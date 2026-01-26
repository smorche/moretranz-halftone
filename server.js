import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "1mb" }));

/** ---------------------------
 *  Config
 *  --------------------------*/
const requiredEnv = [
  "R2_ENDPOINT",
  "R2_BUCKET",
  "R2_REGION",
  "R2_ACCESS_KEY_ID",
  "R2_SECRET_ACCESS_KEY",
];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024);
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 50_000_000);

const HARD_MAX_WIDTH = Number(process.env.HARD_MAX_WIDTH || 3600);
const SOFT_MAX_WIDTH = Number(process.env.SOFT_MAX_WIDTH || 3200);
const MIN_CELL = Number(process.env.MIN_CELL || 8);
const MAX_CELL = Number(process.env.MAX_CELL || 30);

const MAX_CONCURRENT_JOBS = Number(process.env.MAX_CONCURRENT_JOBS || 1);
const MAX_CELLS_ESTIMATE = Number(process.env.MAX_CELLS_ESTIMATE || 700_000);

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function reqId() {
  return crypto.randomBytes(8).toString("hex");
}

function errJson(code, message, details) {
  return { error: { code, message, ...(details ? { details } : {}) } };
}

function logWithReq(req, ...args) {
  console.log(`[${req._rid}]`, ...args);
}

async function streamToBuffer(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function sanitizeFilename(name) {
  return String(name || "file").replace(/[^a-zA-Z0-9._-]/g, "_");
}

function luminance255(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** ---------------------------
 *  CORS
 *  --------------------------*/
const ALLOWED_ORIGINS = new Set([
  "https://moretranz-halftone.pages.dev",
  "https://moretranz.com",
  "https://www.moretranz.com",
]);

app.use(
  cors({
    origin: (origin, cb) => {
      if (!origin) return cb(null, true);
      if (ALLOWED_ORIGINS.has(origin)) return cb(null, true);
      return cb(new Error(`CORS blocked for origin: ${origin}`));
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
    optionsSuccessStatus: 204,
  })
);

/** ---------------------------
 *  S3 (Cloudflare R2)
 *  --------------------------*/
const s3 = new S3Client({
  region: process.env.R2_REGION,
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

/** ---------------------------
 *  Middleware: request id
 *  --------------------------*/
app.use((req, _res, next) => {
  req._rid = reqId();
  next();
});

/** ---------------------------
 *  Concurrency guard
 *  --------------------------*/
let ACTIVE_JOBS = 0;
function tooBusyResponse(res) {
  return res
    .status(429)
    .json(errJson("BUSY", "Processor busy. Please wait a moment and try again."));
}

/**
 * Render SVG onto a guaranteed transparent canvas.
 * This avoids any surprises where SVG render introduces non-zero alpha outside shapes.
 */
async function renderSvgToPng({ width, height, svgBuffer }) {
  return sharp({
    create: {
      width,
      height,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    },
  })
    .composite([{ input: svgBuffer }])
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

/**
 * Compute basic alpha stats + a histogram for auto-thresholding.
 */
function alphaStatsAndHist(rgba) {
  const hist = new Uint32Array(256);

  let nonZero = 0;
  let ge48 = 0;
  let ge100 = 0;
  let ge200 = 0;

  for (let i = 3; i < rgba.length; i += 4) {
    const a = rgba[i];
    hist[a]++;

    if (a > 0) nonZero++;
    if (a >= 48) ge48++;
    if (a >= 100) ge100++;
    if (a >= 200) ge200++;
  }

  const total = rgba.length / 4;
  const stats = {
    total,
    nonZeroPct: nonZero / total,
    ge48Pct: ge48 / total,
    ge100Pct: ge100 / total,
    ge200Pct: ge200 / total,
  };

  return { stats, hist };
}

/**
 * Given an alpha histogram, choose a "tight" threshold:
 * - We look for a high percentile among non-zero pixels to isolate truly solid areas.
 * - This is specifically for "dirty alpha" PNGs where the canvas has lots of semi-opaque haze.
 */
function suggestedTightAlphaThreshold(hist, { percentile = 0.92 } = {}) {
  let nonZeroTotal = 0;
  for (let a = 1; a <= 255; a++) nonZeroTotal += hist[a];
  if (nonZeroTotal === 0) return 255;

  const target = Math.floor(nonZeroTotal * percentile);
  let acc = 0;
  for (let a = 1; a <= 255; a++) {
    acc += hist[a];
    if (acc >= target) return a;
  }
  return 255;
}

/**
 * DTF-safe halftone:
 * - Build dots from input sampling
 * - Render a color SVG to PNG
 * - Render a *mask SVG* (same dots, solid white) to PNG
 * - Use mask alpha ONLY (binary) so no faint pixel haze can create underbase
 */
async function makeColorHalftonePng(inputBuffer, opts) {
  const {
    cellSize,
    maxWidth,
    dotShape,
    dotGain,
    minDot,
    alphaThreshold,
    minCoverage,
    alphaMode, // "binary" | "average"
    autoTightenAlpha, // boolean
  } = opts;

  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  const targetW = clamp(Math.min(meta.width, maxWidth), 256, HARD_MAX_WIDTH);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  const cs = clamp(Math.floor(cellSize), MIN_CELL, MAX_CELL);
  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);
  const cellCount = cols * rows;

  if (cellCount > MAX_CELLS_ESTIMATE) {
    throw new Error(
      `Settings too heavy (estimated cells=${cellCount}). Increase cell size or reduce max width.`
    );
  }

  // raw RGBA pixels
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  // alpha stats + hist
  const { stats, hist } = alphaStatsAndHist(data);

  // log stats for visibility
  console.log("ALPHA STATS", {
    total: stats.total,
    nonZeroPct: stats.nonZeroPct.toFixed(4),
    ge48Pct: stats.ge48Pct.toFixed(4),
    ge100Pct: stats.ge100Pct.toFixed(4),
    ge200Pct: stats.ge200Pct.toFixed(4),
  });

  // DTF reality: if most of the image has high alpha, your threshold must be tight.
  // Auto-tighten when:
  // - a majority has alpha >= 200 OR
  // - a majority has alpha >= 100
  // This matches what your ribbon PNG showed.
  let aThresh = clamp(Math.floor(alphaThreshold), 1, 255);

  const looksDirtyAlpha =
    stats.ge200Pct > 0.50 || stats.ge100Pct > 0.65 || stats.ge48Pct > 0.75;

  if (alphaMode === "binary" && autoTightenAlpha && looksDirtyAlpha) {
    const suggested = suggestedTightAlphaThreshold(hist, { percentile: 0.92 });
    // We never lower the user threshold, only tighten if suggested is higher.
    const tightened = Math.max(aThresh, suggested);
    console.log("AUTO TIGHTEN ALPHA", { aThresh, suggested, tightened });
    aThresh = clamp(tightened, 1, 255);
  }

  const covMin = clamp(Number(minCoverage), 0, 1);
  const dg = clamp(Number(dotGain), 0.6, 2.0);
  const minFrac = clamp(Number(minDot), 0, 0.6);

  const half = cs / 2;

  const colorShapes = [];
  const maskShapes = [];

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      let rSum = 0,
        gSum = 0,
        bSum = 0;

      let solidCount = 0;
      let totalCount = 0;

      for (let y = y0; y < y1; y++) {
        const rowBase = y * w * 4;
        for (let x = x0; x < x1; x++) {
          const idx = rowBase + x * 4;
          const aPx = data[idx + 3];
          totalCount++;

          if (aPx >= aThresh) {
            rSum += data[idx];
            gSum += data[idx + 1];
            bSum += data[idx + 2];
            solidCount++;
          }
        }
      }

      if (!totalCount || solidCount === 0) continue;

      const coverage = solidCount / totalCount;
      if (coverage < covMin) continue;

      const r = Math.round(rSum / solidCount);
      const g = Math.round(gSum / solidCount);
      const b = Math.round(bSum / solidCount);

      let darkness = 1 - luminance255(r, g, b) / 255;
      // dot gain curve (dg>1 darkens midtones)
      darkness = Math.pow(darkness, 1 / dg);

      // minimum dot floor
      const frac = clamp(minFrac + (1 - minFrac) * darkness, 0, 1);
      const radius = half * frac;
      if (radius < 0.15) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;

      // In "average" mode, opacity follows coverage.
      // In DTF "binary" mode, opacity is 1 and alpha comes from mask.
      const fillOpacity =
        alphaMode === "average" ? clamp(coverage, 0, 1).toFixed(4) : "1";

      if (dotShape === "square") {
        const size = radius * 2;
        const x = (cx - size / 2).toFixed(2);
        const y = (cy - size / 2).toFixed(2);
        const s = size.toFixed(2);

        colorShapes.push(
          `<rect x="${x}" y="${y}" width="${s}" height="${s}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
        maskShapes.push(
          `<rect x="${x}" y="${y}" width="${s}" height="${s}" fill="white" />`
        );
      } else if (dotShape === "ellipse") {
        const rx = (radius * 1.2).toFixed(2);
        const ry = (radius * 0.85).toFixed(2);
        const cxs = cx.toFixed(2);
        const cys = cy.toFixed(2);

        colorShapes.push(
          `<ellipse cx="${cxs}" cy="${cys}" rx="${rx}" ry="${ry}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
        maskShapes.push(
          `<ellipse cx="${cxs}" cy="${cys}" rx="${rx}" ry="${ry}" fill="white" />`
        );
      } else {
        const cxs = cx.toFixed(2);
        const cys = cy.toFixed(2);
        const rs = radius.toFixed(2);

        colorShapes.push(
          `<circle cx="${cxs}" cy="${cys}" r="${rs}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
        maskShapes.push(
          `<circle cx="${cxs}" cy="${cys}" r="${rs}" fill="white" />`
        );
      }
    }
  }

  const colorSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  ${colorShapes.join("")}
</svg>`;

  const maskSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  ${maskShapes.join("")}
</svg>`;

  const colorPng = await renderSvgToPng({
    width: w,
    height: h,
    svgBuffer: Buffer.from(colorSvg),
  });
  const maskPng = await renderSvgToPng({
    width: w,
    height: h,
    svgBuffer: Buffer.from(maskSvg),
  });

  if (alphaMode === "binary") {
    // DTF-safe: alpha ONLY comes from mask, binary
    const alphaBin = await sharp(maskPng)
      .ensureAlpha()
      .extractChannel(3)
      .threshold(1) // any dot pixel => 255
      .png()
      .toBuffer();

    const rgb = await sharp(colorPng).ensureAlpha().removeAlpha().png().toBuffer();
    const hardened = await sharp(rgb).joinChannel(alphaBin).png().toBuffer();

    return {
      png: hardened,
      width: w,
      height: h,
      cellCount,
      alphaThresholdUsed: aThresh,
      autoTightened: alphaMode === "binary" && autoTightenAlpha && looksDirtyAlpha,
    };
  }

  return {
    png: colorPng,
    width: w,
    height: h,
    cellCount,
    alphaThresholdUsed: aThresh,
    autoTightened: false,
  };
}

/** ---------------------------
 *  Routes
 *  --------------------------*/
app.get("/health", (_req, res) => res.json({ ok: true }));

const UploadUrlRequest = z.object({
  filename: z.string().min(1),
  contentType: z.enum(["image/png", "image/jpeg", "image/webp"]),
  contentLength: z.number().int().positive().max(MAX_UPLOAD_BYTES),
});

app.post("/v1/halftone/upload-url", async (req, res) => {
  try {
    const parsed = UploadUrlRequest.safeParse(req.body);
    if (!parsed.success) {
      return res
        .status(400)
        .json(errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() }));
    }

    const { filename, contentType, contentLength } = parsed.data;

    const safeName = sanitizeFilename(filename);
    const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const key = `uploads/${imageId}/${safeName}`;

    const cmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key,
      ContentType: contentType,
    });

    const uploadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });

    logWithReq(req, "ISSUED UPLOAD URL", { key, contentType, contentLength });

    return res.json({
      imageId,
      key,
      uploadUrl,
      headers: { "Content-Type": contentType },
      maxBytes: MAX_UPLOAD_BYTES,
    });
  } catch (e) {
    console.error(`[${req._rid}] upload-url error`, e);
    return res.status(500).json(errJson("INTERNAL", "Failed to create upload URL"));
  }
});

const ProcessRequest = z.object({
  key: z.string().min(1),

  cellSize: z.number().int().min(MIN_CELL).max(MAX_CELL).default(12),
  maxWidth: z.number().int().min(256).max(4000).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),

  // Darkness controls
  dotGain: z.number().min(0.6).max(2.0).default(1.35),
  minDot: z.number().min(0.0).max(0.6).default(0.18),

  // ✅ DTF-safe defaults (tighter)
  alphaThreshold: z.number().int().min(1).max(255).default(200),
  minCoverage: z.number().min(0).max(1).default(0.35),

  // Output alpha behavior
  alphaMode: z.enum(["binary", "average"]).default("binary"),

  // ✅ Auto tighten alpha if image looks “dirty”
  autoTightenAlpha: z.boolean().default(true),
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res
      .status(400)
      .json(errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() }));
  }

  if (ACTIVE_JOBS >= MAX_CONCURRENT_JOBS) return tooBusyResponse(res);
  ACTIVE_JOBS++;

  const {
    key,
    cellSize,
    maxWidth,
    dotShape,
    dotGain,
    minDot,
    alphaThreshold,
    minCoverage,
    alphaMode,
    autoTightenAlpha,
  } = parsed.data;

  const maxWidthClamped = clamp(maxWidth, 256, HARD_MAX_WIDTH);

  try {
    const started = Date.now();

    if (maxWidthClamped > SOFT_MAX_WIDTH) {
      logWithReq(req, "HIGH MEMORY REQUEST (soft warning)", {
        maxWidth: maxWidthClamped,
        cellSize,
        dotShape,
      });
    }

    logWithReq(req, "PROCESS PARAMS", {
      key,
      cellSize,
      maxWidth: maxWidthClamped,
      dotShape,
      dotGain,
      minDot,
      alphaThreshold,
      minCoverage,
      alphaMode,
      autoTightenAlpha,
    });

    try {
      const head = await s3.send(new HeadObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
      if (typeof head.ContentLength === "number" && head.ContentLength > MAX_UPLOAD_BYTES) {
        return res
          .status(413)
          .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
      }
    } catch (e) {
      logWithReq(req, "HeadObject warning:", String(e?.name || e));
    }

    const obj = await s3.send(new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res
        .status(413)
        .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    let meta;
    try {
      meta = await sharp(inputBuffer, { failOn: "none" }).metadata();
    } catch (e) {
      logWithReq(req, "BAD IMAGE (sharp decode failed):", e?.message || e);
      return res.status(400).json(errJson("BAD_IMAGE", "Unsupported or corrupted image file"));
    }

    if (!meta.format || !["png", "jpeg", "webp"].includes(meta.format)) {
      return res.status(400).json(
        errJson("BAD_IMAGE", "Unsupported image format (only PNG, JPG, WEBP)", {
          format: meta.format || null,
        })
      );
    }
    if (!meta.width || !meta.height) {
      return res.status(400).json(errJson("BAD_IMAGE", "Could not read image dimensions"));
    }

    const totalPixels = meta.width * meta.height;
    if (totalPixels > MAX_IMAGE_PIXELS) {
      return res.status(413).json(
        errJson("TOO_LARGE", `Image resolution too large (${totalPixels} pixels)`, {
          width: meta.width,
          height: meta.height,
          maxPixels: MAX_IMAGE_PIXELS,
        })
      );
    }

    const { png, width, height, cellCount, alphaThresholdUsed, autoTightened } =
      await makeColorHalftonePng(inputBuffer, {
        cellSize,
        maxWidth: maxWidthClamped,
        dotShape,
        dotGain,
        minDot,
        alphaThreshold,
        minCoverage,
        alphaMode,
        autoTightenAlpha,
      });

    const outId = `ht_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const outputKey = `outputs/${outId}.png`;

    await s3.send(
      new PutObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
        Body: png,
        ContentType: "image/png",
      })
    );

    const downloadUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: outputKey }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - started;
    logWithReq(req, "PROCESS OK", { outputKey, width, height, cellCount, ms, alphaThresholdUsed, autoTightened });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: {
        cellSize,
        maxWidth: maxWidthClamped,
        dotShape,
        dotGain,
        minDot,
        alphaThreshold,
        minCoverage,
        alphaMode,
        autoTightenAlpha,
        alphaThresholdUsed,
        autoTightened,
      },
      perf: { ms, cellCount, outWidth: width, outHeight: height },
      downloadUrl,
    });
  } catch (e) {
    console.error(`[${req._rid}] HALFTONE ERROR:`, e);
    return res.status(500).json(errJson("INTERNAL", "Failed to process halftone"));
  } finally {
    ACTIVE_JOBS--;
  }
});

/** ---------------------------
 *  Start
 *  --------------------------*/
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
