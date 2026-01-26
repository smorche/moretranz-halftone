import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand, HeadObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const BUILD_ID = process.env.BUILD_ID || `dev_${new Date().toISOString()}`;

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
  "R2_SECRET_ACCESS_KEY"
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
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
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
  return res.status(429).json(
    errJson("BUSY", "Processor busy. Please wait a moment and try again.")
  );
}

/**
 * Halftone generator with DTF-safe alpha handling:
 * - Build a hard alpha mask: maskPx = (alpha > alphaThreshold)
 * - Cell emits dots only if mask coverage >= minCoverage
 * - Sample RGB only from masked pixels (true “ink pixels”)
 * - alphaMode=binary => output alpha hardened to 0/255 (DTF safe)
 */
async function makeColorHalftonePng(inputBuffer, {
  cellSize,
  maxWidth,
  dotShape,

  dotGain,
  minDot,

  alphaThreshold,
  minCoverage,
  alphaMode
}) {
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

  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  // Effective thresholds (DTF-friendly guardrails)
  // - alphaThreshold must be >= 1 so alpha=0 never counts
  const aThresh = clamp(Math.floor(alphaThreshold), 1, 255);
  const covMin = clamp(Number(minCoverage), 0, 1);

  const dg = clamp(Number(dotGain), 0.6, 2.0);
  const minFrac = clamp(Number(minDot), 0, 0.6);

  // Precompute alpha mask: 1 byte per pixel
  const mask = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    mask[p] = data[i + 3] > aThresh ? 1 : 0;
  }

  const half = cs / 2;
  const shapes = [];

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      let rSum = 0, gSum = 0, bSum = 0;
      let maskedCount = 0;
      let totalCount = 0;

      for (let y = y0; y < y1; y++) {
        const rowBasePx = y * w;
        const rowBaseByte = rowBasePx * 4;
        for (let x = x0; x < x1; x++) {
          totalCount++;
          const p = rowBasePx + x;
          if (mask[p] === 0) continue;

          const idx = rowBaseByte + x * 4;
          rSum += data[idx];
          gSum += data[idx + 1];
          bSum += data[idx + 2];
          maskedCount++;
        }
      }

      if (!totalCount || maskedCount <= 0) continue;

      const coverage = maskedCount / totalCount;
      if (coverage < covMin) continue;

      const r = Math.round(rSum / maskedCount);
      const g = Math.round(gSum / maskedCount);
      const b = Math.round(bSum / maskedCount);

      const lum = luminance255(r, g, b);
      let darkness = 1 - lum / 255;

      // dot gain curve
      darkness = Math.pow(darkness, 1 / dg);

      const frac = clamp(minFrac + (1 - minFrac) * darkness, 0, 1);
      const radius = half * frac;

      if (radius < 0.15) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;
      const fillOpacity = "1.0000"; // dots are drawn opaque; we harden final alpha for DTF

      if (dotShape === "square") {
        const size = radius * 2;
        shapes.push(
          `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(2)}" width="${size.toFixed(
            2
          )}" height="${size.toFixed(2)}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
      } else if (dotShape === "ellipse") {
        const rx = radius * 1.2;
        const ry = radius * 0.85;
        shapes.push(
          `<ellipse cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" rx="${rx.toFixed(
            2
          )}" ry="${ry.toFixed(2)}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
      } else {
        shapes.push(
          `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${radius.toFixed(
            2
          )}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
      }
    }
  }

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <rect width="100%" height="100%" fill="transparent" />
  ${shapes.join("")}
</svg>`;

  // Render SVG to raw RGBA so we can force true binary alpha if requested
  const rendered = sharp(Buffer.from(svg));
  const { data: outData, info } = await rendered.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  if (alphaMode === "binary") {
    for (let i = 0; i < outData.length; i += 4) {
      const a = outData[i + 3];
      if (a === 0) {
        outData[i] = 0;
        outData[i + 1] = 0;
        outData[i + 2] = 0;
      } else {
        outData[i + 3] = 255;
      }
    }
  }

  const out = await sharp(outData, {
    raw: { width: info.width, height: info.height, channels: 4 }
  })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return { png: out, width: w, height: h, cellCount };
}

/** ---------------------------
 *  Routes
 *  --------------------------*/
app.get("/health", (_req, res) => res.json({ ok: true }));

// NEW: version endpoint to confirm deploy
app.get("/version", (_req, res) => {
  res.json({
    ok: true,
    buildId: BUILD_ID,
    server: "moretranz-halftone",
    alphaMaskMode: "precomputed-mask",
  });
});

const UploadUrlRequest = z.object({
  filename: z.string().min(1),
  contentType: z.enum(["image/png", "image/jpeg", "image/webp"]),
  contentLength: z.number().int().positive().max(MAX_UPLOAD_BYTES)
});

app.post("/v1/halftone/upload-url", async (req, res) => {
  try {
    const parsed = UploadUrlRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json(
        errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() })
      );
    }

    const { filename, contentType, contentLength } = parsed.data;

    const safeName = sanitizeFilename(filename);
    const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const key = `uploads/${imageId}/${safeName}`;

    const cmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key,
      ContentType: contentType
    });

    const uploadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });

    logWithReq(req, "ISSUED UPLOAD URL", { key, contentType, contentLength });

    return res.json({
      imageId,
      key,
      uploadUrl,
      headers: { "Content-Type": contentType },
      maxBytes: MAX_UPLOAD_BYTES
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

  dotGain: z.number().min(0.6).max(2.0).default(1.25),
  minDot: z.number().min(0.0).max(0.6).default(0.18),

  // IMPORTANT: default alphaThreshold bumped to 48 for safer backgrounds
  alphaThreshold: z.number().int().min(0).max(255).default(48),
  minCoverage: z.number().min(0).max(1).default(0.15),
  alphaMode: z.enum(["binary", "average"]).default("binary"),
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json(
      errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() })
    );
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
    alphaMode
  } = parsed.data;

  const maxWidthClamped = clamp(maxWidth, 256, HARD_MAX_WIDTH);

  try {
    const started = Date.now();

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
      buildId: BUILD_ID
    });

    if (maxWidthClamped > SOFT_MAX_WIDTH) {
      logWithReq(req, "HIGH MEMORY REQUEST (soft warning)", { maxWidth: maxWidthClamped, cellSize, dotShape });
    }

    try {
      const head = await s3.send(new HeadObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
      if (typeof head.ContentLength === "number" && head.ContentLength > MAX_UPLOAD_BYTES) {
        return res.status(413).json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
      }
    } catch (e) {
      logWithReq(req, "HeadObject warning:", String(e?.name || e));
    }

    const obj = await s3.send(new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res.status(413).json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    let meta;
    try {
      meta = await sharp(inputBuffer, { failOn: "none" }).metadata();
    } catch {
      return res.status(400).json(errJson("BAD_IMAGE", "Unsupported or corrupted image file"));
    }

    if (!meta.format || !["png", "jpeg", "webp"].includes(meta.format)) {
      return res.status(400).json(
        errJson("BAD_IMAGE", "Unsupported image format (only PNG, JPG, WEBP)", { format: meta.format || null })
      );
    }
    if (!meta.width || !meta.height) return res.status(400).json(errJson("BAD_IMAGE", "Could not read image dimensions"));

    const totalPixels = meta.width * meta.height;
    if (totalPixels > MAX_IMAGE_PIXELS) {
      return res.status(413).json(
        errJson("TOO_LARGE", `Image resolution too large (${totalPixels} pixels)`, {
          width: meta.width, height: meta.height, maxPixels: MAX_IMAGE_PIXELS
        })
      );
    }

    const { png, width, height, cellCount } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth: maxWidthClamped,
      dotShape,
      dotGain,
      minDot,
      alphaThreshold,
      minCoverage,
      alphaMode
    });

    const outId = `ht_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const outputKey = `outputs/${outId}.png`;

    await s3.send(
      new PutObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
        Body: png,
        ContentType: "image/png"
      })
    );

    const downloadUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: outputKey }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - started;

    return res.json({
      ok: true,
      buildId: BUILD_ID,
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
        alphaMode
      },
      perf: { ms, cellCount, outWidth: width, outHeight: height },
      downloadUrl
    });
  } catch (e) {
    console.error(`[${req._rid}] HALFTONE ERROR:`, e);
    return res.status(500).json(errJson("INTERNAL", "Failed to process halftone"));
  } finally {
    ACTIVE_JOBS = Math.max(0, ACTIVE_JOBS - 1);
  }
});

/** ---------------------------
 *  Start
 *  --------------------------*/
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
