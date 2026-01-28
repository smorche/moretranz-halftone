import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand, HeadObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "1mb" }));

/** ---------------------------
 *  Config / Limits
 *  --------------------------*/
const requiredEnv = ["R2_ENDPOINT", "R2_BUCKET", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB default

// Pixel cap for decode safety. 12"x12" @300dpi is 3600x3600=12,960,000 px.
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000);

// Max width (12" @ 300dpi = 3600px)
const HARD_MAX_WIDTH = Number(process.env.HARD_MAX_WIDTH || 3600);

// Halftone controls bounds
const MIN_CELL = Number(process.env.MIN_CELL || 6);
const MAX_CELL = Number(process.env.MAX_CELL || 48);

// Guard against too many SVG shapes (cells)
const MAX_CELLS_ESTIMATE = Number(process.env.MAX_CELLS_ESTIMATE || 260_000);

// Alpha-crop behavior
const ALPHA_BBOX_THRESHOLD = Number(process.env.ALPHA_BBOX_THRESHOLD || 8); // 0..255
const ALPHA_CROP_PADDING = Number(process.env.ALPHA_CROP_PADDING || 2); // px padding around silhouette
const MIN_COVERAGE_TO_DRAW = Number(process.env.MIN_COVERAGE_TO_DRAW || 0.03); // 0..1

function parseCorsOrigins(val) {
  if (!val || val.trim() === "" || val.trim() === "*") return null;
  return new Set(
    val
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
  );
}

const DEFAULT_ALLOWED = new Set([
  "https://moretranz-halftone.pages.dev",
  "https://moretranz.com",
  "https://www.moretranz.com",
]);

const envAllowed = parseCorsOrigins(process.env.CORS_ORIGINS);
const ALLOWED_ORIGINS = envAllowed || DEFAULT_ALLOWED;

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
 *  Helpers
 *  --------------------------*/
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

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function luminance255(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Find bounding box of alpha > threshold.
 * alphaBuf is 1 byte per pixel (0..255), length w*h.
 */
function findAlphaBBox(alphaBuf, w, h, thr) {
  let minX = w, minY = h, maxX = -1, maxY = -1;

  const t = clamp(Math.floor(thr), 0, 255);

  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      const a = alphaBuf[rowOff + x];
      if (a > t) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0 || maxY < 0) return null; // no nontransparent pixels
  return { left: minX, top: minY, right: maxX, bottom: maxY };
}

/**
 * Full-color halftone (transparent output) with alpha-crop:
 * - crop to alpha silhouette first (prevents rectangle artifacts)
 * - in each cell:
 *   - compute alpha coverage (how much of the cell is nontransparent)
 *   - compute alpha-weighted average color (ignoring low-alpha pixels)
 *   - dot size based on luminance + strength (darker => bigger)
 *   - dot opacity tied to coverage (preserves cutout edges)
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  // Decode
  const base = sharp(inputBuffer, { failOn: "none" });
  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  // Ensure alpha exists; grab alpha channel for bbox detection
  // unpremultiply helps when alpha is present (prevents color contamination math)
  const ensured = base.ensureAlpha().unpremultiply();

  const alphaRaw = await ensured.extractChannel(3).raw().toBuffer();
  const bbox = findAlphaBBox(alphaRaw, meta.width, meta.height, ALPHA_BBOX_THRESHOLD);

  if (!bbox) {
    throw new Error("Image appears fully transparent (no alpha pixels > threshold)");
  }

  // Apply padding and clamp
  const pad = Math.max(0, Math.floor(ALPHA_CROP_PADDING));
  const left = clamp(bbox.left - pad, 0, meta.width - 1);
  const top = clamp(bbox.top - pad, 0, meta.height - 1);
  const right = clamp(bbox.right + pad, 0, meta.width - 1);
  const bottom = clamp(bbox.bottom + pad, 0, meta.height - 1);

  const cropW = right - left + 1;
  const cropH = bottom - top + 1;

  // Crop to silhouette area FIRST
  let cropped = ensured.extract({ left, top, width: cropW, height: cropH });

  // Resize down to maxWidth (hard clamp to 3600)
  const targetW = clamp(Math.min(cropW, maxWidth), 256, HARD_MAX_WIDTH);
  cropped = cropped.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await cropped.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  const cs = clamp(Math.floor(cellSize), MIN_CELL, MAX_CELL);
  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);
  const cellCount = cols * rows;

  if (cellCount > MAX_CELLS_ESTIMATE) {
    throw new Error(
      `Settings too heavy: estimated cells=${cellCount}. Increase Cell size or reduce Max width.`
    );
  }

  const { data } = await cropped.ensureAlpha().unpremultiply().raw().toBuffer({ resolveWithObject: true });

  // Strength mapping:
  // - baseline 100
  // - allow up to 250 for noticeably richer color / dot coverage
  const sNorm = clamp((Number(strength) || 100) / 100, 0, 2.5); // 0..2.5
  const curveExp = 1 / Math.max(0.25, sNorm); // higher strength => darker midtones
  const radiusGain = 0.95 + 0.85 * sNorm; // coverage gain
  const alphaGain = clamp(0.95 + 0.40 * sNorm, 0, 1.6); // “ink gain”
  const minDotRadiusPx = 0.28;

  // When computing average color, ignore very low alpha pixels
  const ALPHA_SAMPLE_CUTOFF = 16 / 255; // ignore near-transparent pixels for color avg

  const half = cs / 2;
  const shapes = [];

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      const totalPx = (x1 - x0) * (y1 - y0);
      if (totalPx <= 0) continue;

      // Coverage: fraction of pixels with alpha > bbox threshold
      // This is the key that prevents “rectangle halftone”
      let opaqueCount = 0;

      // Alpha-weighted color average
      let aSum = 0;
      let rSum = 0, gSum = 0, bSum = 0;

      for (let y = y0; y < y1; y++) {
        const rowBase = (y * w) * 4;
        for (let x = x0; x < x1; x++) {
          const idx = rowBase + x * 4;
          const a255 = data[idx + 3];
          if (a255 > ALPHA_BBOX_THRESHOLD) opaqueCount++;

          const a = a255 / 255;
          if (a <= ALPHA_SAMPLE_CUTOFF) continue;

          aSum += a;
          rSum += data[idx] * a;
          gSum += data[idx + 1] * a;
          bSum += data[idx + 2] * a;
        }
      }

      const coverage = clamp(opaqueCount / totalPx, 0, 1);
      if (coverage < MIN_COVERAGE_TO_DRAW) continue; // hard skip near-empty cells

      if (aSum <= 1e-6) continue;

      const r = Math.round(rSum / aSum);
      const g = Math.round(gSum / aSum);
      const b = Math.round(bSum / aSum);

      const lum = luminance255(r, g, b);
      const darkness = 1 - lum / 255;

      // Strength curves the darkness response (increases dot size for midtones)
      const darknessCurved = clamp(Math.pow(darkness, curveExp), 0, 1);

      // Mix in coverage so edge cells don’t fill the box
      const combined = clamp(darknessCurved * (0.60 + 0.40 * coverage), 0, 1);

      const radius = half * clamp((0.05 + 0.95 * combined) * radiusGain, 0, 1);
      if (radius < minDotRadiusPx) continue;

      // Opacity is driven by coverage (shape fidelity), plus “ink gain” via strength
      const op = clamp(coverage * alphaGain, 0, 1).toFixed(4);

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;
      const fill = `rgb(${r},${g},${b})`;

      if (dotShape === "square") {
        const size = radius * 2;
        shapes.push(
          `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(2)}" width="${size.toFixed(
            2
          )}" height="${size.toFixed(2)}" fill="${fill}" fill-opacity="${op}" />`
        );
      } else if (dotShape === "ellipse") {
        const rx = radius * 1.2;
        const ry = radius * 0.85;
        shapes.push(
          `<ellipse cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" rx="${rx.toFixed(
            2
          )}" ry="${ry.toFixed(2)}" fill="${fill}" fill-opacity="${op}" />`
        );
      } else {
        shapes.push(
          `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${radius.toFixed(
            2
          )}" fill="${fill}" fill-opacity="${op}" />`
        );
      }
    }
  }

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <rect width="100%" height="100%" fill="transparent" />
  ${shapes.join("")}
</svg>`;

  const out = await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return {
    png: out,
    width: w,
    height: h,
    cellCount,
    croppedFrom: { left, top, width: cropW, height: cropH },
  };
}

/** ---------------------------
 *  Middleware: request id
 *  --------------------------*/
app.use((req, _res, next) => {
  req._rid = reqId();
  next();
});

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
      return res.status(400).json(
        errJson("BAD_REQUEST", "Invalid request", {
          issues: parsed.error.flatten(),
        })
      );
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
  maxWidth: z.number().int().min(256).max(HARD_MAX_WIDTH).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  strength: z.number().int().min(0).max(250).default(120),
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json(
      errJson("BAD_REQUEST", "Invalid request", {
        issues: parsed.error.flatten(),
      })
    );
  }

  const { key, cellSize, maxWidth, dotShape, strength } = parsed.data;

  try {
    logWithReq(req, "PROCESS PARAMS:", { key, cellSize, maxWidth, dotShape, strength });

    // HeadObject size guard (when available)
    try {
      const head = await s3.send(
        new HeadObjectCommand({
          Bucket: process.env.R2_BUCKET,
          Key: key,
        })
      );

      if (typeof head.ContentLength === "number" && head.ContentLength > MAX_UPLOAD_BYTES) {
        return res
          .status(413)
          .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
      }
    } catch (e) {
      logWithReq(req, "HeadObject warning:", String(e?.name || e));
    }

    // Download from R2
    const obj = await s3.send(
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: key,
      })
    );

    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res
        .status(413)
        .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    // Validate format quickly
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

    if (!meta.width || !meta.height) return res.status(400).json(errJson("BAD_IMAGE", "Could not read image dimensions"));

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

    const t0 = Date.now();

    // Generate halftone
    const { png, width, height, cellCount, croppedFrom } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth,
      dotShape,
      strength,
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
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
      }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - t0;
    logWithReq(req, "PROCESS OK:", { outputKey, width, height, cellCount, ms, croppedFrom });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      stats: { width, height, cellCount, ms, croppedFrom },
      downloadUrl,
    });
  } catch (e) {
    const msg = String(e?.message || e);
    console.error(`[${req._rid}] HALFTONE ERROR:`, e);

    if (msg.includes("Settings too heavy")) {
      return res.status(400).json(errJson("TOO_HEAVY", msg, { maxCells: MAX_CELLS_ESTIMATE }));
    }

    // Helpful error when alpha bbox fails
    if (msg.includes("fully transparent")) {
      return res.status(400).json(errJson("BAD_IMAGE", msg));
    }

    return res.status(500).json(errJson("INTERNAL", "Failed to process halftone"));
  }
});

/** ---------------------------
 *  Start
 *  --------------------------*/
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
