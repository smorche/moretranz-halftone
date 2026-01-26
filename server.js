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

// Upload bytes: keep at 25MB unless you explicitly raise it.
// 12" @ 300dpi PNG can easily be several MB to tens of MB depending on complexity.
const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024);

// Pixel guard: 12" @ 300dpi = 3600 px wide. Height varies.
// A “12x12 @ 300dpi” would be 3600*3600 = 12.96MP, which is fine.
// 50MP default is conservative but safe; you can bump if needed.
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 50_000_000);

// Halftone limits
const HARD_MAX_WIDTH = Number(process.env.HARD_MAX_WIDTH || 3600); // supports 12"@300dpi width
const SOFT_MAX_WIDTH = Number(process.env.SOFT_MAX_WIDTH || 3200); // warns above this
const MIN_CELL = Number(process.env.MIN_CELL || 8);
const MAX_CELL = Number(process.env.MAX_CELL || 30);

// Concurrency guard: keep at 1 to avoid stacked peak memory spikes
const MAX_CONCURRENT_JOBS = Number(process.env.MAX_CONCURRENT_JOBS || 1);

// Optional: reject super heavy requests by estimate (safety)
const MAX_CELLS_ESTIMATE = Number(process.env.MAX_CELLS_ESTIMATE || 700_000); // conservative

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
      if (!origin) return cb(null, true); // server-to-server/curl
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
  region: process.env.R2_REGION, // often "auto"
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
 * Full-color halftone (RGBA-safe):
 * - sample average RGBA in each cell
 * - dot color = sampled RGB
 * - dot opacity = sampled alpha
 * - dot radius = based on luminance (darker -> larger dot)
 * - output is transparent PNG
 *
 * Memory notes:
 * - Build shapes using an array (less string churn)
 * - Avoid creating huge intermediate buffers unnecessarily
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape }) {
  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  // Clamp maxWidth to supported range (12"@300dpi => 3600px)
  const targetW = clamp(Math.min(meta.width, maxWidth), 256, HARD_MAX_WIDTH);

  // Resize down to targetW (keep aspect)
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  // Estimate workload & optionally reject insane combos (safety)
  const cs = clamp(Math.floor(cellSize), MIN_CELL, MAX_CELL);
  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);
  const cellCount = cols * rows;

  if (cellCount > MAX_CELLS_ESTIMATE) {
    throw new Error(
      `Settings too heavy (estimated cells=${cellCount}). Increase cell size or reduce max width.`
    );
  }

  // Get raw RGBA pixels
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const half = cs / 2;
  const shapes = [];

  // Build dots
  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      // Average RGBA in the cell (box average)
      let rSum = 0, gSum = 0, bSum = 0, aSum = 0, count = 0;

      for (let y = y0; y < y1; y++) {
        const rowBase = (y * w) * 4;
        for (let x = x0; x < x1; x++) {
          const idx = rowBase + x * 4;
          rSum += data[idx];
          gSum += data[idx + 1];
          bSum += data[idx + 2];
          aSum += data[idx + 3];
          count++;
        }
      }

      if (!count) continue;

      const r = Math.round(rSum / count);
      const g = Math.round(gSum / count);
      const b = Math.round(bSum / count);
      const a = Math.round(aSum / count);

      // Skip fully transparent cells (keeps “shape-only” PNGs intact)
      if (a <= 0) continue;

      const lum = luminance255(r, g, b);
      const darkness = 1 - lum / 255;

      // Dot size: keep a floor so light areas still have some dots,
      // but not so large it fills everything.
      // This is the main "look" knob.
      const radius = clamp(half * (0.12 + 0.98 * darkness), 0, half);

      // Skip ultra tiny dots
      if (radius < 0.35) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;
      const fillOpacity = (a / 255).toFixed(4);

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

  // Render SVG -> PNG (transparent)
  const out = await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return { png: out, width: w, height: h, cellCount };
}

/** ---------------------------
 *  Routes
 *  --------------------------*/
app.get("/health", (_req, res) => res.json({ ok: true }));

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

  // UI default 12; we allow 8..30 here to reduce CPU + memory risk
  cellSize: z.number().int().min(MIN_CELL).max(MAX_CELL).default(12),

  // UI can request up to 3600 for 12"@300dpi; server clamps to HARD_MAX_WIDTH
  maxWidth: z.number().int().min(256).max(4000).default(2000),

  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json(
      errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() })
    );
  }

  if (ACTIVE_JOBS >= MAX_CONCURRENT_JOBS) {
    return tooBusyResponse(res);
  }

  ACTIVE_JOBS++;

  const { key, cellSize, maxWidth, dotShape } = parsed.data;

  // Clamp maxWidth server-side, regardless of what UI sends
  const maxWidthClamped = clamp(maxWidth, 256, HARD_MAX_WIDTH);

  try {
    const started = Date.now();

    if (maxWidthClamped > SOFT_MAX_WIDTH) {
      logWithReq(req, "HIGH MEMORY REQUEST (soft warning)", { maxWidth: maxWidthClamped, cellSize, dotShape });
    }

    logWithReq(req, "PROCESS PARAMS", { key, cellSize, maxWidth: maxWidthClamped, dotShape });

    // HeadObject (optional)
    try {
      const head = await s3.send(new HeadObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
      if (typeof head.ContentLength === "number" && head.ContentLength > MAX_UPLOAD_BYTES) {
        return res.status(413).json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
      }
    } catch (e) {
      logWithReq(req, "HeadObject warning:", String(e?.name || e));
    }

    // GetObject
    const obj = await s3.send(new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res.status(413).json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    // Validate image
    let meta;
    try {
      meta = await sharp(inputBuffer, { failOn: "none" }).metadata();
    } catch (e) {
      logWithReq(req, "BAD IMAGE (sharp decode failed):", e?.message || e);
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

    // Process
    const { png, width, height, cellCount } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth: maxWidthClamped,
      dotShape
    });

    // Store output
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
    logWithReq(req, "PROCESS OK", { outputKey, width, height, cellCount, ms });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth: maxWidthClamped, dotShape },
      perf: { ms, cellCount, outWidth: width, outHeight: height },
      downloadUrl
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
