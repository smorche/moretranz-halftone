// server.js (FULL REPLACEMENT)

import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import { z } from "zod";
import sharp from "sharp";
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

/**
 * ─────────────────────────────────────────────────────────────
 * Sharp resource controls (helps avoid memory spikes)
 * ─────────────────────────────────────────────────────────────
 */
sharp.cache(false);
// Keep this conservative; raise if Standard instance is stable.
// 2 is a good default for many Render boxes.
sharp.concurrency(Number(process.env.SHARP_CONCURRENCY || 2));

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

// Upload size (bytes)
const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB default

// Pixel guard (total pixels). 3600x3600 = 12.96MP; this allows larger, but not insane files.
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000); // 60MP default

// Max output width. 12" @ 300dpi = 3600px.
const MAX_MAXWIDTH = Number(process.env.MAX_MAXWIDTH || 3600);

// Alpha cutoff: anything below this alpha (0-255) is treated as fully transparent for dot generation.
// This is the key fix for “faint rectangle” on transparent PNGs.
const ALPHA_CUTOFF = Number(process.env.ALPHA_CUTOFF || 24);

// Allow origins (exact match). You can also add more later.
const ALLOWED_ORIGINS = new Set([
  "https://moretranz-halftone.pages.dev",
  "https://moretranz.com",
  "https://www.moretranz.com",
]);

/**
 * Manual CORS middleware:
 * - never throws (prevents “Failed to fetch” due to middleware error)
 * - cleanly answers OPTIONS preflight
 */
app.use((req, res, next) => {
  const origin = req.headers.origin;

  // Allow server-to-server requests (no Origin header)
  if (!origin) return next();

  const allowed = ALLOWED_ORIGINS.has(origin);

  if (allowed) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Max-Age", "86400");
  }

  if (req.method === "OPTIONS") {
    // If origin not allowed, reply 403 instead of crashing.
    return res.status(allowed ? 204 : 403).send("");
  }

  if (!allowed) {
    return res.status(403).json({
      error: {
        code: "CORS_BLOCKED",
        message: `CORS blocked for origin: ${origin}`,
      },
    });
  }

  return next();
});

/** ---------------------------
 *  S3 (Cloudflare R2)
 *  --------------------------*/
const s3 = new S3Client({
  region: process.env.R2_REGION, // usually "auto"
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
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function luminance255(r, g, b) {
  // relative luminance approximation in sRGB space (good enough here)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Full-color halftone (transparent PNG output):
 * FIXES:
 * - alpha-weighted color averaging (transparent pixels do NOT wash out color)
 * - alpha cutoff (removes faint “rectangle” dotting)
 * - improved Strength curve (lets you get much darker / richer output)
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  // Enforce maxWidth upper bound
  const safeMaxWidth = clamp(Math.floor(maxWidth), 256, MAX_MAXWIDTH);

  // Resize down to safeMaxWidth (keep aspect)
  const targetW = Math.min(meta.width, safeMaxWidth);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  // Raw RGBA pixels
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  // Clamp cell size for CPU sanity
  const cs = clamp(Math.floor(cellSize), 6, 80);
  const half = cs / 2;

  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);

  // Strength: baseline 100; allow stronger curves beyond that
  // We’ll use it to:
  // 1) increase dot coverage (radius)
  // 2) apply a mild “contrast” curve to darkness
  const s = clamp(Number(strength || 100), 50, 220); // allow more headroom
  const strengthFactor = s / 100; // 1.0 at 100
  // Gamma: lower gamma makes midtones darker faster
  const gamma = clamp(1.05 - (strengthFactor - 1) * 0.25, 0.55, 1.05);

  let shapes = "";

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      // Alpha-weighted average:
      // - sum RGB weighted by alpha
      // - sum alpha
      let rSum = 0, gSum = 0, bSum = 0;
      let aSum = 0;
      let count = 0;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = (y * w + x) * 4;
          const a255 = data[idx + 3];
          const a = a255 / 255;

          // Still count pixels for average alpha
          count++;
          aSum += a;

          // Only contribute color if alpha meaningfully present
          if (a255 > 0) {
            rSum += data[idx] * a;
            gSum += data[idx + 1] * a;
            bSum += data[idx + 2] * a;
          }
        }
      }

      if (count === 0) continue;

      const aAvg = aSum / count; // 0..1
      const aAvg255 = Math.round(aAvg * 255);

      // IMPORTANT: kill “almost transparent” background
      if (aAvg255 < ALPHA_CUTOFF) continue;

      // Un-premultiply to get true average color of visible pixels
      // If aSum is tiny, avoid divide-by-zero; skip in that case
      if (aSum <= 1e-6) continue;

      const r = clamp(Math.round(rSum / aSum), 0, 255);
      const g = clamp(Math.round(gSum / aSum), 0, 255);
      const b = clamp(Math.round(bSum / aSum), 0, 255);

      // Darkness from luminance
      const lum = luminance255(r, g, b); // 0..255
      let darkness = 1 - lum / 255; // 0..1

      // Apply strength curve:
      // - gamma curve boosts midtones when strength > 100
      darkness = Math.pow(clamp(darkness, 0, 1), gamma);

      // Convert darkness into radius coverage:
      // baselineMin prevents dots disappearing too easily.
      // strengthFactor allows coverage beyond baseline, but clamp to full cell.
      const baselineMin = 0.18; // slightly higher than before (less washed out)
      let coverage = baselineMin + (1 - baselineMin) * darkness;

      // Scale coverage with strength (>1 increases coverage)
      coverage *= strengthFactor;

      // Clamp to [0..1.15] to allow “fatter” dots but avoid insane overlap
      coverage = clamp(coverage, 0, 1.15);

      const radius = clamp(half * coverage, 0, half);

      if (radius < 0.35) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;

      // For alpha: keep it tied to average alpha; also slightly boost opacity
      // as strength increases (helps edges read better).
      const opacityBoost = clamp(0.85 + (strengthFactor - 1) * 0.25, 0.85, 1.15);
      const fillOpacity = clamp(aAvg * opacityBoost, 0, 1).toFixed(4);

      if (dotShape === "square") {
        const size = radius * 2;
        shapes += `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(
          2
        )}" width="${size.toFixed(2)}" height="${size.toFixed(
          2
        )}" rx="0" ry="0" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      } else if (dotShape === "ellipse") {
        const rx = radius * 1.2;
        const ry = radius * 0.85;
        shapes += `<ellipse cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" rx="${rx.toFixed(
          2
        )}" ry="${ry.toFixed(2)}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      } else {
        shapes += `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${radius.toFixed(
          2
        )}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      }
    }
  }

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <rect width="100%" height="100%" fill="transparent" />
  ${shapes}
</svg>`;

  const out = await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return { png: out, width: w, height: h };
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
  cellSize: z.number().int().min(6).max(80).default(12),
  maxWidth: z.number().int().min(256).max(MAX_MAXWIDTH).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  // Strength slider: baseline 100; allow richer/darker up to 220
  strength: z.number().int().min(50).max(220).default(100),
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

  const t0 = Date.now();

  try {
    logWithReq(req, "PROCESS PARAMS:", { key, cellSize, maxWidth, dotShape, strength });

    // 1) HeadObject check
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

    // 2) Download from R2
    const obj = await s3.send(
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: key,
      })
    );

    if (!obj.Body) {
      return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));
    }

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) {
      return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    }
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res
        .status(413)
        .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    // 3) Validate decode
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

    // 4) Generate halftone PNG
    const { png, width, height } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth,
      dotShape,
      strength,
    });

    // 5) Store output in R2
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

    // 6) Signed download URL
    const downloadUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
      }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - t0;

    logWithReq(req, "PROCESS OK:", { outputKey, width, height, ms });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength, alphaCutoff: ALPHA_CUTOFF },
      ms,
      downloadUrl,
    });
  } catch (e) {
    console.error(`[${req._rid}] HALFTONE ERROR:`, e);
    return res.status(500).json(errJson("INTERNAL", "Failed to process halftone"));
  }
});

/** ---------------------------
 *  Start
 *  --------------------------*/
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
