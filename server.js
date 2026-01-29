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
const requiredEnv = ["R2_ENDPOINT", "R2_BUCKET", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 50_000_000); // 50MP

// 12" @ 300dpi = 3600px
const MAX_WIDTH_PX = Number(process.env.MAX_WIDTH_PX || 3600);

// Treat nearly-transparent pixels as background for sampling
const ALPHA_EPS = Number(process.env.ALPHA_EPS || 8); // 0..255

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
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** ---------------------------
 *  CORS
 *  --------------------------*/
// Allow:
// - your Pages app (any *.pages.dev, but we’ll still keep it tight)
// - moretranz.com
// - localhost for dev
function isAllowedOrigin(origin) {
  if (!origin) return true; // server-to-server / curl
  if (origin === "https://moretranz.com") return true;
  if (origin === "https://www.moretranz.com") return true;

  // Pages domains (your tool lives here)
  // Examples:
  //   https://moretranz-halftone.pages.dev
  //   https://<branch>.<project>.pages.dev (if you ever enable preview deployments)
  try {
    const u = new URL(origin);
    if (u.hostname === "moretranz-halftone.pages.dev") return true;
    if (u.hostname.endsWith(".pages.dev")) return true;
    if (u.hostname === "localhost" || u.hostname === "127.0.0.1") return true;
  } catch {
    return false;
  }
  return false;
}

app.use(
  cors({
    origin: (origin, cb) => {
      if (isAllowedOrigin(origin)) return cb(null, true);
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
  region: process.env.R2_REGION, // usually "auto"
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

/** ---------------------------
 *  Halftone generator (ALPHA-CORRECT)
 *  --------------------------*/
/**
 * Full-color halftone:
 * - alpha-weighted sampling (premultiplied) so transparent BG does NOT wash colors
 * - dot radius can slightly exceed cell to avoid "faded" output
 * - "strength" meaningfully increases dot coverage + midtone density
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);

  // Cap to 12" @ 300dpi by default (3600px)
  const cappedMaxWidth = clamp(Math.floor(maxWidth || MAX_WIDTH_PX), 256, MAX_WIDTH_PX);

  // Resize down to maxWidth (keep aspect)
  const targetW = Math.min(meta.width, cappedMaxWidth);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  // Raw RGBA
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  // Cell sizing
  const cs = clamp(Math.floor(cellSize), 4, 80);
  const half = cs / 2;

  // Strength: 100 = baseline; allow up to 220 in UI comfortably
  // Map strength -> coverage multiplier + curve for midtones
  const s = clamp(Number(strength ?? 100), 0, 260) / 100; // 0.0 .. 2.6
  // Higher strength => slightly lower gamma => denser midtones
  const gamma = clamp(1.0 / (0.85 + 0.55 * s), 0.45, 1.2);

  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);

  // Build SVG shapes in an array (less memory churn than string concat)
  const shapes = [];

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      // Alpha-weighted (premultiplied) sampling:
      // rSum += r * a, aSum += a, final r = rSum/aSum
      let rSum = 0,
        gSum = 0,
        bSum = 0;
      let aSum = 0; // sum of alpha in 0..1
      let aRawSum = 0; // sum of raw alpha 0..255 (for avg alpha)

      let count = 0;

      for (let y = y0; y < y1; y++) {
        const rowIdx = y * w;
        for (let x = x0; x < x1; x++) {
          const idx = (rowIdx + x) * 4;
          const a255 = data[idx + 3];

          count++;

          // Ignore near-transparent pixels for color sampling to prevent background wash
          if (a255 <= ALPHA_EPS) continue;

          const a = a255 / 255;
          aSum += a;
          aRawSum += a255;

          rSum += data[idx] * a;
          gSum += data[idx + 1] * a;
          bSum += data[idx + 2] * a;
        }
      }

      if (count === 0) continue;

      // If effectively transparent cell, skip output entirely
      if (aSum <= 0) continue;

      // Average alpha for edge softness (0..1)
      const avgAlpha = clamp(aRawSum / (count * 255), 0, 1);

      // Alpha-weighted average color
      const r = Math.round(rSum / aSum);
      const g = Math.round(gSum / aSum);
      const b = Math.round(bSum / aSum);

      // Darkness based on luminance, but with strength curve
      const lum = luminance255(r, g, b); // 0..255
      const baseDark = 1 - lum / 255; // 0..1

      // Apply strength multiplier, then gamma curve for midtones
      let dark = clamp(baseDark * s, 0, 1);
      dark = Math.pow(dark, gamma);

      // Also respect alpha: semi-transparent edges should not produce full dots
      // Use sqrt so edges stay present but don't "fill in" too hard
      const alphaScale = Math.sqrt(avgAlpha);

      // Dot size:
      // - baseline gives visible dots even in light areas
      // - allow slight overlap (up to 1.15 * half) to reduce washed-out look
      const minFrac = 0.08; // raises floor so output doesn't vanish
      const maxFrac = 1.15; // allows overlap (darkens overall look)
      let radius = half * (minFrac + (maxFrac - minFrac) * dark);
      radius *= alphaScale;

      // Skip near-invisible dots
      if (radius < 0.35) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;
      const fillOpacity = avgAlpha.toFixed(4);

      if (dotShape === "square") {
        const size = radius * 2;
        shapes.push(
          `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(
            2
          )}" width="${size.toFixed(2)}" height="${size.toFixed(
            2
          )}" fill="${fill}" fill-opacity="${fillOpacity}" />`
        );
      } else if (dotShape === "ellipse") {
        const rx = radius * 1.18;
        const ry = radius * 0.86;
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

  const out = await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return { png: out, width: w, height: h, cellsUsed: shapes.length };
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
  cellSize: z.number().int().min(4).max(80).default(12),
  maxWidth: z.number().int().min(256).max(MAX_WIDTH_PX).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  strength: z.number().int().min(0).max(260).default(120),
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

  const started = Date.now();

  try {
    logWithReq(req, "PROCESS PARAMS:", { key, cellSize, maxWidth, dotShape, strength });

    // 1) HeadObject check
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

    // 2) Download
    const obj = await s3.send(new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key }));
    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    if (inputBuffer.length > MAX_UPLOAD_BYTES) {
      return res
        .status(413)
        .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
    }

    // 3) Validate image
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

    // 4) Generate halftone (ALPHA-CORRECT)
    const { png, width, height, cellsUsed } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth,
      dotShape,
      strength,
    });

    // 5) Store output
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
      new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: outputKey }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - started;
    logWithReq(req, "PROCESS OK:", { outputKey, width, height, ms, cellsUsed });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      width,
      height,
      cellsUsed,
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
