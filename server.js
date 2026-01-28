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
  "R2_SECRET_ACCESS_KEY",
];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000); // 60MP safe-ish
const MAX_CELLS = Number(process.env.MAX_CELLS || 450_000); // guard for CPU/RAM

// Optional: reduce Sharp memory pressure on small instances
// sharp.cache(false);
// sharp.concurrency(1);

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
  return String(name).replace(/[^a-zA-Z0-9._-]/g, "_");
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
// Allow your prod domains + ANY Cloudflare Pages preview under *.pages.dev
const ALLOWED_EXACT = new Set([
  "https://moretranz.com",
  "https://www.moretranz.com",
  "https://moretranz-halftone.pages.dev",
]);

function isAllowedOrigin(origin) {
  if (!origin) return true; // server-to-server / curl
  if (ALLOWED_EXACT.has(origin)) return true;

  // Allow any Pages project / preview: https://<anything>.pages.dev
  // (Includes preview URLs like https://<hash>.<project>.pages.dev)
  try {
    const u = new URL(origin);
    if (u.protocol !== "https:") return false;
    if (u.hostname.endsWith(".pages.dev")) return true;
  } catch {
    return false;
  }

  return false;
}

app.use(
  cors({
    origin: (origin, cb) => {
      if (isAllowedOrigin(origin)) return cb(null, true);
      // Clean deny (don’t throw; throwing can appear like random “Failed to fetch”)
      console.warn(`CORS denied for origin: ${origin}`);
      return cb(null, false);
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
 *  Halftone generator (alpha-correct)
 *  --------------------------*/
/**
 * Full-color halftone:
 * - alpha-weighted avg color per cell (fixes washed-out / rectangle issues)
 * - dot color = sampled RGB
 * - dot radius = based on luminance + Strength
 * - output = transparent PNG
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  // 12" @ 300dpi = 3600 px wide
  const targetW = Math.min(meta.width, maxWidth);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  // raw RGBA pixels
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const cs = clamp(Math.floor(cellSize), 4, 80);
  const half = cs / 2;

  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);
  const cells = cols * rows;

  if (cells > MAX_CELLS) {
    // Recommend a larger cellSize instead of burning CPU/RAM
    const suggested = Math.ceil(Math.sqrt((w * h) / MAX_CELLS));
    throw new Error(
      `Too many cells (${cells.toLocaleString()}) for this image/settings. Increase Cell size to ~${suggested}+ or reduce Max width.`
    );
  }

  // Strength: 100 = baseline, higher = darker/more coverage
  // We apply strength primarily by increasing dot radius via a curve.
  const s = clamp(Number(strength) || 100, 40, 220) / 100; // 0.40..2.20
  // gamma < 1 increases midtone darkness; gamma > 1 lightens midtones
  const gamma = clamp(1 / s, 0.35, 2.2);

  let shapes = "";

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      // Alpha-weighted sums
      let aSum = 0;
      let rSum = 0;
      let gSum = 0;
      let bSum = 0;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = (y * w + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const a = data[idx + 3]; // 0..255

          if (a === 0) continue;

          aSum += a;
          rSum += r * a;
          gSum += g * a;
          bSum += b * a;
        }
      }

      // If fully transparent cell, skip (THIS fixes “rectangle” output)
      if (aSum <= 0) continue;

      // Average RGB weighted by alpha
      const r = Math.round(rSum / aSum);
      const g = Math.round(gSum / aSum);
      const b = Math.round(bSum / aSum);

      // Average alpha (normalized to 0..1)
      const avgA = clamp(aSum / (255 * (x1 - x0) * (y1 - y0)), 0, 1);
      if (avgA < 0.02) continue; // tiny alpha -> skip noise

      // Luminance + strength curve
      const lum = luminance255(r, g, b); // 0..255
      const darkness = 1 - lum / 255; // 0..1

      // Strength influences darkness curve + overall coverage
      const curved = Math.pow(clamp(darkness, 0, 1), gamma);
      const coverage = clamp(curved * s, 0, 1);

      // Radius: allow larger max coverage than before
      const radius = clamp(half * (0.05 + 0.95 * coverage), 0, half);

      if (radius < 0.35) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;
      const fillOpacity = avgA.toFixed(4);

      if (dotShape === "square") {
        const size = radius * 2;
        shapes += `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(
          2
        )}" width="${size.toFixed(2)}" height="${size.toFixed(
          2
        )}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
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

  return { png: out, width: w, height: h, cells };
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
  // allow 3600 for 12" @ 300dpi
  maxWidth: z.number().int().min(256).max(3600).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  // Strength: 100 baseline, higher = darker/more coverage
  strength: z.number().int().min(40).max(220).default(120),
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

    // 2) Download object
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

    // 4) Generate halftone
    const t0 = Date.now();
    let result;
    try {
      result = await makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength });
    } catch (e) {
      // Return as 400 (user settings) instead of 500
      return res.status(400).json(errJson("SETTINGS_TOO_HEAVY", String(e?.message || e)));
    }
    const { png, width, height, cells } = result;

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
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
      }),
      { expiresIn: 600 }
    );

    const ms = Date.now() - t0;
    logWithReq(req, "PROCESS OK:", { outputKey, width, height, cells, ms });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      stats: { width, height, cells, ms },
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
