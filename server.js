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
  HeadObjectCommand
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "2mb" }));

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

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000); // 60MP safety

// 12" @ 300dpi = 3600px
const DEFAULT_MAX_WIDTH = Number(process.env.DEFAULT_MAX_WIDTH || 3600);
const MAX_MAX_WIDTH = Number(process.env.MAX_MAX_WIDTH || 4000);

const ALLOWED_ORIGINS = new Set([
  "https://moretranz-halftone.pages.dev",
  "https://moretranz.com",
  "https://www.moretranz.com",
  "http://localhost:5173",
  "http://localhost:3000",
  "http://localhost:8080"
]);

app.use(
  cors({
    origin: (origin, cb) => {
      // allow server-to-server / curl / Postman
      if (!origin) return cb(null, true);
      if (ALLOWED_ORIGINS.has(origin)) return cb(null, true);
      return cb(new Error(`CORS blocked for origin: ${origin}`));
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
    optionsSuccessStatus: 204
  })
);

/** ---------------------------
 *  S3 (Cloudflare R2)
 *  --------------------------*/
const s3 = new S3Client({
  region: process.env.R2_REGION, // "auto" typically
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
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
  return String(name || "file")
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .slice(0, 180);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function luminance255(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function uid() {
  return crypto.randomBytes(6).toString("hex");
}

/**
 * Strength curve that actually darkens:
 * - strength = 100 => baseline
 * - strength > 100 => more dot coverage in midtones/highlights
 * - strength < 100 => lighter
 *
 * coverage = 1 - (1 - darkness)^s, where s=strength/100
 */
function coverageFromDarkness(darkness01, strength) {
  const s = clamp(strength / 100, 0.25, 3.0); // allow 25..300 mapped
  return 1 - Math.pow(1 - clamp(darkness01, 0, 1), s);
}

/**
 * Raster halftone renderer:
 * - reads RGBA
 * - averages per cell using alpha-weighted color
 * - draws dots directly into output RGBA buffer
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  const base = sharp(inputBuffer, { failOn: "none" });
  const meta = await base.metadata();

  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  const targetW = Math.min(meta.width, maxWidth);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const cs = clamp(Math.floor(cellSize), 4, 80);
  const half = cs / 2;

  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);

  const out = Buffer.alloc(w * h * 4, 0);

  function blendPixel(pxIdx, r, g, b, a01) {
    if (a01 <= 0) return;

    const i = pxIdx * 4;

    const pr = out[i];
    const pg = out[i + 1];
    const pb = out[i + 2];
    const pa01 = out[i + 3] / 255;

    const outA = a01 + pa01 * (1 - a01);
    if (outA <= 0) return;

    const outR = (r * a01 + pr * pa01 * (1 - a01)) / outA;
    const outG = (g * a01 + pg * pa01 * (1 - a01)) / outA;
    const outB = (b * a01 + pb * pa01 * (1 - a01)) / outA;

    out[i] = clamp(Math.round(outR), 0, 255);
    out[i + 1] = clamp(Math.round(outG), 0, 255);
    out[i + 2] = clamp(Math.round(outB), 0, 255);
    out[i + 3] = clamp(Math.round(outA * 255), 0, 255);
  }

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      let rSum = 0, gSum = 0, bSum = 0;
      let aSum = 0;
      let count = 0;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = (y * w + x) * 4;
          const a = data[idx + 3];
          if (a > 0) {
            rSum += data[idx] * a;
            gSum += data[idx + 1] * a;
            bSum += data[idx + 2] * a;
            aSum += a;
          }
          count++;
        }
      }

      if (aSum <= 0) continue;

      const r = Math.round(rSum / aSum);
      const g = Math.round(gSum / aSum);
      const b = Math.round(bSum / aSum);

      const aAvg = aSum / count;
      const a01 = clamp(aAvg / 255, 0, 1);

      const lum = luminance255(r, g, b);
      const darkness = 1 - lum / 255;
      const coverage = coverageFromDarkness(darkness, strength);

      const s = clamp(strength / 100, 0.25, 3.0);
      const minFrac = clamp(0.03 * (s - 1), 0, 0.10);
      const frac = clamp(minFrac + coverage * (1 - minFrac), 0, 1);

      let rx = half * frac;
      let ry = half * frac;

      if (dotShape === "ellipse") {
        rx = rx * 1.25;
        ry = ry * 0.85;
        rx = clamp(rx, 0, half * 1.4);
        ry = clamp(ry, 0, half * 1.2);
      }

      if (rx < 0.35 || ry < 0.35) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      let minX = clamp(Math.floor(cx - rx), 0, w - 1);
      let maxX = clamp(Math.ceil(cx + rx), 0, w - 1);
      let minY = clamp(Math.floor(cy - ry), 0, h - 1);
      let maxY = clamp(Math.ceil(cy + ry), 0, h - 1);

      if (dotShape === "square") {
        const halfSize = rx;
        minX = clamp(Math.floor(cx - halfSize), 0, w - 1);
        maxX = clamp(Math.ceil(cx + halfSize), 0, w - 1);
        minY = clamp(Math.floor(cy - halfSize), 0, h - 1);
        maxY = clamp(Math.ceil(cy + halfSize), 0, h - 1);

        for (let y = minY; y <= maxY; y++) {
          const rowBase = y * w;
          for (let x = minX; x <= maxX; x++) {
            blendPixel(rowBase + x, r, g, b, a01);
          }
        }
      } else {
        const invRx2 = 1 / (rx * rx);
        const invRy2 = 1 / (ry * ry);

        for (let y = minY; y <= maxY; y++) {
          const dy = y + 0.5 - cy;
          const dy2 = dy * dy * invRy2;
          const rowBase = y * w;

          for (let x = minX; x <= maxX; x++) {
            const dx = x + 0.5 - cx;
            const v = dx * dx * invRx2 + dy2;
            if (v <= 1) {
              blendPixel(rowBase + x, r, g, b, a01);
            }
          }
        }
      }
    }
  }

  const png = await sharp(out, { raw: { width: w, height: h, channels: 4 } })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();

  return { png, width: w, height: h };
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
    const imageId = `img_${Date.now()}_${uid()}`;
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
  cellSize: z.number().int().min(4).max(80).default(10),
  maxWidth: z.number().int().min(256).max(MAX_MAX_WIDTH).default(DEFAULT_MAX_WIDTH),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  // Only the DTF-aligned label values
  strength: z.number().int().refine((v) => [100, 150, 200, 250].includes(v), {
    message: "Strength must be one of: 100, 150, 200, 250"
  }).default(150)
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json(
      errJson("BAD_REQUEST", "Invalid request", { issues: parsed.error.flatten() })
    );
  }

  const { key, cellSize, maxWidth, dotShape, strength } = parsed.data;

  try {
    logWithReq(req, "PROCESS PARAMS:", { key, cellSize, maxWidth, dotShape, strength });

    try {
      const head = await s3.send(
        new HeadObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key })
      );
      if (typeof head.ContentLength === "number" && head.ContentLength > MAX_UPLOAD_BYTES) {
        return res
          .status(413)
          .json(errJson("TOO_LARGE", `File exceeds max upload size (${MAX_UPLOAD_BYTES} bytes)`));
      }
    } catch (e) {
      logWithReq(req, "HeadObject warning:", String(e?.name || e));
    }

    const obj = await s3.send(
      new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key })
    );

    if (!obj.Body) return res.status(404).json(errJson("NOT_FOUND", "Input object not found"));

    const inputBuffer = await streamToBuffer(obj.Body);

    if (inputBuffer.length <= 0) {
      return res.status(400).json(errJson("BAD_IMAGE", "Uploaded file is empty"));
    }
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
          format: meta.format || null
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
          maxPixels: MAX_IMAGE_PIXELS
        })
      );
    }

    const t0 = Date.now();

    const { png, width, height } = await makeColorHalftonePng(inputBuffer, {
      cellSize,
      maxWidth,
      dotShape,
      strength
    });

    const outId = `ht_${Date.now()}_${uid()}`;
    const outputKey = `outputs/${outId}.png`;

    await s3.send(
      new PutObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
        Body: png,
        ContentType: "image/png"
      })
    );

    // Force a real download (best UX) via response headers on signed URL:
    const downloadCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outputKey,
      ResponseContentType: "image/png",
      ResponseContentDisposition: `attachment; filename="halftone.png"`
    });

    const downloadUrl = await getSignedUrl(s3, downloadCmd, { expiresIn: 600 });

    const ms = Date.now() - t0;

    logWithReq(req, "PROCESS OK:", { outputKey, width, height, ms });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      downloadUrl,
      ms
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
