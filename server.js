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

// Upload bytes limit (client -> R2)
const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 25 * 1024 * 1024); // 25MB default

// Total pixels limit we will accept for decoding (server-side protection)
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000); // default 60MP

// Halftone runtime guards
const HARD_MAX_WIDTH = Number(process.env.HARD_MAX_WIDTH || 3600); // 12" @ 300dpi = 3600px
const MIN_CELL = Number(process.env.MIN_CELL || 6);
const MAX_CELL = Number(process.env.MAX_CELL || 48);

// Prevent “death by dots”: w*h/cell^2 grows quickly.
// Example: 3600x3600 with cell=8 -> ~202,500 cells (ok-ish); cell=6 -> 360,000+ (heavy)
const MAX_CELLS_ESTIMATE = Number(process.env.MAX_CELLS_ESTIMATE || 260_000);

// NOTE: You can optionally set this env var to allow additional origins.
// Example: CORS_ORIGINS=https://moretranz-halftone.pages.dev,https://moretranz.com
function parseCorsOrigins(val) {
  if (!val || val.trim() === "" || val.trim() === "*") return null; // null -> use default set below
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
      // allow server-to-server / curl / Postman (no Origin header)
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
  return String(name || "file").replace(/[^a-zA-Z0-9._-]/g, "_");
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function luminance255(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Full-color halftone (transparent output) with correct alpha handling:
 * - For each cell, compute alpha-weighted average color (ignoring near-transparent pixels)
 * - coverage = alphaSum / totalPixelsInCell
 * - dot radius is based on luminance + coverage (keeps cutout edges crisp)
 * - dot opacity uses coverage (NOT average alpha), improved “Strength” control
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
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

  // Safety guard for CPU/RAM
  if (cellCount > MAX_CELLS_ESTIMATE) {
    throw new Error(
      `Settings too heavy: estimated cells=${cellCount}. Increase Cell size or reduce Max width.`
    );
  }

  // Raw RGBA
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  // Strength mapping: 100 = baseline; allow richer/darker by pushing beyond 100
  const s = clamp((Number(strength) || 100) / 100, 0, 2); // 0..2
  const curveExp = 1 / Math.max(0.35, s); // stronger => darker midtones
  const radiusGain = 0.90 + 0.70 * s; // stronger => more dot coverage
  const alphaGain = clamp(0.90 + 0.35 * s, 0, 1.35); // stronger => “ink gain”
  const minDot = 0.28;

  const half = cs / 2;
  const shapes = [];

  // ignore very low-alpha pixels so background doesn’t wash out the cell average
  const ALPHA_CUTOFF = 8 / 255; // ~3%

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      let rPremul = 0,
        gPremul = 0,
        bPremul = 0;

      let alphaSum = 0; // sum of alphas (0..1)
      let considered = 0;

      for (let y = y0; y < y1; y++) {
        const rowBase = (y * w) * 4;
        for (let x = x0; x < x1; x++) {
          const idx = rowBase + x * 4;
          const a = data[idx + 3] / 255;
          considered++;

          if (a <= ALPHA_CUTOFF) continue;

          rPremul += data[idx] * a;
          gPremul += data[idx + 1] * a;
          bPremul += data[idx + 2] * a;
          alphaSum += a;
        }
      }

      if (!considered) continue;

      const coverage = clamp(alphaSum / considered, 0, 1);
      if (coverage <= 0.005) continue; // preserves cutout shape

      if (alphaSum <= 1e-6) continue;

      const r = Math.round(rPremul / alphaSum);
      const g = Math.round(gPremul / alphaSum);
      const b = Math.round(bPremul / alphaSum);

      const lum = luminance255(r, g, b);
      const darkness = 1 - lum / 255;
      const darknessCurved = clamp(Math.pow(darkness, curveExp), 0, 1);

      // coverage influences edge cells so we don’t “fill rectangles”
      const combined = clamp(darknessCurved * (0.55 + 0.45 * coverage), 0, 1);

      const radius = half * clamp((0.06 + 0.94 * combined) * radiusGain, 0, 1);
      if (radius < minDot) continue;

      const op = clamp(coverage * alphaGain, 0, 1).toFixed(4);

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;
      const fill = `rgb(${r},${g},${b})`;

      if (dotShape === "square") {
        const size = radius * 2;
        shapes.push(
          `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(
            2
          )}" width="${size.toFixed(2)}" height="${size.toFixed(
            2
          )}" fill="${fill}" fill-opacity="${op}" />`
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

  return { png: out, width: w, height: h, cellCount };
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

    // IMPORTANT: do NOT include ContentLength in signed headers unless client sends it exactly.
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

const DownloadUrlRequest = z.object({ key: z.string().min(1) });

app.post("/v1/halftone/download-url", async (req, res) => {
  try {
    const parsed = DownloadUrlRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json(
        errJson("BAD_REQUEST", "Invalid request", {
          issues: parsed.error.flatten(),
        })
      );
    }

    const cmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: parsed.data.key,
    });

    const downloadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });
    return res.json({ downloadUrl });
  } catch (e) {
    console.error(`[${req._rid}] download-url error`, e);
    return res.status(500).json(errJson("INTERNAL", "Failed to create download URL"));
  }
});

const ProcessRequest = z.object({
  key: z.string().min(1),

  // Keep customer-adjustable but safe defaults
  cellSize: z.number().int().min(MIN_CELL).max(MAX_CELL).default(12),

  // Accept up to 12" @ 300dpi (3600px); hard clamp enforced server-side
  maxWidth: z.number().int().min(256).max(HARD_MAX_WIDTH).default(2000),

  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),

  // Strength: 100 baseline; allow richer/darker up to 200 safely
  strength: z.number().int().min(0).max(200).default(100),
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

    // 1) HeadObject check (exists, size guard when available)
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

    // 2) Download object bytes from R2
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

    // 3) Validate that Sharp can read it
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

    // 4) Generate halftone PNG (transparent)
    const { png, width, height, cellCount } = await makeColorHalftonePng(inputBuffer, {
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

    // 6) Provide a signed download URL for output
    const downloadUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
      }),
      { expiresIn: 600 }
    );

    logWithReq(req, "PROCESS OK:", { outputKey, width, height, cellCount });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      stats: { width, height, cellCount },
      downloadUrl,
    });
  } catch (e) {
    const msg = String(e?.message || e);
    console.error(`[${req._rid}] HALFTONE ERROR:`, e);

    // If we can classify “too heavy” quickly, return a 400 with a helpful message
    if (msg.includes("Settings too heavy")) {
      return res.status(400).json(errJson("TOO_HEAVY", msg, { maxCells: MAX_CELLS_ESTIMATE }));
    }

    return res.status(500).json(errJson("INTERNAL", "Failed to process halftone"));
  }
});

/** ---------------------------
 *  Start
 *  --------------------------*/
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
