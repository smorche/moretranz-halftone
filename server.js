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

const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES || 40 * 1024 * 1024); // 40MB default
const MAX_IMAGE_PIXELS = Number(process.env.MAX_IMAGE_PIXELS || 60_000_000); // 60MP default

// Alpha handling knobs
const ALPHA_PIXEL_THRESHOLD = Number(process.env.ALPHA_PIXEL_THRESHOLD || 8); // 0..255; treat <= this as transparent
const CELL_ALPHA_COVERAGE_MIN = Number(process.env.CELL_ALPHA_COVERAGE_MIN || 0.03); // 0..1; skip cells below this coverage

function parseCorsOrigins(val) {
  if (!val || val.trim() === "" || val.trim() === "*") return "*";
  return val
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

// Default allow-list (can extend via env ALLOWED_ORIGINS="https://...,https://..."
const DEFAULT_ALLOWED = [
  "https://moretranz-halftone.pages.dev",
  "https://moretranz.com",
  "https://www.moretranz.com",
];

const ENV_ALLOWED = parseCorsOrigins(process.env.ALLOWED_ORIGINS || "");
const ALLOWED_ORIGINS =
  ENV_ALLOWED === "*"
    ? "*"
    : new Set([...(Array.isArray(ENV_ALLOWED) ? ENV_ALLOWED : []), ...DEFAULT_ALLOWED]);

app.use(
  cors({
    origin: (origin, cb) => {
      // allow server-to-server / curl / Postman (no Origin header)
      if (!origin) return cb(null, true);

      if (ALLOWED_ORIGINS === "*") return cb(null, true);

      // exact match allow-list
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
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function luminance255(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Full-color halftone (alpha-correct + stronger strength):
 * - samples ONLY pixels with alpha > threshold
 * - uses alpha-coverage threshold to avoid "faint rectangle" output
 * - dot color = alpha-weighted sampled RGB
 * - dot opacity = cell alpha coverage (preserves smooth edges)
 * - dot size uses luminance curve + strength multiplier (stronger midtones)
 */
async function makeColorHalftonePng(inputBuffer, { cellSize, maxWidth, dotShape, strength }) {
  const t0 = Date.now();

  // Decode & normalize to RGBA
  const base = sharp(inputBuffer, { failOn: "none" });

  const meta = await base.metadata();
  if (!meta.width || !meta.height) throw new Error("Could not read image dimensions");

  const pixels = meta.width * meta.height;
  if (pixels > MAX_IMAGE_PIXELS) {
    throw new Error(`Image too large: ${pixels} pixels exceeds limit ${MAX_IMAGE_PIXELS}`);
  }

  // Resize down to maxWidth (keep aspect); NOTE: 12" @ 300dpi => 3600px (allowed)
  const targetW = Math.min(meta.width, maxWidth);
  const resized = base.resize({ width: targetW, withoutEnlargement: true });

  const rMeta = await resized.metadata();
  const w = rMeta.width;
  const h = rMeta.height;
  if (!w || !h) throw new Error("Could not read resized dimensions");

  // Get raw RGBA pixels
  const { data } = await resized.ensureAlpha().raw().toBuffer({ resolveWithObject: true });

  const cs = clamp(Math.floor(cellSize), 4, 80);
  const half = cs / 2;

  const cols = Math.ceil(w / cs);
  const rows = Math.ceil(h / cs);

  // Strength tuning:
  // strength=100 baseline. Allow stronger midtone boost without crushing highlights.
  // We do:
  //   darkness = 1 - lum/255
  //   darknessCurve = darkness^(gamma)  (gamma<1 boosts midtones)
  //   sizeFactor = clamp(darknessCurve * (strength/100), 0, 1)
  const s = clamp(Number(strength ?? 100), 0, 250);
  const strengthMult = s / 100;
  const gamma = 0.70; // <1 boosts midtones; adjust here if needed

  let shapes = "";
  let cellsUsed = 0;

  for (let row = 0; row < rows; row++) {
    const y0 = row * cs;
    const y1 = Math.min(y0 + cs, h);

    for (let col = 0; col < cols; col++) {
      const x0 = col * cs;
      const x1 = Math.min(x0 + cs, w);

      const cellPixelCount = (x1 - x0) * (y1 - y0);
      if (cellPixelCount <= 0) continue;

      // Alpha-correct sampling:
      // - only include pixels with alpha > threshold
      // - compute alpha coverage over the full cell to preserve edges
      let sumAlpha = 0;        // sum of alpha (0..255) for included pixels
      let sumRA = 0;           // sum of r*alpha
      let sumGA = 0;           // sum of g*alpha
      let sumBA = 0;           // sum of b*alpha

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = (y * w + x) * 4;
          const a = data[idx + 3];

          // Track coverage over whole cell (including edge pixels)
          if (a > 0) {
            // Only sample color if above threshold (prevents transparent background diluting RGB)
            if (a > ALPHA_PIXEL_THRESHOLD) {
              sumAlpha += a;
              sumRA += data[idx] * a;
              sumGA += data[idx + 1] * a;
              sumBA += data[idx + 2] * a;
            } else {
              // still count alpha towards coverage if it's >0 but below threshold?
              // For very soft edges, keep a tiny contribution to coverage but not color.
              // This avoids random background dots while preserving antialias edges.
              sumAlpha += a * 0.25;
            }
          }
        }
      }

      // Compute alpha coverage as a fraction of full cell area
      // coverage = average alpha / 255 over all cell pixels
      const coverage = clamp(sumAlpha / (cellPixelCount * 255), 0, 1);

      // Skip cells that are effectively transparent — THIS removes the rectangle artifact
      if (coverage < CELL_ALPHA_COVERAGE_MIN) continue;

      // If we have almost no sampled alpha above threshold, skip (avoid weird dots on faint edges)
      if (sumAlpha <= 0) continue;

      // Alpha-weighted average RGB
      const r = clamp(Math.round(sumRA / sumAlpha), 0, 255);
      const g = clamp(Math.round(sumGA / sumAlpha), 0, 255);
      const b = clamp(Math.round(sumBA / sumAlpha), 0, 255);

      // Luminance -> darkness
      const lum = luminance255(r, g, b);
      let darkness = 1 - lum / 255; // 0..1

      // Apply curve + strength multiplier
      const darknessCurve = Math.pow(clamp(darkness, 0, 1), gamma);
      const sizeFactor = clamp(darknessCurve * strengthMult, 0, 1);

      // radius based on sizeFactor; keep a small floor to avoid "missing" dark areas
      const radius = clamp(half * (0.10 + 0.90 * sizeFactor), 0, half);

      if (radius < 0.45) continue;

      const cx = x0 + (x1 - x0) / 2;
      const cy = y0 + (y1 - y0) / 2;

      const fill = `rgb(${r},${g},${b})`;

      // Use coverage as opacity (keeps smooth edges, but doesn't paint the whole rectangle)
      // Clamp so edges don't get *too* faint at high detail
      const fillOpacity = clamp(coverage, 0, 1).toFixed(4);

      if (dotShape === "square") {
        const size = radius * 2;
        shapes += `<rect x="${(cx - size / 2).toFixed(2)}" y="${(cy - size / 2).toFixed(
          2
        )}" width="${size.toFixed(2)}" height="${size.toFixed(
          2
        )}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      } else if (dotShape === "ellipse") {
        const rx = radius * 1.15;
        const ry = radius * 0.85;
        shapes += `<ellipse cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" rx="${rx.toFixed(
          2
        )}" ry="${ry.toFixed(2)}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      } else {
        shapes += `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${radius.toFixed(
          2
        )}" fill="${fill}" fill-opacity="${fillOpacity}" />`;
      }

      cellsUsed++;
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

  const ms = Date.now() - t0;
  return { png: out, width: w, height: h, ms, cellsUsed };
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
  cellSize: z.number().int().min(4).max(80).default(12),
  maxWidth: z.number().int().min(256).max(4000).default(2000), // 12"@300dpi = 3600px is allowed
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle"),
  strength: z.number().int().min(0).max(250).default(100),
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

    // 2) Download object bytes from R2 (server-side)
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

    // 4) Generate halftone PNG (transparent)
    const { png, width, height, ms, cellsUsed } = await makeColorHalftonePng(inputBuffer, {
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

    logWithReq(req, "PROCESS OK:", { outputKey, width, height, ms, cellsUsed });

    return res.json({
      ok: true,
      inputKey: key,
      outputKey,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape, strength },
      stats: { width, height, ms, cellsUsed },
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
