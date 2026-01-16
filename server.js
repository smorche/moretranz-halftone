import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();

/**
 * NOTE:
 * - Requests to /upload-url and /process are small JSON; keep limit low.
 * - The actual image data is uploaded directly to R2 via a presigned URL.
 */
app.use(express.json({ limit: "1mb" }));

app.use(
  cors({
    origin: true, // allow all origins (safe for now); tighten later to your Shopify domain(s)
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  })
);

// --- Required env vars ---
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

const s3 = new S3Client({
  region: process.env.R2_REGION, // usually "auto" for Cloudflare R2
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

// --- Basic routes ---
app.get("/", (_req, res) => {
  res
    .status(200)
    .type("text/plain")
    .send("MoreTranz Halftone API is running. Try GET /health");
});

app.get("/health", (_req, res) => res.json({ ok: true }));

// --- Upload URL (presigned PUT) ---
const UploadUrlRequest = z.object({
  filename: z.string().min(1),
  contentType: z.enum(["image/png", "image/jpeg", "image/webp"]),
  contentLength: z.number().int().positive().max(25 * 1024 * 1024)
});

app.post("/v1/halftone/upload-url", async (req, res) => {
  try {
    const parsed = UploadUrlRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: {
          code: "BAD_REQUEST",
          message: "Invalid request",
          details: parsed.error.flatten()
        }
      });
    }

    const { filename, contentType } = parsed.data;

    const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const key = `uploads/${imageId}/${safeName}`;

    // IMPORTANT for R2 presigned PUT:
    // Keep SignedHeaders minimal. We'll only require Content-Type.
    const cmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key,
      ContentType: contentType
    });

    const uploadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });

    return res.json({
      imageId,
      key,
      uploadUrl,
      headers: { "Content-Type": contentType },
      maxBytes: 25 * 1024 * 1024
    });
  } catch (err) {
    console.error("upload-url error:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to create upload URL" }
    });
  }
});

// --- Download URL (presigned GET) ---
const DownloadUrlRequest = z.object({
  key: z.string().min(1)
});

app.post("/v1/halftone/download-url", async (req, res) => {
  try {
    const parsed = DownloadUrlRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: {
          code: "BAD_REQUEST",
          message: "Invalid request",
          details: parsed.error.flatten()
        }
      });
    }

    const cmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: parsed.data.key
    });

    const downloadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });
    return res.json({ downloadUrl });
  } catch (err) {
    console.error("download-url error:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to create download URL" }
    });
  }
});

// --- Helpers ---
async function streamToBuffer(body) {
  // AWS SDK v3 GetObject Body can be a stream in Node
  const chunks = [];
  for await (const chunk of body) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

/**
 * Full-color halftone:
 * - Sample the image on a grid (cellSize)
 * - For each cell, compute average RGB and luminance
 * - Use luminance to control dot size (darker -> bigger)
 * - Render dots on a transparent background via SVG -> PNG
 */
async function makeHalftonePng({
  inputBuffer,
  cellSize,
  maxWidth,
  dotShape
}) {
  const img = sharp(inputBuffer, { failOn: "none" });

  // Normalize orientation and convert to RGB
  // (RGB-only mode per your requirement)
  const metadata = await img.metadata();
  const inW = metadata.width || 0;
  const inH = metadata.height || 0;
  if (!inW || !inH) {
    throw new Error("Unable to read input image dimensions");
  }

  // Resize down if needed (keeps cost predictable)
  const targetW = maxWidth && inW > maxWidth ? maxWidth : inW;

  const resized = img
    .rotate()
    .resize({ width: targetW, withoutEnlargement: true })
    .removeAlpha() // ensure we work in RGB; output will be transparent anyway
    .toColourspace("rgb");

  const { data, info } = await resized
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;
  const channels = info.channels; // should be 3 (RGB)
  if (channels < 3) {
    throw new Error(`Unexpected channel count: ${channels}`);
  }

  const cs = clamp(cellSize, 4, 64);
  const rMax = cs * 0.5;

  // Build SVG dots (transparent background)
  // For large images, SVG can get big; we cap maxWidth and cellSize range.
  let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  svg += `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;

  for (let y = 0; y < height; y += cs) {
    for (let x = 0; x < width; x += cs) {
      // Compute average color within this cell
      const xEnd = Math.min(x + cs, width);
      const yEnd = Math.min(y + cs, height);

      let rSum = 0,
        gSum = 0,
        bSum = 0,
        count = 0;

      for (let yy = y; yy < yEnd; yy++) {
        for (let xx = x; xx < xEnd; xx++) {
          const idx = (yy * width + xx) * channels;
          rSum += data[idx];
          gSum += data[idx + 1];
          bSum += data[idx + 2];
          count++;
        }
      }

      if (count === 0) continue;

      const r = Math.round(rSum / count);
      const g = Math.round(gSum / count);
      const b = Math.round(bSum / count);

      // Relative luminance (sRGB approximation)
      // 0 = black, 255 = white
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

      // Dot size: darker -> larger
      // Scale lum (0..255) into radius (rMax..0)
      const t = 1 - lum / 255; // 1=dark, 0=light
      const radius = rMax * t;

      // Skip near-white dots to reduce SVG size
      if (radius < 0.35) continue;

      const cx = x + cs / 2;
      const cy = y + cs / 2;
      const fill = `rgb(${r},${g},${b})`;

      if (dotShape === "square") {
        const size = radius * 2;
        const sx = cx - size / 2;
        const sy = cy - size / 2;
        svg += `<rect x="${sx.toFixed(2)}" y="${sy.toFixed(2)}" width="${size.toFixed(
          2
        )}" height="${size.toFixed(2)}" fill="${fill}" />`;
      } else if (dotShape === "ellipse") {
        // Slightly stretched ellipse for a different look
        const rx = radius;
        const ry = radius * 0.75;
        svg += `<ellipse cx="${cx.toFixed(2)}" cy="${cy.toFixed(
          2
        )}" rx="${rx.toFixed(2)}" ry="${ry.toFixed(
          2
        )}" fill="${fill}" />`;
      } else {
        // circle (default)
        svg += `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(
          2
        )}" r="${radius.toFixed(2)}" fill="${fill}" />`;
      }
    }
  }

  svg += `</svg>`;

  // Render SVG -> PNG with transparent background
  const outPng = await sharp(Buffer.from(svg))
    .png() // transparent output
    .toBuffer();

  return { outPng, width, height };
}

// --- Process endpoint (server-side processing) ---
const ProcessRequest = z.object({
  key: z.string().min(1),

  // Halftone controls
  cellSize: z.number().int().min(4).max(64).default(12),
  maxWidth: z.number().int().min(200).max(4000).default(2000),
  dotShape: z.enum(["circle", "square", "ellipse"]).default("circle")
});

app.post("/v1/halftone/process", async (req, res) => {
  try {
    const parsed = ProcessRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: {
          code: "BAD_REQUEST",
          message: "Invalid request",
          details: parsed.error.flatten()
        }
      });
    }

    const { key, cellSize, maxWidth, dotShape } = parsed.data;

    // 1) Download original from R2
    const getCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key
    });

    const getResp = await s3.send(getCmd);
    if (!getResp.Body) {
      return res.status(404).json({
        error: { code: "NOT_FOUND", message: "Source image not found" }
      });
    }

    const inputBuffer = await streamToBuffer(getResp.Body);

    // 2) Generate halftone PNG (transparent)
    const { outPng, width, height } = await makeHalftonePng({
      inputBuffer,
      cellSize,
      maxWidth,
      dotShape
    });

    // 3) Upload result to R2 (server-side, no CORS issues)
    const outKey = key
      .replace(/^uploads\//, "outputs/")
      .replace(/\.[a-zA-Z0-9]+$/, "") + `_halftone_${dotShape}_c${cellSize}.png`;

    const putCmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey,
      Body: outPng,
      ContentType: "image/png"
    });

    await s3.send(putCmd);

    // 4) Return presigned download URL
    const dlCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey
    });

    const downloadUrl = await getSignedUrl(s3, dlCmd, { expiresIn: 600 });

    return res.json({
      outputKey: outKey,
      downloadUrl,
      meta: {
        width,
        height,
        cellSize,
        dotShape,
        transparent: true,
        mode: "RGB"
      }
    });
  } catch (err) {
    console.error("process error:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to process halftone" }
    });
  }
});

// --- Start server ---
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
