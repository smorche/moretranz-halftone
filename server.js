import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "2mb" }));

// CORS (broad for now)
app.use(
  cors({
    origin: true,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  })
);

// Explicit OPTIONS handler (helps with some proxies)
app.options("*", cors());

const requiredEnv = ["R2_ENDPOINT", "R2_BUCKET", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const s3 = new S3Client({
  region: process.env.R2_REGION, // typically "auto" for R2
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

app.get("/health", (_req, res) => res.json({ ok: true }));

// ------------------------------
// Helpers
// ------------------------------
async function streamToBuffer(body) {
  if (!body) return Buffer.alloc(0);

  // In Node 18+, AWS SDK v3 Body is usually a readable stream
  if (Buffer.isBuffer(body)) return body;

  const chunks = [];
  for await (const chunk of body) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function safeKey(key) {
  // Prevent path traversal-like weirdness and keep it in a bucket namespace
  // Allow slashes but remove leading slash and collapse ".."
  return key.replace(/^\/+/, "").replace(/\.\./g, "_");
}

// Draw a dot (circle or ellipse) into an RGBA buffer with alpha blending
function drawDotRGBA(out, width, height, cx, cy, rx, ry, r, g, b, a) {
  if (rx <= 0 || ry <= 0 || a <= 0) return;

  const x0 = Math.max(0, Math.floor(cx - rx));
  const x1 = Math.min(width - 1, Math.ceil(cx + rx));
  const y0 = Math.max(0, Math.floor(cy - ry));
  const y1 = Math.min(height - 1, Math.ceil(cy + ry));

  const invRx2 = 1 / (rx * rx);
  const invRy2 = 1 / (ry * ry);

  // alpha in 0..255
  const srcA = a;

  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    const dy2 = dy * dy;

    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const inside = (dx * dx) * invRx2 + dy2 * invRy2 <= 1;
      if (!inside) continue;

      const idx = (y * width + x) * 4;
      const dstR = out[idx];
      const dstG = out[idx + 1];
      const dstB = out[idx + 2];
      const dstA = out[idx + 3];

      // Source-over alpha blend
      const sA = srcA / 255;
      const dA = dstA / 255;
      const outA = sA + dA * (1 - sA);

      if (outA <= 0) {
        out[idx] = 0;
        out[idx + 1] = 0;
        out[idx + 2] = 0;
        out[idx + 3] = 0;
        continue;
      }

      const outR = (r * sA + dstR * dA * (1 - sA)) / outA;
      const outG = (g * sA + dstG * dA * (1 - sA)) / outA;
      const outB = (b * sA + dstB * dA * (1 - sA)) / outA;

      out[idx] = Math.round(outR);
      out[idx + 1] = Math.round(outG);
      out[idx + 2] = Math.round(outB);
      out[idx + 3] = Math.round(outA * 255);
    }
  }
}

// ------------------------------
// Upload URL
// ------------------------------
const UploadUrlRequest = z.object({
  filename: z.string().min(1),
  contentType: z.enum(["image/png", "image/jpeg", "image/webp"]),
  contentLength: z.number().int().positive().max(25 * 1024 * 1024)
});

app.post("/v1/halftone/upload-url", async (req, res) => {
  const parsed = UploadUrlRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const { filename, contentType } = parsed.data;

  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  const key = `uploads/${imageId}/${safeName}`;

  // IMPORTANT: do NOT include ContentLength in the signed headers
  // (R2/S3 presign can fail if the browser doesn't send exact content-length)
  const cmd = new PutObjectCommand({
    Bucket: process.env.R2_BUCKET,
    Key: key,
    ContentType: contentType
  });

  const uploadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });

  res.json({
    imageId,
    key,
    uploadUrl,
    headers: { "Content-Type": contentType },
    maxBytes: 25 * 1024 * 1024
  });
});

// ------------------------------
// Download URL
// ------------------------------
const DownloadUrlRequest = z.object({ key: z.string().min(1) });

app.post("/v1/halftone/download-url", async (req, res) => {
  const parsed = DownloadUrlRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const cmd = new GetObjectCommand({
    Bucket: process.env.R2_BUCKET,
    Key: safeKey(parsed.data.key)
  });

  const downloadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });
  res.json({ downloadUrl });
});

// ------------------------------
// Halftone Process
// ------------------------------
const ProcessRequest = z.object({
  key: z.string().min(1),
  cellSize: z.number().int().min(4).max(80).default(12),
  maxWidth: z.number().int().min(256).max(4000).default(2000),
  dotShape: z.enum(["circle", "ellipse"]).default("circle")
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const { key, cellSize, maxWidth, dotShape } = parsed.data;

  console.log("PROCESS PARAMS:", { key, cellSize, maxWidth, dotShape });

  try {
    // 1) Download original from R2
    const getCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: safeKey(key)
    });

    const getResp = await s3.send(getCmd);
    const inputBuffer = await streamToBuffer(getResp.Body);

    if (!inputBuffer || inputBuffer.length === 0) {
      return res.status(400).json({
        error: { code: "BAD_REQUEST", message: "Input file is empty or not found" }
      });
    }

    // 2) Decode + resize + force RGBA + get raw pixels
    const { data: rgba, info } = await sharp(inputBuffer)
      .rotate() // respect EXIF orientation
      .resize({ width: maxWidth, withoutEnlargement: true })
      .ensureAlpha() // REQUIRED for transparency-safe pipeline
      .raw()
      .toBuffer({ resolveWithObject: true });

    console.log("IMAGE INFO:", info);

    const width = info.width;
    const height = info.height;
    const channels = info.channels;

    if (channels !== 4) {
      // ensureAlpha() should guarantee 4
      throw new Error(`Expected 4 channels (RGBA) but got ${channels}`);
    }

    // 3) Create transparent output canvas
    const out = Buffer.alloc(width * height * 4, 0);

    // 4) Full-color halftone per cell
    //    - Average color per cell
    //    - Luminance determines dot size
    //    - Color uses average RGB
    //    - Output remains transparent outside dots
    const cellsX = Math.ceil(width / cellSize);
    const cellsY = Math.ceil(height / cellSize);

    for (let cy = 0; cy < cellsY; cy++) {
      for (let cx = 0; cx < cellsX; cx++) {
        const xStart = cx * cellSize;
        const yStart = cy * cellSize;
        const xEnd = Math.min(width, xStart + cellSize);
        const yEnd = Math.min(height, yStart + cellSize);

        let rSum = 0,
          gSum = 0,
          bSum = 0,
          aSum = 0,
          count = 0;

        for (let y = yStart; y < yEnd; y++) {
          for (let x = xStart; x < xEnd; x++) {
            const idx = (y * width + x) * 4;
            const a = rgba[idx + 3];
            // Include all pixels, but alpha-weight the color
            rSum += rgba[idx] * a;
            gSum += rgba[idx + 1] * a;
            bSum += rgba[idx + 2] * a;
            aSum += a;
            count++;
          }
        }

        if (count === 0) continue;

        const avgA = aSum / count; // 0..255 (not normalized)
        if (avgA <= 1) continue; // mostly transparent cell -> skip

        // Alpha-weighted average RGB
        const denom = aSum || 1;
        const avgR = rSum / denom;
        const avgG = gSum / denom;
        const avgB = bSum / denom;

        // Luminance (0..1), using sRGB coefficients
        const lum = clamp01((0.2126 * avgR + 0.7152 * avgG + 0.0722 * avgB) / 255);

        // Darker -> bigger dot
        const radiusMax = (cellSize / 2) * 0.98;
        const radius = radiusMax * (1 - lum);

        // Dot alpha: keep based on cell alpha (so transparent inputs stay transparent)
        const dotAlpha = Math.round(clamp01(avgA / 255) * 255);

        const centerX = xStart + (xEnd - xStart) / 2;
        const centerY = yStart + (yEnd - yStart) / 2;

        let rx = radius;
        let ry = radius;

        // Slightly elliptical option (more “printy” look)
        if (dotShape === "ellipse") {
          rx = radius * 1.15;
          ry = radius * 0.85;
        }

        // Skip tiny dots (speeds up)
        if (rx < 0.35 || ry < 0.35) continue;

        drawDotRGBA(
          out,
          width,
          height,
          centerX,
          centerY,
          rx,
          ry,
          avgR,
          avgG,
          avgB,
          dotAlpha
        );
      }
    }

    // 5) Encode to transparent PNG
    const outputPng = await sharp(out, { raw: { width, height, channels: 4 } })
      .png({ compressionLevel: 9 })
      .toBuffer();

    // 6) Upload output to R2
    const outId = `ht_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const outKey = `outputs/${outId}.png`;

    const putCmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey,
      Body: outputPng,
      ContentType: "image/png"
    });

    await s3.send(putCmd);

    // 7) Provide a download URL (presigned)
    const dlCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey
    });

    const downloadUrl = await getSignedUrl(s3, dlCmd, { expiresIn: 600 });

    res.json({
      ok: true,
      inputKey: key,
      outputKey: outKey,
      downloadUrl,
      format: "png",
      transparent: true,
      params: { cellSize, maxWidth, dotShape }
    });
  } catch (err) {
    console.error("HALFTONE ERROR:", err);
    res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to process halftone" }
    });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
