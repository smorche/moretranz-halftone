import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.use(
  cors({
    origin: true, // allow all origins (safe for now)
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  })
);

const requiredEnv = ["R2_ENDPOINT", "R2_BUCKET", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const s3 = new S3Client({
  region: process.env.R2_REGION, // "auto"
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

async function streamToBuffer(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function rgbaToHex(r, g, b) {
  const toHex = (x) => x.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

app.get("/health", (_req, res) => res.json({ ok: true }));

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
const ProcessRequest = z.object({
  key: z.string().min(1),
  maxWidth: z.number().int().min(64).max(5000).default(2000),
  cellSize: z.number().int().min(2).max(80).default(12)
});

async function streamToBuffer(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return Buffer.concat(chunks);
}

app.post("/v1/halftone/process", async (req, res) => {
  try {
    const parsed = ProcessRequest.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
      });
    }

    const { key, maxWidth, cellSize } = parsed.data;

    const getCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key
    });

    const obj = await s3.send(getCmd);
    if (!obj.Body) {
      return res.status(404).json({ error: { code: "NOT_FOUND", message: "Object body is empty" } });
    }

    const inputBuffer = await streamToBuffer(obj.Body);

    const outputBuffer = await sharp(inputBuffer)
      .rotate()
      .resize({ width: maxWidth, withoutEnlargement: true })
      .png()
      .toBuffer();

    const outKey = key
      .replace(/^uploads\//, "outputs/")
      .replace(/\.[a-zA-Z0-9]+$/, ".png");

    const putCmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey,
      Body: outputBuffer,
      ContentType: "image/png"
    });

    await s3.send(putCmd);

    const downloadCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey
    });

    const downloadUrl = await getSignedUrl(s3, downloadCmd, { expiresIn: 600 });

    res.json({
      ok: true,
      inputKey: key,
      outputKey: outKey,
      downloadUrl,
      settings: { maxWidth, cellSize }
    });
  } catch (err) {
    console.error("PROCESS_ERROR:", err);
    res.status(500).json({
      error: { code: "INTERNAL", message: "Processing failed" }
    });
  }
});
  }

  const { filename, contentType } = parsed.data;

  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  const key = `uploads/${imageId}/${safeName}`;

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
    Key: parsed.data.key
  });

  const downloadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });
  res.json({ downloadUrl });
});

/**
 * Full-color halftone processor (transparent PNG output)
 * Input: { key, cellSize?, maxWidth? }
 * Output: { outKey, downloadUrl, width, height, cellSize }
 */
const ProcessRequest = z.object({
  key: z.string().min(1),
  cellSize: z.number().int().min(4).max(80).default(12),
  maxWidth: z.number().int().min(200).max(4000).default(2000)
});

app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const { key, cellSize, maxWidth } = parsed.data;

  // 1) Download source image from R2
  const obj = await s3.send(
    new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key
    })
  );

  const inputBuf = await streamToBuffer(obj.Body);

  // 2) Decode to RGBA, optionally downscale for speed/cost
  let img = sharp(inputBuf, { failOn: "none" }).ensureAlpha();
  const meta = await img.metadata();
  if (meta.width && meta.width > maxWidth) img = img.resize({ width: maxWidth });

  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const width = info.width;
  const height = info.height;

  // 3) Build SVG halftone dots (transparent background)
  const radiusMax = cellSize / 2;
  let circles = "";

  for (let y = 0; y < height; y += cellSize) {
    for (let x = 0; x < width; x += cellSize) {
      const sx = clamp(x + Math.floor(cellSize / 2), 0, width - 1);
      const sy = clamp(y + Math.floor(cellSize / 2), 0, height - 1);
      const idx = (sy * width + sx) * 4;

      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const a = data[idx + 3];

      // preserve transparency: skip near-transparent pixels
      if (a < 10) continue;

      // luminance drives dot size (darker => bigger dot)
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const t = 1 - lum / 255; // 0 bright -> 1 dark
      const rad = clamp(t * radiusMax, 0.4, radiusMax);

      const fill = rgbaToHex(r, g, b);
      const opacity = a / 255;

      const cx = x + radiusMax;
      const cy = y + radiusMax;

      circles += `<circle cx="${cx}" cy="${cy}" r="${rad}" fill="${fill}" fill-opacity="${opacity.toFixed(
        3
      )}" />`;
    }
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${circles}</svg>`;

  // 4) Render SVG -> transparent PNG
  const outPng = await sharp(Buffer.from(svg)).png().toBuffer();

  // 5) Upload output to R2
  const outKey = key.replace(/^uploads\//, "outputs/").replace(/\.[^.]+$/, "") + `_halftone.png`;

  await s3.send(
    new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: outKey,
      Body: outPng,
      ContentType: "image/png"
    })
  );

  // 6) Return signed download URL
  const downloadUrl = await getSignedUrl(
    s3,
    new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: outKey }),
    { expiresIn: 600 }
  );

  res.json({ outKey, downloadUrl, width, height, cellSize });
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
