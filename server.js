// server.js
import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();

// Parse JSON bodies (your API only; uploads go direct-to-R2)
app.use(express.json({ limit: "2mb" }));

// CORS for your API (NOT for R2). This allows browsers to call your API from any site.
app.use(
  cors({
    origin: true,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  })
);

// Respond to preflight requests
app.options("*", cors());

// --- Env validation ---
const requiredEnv = ["R2_ENDPOINT", "R2_BUCKET", "R2_REGION", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"];
for (const k of requiredEnv) {
  if (!process.env[k]) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

// --- Cloudflare R2 (S3 compatible) client ---
const s3 = new S3Client({
  region: process.env.R2_REGION, // usually "auto" for R2
  endpoint: process.env.R2_ENDPOINT, // e.g. https://<accountid>.r2.cloudflarestorage.com
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

// --- Utility: stream -> Buffer (GetObject returns a stream body) ---
function streamToBuffer(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("end", () => resolve(Buffer.concat(chunks)));
    stream.on("error", reject);
  });
}

// --- Basic routes ---
app.get("/", (_req, res) => {
  res.type("text").send("MoreTranz Halftone API is running. Try GET /health");
});

app.get("/health", (_req, res) => res.json({ ok: true }));

// --- Request schemas ---
const UploadUrlRequest = z.object({
  filename: z.string().min(1),
  contentType: z.enum(["image/png", "image/jpeg", "image/webp"]),
  // This is the size of the file you will PUT to the presigned URL
  contentLength: z.number().int().positive().max(25 * 1024 * 1024)
});

const DownloadUrlRequest = z.object({
  key: z.string().min(1)
});

const ProcessRequest = z.object({
  // R2 object key returned from upload-url step
  key: z.string().min(1),
  // Processing controls (starter pipeline)
  maxWidth: z.number().int().positive().max(4000).default(2000),
  outputType: z.enum(["image/png", "image/jpeg", "image/webp"]).default("image/png")
});

// --- 1) Generate an upload URL (client will PUT file directly to R2) ---
app.post("/v1/halftone/upload-url", async (req, res) => {
  const parsed = UploadUrlRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const { filename, contentType } = parsed.data;

  // Sanitize filename for object key safety
  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  const key = `uploads/${imageId}/${safeName}`;

  try {
    // NOTE: We do NOT include ContentLength in the signed request,
    // because browsers can trigger preflight or send different headers.
    // R2 will still store the content you PUT.
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
    console.error("UPLOAD-URL ERROR:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to create upload URL" }
    });
  }
});

// --- 2) Generate a download URL for an existing R2 object ---
app.post("/v1/halftone/download-url", async (req, res) => {
  const parsed = DownloadUrlRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  try {
    const cmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: parsed.data.key
    });

    const downloadUrl = await getSignedUrl(s3, cmd, { expiresIn: 600 });
    return res.json({ downloadUrl });
  } catch (err) {
    console.error("DOWNLOAD-URL ERROR:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Failed to create download URL" }
    });
  }
});

// --- 3) Process an uploaded image (download from R2 -> sharp -> write to R2) ---
app.post("/v1/halftone/process", async (req, res) => {
  const parsed = ProcessRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: { code: "BAD_REQUEST", message: "Invalid request", details: parsed.error.flatten() }
    });
  }

  const { key, maxWidth, outputType } = parsed.data;

  try {
    // (a) Download original from R2
    const getCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key
    });

    const obj = await s3.send(getCmd);

    if (!obj?.Body) {
      return res.status(404).json({
        error: { code: "NOT_FOUND", message: "Object not found or empty" }
      });
    }

    const inputBuffer = await streamToBuffer(obj.Body);

    // (b) Starter processing pipeline (NOT final halftone algorithm)
    const format = outputType === "image/jpeg" ? "jpeg" : outputType === "image/webp" ? "webp" : "png";
    const processedBuffer = await sharp(inputBuffer)
      .rotate() // honor EXIF orientation
      .resize({ width: maxWidth, withoutEnlargement: true })
      .grayscale()
      .normalise()
      .toFormat(format)
      .toBuffer();

    // (c) Save processed result back to R2
    const ext = format === "jpeg" ? "jpg" : format;
    const processedKey = key
      .replace(/^uploads\//, "processed/")
      .replace(/\.[^.]+$/, `.${ext}`);

    const putCmd = new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: processedKey,
      Body: processedBuffer,
      ContentType: outputType
    });

    await s3.send(putCmd);

    // (d) Return a signed download URL for the processed file
    const dlCmd = new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: processedKey
    });

    const downloadUrl = await getSignedUrl(s3, dlCmd, { expiresIn: 600 });

    return res.json({
      ok: true,
      inputKey: key,
      processedKey,
      downloadUrl,
      bytes: processedBuffer.length
    });
  } catch (err) {
    console.error("PROCESS ERROR:", err);
    return res.status(500).json({
      error: { code: "INTERNAL", message: "Processing failed" }
    });
  }
});

// --- Start server ---
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
