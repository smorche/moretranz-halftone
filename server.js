import "dotenv/config";
import express from "express";
import cors from "cors";
import { z } from "zod";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.use(
  cors({
    origin: [
      "https://moretranz.com",
      "https://www.moretranz.com",
      "http://localhost:3000"
    ],
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
  }

  const { filename, contentType, contentLength } = parsed.data;

  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
  const imageId = `img_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  const key = `uploads/${imageId}/${safeName}`;

  const cmd = new PutObjectCommand({
    Bucket: process.env.R2_BUCKET,
    Key: key,
    ContentType: contentType,
    ContentLength: contentLength
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

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on port ${port}`));
