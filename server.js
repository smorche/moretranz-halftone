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

/* ---------------------------
   App + middleware
---------------------------- */
const app = express();
app.use(express.json({ limit: "1mb" }));

app.use(
  cors({
    origin: [
      "https://moretranz-halftone.pages.dev",
      "https://moretranz.com",
      "https://www.moretranz.com"
    ],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  })
);

/* ---------------------------
   Limits / constants
---------------------------- */
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25MB
const MAX_IMAGE_PIXELS = 60_000_000;       // safety ceiling
const MAX_PRINT_WIDTH_PX = 3600;           // 12" @ 300 DPI
const ALPHA_THRESHOLD = 10;                // 0–255

/* ---------------------------
   R2 client
---------------------------- */
const s3 = new S3Client({
  region: process.env.R2_REGION,
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

/* ---------------------------
   Helpers
---------------------------- */
const reqId = () => crypto.randomBytes(8).toString("hex");

const streamToBuffer = async (stream) => {
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  return Buffer.concat(chunks);
};

const luminance = (r, g, b) =>
  0.2126 * r + 0.7152 * g + 0.0722 * b;

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

/* ---------------------------
   Core halftone logic (FIXED)
---------------------------- */
async function makeColorHalftonePng(
  inputBuffer,
  { cellSize, maxWidth, dotShape }
) {
  const base = sharp(inputBuffer, { failOn: "none" }).ensureAlpha();

  const meta = await base.metadata();
  if (!meta.width || !meta.height) {
    throw new Error("Invalid image");
  }

  const scaleWidth = Math.min(meta.width, maxWidth, MAX_PRINT_WIDTH_PX);

  const resized = sharp(inputBuffer)
    .ensureAlpha()
    .resize({ width: scaleWidth, withoutEnlargement: true });

  const { data, info } = await resized
    .raw()
    .toBuffer({ resolveWithObject: true });

  const w = info.width;
  const h = info.height;

  const cs = clamp(cellSize, 6, 40);
  const half = cs / 2;

  let shapes = "";

  for (let y = 0; y < h; y += cs) {
    for (let x = 0; x < w; x += cs) {
      let r = 0, g = 0, b = 0, a = 0, count = 0;

      for (let yy = y; yy < Math.min(y + cs, h); yy++) {
        for (let xx = x; xx < Math.min(x + cs, w); xx++) {
          const i = (yy * w + xx) * 4;
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          a += data[i + 3];
          count++;
        }
      }

      if (!count) continue;

      r /= count; g /= count; b /= count; a /= count;

      if (a < ALPHA_THRESHOLD) continue;

      const darkness = 1 - luminance(r, g, b) / 255;
      const radius = clamp(half * darkness, 0.5, half);

      const cx = x + cs / 2;
      const cy = y + cs / 2;
      const fill = `rgb(${r|0},${g|0},${b|0})`;
      const opacity = (a / 255).toFixed(3);

      if (dotShape === "square") {
        shapes += `<rect x="${(cx-radius).toFixed(2)}" y="${(cy-radius).toFixed(2)}"
          width="${(radius*2).toFixed(2)}" height="${(radius*2).toFixed(2)}"
          fill="${fill}" fill-opacity="${opacity}" />`;
      } else if (dotShape === "ellipse") {
        shapes += `<ellipse cx="${cx}" cy="${cy}"
          rx="${radius*1.2}" ry="${radius*0.85}"
          fill="${fill}" fill-opacity="${opacity}" />`;
      } else {
        shapes += `<circle cx="${cx}" cy="${cy}"
          r="${radius}" fill="${fill}" fill-opacity="${opacity}" />`;
      }
    }
  }

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg"
     width="${w}" height="${h}"
     viewBox="0 0 ${w} ${h}">
${shapes}
</svg>`;

  return sharp(Buffer.from(svg))
    .png({ compressionLevel: 9 })
    .toBuffer();
}

/* ---------------------------
   Routes
---------------------------- */
app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/v1/halftone/upload-url", async (req, res) => {
  const { filename, contentType, contentLength } = req.body;

  if (!filename || !contentType || contentLength > MAX_UPLOAD_BYTES) {
    return res.status(400).json({ error: "Invalid upload" });
  }

  const key = `uploads/img_${Date.now()}/${filename}`;

  const uploadUrl = await getSignedUrl(
    s3,
    new PutObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: key,
      ContentType: contentType
    }),
    { expiresIn: 600 }
  );

  res.json({ key, uploadUrl, headers: { "Content-Type": contentType } });
});

app.post("/v1/halftone/process", async (req, res) => {
  const { key, cellSize = 12, maxWidth = 2000, dotShape = "circle" } = req.body;

  try {
    const obj = await s3.send(
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: key
      })
    );

    const input = await streamToBuffer(obj.Body);

    const png = await makeColorHalftonePng(input, {
      cellSize,
      maxWidth,
      dotShape
    });

    const outputKey = `outputs/ht_${Date.now()}.png`;

    await s3.send(
      new PutObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey,
        Body: png,
        ContentType: "image/png"
      })
    );

    const downloadUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({
        Bucket: process.env.R2_BUCKET,
        Key: outputKey
      }),
      { expiresIn: 600 }
    );

    res.json({ downloadUrl, outputKey });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process halftone" });
  }
});

/* ---------------------------
   Start server
---------------------------- */
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`API listening on ${port}`));
