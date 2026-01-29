import express from "express";
import cors from "cors";
import sharp from "sharp";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import crypto from "crypto";

const app = express();
app.use(express.json({ limit: "50mb" }));

app.use(cors({
  origin: [
    "https://moretranz-halftone.pages.dev",
    "http://localhost:8788"
  ]
}));

/* ------------------ CONFIG ------------------ */

const PORT = process.env.PORT || 10000;

const R2 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

const BUCKET = process.env.R2_BUCKET;

/* ------------------ HELPERS ------------------ */

function uid() {
  return crypto.randomBytes(8).toString("hex");
}

/* ------------------ HALFTONE CORE ------------------ */

async function generateHalftone({
  inputBuffer,
  cellSize,
  maxWidth,
  strength
}) {
  // Clamp inputs
  cellSize = Math.max(4, Math.min(cellSize, 40));
  strength = Math.max(50, Math.min(strength, 200));

  // Normalize strength → print-style behavior
  const gamma = 1.0 - ((strength - 100) / 300);   // 100=1.0, 160≈0.8
  const radiusBoost = 1 + ((strength - 100) / 150);
  const minDot = (strength - 80) / 400;

  // Load & resize
  const image = sharp(inputBuffer, { unlimited: true })
    .ensureAlpha();

  const meta = await image.metadata();

  const scale = Math.min(1, maxWidth / meta.width);
  const width = Math.round(meta.width * scale);
  const height = Math.round(meta.height * scale);

  const resized = await image
    .resize(width, height, { fit: "inside" })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { data, info } = resized;

  const out = Buffer.alloc(info.width * info.height * 4, 0);

  for (let y = 0; y < info.height; y += cellSize) {
    for (let x = 0; x < info.width; x += cellSize) {

      let sum = 0;
      let count = 0;
      let alphaSum = 0;

      for (let cy = 0; cy < cellSize; cy++) {
        for (let cx = 0; cx < cellSize; cx++) {
          const px = x + cx;
          const py = y + cy;
          if (px >= info.width || py >= info.height) continue;

          const i = (py * info.width + px) * 4;
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const a = data[i + 3] / 255;

          const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
          sum += Math.pow(lum, gamma);
          alphaSum += a;
          count++;
        }
      }

      if (!count) continue;

      const avg = sum / count;
      const alpha = alphaSum / count;

      const coverage = Math.max(minDot, 1 - avg) * radiusBoost;
      const radius = Math.min(cellSize / 2, coverage * cellSize / 2);

      const cx0 = x + cellSize / 2;
      const cy0 = y + cellSize / 2;

      for (let cy = 0; cy < cellSize; cy++) {
        for (let cx = 0; cx < cellSize; cx++) {
          const px = Math.floor(cx0 - cellSize / 2 + cx);
          const py = Math.floor(cy0 - cellSize / 2 + cy);

          if (px < 0 || py < 0 || px >= info.width || py >= info.height) continue;

          const dx = cx - cellSize / 2;
          const dy = cy - cellSize / 2;
          if (dx * dx + dy * dy > radius * radius) continue;

          const o = (py * info.width + px) * 4;
          out[o] = 0;
          out[o + 1] = 0;
          out[o + 2] = 0;
          out[o + 3] = Math.round(alpha * 255);
        }
      }
    }
  }

  return sharp(out, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  }).png();
}

/* ------------------ ROUTES ------------------ */

app.post("/process", async (req, res) => {
  try {
    const { key, cellSize, maxWidth, strength } = req.body;

    const original = await R2.send(new GetObjectCommand({
      Bucket: BUCKET,
      Key: key
    }));

    const inputBuffer = Buffer.from(await original.Body.transformToByteArray());

    const png = await generateHalftone({
      inputBuffer,
      cellSize,
      maxWidth,
      strength
    });

    const outputKey = `outputs/ht_${Date.now()}_${uid()}.png`;

    await R2.send(new PutObjectCommand({
      Bucket: BUCKET,
      Key: outputKey,
      Body: await png.toBuffer(),
      ContentType: "image/png"
    }));

    res.json({ outputKey });

  } catch (err) {
    console.error("HALFTONE ERROR:", err);
    res.status(500).json({
      error: "Failed to process halftone"
    });
  }
});

/* ------------------ START ------------------ */

app.listen(PORT, () => {
  console.log(`API listening on port ${PORT}`);
});
