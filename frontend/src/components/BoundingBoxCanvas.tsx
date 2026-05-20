"use client";
import { useEffect, useRef } from "react";
import type { Detection } from "@/lib/api";

const CLASS_COLORS: Record<string, string> = {
  soja_boa: "#22c55e",
  soja_verde: "#84cc16",
  soja_meia_lua: "#f59e0b",
  soja_ardida: "#ef4444",
  soja_quebrada: "#8b5cf6",
};

interface Props {
  imageDataUrl: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}

export default function BoundingBoxCanvas({
  imageDataUrl,
  detections,
  imageWidth,
  imageHeight,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const displayWidth = canvas.parentElement?.clientWidth ?? imageWidth;
      const scale = displayWidth / imageWidth;
      canvas.width = displayWidth;
      canvas.height = imageHeight * scale;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const color = CLASS_COLORS[det.class] ?? "#6b7280";
        const sx1 = x1 * scale;
        const sy1 = y1 * scale;
        const sw = (x2 - x1) * scale;
        const sh = (y2 - y1) * scale;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, sw, sh);

        const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "12px sans-serif";
        const textW = ctx.measureText(label).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - 18, textW, 18);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, sx1 + 4, sy1 - 4);
      }
    };
    img.src = imageDataUrl;
  }, [imageDataUrl, detections, imageWidth, imageHeight]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-xl shadow-md"
      aria-label="Imagem com detecções de grãos de soja"
    />
  );
}
