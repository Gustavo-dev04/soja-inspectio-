"""
Gradio app para Hugging Face Space — Inspeção de Soja.
Deploy como Space com SDK: gradio.
Coloque best.pt em weights/best.pt ou defina HF_MODEL_PATH.
"""

import os

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

MODEL_PATH = os.getenv("HF_MODEL_PATH", "weights/best.pt")
model = YOLO(MODEL_PATH)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "soja_boa":      (34,  197,  94),
    "soja_verde":    (132, 204,  22),
    "soja_meia_lua": (245, 158,  11),
    "soja_ardida":   (239,  68,  68),
    "soja_quebrada": (139,  92, 246),
}


def predict(image: np.ndarray, conf_threshold: float = 0.25):
    results = model.predict(source=image, conf=conf_threshold, verbose=False)

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    summary: dict[str, int] = {}

    for result in results:
        names = result.names
        for box in result.boxes:
            cls_name = names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            color = CLASS_COLORS.get(cls_name, (107, 114, 128))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 14)), f"{cls_name} {conf:.0%}", fill=color)
            summary[cls_name] = summary.get(cls_name, 0) + 1

    total = sum(summary.values())
    resumo = (
        "\n".join(
            f"{k}: {v} ({v / total * 100:.1f}%)"
            for k, v in sorted(summary.items(), key=lambda x: -x[1])
        )
        or "Nenhum grão detectado."
    )

    return pil_img, resumo


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Imagem de Grãos de Soja", type="numpy"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confiança Mínima"),
    ],
    outputs=[
        gr.Image(label="Detecções"),
        gr.Textbox(label="Resumo por Classe"),
    ],
    title="Inspeção de Grãos de Soja — YOLOv8",
    description=(
        "Envie uma imagem de grãos de soja para identificar defeitos automaticamente.\n"
        "Classes: soja_boa · soja_verde · soja_meia_lua · soja_ardida · soja_quebrada"
    ),
)

if __name__ == "__main__":
    demo.launch()
