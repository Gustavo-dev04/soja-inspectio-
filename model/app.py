import os
import json
import numpy as np
import cv2
import gradio as gr
import tensorflow as tf
from PIL import Image

MODEL_PATH   = "soja_model_final.keras"
CLASSES_PATH = "soja_classes.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

# Portuguese display labels
PT_LABELS = {
    "Broken soybeans":       "Quebrado",
    "Immature soybeans":     "Imaturo",
    "Intact soybeans":       "Intacto",
    "Skin-damaged soybeans": "Casca danificada",
    "Spotted soybeans":      "Manchado",
}

# RGB color per class index: Broken, Immature, Intact, Skin-damaged, Spotted
CLASS_COLORS = [
    (231, 76,  60),   # red      — broken
    (243, 156, 18),   # orange   — immature
    (46,  204, 113),  # green    — intact
    (155, 89,  182),  # purple   — skin-damaged
    (52,  152, 219),  # blue     — spotted
]

IMG_SIZE = (224, 224)


def _preprocess(img_array: np.ndarray) -> np.ndarray:
    img = tf.image.resize(img_array, IMG_SIZE)
    return np.expand_dims(img.numpy(), 0)


def _predict(crop: np.ndarray):
    pred = model.predict(_preprocess(crop), verbose=0)[0]
    idx  = int(np.argmax(pred))
    return idx, float(pred[idx]), pred


# ── Mode 1: single grain ─────────────────────────────────────────────
def classify_single(image: Image.Image):
    if image is None:
        return "Nenhuma imagem recebida.", {}

    arr           = np.array(image.convert("RGB"))
    idx, conf, pv = _predict(arr)
    pt_label      = PT_LABELS[CLASS_NAMES[idx]]
    label_str     = f"{pt_label} — {conf:.1%}"

    probs = {PT_LABELS[CLASS_NAMES[i]]: float(pv[i]) for i in range(len(CLASS_NAMES))}
    return label_str, probs


# ── Mode 2: multi-grain with OpenCV segmentation ─────────────────────
def segment_and_classify(image: Image.Image, conf_threshold: float = 0.6):
    if image is None:
        return None, "Nenhuma imagem recebida."

    orig    = np.array(image.convert("RGB"))
    h, w    = orig.shape[:2]
    annotated = orig.copy()

    # --- Segmentation on dark background ---
    gray    = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area relative to image size
    min_area = (h * w) * 0.0015   # 0.15% — ignora ruído
    max_area = (h * w) * 0.25     # 25%   — ignora objetos enormes

    class_count  = {PT_LABELS[n]: 0 for n in CLASS_NAMES}
    total_grains = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = max(8, int(min(bw, bh) * 0.1))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(w, x + bw + pad)
        y2  = min(h, y + bh + pad)

        crop          = orig[y1:y2, x1:x2]
        idx, conf, _  = _predict(crop)
        name          = CLASS_NAMES[idx]
        pt_label      = PT_LABELS[name]
        color         = CLASS_COLORS[idx]
        total_grains += 1

        if conf < conf_threshold:
            label_text = f"? {conf:.0%}"
            draw_color = (180, 180, 180)
        else:
            label_text = f"{pt_label} {conf:.0%}"
            draw_color = color
            class_count[pt_label] += 1

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, 2)

        # Label background + text
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.65, bw / 250))
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        lx2 = min(w, x1 + tw + 6)
        ly1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, ly1), (lx2, y1), draw_color, -1)
        cv2.putText(
            annotated, label_text,
            (x1 + 3, y1 - 4),
            font, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Build summary
    if total_grains == 0:
        msg = (
            "Nenhum grão detectado.\n\n"
            "Dicas para melhor resultado:\n"
            "• Fundo preto fosco (cartolina ou pano)\n"
            "• Grãos separados, sem se encostar\n"
            "• Luz vinda de cima, sem sombras fortes\n"
            "• Câmera na vertical, ~30cm de distância"
        )
    else:
        lines = [f"Total detectado: {total_grains} grão(s)\n"]
        for pt, n in sorted(class_count.items(), key=lambda kv: -kv[1]):
            if n > 0:
                pct = n / total_grains * 100
                lines.append(f"  {pt}: {n}  ({pct:.0f}%)")
        uncertain = total_grains - sum(class_count.values())
        if uncertain > 0:
            lines.append(f"  Incertos (?): {uncertain}")
        msg = "\n".join(lines)

    return Image.fromarray(annotated), msg


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Classificador de Grãos de Soja", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Classificador de Grãos de Soja
        **Demo FATEC** &nbsp;·&nbsp; EfficientNet-B0 + OpenCV
        &nbsp;·&nbsp; 5 classes: Intacto · Imaturo · Quebrado · Casca danificada · Manchado
        """
    )

    with gr.Tabs():

        with gr.Tab("Um grão"):
            gr.Markdown("Foto de **um único grão** — classificação direta.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp1 = gr.Image(
                        type="pil",
                        label="Foto do grão",
                        sources=["upload", "webcam"],
                    )
                    btn1 = gr.Button("Classificar", variant="primary")
                with gr.Column(scale=1):
                    out1_label = gr.Textbox(label="Resultado", interactive=False)
                    out1_probs = gr.Label(
                        label="Probabilidades por classe",
                        num_top_classes=5,
                    )
            btn1.click(
                classify_single,
                inputs=[inp1],
                outputs=[out1_label, out1_probs],
            )

        with gr.Tab("Vários grãos"):
            gr.Markdown(
                "Foto com **vários grãos** sobre **fundo escuro**.  \n"
                "O sistema segmenta cada grão com OpenCV e classifica um por um."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inp2 = gr.Image(
                        type="pil",
                        label="Foto dos grãos",
                        sources=["upload", "webcam"],
                    )
                    conf_slider = gr.Slider(
                        0.3, 0.9, value=0.6, step=0.05,
                        label="Limiar de confiança (grão marcado com ? se abaixo)",
                    )
                    btn2 = gr.Button("Analisar", variant="primary")
                with gr.Column(scale=1):
                    out2_img  = gr.Image(label="Imagem anotada")
                    out2_text = gr.Textbox(label="Resumo", lines=10, interactive=False)
            btn2.click(
                segment_and_classify,
                inputs=[inp2, conf_slider],
                outputs=[out2_img, out2_text],
            )

        with gr.Tab("Instruções"):
            gr.Markdown(
                """
                ## Setup para foto ideal

                1. **Fundo:** cartolina ou pano **preto fosco** (não refletivo)
                2. **Grãos:** espalhados **sem se encostar** (separados ≥1 cm)
                3. **Luz:** vinda de **cima**, difusa — sem sombras duras
                4. **Câmera:** vertical, ~30 cm de distância, sem zoom

                ---

                ## Classes

                | Classe | PT | Descrição |
                |--------|----|-----------|
                | intact | Intacto | Grão saudável, sem defeitos |
                | immature | Imaturo | Verde ou desenvolvimento incompleto |
                | broken | Quebrado | Partido ou com fragmento faltando |
                | skin-damaged | Casca danificada | Casca rasgada, descascada ou abrasada |
                | spotted | Manchado | Manchas escuras ou colorações anormais |

                ---

                ## Limiar de confiança

                Grãos com confiança abaixo do limiar aparecem como `?` cinza na imagem.
                Reduza o limiar se poucos grãos estiverem sendo classificados.

                ---

                ## Referências

                - **Modelo:** EfficientNet-B0 (ImageNet) + fine-tuning — TensorFlow/Keras
                - **Dataset:** SoyaBeans Classifications v2 — Roboflow, MIT License
                - **Segmentação:** OpenCV (grayscale → Otsu threshold → findContours)
                - **Critério:** norma chinesa GB1352-2009
                """
            )

demo.launch()
