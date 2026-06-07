import io
import os
import json
import uuid
import numpy as np
import cv2
import gradio as gr
import tensorflow as tf
from PIL import Image
from datetime import datetime
from huggingface_hub import HfApi

MODEL_PATH   = "soja_model_final.keras"
CLASSES_PATH = "soja_classes.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

PT_LABELS = {
    "Broken soybeans":       "Quebrado",
    "Immature soybeans":     "Imaturo",
    "Intact soybeans":       "Intacto",
    "Skin-damaged soybeans": "Casca danificada",
    "Spotted soybeans":      "Manchado",
}

PT_TO_CLASS = {v: k for k, v in PT_LABELS.items()}
PT_OPTIONS  = list(PT_LABELS.values())

CLASS_COLORS = [
    (231, 76,  60),
    (243, 156, 18),
    (46,  204, 113),
    (155, 89,  182),
    (52,  152, 219),
]

IMG_SIZE = (224, 224)

CORRECTIONS_REPO = os.getenv("CORRECTIONS_DATASET_REPO", "Guguinhaxd/soja-correction")
HF_TOKEN         = os.getenv("SOJA_CORRECTIONS", None)


# ── Core helpers ─────────────────────────────────────────────────────
def _preprocess(img_array: np.ndarray) -> np.ndarray:
    img = tf.image.resize(img_array, IMG_SIZE)
    return np.expand_dims(img.numpy(), 0)


def _predict(crop: np.ndarray):
    pred = model.predict(_preprocess(crop), verbose=0)[0]
    idx  = int(np.argmax(pred))
    return idx, float(pred[idx]), pred


def _crop_single_grain(arr: np.ndarray) -> np.ndarray:
    """Detect the largest object and return a tight crop (falls back to full image)."""
    h, w    = arr.shape[:2]
    gray    = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr
    largest = max(contours, key=cv2.contourArea)
    if not (0.02 < cv2.contourArea(largest) / (h * w) < 0.95):
        return arr
    x, y, bw, bh = cv2.boundingRect(largest)
    pad = max(15, int(min(bw, bh) * 0.12))
    return arr[max(0, y-pad):min(h, y+bh+pad), max(0, x-pad):min(w, x+bw+pad)]


# ── Feedback / active learning ────────────────────────────────────────
def save_correction(image: Image.Image, correct_pt_label: str):
    """Save a misclassified image with the correct label to the HF corrections dataset."""
    if image is None:
        return "Nenhuma imagem para corrigir."
    if not correct_pt_label:
        return "Selecione a classe correta antes de enviar."
    if not HF_TOKEN:
        return "⚠️ Secret SOJA_CORRECTIONS não configurado no Space. Contate o administrador."

    try:
        correct_class = PT_TO_CLASS[correct_pt_label]
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        buf.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid       = uuid.uuid4().hex[:8]
        path      = f"{correct_class}/{timestamp}_{uid}.jpg"

        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=buf,
            path_in_repo=path,
            repo_id=CORRECTIONS_REPO,
            repo_type="dataset",
            commit_message=f"correction: {correct_class} {timestamp}",
        )
        return f"✅ Salvo como '{correct_pt_label}'. Obrigado — vai para o dataset de treino!"
    except Exception as e:
        return f"Erro ao salvar: {e}"


# ── Mode 1: single grain ─────────────────────────────────────────────
def classify_single(image: Image.Image):
    if image is None:
        return "Nenhuma imagem recebida.", {}
    arr           = np.array(image.convert("RGB"))
    crop          = _crop_single_grain(arr)
    idx, conf, pv = _predict(crop)
    pt_label      = PT_LABELS[CLASS_NAMES[idx]]
    probs         = {PT_LABELS[CLASS_NAMES[i]]: float(pv[i]) for i in range(len(CLASS_NAMES))}
    return f"{pt_label} — {conf:.1%}", probs


# ── Mode 2: multi-grain ───────────────────────────────────────────────
def segment_and_classify(image: Image.Image, conf_threshold: float = 0.6):
    if image is None:
        return None, "Nenhuma imagem recebida."

    orig      = np.array(image.convert("RGB"))
    h, w      = orig.shape[:2]
    annotated = orig.copy()

    gray    = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area    = (h * w) * 0.0015
    max_area    = (h * w) * 0.25
    class_count = {PT_LABELS[n]: 0 for n in CLASS_NAMES}
    total       = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad  = max(8, int(min(bw, bh) * 0.1))
        x1   = max(0, x - pad);  y1 = max(0, y - pad)
        x2   = min(w, x+bw+pad); y2 = min(h, y+bh+pad)
        crop = orig[y1:y2, x1:x2]
        idx, conf, _ = _predict(crop)
        name  = CLASS_NAMES[idx]
        pt    = PT_LABELS[name]
        color = CLASS_COLORS[idx]
        total += 1

        if conf < conf_threshold:
            label_text = f"? {conf:.0%}"; draw_color = (180, 180, 180)
        else:
            label_text = f"{pt} {conf:.0%}"; draw_color = color
            class_count[pt] += 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, 2)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscale = max(0.4, min(0.65, bw / 250))
        (tw, th), _ = cv2.getTextSize(label_text, font, fscale, 1)
        lx2 = min(w, x1+tw+6); ly1 = max(0, y1-th-8)
        cv2.rectangle(annotated, (x1, ly1), (lx2, y1), draw_color, -1)
        cv2.putText(annotated, label_text, (x1+3, y1-4), font, fscale, (255,255,255), 1, cv2.LINE_AA)

    if total == 0:
        msg = (
            "Nenhum grão detectado.\n\n"
            "Dicas:\n"
            "• Fundo preto fosco (cartolina ou pano)\n"
            "• Grãos separados, sem encostar\n"
            "• Luz vinda de cima, sem sombras fortes\n"
            "• Câmera vertical, ~30cm de distância"
        )
    else:
        lines = [f"Total: {total} grão(s)\n"]
        for pt, n in sorted(class_count.items(), key=lambda kv: -kv[1]):
            if n > 0:
                lines.append(f"  {pt}: {n}  ({n/total*100:.0f}%)")
        uncertain = total - sum(class_count.values())
        if uncertain:
            lines.append(f"  Incertos (?): {uncertain}")
        msg = "\n".join(lines)

    return Image.fromarray(annotated), msg


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Classificador de Grãos de Soja") as demo:
    gr.Markdown(
        """
        # Classificador de Grãos de Soja
        **Demo FATEC** &nbsp;·&nbsp; EfficientNet-B0 + OpenCV
        &nbsp;·&nbsp; 5 classes: Intacto · Imaturo · Quebrado · Casca danificada · Manchado
        """
    )

    with gr.Tabs():

        # ── Aba 1: Um grão ────────────────────────────────────────────
        with gr.Tab("Um grão"):
            gr.Markdown("Foto de **um único grão** — o sistema detecta e recorta o grão automaticamente.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp1 = gr.Image(type="pil", label="Foto do grão", sources=["upload", "webcam"])
                    btn1 = gr.Button("Classificar", variant="primary")
                with gr.Column(scale=1):
                    out1_label = gr.Textbox(label="Resultado", interactive=False)
                    out1_probs = gr.Label(label="Probabilidades por classe", num_top_classes=5)

            btn1.click(classify_single, inputs=[inp1], outputs=[out1_label, out1_probs])

            # ── Bloco de correção ──────────────────────────────────────
            gr.Markdown("---")
            gr.Markdown("### Resultado errado?")
            gr.Markdown(
                "Se o modelo errou, selecione a classe correta abaixo e clique em **Enviar correção**. "
                "A imagem é salva automaticamente no dataset de treino para melhorar o modelo."
            )
            with gr.Row():
                correction_dd  = gr.Dropdown(
                    choices=PT_OPTIONS,
                    label="Classe correta",
                    interactive=True,
                )
                correction_btn = gr.Button("Enviar correção", variant="secondary")
            correction_status = gr.Textbox(label="Status da correção", interactive=False)

            correction_btn.click(
                save_correction,
                inputs=[inp1, correction_dd],
                outputs=[correction_status],
            )

        # ── Aba 2: Vários grãos ───────────────────────────────────────
        with gr.Tab("Vários grãos"):
            gr.Markdown(
                "Foto com **vários grãos** sobre **fundo escuro**.  \n"
                "O sistema segmenta e classifica cada grão individualmente."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inp2        = gr.Image(type="pil", label="Foto dos grãos", sources=["upload", "webcam"])
                    conf_slider = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Limiar de confiança")
                    btn2        = gr.Button("Analisar", variant="primary")
                with gr.Column(scale=1):
                    out2_img  = gr.Image(label="Imagem anotada")
                    out2_text = gr.Textbox(label="Resumo", lines=10, interactive=False)

            btn2.click(segment_and_classify, inputs=[inp2, conf_slider], outputs=[out2_img, out2_text])

        # ── Aba 3: Instruções ─────────────────────────────────────────
        with gr.Tab("Instruções"):
            gr.Markdown(
                """
                ## Setup para foto ideal

                1. **Fundo:** cartolina ou pano **preto fosco** (não refletivo)
                2. **Grãos:** separados **sem encostar** (≥1 cm entre eles)
                3. **Luz:** vinda de **cima**, difusa — sem flash, sem sombra dura
                4. **Câmera:** vertical, ~30 cm de distância, sem zoom

                ---

                ## Classes

                | Classe | Descrição |
                |--------|-----------|
                | Intacto | Grão saudável, sem defeitos |
                | Imaturo | Verde ou desenvolvimento incompleto |
                | Quebrado | Partido ou com fragmento faltando |
                | Casca danificada | Casca rasgada, descascada ou abrasada |
                | Manchado | Manchas escuras ou colorações anormais |

                ---

                ## Como o feedback melhora o modelo

                Cada vez que você corrige um resultado errado, a imagem é salva no dataset de treino.
                Após acumular correções, um novo treino (fine-tuning) com essas imagens adapta o modelo
                à câmera e iluminação reais — reduzindo erros progressivamente.

                ---

                ## Referências

                - **Modelo:** EfficientNet-B0 (ImageNet) + fine-tuning — TensorFlow/Keras
                - **Dataset:** SoyaBeans Classifications v2 — Roboflow, MIT License
                - **Segmentação:** OpenCV (grayscale → Otsu threshold → findContours)
                """
            )

demo.launch()
