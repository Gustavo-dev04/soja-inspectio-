import io
import os
import json
import glob
import hashlib
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


def _crop_single_grain(arr: np.ndarray):
    """Detect the largest object and return (crop, found).

    found=False means no usable grain was detected (no contour or too small) —
    callers that feed the training dataset must NOT save in that case.
    """
    h, w    = arr.shape[:2]
    gray    = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr, False
    largest = max(contours, key=cv2.contourArea)
    ratio   = cv2.contourArea(largest) / (h * w)
    if ratio <= 0.02:
        return arr, False           # só ruído — nenhum grão de verdade
    if ratio >= 0.95:
        return arr, True            # imagem já é o grão enquadrado
    x, y, bw, bh = cv2.boundingRect(largest)
    pad = max(15, int(min(bw, bh) * 0.12))
    return arr[max(0, y-pad):min(h, y+bh+pad), max(0, x-pad):min(w, x+bw+pad)], True


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
        # Salva o grão JÁ RECORTADO — mesmo preprocessamento da classificação.
        # Garante que o fine-tuning treine no mesmo enquadramento que o modelo vê em produção.
        cropped, found = _crop_single_grain(np.array(image.convert("RGB")))
        if not found:
            return ("❌ Nenhum grão detectado na foto — correção NÃO salva. "
                    "Use fundo escuro e deixe o grão bem visível.")
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            return ("❌ Grão muito pequeno na foto (recorte < 50×50 px) — "
                    "aproxime a câmera e tente de novo.")

        buf = io.BytesIO()
        Image.fromarray(cropped).save(buf, format="JPEG", quality=92)
        buf.seek(0)

        # Nome = hash MD5 do conteúdo → mesmo grão enviado 2x cai no mesmo path
        md5  = hashlib.md5(buf.getvalue()).hexdigest()
        path = f"{correct_class}/{md5}.jpg"

        api = HfApi(token=HF_TOKEN)
        if api.file_exists(CORRECTIONS_REPO, path, repo_type="dataset"):
            return "⚠️ Esta imagem já está no dataset (duplicada) — não salvei de novo."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        api.upload_file(
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
    crop, _found  = _crop_single_grain(arr)
    idx, conf, pv = _predict(crop)
    pt_label      = PT_LABELS[CLASS_NAMES[idx]]
    probs         = {PT_LABELS[CLASS_NAMES[i]]: float(pv[i]) for i in range(len(CLASS_NAMES))}
    return f"{pt_label} — {conf:.1%}", probs


# ── Status do sistema ─────────────────────────────────────────────────
def get_status() -> str:
    """Versão do modelo, correções acumuladas e última avaliação registrada."""
    lines = []

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        size  = os.path.getsize(MODEL_PATH) / 1e6
        lines.append(f"Modelo: {MODEL_PATH} ({size:.0f} MB)")
        lines.append(f"Atualizado em: {mtime:%d/%m/%Y %H:%M} UTC")
    except OSError:
        lines.append("Modelo: arquivo não encontrado!")

    if HF_TOKEN:
        try:
            files = HfApi(token=HF_TOKEN).list_repo_files(
                CORRECTIONS_REPO, repo_type="dataset")
            imgs   = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            treino = [f for f in imgs if not f.startswith("holdout/")]
            hold   = [f for f in imgs if f.startswith("holdout/")]
            lines.append(f"Correções acumuladas: {len(treino)} treino + {len(hold)} holdout")
        except Exception as e:
            lines.append(f"Correções: erro ao consultar dataset ({e})")
    else:
        lines.append("Correções: secret SOJA_CORRECTIONS não configurado")

    evals = sorted(glob.glob("results/eval_*.json"))
    if evals:
        try:
            with open(evals[-1]) as f:
                ev = json.load(f)
            lines.append(f"Última avaliação: {ev.get('timestamp', '?')}")
            lines.append(f"  Holdout (fotos reais): {ev.get('baseline_holdout', 0):.0%}"
                         f" → {ev.get('new_holdout', 0):.0%}")
            lines.append(f"  Teste original (lab):  {ev.get('new_original', 0):.0%}")
            per_cls = ev.get("per_class_holdout", {})
            if per_cls:
                detalhes = ", ".join(
                    f"{k}: {v:.0%}" for k, v in per_cls.items() if v is not None)
                lines.append(f"  Por classe: {detalhes}")
        except Exception as e:
            lines.append(f"Última avaliação: erro ao ler ({e})")
    else:
        lines.append("Última avaliação: nenhuma registrada ainda "
                     "(roda o finetune.ipynb e a Célula 11 sobe o eval pra cá)")

    return "\n".join(lines)


def _split_touching(orig: np.ndarray, cnt) -> list:
    """Separa grãos encostados (blob único) via watershed; devolve contornos individuais.

    Roda só no ROI do contorno (barato em CPU). Se o distance transform achar
    uma única semente, o blob era um grão só — devolve o contorno original.
    """
    x, y, bw, bh = cv2.boundingRect(cnt)
    roi_mask = np.zeros((bh, bw), np.uint8)
    cv2.drawContours(roi_mask, [cnt - np.array([[x, y]])], -1, 255, -1)

    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    n_seeds, markers = cv2.connectedComponents(sure_fg)
    if n_seeds <= 2:                      # fundo + 1 semente → não era blob duplo
        return [cnt]

    markers = markers + 1                 # fundo=1, sementes=2..n
    unknown = cv2.subtract(roi_mask, sure_fg)
    markers[unknown == 255] = 0           # região incerta que o watershed resolve

    roi_bgr = cv2.cvtColor(orig[y:y+bh, x:x+bw], cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(roi_bgr, markers)

    pieces = []
    for m in range(2, n_seeds + 1):
        seg = np.where((markers == m) & (roi_mask > 0), 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            pieces.append(c + np.array([[x, y]]))
    return pieces if pieces else [cnt]


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

    # Watershed condicional: blobs com área > 1.8× a mediana dos DEMAIS contornos
    # são suspeitos de conter 2+ grãos encostados e passam pela separação; o resto
    # segue direto (mais barato em CPU). Mediana leave-one-out: com poucos grãos no
    # frame, a mediana global seria puxada pelo próprio blob e o gatilho falharia.
    # Contorno sozinho no frame é sempre suspeito (o contador de sementes interno
    # do watershed decide se separa). Filtro de área reaplicado APÓS a separação.
    pre = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    final_contours = []
    if pre:
        areas = np.array([cv2.contourArea(c) for c in pre])
        for i, c in enumerate(pre):
            others  = np.delete(areas, i)
            suspect = len(others) == 0 or areas[i] > 1.8 * float(np.median(others))
            if suspect:
                final_contours.extend(_split_touching(orig, c))
            else:
                final_contours.append(c)

    for cnt in final_contours:
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
                "A imagem é salva automaticamente no dataset de treino para melhorar o modelo.\n\n"
                "💡 **Coleta rápida:** para juntar muitas fotos de uma classe, selecione a classe "
                "uma vez e fique tirando foto + **Enviar correção** em sequência — a foto limpa "
                "sozinha após cada envio (não precisa clicar em Classificar)."
            )
            with gr.Row():
                correction_dd  = gr.Dropdown(
                    choices=PT_OPTIONS,
                    label="Classe correta",
                    interactive=True,
                )
                correction_btn = gr.Button("Enviar correção", variant="secondary")
            correction_status = gr.Textbox(label="Status da correção", interactive=False)

            # Após enviar, limpa só a FOTO (mantém a classe selecionada).
            # Permite coletar muitos grãos da mesma classe em sequência rápida:
            # seleciona a classe uma vez → snap → enviar → snap → enviar...
            correction_btn.click(
                save_correction,
                inputs=[inp1, correction_dd],
                outputs=[correction_status],
            ).then(
                lambda: None,
                outputs=[inp1],
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

        # ── Aba 4: Status ─────────────────────────────────────────────
        with gr.Tab("Status"):
            gr.Markdown("## Status do sistema\nVersão do modelo, correções acumuladas e última avaliação.")
            status_btn = gr.Button("Atualizar status", variant="secondary")
            status_out = gr.Textbox(label="Status", lines=10, interactive=False)
            status_btn.click(get_status, inputs=[], outputs=[status_out])

demo.launch()
