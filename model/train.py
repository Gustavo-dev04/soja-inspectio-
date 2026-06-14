"""
Soja Inspection — Script de Treino YOLOv8
Desenhado para Google Colab (GPU recomendada).

Uso no Colab:
  1. !pip install ultralytics roboflow huggingface_hub -q
  2. Execute as seções em ordem
"""

import os
from pathlib import Path


# ── 1. Download do Dataset (Roboflow) ──────────────────────────────────
def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    location: str = "./datasets/soja",
) -> str:
    """Faz download do dataset e retorna o caminho do data.yaml."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov8", location=location)
    )
    return dataset.location


# ── 2. Treino ───────────────────────────────────────────────────────
def train(
    data_yaml: str = "model/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_variant: str = "yolov8n.pt",   # nano → mais rápido; yolov8s/m para mais acurácia
    project_name: str = "soja-inspection",
    run_name: str = "train-v1",
) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_variant)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_name,
        name=run_name,
        device=0,       # GPU 0; use "cpu" sem GPU
        patience=20,    # early stopping
        save=True,
        plots=True,
    )
    best = Path(f"{project_name}/{run_name}/weights/best.pt")
    print(f"Melhores pesos salvos em: {best}")
    return best


# ── 3. Avaliação ────────────────────────────────────────────────────
def evaluate(weights_path: str, data_yaml: str = "model/data.yaml"):
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


# ── 4. Export ONNX ──────────────────────────────────────────────────
def export_onnx(weights_path: str):
    from ultralytics import YOLO

    model = YOLO(weights_path)
    model.export(format="onnx", imgsz=640, simplify=True)
    print("Modelo exportado para ONNX.")


# ── 5. Envio ao Hugging Face Hub ──────────────────────────────────────
def push_to_hub(weights_path: str, repo_id: str, hf_token: str):
    """Envia best.pt para um repositório HF (ex: 'seu-user/soja-yolov8')."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=weights_path,
        path_in_repo="weights/best.pt",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
    print(f"Pesos enviados para https://huggingface.co/{repo_id}")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina YOLOv8 para inspeção de grãos de soja"
    )
    parser.add_argument("--data", default="model/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Variante do modelo: yolov8n/s/m/l/x.pt",
    )
    args = parser.parse_args()

    weights = train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_variant=args.model,
    )
    evaluate(str(weights), args.data)
