"""
Sinnet — Script de Treino (Modelo Leve)
Arquitetura: YOLOv8s | ~11M parâmetros
GPU: T4 gratuita no Google Colab

Uso no Colab:
  !pip install ultralytics roboflow -q
  !python model/sinnet/train.py
"""

import yaml
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str = "model/sinnet/data.yaml",
    config_yaml: str = "model/sinnet/config.yaml",
) -> Path:
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"])
    model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        weight_decay=cfg["weight_decay"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        save=True,
        plots=True,
    )

    best = Path(f"{cfg['project']}/{cfg['name']}/weights/best.pt")
    print(f"\n[Sinnet] Treino concluído. Melhores pesos: {best}")
    return best


def evaluate(weights_path: str, data_yaml: str = "model/sinnet/data.yaml"):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"\n[Sinnet] mAP50   : {metrics.box.map50:.4f}")
    print(f"[Sinnet] mAP50-95: {metrics.box.map:.4f}")
    print(f"[Sinnet] Precision: {metrics.box.mp:.4f}")
    print(f"[Sinnet] Recall  : {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    weights = train()
    evaluate(str(weights))
