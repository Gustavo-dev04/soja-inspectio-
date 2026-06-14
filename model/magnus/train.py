"""
Magnus — Script de Treino (Modelo Pesado)
Arquitetura: RT-DETR-X | ~67-100M parâmetros
GPU: A100 obrigatório (RunPod ~R$15/sessão, Colab Pro ~R$25)

Uso:
  # RunPod ou Colab Pro+
  !pip install ultralytics pyyaml -q
  !python model/magnus/train.py
"""

import yaml
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str = "model/magnus/data.yaml",
    config_yaml: str = "model/magnus/config.yaml",
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
        warmup_epochs=cfg["warmup_epochs"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        copy_paste=cfg["copy_paste"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        save=True,
        plots=True,
    )

    best = Path(f"{cfg['project']}/{cfg['name']}/weights/best.pt")
    print(f"\n[Magnus] Treino concluído. Melhores pesos: {best}")
    return best


def evaluate(weights_path: str, data_yaml: str = "model/magnus/data.yaml"):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"\n[Magnus] mAP50   : {metrics.box.map50:.4f}")
    print(f"[Magnus] mAP50-95: {metrics.box.map:.4f}")
    print(f"[Magnus] Precision: {metrics.box.mp:.4f}")
    print(f"[Magnus] Recall  : {metrics.box.mr:.4f}")
    return metrics


def export_tensorrt(weights_path: str):
    """Exporta para TensorRT — necessário para Escopo 3 (Jetson AGX)."""
    model = YOLO(weights_path)
    model.export(format="engine", imgsz=1280, half=True, simplify=True)
    print("[Magnus] Exportado para TensorRT (.engine) — pronto para Jetson AGX.")


if __name__ == "__main__":
    weights = train()
    evaluate(str(weights))
