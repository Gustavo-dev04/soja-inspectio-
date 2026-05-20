import base64
import io
import os
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        path = os.getenv("MODEL_PATH", "yolov8n.pt")
        _model = YOLO(path)
    return _model


def decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def run_inference(image: Image.Image, conf_threshold: float = 0.25) -> dict[str, Any]:
    model = get_model()
    img_array = np.array(image)

    results = model.predict(source=img_array, conf=conf_threshold, verbose=False)

    detections: list[dict] = []
    class_counts: dict[str, int] = {}

    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    w, h = image.size
    return {
        "total_graos": len(detections),
        "detections": detections,
        "class_counts": class_counts,
        "image_width": w,
        "image_height": h,
    }
