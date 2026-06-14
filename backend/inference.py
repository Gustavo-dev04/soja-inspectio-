import base64
import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

CLASS_NAMES = ["broken", "immature", "intact", "skin-damaged", "spotted"]

_model: YOLO | None = None
_yolo_to_ours: list[int] | None = None


def get_model() -> YOLO:
    global _model, _yolo_to_ours
    if _model is None:
        path = os.getenv("MODEL_PATH", "soja_yolo11s_finetuned.pt")
        _model = YOLO(path)
        _yolo_to_ours = [CLASS_NAMES.index(_model.names[i]) for i in range(len(_model.names))]
    return _model


def decode_base64_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _segment_grains(img_array: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.001
    max_area = h * w * 0.5
    pad = 4
    boxes: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area or cv2.contourArea(cnt) > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        boxes.append((x1, y1, x2, y2))

    return boxes


def run_inference(image: Image.Image) -> dict[str, Any]:
    model = get_model()
    img_array = np.array(image)
    boxes = _segment_grains(img_array)

    detections: list[dict] = []
    class_counts: dict[str, int] = {}

    for (x1, y1, x2, y2) in boxes:
        crop = img_array[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        result = model.predict(Image.fromarray(crop), imgsz=224, verbose=False)[0]
        yolo_probs = result.probs.data.cpu().numpy()

        pv = np.zeros(len(CLASS_NAMES), dtype=float)
        for yolo_idx, our_idx in enumerate(_yolo_to_ours):
            pv[our_idx] = float(yolo_probs[yolo_idx])

        our_idx = int(np.argmax(pv))
        conf = float(pv[our_idx])
        cls_name = CLASS_NAMES[our_idx]

        detections.append({
            "class": cls_name,
            "class_id": our_idx,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
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
