"""Optional local YOLO26s object detector used beside the defect model.

The COCO-pretrained detector recognizes common objects and checks that an item is
present in frame. It does not influence the anomaly decision: product-specific
surface defects require the PatchCore reference model or custom YOLO training.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT / ".cache" / "yolo"
WEIGHTS = CACHE_DIR / "yolo26s.pt"
MODEL_NAME = "YOLO26s"
_model = None
_lock = threading.Lock()


def status() -> dict:
    return {"model": MODEL_NAME, "task": "generic object detection", "source": "Ultralytics pretrained COCO", "ready": WEIGHTS.is_file()}


def prepare():
    """Load the cached detector, downloading official pretrained weights once."""
    global _model
    with _lock:
        if _model is not None:
            return _model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Install project requirements and retry.") from exc
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if WEIGHTS.is_file():
            _model = YOLO(str(WEIGHTS))
            return _model
        previous = Path.cwd()
        try:
            os.chdir(CACHE_DIR)
            _model = YOLO(WEIGHTS.name)
        finally:
            os.chdir(previous)
        if not WEIGHTS.is_file():
            raise RuntimeError("YOLO26s weights could not be downloaded. Check the internet connection and retry.")
        return _model


def detect(image: Image.Image, confidence: float = 0.25) -> dict:
    """Return generic detections plus a rendered detector overlay when available."""
    started = time.perf_counter()
    try:
        model = prepare()
        prediction = model.predict(source=np.asarray(image.convert("RGB")), imgsz=640, conf=confidence, device="cpu", verbose=False)[0]
        names = prediction.names
        detections = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = (round(float(value), 1) for value in box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            detections.append({"label": str(names[class_id]), "confidence": round(float(box.conf[0].item()), 4), "box": [x1, y1, x2, y2]})
        overlay = image.convert("RGB").copy()
        canvas = ImageDraw.Draw(overlay)
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            canvas.rectangle((x1, y1, x2, y2), outline=(26, 213, 188), width=max(2, round(min(image.size) / 300)))
            canvas.text((x1 + 4, max(2, y1 - 16)), f"{item['label']} {item['confidence']:.0%}", fill=(7, 24, 43), stroke_width=2, stroke_fill=(239, 255, 251))
        return {**status(), "available": True, "detections": detections, "overlay": overlay, "latency_ms": round((time.perf_counter() - started) * 1000, 1), "note": "YOLO boxes check object presence. The product-specific anomaly decision remains independent."}
    except Exception as exc:
        return {**status(), "available": False, "detections": [], "error": str(exc)[:280], "latency_ms": round((time.perf_counter() - started) * 1000, 1), "note": "The defect inspection result is still produced by the calibrated product model."}
