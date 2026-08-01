"""FastAPI + ONNX Runtime serving for a YOLOv8 detector.

Endpoints:
    POST /predict   multipart image -> detections JSON
    GET  /healthz   liveness/readiness
    GET  /metrics   Prometheus exposition

Every prediction is also appended as JSONL to PRED_LOG_DIR so the Evidently
drift CronJob can compare live traffic against the reference window.
"""
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
PRED_LOG_DIR = Path(os.environ.get("PRED_LOG_DIR", "/data/predictions"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

REQUESTS = Counter("inference_requests_total", "Inference requests", ["status"])
LATENCY = Histogram("inference_latency_seconds", "End-to-end inference latency",
                    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
DETECTIONS = Counter("detections_total", "Detections by class", ["class_name"])
CONFIDENCE = Histogram("detection_confidence", "Detection confidence scores",
                       buckets=(0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95))
MODEL_INFO = Gauge("model_version_info", "Deployed model version", ["version"])
MODEL_INFO.labels(version=MODEL_VERSION).set(1)

app = FastAPI(title="cv-detector")
session: ort.InferenceSession | None = None


@app.on_event("startup")
def load_model() -> None:
    global session
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    PRED_LOG_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(img: Image.Image) -> tuple[np.ndarray, float, float]:
    w0, h0 = img.size
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]           # (1, 3, H, W)
    return x, w0 / IMG_SIZE, h0 / IMG_SIZE


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(output: np.ndarray, sx: float, sy: float) -> list[dict]:
    # YOLOv8 ONNX output: (1, 4 + num_classes, 8400)
    preds = output[0].T                       # (8400, 84)
    boxes_cxcywh, class_scores = preds[:, :4], preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    mask = confidences >= CONF_THRESHOLD
    if not mask.any():
        return []
    boxes_cxcywh, class_ids, confidences = boxes_cxcywh[mask], class_ids[mask], confidences[mask]

    cx, cy, w, h = boxes_cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(boxes, confidences, IOU_THRESHOLD)
    dets = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cls = int(class_ids[i])
        dets.append({
            "class_id": cls,
            "class_name": COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls),
            "confidence": round(float(confidences[i]), 4),
            "box": [round(float(v), 1) for v in (x1 * sx, y1 * sy, x2 * sx, y2 * sy)],
        })
    return dets


def log_prediction(img: Image.Image, dets: list[dict], latency: float) -> None:
    """Append per-request features for the drift job (data drift + prediction drift)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "width": img.size[0],
        "height": img.size[1],
        "n_detections": len(dets),
        "mean_confidence": float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
        "top_class": dets[0]["class_name"] if dets else "none",
        "latency_s": round(latency, 4),
    }
    day_file = PRED_LOG_DIR / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with open(day_file, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(503, "model not loaded")
    start = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(await file.read()))
        x, sx, sy = preprocess(img)
        output = session.run(None, {session.get_inputs()[0].name: x})[0]
        dets = postprocess(output, sx, sy)
        latency = time.perf_counter() - start

        LATENCY.observe(latency)
        REQUESTS.labels(status="ok").inc()
        for d in dets:
            DETECTIONS.labels(class_name=d["class_name"]).inc()
            CONFIDENCE.observe(d["confidence"])
        log_prediction(img, dets, latency)
        return {"detections": dets, "latency_ms": round(latency * 1000, 1),
                "model_version": MODEL_VERSION}
    except HTTPException:
        raise
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(500, f"inference failed: {exc}") from exc


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": session is not None,
            "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
