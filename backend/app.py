"""Local-only API for Inspectra. Run with uvicorn backend.app:app on 127.0.0.1."""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.store import Store, now
from backend.yolo import detect as yolo_detect, status as yolo_status

PROJECT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_BATCH = 32
MAX_REQUEST_BYTES = 110 * 1024 * 1024
LOGGER = logging.getLogger("inspectra")
Split = Literal["train", "validation", "test"]
Label = Literal["good", "defect"]


def fit(*args, **kwargs):
    from backend.engine import fit as engine_fit
    return engine_fit(*args, **kwargs)


def predict(*args, **kwargs):
    from backend.engine import predict as engine_predict
    return engine_predict(*args, **kwargs)


def assess_quality(*args, **kwargs):
    from backend.engine import assess_quality as engine_quality
    return engine_quality(*args, **kwargs)


def decode_image(payload: bytes) -> Image.Image:
    if not payload:
        raise HTTPException(400, "The image file is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Each image must be 10 MB or smaller.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise HTTPException(415, "Use a JPEG, PNG, WebP, or BMP image.")
                if opened.width * opened.height > MAX_PIXELS:
                    raise HTTPException(413, "Images must contain no more than 20 million pixels.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise HTTPException(415, "Use a single still image, not an animated image.")
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, "This file could not be decoded as a supported image.") from None


def digest_image(image: Image.Image) -> str:
    hasher = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    hasher.update(image.convert("RGB").tobytes())
    return hasher.hexdigest()


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def checked_prediction(image: Image.Image, model_path: str | Path) -> tuple[float, np.ndarray]:
    score, anomaly_map = predict(image, Path(model_path))
    score, anomaly_map = float(score), np.asarray(anomaly_map, dtype=np.float32)
    if not math.isfinite(score) or anomaly_map.ndim != 2 or not anomaly_map.size or not np.all(np.isfinite(anomaly_map)):
        raise RuntimeError("The model returned invalid numeric output; inspect the model and dataset before retrying.")
    if anomaly_map.shape != (image.height, image.width):
        anomaly_map = np.asarray(Image.fromarray(anomaly_map).resize(image.size, Image.Resampling.BILINEAR))
    return score, anomaly_map


def decide(score: float, thresholds: dict) -> str:
    if score <= thresholds["pass"]:
        return "pass"
    if score >= thresholds["fail"]:
        return "fail"
    return "review"


def calibrate(records: list[dict], model_path: Path, progress) -> dict:
    """Only explicit validation records may influence thresholds or heatmap scale."""
    scores = {"good": [], "defect": []}
    map_highs = []
    for index, record in enumerate(records):
        if record["split"] != "validation":
            raise ValueError("Calibration accepts validation images only.")
        image = open_image(record["path"])
        issues = assess_quality(image)
        if issues:
            raise ValueError(f"Validation image '{record['name']}' needs recapture: {' '.join(issues)}")
        score, anomaly_map = checked_prediction(image, model_path)
        scores[record["label"]].append(score)
        map_highs.append(float(np.quantile(anomaly_map, 0.995)))
        progress(f"Calibrating on validation images {index + 1}/{len(records)}")
    if min(len(scores["good"]), len(scores["defect"])) < 2:
        raise ValueError("Calibration needs at least two good and two defective validation images.")
    pass_threshold = float(np.quantile(scores["good"], 0.95))
    epsilon = max(abs(pass_threshold) * 0.02, 1e-6)
    fail_threshold = max(float(np.quantile(scores["defect"], 0.25)), pass_threshold + epsilon)
    overlap = min(scores["defect"]) <= max(scores["good"])
    notes = ["Thresholds use validation images only. Raw anomaly scores are not defect probabilities."]
    if overlap:
        notes.append("Validation scores overlap: some defects may pass, and some good products may require review or fail. Improve representative data before operational use.")
    if min(len(scores["good"]), len(scores["defect"])) < 20:
        notes.append("Small validation sample: thresholds are provisional. Collect more separate capture sessions.")
    return {
        "thresholds": {"pass": pass_threshold, "fail": fail_threshold},
        "heatmap_max": max(max(map_highs), fail_threshold, 1e-6),
        "calibration": {
            "good_count": len(scores["good"]), "defect_count": len(scores["defect"]),
            "good_score_range": [min(scores["good"]), max(scores["good"])],
            "defect_score_range": [min(scores["defect"]), max(scores["defect"])],
            "overlap": overlap, "notes": notes,
        },
    }


def visualizations(image: Image.Image, anomaly_map: np.ndarray, scale: float) -> tuple[Image.Image, Image.Image]:
    """A fixed model-level scale makes two image heatmaps directly comparable."""
    normalized = np.clip(anomaly_map / scale, 0, 1)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([[15, 30, 70], [35, 91, 181], [27, 186, 174], [249, 191, 50], [220, 51, 62]])
    rgb = np.stack([np.interp(normalized, stops, colors[:, i]) for i in range(3)], axis=-1)
    alpha = (0.1 + normalized * 0.58)[..., None]
    overlay = np.asarray(image, dtype=np.float32) * (1 - alpha) + rgb * alpha
    return Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


class ProductBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ReviewBody(BaseModel):
    decision: Literal["pass", "fail", "review"]
    note: str = Field(default="", max_length=1000)


def create_app(data_dir: Path | str | None = None, *, seed_demo: bool = True) -> FastAPI:
    directory = Path(data_dir or os.environ.get("INSPECTRA_DATA_DIR", PROJECT / "data")).resolve()
    media_dir = directory / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    store = Store(directory)
    job_lock = threading.Lock()
    active = {"id": None}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspectra-job")

    @asynccontextmanager
    async def lifespan(application):
        store.fail_interrupted_jobs()
        if seed_demo and not store.product("demo-label"):
            from backend.demo import create_demo_dataset
            records = create_demo_dataset(media_dir / "demo")
            for record in records:
                record["digest"] = digest_image(open_image(record["path"]))
            store.add_product("demo-label", "Packaging label · demo", "synthetic-demo")
            store.add_samples("demo-label", records, seed=True)
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    api = FastAPI(title="Inspectra", version="1.0.0", lifespan=lifespan)
    api.state.store = store
    api.state.data_dir = directory

    @api.middleware("http")
    async def local_request_limits(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).hostname not in {"localhost", "127.0.0.1", "::1"}:
                return JSONResponse({"detail": "Only local application requests are accepted."}, status_code=403)
            length = request.headers.get("content-length")
            if length:
                try:
                    if int(length) > MAX_REQUEST_BYTES:
                        return JSONResponse({"detail": "This upload is too large. Upload at most 100 MB per batch."}, status_code=413)
                except ValueError:
                    return JSONResponse({"detail": "Invalid request size."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @api.exception_handler(RequestValidationError)
    async def request_validation_error(request, exc):
        messages = [f"{'.'.join(str(part) for part in item['loc'] if part != 'body')}: {item['msg']}" for item in exc.errors()]
        return JSONResponse({"detail": "; ".join(messages)}, status_code=422)

    def product_or_404(product_id: str) -> dict:
        product = store.product(product_id)
        if not product:
            raise HTTPException(404, "Product not found. Choose or create a product first.")
        return product

    def media_url(path: str | Path) -> str:
        relative = Path(path).resolve().relative_to(media_dir)
        return "/media/" + quote(relative.as_posix(), safe="/")

    def public_model(model: dict | None) -> dict | None:
        return {key: value for key, value in model.items() if key != "path"} if model else None

    def job_progress(product_id: str, job: dict, message: str):
        job["message"] = message
        store.save_job(product_id, job)

    def train_job(product_id: str, job: dict, records: list[dict]):
        train = [row for row in records if row["split"] == "train" and row["label"] == "good"]
        validation = [row for row in records if row["split"] == "validation"]
        version_id = str(uuid4())
        model_path = directory / "models" / product_id / f"{version_id}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        progress = lambda message: job_progress(product_id, job, message)
        metadata = fit([Path(row["path"]) for row in train], model_path, progress=progress)
        if not model_path.is_file():
            raise RuntimeError("Training did not produce a model file. The previous model remains active.")
        calibration = calibrate(validation, model_path, progress)
        sources = {row["source"] for row in train + validation}
        model = {
            **metadata, **calibration, "id": version_id, "path": str(model_path.resolve()),
            "ready": True, "calibrated": True, "trained_at": now(),
            "normal_count": len(train), "source": next(iter(sources)) if len(sources) == 1 else "mixed",
            "training_sample_ids": [row["id"] for row in train],
            "validation_sample_ids": [row["id"] for row in validation],
        }
        store.activate_model(product_id, model)
        job["model_id"] = version_id

    def evaluation_job(product_id: str, job: dict, records: list[dict], model: dict):
        confusion = {label: {decision: 0 for decision in ("pass", "review", "fail", "recapture")} for label in ("good", "defect")}
        latencies = []
        for index, row in enumerate(records):
            image = open_image(row["path"])
            started = time.perf_counter()
            if assess_quality(image):
                decision = "recapture"
            else:
                score, _ = checked_prediction(image, model["path"])
                decision = decide(score, model["thresholds"])
            latencies.append((time.perf_counter() - started) * 1000)
            confusion[row["label"]][decision] += 1
            job_progress(product_id, job, f"Evaluating untouched test images {index + 1}/{len(records)}")
        good, defect = sum(confusion["good"].values()), sum(confusion["defect"].values())
        sources = {row["source"] for row in records}
        source = next(iter(sources)) if len(sources) == 1 else "mixed"
        notes = [
            "Defect recall = (defects sent to review + defects failed) / all defective test images. Recaptures are not counted as detected defects.",
            "False reject rate = good test images failed / all good test images. Review rate is reported separately.",
            "Latency measures image quality checks and model inference; it excludes camera capture, upload, and dashboard rendering.",
            "Test images are evaluated with frozen validation thresholds; this evaluation never tunes the model.",
        ]
        if source == "synthetic-demo":
            notes.append("These are deliberately generated demonstration images. Results do not establish performance on real factory or camera data.")
        elif source == "mixed":
            notes.append("This test set mixes data sources. Evaluate a separate real-camera test set before making operational claims.")
        report = {
            "model_id": model["id"], "created_at": now(), "source": source,
            "total": len(records), "good_count": good, "defect_count": defect,
            "defect_recall": (confusion["defect"]["review"] + confusion["defect"]["fail"]) / defect,
            "false_reject_rate": confusion["good"]["fail"] / good,
            "review_rate": (confusion["good"]["review"] + confusion["defect"]["review"]) / len(records),
            "recapture_rate": (confusion["good"]["recapture"] + confusion["defect"]["recapture"]) / len(records),
            "mean_latency_ms": float(np.mean(latencies)), "confusion": confusion, "notes": notes,
            "thresholds": model["thresholds"], "sample_ids": [row["id"] for row in records],
        }
        store.save_report(product_id, report)

    def run_job(product_id: str, job: dict, work, args):
        try:
            job.update(status="running", message="Starting model job…", started_at=now())
            store.save_job(product_id, job)
            work(product_id, job, *args)
            job.update(status="completed", message="Model fitted and calibrated." if job["type"] == "train" else "Test evaluation completed.", error=None)
        except Exception as exc:
            LOGGER.exception("%s job failed", job["type"])
            message = str(exc)[:600] if isinstance(exc, (ValueError, RuntimeError, FileNotFoundError)) else "The operation failed. Check the server log for details and retry."
            job.update(status="failed", message=message, error=message)
        finally:
            job["finished_at"] = now()
            store.save_job(product_id, job)
            with job_lock:
                active["id"] = None

    def enqueue(product_id: str, job_type: str, work, *args) -> dict:
        with job_lock:
            if active["id"]:
                raise HTTPException(409, "A model job is already running. Wait for it to finish.")
            job = {"id": str(uuid4()), "type": job_type, "status": "queued", "message": "Waiting to start…", "error": None, "created_at": now()}
            store.save_job(product_id, job)
            active["id"] = job["id"]
            executor.submit(run_job, product_id, job.copy(), work, args)
            return job

    @api.get("/api/products")
    def products():
        return {"items": store.products()}

    @api.post("/api/products", status_code=201)
    def add_product(body: ProductBody):
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "Enter a product name.")
        return store.add_product(str(uuid4()), name, "user-upload")

    @api.get("/api/status")
    def status(product_id: str = "demo-label"):
        product = product_or_404(product_id)
        model = store.active_model(product_id)
        return {"app_version": "1.1.0", "device": model.get("device", "cpu") if model else "cpu", "product": {key: product[key] for key in ("id", "name", "source")}, "model": public_model(model), "object_detector": yolo_status(), "dataset": store.dataset_counts(product_id), "stats": store.stats(product_id), "job": store.latest_job(product_id)}

    @api.get("/api/yolo/status")
    def yolo_detector_status():
        return yolo_status()

    @api.get("/api/samples")
    def samples(product_id: str = "demo-label", split: Split | None = "test"):
        product_or_404(product_id)
        return {"items": [{key: row[key] for key in ("id", "name", "label", "split", "source", "session_id")} | {"url": media_url(row["path"])} for row in store.samples(product_id, split)]}

    @api.post("/api/dataset", status_code=201)
    async def add_dataset(files: list[UploadFile] = File(...), split: Split = Form(...), label: Label = Form(...), session_id: str = Form(...), product_id: str = "demo-label"):
        product_or_404(product_id)
        if split == "train" and label != "good":
            raise HTTPException(422, "Training accepts good products only. Put defects in validation or test.")
        session_id = session_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", session_id):
            raise HTTPException(422, "Session ID must be 1–80 letters, numbers, spaces, dots, underscores, or hyphens and start with a letter or number.")
        if len(files) > MAX_BATCH:
            raise HTTPException(413, f"Upload at most {MAX_BATCH} images in one batch.")
        records, written = [], []
        total = 0
        try:
            for upload in files:
                payload = await upload.read(MAX_IMAGE_BYTES + 1)
                total += len(payload)
                if total > 100 * 1024 * 1024:
                    raise HTTPException(413, "Upload at most 100 MB per batch.")
                image = await run_in_threadpool(decode_image, payload)
                sample_id = str(uuid4())
                destination = media_dir / "dataset" / product_id / f"{sample_id}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                await run_in_threadpool(image.save, destination, "PNG")
                written.append(destination)
                records.append({"id": sample_id, "name": (upload.filename or "image").replace("\\", "/").rsplit("/", 1)[-1][:160], "path": str(destination), "label": label, "split": split, "session_id": session_id, "source": "user-upload", "digest": digest_image(image)})
            try:
                store.add_samples(product_id, records)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from None
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        finally:
            for upload in files:
                await upload.close()
        return {"added": len(records), "message": f"Added {len(records)} {label} images to {split}. Keep future capture sessions separate across splits."}

    def inspect_image(product_id: str, image: Image.Image, source: str) -> dict:
        started = time.perf_counter()
        model = store.active_model(product_id)
        quality_issues = assess_quality(image)
        inspection_id = str(uuid4())
        folder = media_dir / "inspections" / inspection_id
        score, heatmap_url, overlay_url, yolo_overlay_url = None, None, None, None
        object_detection = yolo_detect(image)
        if quality_issues:
            decision = "recapture"
        else:
            if not model or not model.get("calibrated"):
                raise HTTPException(409, "Fit and calibrate a model for this product before inspecting images.")
            try:
                score, anomaly_map = checked_prediction(image, model["path"])
            except Exception as exc:
                LOGGER.exception("Inspection model inference failed")
                message = str(exc)[:400] if isinstance(exc, (RuntimeError, FileNotFoundError)) else "Model inference failed. Check the server log and model before retrying."
                raise HTTPException(503, message) from None
            decision = decide(score, model["thresholds"])
            heatmap, overlay = visualizations(image, anomaly_map, model["heatmap_max"])
            folder.mkdir(parents=True, exist_ok=True)
            heatmap.save(folder / "heatmap.png")
            overlay.save(folder / "overlay.png")
            heatmap_url, overlay_url = media_url(folder / "heatmap.png"), media_url(folder / "overlay.png")
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / "original.png")
        yolo_overlay = object_detection.pop("overlay", None)
        if yolo_overlay is not None:
            yolo_overlay.save(folder / "yolo.png")
            yolo_overlay_url = media_url(folder / "yolo.png")
        item = {
            "id": inspection_id, "image_url": media_url(folder / "original.png"),
            "heatmap_url": heatmap_url, "overlay_url": overlay_url, "decision": decision, "score": score,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_id": model["id"] if model else None,
            "thresholds": model["thresholds"] if model else {"pass": None, "fail": None},
            "quality": {"ok": not quality_issues, "warnings": quality_issues},
            "object_detection": object_detection, "yolo_overlay_url": yolo_overlay_url,
            "source": source, "created_at": now(), "review": None,
        }
        store.save_inspection(product_id, item)
        return item

    @api.post("/api/inspect")
    async def inspect(product_id: str = "demo-label", file: UploadFile | None = File(None), sample_id: str | None = Form(None)):
        product_or_404(product_id)
        if (file is None) == (not sample_id):
            raise HTTPException(422, "Provide exactly one image file or sample ID.")
        if file:
            try:
                image = await run_in_threadpool(decode_image, await file.read(MAX_IMAGE_BYTES + 1))
            finally:
                await file.close()
            source = "user-upload"
        else:
            sample = store.sample(product_id, sample_id)
            if not sample:
                raise HTTPException(404, "Sample not found for this product.")
            image = await run_in_threadpool(open_image, sample["path"])
            source = sample["source"]
        return await run_in_threadpool(inspect_image, product_id, image, source)

    @api.get("/api/inspections")
    def inspections(product_id: str = "demo-label", limit: int = Query(50, ge=1, le=500)):
        product_or_404(product_id)
        return {"items": store.inspections(product_id, limit)}

    @api.get("/api/inspections/export")
    def export(product_id: str = "demo-label"):
        product_or_404(product_id)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["id", "created_at", "machine_decision", "raw_anomaly_score", "latency_ms", "model_id", "pass_threshold", "fail_threshold", "source", "human_decision", "human_note"])
        for item in store.inspections(product_id):
            review = item.get("review") or {}
            note = review.get("note", "")
            if note.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                note = "'" + note
            writer.writerow([item["id"], item["created_at"], item["decision"], item["score"], item["latency_ms"], item["model_id"], item["thresholds"]["pass"], item["thresholds"]["fail"], item["source"], review.get("decision", ""), note])
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="inspectra-inspections.csv"'})

    @api.post("/api/inspections/{inspection_id}/review")
    def review(inspection_id: str, body: ReviewBody, product_id: str = "demo-label"):
        product_or_404(product_id)
        item = store.review_inspection(product_id, inspection_id, {"decision": body.decision, "note": body.note.strip(), "created_at": now()})
        if not item:
            raise HTTPException(404, "Inspection not found for this product.")
        return item

    @api.post("/api/train", status_code=202)
    def train(product_id: str = "demo-label"):
        product_or_404(product_id)
        records = store.samples(product_id)
        counts = store.dataset_counts(product_id)
        if counts["train_good"] < 5 or counts["validation_good"] < 2 or counts["validation_defect"] < 2:
            raise HTTPException(422, "Fit requires at least 5 good training images, 2 good validation images, and 2 defective validation images from separate sessions.")
        return enqueue(product_id, "train", train_job, records)

    @api.post("/api/evaluate", status_code=202)
    def evaluate(product_id: str = "demo-label"):
        product_or_404(product_id)
        model = store.active_model(product_id)
        if not model or not model.get("calibrated"):
            raise HTTPException(409, "Fit and calibrate this product's model before evaluation.")
        records = store.samples(product_id, "test")
        if not all(any(row["label"] == label for row in records) for label in ("good", "defect")):
            raise HTTPException(422, "Evaluation requires both good and defective test images from separate capture sessions.")
        return enqueue(product_id, "evaluate", evaluation_job, records, model)

    @api.get("/api/evaluation")
    def evaluation(product_id: str = "demo-label"):
        product_or_404(product_id)
        return {"report": store.report(product_id)}

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str, product_id: str = "demo-label"):
        product_or_404(product_id)
        job = store.job(product_id, job_id)
        if not job:
            raise HTTPException(404, "Job not found for this product.")
        return job

    api.mount("/media", StaticFiles(directory=media_dir), name="media")
    frontend = PROJECT / "frontend" / "dist"

    @api.get("/{requested_path:path}", include_in_schema=False)
    def frontend_or_404(requested_path: str):
        if requested_path == "api" or requested_path.startswith("api/"):
            raise HTTPException(404, "API route not found.")
        requested = (frontend / requested_path).resolve()
        if not requested.is_relative_to(frontend.resolve()):
            raise HTTPException(404, "File not found.")
        if requested.is_file():
            return FileResponse(requested)
        index = frontend / "index.html"
        if index.is_file() and "." not in Path(requested_path).name:
            return FileResponse(index)
        raise HTTPException(404, "Frontend is not built yet. Build frontend/dist or run the Vite development server.")

    return api


app = create_app()
