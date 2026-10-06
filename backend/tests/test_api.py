"""API integrity tests. Only this test module substitutes deterministic model calls."""
from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import app as module


def png(value: int, *, size=(32, 24)) -> bytes:
    image = Image.new("RGB", size, (value, value, value))
    if value:
        image.putpixel((1, 1), (min(value + 1, 255), value, value))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    def fake_fit(paths, destination, progress=None):
        destination.write_bytes(b"test-only-model")
        if progress:
            progress("Test model fitted")
        return {"algorithm": "test-double", "backbone": "test-double", "normal_count": len(paths), "input_size": 32, "device": "cpu"}

    def fake_predict(image, path):
        assert Path(path).is_file()
        score = image.getpixel((0, 0))[0] / 255
        return score, np.full((image.height, image.width), score, dtype=np.float32)

    monkeypatch.setattr(module, "fit", fake_fit)
    monkeypatch.setattr(module, "predict", fake_predict)
    monkeypatch.setattr(module, "assess_quality", lambda image: ["Image is too dark. Recapture with more light."] if image.getpixel((0, 0))[0] == 0 else [])
    monkeypatch.setattr(module, "yolo_detect", lambda image: {"model": "YOLO26s", "available": True, "detections": [{"label": "bottle", "confidence": 0.91, "box": [1, 1, 8, 8]}], "latency_ms": 4.2, "overlay": image.copy(), "note": "test detector"})
    app = module.create_app(tmp_path / "state", seed_demo=False)
    app.state.store.add_product("demo-label", "Test fixture", "user-upload")
    with TestClient(app) as connection:
        yield connection


def upload(client, values, *, split="train", label="good", session="train-a", product="demo-label"):
    return client.post(f"/api/dataset?product_id={product}", data={"split": split, "label": label, "session_id": session}, files=[("files", (f"image-{value}.png", png(value), "image/png")) for value in values])


def dataset(client, product="demo-label"):
    for values, split, label, session in [
        ([40, 41, 42, 43, 44], "train", "good", "train-a"),
        ([50, 52], "validation", "good", "validation-a"),
        ([150, 160], "validation", "defect", "validation-b"),
        ([49, 53], "test", "good", "test-a"),
        ([170, 180], "test", "defect", "test-b"),
    ]:
        response = upload(client, values, split=split, label=label, session=session, product=product)
        assert response.status_code == 201, response.text


def wait_job(client, response, product="demo-label"):
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    for _ in range(250):
        job = client.get(f"/api/jobs/{job_id}?product_id={product}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail("Test job did not finish")


def train(client):
    job = wait_job(client, client.post("/api/train"))
    assert job["status"] == "completed", job
    return client.get("/api/status").json()["model"]


def test_products_start_untrained_and_are_isolated(client):
    response = client.get("/api/status").json()
    assert response["model"] is None
    assert response["stats"]["total"] == 0
    product = client.post("/api/products", json={"name": "Camera labels"}).json()
    assert product["source"] == "user-upload"
    assert upload(client, [60], product=product["id"]).status_code == 201
    assert client.get("/api/status").json()["dataset"]["train_good"] == 0
    assert client.get(f"/api/status?product_id={product['id']}").json()["dataset"]["train_good"] == 1
    assert client.get("/api/status?product_id=does-not-exist").status_code == 404
    assert client.post("/api/products", json={"name": " "}).status_code == 422


def test_dataset_rejects_defect_training_and_leakage(client):
    assert upload(client, [40], label="defect").status_code == 422
    assert upload(client, [40]).status_code == 201
    response = upload(client, [41], split="validation", session="train-a")
    assert response.status_code == 409
    assert "Session" in response.json()["detail"]
    response = upload(client, [40], split="test", session="test-a")
    assert response.status_code == 409
    assert "duplicates cannot cross splits" in response.json()["detail"]
    assert upload(client, [40]).status_code == 409
    assert client.get("/api/status").json()["dataset"]["train_good"] == 1


def test_batch_rolls_back_completely_if_any_image_conflicts(client):
    assert upload(client, [40]).status_code == 201
    response = upload(client, [41, 40], split="test", session="test-a")
    assert response.status_code == 409
    assert client.get("/api/status").json()["dataset"]["test_good"] == 0
    files = list(client.app.state.data_dir.glob("media/dataset/**/*.png"))
    assert len(files) == 1


def test_image_validation_and_upload_limit(client):
    response = client.post("/api/inspect", files={"file": ("attack.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    response = client.post("/api/inspect", files={"file": ("big.png", b"x" * (module.MAX_IMAGE_BYTES + 1), "image/png")})
    assert response.status_code == 413
    response = client.post("/api/inspect", files={"file": ("photo.png", png(60), "image/png")}, data={"sample_id": "abc"})
    assert response.status_code == 422
    assert client.post("/api/inspect").status_code == 422
    assert client.get("/api/samples?split=wrong").status_code == 422


def test_recapture_never_passes_and_does_not_require_model(client):
    response = client.post("/api/inspect", files={"file": ("dark.png", png(0), "image/png")})
    assert response.status_code == 200
    item = response.json()
    assert item["decision"] == "recapture"
    assert item["score"] is None
    assert item["heatmap_url"] is None
    assert item["quality"]["ok"] is False
    assert item["model_id"] is None
    assert client.get(item["image_url"]).status_code == 200
    assert client.post("/api/inspect", files={"file": ("good.png", png(50), "image/png")}).status_code == 409


def test_training_requires_separate_validation_examples(client):
    assert client.post("/api/train").status_code == 422
    assert upload(client, [40, 41, 42, 43, 44]).status_code == 201
    assert client.post("/api/train").status_code == 422
    assert client.post("/api/evaluate").status_code == 409


def test_calibration_uses_validation_only_and_saves_version(client, monkeypatch):
    dataset(client)
    seen_fit, seen_predictions = [], []
    original_fit, original_predict = module.fit, module.predict

    def capture_fit(paths, destination, progress=None):
        seen_fit.extend(Image.open(path).getpixel((0, 0))[0] for path in paths)
        return original_fit(paths, destination, progress)

    def capture_predict(image, path):
        seen_predictions.append(image.getpixel((0, 0))[0])
        return original_predict(image, path)

    monkeypatch.setattr(module, "fit", capture_fit)
    monkeypatch.setattr(module, "predict", capture_predict)
    model = train(client)
    assert sorted(seen_fit) == [40, 41, 42, 43, 44]
    assert sorted(seen_predictions) == [50, 52, 150, 160]
    assert model["thresholds"]["pass"] == pytest.approx(np.quantile([50 / 255, 52 / 255], 0.95))
    assert model["thresholds"]["fail"] > model["thresholds"]["pass"]
    assert model["calibration"]["overlap"] is False
    assert model["calibration"]["good_count"] == 2
    assert "path" not in model
    assert len(list(client.app.state.data_dir.glob("models/**/*.pt"))) == 1


def test_inspection_artifacts_reviews_csv_and_version_history(client):
    dataset(client)
    model = train(client)
    sample = client.get("/api/samples").json()["items"][0]
    item = client.post("/api/inspect", data={"sample_id": sample["id"]}).json()
    assert item["decision"] in {"pass", "review", "fail"}
    assert item["model_id"] == model["id"]
    assert item["thresholds"] == model["thresholds"]
    assert item["object_detection"]["model"] == "YOLO26s"
    assert item["object_detection"]["detections"][0]["label"] == "bottle"
    for key in ["image_url", "heatmap_url", "overlay_url"]:
        artifact = client.get(item[key])
        assert artifact.status_code == 200
        assert Image.open(io.BytesIO(artifact.content)).size == (32, 24)
    review = client.post(f"/api/inspections/{item['id']}/review", json={"decision": "fail", "note": "=SUM(1,2)"}).json()
    assert review["decision"] == item["decision"]
    assert review["review"]["decision"] == "fail"
    exported = list(csv.DictReader(io.StringIO(client.get("/api/inspections/export").text)))
    assert exported[0]["machine_decision"] == item["decision"]
    assert exported[0]["human_decision"] == "fail"
    assert exported[0]["human_note"].startswith("'=")
    new_model = train(client)
    assert model["id"] != new_model["id"]
    history = client.get("/api/inspections").json()["items"]
    assert history[0]["model_id"] == model["id"]
    assert len(list(client.app.state.data_dir.glob("models/**/*.pt"))) == 2


def test_product_isolation_for_samples_reviews_jobs_and_reports(client):
    dataset(client)
    train(client)
    sample = client.get("/api/samples").json()["items"][0]
    item = client.post("/api/inspect", data={"sample_id": sample["id"]}).json()
    product_id = client.post("/api/products", json={"name": "Separate item"}).json()["id"]
    assert client.post(f"/api/inspect?product_id={product_id}", data={"sample_id": sample["id"]}).status_code == 404
    assert client.post(f"/api/inspections/{item['id']}/review?product_id={product_id}", json={"decision": "pass"}).status_code == 404
    assert client.get(f"/api/inspections?product_id={product_id}").json()["items"] == []
    assert client.get(f"/api/evaluation?product_id={product_id}").json()["report"] is None


def test_evaluation_uses_frozen_thresholds_and_only_test_images(client, monkeypatch):
    dataset(client)
    model = train(client)
    seen = []
    original_predict = module.predict

    def capture(image, path):
        seen.append(image.getpixel((0, 0))[0])
        return original_predict(image, path)

    monkeypatch.setattr(module, "predict", capture)
    job = wait_job(client, client.post("/api/evaluate"))
    assert job["status"] == "completed"
    report = client.get("/api/evaluation").json()["report"]
    assert sorted(seen) == [49, 53, 170, 180]
    assert report["total"] == 4
    assert report["defect_recall"] == 1.0
    assert report["false_reject_rate"] == 0.0
    assert report["review_rate"] == 0.25
    assert report["confusion"]["good"] == {"pass": 1, "review": 1, "fail": 0, "recapture": 0}
    assert client.get("/api/status").json()["model"]["thresholds"] == model["thresholds"]
    assert client.get("/api/status").json()["stats"]["total"] == 0


def test_failed_retraining_keeps_previous_model(client, monkeypatch):
    dataset(client)
    model = train(client)

    def broken_fit(*args, **kwargs):
        raise RuntimeError("Pretrained weights unavailable. Download weights and retry.")

    monkeypatch.setattr(module, "fit", broken_fit)
    job = wait_job(client, client.post("/api/train"))
    assert job["status"] == "failed"
    assert "weights unavailable" in job["error"]
    assert client.get("/api/status").json()["model"]["id"] == model["id"]


def test_concurrent_jobs_are_rejected_without_blocking_previous_model(client, monkeypatch):
    dataset(client)
    old_model = train(client)
    entered, release = threading.Event(), threading.Event()
    original_fit = module.fit

    def slow_fit(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "fit", slow_fit)
    response = client.post("/api/train")
    try:
        assert entered.wait(2)
        assert client.post("/api/train").status_code == 409
        assert client.post("/api/evaluate").status_code == 409
        result = client.post("/api/inspect", files={"file": ("good.png", png(49), "image/png")}).json()
        assert result["model_id"] == old_model["id"]
    finally:
        release.set()
    assert wait_job(client, response)["status"] == "completed"


def test_heatmap_scale_is_fixed_across_images():
    base = Image.new("RGB", (10, 10), "white")
    first = np.full((10, 10), 0.2, dtype=np.float32)
    second = first.copy()
    second[0, 0] = 1.0
    heatmap_a, _ = module.visualizations(base, first, 2.0)
    heatmap_b, _ = module.visualizations(base, second, 2.0)
    assert heatmap_a.getpixel((5, 5)) == heatmap_b.getpixel((5, 5))
    assert heatmap_b.getpixel((0, 0)) != heatmap_b.getpixel((5, 5))


def test_cross_site_mutations_and_unknown_api_are_rejected(client):
    assert client.post("/api/products", json={"name": "Attack"}, headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/api/status").headers["cache-control"] == "no-store"


def test_persistence_marks_interrupted_jobs_failed(tmp_path):
    from backend.store import Store, now
    store = Store(tmp_path)
    store.add_product("product", "Persisted", "user-upload")
    store.save_job("product", {"id": "job", "created_at": now(), "status": "running", "type": "train"})
    reopened = Store(tmp_path)
    reopened.fail_interrupted_jobs()
    assert reopened.product("product")["name"] == "Persisted"
    assert reopened.job("product", "job")["status"] == "failed"
