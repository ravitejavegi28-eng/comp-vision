"""Local PatchCore-style anomaly detector with frozen pretrained ResNet18 features.

The model stores representative normal image patches and scores a new patch by
its nearest normal neighbor. It has no external inference API and never uses
random fallback weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / ".cache" / "resnet18-f37072fd.pth"
INPUT_SIZE = 224
MAX_CANDIDATES = 16_000
MAX_MEMORY = 512
LOCK = threading.RLock()
_CACHE: dict[str, dict] = {}


def _torch():
    import torch
    from torchvision.models import resnet18
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    return torch, resnet18


def _backbone():
    torch, resnet18 = _torch()
    if not WEIGHTS.is_file():
        raise RuntimeError("Pretrained weights are missing. Run: python scripts/prepare_weights.py (internet required once).")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tensor(image: Image.Image):
    torch, _ = _torch()
    image = ImageOps.exif_transpose(image).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def _features(model, batch):
    """Concatenate ResNet layer2 and layer3 into normalized 28x28 patch vectors."""
    torch, _ = _torch()
    with torch.inference_mode():
        value = model.conv1(batch)
        value = model.bn1(value)
        value = model.relu(value)
        value = model.maxpool(value)
        value = model.layer1(value)
        layer2 = model.layer2(value)
        layer3 = model.layer3(layer2)
        layer3 = torch.nn.functional.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([layer2, layer3], dim=1)
        features = torch.nn.functional.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        features = torch.nn.functional.normalize(features, dim=1)
        return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu()


def _representative(features):
    """Deterministic patch sampling sized for a local laptop memory bank."""
    torch, _ = _torch()
    count = min(MAX_MEMORY, len(features))
    if count == len(features):
        return features
    indices = torch.randperm(len(features), generator=torch.Generator().manual_seed(42))[:count]
    return features[indices]


def fit(paths: list[Path], destination: Path, progress: Callable[[str], None] | None = None) -> dict:
    if len(paths) < 5:
        raise ValueError("At least five separate good training images are required.")
    if len(paths) > 1000:
        raise ValueError("This local prototype supports at most 1,000 training images per fit. Curate a representative subset.")
    progress = progress or (lambda _: None)
    with LOCK:
        torch, _ = _torch()
        progress("Loading the pretrained ResNet18 feature extractor")
        model = _backbone()
        candidates = []
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                issues = assess_quality(image)
                if issues:
                    raise ValueError(f"Training image {Path(path).name}: {issues[0]}")
                candidates.append(_features(model, _tensor(image)))
            progress(f"Extracting normal features: {index + 1}/{len(paths)} images")
        features = torch.cat(candidates)
        if len(features) > MAX_CANDIDATES:
            features = features[torch.randperm(len(features), generator=torch.Generator().manual_seed(42))[:MAX_CANDIDATES]]
        progress("Selecting the representative PatchCore memory bank")
        memory_bank = _representative(features)
        metadata = {
            "algorithm": "PatchCore", "backbone": "ResNet18 / ImageNet", "normal_count": len(paths),
            "input_size": INPUT_SIZE, "device": "cpu", "memory_patches": int(memory_bank.shape[0]),
            "candidate_limit": MAX_CANDIDATES, "implementation": "Local deterministic nearest-neighbor feature memory bank",
            "score_definition": "Maximum nearest-neighbor patch distance; not a probability",
            "roi": "Whole uploaded or camera-captured image; maintain consistent framing",
        }
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save({"format": 2, "metadata": metadata, "memory_bank": memory_bank}, temporary)
        temporary.replace(destination)
        _CACHE.clear()
        _CACHE[str(destination.resolve())] = {"model": model, "memory_bank": memory_bank}
        progress("Model fitted; calibrating on separate validation samples")
        return metadata


def _load(model_path: Path):
    key = str(Path(model_path).resolve())
    if key not in _CACHE:
        torch, _ = _torch()
        checkpoint = torch.load(key, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != 2:
            raise ValueError("Unsupported model format. Fit a new model using the Dataset & model page.")
        _CACHE.clear()
        _CACHE[key] = {"model": _backbone(), "memory_bank": checkpoint["memory_bank"].float()}
    return _CACHE[key]


def predict(image: Image.Image, model_path: Path) -> tuple[float, np.ndarray]:
    with LOCK:
        torch, _ = _torch()
        artifact = _load(model_path)
        embedding = _features(artifact["model"], _tensor(image))
        distances = (2.0 - 2.0 * (embedding @ artifact["memory_bank"].T)).clamp_min(0).sqrt()
        patch_scores = distances.min(dim=1).values.reshape(28, 28)
        score = float(patch_scores.max())
        heat = np.asarray(Image.fromarray(patch_scores.numpy().astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32)
        if not np.isfinite(score) or not np.isfinite(heat).all():
            raise RuntimeError("The model returned invalid scores. Recapture the image or fit a new reference.")
        return score, heat


def assess_quality(image: Image.Image) -> list[str]:
    warnings = []
    if min(image.size) < 128:
        warnings.append("Image is too small. Capture at least 128 pixels on both sides.")
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    if float(gray.mean()) < 12:
        warnings.append("Image is too dark. Increase lighting and recapture.")
    elif float(gray.mean()) > 248:
        warnings.append("Image is overexposed or blank. Reframe the product and reduce exposure.")
    elif float(gray.std()) < 3:
        warnings.append("Image has too little detail. Place the product in the inspection area and recapture.")
    return warnings
