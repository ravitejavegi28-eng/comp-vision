# Inspectra

Local visual quality inspection workstation for a repeated product. It builds a PatchCore-style nearest-neighbor reference from good images, uses separate validation images to set thresholds, and evaluates only on held-out test images.

## Start

```powershell
cd "D:\HRM Solutions\quality-inspector"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`. The first launch includes a clearly marked synthetic packaging demo. Select **Fit demo model**, wait for calibration, inspect held-out samples, then use **Evaluation**.

## Real product workflow

1. In **Dataset & model**, create a product workspace.
2. Upload at least five good training images, two separate good validation images, and two separate defective validation images. Keep each capture session in one split.
3. Fit and calibrate. Inspect images through upload or camera capture.
4. Upload untouched good and defective test images and run **Evaluation**. Do not reuse them while fitting or calibrating.

The model's score is a visual distance, not a probability. Keep the camera, product framing, and lighting consistent with the normal reference. Synthetic samples are software fixtures and do not represent factory performance.

## Technical notes

- Local FastAPI server, SQLite metadata, local images, and no external inference API.
- Frozen ImageNet-pretrained ResNet18 extracts layer-2 and layer-3 patch features. A deterministic normal-feature memory bank drives anomaly heatmaps.
- `scripts/prepare_weights.py` downloads and verifies the one-time public ResNet18 checkpoint; it is already cached locally for this workspace.
- The locally cached YOLO26s checkpoint adds a COCO object-presence check to every inspection. It does not claim to recognize scratches, tears, or packaging faults; train a custom YOLO detector only after collecting labelled defect boxes.
