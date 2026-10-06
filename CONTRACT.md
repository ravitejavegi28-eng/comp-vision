# Quality Inspector implementation contract

Standalone local app in this directory; do not edit the surrounding HRM website.

## Architecture and ownership
- `backend/app.py`, `backend/store.py`, `backend/tests/`: API agent.
- `frontend/`: frontend agent.
- `backend/engine.py`, `backend/demo.py`, scripts, documentation, dependency setup: root agent.
- Backend serves built `frontend/dist` and `/media` from local project data. Bind 127.0.0.1:8787.
- All data local, no secrets or external inference APIs. Explicit synthetic demo provenance.

## Engine interface (root implements)
`from backend.engine import fit, predict, assess_quality`

`fit(paths: list[Path], destination: Path, progress: Callable[[str],None] | None = None) -> dict`
fits a local PatchCore-style nearest-neighbor memory bank over pretrained ResNet18 features, saves it to destination, and returns algorithm, backbone, normal_count, input_size, device. Raises helpful errors if weights are unavailable; no silent fake inference.

`predict(image: PIL.Image.Image, model_path: Path) -> tuple[float, np.ndarray]`
returns raw image anomaly score and 2D anomaly map at image size. Model cached, inference serialized.

`assess_quality(image: PIL.Image.Image) -> list[str]`
returns actionable issues for very small, blank, extremely dark/bright images. Poor quality means RECAPTURE, never PASS. No claims of guaranteed focus/framing detection.

`from backend.demo import create_demo_dataset`
`create_demo_dataset(directory: Path) -> list[dict]` writes deterministic synthetic packaging images once, returns records with id, path (absolute str), name, label (`good`/`defect`), split (`train`/`validation`/`test`), session_id, source (`synthetic-demo`). Training good only. Defect families reserved in test. Does not train model.

## API contract
JSON errors: HTTP error with `detail` human readable. No stack traces to client.
All product-scoped endpoints take query `product_id`, default `demo-label`.

- GET `/api/products` -> `{items:[{id,name,source}]}`; demo-label exists seeded at first launch.
- POST `/api/products` JSON `{name}` -> product (id generated); own data isolated from demo.
- GET `/api/status` -> `{app_version, device, product:{id,name,source}, model:null|{id,ready,algorithm,backbone,trained_at,normal_count,thresholds:{pass,fail},calibrated}, dataset:{train_good,validation_good,validation_defect,test_good,test_defect}, stats:{total,pass,review,fail,recapture}, job:null|{id,type,status,message,error}}`
- GET `/api/samples?split=test` -> `{items:[{id,name,label,split,url,source,session_id}]}`
- POST `/api/inspect` multipart fields `file` OR `sample_id` -> `{id,image_url,heatmap_url,overlay_url,decision,score,latency_ms,model_id,thresholds:{pass,fail},quality:{ok,warnings:[]},source,created_at,review:null}`. decisions lowercase pass/review/fail/recapture. `score` null for recapture. Raw scores not percentages.
- GET `/api/inspections?limit=50` -> `{items:[inspection...]}` newest first.
- POST `/api/inspections/{id}/review` JSON `{decision:"pass"|"fail"|"review",note}` -> updated inspection. Human correction does not change original machine decision.
- GET `/api/inspections/export` -> CSV download.
- POST `/api/dataset` multipart `files` (multiple), `split`, `label`, `session_id` -> `{added, message}`. Limit upload sizes, accept image decoding only, reject defects in train, same session across splits, exact duplicate across splits. Product isolation enforced.
- POST `/api/train` -> `{id,type,status,message}` background job. Require train normal >=5 and separate validation good and defective >=2 each. Fit to a NEW version then calibrate on validation only; atomic activation. Keep previous model usable if job fails. Test never used for tuning.
- POST `/api/evaluate` -> job; requires fitted calibrated model and test good/defective. POST blocked while other job running.
- GET `/api/evaluation` -> `{report:null|{model_id,created_at,source,total,good_count,defect_count,defect_recall,false_reject_rate,review_rate,mean_latency_ms,confusion:{good:{pass,review,fail,recapture},defect:{pass,review,fail,recapture}},notes:[]}}`. Recall counts review/fail as caught, definition explicit. Recapture reported separately.
- GET `/api/jobs/{id}` -> job.

## Calibration
Use validation only. Pass threshold = upper quantile of good scores; fail threshold chosen using validation defect scores and always > pass. Report overlap; do not claim calibrated probabilities. Heatmaps use model-level fixed scale, never per-image normalization. Thresholds/version saved with every inspection. Empty model UI clearly prompts fit. Backend may train demo on demand from button, startup need not wait.

## Frontend experience
Brand: Inspectra, local visual quality inspection workstation. Ice gray background #eef2f6, navy #18324b, cobalt #275bce, teal #168376, amber #aa6e12, red #c3444f. Segoe UI/system sans, strong legible tabular numbers. Calm white chrome around a large dark blue inspection stage; tiny calibration corner markers only inside image viewer. No fake stats. Sidebar tabs Inspect, Dataset & model, Evaluation, History. Product selector; persistent synthetic demo badge; API status. Inspect includes upload, camera capture with permission/error states, sample gallery, heatmap toggle, result card, review. Dataset allows upload with split/label/session, new product, fit and progress. Evaluation includes metrics definitions and real report. History includes detail/review and CSV. Responsive and keyboard accessible.

Use React + Vite JS (avoid unnecessary TS setup), lucide-react. Same origin `/api` and `/media`; Vite proxy to :8787. Production frontend served by FastAPI.
