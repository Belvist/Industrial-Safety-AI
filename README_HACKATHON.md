# Industrial Safety AI — LinQ hackathon quickstart

Video monitoring for industrial safety: cameras → neural detection → worker
tracking → behaviour rules → confirmed incidents → operator journal.

**Current acceptance (24 October 2026):** motion tracking and predictive zone
warnings now run independently from asynchronous PPE. Real licensed industrial
video, source-bound event timelines, actual camera/zone statistics and a
reproducible browser journey are documented in [critical acceptance](docs/acceptance/README.md).
The older remote LinQ worker result in [LINQ_HARDWARE_REPORT.md](docs/LINQ_HARDWARE_REPORT.md)
is historical. **This new independent pipeline has not been reverified on LinQ;
real TPU 5 FPS and scenario C remain blocked by missing hardware access.**
CPU person performance is reported separately. Running has controlled tests;
real positive running recognition and all eight scenarios are not claimed.

## One-command start

```bash
python run.py
```

`run.py` creates the virtual environment, installs backend/frontend
dependencies, downloads the model (if absent), builds the UI and starts the
server at `http://127.0.0.1:8000`.

Manual start:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e "./backend[dev,vision,linq]" -c backend/constraints.txt
cd frontend && npm ci && npm run build && cd ..
python backend/scripts/download_model.py
python -m uvicorn app.main:app --app-dir backend --port 8000
```

## Run against LinQ TPU

The adapter speaks the organizers' RabbitMQ protocol; the compiled program
must already be in the shared cache (see `docs/LINQ_HARDWARE_REPORT.md`).

```bash
export INFERENCE_PROVIDER=linq
export LINQ_AMQP_HOST=10.0.114.30
export LINQ_TPU_PROGRAM=ppe_nosig_p999.tpu
export LINQ_TPU_CACHE_DIR=/auto/cache/hackathon93154
# credentials come from the organizers; do not commit them
export LINQ_AMQP_USER=...
export LINQ_AMQP_PASSWORD=...
python run.py
```

`GET /api/v1/system/info` shows `protocol`, `program`, `worker_verified`,
`worker_pid`, `worker_inference_ms` and `readiness`. `worker_verified=true`
means a TPU worker answered successfully; it does not mean the protocol
exposes hardware counters. Without `LINQ_AMQP_HOST` the same adapter falls
back to local ONNX.

## Demo in 60 seconds

1. Open `http://127.0.0.1:8000`.
2. **Камеры и правила** → check **Контроль бега по цеху**.
3. Start the browser camera (or upload an MP4).
4. Walk normally — no incident.
5. Run across the frame for ~1.5 s — one confirmed **Бег по цеху** incident
   with snapshot, camera, time and type.
6. Stop/walk — the incident becomes **Завершён**.
7. **Журнал инцидентов** → filter by type, camera and status.

Full script: `docs/DEMO_SCRIPT.md`.

## What is implemented

| Area | Status |
|---|---|
| FastAPI + React, file/browser/RTSP inputs | Implemented; RTSP/physical devices not verified here |
| ONNX PPE detection, tracking, danger zones | Implemented, existing reports |
| Scenario 01 — running (trajectory-based) | Implemented + unit/pipeline tests |
| Temporal confirmation, one alert per event | Implemented + tests |
| Journal filters type/camera/zone/time | Implemented + tests |
| Snapshot, feedback, Telegram transport | Snapshot/feedback tested; live Telegram not verified |
| LinQ TPU | Real RabbitMQ TPU program; `worker_verified`; parity gap on 2 low-confidence test detections; end-to-end 1.4–2.4 s |
| Scenarios 02–08 | Not implemented; not claimed |

## Tests

```bash
cd backend && python -m ruff check . && python -m pytest -q
cd frontend && npm run test && npm run build
```

## Security

Never commit `.ovpn`, VPN/Jupyter credentials, API tokens or private video.
The organizers' profile contains a private key. Rotate credentials that were
shared outside the private team channel.

## Concurrent pipeline / operator workspace update

See [the 5 FPS benchmark report](docs/LINQ_5FPS_BENCHMARK.md) for the bounded
AMQP owner thread, parallel CPU person branch, ordered analytics, honest FPS
metrics and the new desktop/mobile workspace. Keep `LINQ_MAX_IN_FLIGHT=1` until
organizers authorize a measured higher setting. This environment has no LinQ
VPN/broker access, so this update is **not a new hardware 5 FPS demonstration**.
The existing `demo-stable-c809154` tag is retained.

## Operator workspace (24 Oct 2026)

The default UI is now a rounded monochrome product workspace with simple navigation,
focused video, optional People/Radar/Events panels and a separate review queue.
Manrope is bundled locally, with Cyrillic support. Light theme is available. Start with **«Подключить первую камеру»** and follow
source → checks → observation. Ctrl/⌘ B toggles navigation; F opens focus mode.
See [the operator journey](docs/PRODUCT_WORKSPACE.md) and
[the three-minute demo](docs/DEMO_SCRIPT.md).

H.264 playback/review/export has its own isolated browser regression:
`python e2e/test_incident_review.py`. Its fixtures are explicitly labelled and
are not accuracy, physical-camera or TPU performance evidence.
