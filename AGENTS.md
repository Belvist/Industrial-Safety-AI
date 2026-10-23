# AGENTS.md — project memory

## What this project is

Industrial Safety AI: FastAPI + React video monitoring for industrial safety.
Pipeline: camera → ONNX detection → worker tracking → rules → TemporalEngine →
incidents → SQLite/snapshots/Telegram → operator UI. LinQ TPU is the required
hackathon accelerator (see `docs/LINQ_CASE_TRACEABILITY.md`).

## Current state (23 Oct 2026)

- Git history was consolidated into a single `Initial commit` at the owner's
  request; `main` was force-pushed and other remote branches deleted.
- Scenario 01 (running) implemented in `backend/app/services/behavior.py`,
  integrated in `runtime.process`, configurable per camera (`running_enabled`,
  `running_speed`, `running_min_seconds`).
- Incident journal filters implemented server-side: `type`, `camera_id`,
  `zone`, `from_time`, `to_time`, `status`, pagination. SQLite migration adds
  `zone_name` and indexes.
- LinQ adapter is real: RabbitMQ `tpu.direct`/`input`, program
  `ppe_nosig_p999.tpu` in `/auto/cache/hackathon93154`, protocol
  `linq-rabbitmq-tpu-v1`. Real worker replies: `status_code=0`,
  `output0=(1,18,8400)`, `worker_verified=true`.
- Program SHA-256: `ac9fc87355edf9fb5552537c02f4b1a584370859dc6a7b96a4e6f1ec4b22eabd`.
- Critical compile workaround: the final `/model.22/Sigmoid` is removed before
  quantization, and calibration uses `activation_percentile=99.9`. Without
  this the final conv+bias+LUT mis-scales class scores (0.87 -> 0.12).
  Host-side sigmoid is applied in `LinqProvider._decode`.
- Parity: val 20/20 detections; test 11/14 (2 low-confidence `no_helmet`
  misses, 1 extra). End-to-end p50 1.4–2.4 s; worker p50 55–75 ms. Do not
  report worker-only FPS as end-to-end. No independent hardware counters:
  `hardware_verified=false`, `worker_verified=true`.
- Settings: `LINQ_AMQP_HOST` enables remote mode; `LINQ_TPU_PROGRAM`,
  `LINQ_TPU_CACHE_DIR`; `pika`/`protobuf` are the `linq` extra. Wire format
  is vendored in `backend/app/inference/tpu_wire/` (see `THIRD_PARTY.md`).
- Scenarios 02–08 are not implemented and must not be claimed.

## Frozen / do not do

- Do not commit `.ovpn`, VPN/Jupyter credentials, tokens, private video.
- Do not claim hardware verification without vendor telemetry.
- Do not turn "walking" into "running" by widening thresholds without data.
- Do not rewrite `main` again without explicit owner confirmation.

## Commands

```bash
cd backend && python -m ruff check . && python -m pytest -q
cd frontend && npm run test && npm run build
python run.py            # one-command local start (installs vision+linq extras)
# real TPU: export INFERENCE_PROVIDER=linq LINQ_AMQP_HOST=10.0.114.30 ...
```

## Key docs

- `README_HACKATHON.md` — quickstart and demo.
- `docs/LINQ_CASE_TRACEABILITY.md` — requirement → evidence matrix.
- `docs/LINQ_HARDWARE_REPORT.md` — real TPU evidence, parity and timings.
- `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION_CHECKLIST.md`.
