# AGENTS.md — project memory

## What this project is

Industrial Safety AI: FastAPI + React video monitoring for industrial safety.
Pipeline: camera → ONNX detection → worker tracking → rules → TemporalEngine →
incidents → SQLite/snapshots/Telegram → operator UI. LinQ TPU is the required
hackathon accelerator (see `docs/LINQ_CASE_TRACEABILITY.md`).

## Current state (23 Oct 2026)

- Git history consolidated into single `Initial commit`; `main` force-pushed.
  Working branch: `hackathon/linq-final-2026` (pushed to origin).
- Scenario 01 (running) implemented in `backend/app/services/behavior.py`,
  integrated in `runtime.process`, configurable per camera (`running_enabled`,
  `running_speed`, `running_min_seconds`). Unit + pipeline tests pass.
- Incident journal filters implemented server-side: `type`, `camera_id`,
  `zone`, `from_time`, `to_time`, `status`, pagination. SQLite migration adds
  `zone_name` and indexes.
- **Safety Radar (killer feature) implemented**: `backend/app/services/safety_radar.py`
  — trajectory analysis (foot-point, speed), zone approach (WARNING), zone cross (CRITICAL),
  risk levels NORMAL/WARNING/CRITICAL/UNKNOWN, Evidence Replay events with trajectory,
  Risk Analytics event storage. Integrated in `runtime.process`, exposed via `session.latest.radar`.
- LinQ adapter real: RabbitMQ `tpu.direct`/`input`, program
  `ppe_nosig_p999.tpu` in `/auto/cache/hackathon93154`, protocol
  `linq-rabbitmq-tpu-v1`. Real worker replies: `status_code=0`,
  `output0=(1,18,8400)`, `worker_verified=true`.
- Program SHA-256: `ac9fc87355edf9fb5552537c02f4b1a584370859dc6a7b96a4e6f1ec4b22eabd`.
- Critical compile workaround: final `/model.22/Sigmoid` removed before
  quantization, calibration `activation_percentile=99.9`. Without this
  final conv+bias+LUT mis-scales class scores (0.87 -> 0.12).
  Host-side sigmoid in `LinqProvider._decode`.
- Parity: val 20/20 detections; test 11/14 (2 low-confidence `no_helmet`
  misses, 1 extra). End-to-end wall p50 0.98–1.26 s (14 frames), worker p50 73.7 ms.
  No independent hardware counters: `hardware_verified=false`, `worker_verified=true`.
- Settings: `LINQ_AMQP_HOST` enables remote mode; `LINQ_TPU_PROGRAM`,
  `LINQ_TPU_CACHE_DIR`; `pika`/`protobuf` are `linq` extra. Wire format
  vendored in `backend/app/inference/tpu_wire/` (see `THIRD_PARTY.md`).
- **Safety Radar** separate service, works with both LinQ and local ONNX.
- Scenarios 02–08: 02–04 BLOCKED (no models), 05 IMPLEMENTED (zones),
  06 BLOCKED, 07 PARTIAL (fall detection needs Fall/Idle/Sit split),
  08 NOT_IMPLEMENTED. Must not claim as implemented.
- Tests: 189 backend (ruff clean), 32 frontend (vitest), all passing.

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
