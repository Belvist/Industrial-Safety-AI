# LinQ case traceability

Status date: 24 October 2026. Source of truth for the case: `КЕЙС_LinQ.pdf`
(slides 4, 13, 15, 18, 19) and the two LinQ procedure PDFs. This file maps
every claim to code, tests and an honest verification level. No requirement
is marked PASS on the basis of a plan or a placeholder. Hardware evidence is
separate from adapter tests.

Verification levels:

- **PASS** — implemented and covered by an automated test in this repository.
- **PARTIAL** — implemented, but only part of the real-world contract is proven.
- **UNTESTED** — code exists, no reproducible evidence yet.
- **FAIL** — known gap; not implemented.
- **DONE** — obtained/executed manually; the exact artifact or command is recorded, but it is not an automated CI test.
- **BLOCKED** — cannot be verified without external access (VPN/SDK/device).

## Critical acceptance update (24 October)

The current real-video matrix and limitations are in
[`docs/acceptance/RESULTS.md`](../../docs/acceptance/RESULTS.md). That report distinguishes code
coverage from actual inference and hardware proof. CPU person/Radar are now
independent of PPE, real industrial A/B clips show predicted WARNING before
entry and an actual saved incident. The new pipeline has **not** been verified
on LinQ; scenario C and five TPU-complete frames/second remain BLOCKED.

## Slide 13 — mandatory requirements

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Accept video streams or image series | PARTIAL | `backend/app/main.py` (`/cameras/{id}/start`, `/frame`, `/video`), `backend/app/services/runtime.py` source loop. File/browser paths are exercised by tests; RTSP/physical camera is not run in CI. |
| 2 | Neural detection of safety violations | PARTIAL | PPE and danger zone: `backend/app/services/engine.py`, `backend/app/inference/local_onnx.py`, tests `backend/tests/test_runtime.py`, `test_engine.py`. Behaviour scenario 01 (running): `backend/app/services/behavior.py`, tests `backend/tests/test_behavior.py` (unit + synthetic pipeline). No real-video event recall yet. |
| 3 | Alert with frame, camera, time, type | PASS | Incident record + snapshot in `runtime.process`, `storage.create_incident`; API `/api/v1/incidents/{id}/snapshot`; tested by `test_runtime.py` and `test_notifications.py` (transport mocked). Live Telegram delivery is UNTESTED. |
| 4 | One alert per continuous violation | PASS | `TemporalEngine` + `Storage.finish_incident`; tests `test_state_machine.py`, `test_runtime.py`, `test_two_worker_scenario.py`. |
| 5 | Journal filters: type / camera / zone / time | PASS | `Storage.incidents`, `GET /api/v1/incidents` (`type`, `camera_id`, `zone`, `from_time`, `to_time`, `status`, pagination); tests `backend/tests/test_incident_filters.py`. UI controls in `frontend/src/App.tsx` + `hooks/useSafety.ts`. |
| 6 | Inference on LinQ TPU with measured speed | PARTIAL | Real remote TPU program executed over RabbitMQ: `ppe_nosig_p999.tpu`, worker `status_code=0`, `output0=(1,18,8400)`. Adapter `protocol=linq-rabbitmq-tpu-v1`, `worker_verified=true`. Parity: val 20/20 detections, test 11/14 matched. Speed: worker p50 55.6–75.4 ms, end-to-end p50 1.36–2.42 s. No independent hardware counters; `hardware_verified=false`. See `docs/LINQ_HARDWARE_REPORT.md`. |

## Slide 4 — behavioural scenarios

| # | Scenario | Status | Notes |
|---|---|---|---|
| 01 | Running / sharp movement | PARTIAL | Implemented from Worker trajectories (speed normalised by person height, straightness guard, temporal confirmation). Unit + synthetic pipeline tests pass. Real-video event recall / false alarms per hour: UNTESTED. |
| 02 | Dangerous approach to moving machinery | FAIL | A polygon zone exists, but it does not detect moving machinery. Not claimed as scenario 02. |
| 03 | Phone use | FAIL | Only a legacy `smoking` flag exists for another class; no phone model. |
| 04 | Smoking / open fire | FAIL | `smoking` is a config flag consumed only when the loaded model emits a `smoking` class; no verified smoking model. Do not present it as working. |
| 05 | Crossing barriers / under conveyor or load | PARTIAL | Named polygon zones, foot-point crossing and `zone_name` on incidents are implemented and tested. There is no conveyor/load detector, so only the drawn-zone part is covered. |
| 06 | Work at height without harness | FAIL | No detector. |
| 07 | Fall / long immobility | FAIL | Not implemented. `TemporalEngine`/tracking exist, but there is no Fall/Idle/Sit split; a bbox ratio alone is not evidence. |
| 08 | Unsafe load carrying (optional) | FAIL | Not implemented. |

## Desired capabilities (slide 13)

| Capability | Status | Evidence |
|---|---|---|
| Custom danger zones | PASS | Polygon editor + `danger_zone` rule; `test_contracts.py`, UI. |
| Operator confirm/reject | PASS | `POST /api/v1/incidents/{id}/feedback`; tested in `test_runtime.py`. |
| Statistics by type/camera/time | PARTIAL | `Storage.stats` covers total/active/today/false-positive; Camera/type/zone/day counts and recorded WARNING/CRITICAL transitions now implemented in `risk_statistics`, operator-only API/UI, with scope tests and actual video ledger. Field accuracy is not validated. |
| Multiple concurrent streams | PARTIAL | Multi-camera UI and per-camera workers; two concurrent real-file streams measured for 60 seconds on 2 CPU cores; physical cameras and realtime RTSP capacity remain untested. |
| Scalability | UNTESTED | Architecture is per-camera threads; two real-file streams measured at about 2.96 FPS per camera; no claim of 5 FPS per live camera or wider scalability. |

## Hardware / vendor boundary

| Item | Status | Detail |
|---|---|---|
| LinQ SDK `tpu_framework.pdf` | DONE | Obtained; used for the compile/quantize pipeline. |
| Python API example `usage_example_python_api.zip` | DONE | Obtained; the RabbitMQ client follows its `PyTask`/headers contract. |
| VPN to `hackathon.lin-q.ru` | DONE | OpenVPN tunnel established; credentials/profile remain outside Git. |
| Real TPU inference + p50/p95/FPS | PARTIAL | Worker telemetry (`status_code`, `inference_time`) and wall timings measured; no vendor hardware counters. CPU/CoreML numbers are not reported as LinQ. |

## What this repository does **not** claim

- That the current model detects all eight behavioural scenarios.
- That `/v1/detect` is the organizers' official API.
- That CPU or CoreML execution is LinQ accelerator execution.
- That PPE mAP is event-level accuracy on a real plant.
