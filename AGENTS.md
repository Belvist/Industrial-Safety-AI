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
- Neural split: PPE on TPU; separate `models/person.onnx` on CPU (PPE model's
  person head is dead, max ~0.002). `system/info` reports
  `ppe_runtime=tpu`, `person_detector=cpu`.
- Settings: `LINQ_AMQP_HOST` enables remote mode; `LINQ_TPU_PROGRAM`,
  `LINQ_TPU_CACHE_DIR`; `pika`/`protobuf` are `linq` extra. Wire format
  vendored in `backend/app/inference/tpu_wire/` (see `THIRD_PARTY.md`).
- **Safety Radar** separate service, works with both LinQ and local ONNX.
- Scenarios 02–08: 02–04 BLOCKED (no models), 05 PARTIAL (named polygon
  zones only; no conveyor/load detector), 06 BLOCKED, 07 FAIL (no
  Fall/Idle/Sit split), 08 NOT_IMPLEMENTED. Must not claim as implemented.
- Tests: 209+ backend (ruff clean), 32 frontend (vitest), all passing.
- Fully automatic LinQ E2E (no scripted boxes): real video → TPU PPE +
  CPU person → Track → Radar CRITICAL → 1 incident → MP4 evidence.

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

## Operator workspace / bounded pipeline update (23 Oct 2026)

- Work is based on `hackathon/linq-final-2026` at `8f581d6`; `main` and the
  existing `demo-stable-c809154` tag are untouched.
- `AmqpPipeline` has a single Pika owner thread, bounded admission, UUID response
  correlation, consuming callbacks, timeouts and reconnect-on-next-request.
- CPU person and remote PPE are parallel on the same frame. Server sources and
  browser requests commit analytics in order with session/rules generation guards.
- `LINQ_MAX_IN_FLIGHT=1` remains the default. No configured VPN/LinQ access was
  available here: **real 5 FPS acceptance is still blocked**, not demonstrated.
- Separate capture/TPU/analytics/UI FPS, stage percentiles and age/drop metrics;
  no UI polling speed is used as inference FPS.
- Desktop sidebar/mobile bottom navigation; distinct overview; timestamp-safe
  evidence and frame payloads. H.264 fallback uses an installed FFmpeg/libx264.
- Security fixes and dependency audit: see `docs/SECURITY_PIPELINE_REVIEW.md`.
- Reproducible real/controlled procedures: `docs/LINQ_5FPS_BENCHMARK.md`.
- Hosted CI at `8f581d6` never started jobs due to GitHub billing/spending limit;
  do not describe it as an observed code failure or as green CI.

## Monochrome operator UX (24 Oct 2026)

- User explicitly replaced the previous purple style with Cursor/OpenCode-like
  monochrome chrome. Default dark; neutral light toggle. Do not restore purple.
- `frontend/src/workspace/` owns the new product flow; `WorkspaceShell` contains
  activity navigation, camera explorer (Ctrl/Meta B), editor header and status bar.
- One focused observation view; optional People/Radar/Events inspector. Keep source
  connection and zone drawing in dialogs, settings separate, evidence in a drawer.
- `feedback` filters the review queue independently of physical incident status.
  Operator-only CSV export uses one SQLite read snapshot, all matching rows up to
  10,000, and spreadsheet formula protection.
- Browser tests use simulated media or explicitly labelled isolated incident
  fixtures; never describe those as real hardware, real incidents or model recall.


## Softer product identity (latest UI direction, 24 Oct 2026)

- User clarified that Cursor/OpenCode were usability references, not a request
  to copy an IDE. The activity rail, editor tabs and bottom status strip are gone.
- `WorkspaceShell` now uses one rounded sidebar with three main destinations.
  Camera shortcuts are collapsed by default. Ctrl/Meta B toggles navigation.
- Preserve generous spacing and rounded controls/cards/dialogs. Final skin is in
  `frontend/src/operator.css`; shared radius tokens live in `application.css`.
- Manrope variable WOFF2 is bundled locally with Latin/Cyrillic subsets. No
  runtime font CDN requests. Keep its OFL notice in source and distribution.

## Critical real-video acceptance (24 October 2026)

- Stop cosmetic work. CPU person/tracking/Radar are independent from PPE by default.
- IndependentPpe snapshots source frame/tracks/continuity version; no old PPE boxes
  are appended to the current scene. Occlusion/crossing invalidates PPE continuity.
- WARNING uses reliable predicted nearest boundary entry; stop/parallel cancel it.
  Actual warning-to-entry lead is distinct from predicted ETA and end-to-end delay.
- Incidents use per-worker/per-zone/per-entry keys. UNKNOWN never proves clearance.
- New SQLite observation_events back timeline and camera/zone/date counts; scopes tested.
- Replay is published atomically after encoding/transcode; temporary MPEG-4 is never public.
- Real source is Intel CC BY footage, with source/model checksums in docs/acceptance.
  Demo polygons were chosen on that footage, not independent holdout validation.
- Requested YouTube clip was proxy-blocked. No configured VPN/LinQ in this session.
  New remote TPU acceptance remains BLOCKED. See docs/acceptance/RESULTS.md.

## Bounded PPE window and finalization follow-up

- See docs/finalization-20261024/RESULTS.md. Prior f27f391 A/B/tag/assets remain intact.
- PPE_MAX_IN_FLIGHT_PER_CAMERA defaults 1; effective cap also respects LINQ_MAX_IN_FLIGHT.
- Pending replies ordered by source time, expired heads quarantined; global FIFO has N workers/8 queue slots, daemon workers with explicit shutdown.
- Session metrics are captured at admission; old work cannot inflate new session FPS.
- ONNX idle spinning disabled; Replay directly encodes JPEG to H.264. TARGET_FPS=6 is measured CPU headroom, not TPU FPS.
- Same-target 3-repeat before/after measurements, actual replay/reentry/two-zone/negative-walking cases are published. Positive running, real overlap/physical ID-switch accuracy and LinQ remain unverified.
- Provisional manual PPE evaluation is evaluation-only (never ROI input); substantial FN, no claimed model-accuracy fix; 72px guard uncalibrated.
- PPE lifecycle follow-up: `dea4b34` separates RPC deadline, source freshness, display TTL, and incident freshness; regression coverage includes delayed replies, stale source, identity loss, restart generation, timeout, and out-of-order completion. Backend suite: 279 passed, 1 skipped; Ruff clean.
- Zone episode dedupe follow-up: `aeae3c4` reuses an interrupted danger-zone incident only for the same camera/zone/Radar episode, with runtime regression for a short unknown gap.
- Zone trace audit: pre-fix source SHA `5451228a...` had 11 rows across 10 episode IDs, including track 66 duplicated under one episode after observation loss. Post-fix rerun on clean SHA `8b13d40` processed the same source for 60.00 s at 5.00 analytics FPS (300 frames): 10 zone incidents / 10 episode IDs, no repeated episode key. This supports the targeted duplicate fix only; without visual adjudication, 10 IDs do not establish 10 physical violations. Tracks 3/5/7/12/14 were simultaneous distinct track IDs but physical identity was not independently labeled.
- Full-clip speed diagnostics: 650 analytics frames at 5.00 FPS, 0 running incidents; a brief 1.913 person-heights/s peak lacked temporal confirmation and the track then disappeared. No independent interval ground truth.
- 24 Oct follow-up: `dea4b34`, `aeae3c4`, `8b13d40`, `7456dd7` were pushed without history rewrite to `judogaaa/Industrial-Safety-AI` earlier, then the full repository at `7456dd7` was published to the newly requested public `Belvist/Industrial-Safety-AI` on `main` and `fix/real-detection-accuracy`. Both target refs were confirmed at `7456dd7`; source `origin` still tracks judogaaa.
- Pexels exploratory media were downloaded outside the repo under `/tmp`, not committed. Running clips: Pexels 1322538 (5.239 s, 1920x1080, SHA-256 `6832215b033639ba3710b37a368b3c4ef210f13387c79181eba730f8e39d8d81`) and 7866029 (10.01 s, 1080x1920, SHA-256 `d12e44f30ced74788bc83c44710cdbc137608ec09ea879ff6829d7fdf7ca5def`). Pexels license page marks them Free; retained source URLs in task history. A 7866007 walking candidate (10.01 s, 1440x2560, SHA-256 `1a6d7b51f40c1a89407f499d037de6d2c8681dbb21b9173537872c00009d2677`) is from the same creator/setup, not an independent holdout. Actual isolated backend ingestion of clip 7866029 processed 60 frames at 6.02 CPU person/Radar FPS, track 1, final speed 0.132 person-heights/s, zero running incidents. A subsequent 7866007 upload processed 60 frames at 6.02 FPS, zero running incidents; final tracks had speeds 0.091 and 0.657, with one small track below the PPE observability guard. These are exploratory results only: visual interval labels and camera-motion annotation were not available, so TP/FP/FN and running-vs-walking accuracy are not established. No real positive running incident was confirmed.
- New isolated uvicorn PID 42551 used `127.0.0.1:8011`, separate `/tmp/industrial-safety-isolated` SQLite/data, confirmed model ready on CPU and no initial incidents. The old process on port 8000 was left untouched. Browser E2E was not run; no browser automation script/dependency is configured. Real LinQ unavailable in this run; TPU replies/s and PPE applied/s not measured. No zone physical-video adjudication performed because visual inspection was unavailable.
- Final checks on this code: backend Ruff clean, 279 passed / 1 skipped; frontend Vitest 39 passed; TypeScript and Vite production build pass. Do not describe the blocked real-video/browser/LinQ requirements as accepted.

## PPE accuracy investigation (24 Oct 2026)

- New isolated feature branch `fix/ppe-always-on-accuracy` started from `0905ebe`; do not modify main or stable tags. No production thresholds or user `.env` changed.
- `backend/scripts/profile_motion.py` now defaults helmet/vest ON, supports explicit `--motion-only` and `--model`, and reports model SHA/provider/rule flags and separate rates. `INCOMPLETE` means the source ended before the full measurement window.
- Four-model diagnostic sweep across `ppe.onnx`, `ppe-v3-clean.onnx`, `ppe-v4.onnx`, `ppe-v4cloud.onnx` found no credible accuracy fix. The 124-image `ppe_v3_clean` test has 30 near-duplicate cross-split pairs and two label conflicts (`datasets/ppe_v3_clean/audit_leaks.json`); do not use it as an independent holdout or to tune production thresholds. No decoder defect established; no postprocess changes made.
- Same-distribution box metrics and model checksums, plus a short isolated `ppe-v4` CPU throughput measurement, are recorded in `docs/finalization-20261024/RESULTS.md`; neither constitutes accuracy acceptance.
- Current outstanding PPE acceptance requires licensed independent A–H video with verified annotations, per-class TP/FP/FN/UNKNOWN and timing, evidence replay/journal checks, and real browser/LinQ where available. Preserve honest degraded/blocked status; do not claim recall gain from this diagnostic split.
- Dataset source audit follow-up: Atharion v1.0 archive SHA-256 `f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5`; archive metadata has 791/170/169 train/valid/test images, 3 classes Person/Hardhat/Safety Vest, 5355 boxes. `NOTICE.md` identifies 1026 images from Ultralytics Construction-PPE (AGPL-3.0) and 104 from Construction Site Safety (CC BY 4.0); source attribution is per-image. It is positive-only PPE data: lack of a Hardhat/Vest box is not a verified absence.
- Cross-check of all 169 Atharion test images against 33,343 local dataset images found 3 exact SHA matches, all in css-kaggle-v28, and 110 pHash-distance<=5 candidates (102 Ultralytics-source, 8 css-kaggle-v28). pHash is a candidate-overlap detector, not scene-level adjudication; the test split is not independent and was not evaluated.
- Three additional Pexels/Pixabay clips were downloaded for exploration, but no human visual adjudication was possible in this session; no A–H conditions or PPE states are verified. Keep them out of ground-truth metrics until labeled by a qualified reviewer. Downloaded artifacts and SHA-256: Pexels 8965526 (`pexels-8965526.mp4`, `79f6b0c2d09c3922e8aeeea8e9814c5547f84d38b6f2bd85dc0d9688e728dd0b`); Pexels 19832500 (`pexels-19832500.mp4`, `931a1db66e4578c8767b4c1133310ebdf8378749ee82b6ca5e310d3507fcfd51`); Pixabay 348897 (`pixabay-348897.mp4`, `0ce9570e79c8dee8f51f9514a01e2df1ae662791ca1add8d46441b5d8179a3d4`). Media remain outside Git; exact source pages/terms and frame-level annotations are not yet recorded in a checked-in manifest.
- No independent holdout or production accuracy improvement has been established. Continue pursuing licensed independent, scene-separated sources and explicitly verified labels; report unsupported A–H strata as BLOCKED.
- Follow-up validation: 17 manually labeled person boxes were all matched by full-frame `person.onnx` at IoU >=0.4; naive overlapping 640px tiles matched 13/17 with 45 strong detections, while two-window crops matched 17/17 but raised mean inference latency 54.3→75.2ms and strong detections 19→25. No multi-scale rescue was merged.
- Re-run on those same 17 samples with `ppe.onnx`: full-frame helmet present/missing/unknown 5/0/12 and vest 0/0/17; ROI-auto 6/1/10 and 3/0/14. Known-label violation FN counts remain 5/10 full-frame and 4/10 ROI-auto. These are single-frame diagnostic observations, not independent/event acceptance.
- Monitor warning overlay now lives on the video surface; explicit higher-specificity CSS protects overlay layout styles from generic `.p-inform.p-warning`. Monitor/focus sizing uses container width rather than viewport-height-derived width. Vitest/build pass; browser layout/geometry remains unverified.
- Corrected Atharion overlap note: confirmed 3 exact SHA matches (all CSS); 110 images met pHash<=5 and may include exact matches. Do not count 110 as additional to exact overlap.
- Point-in-time `/system/info`: port 8000 LinQ readiness `failed`, `hardware_verified=false`, 0 successful TPU results in the current 60s window, 93 failures/246 calls total, and PPE full-frame fallback `roi too small or too large`; PPE is still attempted via fallback. Port 8011 uses CPU `ppe-v4.onnx`, with zero stored incidents at check time. Neither is browser/camera acceptance.

## Detector-to-track investigation (25 Oct 2026)

- Isolated live CPU ONNX E2E ran on `127.0.0.1:8027` with separate `/tmp/industrial-frame-trace/` state; it passed the existing real-video critical-warning browser path with helmet/vest enabled. Do not use that result as LinQ or hardware acceptance. Screenshot: `/tmp/industrial-frame-trace/browser-final/critical.png`.
- A direct same-call frame trace on source frame 165 produced one strong person detection → one tracker input → Track ID 1 → one `latest`/API track. On frame 527, two separate strong person detections → two tracker inputs → two distinct IDs/API tracks. This does not reproduce the historical 3 detections → 2 tracks discrepancy.
- Historical frame 6299's original BGR pixels are unavailable; a saved/re-encoded JPEG is not identical. The earlier 3-detection result and 2-track result came from separate inference calls. Cause remains UNKNOWN; no tracker defect or threshold change is justified.
- `system/info` now exposes CPU person model readiness, safe exception class only, model SHA-256 and active providers; runtime `independent_motion_active` reflects actual CPU motion availability. Tests cover successful and failed CPU model loading and tracker strong-person cardinality.
- Added `e2e/test_real_video.py --minimum-tracks` for API plus rendered-box assertions. Minimum 1 passed; minimum 2 did not pass within the 45-second live source observation (maximum simultaneous API tracks was 1). Therefore live multi-track browser acceptance remains BLOCKED; no confidence thresholds were loosened. Verify sequence alignment and reporting behavior before relying on this optional mode further.
- Final checks: Ruff clean; backend 281 passed / 1 skipped; frontend 39 passed; TypeScript/Vite build passed. Real LinQ was not available. The isolated 8027 process must be stopped after its final use; do not touch the existing 8000/8011 processes.

## LinQ defense submission (25 Oct 2026)

- Submission README/QA/demo now point to `Belvist/Industrial-Safety-AI`, avoid presenting running as a proven positive, label prior LinQ metrics as historical, and distinguish worker inference latency from full RPC latency.
- Candidate revision `8c7f7cbd5fcc385d650e1ea70181056a4b591d97` was first pushed to `fix/ppe-always-on-accuracy` and tag `linq-defense-20261025`. A clean clone resolved to that SHA, but showed that no ONNX models are tracked and that `backend/scripts/download_model.py` verifies files rather than downloading missing registry models; manual-model startup remains unverified. That initial tag is not the final package.
- Follow-up README clarifies model provisioning and the exact behavior of the bootstrap command. The final release tag will identify the docs-corrected revision; keep the limitation explicit and do not claim clean-clone demo acceptance.
- This environment has no `LINQ_AMQP_HOST`; no new LinQ inference/latency acceptance was possible. Current-revision LinQ remains BLOCKED. Personal-account portal submission and checkpoint attendance cannot be verified here.
- Full checks on the code-bearing revision: Ruff clean, backend 281 passed / 1 skipped, frontend Vitest 39 passed, TypeScript/Vite build passed. Real-video Radar warning/critical/incidence/H.264 E2E had passed in isolated CPU mode earlier; it is not LinQ acceptance.
