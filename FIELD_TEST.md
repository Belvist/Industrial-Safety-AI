# FIELD TEST — Industrial Safety AI

Tested version: branch `fix/field-test-rc-20261008`, commit `a0048e2`.
CI: 5/5 green (backend ubuntu/macos/windows, frontend, e2e).
No merge to `main`. Not production-ready.

## Run

Mac:

```bash
git checkout fix/field-test-rc-20261008
python3.11 run.py            # setup, model download, http://127.0.0.1:8000
```

Windows (PowerShell):

```powershell
git checkout fix/field-test-rc-20261008
py -3.11 run.py
```

Phone as camera (same Wi-Fi, trusted network only):

```bash
python3.11 run.py --lan     # scan QR in "Камеры и правила", accept the cert
```

For speed on Mac, add to `.env`: `ONNX_PROVIDERS=CoreMLExecutionProvider`.

## Expected on a healthy run

- `GET /health` → ok; `system/info` → `model_ready: true`.
- Person in frame → `Worker #N` with stable ID.
- Helmet + vest on → statuses present, zero incidents.
- Helmet off ~2 s → exactly one `no_helmet` ACTIVE + snapshot.
- Helmet back on ~2 s → same incident RESOLVED (`violation_cleared`).
- Walk into drawn polygon → CRITICAL `danger_zone` ACTIVE; walk out → RESOLVED.
- Camera lost mid-violation → INTERRUPTED (`observation_lost`/`frame_gap`), never silent RESOLVED.

## Verified this release (2026-10-08, by agent)

| Check | Result |
|---|---|
| CI backend ubuntu/macos/windows + frontend + e2e | PASS |
| Weights present, SHA recorded, 7 classes supported (no smoking in weights) | PASS |
| bus.jpg real inference: 3 persons, weak no_helmet/no_vest visible, 0 incidents | PASS |
| Scripted API scenario 8/8 (ACTIVE → snapshot/SQLite → RESOLVED → INTERRUPTED + ended_at) | PASS |
| MP4 upload → processing → ended, no crash | PASS |
| Polygon zone PUT round-trip | PASS |
| Physical laptop/USB camera | UNTESTED |
| Physical phone pairing | UNTESTED |
| Real Telegram delivery | UNTESTED (no token; format asserted in tests) |
| RTSP live camera | UNTESTED |
| Fine-tuned model accuracy | OPEN (weights unchanged; no labeled data) |
| LinQ hardware | OPEN (adapter only) |

## 10-minute tester protocol

1. (1 min) Open the app, check health + model_ready.
2. (2 min) Stand in frame with helmet + vest → expect Normal, no incidents.
3. (2 min) Remove helmet → one incident with photo in ~2 s.
4. (1 min) Put helmet on → RESOLVED.
5. (2 min) Draw a zone, walk in/out → CRITICAL pair of events.
6. (1 min) Stop camera mid-violation → INTERRUPTED.
7. (1 min) Save for each step: short video, screenshot with detections, incident card.

If helmets misdetect: do NOT change thresholds. Record video, confidence, bbox and association outcome for the fix report.
