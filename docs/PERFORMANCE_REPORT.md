# Performance report (sprint `feature/security-model-quality`)

Mac, venv python3.12, bus.jpg 1080×810, dual-model unless noted. Live camera
numbers need footage; below are headless ceilings, not camera FPS.

## Throughput (full `Runtime.process`, JPEG transport included)

| Provider | ms/frame | FPS ceiling | 5/10/15 sustain? |
|---|---|---|---|
| CPU | 693 | 1.4 | no (matches observed ~1.8) |
| CoreML | 28 | 35.9 | yes, ~2x headroom at 15 |

Pure inference: CPU 437–448 ms / CoreML 16 ms. `TARGET_FPS=5` unchanged:
it paces only video/RTSP workers (browser is client-paced); raising needs
live validation under network + SQLite + Telegram load.

## Latency budget (CoreML, per frame)

Inference ~16 ms, JPEG encode/decode + tracking + rules + SQLite ~12 ms.
Browser transport round-trip not included — measure on live camera.

## Memory/CPU/stability

Not measured under long runs yet — required before pilot claims.
`session.frames` bounded (4), notifier queue bounded (100), snapshots
retained per `SNAPSHOT_RETENTION_DAYS` (default keep-forever, local demo).
