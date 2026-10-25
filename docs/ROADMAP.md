# Roadmap

## M0 — Repository bootstrap
- Cross-platform backend skeleton
- Stable inference contract
- CI on Linux, Windows and macOS

## M1 — Camera layer
- Browser camera via getUserMedia
- USB/built-in cameras on Windows and macOS
- MP4 source
- RTSP source

## M2 — PPE inference
- Select an open-source PPE checkpoint with compatible licensing
- Export/verify ONNX
- Normalize person / helmet / no_helmet / vest / no_vest detections
- Benchmark latency and FPS

## M3 — Safety logic
- Tracking
- Helmet/vest association
- Normalized polygon zones
- Temporal confirmation
- Incident deduplication and ACTIVE/RESOLVED lifecycle

## M4 — Product layer
- SQLite persistence
- Snapshot evidence
- Telegram alerts
- React monitoring/dashboard UI
- False-positive / confirmed feedback

## M5 — LinQ
- Implement LinqProvider against organizer SDK/API
- Keep normalized Detection contract unchanged
- Measure real LinQ latency/FPS


## v0.2 implementation status

Implemented: browser/USB camera selection, uploaded video and RTSP workers; verified ONNX downloads; person/PPE inference; IoU tracking and PPE association; normalized zones; confirmation/deduplication; ACTIVE/RESOLVED/INTERRUPTED lifecycle; SQLite; evidence; Telegram send/status; React dashboard and feedback; local launchers.

Next: validate Windows + A4Tech and MacBook cameras on physical devices; collect industrial validation videos and benchmark PPE precision/recall; upgrade tracking for occlusion/crossing; resolve upstream model-license inconsistencies before redistribution/commercial packaging; add reconnect/retry/retention policies; implement LinQ against the organizer SDK.
