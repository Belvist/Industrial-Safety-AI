# Architecture

## Goal

Keep camera acquisition, inference hardware, safety logic and UI independent.

```text
Browser webcam / USB ─┐
RTSP / IP camera ─────┼─> Frame ─> InferenceProvider ─> Detection[]
MP4 / test video ─────┘                    │
                              Local ONNX now / LinQ later
                                           │
                                           v
                            Tracking -> Safety Rules
                                           │
                                           v
                         Temporal confirmation (~2 s)
                                           │
                                           v
                            Incident ACTIVE / RESOLVED
                                 │              │
                              SQLite        Telegram
                                 └──── Dashboard ────┘
```

## Invariants

1. Downstream code never consumes raw model tensors. It consumes normalized `Detection` objects.
2. LinQ is an adapter, not a rewrite. Switching inference hardware must not change rules, incidents, camera management, Telegram, or the dashboard.
3. Browser cameras use `getUserMedia` for portability across macOS, Windows and USB webcams.
4. RTSP, HTTP streams and video files are backend-owned frame sources.
5. Danger-zone polygons are stored in normalized `0..1` coordinates, never fixed pixels.
6. A single noisy frame does not create an incident. Violations require temporal confirmation.
7. One persistent violation maps to one incident; no per-frame alert spam.
8. Model files and captured evidence are runtime assets and are not committed to Git.

## MVP contract

```text
camera
 -> person/PPE detection
 -> stable track
 -> no-helmet OR danger-zone rule
 -> 2-second confirmation
 -> one ACTIVE incident
 -> evidence snapshot
 -> Telegram notification
 -> RESOLVED after the violation clears
```


## Local MVP implementation (v0.2)

FastAPI owns a single Runtime and a CameraSession per configured camera. SQLite persists camera rules, incident lifecycle, evidence filenames, notification status and operator verdicts. Model files and data paths resolve from the repository root. Run one backend worker bound to 127.0.0.1.

The ONNX adapter composes a dedicated YOLOv8n person detector and YOLOv8m PPE detector; everything downstream still consumes Detection objects with normalized xyxy boxes. Input frames are uint8 BGR. Browser cameras send sequential JPEG requests; video/RTSP sources use background workers. A lock prevents concurrent processing within a camera; ONNX sessions serialize their own inference calls. Slow video inference skips frames according to the source timeline.

Tracking uses greedy one-to-one IoU matching, not persistent identity recognition. PPE is associated to one worker by anatomical region and distance. Missing PPE requires explicit negative detection; absent detections mean unknown. Danger-zone checks use the bottom-center point of the person's box.

Temporal confirmation resets on a missed violation; a gap >3 seconds interrupts confirmed incidents and resets pending observations. Resolution requires the configured absence duration. Stopping/restarting a source, changing rules, or restarting the server interrupts active incidents. INTERRUPTED is an additional lifecycle state and must not be presented as proof that the violation cleared.

Evidence is an annotated JPEG. Telegram uses a bounded background queue; delivery failure is recorded and does not stop inference. The UI uses polling, not WebSocket, and FastAPI serves the built React app. Local-origin checks protect mutations; this is a single-operator local service without remote authentication.


## LinQ track (hackathon 2026, case "LinQ / ХайТэк")

MVP scope per project description §11: `no_helmet`, `no_vest`, `danger_zone`
(on by default). `no_mask` (respirator) and `smoking` are implemented as
post-MVP extensions, off by default per camera, enabled once the model
provides the classes — no new infrastructure, just rule flags.

`LinqProvider` (`backend/app/inference/linq.py`) is the only file that knows
about accelerator wire details, with three modes:

- `linq_remote` — `LINQ_ENDPOINT` (+ `LINQ_API_KEY`) set: JPEG frames POSTed to
  `{endpoint}/v1/detect`, response normalized to `Detection`. Used at the event.
- `linq_local` — ONNX Runtime with `LinQExecutionProvider` when installed.
- dev fallback — same ONNX graph on CPU while access is not issued.

Switching `INFERENCE_PROVIDER=local_onnx|linq` changes no rules, incidents,
storage, Telegram or dashboard code. `GET /api/v1/system/info` exposes the
active mode under `inference`.


## Explicit state boundaries

See [STATE_MACHINES.md](STATE_MACHINES.md) for transition diagrams, the client/backend contract and all interruption/error paths. Camera lifecycle transitions are centralized, commands are serialized, frame requests are scoped to a session, and HTTP response schemas validate normalized model output. The frontend owns an independent capture state machine and command machine; it renders backend safety decisions. Every detected object has a class/confidence label, confidence card and a versioned frame source, so the overlay matches the actual inferred frame.
