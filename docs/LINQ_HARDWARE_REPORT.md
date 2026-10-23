# LinQ hardware report

Status: **REAL REMOTE TPU INFERENCE VERIFIED — PARTIAL for the case.**

A compiled `.tpu` program was executed by the organizers' TPU worker over
RabbitMQ. The worker returned `status_code = 0` and its own
`inference_time`; detection parity is measured on fixed images. This is
**not** a claim of independent hardware counters: the protocol exposes only
the worker's report, and the adapter therefore keeps
`hardware_verified = false` while `worker_verified = true`.

No CPU/CoreML number is presented as LinQ.

## Evidence chain

| Step | Evidence |
|---|---|
| VPN to `hackathon.lin-q.ru` | OpenVPN 2.7.8, tunnel `utun5`, route `10.0.114.0/24` |
| Jupyter / shared cache | `/auto/cache/hackathon93154/` writable from the kernel |
| Program compiled | `ppe_nosig_p999_b1_o1_h1m.tpu`, 22 055 100 bytes |
| Program SHA-256 | `ac9fc87355edf9fb5552537c02f4b1a584370859dc6a7b96a4e6f1ec4b22eabd` |
| Program on worker | `file:///auto/cache/hackathon93154/ppe_nosig_p999.tpu?raw=False` |
| Source model SHA-256 | `b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246` (`models/ppe.onnx`) |
| Opset-18 export SHA-256 | `0bb2c8ab5ca7ab967538e855c2bfc6de4e0e07bf11646d0afcf512266fa620b5` |
| Worker reply | `worker_pid = 7` / `8`, `status_code = 0`, `output0 = (1, 18, 8400) float32` |
| Adapter | `protocol = linq-rabbitmq-tpu-v1`, `worker_verified = true` |

The program is uploaded through the Jupyter contents API and copied to
`/auto/cache/hackathon93154/ppe_nosig_p999.tpu`. The uploaded copy was
SHA-256 checked after the copy.

## Model pipeline (reproducible)

1. `models/ppe.onnx` (opset 20) converted to opset 18 with
   `onnx.version_converter`; tensor names preserved: `images`, `output0`.
2. The final `/model.22/Sigmoid` node is removed **only** at the output.
   The program emits class **logits**; the host applies sigmoid in
   `LinqProvider._decode`.
3. Calibration: 24 real frames from `ppe_public_v3/val/images`, same
   letterbox + BGR + `/255` preprocessing as `local_onnx.py`.
4. `preliminary_model_conditioning` → `RegularModel` →
   `calibrate_torch(activation_percentile=99.9)` → `quantize` →
   `as_torch_graph(batch_size=1)` → `export_to_pt2`.
5. `Network.from_quantized_model` → `compiler.compile_(TPU_H1M, net, O1)`.
6. Compilation ran in Docker `python:3.12.11-slim` with
   `--platform linux/amd64`; the vendor wheels are linux_x86_64 / cp312.

### Why the final Sigmoid was removed

With the original graph the quantizer fused the final conv + bias + Sigmoid
into a LUT and mis-scaled the class branch. Measured on `coco_000135.jpg`:

- fp32 local top score `0.873` → quantized TPU top score `0.1237`;
- the LUT input logits were all negative (`-83.25 … -8.51`), although the
  fp32 winning logit was positive;
- the quantized final-conv bias was stored as `-0.43` while the source bias
  was `-6.41`, and the weights were not on the same scale.

Removing the final Sigmoid and calibrating at the 99.9th percentile restored
the winning class and logit: TPU logit `2.0` vs fp32 `1.93` on the same
frame. The host-side sigmoid is explicit and covered by tests.

## Parity (fixed images)

Val set, 8 images / 20 detections:

| Metric | Result |
|---|---|
| Detections matched | 20 / 20 |
| Missing | 0 |
| Extra | 0 |
| Mean IoU range | 0.827 – 0.936 |
| Max confidence delta | 0.148 |

Test set, 6 images:

| Metric | Result |
|---|---|
| Local detections | 14 |
| TPU detections | 12 |
| Matched (IoU ≥ 0.5) | 11 |
| Missing | 2 |
| Extra | 1 |

The miss is a low-confidence negative class (`no_helmet`) and one weak
helmet; it is a real gap, not hidden. The test set is small and is not an
event-level accuracy claim.

## Measured speed

Environment: Mac client → OpenVPN → RabbitMQ `10.0.114.30:5672` → shared
TPU worker. The broker round-trip dominates; the worker time excludes queue
and network transfer.

14-frame run (8 val + 6 test images) with `LinqProvider`:

| Metric | p50 | p95 |
|---|---|---|
| Total wall (preprocess + RabbitMQ + decode) | 984.8 ms | 1258.6 ms |
| Worker `inference_time` (reported by TPU worker) | 73.7 ms | 80.2 ms |

Earlier 20-iteration stage breakdown on `coco_000124.jpg`:

| Stage | p50 | p95 |
|---|---|---|
| Letterbox + preprocess | 1.15 ms | 3.69 ms |
| Publish to broker | 314.1 ms | 429.8 ms |
| Wait for reply | 683.3 ms | 1006.4 ms |
| Host decode (sigmoid + NMS) | 1.66 ms | 3.52 ms |
| Worker `inference_time` | 55.63 ms | 86.27 ms |

30-iteration end-to-end run (earlier):

| Metric | p50 | p95 |
|---|---|---|
| Provider `predict` / wall | 2418.5 ms | 2715.0 ms |
| Worker `inference_time` | 75.4 ms | 81.6 ms |

The broker/VPN path is variable (wall p50 0.98–2.42 s across runs).
Wall throughput ~0.4–1.0 FPS in this setup. Worker-only throughput
~12–18 FPS (73–80 ms), but **not** end-to-end and not used as headline.

All measurements use the compiled `ppe_nosig_p999.tpu` program
(SHA-256 `ac9fc87355edf9fb5552537c02f4b1a584370859dc6a7b96a4e6f1ec4b22eabd`)
over RabbitMQ `tpu.direct`/`input` with protocol `linq-rabbitmq-tpu-v1`.

## Adapter truth

`backend/app/inference/linq.py` reports:

- `mode = "linq_remote"` when `LINQ_AMQP_HOST` is set;
- `readiness = "configured"` after load, `"reachable"` only after a broker
  connection, `"inference_verified"` only after a completed TPU reply;
- `worker_verified = true` only when a worker replied with
  `status_code = 0` and a positive `inference_time`;
- `hardware_verified = false` for the remote path: the protocol has no
  independent hardware counters;
- `protocol = "linq-rabbitmq-tpu-v1"`.

`backend/app/inference/tpu_wire/` is the organizers' wire format, vendored
verbatim (see `THIRD_PARTY.md`). `pika` and `protobuf` are the `linq` extra.

## Remaining risks

- Two low-confidence `no_helmet` detections missed on the small test set;
  more calibration/parity work is needed before claiming PPE recall.
- No automated CI test against the live TPU service; the hardware test is
  manual and depends on the VPN and the shared worker.
- Broker/VPN latency is high and variable; do not present worker-only FPS as
  end-to-end.
- The VPN profile, Jupyter password and any tokens must never enter Git,
  archives, screenshots or slides. Rotate credentials shared in chat.

## Local (non-LinQ) reference

Local ONNX/CoreML numbers exist in `docs/PERFORMANCE_REPORT.md` and
`research/history/`. They are useful for the application loop, but they are
**not** evidence for the case's hardware requirement.

## End-to-end through LinQ (video -> TPU -> radar -> incident -> evidence)

### Neural split (honest)

The compiled PPE program has no usable `person` head (max person score ~0.002
on real frames), so worker tracking uses a **separate person detector**
(`models/person.onnx`) on CPU. PPE detection runs on the LinQ TPU. The adapter
reports this split in `system/info`:

```
"ppe_runtime": "tpu", "person_detector": "cpu"
```

Tracking, Safety Radar, TemporalEngine, storage and UI run on CPU.

### Fully automatic run (no scripted boxes)

Input: `footage_v4/yt_eval/3SuIh5zgKQQ.mp4` (854x480, 30 fps), first 60
frames. Zone: normalized rectangle x 0.80-0.99, y 0.20-0.62
("Зона техники"). No bounding boxes are injected anywhere.

Observed:

1. Each frame is sent to `ppe_nosig_p999.tpu` over RabbitMQ
   (`worker_verified=true`, worker PID 7).
2. `models/person.onnx` (CPU) returns real person detections; the tracker
   produces Track #1 with a stable ID.
3. Safety Radar reaches `CRITICAL` (foot point inside the polygon) and
   emits `zone_cross` with the explanation.
4. Radar CRITICAL is fed into TemporalEngine as `danger_zone`; it confirms
   **exactly one incident**.
5. One MP4 evidence clip is written asynchronously and served as `video/mp4`
   with `X-API-Token` auth and a path-traversal guard.

### Approach warning on a second real clip

Input: `footage_v4/yt_eval/UVh_Jv0n1ZQ.mp4`, zone x 0.10-0.315. The worker
walks toward the boundary; Radar emits `zone_approach / WARNING` with
`distance=0.015 norm` (no metres, no time-to-collision). A single-frame
boundary touch does **not** create an incident — temporal confirmation is
required.

### Limits of this evidence

- It is a real-video run, not a labelled event benchmark. Event-level
  precision/recall and false alarms/hour are not measured.
- The person detector is CPU, not TPU. Only the PPE graph is on LinQ.
- `hardware_verified=false`: no independent hardware counters.

## Honest limits after the fixes

- `hardware_verified=false`: the protocol exposes only `worker_pid`,
  `status_code` and `inference_time`, no independent hardware counters.
- Event-level precision/recall and false alarms per hour are not measured:
  the available footage has no event-level ground truth.
- Radar thresholds are normalized-image units. No metres, m/s or
  time-to-collision are shown without camera calibration.
- Scenarios 02-04, 06 are BLOCKED; 07 is FAIL; 08 is NOT_IMPLEMENTED.
