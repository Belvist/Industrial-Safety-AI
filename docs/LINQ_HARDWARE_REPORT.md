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

20-iteration stage breakdown on `coco_000124.jpg`:

| Stage | p50 | p95 |
|---|---|---|
| Letterbox + preprocess | 1.15 ms | 3.69 ms |
| Publish to broker | 314.1 ms | 429.8 ms |
| Wait for reply | 683.3 ms | 1006.4 ms |
| Host decode (sigmoid + NMS) | 1.66 ms | 3.52 ms |
| Worker `inference_time` | 55.63 ms | 86.27 ms |

30-iteration end-to-end run:

| Metric | p50 | p95 |
|---|---|---|
| Provider `predict` / wall | 2418.5 ms | 2715.0 ms |
| Worker `inference_time` | 75.4 ms | 81.6 ms |

A second 20-iteration run measured wall p50 `1362.8 ms`, p95 `1538.4 ms`;
the broker/VPN path is variable. Wall throughput is therefore ~0.4–0.7 FPS
in this setup. Worker-only throughput is ~13–18 FPS, but it is **not**
end-to-end and is not used as the headline number.

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
