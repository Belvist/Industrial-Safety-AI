# LinQ 5 FPS: implementation, measurements and limits

**Status: implementation and local validation completed; real LinQ 5 FPS acceptance is BLOCKED.**

Date: 2026-10-23. Base: `hackathon/linq-final-2026` at
`8f581d6dfbd64b9ea929389b7d71a5f771a7bd82`.
The managed executor has no configured VPN, LinQ endpoint, outbound identity or
credential binding. No remote worker request was sent by this work. The previous
hardware report is historical evidence, not a measurement of this implementation.

## Baseline versus after

| Measurement | Historical baseline | New real-worker measurement |
|---|---:|---:|
| Remote wall throughput | about 0.4–1.0 FPS | unavailable: access blocked |
| Remote wall latency p50 / p95 | 984.8 / 1258.6 ms (14 images) | unavailable |
| Worker inference p50 / p95 | 73.7 / 80.2 ms | unavailable |
| Analytics FPS, 60 s with TPU + CPU person | not measured independently | unavailable |
| Dropped/stale frames and maximum data age, hardware run | not measured | unavailable |
| In-flight policy | serialized lock; one request | configurable 1–12, **default 1** |
| CPU person versus TPU PPE | sequential | parallel, same source-frame object |

Baseline source: [LINQ_HARDWARE_REPORT.md](LINQ_HARDWARE_REPORT.md).
The 314 ms publish and 683 ms wait measurements come from a separate older run;
worker inference is inside the response wait, so it is not added to that wait.
No conclusion about capacity scaling follows from worker-only inference time.

**Actual TPU FPS and analytics FPS of the new remote deployment remain unknown.
Five FPS has not been demonstrated, and this performance task is not closed.**

## Architecture implemented

- `AmqpPipeline` owns Pika connection/channel in a single I/O thread. Real Pika
  uses `basic_consume` and `process_data_events`, avoiding a `basic_get` broker
  round trip for every result. The polling path exists for older controlled peers.
- UUID message IDs and a pending-Future map correlate out-of-order replies. Queues
  and admission are bounded. The reply queue is exclusive, auto-deleted and expires.
  Messages have broker TTL. Timeouts remove pending entries; stale/unknown IDs do
  not update any frame. Connection failure rejects outstanding work; a new request
  reconnects without replaying uncertain publishes. Shutdown rejects pending work.
- Cancellation is local and TTL-based. The vendor protocol has no operation to
  cancel an already running TPU task or inspect the shared broker queue depth.
- CPU person inference starts independently of remote PPE and joins only the same
  frame. Configured remote inference has no hidden local PPE fallback. Model size
  remains 640×640; confidence thresholds, model programs and calibration are unchanged.
- OpenCV capture owns its own thread. RTSP continuously decodes and retains at most
  one unscheduled latest frame. File input uses bounded backpressure and media time;
  this is explicitly labelled file analysis, not realtime playback.
- Source inference has a bounded worker pool; analytics commits Futures in source
  order. State locks are released during inference. Session and rules generations
  invalidate late results after stop/restart or geometry changes.
- Browser capture uses a separate timer and the server's `frame_capacity`. HTTP
  requests carry increasing frame IDs and capture elapsed time. Commits are ordered;
  reversed/duplicate source IDs are discarded. Every request is aborted on client stop.
- JPEG loads retain the exact result payload associated with those pixels. New
  polling results cannot put new detection boxes on an old image.

## Metrics and units

Provider and per-camera `metrics` contain bounded samples with p50/p95/p99/max,
plus independent rolling 60-second event counts and actual elapsed windows.
Current stages cover capture/decode, preprocessing, tensor serialization/size,
publish, complete RPC, reply deserialization, reported worker inference,
postprocess/NMS, CPU person, total inference, tracking/assignment, radar, JPEG,
ordered analytics/end-to-end and evidence encoding. The API also measures response
header preparation time. Samples are bounded to the latest 4096 per stage.

- `capture`: decoded frames (server sources), received/admitted camera frames (browser).
- `tpu`: successful per-frame remote PPE results, recorded before the CPU join.
- `analytics`: ordered frames actually committed to tracking, rules and radar.
- UI FPS: unique images loaded by the current viewport over its own elapsed window.
- `sampled_out`, `dropped`, `stale`: source sampling, latest-frame replacement and
  superseded results. These are different operations, not one invented loss number.
- `data_age_ms`: capture/admission to commit plus time since that result. Browser
  network transfer before admission is not included; capture elapsed time preserves
  relative source timing, not calibrated absolute client/server clock alignment.

Broker queue latency is unavailable without synchronized vendor telemetry.
A worker-reported inference duration is never treated as end-to-end latency.
Browser diagnostics also expose bounded p50/p95/p99 samples for complete API
round trips and authenticated frame fetch-to-display time; no URLs or credentials
are recorded in these timings. API polling is up to 10 Hz and is never reported
as inference FPS. Browser/E2E
validation measures user flows, not remote TPU performance.

## Controlled transport test — explicitly NOT hardware

A reproducible, synthetic peer exercises the actual transport, protobuf format,
correlation map, CPU/TPU synchronization, runtime ordering and failure handling.
Its configured worker timing is a fixture; it is not vendor telemetry. Small
synthetic input tensors are used, so its speed cannot predict full-frame VPN speed.

[Machine-readable 60-second result](benchmarks/controlled-amqp-60s.json):

| Controlled transport result | Value |
|---|---:|
| Duration | 60.089 s |
| Successful replies | 1864 |
| Controlled replies/sec | 31.021 |
| RPC p50 / p95 / p99 | 27.850 / 29.248 / 31.259 ms |
| Complete synthetic send p50 / p95 / p99 | 28.039 / 29.477 / 31.441 ms |
| In-flight / observed peak | 4 / 4 |
| Failures / pending at end | 0 / 0 |
| Real worker verified | **false** |

Separate bounded tests cover limits 1, 2, 4, 6, 8 and 12; reversed replies;
timeout followed by a late reply; disconnect/reconnect; concurrent CPU/TPU
execution; out-of-order source inference with ordered commits; concurrent browser
frames; stale IDs and stop/restart isolation. A synthetic two-worker zone scenario
creates two independent active incidents. An uninterrupted tracked worker keeps one.
Unknown identity after actual tracking loss is not globally merged by zone: this
prevents suppressing a second worker's incident. Real-world ID-switch rates and
labelled event precision/recall remain unmeasured.

Reproduce the controlled tests:

```bash
RUN_CONTROLLED_SOAK=1 CONTROLLED_SOAK_REPORT=/tmp/controlled-amqp.json \
  python -m pytest backend/tests/test_concurrent_pipeline.py -q
```

## Real-worker benchmark procedure

Connect the executor through the supported VPN/network configuration and configure
LinQ settings privately. Load the CPU person model and use actual multi-worker
video. Obtain the permitted concurrency from the organizers before running.
The harness requires `--approved-max`; levels exceeding it are never attempted.
Do not copy credentials, VPN files or private video into Git.

```bash
python backend/scripts/benchmark_linq.py \
  --video /path/to/real-workers.mp4 \
  --approved-max 4 --levels 1,2,4,6,8,12 --seconds 60 \
  --output /tmp/linq-benchmark.json
```

Here `4` is an example, **not an asserted quota**. The script runs permitted levels
sequentially after warmup, stops escalating after observed failures, and requires
remote PPE plus a loaded CPU person detector. It records real wall duration,
TPU replies, ordered analytics frames, CPU time, stage percentiles, pending work
and peak in-flight. Video is looped in saturated file-throughput mode, explicitly
labelled; this is not a separate realtime RTSP, event-recall or UI benchmark.

The acceptance flag requires >=300 ordered frames, >=5 analytics FPS and >=5
TPU FPS over >=60 real seconds with no recorded failures. Drain work after the
window does not count as completed analytics in that window. UI FPS and broker
queue latency remain null in this CLI harness and require browser/vendor measurement.
A blocked-precondition check is recorded in
[hardware-blocked.json](benchmarks/hardware-blocked.json); it made no hardware call.

After a stable level is measured, run live RTSP, labelled multi-worker footage,
TPU failure/recovery and browser tests in that deployment. Capture the maximum
frame age, dropped/stale counts and event evidence. If replies/sec does not scale,
keep the measured safe limit; investigate broker/VPN and worker scheduling with the
organizers. Do not increase ROI workload or claim batch support without a verified
vendor contract. A faster CPU-only radar remains a distinct, honestly labelled
fallback plan, not proof of five TPU PPE inferences per second.

## Evidence and local validation

Replay includes the event frame before encoding is scheduled, remains scoped to
its source session and uses source timestamps. Slow analytics no longer produces
an accelerated 5 FPS clip. H.264 encoding/MP4 frame count is checked with ffprobe;
OpenCV falls back to an installed FFmpeg/libx264 encoder when needed.

Local backend, frontend tests/build, QR phone flow and operator desktop/mobile
Chromium E2E pass; E2E uses an emulated webcam and real local ONNX weights, not
remote TPU or a physical phone. Dependency audits report zero known vulnerabilities
for the npm tree and the audited project Python dependency closure after updates.
See [the focused security review](SECURITY_PIPELINE_REVIEW.md).

GitHub Actions run [40031674695](https://github.com/judogaaa/Industrial-Safety-AI/actions/runs/40031674695)
failed before any jobs started. The check annotation states: "recent account
payments have failed or your spending limit needs to be increased". Logs do not
exist because jobs never ran. Local test success does not resolve account billing
or establish a green hosted CI run. The existing `demo-stable-c809154` tag is unchanged.

Final local checks: 244 Python tests passed, 4 skipped; 32 frontend tests passed;
TypeScript/Vite production build passed. The controlled 60-second suite passed
16 tests when explicitly enabled. Both browser E2E scripts passed in Chromium.
