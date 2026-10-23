# Demo script — Industrial Safety AI × LinQ

Target length: 2–3 minutes live. The 90-second version is marked **[90s]**.

## Before the demo

1. Start the application: `python run.py` (or the one-command path in
   `README_HACKATHON.md`).
2. Open the operator UI and pick a camera.
3. In **Камеры и правила**, enable **Контроль бега по цеху**.
   Optional: enable helmet/vest and draw a danger zone.
4. Open `docs/LINQ_HARDWARE_REPORT.md` and say plainly whether LinQ is
   verified or blocked. Never show CPU/CoreML as TPU.

## Script

1. **[90s] Show normal work.** Start the browser camera or a video file.
   Point at the calm-state chip: workers are tracked, no incident is created.
2. **[90s] Trigger scenario 01.** Have a person walk, then run across the
   frame for at least ~1.5 s. The alert banner moves from “ПРОВЕРКА” to
   “НАРУШЕНИЕ: Бег по цеху” after temporal confirmation.
3. Show that exactly **one** incident appears. Open it: snapshot, camera,
   timestamp, type, confidence, duration.
4. Keep the person still/walking. Let the rule clear; the incident becomes
   **Завершён** (`violation_cleared`).
5. Open **Журнал инцидентов**. Filter by type `Бег по цеху`, by camera, and
   by status. This is server-side filtering (case slide 13, item 5).
6. Stop the camera mid-incident to show **Прерван** (`monitoring_stopped`),
   not a false “resolved”.
7. Optional: show PPE/zone rules as additional functions, not as the main
   behavioural case.
8. Hardware slide: state LinQ status honestly (verified / blocked) and the
   exact next step from `docs/LINQ_HARDWARE_REPORT.md`.

## What to say about LinQ

- The application is provider-agnostic: `Detection` → tracking → rules →
  TemporalEngine → incidents.
- The remote adapter speaks the organizers' RabbitMQ protocol and runs the
  compiled `ppe_nosig_p999.tpu` program. `system/info` shows `protocol`,
  `program`, `worker_verified`, `worker_pid`, `worker_inference_ms` and
  `readiness`.
- Say the honest number: worker inference p50 ~55–75 ms, but end-to-end
  p50 ~1.4–2.4 s because of the VPN/RabbitMQ round-trip. Do not present
  worker-only FPS as end-to-end.
- Do not claim independent hardware counters: the protocol exposes only the
  worker's `status_code` and `inference_time`. `hardware_verified` stays
  false; `worker_verified` is true.
- Parity is measured: val 20/20 detections; test 11/14 (two low-confidence
  `no_helmet` misses, one extra). State it if asked.

## Failure fallbacks

- Running does not fire: check that the rule is enabled, the person crosses
  enough of the frame, and `confirm_seconds` has elapsed. Show the rule state
  (PENDING) instead of pretending.
- Camera unavailable: use a recorded MP4 upload; label it as a recording.
- No network: run fully local and say so; do not claim LinQ.
