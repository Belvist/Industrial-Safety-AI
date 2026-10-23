# V4-cloud single vs dual + full Runtime + video tooling (one report)

Date: 2026-10-12. Commit: 547aa71 (plus this work, SHA on push).
Model: `models/ppe-v4cloud.onnx` throughout. Production weights/settings untouched.
No training. Reproduce: commands in §5.

## 1. Honest SINGLE vs DUAL A/B (same weights/frames/thresholds/ROI off)

Script: `training/eval_single_dual.py` → `training/eval_single_dual/single_dual.json`.
PPE outputs verified byte-identical single vs dual (assert in script); only the
person source differs.

| person source | person TP/FP/FN (P/R) | ms p50/p95 (CPU) |
|---|---|---|
| dedicated person.onnx | 191/76/27 (0.715/0.876) | 434/461 |
| intrinsic V4-cloud | 188/53/30 (0.780/0.862) | 359/390 |

Worker-band assignability (emulated with detection boxes — approximation):
no_helmet single 23/31 vs dual 29/31; vest single 124/131 vs dual 117/131;
no_vest single 52/61 vs dual 57/61.

Corrections to the previous report: per-model saving is ~60–75 ms on CPU
(434→359), not 150 ms (that compared different models); fewer worker IDs
alone prove nothing — the assignability table above is the real comparison,
and it shows single LOSES 6 no_helmet GT coverages while winning vests.
Net: single is cleaner/faster, but dual covers more violation heads on stills.
Video verdict dominates: see §3 (UVh: 0 FP incidents both, 46 vs 50 IDs).

## 2. Full Runtime on Mac/CoreML (single V4-cloud, prod thresholds, ROI off)

Script: `training/bench_runtime.py`. Input: 13.5-min concat of the 6 real
clips (854x480@25). Isolated data dir + tmp DB. Transport = video file
decode, NOT a live camera (stated; capture path untested — see §4).

- Paced run (target 5 FPS): 3444 frames / 716 s = **4.81 FPS achieved**,
  per-frame end-to-end p50 **38.2 ms** / p95 **48.4 ms**, CPU mean 106% /
  max 226%, RSS mean 348 MB, providers CoreML+CPU-fallback, 2 incidents /
  1 snapshot in DB.
- Unpaced run (target 30 FPS): 20301 frames / 696 s = **29.17 FPS achieved**,
  p50 **32.5 ms** / p95 **34.5 ms**, CPU mean 114%, RSS mean 428 MB,
  2 incidents / 1 snapshot. Full results: `research/history/bench_runtime_fast/`
  (archived; superseded by `training/bench_runtime_v4/` and
  `training/bench_runtime_prod/` with total+verified counts).
- Reading: per-frame cost ~33–38 ms ⇒ compute headroom ≈ 20+ FPS; the 4.81 is
  pacing granularity, not a compute wall. Target_FPS=5 is sustainable.

## 3. Video A/B tooling for the owner

- `backend/scripts/record_clip.py`: record from camera index or cut from file
  (`--source 0 --seconds 180 --out footage_v4/raw/sceneA_cam1.mp4`), writes
  meta sidecar. Tested from file (70 frames + sidecar OK).
- `backend/scripts/label_events.py`: keyboard interval labeling
  (space=play, 1/2=type, w=worker, [=start, ]=end, u=undo, s=save, q=quit)
  straight to events.json; `--dump-frames` headless fallback tested.
  Box-less events score as worker_unverified (scorer already handles this).
- Protocol reminder: 3 scenes × 4 combos, static camera, stable worker ids,
  clean footage for FP/hour, frozen holdout scenes.

## 4. NOT TESTED (needs owner hardware/setup)

- Live camera transport (RTSP/USB/phone): file decode measured instead.
- Telegram notifier path (disabled in bench; code path exists).
- `run.py` HTTP server under load (Runtime class driven directly).
- Violation footage of any kind: event recall still unmeasurable (0 episodes).
- Short owner checklist: 1) `record_clip.py --source 0` shows your camera;
  2) record 3×2–3 min per scenario; 3) label with `label_events.py`;
  4) send clips+events back — scoring runs unchanged.

## 5. Conclusion

Single V4-cloud is viable as the unified-model base (cleaner persons, faster,
no incident regression on available footage), with one measured cost: −6
no_helmet GT coverages on stills vs dual. Whether that cost matters in live
tracking is decidable only on violation footage — which is now collectible
with the tools above. No custom multitask architecture needed yet; no V5;
no retraining; production untouched.
