# PPE training audit — notes (facts, no claims)

## Run
- Machine: Apple M5, 16 GB RAM, PyTorch MPS. Env: `.venv-train` (torch 2.14.1, ultralytics 8.4.174).
- Base weights: `yolo11m.pt` (38.8 MB). Config: 640px, batch 4, workers 0, seed 17, patience 12.
- Epochs: 25 total. Epoch 1 first launch, then machine memory stop; resumed with
  `--resume` for epochs 2–25 (24 epochs, 1.55 h). Stop reason: epoch limit reached,
  early stopping did NOT trigger.
- Splits (ppe_public_v1): train 1304 / val 176 / test 124 = 1604 images.
- Dataset QA: 5 edge boxes clipped (pixel-rounding overflow ≤0.5%), 0 dropped,
  0 exact duplicates. `mask` has 1 box total, `no_mask` 52 — respirator classes
  cannot be learned from this data.

## Held-out source test (ultralytics, in industrial_eval.json)
- P 0.72 / R 0.63 / mAP50 0.58 / mAP50-95 0.31.
- WARNING (also in json): source-test metrics are NOT industrial accuracy.

## Old vs candidate ONNX, same 124-image test, same provider pipeline
(see `onnx_ab_comparison.txt` for the full log; detector conf 0.4 flat for both)
- helmet: AP50 0.49 → 0.74, R 0.56 → 0.78. vest: AP50 0.30 → 0.67, R 0.32 → 0.76.
- no_helmet: AP50 0.52 → 0.21, P 1.00 → 0.62 (6 FP), R 0.52 → 0.32. REGRESSION on
  this metric; test has only 31 no_helmet instances — small sample.
- no_vest: AP50 0.27 → 0.34, R 0.28 → 0.52, but P 0.90 → 0.54 (28 background FP).
  Recall/precision tradeoff, false-alarm risk to watch live.
- mask/no_mask: 0.00 both (0–2 instances in test) — untestable here.
- mAP50 0.324 → 0.380, mAP50-95 0.185 → 0.195 (now via fixed mean_ap_images).
- Latency: ~500 ms/frame both (CPU).

## Conflict to resolve live
On 4 real user snapshots the candidate raised no_helmet 0.40–0.47 weak to
0.67–0.80 strong (above the 0.60 violation threshold), while the test-set AP for
no_helmet regressed. Both facts are true; the test sample is tiny. Live-camera
validation decides. Candidate NOT deployed (models/ppe.onnx untouched).

## ROI mode (same raw frames, deployed weights — isolates ROI from retraining)
See `roi_ab.json`: no_helmet 0.40→0.42, 0.47→0.43 (still weak), +15–40 ms/frame.
ROI alone does not cross thresholds on these frames.

## v2: warm-start continuation to 50 total epochs
- `--resume` on a finished run silently restarts (ultralytics strips optimizer
  state on clean finish); added `--warm-start <best.pt>` into new
  industrial-ppe-v2 dir instead. v1 best.pt untouched.
- v2 test (ultralytics): P 0.75 / R 0.64 / mAP50 0.62 / mAP50-95 0.33.
- Same 124-image provider A/B, old / v1 / v2 AP50:
  helmet 0.49 / 0.74 / 0.80; no_helmet 0.52 / 0.21 / 0.28 (v2 P recovers to
  0.83, FP 6->2, R still 0.32); vest 0.30 / 0.67 / 0.66;
  no_vest 0.27 / 0.34 / 0.44 (v2 P 0.61, R 0.62); mAP50 0.324 / 0.380 / 0.409.
- BUT v2 on the 4 real snapshots is shakier than v1: 0.67->0.61,
  0.80->0.21 weak, 0.78->0.77, 0.71->0.50 weak. Test-AP up, snapshot
  confidence unstable frame-to-frame.
- Verdict unchanged: NO deployment. v1 wins snapshots, v2 wins test averages;
  both need live-camera validation with temporal confirmation and FP/hour.

## v3: auditor-supplied dataset (2369 imgs, no_helmet 951), 40 epochs from V2 best
- Verified on unpack: 2069/176/124, 9783 boxes, geometry clean, 0 exact dups.
- V3 test set is byte-identical to v1 test (124/124) -> all A/B comparable.
- Same-test AP50, orig / v1 / v2 / v3: helmet .49/.74/.80/.80;
  no_helmet .52/.21/.28/.40 (v3 P .78 R .45, 14/31, 4 FP);
  vest .30/.67/.66/.69; no_vest .27/.34/.44/.49;
  mAP50 .324/.380/.409/.438.
- Real snapshots (strong only): f1 orig--/v1.67/v2.61/v3.82;
  f2 --/.80/--/.81; f3 --/.78/.77/.77; f4 --/.71/--/.74.
  V3 fires strong no_helmet on all 4 frames where orig fires on none.
- Verdict: best candidate on both axes, still below orig on test no_helmet AP.
  NO deploy without live validation + FP/hour.
## v3-clean: from-scratch on leak-free data, 50 epochs
- Same-test AP50, orig / v3 / v3-clean: helmet .49/.80/.81;
  no_helmet .52/.40/.42 (clean: P .82 R .45, 14/31, 3 FP);
  vest .30/.69/.70; no_vest .27/.49/.51;
  mAP50 .326/.438/.446.
- Real snapshots (strong only): .82/.84/.69/.73 on 4/4 frames — best yet.
- Base re-run on cleaned labels moved only +0.002 (label fix is honest, tiny).
- Verdict: best candidate on both axes; no_helmet still trails orig on test AP.
  NO deploy without live FP/hour validation.

## Production-threshold A/B (fixed test, thresholds as in app)
- @0.4: v3 vs v3-clean mAP .441/.446 (no_helmet AP .40/.42, no_vest .51/.51).
- @prod (no_helmet/no_vest 0.60): orig collapses — no_helmet R .23 (7/31),
  no_vest R .00, vest R .02, mAP .167. This is why live showed zeros.
- @prod: v3 mAP .380 (no_helmet AP .28, 10/31; no_vest AP .34);
  v3-clean mAP .360 (no_helmet AP .25, 8/31 P.89; no_vest AP .25).
- Verdict: v3-clean ~= v3, both far above orig at violation level; per-frame
  no_helmet recall stays poor everywhere (23-32%) — temporal confirmation
  must carry it. NO deploy without live FP/hour.

## Incident: v1 results.csv lost
The 2-minute rogue fresh-start run (before the warm-start fix) overwrote
v1 run-dir previews and deleted v1 results.csv before I killed it. v1 plots
(results.png etc.), industrial_eval.json, weights and val images are intact;
v1 per-epoch numbers survive as the curve shape in results.png. v2 results.csv
is complete. Lesson: never start training into an existing run dir.

## Files
- SHAs: `best_sha256` in industrial_eval.json; onnx sha in
  models/ppe-candidate.onnx.sha256. Best .pt (161 MB) and candidate .onnx
  (80 MB) are local-only, NOT in git.
- training/evaluate.py: fixed mean_ap() → mean_ap_images() for per-image dicts
  (+ tests/test_metrics_images.py regression test).
