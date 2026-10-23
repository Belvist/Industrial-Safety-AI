# Diagnostic report: V3-clean vs V4 — what actually blocks violations

Date: 2026-10-11. Commit: see `training/diag_v4/meta.json`.
Weights: base `models/ppe-v3-clean.onnx`, candidate `models/ppe-v4.onnx`
(SHAs in meta.json). Fixed production thresholds
(person 0.45, helmet/vest 0.50, no_helmet/no_vest 0.60), IoU 0.5, dual
pipeline. Test: 124 same-distribution stills (GT 31 no_helmet / 61 no_vest).

Reproduce:
```
.venv/bin/python training/diag_objects.py --data datasets/ppe_v3_clean/data.yaml \
  --base models/ppe-v3-clean.onnx --candidate models/ppe-v4.onnx \
  --person-model models/person.onnx --out training/diag_v4
.venv/bin/python -m pytest backend/tests/test_evaluate_folder_events.py -q
```

## 1. Config matrix (same frames, nothing mixed)

| model | roi | EP | p50/p95 ms | FPS | nh R/P | nv R/P |
|---|---|---|---|---|---|---|
| base | off | CPU | 444.9/530.7 | 2.20 | 0.258/0.889 | 0.328/0.769 |
| base | off | CoreML | 37.6/43.1 | 25.59 | 0.258/0.889 | 0.344/0.750 |
| base | auto | CPU | 470.4/852.1 | 1.84 | 0.258/0.889 | 0.295/0.720 |
| base | auto | CoreML | 37.1/69.8 | 23.34 | 0.258/0.889 | 0.311/0.704 |
| V4 | off | CPU | 440.2/472.2 | 2.24 | 0.452/0.824 | 0.311/0.633 |
| V4 | off | CoreML | 36.3/37.7 | 27.01 | 0.452/0.824 | 0.311/0.633 |
| V4 | auto | CPU | 441.9/832.6 | 1.96 | 0.452/0.778 | 0.279/0.586 |
| V4 | auto | CoreML | 37.0/69.0 | 23.43 | 0.452/0.778 | 0.279/0.586 |

Findings:
- **CoreML works on both models: ~36–37 ms p50 → 23–27 FPS inference.**
  The 2 FPS verdict measured the CPU path only. Full-app FPS (transport +
  tracking + incident save) still unmeasured — that, not the model, is next.
- **ROI auto gives zero recall gain** on this set (nh_R identical everywhere),
  lowers precision slightly and blows CPU p95 (852 ms double-inference).
- CPU↔CoreML recall identical except base no_vest 0.328 vs 0.344 (one box of
  numeric noise between execution providers — disclosed, not hidden).
- Person metrics byte-identical across all 8 configs (separate person model).

## 2. Object-level causes (reference: off/CPU; sums assert-equal to match_images)

V4 no_helmet FN=17: bad_iou 5, below_threshold 3, no_detection 4,
taken_by_other_gt 4, no_person 1.
V4 no_vest FN=42: taken_by_other_gt 17, below_threshold 10, bad_iou 8,
wrong_class 3, no_detection 3, no_assign 1.
V4 FP: no_helmet 3 (background 2, wrong_class 1); no_vest 11
(background 7, mislocalized 2, wrong_class 2).
Base numbers in `causes.json`; per-object records in `objects_fn.json` /
`objects_fp.json` (+ CSV); one crop per violation-class error in `crops/`.

Cause definitions: wrong_class = strong other-class pred IoU≥0.5 here;
bad_iou = strong same-class exists but best IoU<0.5; below_threshold =
same-class visible only under 0.60 (incl. weak); no_person = no strong
person covering the GT center; no_assign = person present but band rule
fails (emulated with detection boxes — approximation, documented);
no_detection = nothing; taken_by_other_gt = same-class IoU≥0.5 pred
consumed by an adjacent GT first (crowd disambiguation / greedy matching).

## 3. Conclusion: what blocks most

1. **Model-side detection (threshold + localization), not person/tracking.**
   Person/assign causes total ≤2 per violation class on stills. The reviewer's
   tracking hypothesis does not reproduce here.
2. **no_vest: 10 misses sit below 0.60** → threshold retune candidate (FP cost
   must be measured as incidents/hour, not boxes). 17 more are crowd
   disambiguation (adjacent torsos, greedy matching) — tighter boxes or
   per-worker reasoning, not epochs.
3. **no_helmet: 12/17 are model-side** (bad_iou 5, below 3, no_detection 4);
   small/distant heads dominate the miss sheets.
4. **FP background** (helmet 13–16, vest 14–15, person 68) is the FP/hour
   reservoir — same for both models.

## 4. Video benchmark tooling (no footage yet — tool ready, tested)

`backend/scripts/evaluate_folder.py --events` now does track-aware matching:
GT events carry `worker` + sparse `boxes`; a detection matches only on
type + time overlap + spatial agreement (≥50% of common frames IoU≥0.3).
Alarm-on-wrong-person = FP + FN (`worker_mismatch`), never a match. Events
without boxes fall back to time-only flagged `worker_unverified`.
Tests: `backend/tests/test_evaluate_folder_events.py` (6 passed).

Labeling protocol for the owner (3 scenes × 4 combos, static camera):
per worker a stable id + violation intervals + 2–4 boxes per episode at
`--fps` frame numbering; clean footage for FP/hour; separate scenes frozen
for the final holdout. No GT = no recall claims.

## 5. Honesty limits of this report

- Same-distribution stills; CCTV generalization unproven.
- Band emulation uses detection boxes, not tracker IDs (still images).
- 1520/2273 no_vest training boxes are derived (see V4_TRAIN_QA.md).
- CI covers code, not vision. Weights/datasets are SHA-referenced, not in git.
- Video criteria (recall/FP-hour/latency/FPS-live) remain unproven without
  labeled violation footage.
