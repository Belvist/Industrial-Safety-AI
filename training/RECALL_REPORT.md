# Recall without training: final comparison (noV5, no retraining)

Date: 2026-10-11. Commit: see meta in `training/sweep_recall/results.json`.
Models (SHA in results.json): v3-clean, v4, v4cloud. Fixed split: 124
same-distribution stills (GT 31 no_helmet / 61 no_vest). Dual pipeline,
ROI off, CPU. Thresholds swept in software from one pass at floor 0.05.
**Test used for diagnostics only — no production setting was derived here.**

Reproduce:
```
.venv/bin/python training/sweep_recall.py --data datasets/ppe_v3_clean/data.yaml \
  --models v3-clean:models/ppe-v3-clean.onnx v4:models/ppe-v4.onnx \
           v4cloud:models/ppe-v4cloud.onnx \
  --person-model models/person.onnx --out training/sweep_recall
.venv/bin/python training/sweep_temporal.py --out training/sweep_temporal
```

## 1. Threshold frontier (box recall; FP in brackets)

no_helmet (31): v4cloud peaks 19 (FP 3) @0.25; v4 18 (FP 11) @0.25.
no_vest (61): v3-clean 45 (FP 28) @0.25; v4cloud 40 (FP 43) @0.25,
35 (FP 27) @0.45. Full grid in `sweep_recall/results.csv`.

**Invisible even at 0.25** (no same-class detection ≥0.25 anywhere —
no threshold recovers these, only data/architecture):
no_helmet 12–14/31 (~40%), no_vest 16–25/61 (~30–40%). Full lists:
`sweep_recall/invisible.json`.

## 2. Split pick verdict: unnecessary

Best single model (V4-cloud) beats the V4-nh + V4cloud-nv combo on both
counts: nh 19/3 vs 18/11 @0.25. One model = half the inference cost
(~430 ms saved per frame on CPU). Recommended operating points (F1-optimal,
diagnostic only): nh 0.25 (19/31, FP 3), nv 0.45 (35/61, FP 27).

## 3. Temporal accumulation cost (9.6 min harsh workwear footage, all FP)

FP/hour by threshold (windows 4/3, 6/4, 8/5 — identical: persistent FPs
don't flicker, so wider windows do NOT help):
0.25 → 31–50/h; 0.35 → 19–38/h; 0.45 → 19–25/h; 0.60 → 0–6/h
(v4cloud@0.60: 0 incidents). Full grid: `sweep_temporal/temporal.csv`.

Conclusion: box recall 66% costs ~44 FP/hour — two orders of magnitude over
the ≤1/hour budget. Thresholds alone cannot reach the acceptance criteria;
the FP budget forces thresholds ≥ ~0.5, capping box recall at ~40–45%.

## 4. part_proto split costs (CoreML, measured after the person-per-crop fix)

(Code + runs archived at `research/part_crops/` after the negative verdict.)

person-only 7.3 ms, PPE full-frame 44.6 ms, PPE per-crop 38.9 ms (crops are
letterboxed to 640, so barely cheaper). Prior +196 ms/frame figure included
~19% waste from re-running person per crop — fixed, conclusion unchanged
(crops lose on accuracy regardless).

## 5. Fundamentally unfixable by post-processing

- The invisible set above (needs data or architecture).
- Systematic stripe/uniform FPs (temporal consensus can't vote out a
  persistent false detection — proven by window invariance).
- Event recall on real violations: unmeasurable, 0 labeled episodes exist.

## 6. ONE proposed experiment: head/torso crop classifiers (NOT run)

- Data (free labels): head crops from helmet/no_helmet boxes (~4000+, hb
  included), torso crops from vest/no_vest boxes. 3-way labels:
  present/absent/unknown (occluded/tiny → unknown, human-verified sample).
- Model: small CNN (MobileNet-class), input = crop, output = 3 states.
  No YOLO retraining, no full-frame changes.
- Pipeline: person box → geometric head/torso split (existing splitter) →
  classifiers vote into TemporalEngine alongside (not instead of) box path.
- Eval: same 124 (must recover a stated share of the invisible set) +
  clean videos (must not raise FP/hour) + violation footage when labeled.
- Success bar: +10pp event recall with FP/hour flat, else drop it.
- NOT launched: awaiting violation footage + owner go-ahead.

## 7. Labeling protocol for the owner (unchanged)

3 scenes × 4 combos, static camera, stable worker ids + violation intervals
+ 2–4 boxes per episode at 5 fps numbering; clean footage for FP/hour;
frozen holdout scenes. Template: `training/video_events_TEMPLATE.json`.
