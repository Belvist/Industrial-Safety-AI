# Reproducible A/B: base vs candidate (same images, same thresholds)

- commit: `480f1f2dc46a2d7d17dcbb1229f384ee966ebd24`
- base: `models/ppe-v4.onnx` sha `688bbf4d68217d1ff722aac06b2bae0a0296b6e5938d4b29f68623edd2d09f88`
- candidate: `models/ppe-v4cloud.onnx` sha `f02ceb50ff1c2b4b41351e356c797dab866e5d900f4766b67b7cf542f6a55c46`
- thresholds: `{'person': 0.45, 'helmet': 0.5, 'no_helmet': 0.6, 'vest': 0.5, 'no_vest': 0.6}`, IoU 0.5, dual pipeline (person model `models/person.onnx`), ROI off
- test: `datasets/ppe_v3_clean/data.yaml` — 124 images, GT {'person': 218, 'helmet': 227, 'no_helmet': 31, 'vest': 131, 'no_vest': 61}
- mask/no_mask GT excluded from scoring (never remapped)

| class | base tp/fp/fn (P/R) | candidate tp/fp/fn (P/R) |
|---|---|---|
| person | 191/76/27 (0.715/0.876) | 191/76/27 (0.715/0.876) |
| helmet | 177/28/50 (0.863/0.78) | 191/25/36 (0.884/0.841) |
| no_helmet | 14/3/17 (0.824/0.452) | 12/1/19 (0.923/0.387) |
| vest | 89/18/42 (0.832/0.679) | 94/20/37 (0.825/0.718) |
| no_vest | 19/11/42 (0.633/0.311) | 27/13/34 (0.675/0.443) |

Latency: base 401.5 ms (p95 419.0), candidate 420.1 ms (p95 437.2) — CPU ONNX, this Mac.

Error sheets: `sheets/<model>_fn.jpg` (every miss, GT green) and `<model>_fp.jpg` (every false alarm, pred red); full lists in `error_frames.json`.

## Independent video (no GT violations found — FP/latency only)

- 3SuIh5zgKQQ / v3c: 0 ACTIVE FP (none), 467.4 ms/frame
- 3SuIh5zgKQQ / v4: 0 ACTIVE FP (none), 467.0 ms/frame
- F6ZxQe9tK4A / v3c: 0 ACTIVE FP (none), 515.2 ms/frame
- F6ZxQe9tK4A / v4: 0 ACTIVE FP (none), 514.9 ms/frame
- UVh_Jv0n1ZQ / v3c: 1 ACTIVE FP (no_vest@11.6s), 522.2 ms/frame
- UVh_Jv0n1ZQ / v4: 1 ACTIVE FP (no_vest@35.2s), 522.5 ms/frame
- V9fiUkXmAB4 / v3c: 0 ACTIVE FP (none), 494.0 ms/frame
- V9fiUkXmAB4 / v4: 0 ACTIVE FP (none), 494.1 ms/frame
- bFZ4Dbc4sos / v3c: 0 ACTIVE FP (none), 506.5 ms/frame
- bFZ4Dbc4sos / v4: 0 ACTIVE FP (none), 506.0 ms/frame
- reHtpEN9CDk / v3c: 0 ACTIVE FP (none), 522.1 ms/frame
- reHtpEN9CDk / v4: 0 ACTIVE FP (none), 521.7 ms/frame

Scope honesty: same-distribution stills (train/test share sources) — this measures in-distribution fit, NOT CCTV generalization. Derived torso boxes in V4 training are disclosed in V4_TRAIN_QA.md; CI tests cover pipeline code, not vision accuracy.

## United 3-way summary (same images, same prod thresholds 0.60)

| class | V3-clean tp/fp/fn (P/R) | V4 tp/fp/fn (P/R) | V4-cloud tp/fp/fn (P/R) |
|---|---|---|---|
| person | 191/76/27 (0.715/0.876) | same (shared person model) | same |
| helmet | 171/20/56 (0.895/0.753) | 177/28/50 (0.863/0.780) | 191/25/36 (0.884/0.841) |
| no_helmet | 8/1/23 (0.889/0.258) | 14/3/17 (0.824/0.452) | 12/1/19 (0.923/0.387) |
| vest | 88/18/43 (0.830/0.672) | 89/18/42 (0.832/0.679) | 94/20/37 (0.825/0.718) |
| no_vest | 20/6/41 (0.769/0.328) | 19/11/42 (0.633/0.311) | 27/13/34 (0.675/0.443) |

## Weights registry (binaries NOT in git — sizes below; verify by SHA)

- prod `models/ppe.onnx` sha `b99fed37…` — UNTOUCHED, still serving.
- `models/ppe-v3-clean.onnx` sha `b270cd96…` (baseline).
- `models/ppe-v4.onnx` sha `688bbf4d…` (80MB; train: 1608 imgs, 80 epochs, MPS).
- `models/ppe-v4cloud.onnx` sha `f02ceb50…` (80MB; train: 5059 imgs, 80 epochs,
  patience 15, batch 24, A10; `training/runs/industrial-ppe-v4cloud/weights/best.pt`
  sha `ded90261…`, 40MB).
- `models/person.onnx` sha `190ba5f1…` (shared detector, all runs).
- Reproduce either model: dataset recipe in `training/V4_TRAIN_QA.md` +
  `ppe_public_v3/TRAIN_V3.py --weights yolo11m.pt --epochs 80 --patience 15`
  (seed 17).

## Deployment status: NO-GO (unchanged)

Neither candidate replaces prod without violation footage (event recall,
FP/hour, live FPS). This push publishes evidence, not a release.
