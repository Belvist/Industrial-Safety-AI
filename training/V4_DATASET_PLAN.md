# V4 dataset plan — violation-focused, leak-free from day one

Status: PIPELINE IMPLEMENTED, SELF-TEST GREEN, FOOTAGE MISSING. Do not start
V4 training until the footage below exists. No V4 epochs until then — more
epochs on old data is a bad spend (auditor verdict, Oct 2026).

Pipeline self-check (no footage, weights, or network needed):

```
.venv/bin/python training/build_ppe_v4.py --self-test
```

plus `backend/tests/test_v4_dataset_pipeline.py` in CI (scene isolation,
audit parity, old-data guard, split gates).

## Target
- 300–600 NEW labeled frames from independent videos/cameras (not Roboflow
  re-crops, not frames already in v1/v3).
- 5 classes: `helmet, no_helmet, vest, no_vest, person` (respirators stay a
  separate future model — mask has 1 box total, unlearnable here).
- Minimum 150 verified `no_helmet` + 150 verified `no_vest` boxes, plus
  negatives (correct PPE) from the SAME scenes (else the model learns
  "this camera = violation").
- Untouched holdout: whole scene clusters, `--holdout-frac 0.15`, used ONLY
  for the final evaluation. Old test/val and `review/fn_*` crops NEVER train.

## Collection protocol (owner films, agent assembles)
1. cuda: helmet on/off × vest on/off × 2+ people × near/far × day/evening light.
   Short video clips (10–20 s), then extract 1 frame / 2 s (no bursts).
2. Variety: ≥3 locations, ≥2 cameras (phone + laptop/USB), portrait + landscape.
3. Consent: only people who agreed; no public-space strangers in close-up.
4. Drop footage into `footage_v4/` (git-ignored): one subfolder per
   video/scene (`scene_<place>_<date>/`), keep original filenames.

## Licenses
- New self-recorded footage: owned, no restriction.
- Any public supplement: CC BY 4.0 or more permissive ONLY; record source +
  URL in the dataset CREDITS.md. `hf-vision/hardhat`-style unclear sources
  are private-experiment-only and must NOT enter V4.
- Stock/watermarked photos (e.g. Dreamstime crops found in v3): EXCLUDE —
  they cluster across splits and poison independence.

## Label QA (before any training)
- Two-person rule for violations: every `no_helmet`/`no_vest` box confirmed
  against the frame; helmet↔no_helmet / vest↔no_vest overlaps (IoU>0.5) are
  auto-flagged by the builder and must resolve to zero.
- Spot-check contact sheet (16+ samples) saved next to the dataset.

## Assembly (raw MP4 in, audited dataset out)

Split key is `scene_id`: all clips of one shoot share a scene and own
exactly one split — so different videos of the same shoot can never land in
train and holdout at once (dissimilar frames of one video can't leak either).
pHash stays as a warning audit only; exact pixel duplicates across splits are
a hard failure.

```
footage_v4/clip_yard_morning/
  clip.mp4        # exactly one video per dir (.mp4/.mov/.avi/.mkv)
  LICENSE.json  # {"origin": "self-recorded"|"public", "license": "owned"|"CC BY 4.0",
                  #  "source_url": null|"...", "consent": true,
                  #  "scene_id": "shoot_yard_20261012"}  # optional, default: dir name.
                  #  ALL clips of one shoot MUST share one scene_id.
  labels/f000120.txt   # hand-checked YOLO labels (class ids below)
```

Fixed class order: `0 helmet, 1 no_helmet, 2 vest, 3 no_vest, 4 person`.

```
# 1. Extract frames (1 frame / 2 s per protocol) -> footage_v4/_staging/
python training/build_ppe_v4.py --footage footage_v4 --stage extract

# 2. Draft pre-labels with existing weights (optional; hand labels/ win).
#    Drafts NEVER train without review (build fails unless --allow-drafts).
python training/build_ppe_v4.py --footage footage_v4 --stage autolabel \
  --autolabel-weights models/ppe-v3-clean.onnx --person-model models/person.onnx

# 3. Review drafts into footage_v4/<video>/labels/, then build:
python training/build_ppe_v4.py \
  --footage footage_v4 --output datasets/ppe_v4 --holdout datasets/ppe_v4_holdout \
  --max-images 600 --min-per-class 150 --holdout-frac 0.15

# ...or all stages at once (needs confirmed labels already present):
python training/build_ppe_v4.py --footage footage_v4 --output datasets/ppe_v4 \
  --holdout datasets/ppe_v4_holdout --max-images 600 --min-per-class 150
```

Gates (fail by design, no silent mixing): `LICENSE.json` valid + consent +
scene_id; exactly one video per dir; every frame labeled; 0 opposite-label
conflicts (IoU>0.5, either label order); 0 pixel-matches against old data
(`--forbid-dirs` defaults cover v1/v3/v3-clean + old test/val — review crops
must never train; a non-empty list that hashes 0 images fails as vacuous —
pass `--forbid-dirs` bare only when no old data exists at all); ≥3 scenes for train/val/holdout; `--min-per-class` violations
present; outputs must be empty dirs. Products: `train|val/images+labels`,
`data.yaml`, `audit.json` AND `audit_leaks.json` (byte-identical — TRAIN_V3.py
accepts either), `manifest.json` (scene_id/video_id/sha/draft/license per
frame), `CREDITS.md`, `contact_sheet.jpg`, holdout dir with
`HOLDOUT_README.md`. Old test/val and `review/fn_*` crops NEVER train.

## Training (only after the above is green)
```
python ppe_public_v3/TRAIN_V3.py --dataset datasets/ppe_v4 \
  --classes helmet no_helmet vest no_vest person \
  --weights yolo11m.pt --epochs 50 --patience 12 --name industrial-ppe-v4
```
- Audit gate accepts `audit_leaks.json` or `audit.json` (legacy V3 dir still
  uses `AUDIT_VERIFIED.json`).
- Weights bootstrap: explicit `--weights` > V2 `best.pt` when present >
  `yolo11m.pt` (repo-root copy or Ultralytics auto-download on first use).
  A clean checkout no longer stops on a missing V2 checkpoint.
- Warm-start from v3-clean is FORBIDDEN for the 5-class head (shape mismatch);
from-scratch transfer from yolo11m only. Then: same-test A/B vs v3-clean,
production-threshold run, live FP/hour — no prod swap without them.

## Threshold-sweep CLI truth (audit note)

`training/threshold_sweep.py` runs `ROI=off` only (recorded as `roi_mode:
off` in `sweep.json`) with `--grid` thresholds (default 0.30–0.70); there is
no `--reuse-cache` flag — re-runs cost one ONNX pass per model. Event-level
≥80% needs video measurement (`backend/scripts/evaluate_folder.py --events`),
not the still-image table.
