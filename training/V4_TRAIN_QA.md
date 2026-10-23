# V4 TRAIN — QA report (SUPERSEDED by the expanded merge, see WORKLOG Запрос 34-37)

> This document describes the FIRST merge (1608 images). The current TRAIN is
> 5059 images (4554 train + 505 val): +957 hh accepts (2nd review round),
> +50 hh-bg, +2832 hb1204, +~10 coco-full. See `training/eval_ab_v4cloud_final/`
> for the 3-way A/B. Original first-merge report preserved below unchanged.

# V4 TRAIN — QA report (final, human-reviewed)

Date: 2026-10-10. Builder: `training/assemble_v4_train.py`
(frozen `build_ppe_v4.py` / `TRAIN_V3.py` / tests untouched).
Decisions: `training/v4_review_decisions.json` (370 verdicts, all reviewed).
Heavy files (images/labels/zip) live only in git-ignored `datasets/v4_train/`.

## Final set: 1448 train + 160 val (5 classes, strict gate PASS)

| class | boxes |
|---|---|
| helmet | 2787 |
| no_helmet | **1706** (was 184 in V3-clean train alone) |
| vest | 1110 |
| no_vest | **2273** (753 base + 1520 human-confirmed derived torso crops) |
| person | 3951 |

Groups: `v3c-rf100`, `v3c-coco`, `hh` (297 accepted), `hh-bg` (50 backgrounds).
Split: sha-stratified 90/10 per group (seed 17). No test/holdout by design
(holdout = brand-new scenes only). Old test/val never included (pixel-sha
forbid). TRAIN_V3 strict gate: conflicts 0, duplicates 0, drafts 0.

## Licenses (machine-checked, not memory)

- Kept: `rf100_*` — LibreYOLO/construction-safety-gsnvb, **CC BY 4.0** (HF API tag).
- Kept: `coco_*` — keremberke/construction-safety-object-detection,
  **CC BY 4.0** (`README.dataset.txt`).
- EXCLUDED: 762 `head_*` — hf-vision/hardhat has NO license tag and a
  "More Information needed" card → blocked until clarified.
- Added: keremberke/hard-hat-detection, **CC BY 4.0** (`README.dataset.txt` +
  loader header), HF sha `694c6135…`, `data/train.zip` sha256 recorded in
  `SOURCES.md`, deterministic filename-order quota (600 no-hardhat +
  600 hardhat-only + 60 background candidates).
- NOT included: Roboflow Safety Vests (CC BY 4.0, needs API key — owner action,
  exact command in `SOURCES.md`).

## Review protocol (the core of this QA)

1. Programmatic triage of 589 no-hardhat candidates: source-box sanity,
   head-in-person ≥50%, person-has-head ≥40% (or edge-cropped), vest rule.
   Result: 272 auto-reject (204 head-without-person, 50 person-without-head,
   18 bad-geometry), 317 to human review. Triage NEVER accepts.
2. **All 317 + 53 background candidates visually reviewed** (24 contact sheets,
   color-coded: H/N/V/P/D): accept iff every visible head/person/torso is
   correctly labeled. Doubt = reject. No boxes invented; vest drafts only
   confirmed, never drawn.
3. Verdicts: **297 accept / 20 reject** (t2), **50 / 3** (bg).
   Reject pattern (validates the review): 13× hi-vis vest labeled no_vest
   (detector missed them), 2× red hard hats labeled no-hardhat (confirmed by
   2× zoom crops — source noise pattern, documented), 1× unlabeled baby,
   3× visible arm fragments (bg), 1× orange uniform edge case.
4. no_vest completion: 1520 human-confirmed derived torso crops (uniform
   person-relative rule, drawn magenta on sheets). `no_vest` is never
   inferred from a missing detection.
5. Empty-label coco base (175 machinery/landscape images found during audit):
   person-detector filter (<0.5) → 159 kept as background, 16 dropped
   (visible people) + 32-sample visual spot check.

## Known limitations (honest)

- Derived torso boxes are uniform crops, not hand-drawn (count in audit).
- Old curated images carry no scene_id; intra-old-data scene leakage possible
  but bounded: holdout will be new scenes; pixel-dup hard-drop enforced.
- Sub-threshold tiny figures treated as background (consistent base+new).
- Operator-in-cab edge cases may slip through background filter.
- ok-pool (417 hardhat-only drafts) and Safety Vests still pending.

## Next

Human footage ≥3 scenes (no_helmet ≥200) → final assembly → ONE V4
(YOLO11m, 5 classes, 640) → ONNX → V3-clean A/B on independent video.
Production weights untouched until then.
