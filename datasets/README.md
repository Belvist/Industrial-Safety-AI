# datasets

Private by default. NEVER commit worker photos or site footage to public GitHub
without written permission. This directory holds only tooling docs; the data
itself lives outside the repo (e.g. `../datasets/`) and is gitignored.

## Scenarios to shoot (see `docs/CAMERA_VALIDATION_PLAN.md` A–O)

Worker with/without helmet and vest, putting on/taking off, respirator if worn,
near/far/side/back views, occlusion, poor light, 2+ workers, PPE on the floor,
PPE in hands (NOT worn — critical negatives), person leaving the frame.

## Label format (YOLO detection, normalized 0..1)

```text
dataset_root/
  clip-001-corridor-no-helmet/
    images/f0001.jpg ...
    labels/f0001.txt ...     # "<class_id> <x_center> <y_center> <w> <h>" per line
  clip-002-.../
```

Class ids (`training/prepare_dataset.py --classes`):

```text
0 person, 1 helmet, 2 no_helmet, 3 vest, 4 no_vest, 5 mask, 6 no_mask
```

## Event labels (incident-level scoring)

`events.json` next to clips: `{"events": [{"clip": "clip-001", "type": "no_helmet",
"start_frame": 20, "end_frame": 80}]}` — consumed by
`backend/scripts/evaluate_folder.py --events` per clip folder.

## Split policy (no leakage)

Split at CLIP level (never shuffle frames across clips): all frames of one
recording go to exactly one of train/validation/test. Prefer separating
cameras/sites/sessions across splits. See `training/prepare_dataset.py`.

## Licensing

Record every source dataset + weight license in `THIRD_PARTY.md`. Ultralytics
tooling is AGPL-3.0 — fine-tuning with it has license consequences for the
resulting weights; resolve before any commercial use.
