# PPE confidence experiment — same model, two image-processing modes

## Purpose

The dedicated COCO person model may see a worker at ~0.9 confidence while the
PPE detector emits `no_helmet=0.40-0.47` and `no_vest=0.16`. These values
come from three local user snapshots reported by the agent, not a validated
dataset. The image-level confidence is not a precision/recall metric.

Current change **does not retrain or replace** the PPE weights.
`PPE_ROI_MODE=auto` uses 1-2 person crops for PPE ONNX inference, expanded
above the worker box to include the helmet. For 3+ workers, very large crops,
very small crops, no worker or `USE_SEPARATE_PERSON_MODEL=false`, it
falls back to the exact old full-frame pass.

The detector retains its class-specific thresholds:
`NO_HELMET_THRESHOLD=0.60`, `NO_VEST_THRESHOLD=0.60`.
Weak boxes remain non-actionable. Class support is unchanged: the present
local ONNX model does **not** detect smoking.

## Run A/B on real camera images

Use original frames without annotations. Annotated incident snapshots are
a lower-quality substitute because the drawn rectangles alter the pixels.

```bash
python3.11 backend/scripts/compare_ppe_roi.py /path/to/raw1.jpg /path/to/raw2.jpg --runs 5 --out data/ppe_ab.json
```

Windows: `py -3.11 backend/scripts/compare_ppe_roi.py C:\\test\\raw1.jpg`
(use the project's installed virtual environment, if dependencies are not on
the system Python).

For each raw frame, compare the *same* model:
- `off`: full-frame PPE inference (baseline).
- `auto`: PPE inference from person crops where eligible.

Inspect class, bbox, confidence, weak/strong and latency. A candidate only
deserves promotion if independent labeled footage confirms improved recall
without worse false-positive rate or unusable FPS. Rising confidences alone
are **not proof of correctness**.

## Live test

1. Put `PPE_ROI_MODE=auto` in `.env` and restart the backend.
2. Verify `/api/v1/system/info` returns
   `inference.ppe_roi_mode=auto` (and no `model_error`).
3. Record raw frames with and without a helmet/vest under the same camera
   position, ideally worker side/back and difficult lighting.
4. Show raw detections and incidents. An absent PPE detection means unknown,
   not automatically a missing item.
5. Repeat with `PPE_ROI_MODE=off`. If `auto` hurts FPS or causes wrong
   violations, restore `off` and restart.

## Still needed

- Ground-truth annotations and full-path event tests with the same people.
- Real sustained CPU/CoreML FPS and latency.
- Fine-tuning or alternative licensed weights if both modes miss PPE.
- A separate model/dataset for smoking (not represented in current weights).
