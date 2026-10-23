# Model accuracy report (sprint `feature/security-model-quality`)

Weights unchanged in this sprint: only the system layer around the model.
No "model got X% better" claims without footage numbers (§H7 table still open).

## Current weights (verified on disk, 2026-10-08)

- PPE `ppe.onnx` (Hexmon/vyra-yolo-ppe, CC BY 4.0 card / AGPL-3.0 metadata —
  see THIRD_PARTY.md): emits person, helmet, no_helmet, vest, no_vest, mask,
  no_mask (+ unmapped fall-detected/gloves/goggles/ladder/safety-cone).
- Person `yolov8n.onnx` (Ultralytics, AGPL-3.0 metadata): 80 COCO classes,
  only `person` consumed.
- **Smoking is absent from the weights**: UI disables the rule with
  "не поддерживается моделью" driven by live `supported_classes()`.

## Measured on bus.jpg (CPU, defaults)

- Dual finds 3 persons; single-PPE finds 0 → dual stays recommended.
- `no_vest 0.52` on a street scene: previously a full detection (violation
  risk), now weak-only. `no_helmet 0.38`: previously invisible, now weak.
- Old-vs-new table: `docs/ACCURACY_PERFORMANCE_REPORT.md` (F6).

## Tracking measurement (synthetic, deterministic)

- Two persons crossing over 10 frames: IDs `[1,2]` stable, zero switches.
- 1-frame gap: ID preserved. Verdict: SimpleIoUTracker passes basic proxies;
  ByteTrack deferred until real footage shows switches (no new dependency).

## What the system layer now guarantees

- Per-class thresholds (absence 0.60), display floor 0.25 with weak flags.
- PPE attaches only with box overlap; weak persons spawn no tracks.
- Consensus 3/4 + confirm time; RESOLVED only on positive evidence;
  prolonged unknown → INTERRUPTED/observation_lost.

## Needed for real accuracy work

Footage A–O (`docs/CAMERA_VALIDATION_PLAN.md`), then `training/` pipeline:
`prepare_dataset.py` (verified) → `train.py` → `export_onnx.py` →
`evaluate.py` vs these weights on an independent test split. No training
started: no labeled data exists yet.

## Error table template (fill after footage; no invention)

| Scenario | Expected | Observed (fill) | Error class |
|---|---|---|---|
| Bare head front/side/back | no_helmet | | FN / wrong box |
| Helmet on | helmet, no incident | | FP no_helmet |
| Take off / put on | one incident / RESOLVED | | dup / stuck |
| Helmet on floor | raw only, no incident | | false association |
| Helmet in hands | raw only, no incident | | worn-vs-held |
| Vest on/off | vest / one incident | | FP / FN |
| Vest in hands / on floor | raw only | | false association |
| Two workers close | correct Worker each | | ID switch / cross |
| Far distance | unknown, no incident | | hallucinated PPE |
| Poor light | unknown or confirmed | | flicker incident |
| Partial occlusion | stable ID, no dup | | ID switch |

Consistent `no_helmet` = head region without helmet; `no_vest` = torso
without vest. Never mix class definitions across datasets.
