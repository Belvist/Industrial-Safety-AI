# Unified model: verdict after measurement (no new architecture built)

Date: 2026-10-11. Question: is a separate person.onnx load-bearing, or does
the unification already live inside YOLO? Answer below is measured, not argued.

Reproduce:
```
.venv/bin/python training/eval_person.py --data datasets/ppe_v3_clean/data.yaml \
  --models v3-clean:models/ppe-v3-clean.onnx v4:models/ppe-v4.onnx \
           v4cloud:models/ppe-v4cloud.onnx \
  --person-model models/person.onnx --out training/eval_person
```

## 1. Intrinsic person vs dedicated detector (124 stills, prod thr 0.45)

| model | source | P / R / F1 | tp/fp/fn | ms |
|---|---|---|---|---|
| v3-clean | dedicated | 0.715/0.876/0.788 | 191/76/27 | 452 |
| v3-clean | intrinsic | 0.804/0.885/0.843 | 193/47/25 | 373 |
| v4 | intrinsic | 0.806/0.876/0.840 | 191/46/27 | 387 |
| v4cloud | intrinsic | 0.780/0.862/0.819 | 188/53/30 | 382 |

The dedicated detector adds ~25 false persons at equal recall AND costs an
extra pass. Its `person` outputs were the noisiest part of the pipeline,
not its stabilizer — on stills. Full results: `training/eval_person/person.json`.

## 2. Single-model incidents on UVh (244 s, prod thresholds, 4/3 window)

| config | ACTIVE FP | worker IDs | ms/frame |
|---|---|---|---|
| v4 dual | 1 (no_vest) | 50 | 522 |
| v4cloud dual | 0 | 50 | 412 |
| v4cloud single (intrinsic) | 0 | 46 | 374 |

Fewer ghost tracks (46 vs 50), no new incidents, ~150 ms saved per frame.
Single-model dumps: `footage_v4/yt_eval/dets_v4csingle_*.json` (git-ignored).

## 3. Conclusion

- **The unification already exists inside YOLO**: ppe-v4cloud.onnx IS one
  ONNX with person + PPE heads (5 classes, verified names metadata). No
  weight merging (impossible/meaningless), no custom architecture needed
  for the prototype: stop discarding its person outputs, validate tracking.
- **Custom multitask heads deferred**: justified only if (a) violation
  footage shows workers lost by intrinsic person, or (b) a distilled
  student must run where the 80MB model cannot. Neither is proven.
- **Distillation/fine-tune: NOT launched** (needs violation footage first).

## 4. Honesty limits

- Stills only for person quality; tracking recall on missed persons (188 vs
  191) needs violation video with labeled workers.
- Single-model path not yet enabled in production config (still dual);
  switch requires the video proof above + owner approval.
- Weights/datasets SHA-referenced, not in git. CI covers code, not vision.
