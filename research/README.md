# Research archive (not production evidence)

Historical ML experiments, preserved with their reports — moved out of the
working tree so the active product stays unambiguous. Do NOT train from here
without re-validation. Provenance (commit, weights SHA, thresholds) lives in
each sibling report under training/.

- part_crops/: tight-crop reuse experiment (negative result, see WORKLOG).
- history/: early A/B text notes; bench_runtime[_fast] (incident counts here
  used len(dict) — superseded by bench_runtime_v4/_prod with total+verified).
- diag_crops/: 144 per-object error crops for training/diag_v4/.
- sweep_review/: 21 review images for training/sweep_v3_v3clean/.
