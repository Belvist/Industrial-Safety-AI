# Regression set (re-runnable after any model/threshold change)

Layout (gitignored, lives next to the repo, e.g. `../regression/`):

```text
regression/
  clean-corridor/
    frames/f0001.jpg ... f0120.jpg   # person walking, full PPE, no violations
    events.json                       # {"events": []}
    expected.md                       # 0 incidents, IDs stable
  no-helmet-then-helmet/
    frames/*.jpg
    events.json                       # [{"type": "no_helmet", "start_frame": 20, "end_frame": 80}]
    expected.md                       # exactly 1 no_helmet ACTIVE, then RESOLVED
  vest-on-floor/
    frames/*.jpg
    events.json                       # {"events": []}
    expected.md                       # 0 incidents (G1)
```

Run:

```bash
python backend/scripts/evaluate_folder.py ../regression/<clip>/frames \
  --events ../regression/<clip>/events.json --fps 5
```

Rules:

- Every clip ships `expected.md` with exact expected counts (no "looks fine").
- A model/threshold/tracker change is accepted only if ALL clips still match.
- Clips come from `docs/CAMERA_VALIDATION_PLAN.md` scenarios A–O once shot.
- Current status: no clips recorded yet — the bus.jpg smoke above is NOT
  a regression clip, only a tool self-check.
