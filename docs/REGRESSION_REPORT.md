# Regression report (sprint `feature/security-model-quality`)

Re-run after any model/threshold/tracker change. Green CI is necessary but
not sufficient: these gates protect behavior CI cannot see without weights.

## Automated (CI)

- `pytest backend/tests` — currently 94 pass (includes threshold, consensus,
  unknown-semantics, token, retention, readiness suites).
- `npm test` + `npm run build` — currently 19 pass (pairing auth, QR URL, labels).
- `ruff` backend + `run.py`.

## Tool-driven (needs weights, local)

- `scripts/smoke_model.py <frame>` — single-image sanity.
- `scripts/benchmark_models.py` — dual vs single on same frames.
- `scripts/evaluate_folder.py <dir> [--labels] [--events]` — detector stats,
  P/R with labels, incident-level scoring with GT events.
- `docs/REGRESSION_SET.md` — clip layout; **no clips recorded yet**.

## Manual (needs camera)

- `docs/CAMERA_VALIDATION_PLAN.md` scenarios A–O on Mac + Windows/A4Tech.
- Phone QR pairing in `--lan` mode; Telegram delivery check.
- E2E checklist §7 of the sprint (14 items) against live UI.

## State machines covered by tests

Camera lifecycle, TemporalEngine phases incl. lost, browser capture machine,
UI command machine — transitions plus failure injection (double start,
stale stop, model-missing, snapshot-failure, queue-full, secret redaction).
