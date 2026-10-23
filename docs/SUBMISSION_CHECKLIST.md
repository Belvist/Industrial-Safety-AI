# Submission checklist (deadline 25.10, 10:00 MSK)

## Repository

- [ ] Submission branch pushed (do not rewrite `main` history).
- [ ] `README_HACKATHON.md` opens with a one-command start and a demo path.
- [ ] `docs/LINQ_CASE_TRACEABILITY.md` has PASS / PARTIAL / UNTESTED / FAIL /
      DONE / BLOCKED for every case requirement.
- [ ] `docs/LINQ_HARDWARE_REPORT.md` filled **only** with real measurements.
- [ ] `docs/DEMO_SCRIPT.md` matches the live flow.
- [ ] Backend tests: `cd backend && python -m pytest -q`.
- [ ] Backend lint: `cd backend && python -m ruff check .`.
- [ ] Frontend: `cd frontend && npm ci && npm run test && npm run build`.
- [ ] E2E: `python e2e/test_phone_flow.py ...` (or documented as UNTESTED).

## Materials for the personal cabinet

- [ ] Repository link or archive.
- [ ] README with verification steps.
- [ ] Presentation / short video (2–3 minutes).
- [ ] Screenshots of: calm state, active running incident, journal filters,
      LinQ status panel.
- [ ] Sample input and output (safe, no personal data).
- [ ] Third-party/licence file (`THIRD_PARTY.md`).

## Security — non-negotiable

- [ ] No `.ovpn`, VPN login/password, Jupyter password, API token or private
      key in Git, archives, screenshots or slides.
- [ ] Credentials shared in chat are treated as exposed; rotate them with the
      organizers if they were posted anywhere outside the private team chat.
- [ ] Demo footage contains no personal data or is used with permission.
- [ ] Do not publish the private VPN profile.

## Honesty gate

- [ ] No CPU/CoreML number presented as LinQ TPU.
- [ ] No unimplemented scenario shown as working.
- [ ] No invented accuracy, FPS or cost figures.
- [ ] Every UNTESTED/BLOCKED item named explicitly in the defence.
