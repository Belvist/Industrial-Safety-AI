# Browser E2E (Playwright, mock MediaStream)

`test_phone_flow.py` drives two real browser contexts against a live server:
operator mints a pairing code via UI click → phone exchanges it on `/connect`
→ captures on `/phone` with an emulated camera → backend processes frames →
operator sees the running camera → phone stop ends the session and revokes
the token (reuse rejected with 401).

Mock media only: the detector sees the fake-device pattern (no persons).
This proves the user path, not detection quality. Physical-phone check stays
manual (see `docs/CAMERA_VALIDATION_PLAN.md`).

Run locally:

```bash
# terminal 1: server with a master token
DATABASE_URL="sqlite:////tmp/e2e/db.sqlite" DATA_DIR=/tmp/e2e/data \
API_TOKEN=e2e-master-123 PUBLIC_URL=http://127.0.0.1:8002 \
EXTRA_ORIGINS=http://127.0.0.1:8002 \
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --app-dir backend
# terminal 2:
python e2e/test_phone_flow.py --base http://127.0.0.1:8002 --master e2e-master-123
```

Needs: `pip install playwright` + `playwright install chromium`.
Runs in CI (`e2e` job, ubuntu, headless Chromium).

`test_workspace.py` additionally checks desktop/mobile navigation, browser history,
source focus and measured analytics through an emulated webcam and the real backend.
Run against the same server:

```bash
python e2e/test_workspace.py --base http://127.0.0.1:8002 --master e2e-master-123
```

Both scripts accept `--browser-executable /usr/bin/chromium` to use an installed
browser. They do not prove physical-phone support, real-video accuracy or remote TPU FPS.
