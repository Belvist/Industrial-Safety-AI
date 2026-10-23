"""Browser E2E: operator QR -> phone /connect -> /phone capture -> stop.

Mock MediaStream (fake device), real backend + real ONNX inference.
NOT a physical-phone test: getUserMedia is emulated, detector sees the
fake-device pattern (no persons expected). Proves the user path works.

Usage:
    python e2e/test_phone_flow.py --base http://127.0.0.1:8001 --master <token>
"""

import argparse
import json
import re
import urllib.error
import urllib.request

from playwright.sync_api import expect, sync_playwright

CHROMIUM_ARGS = [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
]


def api(base, master, method, path, body=None):
    request = urllib.request.Request(
        base + "/api/v1" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-API-Token": master, "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--master", required=True)
    args = parser.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=CHROMIUM_ARGS)
        operator = browser.new_context()
        operator.grant_permissions(["camera", "microphone"])
        page = operator.new_page()
        page.goto(f"{args.base}/?token={args.master}")
        page.get_by_role("button", name="Камеры и правила").click()

        with page.expect_response(re.compile(r"/api/v1/pairing$")) as pairing:
            page.get_by_role("button", name="Создать код для телефона").click()
        code = pairing.value.json()["code"]
        camera_id = pairing.value.json()["camera_id"]
        print(f"pairing code minted for camera {camera_id[:8]}")

        phone = browser.new_context()
        phone.grant_permissions(["camera", "microphone"])
        mob = phone.new_page()
        mob.goto(f"{args.base}/connect?pair={code}")
        expect(mob.get_by_text("подключён")).to_be_visible(timeout=15000)
        token = mob.evaluate("() => localStorage.getItem('phone_token')")
        assert token, "phone token stored after exchange"
        print("exchange ok, phone token stored")

        mob.get_by_role("button", name="Открыть камеру телефона").click()
        mob.get_by_role("button", name="Начать передачу").click()
        expect(mob.get_by_text("работников")).to_be_visible(timeout=30000)
        print("phone streaming, backend returns results")

        cams = api(args.base, args.master, "GET", "/cameras")
        mine = next(c for c in cams if c["id"] == camera_id)
        assert mine["status"] == "running", mine
        assert mine["latest"] and mine["latest"]["sequence"] > 0
        print(f"operator sees running camera, seq={mine['latest']['sequence']}")

        mob.get_by_role("button", name="Остановить").click()
        expect(mob.get_by_text("остановлена")).to_be_visible(timeout=15000)
        cams = api(args.base, args.master, "GET", "/cameras")
        assert next(c for c in cams if c["id"] == camera_id)["status"] == "stopped"
        print("phone stop ends session; token revoked, restart denied:")
        denied = api_raw(args.base, token, camera_id)
        assert denied == 401, denied
        print("revoked token rejected with 401")
        browser.close()
    print("E2E phone flow: PASS (mock media, not a physical phone)")


def api_raw(base, phone_token, camera_id):
    request = urllib.request.Request(
        base + f"/api/v1/cameras/{camera_id}/start",
        data=json.dumps({"kind": "browser"}).encode(),
        headers={"X-API-Token": phone_token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            return 200
    except urllib.error.HTTPError as exc:
        return exc.code


if __name__ == "__main__":
    main()
