"""Real-browser operator UX regression, with emulated webcam and real backend."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--browser-executable", default=None)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/workspace-e2e"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=args.browser_executable,
            args=[
                "--no-sandbox",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 960}, permissions=["camera"]
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{args.base}/?token={args.master}")
        nav = page.get_by_role("navigation", name="Основная навигация")
        expect(
            page.get_by_role("heading", name="Мониторинг", exact=True)
        ).to_be_visible()
        nav.get_by_role("button", name="Обзор", exact=True).click()
        expect(page.get_by_role("heading", name="Все камеры")).to_be_visible()
        page.get_by_role("button", name="Камера Камера 1", exact=True).click()
        expect(
            page.get_by_role("heading", name="Мониторинг", exact=True)
        ).to_be_visible()
        page.go_back()
        expect(page.get_by_role("heading", name="Обзор", exact=True)).to_be_visible()
        nav.get_by_role("button", name="Камеры и правила", exact=True).click()
        expect(
            page.get_by_role("heading", name="Настройки рабочей зоны")
        ).to_be_visible()
        page.get_by_role("button", name="Подключить камеру", exact=True).click()
        expect(page.get_by_label("Устройство камеры")).to_be_focused()
        page.get_by_role("button", name="Запустить мониторинг", exact=True).click()
        with page.expect_response(
            lambda response: response.url.endswith("/frame") and response.status == 200,
            timeout=30000,
        ) as frame:
            pass
        body = frame.value.json()
        assert body["sequence"] >= 1 and body.get("metrics")
        assert body["metrics"]["rates"]["analytics"]["fps"] > 0
        assert (
            body["metrics"]["rates"].get("tpu") is None
        )  # local ONNX is never labelled TPU
        expect(
            page.get_by_role("button", name="Остановить", exact=True)
        ).to_be_visible()
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(
            path=str(args.artifacts / "workspace-desktop.png"), full_page=True
        )
        page.get_by_role("button", name="Остановить", exact=True).click()
        page.set_viewport_size({"width": 390, "height": 844})
        for name in ["Обзор", "Камеры и правила", "Инциденты", "Мониторинг"]:
            nav.get_by_role("button", name=name, exact=True).click()
            assert page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ), name
        page.screenshot(
            path=str(args.artifacts / "workspace-mobile.png"), full_page=True
        )
        assert not errors, errors
        browser.close()
    print(
        "Workspace E2E: PASS (desktop, mobile, history, source focus, webcam, measured analytics)"
    )


if __name__ == "__main__":
    main()
