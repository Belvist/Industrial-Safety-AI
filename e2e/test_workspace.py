"""Operator journey against a real backend and ONNX, using a simulated webcam.

No production data is seeded. The fake camera has no people; this test does
not prove PPE accuracy, remote TPU throughput or physical-camera compatibility.
"""

import argparse
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--browser-executable", default=None)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/workspace-e2e"))
    args = parser.parse_args()
    saved_name = f"Сборочный цех · {time.time_ns()}"
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
        expect(nav.get_by_role("button", name="Наблюдение", exact=True)).to_be_visible()
        expect(page.locator(".p-local-mark")).to_have_text("Приложение подключено")
        assert "token=" not in page.url
        page.screenshot(
            path=str(args.artifacts / "welcome-desktop.png"), full_page=True
        )
        expect(page.locator(".product-app")).to_have_attribute("data-theme", "dark")
        page.get_by_role("button", name="Светлая тема", exact=True).click()
        expect(page.locator(".product-app")).to_have_attribute("data-theme", "light")
        page.wait_for_timeout(200)
        page.screenshot(path=str(args.artifacts / "welcome-light.png"), full_page=True)
        page.get_by_role("button", name="Тёмная тема", exact=True).click()
        expect(page.locator(".product-app")).to_have_attribute("data-theme", "dark")
        page.locator(".p-app-breadcrumb").click()
        page.keyboard.press("Control+b")
        expect(
            page.get_by_role("complementary", name="Навигация приложения")
        ).not_to_be_visible()
        page.keyboard.press("Control+b")
        expect(page.get_by_role("complementary", name="Навигация приложения")).to_be_visible()
        nav.get_by_role("button", name="Камеры", exact=True).click()
        page.get_by_role("button", name="Новая камера", exact=True).click()
        dialog = page.get_by_role("dialog", name="Новая камера", exact=True)
        dialog.get_by_label("Как назвать камеру").fill("Сборочный цех · E2E")
        dialog.get_by_role("button", name="Видеофайл").click()
        dialog.get_by_role("button", name="Продолжить", exact=True).click()
        expect(dialog.get_by_role("alert")).to_have_text(
            "Выберите видеофайл для анализа."
        )
        dialog.get_by_role("button", name="Веб-камера").click()
        dialog.get_by_role("button", name="Продолжить", exact=True).click()
        expect(
            dialog.get_by_role("heading", name="Что нужно контролировать?")
        ).to_be_visible()
        with page.expect_response(
            lambda r: r.url.endswith("/frame") and r.status == 200, timeout=60000
        ) as frame:
            dialog.get_by_role("button", name="Начать наблюдение", exact=True).click()
        body = frame.value.json()
        assert (
            body["sequence"] >= 1 and body["metrics"]["rates"]["analytics"]["fps"] > 0
        )
        assert body["metrics"]["rates"].get("tpu") is None
        expect(page.locator(".p-observation-status strong")).to_have_text(
            "Наблюдение работает", timeout=30000
        )
        expect(page.locator(".p-inspector")).to_have_count(0)
        expect(page.locator(".video-toolbar")).to_have_count(0)
        page.screenshot(
            path=str(args.artifacts / "observation-desktop.png"), full_page=True
        )
        page.get_by_role("button", name="Люди").click()
        expect(page.get_by_role("heading", name="Люди в кадре")).to_be_visible()
        page.keyboard.press("Escape")
        expect(page.locator(".p-inspector")).to_have_count(0)
        page.get_by_role("button", name="Радар", exact=True).click()
        expect(
            page.get_by_role("heading", name="Safety Radar", exact=True)
        ).to_be_visible()
        page.keyboard.press("Escape")
        page.keyboard.press("f")
        expect(page.locator(".p-focus-mode")).to_be_visible()
        page.screenshot(
            path=str(args.artifacts / "observation-focus.png"), full_page=True
        )
        page.keyboard.press("Escape")
        expect(page.locator(".p-focus-mode")).to_have_count(0)
        page.get_by_role("button", name="Обозначить зону", exact=True).click()
        zone = page.get_by_role("dialog", name="Опасные зоны", exact=True)
        zone.get_by_role("button", name="Сохранить зоны").click()
        expect(zone.get_by_role("alert")).to_contain_text("не меньше трёх точек")
        canvas = zone.locator(".zone-overlay")
        box = canvas.bounding_box()
        for x, y in [(0.2, 0.7), (0.5, 0.3), (0.8, 0.7)]:
            canvas.click(position={"x": box["width"] * x, "y": box["height"] * y})
        zone.get_by_label("Название выбранной зоны").fill("Проход у станка")
        point = zone.get_by_role("button", name="Точка зоны 1")
        point.focus()
        page.keyboard.press("ArrowRight")
        page.screenshot(
            path=str(args.artifacts / "zone-editor-desktop.png"), full_page=True
        )
        zone.get_by_role("button", name="Сохранить зоны").click()
        expect(zone).not_to_be_visible(timeout=15000)
        page.get_by_role("button", name="Настройки камеры", exact=True).click()
        name = page.get_by_label("Название камеры", exact=True)
        name.fill("Несохранённое название")
        nav.get_by_role("button", name="Наблюдение", exact=True).click()
        leave = page.get_by_role("dialog", name="Сохранить изменения?", exact=True)
        expect(leave).to_be_visible()
        leave.get_by_role("button", name="Продолжить редактирование").click()
        expect(name).to_have_value("Несохранённое название")
        page.go_back()
        expect(leave).to_be_visible()
        leave.get_by_role("button", name="Продолжить редактирование").click()
        expect(name).to_have_value("Несохранённое название")
        page.get_by_role("button", name="Отменить", exact=True).click()
        name.fill(saved_name)
        nav.get_by_role("button", name="Наблюдение", exact=True).click()
        leave.get_by_role("button", name="Сохранить и перейти").click()
        expect(
            page.get_by_role("heading", name="Наблюдение", exact=True)
        ).to_be_visible()
        page.get_by_role("button", name=f"Камера {saved_name}", exact=True).click()
        page.get_by_role("button", name="Остановить", exact=True).click()
        expect(page.locator(".p-observation-status strong")).to_have_text(
            "Наблюдение остановлено", timeout=15000
        )
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(args.artifacts / "observation-mobile.png"), full_page=True
        )
        for title in ["Наблюдение", "Камеры", "Инциденты"]:
            nav.get_by_role("button", name=title, exact=True).click()
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()
            assert page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ), title
            page.screenshot(
                path=str(args.artifacts / f"mobile-{title}.png"), full_page=True
            )
        page.get_by_role("button", name="Фильтры").click()
        expect(page.get_by_label("С даты", exact=True)).to_be_visible()
        page.get_by_role("button", name="Система", exact=True).click()
        expect(page.get_by_role("heading", name="Система", exact=True)).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        browser.close()
    print(
        "Workspace E2E: PASS (source wizard, real ONNX, optional panels, focus, zones, unsaved settings, mobile)"
    )


if __name__ == "__main__":
    main()
