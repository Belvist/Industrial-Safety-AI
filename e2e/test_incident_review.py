"""Browser regression of evidence/review/export, with isolated labelled fixtures.

Starts its own temporary backend. The generated picture and H.264 clip are
explicit test fixtures, not detection, incident-accuracy or TPU evidence.
"""

import argparse
import csv
import io
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from app.core.config import Settings
from app.main import create_app
from app.services.runtime import Runtime
from PIL import Image, ImageDraw
from playwright.sync_api import expect, sync_playwright


class FixtureProvider:
    name = "test-fixture"

    def supported_classes(self):
        return ["person", "helmet", "no_helmet", "vest", "no_vest"]

    def predict_sync(self, frame):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-executable", default=None)
    parser.add_argument(
        "--artifacts", type=Path, default=Path("/tmp/incident-review-e2e")
    )
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="safety-review-fixture-") as folder:
        data = Path(folder)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{sock.getsockname()[1]}"
        token = uuid4().hex
        config = Settings(
            _env_file=None,
            api_token=token,
            data_dir=folder,
            database_url=f"sqlite:///{folder}/s.db",
            telegram_enabled=False,
            extra_origins=base,
        )
        runtime = Runtime(config, provider=FixtureProvider())
        camera_id = next(iter(runtime.sessions))
        image = Image.new("RGB", (960, 540), "#263132")
        draw = ImageDraw.Draw(image)
        draw.text(
            (35, 28),
            "TEST FIXTURE / NO REAL INCIDENT / NO TPU",
            fill="white",
            font_size=24,
        )
        draw.rectangle((260, 120, 440, 480), outline="#d3b77e", width=3)
        draw.polygon(
            [(120, 470), (800, 470), (690, 300), (220, 300)], outline="#d3b77e", width=3
        )
        snapshot = "fixture.jpg"
        image.save(data / "snapshots" / snapshot)
        evidence = f"radar_{camera_id}_fixture.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(data / "snapshots" / snapshot),
                "-t",
                "2",
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-movflags",
                "+faststart",
                str(data / "evidence" / evidence),
            ],
            check=True,
        )
        records = [
            runtime.storage.create_incident(
                camera_id,
                "TEST FIXTURE · Цех",
                (i, kind),
                0.91,
                2.3,
                snapshot=snapshot,
                evidence=evidence,
                zone_name="TEST FIXTURE · Проход",
            )
            for i, kind in [(1, "danger_zone"), (2, "no_helmet")]
        ]
        server = uvicorn.Server(
            uvicorn.Config(create_app(config, runtime=runtime), log_level="error")
        )
        thread = threading.Thread(
            target=lambda: server.run(sockets=[sock]), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    executable_path=args.browser_executable, args=["--no-sandbox"]
                )
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"{base}/?token={token}#incidents")
                rows = page.locator(".p-incident-table tbody tr")
                expect(rows).to_have_count(2)
                rows.first.get_by_role("button").click()
                dialog = page.get_by_role("dialog")
                expect(
                    dialog.get_by_role("button", name="Видеодоказательство")
                ).to_be_visible()
                page.wait_for_function(
                    "document.querySelector('video')?.readyState >= 2"
                )
                assert page.locator("video").get_attribute("src").startswith("blob:")
                page.locator("video").evaluate("video => video.play()")
                page.wait_for_function(
                    "document.querySelector('video')?.currentTime > 0.1"
                )
                page.screenshot(
                    path=str(args.artifacts / "evidence-desktop.png"), full_page=True
                )
                dialog.get_by_role("button", name="Снимок", exact=True).click()
                expect(dialog.locator("img")).to_be_visible()
                with page.expect_download() as download:
                    dialog.get_by_role("link", name="Скачать", exact=True).click()
                assert Path(download.value.path()).stat().st_size > 100
                dialog.get_by_role(
                    "button", name="Подтвердить нарушение", exact=True
                ).click()
                expect(dialog.get_by_text("Подтверждено · сохранено")).to_be_visible()
                assert (
                    sum(
                        runtime.storage.get(r["id"])["feedback"] == "confirmed"
                        for r in records
                    )
                    == 1
                )
                assert all(
                    runtime.storage.get(r["id"])["status"] == "active" for r in records
                )
                dialog.get_by_role("button", name="Следующее событие").click()
                dialog.get_by_role(
                    "button", name="Ложное срабатывание", exact=True
                ).click()
                expect(
                    dialog.get_by_text("Ложное срабатывание · сохранено")
                ).to_be_visible()
                page.keyboard.press("Escape")
                expect(
                    page.get_by_role("heading", name="Нет событий для проверки")
                ).to_be_visible()
                page.get_by_role("button", name="Все события", exact=True).click()
                expect(rows).to_have_count(2)
                with page.expect_download() as export:
                    page.get_by_role("button", name="Скачать CSV", exact=True).click()
                text = Path(export.value.path()).read_text(encoding="utf-8-sig")
                assert len(list(csv.reader(io.StringIO(text), delimiter=";"))) == 3
                assert "Подтверждено" in text and "Ложное срабатывание" in text
                page.screenshot(
                    path=str(args.artifacts / "journal-desktop.png"), full_page=True
                )
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                rows.first.get_by_role("button").click()
                expect(dialog).to_be_visible()
                page.wait_for_function(
                    "document.querySelector('video')?.readyState >= 2"
                )
                page.screenshot(
                    path=str(args.artifacts / "evidence-mobile.png"), full_page=True
                )
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                )
                page.keyboard.press("Escape")
                page.screenshot(
                    path=str(args.artifacts / "journal-mobile.png"), full_page=True
                )
                # Failed history requests must not become a false 'no incidents'.
                page.route(
                    "**/api/v1/incidents?**",
                    lambda route: route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"detail":"Test journal outage"}',
                    ),
                )
                page.get_by_role("button", name="На проверке", exact=True).click()
                expect(
                    page.get_by_role("heading", name="Журнал временно недоступен")
                ).to_be_visible()
                expect(
                    page.get_by_role("heading", name="Нет событий для проверки")
                ).to_have_count(0)
                assert not errors, errors
                browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            sock.close()
    print(
        "Incident review E2E: PASS (isolated labelled fixtures, H.264 playback, snapshot, verdicts, CSV, mobile, outage)"
    )


if __name__ == "__main__":
    main()
