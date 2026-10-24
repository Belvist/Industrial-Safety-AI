"""Real licensed video -> real detector -> UI warning -> incident -> H.264.

Unlike the other browser regressions, this test does not seed detections or
incidents. Run only against an isolated local backend. Camera configuration
is an explicit demo boundary, not a claim about the video's workplace rules.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--browser-executable", default=None)
    p.add_argument(
        "--record-video",
        action="store_true",
        help="Record the real browser journey as WebM",
    )
    args = p.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    def api(path, method="GET", body=None, raw=False):
        req = urllib.request.Request(
            args.base + "/api/v1" + path,
            method=method,
            headers={
                "X-API-Token": args.master,
                "Content-Type": "application/octet-stream"
                if raw
                else "application/json",
            },
            data=body
            if raw
            else json.dumps(body).encode()
            if body is not None
            else None,
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    config = json.loads(args.config.read_text())
    camera = api("/cameras", "POST", config)
    cid = camera["id"]
    observations = []
    errors = []
    screenshots = set()
    displayed = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=args.browser_executable, args=["--no-sandbox"]
            )
            page = browser.new_page(
                viewport={"width": 1440, "height": 960},
                record_video_dir=str(args.artifacts / "screen-recording")
                if args.record_video
                else None,
                record_video_size={"width": 1440, "height": 960}
                if args.record_video
                else None,
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.base + "/?token=" + args.master + "#monitor/" + cid)
            expect(
                page.get_by_role("heading", name=config["name"], exact=True)
            ).to_be_visible()
            api(f"/cameras/{cid}/video", "POST", args.video.read_bytes(), True)
            page.get_by_role("button", name="Радар", exact=True).click()
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                view = api(f"/cameras/{cid}")
                frame = view.get("latest")
                if frame:
                    if (
                        not observations
                        or observations[-1]["sequence"] != frame["sequence"]
                    ):
                        observations.append(frame)
                    levels = set(frame.get("radar", {}).get("track_risks", {}).values())
                    for level in ("WARNING", "CRITICAL"):
                        if level in levels and level not in screenshots:
                            expect(
                                page.locator(".p-radar-tracks ." + level.lower())
                            ).to_be_visible(timeout=5000)
                            displayed[level] = time.monotonic()
                            page.screenshot(
                                path=str(args.artifacts / f"{level.lower()}.png"),
                                full_page=True,
                            )
                            screenshots.add(level)
                    if (
                        screenshots == {"WARNING", "CRITICAL"}
                        and frame.get("source_timestamp", 0) >= 10
                    ):
                        break
                page.wait_for_timeout(50)
            assert screenshots == {"WARNING", "CRITICAL"}, screenshots
            incidents = api(f"/incidents?camera_id={cid}")["items"]
            assert len(incidents) == 1, incidents
            incident = incidents[0]
            timeline = api(f"/incidents/{incident['id']}/timeline")["items"]
            warning = next(e for e in timeline if e["kind"] == "warning")
            critical = next(e for e in timeline if e["kind"] == "critical")
            lead = critical["source_seconds"] - warning["source_seconds"]
            assert lead > 0
            toggle = page.get_by_role("button", name="Показать навигацию", exact=True)
            if toggle.is_visible():
                toggle.click()
            page.get_by_role("navigation", name="Основная навигация").get_by_role(
                "button", name="Инциденты", exact=True
            ).click()
            page.get_by_role("button", name="Все события", exact=True).click()
            page.get_by_role("button", name="Фильтры", exact=False).click()
            page.get_by_label("Камера", exact=True).select_option(cid)
            row = page.locator(".p-incident-table tbody tr").filter(
                has_text=config["name"]
            )
            expect(row).to_have_count(1)
            row.get_by_role("button").click()
            dialog = page.get_by_role("dialog", name="Опасная зона", exact=True)
            expect(dialog).to_be_visible()
            dialog.locator("summary").filter(
                has_text="Временная шкала происшествия"
            ).click()
            expect(
                dialog.get_by_text("Раннее предупреждение", exact=True)
            ).to_be_visible()
            video = dialog.locator("video")
            expect(video).to_be_visible(timeout=20000)
            video.evaluate("(v) => v.play()")
            page.wait_for_function(
                'document.querySelector("[role=dialog] video")?.currentTime > 0.2'
            )
            dialog.get_by_text(
                "Раннее предупреждение", exact=True
            ).scroll_into_view_if_needed()
            page.screenshot(
                path=str(args.artifacts / "incident-timeline.png"), full_page=True
            )
            dialog.get_by_role(
                "button", name="Подтвердить нарушение", exact=True
            ).click()
            expect(
                dialog.get_by_text("Подтверждено · сохранено", exact=False)
            ).to_be_visible()
            assert (
                api(f"/incidents/{incident['id']}/timeline")["items"][-1]["kind"]
                == "operator_review"
            )
            timeline = api(f"/incidents/{incident['id']}/timeline")["items"]
            incident = next(
                row
                for row in api(f"/incidents?camera_id={cid}")["items"]
                if row["id"] == incident["id"]
            )
            assert incident["feedback"] == "confirmed"
            assert not errors, errors
            report = {
                "status": "PASS",
                "mode": api("/system/info")["inference"],
                "lead_time_seconds": lead,
                "browser_display_interval_seconds": displayed["CRITICAL"]
                - displayed["WARNING"],
                "incident": incident,
                "timeline": timeline,
                "errors": errors,
                "source": "real video, no seeded detections/incidents",
            }
            (args.artifacts / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False)
            )
            (args.artifacts / "frames.json").write_text(
                json.dumps(observations, indent=2, ensure_ascii=False)
            )
            browser.close()
    finally:
        api(f"/cameras/{cid}/stop", "POST")
    print(
        "Real-video browser E2E PASS; measured warning lead:",
        round(lead, 4),
        "seconds; LinQ:",
        report["mode"].get("worker_verified", False),
    )


if __name__ == "__main__":
    main()
