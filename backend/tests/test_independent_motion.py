"""Controlled synchronization tests. Fake detections are never video acceptance evidence."""

import threading
import time

import numpy as np

from app.core.config import Settings
from app.domain.models import BoundingBox, Detection
from app.services.runtime import Runtime


class SplitProvider:
    name = "controlled-split"

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.people = True

    def motion_capable(self):
        return True

    def predict_people(self, frame):
        x = float(frame[0, 0, 0]) / 1000
        return (
            [Detection("person", 0.95, BoundingBox(0.15 + x, 0.1, 0.45 + x, 0.9))]
            if self.people
            else []
        )

    def predict_ppe(self, frame, people):
        self.started.set()
        self.release.wait(3)
        p = people[0].bbox
        return [
            Detection(
                "helmet", 0.99, BoundingBox(p.x1 + 0.02, p.y1 + 0.01, p.x1 + 0.15, p.y1 + 0.2)
            )
        ]


def runtime(tmp_path, provider):
    return Runtime(
        Settings(
            _env_file=None,
            data_dir=str(tmp_path),
            database_url=f"sqlite:///{tmp_path}/s.db",
            telegram_enabled=False,
        ),
        provider,
    )


def image(marker=0):
    frame = np.random.default_rng(5).integers(60, 200, size=(240, 320, 3), dtype=np.uint8)
    frame[0, 0, 0] = marker
    return frame


def test_delayed_ppe_does_not_block_motion_and_retains_original_age(tmp_path):
    provider = SplitProvider()
    rt = runtime(tmp_path, provider)
    cam = next(iter(rt.sessions.values()))
    rt.start(cam.id, "browser")
    try:
        begin = time.perf_counter()
        for i in range(8):
            latest = rt.process(cam, image(i * 10), i * 0.2)
        assert time.perf_counter() - begin < 1.0
        assert latest["metrics"]["rates"]["person"]["total"] == 8
        assert latest["metrics"]["rates"]["radar"]["total"] == 8
        assert latest["ppe"]["pending"]
        assert latest["tracks"][0]["ppe"]["helmet"] == "unknown"
        assert provider.started.wait(0.5)
        provider.release.set()
        next(iter(cam.independent_ppe.pending.values()))[0].result(timeout=1)
        latest = rt.process(cam, image(80), 1.6)
        assert latest["tracks"][0]["ppe"]["helmet"] == "present"
        assert latest["tracks"][0]["ppe_age_seconds"] == 1.6
        assert all(d["class_name"] == "person" for d in latest["detections"])
    finally:
        provider.release.set()
        rt.close()


def test_late_ppe_after_a_missing_person_cannot_attach_to_reappearing_track(tmp_path):
    provider = SplitProvider()
    rt = runtime(tmp_path, provider)
    cam = next(iter(rt.sessions.values()))
    rt.start(cam.id, "browser")
    try:
        rt.process(cam, image(), 0)
        assert provider.started.wait(0.5)
        provider.people = False
        rt.process(cam, image(), 0.2)
        provider.people = True
        rt.process(cam, image(), 0.4)
        provider.release.set()
        next(iter(cam.independent_ppe.pending.values()))[0].result(timeout=1)
        latest = rt.process(cam, image(), 0.6)
        assert latest["tracks"][0]["ppe"]["helmet"] == "unknown"
    finally:
        provider.release.set()
        rt.close()


def test_stopped_generation_cannot_commit_a_pending_ppe_result(tmp_path):
    provider = SplitProvider()
    rt = runtime(tmp_path, provider)
    cam = next(iter(rt.sessions.values()))
    rt.start(cam.id, "browser")
    try:
        rt.process(cam, image(), 0)
        assert provider.started.wait(0.5)
        rt.stop(cam.id)
        rt.start(cam.id, "browser")
        provider.release.set()
        latest = rt.process(cam, image(), 1)
        assert latest["ppe"]["source_timestamp"] is None
        assert not rt.storage.incidents()["items"]
    finally:
        provider.release.set()
        rt.close()


def test_ppe_failure_leaves_person_and_radar_available(tmp_path):
    class Failure(SplitProvider):
        def predict_ppe(self, *args):
            raise RuntimeError("controlled remote failure")

    provider = Failure()
    rt = runtime(tmp_path, provider)
    cam = next(iter(rt.sessions.values()))
    rt.start(cam.id, "browser")
    try:
        rt.process(cam, image(), 0)
        future = next(iter(cam.independent_ppe.pending.values()))[0]
        try:
            future.result(timeout=1)
        except RuntimeError:
            pass
        latest = rt.process(cam, image(), 0.2)
        assert latest["sequence"] == 2 and cam.status == "running"
        assert latest["metrics"]["rates"]["ppe_failures"]["total"] == 1
        assert latest["ppe"]["state"] == "unavailable"
        assert latest["tracks"][0]["ppe"]["helmet"] == "unknown"
    finally:
        rt.close()


def test_restart_does_not_attribute_old_frame_inference_to_new_session(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    from app.services.runtime import DiscardedFrame

    provider = SplitProvider()
    provider.release.set()
    rt = runtime(tmp_path, provider)
    cam = next(iter(rt.sessions.values()))
    rt.start(cam.id, "browser")
    old = cam.metrics
    entered, release = threading.Event(), threading.Event()
    mark = old.mark

    def paused_capture(name, count=1):
        if name == "capture":
            entered.set()
            release.wait(2)
        mark(name, count)

    old.mark = paused_capture
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rt.process, cam, image(), 0)
            assert entered.wait(0.5)
            rt.stop(cam.id)
            rt.start(cam.id, "browser")
            new = cam.metrics
            assert new is not old
            release.set()
            with pytest.raises(DiscardedFrame):
                future.result(timeout=1)
            assert all(
                new.snapshot()["rates"].get(k, {}).get("total", 0) == 0
                for k in ["capture", "person", "analytics"]
            )
            assert old.snapshot()["rates"]["person"]["total"] == 1
    finally:
        release.set()
        rt.close()
