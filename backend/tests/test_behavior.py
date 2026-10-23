"""Scenario 01 (running) trajectory rule and its pipeline integration."""

import numpy as np

from app.core.config import Settings
from app.domain.models import BoundingBox, Detection
from app.services.behavior import RunningDetector
from app.services.engine import Track
from app.services.runtime import Runtime
from app.services.schemas import CameraConfig


def _track(track_id, x, height=0.3):
    return Track(
        track_id,
        BoundingBox(x, 0.2, x + height * 0.4, 0.2 + height),
        0.0,
        0.9,
    )


def _walk(detector, xs, height=0.3, step=0.1, threshold=1.8):
    observed = {}
    cleared = {}
    for index, x in enumerate(xs):
        seen, gone = detector.update(
            [_track(1, x, height)], index * step, threshold, 0.4
        )
        observed.update(seen)
        cleared.update(gone)
    return observed, cleared


def test_running_is_detected_after_sustained_motion():
    detector = RunningDetector()
    xs = [0.05 + 0.02 * i for i in range(20)]
    observed, _ = _walk(detector, xs, threshold=0.5)
    assert (1, "running") in observed
    assert observed[(1, "running")] >= 0.5


def test_normal_walk_never_fires_and_reports_cleared():
    detector = RunningDetector()
    xs = [0.05 + 0.005 * i for i in range(20)]
    observed, cleared = _walk(detector, xs)
    assert not observed
    assert (1, "running") in cleared


def test_shaky_camera_jitter_is_not_running():
    detector = RunningDetector()
    xs = [0.3 + (0.03 if i % 2 else -0.03) for i in range(20)]
    observed, _ = _walk(detector, xs, threshold=0.5)
    assert not observed


def test_single_fast_frame_does_not_fire():
    detector = RunningDetector()
    observed = {}
    positions = [0.05, 0.06, 0.5, 0.5]
    for index, x in enumerate(positions):
        seen, _ = detector.update([_track(1, x)], index * 0.1, 1.8, 0.4)
        observed.update(seen)
    assert not observed


def test_disappeared_track_is_forgotten():
    detector = RunningDetector()
    detector.update([_track(1, 0.1)], 0.0, 1.8, 0.4)
    detector.update([], 10.0, 1.8, 0.4)
    assert detector.diagnostics() == {}


class MovingPersonProvider:
    """Person box moving right; no PPE classes, so only behaviour can fire."""

    def __init__(self):
        self.step = 0

    def predict_sync(self, frame):
        x = 0.05 + 0.02 * self.step
        self.step += 1
        return [Detection("person", 0.95, BoundingBox(x, 0.2, x + 0.12, 0.5))]


def test_running_creates_one_incident_in_pipeline(tmp_path):
    config = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/s.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )
    runtime = Runtime(config, MovingPersonProvider())
    try:
        camera = next(iter(runtime.sessions.values()))
        camera.config = CameraConfig(
            name="Цех",
            helmet=False,
            vest=False,
            running_enabled=True,
            running_speed=0.5,
            running_min_seconds=0.8,
            confirm_seconds=0.2,
            clear_seconds=0.2,
            temporal_window=2,
            temporal_required=1,
        )
        runtime.start(camera.id, "browser")
        frame = np.zeros((240, 320, 3), np.uint8)
        for index in range(30):
            runtime.process(camera, frame, index * 0.1)
        incidents = runtime.storage.incidents(types=["running"])["items"]
        assert len(incidents) == 1
        assert incidents[0]["status"] == "active"
        assert incidents[0]["type"] == "running"
    finally:
        runtime.close()
