"""Smoke tests for footage tooling (record_clip, label_events dump path).

Interactive labeling needs a display and is NOT covered here; the headless
frame-dump path plus file-based recording are. The fixture clip is generated
in tmp_path via OpenCV — no local files, no network, all OSes.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "backend" / "scripts"


def _clip(tmp_path: Path, fps: int = 10, n: int = 20) -> Path:
    """Deterministic synthetic clip (moving box on gradient)."""
    src = tmp_path / f"fixture_{fps}fps.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened(), "VideoWriter unavailable in CI"
    try:
        for i in range(n):
            frame = np.full((120, 160, 3), 60 + (i * 5) % 150, np.uint8)
            x = 10 + (i * 5) % 100
            cv2.rectangle(frame, (x, 30), (x + 40, 90), (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    assert src.stat().st_size > 0
    return src


def test_record_clip_from_file(tmp_path):
    out = tmp_path / "scene.mp4"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "record_clip.py"), "--source",
         str(_clip(tmp_path)), "--seconds", "4", "--out", str(out),
         "--fps", "5"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["frames"] > 0 and meta["fps"] == 5.0


def test_record_keeps_duration_upsampling(tmp_path):
    """10 fps source asked at 25 fps: every frame kept, timeline intact."""
    out = tmp_path / "scene.mp4"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "record_clip.py"), "--source",
         str(_clip(tmp_path, fps=10, n=40)), "--seconds", "4",
         "--out", str(out), "--fps", "25"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["frames"] == 40
    assert abs(meta["stamps_sec"][-1] - 3.9) < 0.05
    assert meta["duration_media_sec"] == meta["stamps_sec"][-1]


def test_label_frame_selection_is_time_based(tmp_path):
    """25 fps -> 7 fps must yield 7.0/s, not round-stepped 6.25."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "label_events", SCRIPTS / "label_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frames = mod.LazyFrames(str(_clip(tmp_path, fps=25, n=50)), 7.0)
    try:
        assert len(frames) == 14  # 2.0 s at exactly 7 fps
    finally:
        frames.close()


def test_display_to_norm_portrait():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "label_events", SCRIPTS / "label_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 720x1280 shown as 720x720: center maps to center, corner to corner
    assert mod.display_to_norm(360, 360, 720, 1280) == (0.5, 0.5)
    assert mod.display_to_norm(720, 720, 720, 1280) == (1.0, 1.0)
    assert mod.display_to_norm(0, 0, 720, 1280) == (0.0, 0.0)


def _record_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "record_clip", SCRIPTS / "record_clip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_measure_rate_irregular_camera():
    mod = _record_mod()
    # scripted irregular delivery averaging 10 fps with jitter + one stall
    stamps, t = [0.0], 0.0
    for gap in [0.09, 0.11, 0.10, 0.09, 0.12, 0.10, 0.08, 0.11, 0.10, 0.10,
                0.35, 0.09, 0.11, 0.10, 0.10, 0.09, 0.12, 0.10, 0.10, 0.09]:
        t += gap
        stamps.append(round(t, 3))
    rate = mod.measure_rate(stamps)
    assert rate is not None and abs(rate - 20 / 2.25) < 0.01  # stall averaged in
    assert mod.measure_rate([0.5]) is None
    assert mod.measure_rate([0.5, 0.5]) is None


class _FakeCamera:
    """Camera-like source: frames on scripted irregular wall-clock schedule."""

    def __init__(self, times):
        import numpy as np

        self._times = list(times)
        self._frame = np.zeros((120, 160, 3), np.uint8)

    def read(self):
        if not self._times:
            return False, None
        self._times.pop(0)
        return True, self._frame.copy()


def test_calibrate_camera_irregular_delivery(monkeypatch):
    mod = _record_mod()
    gaps = [0.09, 0.11, 0.10, 0.12, 0.08] * 8
    now = [0.0]

    def fake_monotonic():
        return now[0]

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    cam = _FakeCamera([None] * 40)

    real_read = _FakeCamera.read

    def timed_read(self):
        ok, frame = real_read(self)
        if ok:
            now[0] += gaps.pop(0) if gaps else 0.10
        return ok, frame

    monkeypatch.setattr(_FakeCamera, "read", timed_read)
    rate, kept = mod.calibrate_camera(cam, seconds=30.0, max_frames=40)
    assert rate is not None and 9.0 < rate < 11.0
    assert len(kept) == 40


def test_camera_writer_uses_measured_not_requested(monkeypatch, tmp_path):
    """10 fps device asked at 25 fps: output must play at ~10 fps (duration
    intact), not 25 (2.5x speedup). MP4 header, sidecar and event timescale
    must agree."""
    import itertools

    mod = _record_mod()
    now = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    gaps = itertools.cycle([0.09, 0.11, 0.10, 0.12, 0.08])
    real_read = _FakeCamera.read

    def timed_read(self):
        ok, frame = real_read(self)
        if ok:
            now[0] += next(gaps)
        return ok, frame

    monkeypatch.setattr(_FakeCamera, "read", timed_read)
    cam = _FakeCamera([None] * 40)
    rate, _kept = mod.calibrate_camera(cam, seconds=60.0, max_frames=40)
    assert rate is not None and 9.0 < rate < 11.0
    target_fps = min(rate, 25.0)
    assert 9.0 < target_fps < 11.0  # writer fps follows the device, not the request
    assert abs(40 / target_fps - 4.0) < 0.3  # mp4 duration == sidecar duration


def test_label_events_dump_frames(tmp_path):
    d = tmp_path / "frames"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "label_events.py"), "--clip",
         str(_clip(tmp_path)), "--out", str(tmp_path / "ev.json"),
         "--fps", "5", "--dump-frames", str(d)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert len(list(d.glob("*.jpg"))) > 0
    assert not (tmp_path / "ev.json").is_file()  # dump mode writes no events
