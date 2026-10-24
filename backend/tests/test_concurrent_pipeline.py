"""Concurrency invariants against a controlled AMQP peer, never hardware FPS proof."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.inference.linq import LinqProvider
from app.inference.tpu_wire import PyTask, deserialize, serialize
from app.main import create_app
from app.services.runtime import Runtime
from app.services.schemas import CameraConfig


class Peer:
    def __init__(self):
        self.callback = None
        self.replies = []
        self.ids = []
        self.threads = set()
        self.maximum = 0
        self.drop = False
        self.fail = False
        self.is_open = True

    def thread(self):
        self.threads.add(threading.current_thread().name)
        assert self.threads == {"linq-amqp-io"}

    def basic_consume(self, queue, on_message_callback, auto_ack):
        self.thread()
        self.callback = on_message_callback

    def basic_publish(self, exchange, routing_key, body, properties):
        self.thread()
        if self.fail:
            self.fail = False
            raise ConnectionError("controlled disconnect")
        data, _, _ = deserialize(body)
        marker = float(data["images"].flat[0])
        task_id = properties.headers["tpu_pycloud_message_id"]
        self.ids.append(task_id)
        output = np.full((1, 18, 8400), marker, np.float32)
        reply = serialize(
            PyTask(
                {"output0": output},
                status_code=0,
                inference_time=0.074,
                worker_pid="controlled-peer",
            )
        )
        self.replies.append((time.monotonic() + 0.025, task_id, reply))
        self.maximum = max(self.maximum, len(self.replies))

    def process_data_events(self, time_limit=0):
        self.thread()
        if self.drop:
            return
        # Intentionally complete newest IDs first.
        for ready, key, body in list(reversed(self.replies)):
            if time.monotonic() >= ready:
                self.replies.remove((ready, key, body))
                self.callback(
                    self, None, SimpleNamespace(headers={"tpu_pycloud_message_id": key}), body
                )

    def close(self):
        self.thread()
        self.is_open = False


@pytest.fixture
def wired(monkeypatch):
    providers = []

    def make(limit=4, timeout=1):
        provider = LinqProvider(
            amqp_host="controlled-peer", max_in_flight=limit, timeout_seconds=timeout
        )
        provider.load()
        peer = Peer()

        def connect():
            peer.thread()
            peer.is_open = True
            provider._connection = peer
            provider._channel = peer
            provider._reply_queue = "test-replies"
            return peer

        monkeypatch.setattr(provider, "_connect", connect)
        providers.append(provider)
        return provider, peer

    yield make
    for provider in providers:
        provider.close()


@pytest.mark.parametrize("limit", [1, 2, 4, 6, 8, 12])
def test_out_of_order_replies_are_correlated_and_bounded(wired, limit):
    provider, peer = wired(limit)
    with ThreadPoolExecutor(max_workers=limit) as pool:
        results = list(
            pool.map(
                lambda value: provider._send_tensor(np.full((1, 3, 8, 8), value, np.float32))[
                    0, 0, 0
                ],
                range(36),
            )
        )
    assert results == list(range(36))
    assert len(set(peer.ids)) == 36
    assert peer.maximum <= limit
    assert provider.stats.remote_calls == 36
    assert provider.stats.remote_failures == 0
    assert not provider._transport.pending
    assert peer.threads == {"linq-amqp-io"}


def test_timeout_late_reply_and_reconnect(wired):
    provider, peer = wired(2, 0.08)
    peer.drop = True
    with pytest.raises(RuntimeError, match="не ответил"):
        provider._send_tensor(np.zeros((1, 3, 8, 8), np.float32))
    peer.drop = False
    assert provider._send_tensor(np.ones((1, 3, 8, 8), np.float32))[0, 0, 0] == 1
    peer.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        provider._send_tensor(np.zeros((1, 3, 8, 8), np.float32))
    assert provider._send_tensor(np.full((1, 3, 8, 8), 2, np.float32))[0, 0, 0] == 2
    assert provider.stats.remote_failures == 2
    assert provider.stats.remote_calls == 2


def test_cpu_person_and_tpu_share_frame_and_run_in_parallel(monkeypatch):
    provider = LinqProvider(amqp_host="controlled-peer")
    provider.load()
    barrier = threading.Barrier(2)
    frame = np.zeros((32, 32, 3), np.uint8)

    def branch(actual):
        assert actual is frame
        barrier.wait(timeout=1)
        return []

    provider._person = SimpleNamespace(predict_sync=branch)
    monkeypatch.setattr(provider, "_predict_remote", branch)
    try:
        assert provider.predict_sync(frame) == []
    finally:
        provider.close()


def settings(tmp_path, **kw):
    return Settings(
        _env_file=None, data_dir=str(tmp_path), database_url=f"sqlite:///{tmp_path}/db.sqlite", **kw
    )


class SlowProvider:
    name = "test"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_sync(self, frame):
        self.entered.set()
        assert self.release.wait(2)
        return []


def test_stop_and_rule_change_discard_late_results_without_blocking_state(tmp_path):
    provider = SlowProvider()
    runtime = Runtime(settings(tmp_path), provider)
    cam = next(iter(runtime.sessions.values()))
    runtime.start(cam.id, "browser")
    old = cam.session_id
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(runtime.process, cam, np.zeros((32, 32, 3), np.uint8), 1, old)
        assert provider.entered.wait(1)
        assert cam.summary()["session_id"] == old  # state lock isn't held by inference
        runtime.stop(cam.id)
        runtime.start(cam.id, "browser")
        provider.release.set()
        with pytest.raises(RuntimeError, match="Устаревший"):
            result.result()
    assert cam.latest is None
    assert runtime.storage.incidents()["total"] == 0
    runtime.close()


def test_info_redaction_owner_integrity_and_polygon_validation(tmp_path):
    runtime = Runtime(settings(tmp_path, api_token="operator"), SlowProvider())
    with TestClient(create_app(runtime.settings, runtime)) as client:
        runtime.storage.create_incident("private", "Camera", (1, "no_helmet"), 0.9, 2)
        public = client.get("/api/v1/system/info")
        assert public.json()["stats"]["total"] == 0
        assert public.json()["inference"] is None
        assert public.headers["cache-control"] == "no-store"
        headers = {"X-API-Token": "operator"}
        response = client.post(
            "/api/v1/cameras", headers=headers, json={"name": "test", "owner": "phone:forged"}
        )
        assert response.status_code == 201
        assert response.json()["owner"] is None
        cid = response.json()["id"]
        runtime.sessions[cid].config.owner = "phone:original"
        response = client.put(f"/api/v1/cameras/{cid}", headers=headers, json={"name": "changed"})
        assert response.json()["owner"] == "phone:original"
        other_name = f"radar_{cid}_123_abc.mp4"
        (runtime.data / "evidence" / other_name).write_bytes(b"private")
        first = next(iter(runtime.sessions))
        assert (
            client.get(
                f"/api/v1/cameras/{first}/evidence/{other_name}", headers=headers
            ).status_code
            == 404
        )
    valid = [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(ValueError):
        CameraConfig(
            zones=[
                {"name": "valid", "points": valid},
                {"name": "bad", "points": [[0, 0], [1, 1], [0, 1], [1, 0]]},
            ]
        )


def test_two_workers_same_zone_create_independent_incidents(tmp_path, monkeypatch):
    from app.domain.models import BoundingBox, Detection

    provider = SimpleNamespace(
        name="two-workers",
        predict_sync=lambda _: [
            Detection("person", 0.99, BoundingBox(0.2, 0.2, 0.35, 0.7)),
            Detection("person", 0.99, BoundingBox(0.6, 0.2, 0.75, 0.7)),
        ],
    )
    runtime = Runtime(settings(tmp_path), provider)
    monkeypatch.setattr(runtime, "_save_evidence", lambda *_: None)
    cam = next(iter(runtime.sessions.values()))
    cam.config = CameraConfig(
        zone_enabled=True,
        zone=[(0, 0), (1, 0), (1, 1), (0, 1)],
        confirm_seconds=0.2,
        temporal_required=1,
    )
    runtime.start(cam.id, "browser")
    try:
        for tick in range(8):
            runtime.process(cam, np.zeros((64, 64, 3), np.uint8), tick * 0.2)
        records = runtime.storage.incidents()["items"]
        assert len(records) == 2
        assert {r["track_id"] for r in records} == {1, 2}
        assert all(r["status"] == "active" for r in records)
    finally:
        runtime.close()


def test_video_pipeline_commits_out_of_order_inference_in_source_order(tmp_path, monkeypatch):
    import cv2

    source = tmp_path / "order.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    for value in range(10, 110, 10):
        writer.write(np.full((64, 64, 3), value, np.uint8))
    writer.release()
    completions, commits = [], []

    def predict(frame):
        value = int(frame[0, 0, 0])
        time.sleep(0.08 if value < 15 else 0.002)
        completions.append(value)
        return []

    runtime = Runtime(
        settings(tmp_path, linq_max_in_flight=4), SimpleNamespace(name="linq", predict_sync=predict)
    )
    original = runtime._commit

    def commit(cam, frame, detections, timestamp, captured, ms):
        commits.append(int(frame[0, 0, 0]))
        return original(cam, frame, detections, timestamp, captured, ms)

    monkeypatch.setattr(runtime, "_commit", commit)
    cam = next(iter(runtime.sessions.values()))
    runtime.start(cam.id, "video", str(source))
    try:
        cam.thread.join(timeout=3)
        assert not cam.thread.is_alive()
        assert completions[0] != min(completions)
        assert len(commits) == 10 and commits == sorted(commits) and len(set(commits)) == 10
        assert cam.sequence == 10
    finally:
        runtime.close()


def test_h264_evidence_codec_and_source_session_isolation(tmp_path):
    import json
    import shutil
    import subprocess

    import cv2

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("H.264 verification requires ffmpeg and ffprobe")
    runtime = Runtime(settings(tmp_path), SimpleNamespace(name="test"))
    cam = next(iter(runtime.sessions.values()))
    _, image = cv2.imencode(".jpg", np.full((64, 64, 3), 80, np.uint8))
    frames = [(i, image.tobytes(), image.tobytes()) for i in range(10)]
    name = f"radar_{cam.id}_codec_test.mp4"
    try:
        runtime._write_evidence(name, frames, cam, expected_session="old-session")
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,nb_frames",
                "-of",
                "json",
                str(runtime.data / "evidence" / name),
            ]
        )
        stream = json.loads(output)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert int(stream["nb_frames"]) == 10
    finally:
        runtime.close()


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_CONTROLLED_SOAK") != "1",
    reason="opt-in 60 second controlled AMQP soak, not real hardware",
)
def test_controlled_amqp_soak_60_seconds(wired):
    import json
    import os
    from pathlib import Path

    provider, peer = wired(4)
    tensor = np.zeros((1, 3, 8, 8), np.float32)
    started = time.monotonic()
    successes = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        while time.monotonic() - started < 60:
            futures = [pool.submit(provider._send_tensor, tensor) for _ in range(4)]
            for future in futures:
                assert future.result()[0, 0, 0] == 0
                successes += 1
            time.sleep(0.1)
    elapsed = time.monotonic() - started
    report = {
        "peer": "controlled_fake_not_hardware",
        "real_worker_verified": False,
        "seconds": elapsed,
        "successful_replies": successes,
        "replies_per_second": successes / elapsed,
        "in_flight": 4,
        "peak_pending": peer.maximum,
        "pending_at_end": len(provider._transport.pending),
        "failures": provider.stats.remote_failures,
        "metrics": provider.metrics.snapshot(),
    }
    if os.environ.get("CONTROLLED_SOAK_REPORT"):
        Path(os.environ["CONTROLLED_SOAK_REPORT"]).write_text(json.dumps(report, indent=2))
    assert elapsed >= 60 and successes > 100
    assert not provider._transport.pending
    assert provider.stats.remote_failures == 0


def test_concurrent_browser_frames_keep_capture_order_and_old_ids_are_discarded(tmp_path):
    entered = threading.Event()
    completed, committed = [], []

    def predict(frame):
        marker = int(frame[0, 0, 0])
        if marker == 1:
            entered.set()
            time.sleep(0.08)
        completed.append(marker)
        return []

    runtime = Runtime(
        settings(tmp_path, linq_max_in_flight=2), SimpleNamespace(name="linq", predict_sync=predict)
    )
    cam = next(iter(runtime.sessions.values()))
    runtime.start(cam.id, "browser")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                runtime.process,
                cam,
                np.full((32, 32, 3), 1, np.uint8),
                source_frame_id=1,
                source_elapsed=0.2,
            )
            assert entered.wait(1)
            second = pool.submit(
                runtime.process,
                cam,
                np.full((32, 32, 3), 2, np.uint8),
                source_frame_id=2,
                source_elapsed=0.4,
            )
            committed = [first.result()["sequence"], second.result()["sequence"]]
        assert completed == [2, 1]
        assert committed == [1, 2]
        assert cam.sequence == 2
        with pytest.raises(RuntimeError, match="Устаревший"):
            runtime.process(
                cam, np.zeros((32, 32, 3), np.uint8), source_frame_id=1, source_elapsed=0.2
            )
        assert cam.sequence == 2 and cam.status == "running"
    finally:
        runtime.close()


def test_huge_png_rejected_before_decoder(tmp_path, monkeypatch):
    import struct

    import cv2

    runtime = Runtime(settings(tmp_path), SimpleNamespace(name="test", predict_sync=lambda _: []))
    with TestClient(create_app(runtime.settings, runtime)) as client:
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        sid = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"}).json()[
            "session_id"
        ]

        def forbidden(*_):
            raise AssertionError("oversize image reached the pixel decoder")

        monkeypatch.setattr(cv2, "imdecode", forbidden)
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 100000, 100000)
        response = client.post(
            f"/api/v1/cameras/{cid}/frame", content=raw, headers={"X-Session-Id": sid}
        )
        assert response.status_code == 413
        assert runtime.sessions[cid].status == "running"


def test_replay_preserves_slow_frame_timing_and_event_frame():
    from app.services.replay import replay_frames

    frames = [(index, b"jpeg", b"jpeg", float(index)) for index in range(8)]
    output = replay_frames(frames, 5, event_time=5)
    assert len(output) == 21  # 4 seconds at 5 fps, not 4/5 seconds
    assert output[0][0] == 2 and output[-1][0] == 6
    assert sum(frame[0] == 5 for frame in output) == 5


def test_evidence_is_not_public_until_encoding_completes(tmp_path, monkeypatch):
    """A browser must never fetch the temporary MPEG-4 file before H.264 transcode."""
    import threading

    import cv2

    runtime = Runtime(settings(tmp_path), SimpleNamespace(name="test"))
    cam = next(iter(runtime.sessions.values()))
    name = f"radar_{cam.id}_atomic.mp4"
    image = np.zeros((64, 64, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    frames = [(1, encoded.tobytes(), encoded.tobytes(), 0.0)]
    entered, release = threading.Event(), threading.Event()
    factory = cv2.VideoWriter

    class HeldWriter:
        def __init__(self, *args):
            self.inner = factory(*args)

        def isOpened(self):
            return self.inner.isOpened()

        def write(self, frame):
            entered.set()
            release.wait(3)
            self.inner.write(frame)

        def release(self):
            self.inner.release()

    monkeypatch.setattr(cv2, "VideoWriter", HeldWriter)
    thread = threading.Thread(target=runtime._write_evidence, args=(name, frames, cam))
    thread.start()
    try:
        assert entered.wait(2.5)
        assert not (runtime.data / "evidence" / name).exists()
    finally:
        release.set()
        thread.join(timeout=5)
        runtime.close()
    assert not thread.is_alive()
    assert (runtime.data / "evidence" / name).is_file()
