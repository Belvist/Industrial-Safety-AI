import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.models import BoundingBox, Detection
from app.main import create_app
from app.services.runtime import Runtime
from app.services.schemas import CameraConfig
from app.services.storage import Storage


class ControlledProvider:
    name = "test-provider"
    missing = True
    ppe_confidence = 0.95

    def predict_sync(self, frame):
        assert frame.dtype == np.uint8
        person = Detection("person", 0.99, BoundingBox(0.1, 0.1, 0.8, 0.95))
        ppe = Detection(
            "no_helmet" if self.missing else "helmet",
            self.ppe_confidence,
            BoundingBox(0.3, 0.12, 0.5, 0.25),
        )
        return [person, ppe]


class ThresholdEnforcingProvider:
    """Test double honoring the LocalOnnxProvider.decode contract.

    The real per-class filtering math is unit-tested in test_onnx_decode.py;
    this wrapper lets the full pipeline (tracker/engine/storage/API) run
    against thresholded provider output without shipping weights to CI.
    Below-floor detections are dropped; below-threshold ones are flagged weak.
    """

    def __init__(self, inner, thresholds, floor=0.25):
        self.inner = inner
        self.thresholds = thresholds
        self.floor = floor

    @property
    def name(self):
        return self.inner.name + "+thresholds"

    def predict_sync(self, frame):
        from dataclasses import replace

        kept = []
        for det in self.inner.predict_sync(frame):
            limit = self.thresholds.get(det.class_name, 0)
            if det.confidence < min(limit, self.floor):
                continue
            if det.confidence < limit:
                det = replace(det, weak=True)
            kept.append(det)
        return kept


@pytest.fixture
def runtime(tmp_path):
    config = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/safety.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )
    result = Runtime(config, ControlledProvider())
    yield result
    result.close()


def test_real_image_pipeline_persists_evidence_and_resolves(runtime):
    camera = next(iter(runtime.sessions.values()))
    runtime.start(camera.id, "browser")
    frame = np.full((240, 320, 3), 100, np.uint8)
    for timestamp in [0, 1, 2, 3, 4]:
        runtime.process(camera, frame, timestamp)
    records = runtime.storage.incidents()["items"]
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "active"
    assert record["duration"] == 4
    snapshot = runtime.data / "snapshots" / record["snapshot"]
    assert cv2.imread(str(snapshot)).shape == frame.shape
    runtime.provider.missing = False
    runtime.process(camera, frame, 5)
    runtime.process(camera, frame, 6)
    runtime.process(camera, frame, 7)
    assert runtime.storage.get(record["id"])["status"] == "resolved"
    assert runtime.storage.get(record["id"])["resolution_reason"] == "violation_cleared"
    runtime.provider.missing = True
    for timestamp in [7, 8, 9]:
        runtime.process(camera, frame, timestamp)
    assert runtime.storage.incidents()["total"] == 2
    runtime.stop(camera.id)
    assert runtime.storage.incidents()["items"][0]["status"] == "interrupted"


def test_server_restart_does_not_claim_violation_cleared(runtime):
    record = runtime.storage.create_incident("cam", "Test", (1, "no_helmet"), 0.9, 2)
    reopened = Storage(runtime.settings.storage_path)
    assert reopened.get(record["id"])["status"] == "interrupted"
    assert reopened.get(record["id"])["resolved_at"] is None


def test_frames_with_identical_clock_ticks_do_not_fail(runtime, monkeypatch):
    from app.services import runtime as runtime_module

    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: 100.0)
    camera = next(iter(runtime.sessions.values()))
    runtime.start(camera.id, "browser")
    frame = np.zeros((120, 160, 3), np.uint8)
    for timestamp in [0, 1, 2]:
        result = runtime.process(camera, frame, timestamp)
        assert result["fps"] == 0
        assert camera.status == "running"
    assert runtime.storage.incidents()["total"] == 1


def test_api_frames_settings_history_feedback_and_origin(runtime):
    original = runtime.process
    ticks = iter([0, 1, 2])
    runtime.process = lambda session, frame: original(session, frame, next(ticks))
    with TestClient(create_app(runtime.settings, runtime)) as client:
        camera = client.get("/api/v1/cameras").json()[0]
        cid = camera["id"]
        assert client.get("/api/v1/system/info").json()["model_ready"]
        started = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"})
        assert started.status_code == 200
        headers = {"X-Session-Id": started.json()["session_id"]}
        assert (
            client.post(
                f"/api/v1/cameras/{cid}/frame", content=b"bad image", headers=headers
            ).status_code
            == 400
        )
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
        assert ok
        for _ in range(3):
            response = client.post(
                f"/api/v1/cameras/{cid}/frame", content=encoded.tobytes(), headers=headers
            )
            assert response.status_code == 200, response.text
        result = client.get("/api/v1/incidents?status=active").json()
        assert result["total"] == 1
        incident = result["items"][0]
        image = client.get(f"/api/v1/incidents/{incident['id']}/snapshot")
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        verdict = client.post(
            f"/api/v1/incidents/{incident['id']}/feedback", json={"verdict": "false_positive"}
        )
        assert verdict.json()["feedback"] == "false_positive"
        assert (
            client.post(
                f"/api/v1/cameras/{cid}/stop", headers={"Origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        assert (
            client.put(
                f"/api/v1/cameras/{cid}", json=CameraConfig(name="Цех 2").model_dump()
            ).status_code
            == 200
        )
        assert (
            client.get("/api/v1/incidents").json()["items"][0]["resolution_reason"]
            == "rules_changed"
        )
        assert client.post(f"/api/v1/cameras/{cid}/stop").status_code == 200
        assert (
            client.post(
                f"/api/v1/cameras/{cid}/frame", content=encoded.tobytes(), headers=headers
            ).status_code
            == 409
        )
        assert client.get("/api/v1/incidents/unknown/snapshot").status_code == 404
        assert client.put(f"/api/v1/cameras/{cid}", json={"zone_enabled": True}).status_code == 422


def test_no_model_blocks_monitoring_but_keeps_setup_ui(tmp_path):
    config = Settings(
        _env_file=None,
        model_path=str(tmp_path / "missing.onnx"),
        database_url=f"sqlite:///{tmp_path}/db.sqlite",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert not client.get("/api/v1/system/info").json()["model_ready"]
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        response = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"})
        assert response.status_code == 409
        assert "Модель не найдена" in response.json()["detail"]


def test_old_frames_and_old_stop_cannot_affect_restarted_camera(runtime):
    with TestClient(create_app(runtime.settings, runtime)) as client:
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        old = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"}).json()[
            "session_id"
        ]
        new = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"}).json()[
            "session_id"
        ]
        assert old != new
        _, encoded = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
        assert (
            client.post(
                f"/api/v1/cameras/{cid}/frame",
                content=encoded.tobytes(),
                headers={"X-Session-Id": old},
            ).status_code
            == 409
        )
        assert client.post(f"/api/v1/cameras/{cid}/stop?session_id={old}").status_code == 409
        current = client.post(
            f"/api/v1/cameras/{cid}/frame", content=encoded.tobytes(), headers={"X-Session-Id": new}
        )
        assert current.status_code == 200
        assert client.get(current.json()["frame_url"]).status_code == 200
        assert client.get(f"/api/v1/cameras/{cid}/frame.jpg?session_id={old}").status_code == 409
        assert client.get("/api/v1/cameras").json()[0]["status"] == "running"


def test_canceled_upload_cannot_restart_another_session(runtime, tmp_path):
    camera = next(iter(runtime.sessions.values()))
    upload_id = runtime.reserve_upload(camera.id)
    assert camera.status == "uploading"
    with pytest.raises(RuntimeError):
        runtime.start(camera.id, "browser")
    runtime.stop(camera.id)
    runtime.start(camera.id, "browser")
    current = camera.session_id
    with pytest.raises(RuntimeError):
        runtime.finish_upload(camera.id, upload_id, tmp_path / "old.mp4")
    assert camera.session_id == current and camera.status == "running"


def test_empty_upload_cleans_reserved_state(runtime):
    with TestClient(create_app(runtime.settings, runtime)) as client:
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        assert client.post(f"/api/v1/cameras/{cid}/video", content=b"").status_code == 400
        assert client.get("/api/v1/cameras").json()[0]["status"] == "stopped"
        assert list((runtime.data / "uploads").iterdir()) == []


def test_incident_terminal_state_cannot_be_reopened_in_storage(runtime):
    record = runtime.storage.create_incident("cam", "Цех", (1, "no_helmet"), 0.9, 2)
    runtime.storage.update(record["id"], status="resolved")
    with pytest.raises(ValueError):
        runtime.storage.update(record["id"], status="active")
    with pytest.raises(ValueError):
        runtime.storage.update(record["id"], status="interrupted")
    assert runtime.storage.get(record["id"])["status"] == "resolved"


def test_provider_failure_stops_camera_and_rejects_further_frames(runtime):
    def fail(frame):
        raise ValueError("Model output is malformed")

    runtime.provider.predict_sync = fail
    with TestClient(create_app(runtime.settings, runtime)) as client:
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        started = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"}).json()
        _, encoded = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
        headers = {"X-Session-Id": started["session_id"]}
        response = client.post(
            f"/api/v1/cameras/{cid}/frame", content=encoded.tobytes(), headers=headers
        )
        assert response.status_code == 409
        assert client.get("/api/v1/cameras").json()[0]["status"] == "error"
        assert runtime.storage.incidents()["total"] == 0
        assert (
            client.post(
                f"/api/v1/cameras/{cid}/frame", content=encoded.tobytes(), headers=headers
            ).status_code
            == 409
        )


def test_snapshot_failure_does_not_create_incident_or_notification(runtime, monkeypatch):
    from pathlib import Path

    camera = next(iter(runtime.sessions.values()))
    runtime.start(camera.id, "browser")
    frame = np.zeros((120, 160, 3), np.uint8)
    runtime.process(camera, frame, 0)
    runtime.process(camera, frame, 1)

    def fail(path, data):
        raise OSError("Disk full")

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError):
        runtime.process(camera, frame, 2)
    assert runtime.storage.incidents()["total"] == 0
    assert runtime.notifier.jobs.empty()


def test_weak_no_helmet_below_threshold_never_confirms(tmp_path):
    """End-to-end F2: weak 0.45 stays visible in output yet creates zero incidents."""
    config = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/weak.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )
    inner = ControlledProvider()
    inner.ppe_confidence = 0.45
    assert 0.45 < config.class_thresholds["no_helmet"]
    runtime = Runtime(config, ThresholdEnforcingProvider(inner, config.class_thresholds))
    try:
        camera = next(iter(runtime.sessions.values()))
        runtime.start(camera.id, "browser")
        frame = np.full((240, 320, 3), 100, np.uint8)
        seen_weak = False
        for timestamp in range(8):
            result = runtime.process(camera, frame, timestamp)
            seen_weak |= any(
                d["class_name"] == "no_helmet" and d["weak"]
                for d in result["detections"]
            )
        assert seen_weak
        assert runtime.storage.incidents()["total"] == 0
    finally:
        runtime.close()


def test_frame_result_carries_track_hint_for_associated_ppe(runtime):
    with TestClient(create_app(runtime.settings, runtime)) as client:
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        started = client.post(f"/api/v1/cameras/{cid}/start", json={"kind": "browser"}).json()
        _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
        response = client.post(
            f"/api/v1/cameras/{cid}/frame",
            content=encoded.tobytes(),
            headers={"X-Session-Id": started["session_id"]},
        )
        assert response.status_code == 200
        body = response.json()
        track_ids = {str(t["id"]) for t in body["tracks"]}
        ppes = [d for d in body["detections"] if d["class_name"] == "no_helmet"]
        assert ppes and ppes[0]["track_hint"] in track_ids


def test_token_gates_frame_preview_and_snapshot(tmp_path):
    """F1: <img> URLs carry ?token=; without it preview/snapshot are 401."""
    config = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/token.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
        api_token="secret",
    )
    runtime = Runtime(config, ControlledProvider())
    try:
        original = runtime.process
        ticks = iter([0, 1, 2])
        runtime.process = lambda session, frame: original(session, frame, next(ticks))
        with TestClient(create_app(config, runtime)) as client:
            cid = client.get("/api/v1/cameras?token=secret").json()[0]["id"]
            started = client.post(
                f"/api/v1/cameras/{cid}/start?token=secret", json={"kind": "browser"}
            ).json()
            sid = started["session_id"]
            _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
            headers = {"X-Session-Id": sid}
            for _ in range(3):
                posted = client.post(
                    f"/api/v1/cameras/{cid}/frame?token=secret",
                    content=encoded.tobytes(),
                    headers=headers,
                )
                assert posted.status_code == 200
            frame_url = posted.json()["frame_url"]
            assert client.get(frame_url).status_code == 401
            assert client.get(with_token(frame_url)).status_code == 200
            incident = client.get("/api/v1/incidents?token=secret").json()["items"][0]
            snap = f"/api/v1/incidents/{incident['id']}/snapshot"
            assert client.get(snap).status_code == 401
            assert client.get(with_token(snap)).status_code == 200
    finally:
        runtime.close()


def with_token(url: str) -> str:
    return url + ("&" if "?" in url else "?") + "token=secret"


def test_rtsp_scheme_allowlist_rejects_non_stream_urls(runtime):
    from pydantic import ValidationError

    from app.services.schemas import StartSource

    assert StartSource(kind="rtsp", url="rtsp://cam:554/s").url.startswith("rtsp://")
    for bad in ["file:///etc/passwd", "ftp://host/x", "javascript:alert(1)", "", "notaurl"]:
        with pytest.raises(ValidationError):
            StartSource(kind="rtsp", url=bad)


def test_retention_purges_old_resolved_and_orphans(tmp_path):
    from datetime import UTC, datetime, timedelta

    from app.services.storage import Storage

    storage = Storage(tmp_path / "db.sqlite")
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    old_snap = snaps / "old.jpg"
    old_snap.write_bytes(b"old")
    anchor = datetime.now(UTC) - timedelta(days=40)
    with storage.connection() as db:
        db.execute(
            "INSERT INTO incidents (id, camera_id, camera_name, track_id, type, severity,"
            " status, started_at, resolved_at, confidence, duration, snapshot, feedback)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("old1", "cam", "C", 1, "no_helmet", "warning", "resolved",
             anchor.isoformat(), anchor.isoformat(), 0.9, 5.0, "old.jpg", "unreviewed"),
        )
    orphan = snaps / "orphan.jpg"
    orphan.write_bytes(b"orphan")
    removed = storage.purge_old_resolved(30, snaps)
    assert removed == 2
    assert storage.incidents()["total"] == 0
    assert not old_snap.exists() and not orphan.exists()
    # Retention off: nothing purged, orphans still swept.
    assert storage.purge_old_resolved(0, snaps) == 0
    # Dotfiles (e.g. .gitkeep) are never treated as orphans.
    keep = snaps / ".gitkeep"
    keep.write_bytes(b"")
    assert storage.purge_old_resolved(30, snaps) == 0
    assert keep.exists()
