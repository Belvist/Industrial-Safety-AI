"""Safety Radar: trajectory analysis, risk levels, integration through Runtime.process."""

import numpy as np

from app.core.config import Settings
from app.domain.models import BoundingBox, Detection
from app.services.engine import Track
from app.services.runtime import Runtime
from app.services.safety_radar import SafetyRadar


def _track(tid, x1, y1, x2, y2):
    return Track(
        id=tid,
        bbox=BoundingBox(x1, y1, x2, y2),
        seen=0.0,
        confidence=0.9,
    )


class SequenceProvider:
    """Returns person detections at scripted normalized boxes."""
    name = "sequence"

    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.i = 0

    def predict_sync(self, frame):
        if self.i >= len(self.boxes):
            return []
        x1, y1, x2, y2 = self.boxes[self.i]
        self.i += 1
        return [Detection("person", 0.99, BoundingBox(x1, y1, x2, y2))]


def test_normal_when_far_from_zone():
    radar = SafetyRadar()
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    tracks = [_track(1, 0.1, 0.1, 0.2, 0.2)]
    for t in range(4):
        radar.update(tracks, "cam", float(t))
    assert radar.get_current_risk_level(1, 4.0) == "NORMAL"


def test_unknown_when_track_missing():
    radar = SafetyRadar()
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    assert radar.get_current_risk_level(999, 0.0) == "UNKNOWN"


def test_warning_when_approaching_zone():
    radar = SafetyRadar(zone_threshold_norm=0.25, min_history_points=2)
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    # Move toward the zone from the left, ending just outside it
    boxes = [
        (0.10, 0.70, 0.20, 0.80),
        (0.25, 0.70, 0.35, 0.80),
        (0.45, 0.70, 0.55, 0.80),
    ]
    events = []
    for t, box in enumerate(boxes):
        ev = radar.update([_track(1, *box)], "cam", float(t))
        events.extend(ev)
    assert any(e.level == "WARNING" for e in events)
    assert any(e.risk_type == "zone_approach" for e in events)


def test_critical_when_inside_zone():
    radar = SafetyRadar(zone_threshold_norm=0.15, min_history_points=2)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    boxes = [
        (0.10, 0.40, 0.20, 0.50),
        (0.35, 0.40, 0.45, 0.50),
    ]
    events = []
    for t, box in enumerate(boxes):
        events.extend(radar.update([_track(1, *box)], "cam", float(t)))
    assert any(e.level == "CRITICAL" for e in events)
    assert any(e.risk_type == "zone_cross" for e in events)


def test_parallel_movement_no_warning():
    radar = SafetyRadar(zone_threshold_norm=0.15, min_history_points=2)
    radar.set_zones({"zone": [(0.6, 0.2), (0.9, 0.2), (0.9, 0.4), (0.6, 0.4)]})
    # Move parallel below the zone, never toward it
    boxes = [
        (0.10, 0.70, 0.20, 0.80),
        (0.30, 0.70, 0.40, 0.80),
        (0.50, 0.70, 0.60, 0.80),
        (0.70, 0.70, 0.80, 0.80),
    ]
    events = []
    for t, box in enumerate(boxes):
        events.extend(radar.update([_track(1, *box)], "cam", float(t)))
    assert not any(e.risk_type == "zone_approach" for e in events)


def test_cooldown_prevents_duplicate_events():
    radar = SafetyRadar(cooldown_seconds=10.0, min_history_points=1)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    box = (0.35, 0.40, 0.45, 0.50)
    first = radar.update([_track(1, *box)], "cam", 0.0)
    second = radar.update([_track(1, *box)], "cam", 0.1)
    assert len(first) == 1
    assert len(second) == 0  # cooldown


def test_reset_clears_state():
    radar = SafetyRadar(min_history_points=1)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    radar.update([_track(1, 0.35, 0.4, 0.45, 0.5)], "cam", 0.0)
    assert radar.get_events()
    radar.reset()
    assert radar.get_events() == []
    assert radar.get_current_risk_level(1, 0.0) == "UNKNOWN"


def test_event_limit():
    radar = SafetyRadar(cooldown_seconds=0.0, min_history_points=1)
    radar._max_events = 5
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    for t in range(20):
        radar.update([_track(1, 0.35, 0.4, 0.45, 0.5)], "cam", float(t))
    assert len(radar.get_events()) <= 5


# --- Integration through Runtime.process ---

def _settings(tmp_path):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/safety.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )


def test_runtime_radar_reaches_session_latest(tmp_path):
    """Radar must see person tracks and expose risks through FrameResult."""
    # Realistic per-frame step with high IoU so the tracker keeps one ID.
    provider = SequenceProvider([
        (0.10 + i * 0.03, 0.70, 0.30 + i * 0.03, 0.80) for i in range(18)
    ])
    runtime = Runtime(_settings(tmp_path), provider)
    try:
        cam = next(iter(runtime.sessions.values()))
        # Configure a zone in normalized coordinates
        runtime.sessions[cam.id].config.zone = [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]
        runtime.sessions[cam.id].config.zone_enabled = True
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        latest = None
        for t in range(18):
            latest = runtime.process(cam, frame, float(t))
        assert latest is not None
        assert "radar" in latest
        assert latest["radar"] is not None
        assert "track_risks" in latest["radar"]
        assert latest["radar"]["track_risks"]
        # At t=3 the person is inside the zone -> CRITICAL event
        assert any(e["level"] == "CRITICAL" for e in latest["radar"]["events"])
    finally:
        runtime.close()


def test_runtime_radar_resets_on_restart(tmp_path):
    provider = SequenceProvider([(0.35, 0.40, 0.45, 0.50)] * 6)
    runtime = Runtime(_settings(tmp_path), provider)
    try:
        cam = next(iter(runtime.sessions.values()))
        runtime.sessions[cam.id].config.zone = [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]
        runtime.sessions[cam.id].config.zone_enabled = True
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for t in range(6):
            runtime.process(cam, frame, float(t))
        assert runtime.sessions[cam.id].radar.get_events()
        runtime.stop(cam.id)
        runtime.start(cam.id, "browser")
        assert runtime.sessions[cam.id].radar.get_events() == []
    finally:
        runtime.close()


def test_api_frame_result_exposes_radar(tmp_path):
    """FastAPI must not drop radar (Pydantic response model)."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    provider = SequenceProvider([(0.10 + i * 0.03, 0.70, 0.30 + i * 0.03, 0.80) for i in range(18)])
    settings = _settings(tmp_path)
    runtime = Runtime(settings, provider)
    try:
        with TestClient(create_app(settings, runtime)) as client:
            cam = client.get("/api/v1/cameras").json()[0]
            cam = client.put(
                f"/api/v1/cameras/{cam['id']}",
                json={
                    **cam,
                    "zone": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
                    "zone_enabled": True,
                },
            ).json()
            started = client.post(
                f"/api/v1/cameras/{cam['id']}/start", json={"kind": "browser"}
            ).json()
            import cv2

            _, encoded = cv2.imencode(".jpg", np.full((240, 320, 3), 100, np.uint8))
            headers = {"X-Session-Id": started["session_id"]}
            body = None
            for _ in range(18):
                body = client.post(
                    f"/api/v1/cameras/{cam['id']}/frame",
                    content=encoded.tobytes(),
                    headers=headers,
                ).json()
            assert "radar" in body
            assert body["radar"]["track_risks"]
            assert body["radar"]["events"]
    finally:
        runtime.close()


def test_evidence_replay_file_and_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    provider = SequenceProvider([(0.10 + i * 0.03, 0.70, 0.30 + i * 0.03, 0.80) for i in range(18)])
    settings = _settings(tmp_path)
    runtime = Runtime(settings, provider)
    try:
        with TestClient(create_app(settings, runtime)) as client:
            cam = client.get("/api/v1/cameras").json()[0]
            cam = client.put(
                f"/api/v1/cameras/{cam['id']}",
                json={
                    **cam,
                    "zone": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
                    "zone_enabled": True,
                },
            ).json()
            started = client.post(
                f"/api/v1/cameras/{cam['id']}/start", json={"kind": "browser"}
            ).json()
            import cv2

            _, encoded = cv2.imencode(".jpg", np.full((240, 320, 3), 100, np.uint8))
            headers = {"X-Session-Id": started["session_id"]}
            url = None
            for _ in range(18):
                body = client.post(
                    f"/api/v1/cameras/{cam['id']}/frame",
                    content=encoded.tobytes(),
                    headers=headers,
                ).json()
                for ev in body.get("radar", {}).get("events", []):
                    if ev.get("evidence_url"):
                        url = ev["evidence_url"]
            assert url, "radar event must carry evidence_url"
            import time as _time
            deadline = _time.monotonic() + 5.0
            response = client.get(url)
            while response.status_code == 404 and _time.monotonic() < deadline:
                _time.sleep(0.1)
                response = client.get(url)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("video/mp4")
            assert len(response.content) > 100
            # traversal guard
            assert client.get(
                f"/api/v1/cameras/{cam['id']}/evidence/..%2F..%2Fsecret.mp4"
            ).status_code == 404
    finally:
        runtime.close()


def test_warning_escalates_to_critical_within_cooldown():
    radar = SafetyRadar(cooldown_seconds=10.0, zone_threshold_norm=0.25, min_history_points=2)
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    events = []
    # Approach (WARNING), then enter the zone (CRITICAL) within cooldown
    for t, box in enumerate([
        (0.10, 0.70, 0.20, 0.80),
        (0.45, 0.70, 0.55, 0.80),
        (0.62, 0.62, 0.72, 0.72),
    ]):
        events.extend(radar.update([_track(1, *box)], "cam", float(t)))
    assert any(e.level == "WARNING" for e in events)
    assert any(e.level == "CRITICAL" for e in events), "escalation must bypass cooldown"


def test_get_events_dedupes_by_track_and_type():
    radar = SafetyRadar(cooldown_seconds=0.0, min_history_points=1)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    for t in range(5):
        radar.update([_track(1, 0.35, 0.4, 0.45, 0.5)], "cam", float(t))
    assert len(radar.get_events()) == 1


def test_risk_event_has_declared_evidence_field():
    """P0 regression: evidence must be a real dataclass field, not dynamic."""
    from dataclasses import fields

    from app.services.safety_radar import RiskEvent

    assert "evidence" in {f.name for f in fields(RiskEvent)}
    ev = RiskEvent(1, "cam", "zone_cross", "CRITICAL", 0.0, "m")
    assert ev.evidence is None


def test_critical_state_survives_cooldown_inside_zone():
    """P0 regression: staying in the zone > cooldown stays CRITICAL."""
    radar = SafetyRadar(cooldown_seconds=1.0, min_history_points=2)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]})
    track = _track(1, 0.35, 0.40, 0.45, 0.50)
    for t in range(15):  # 15 s inside the zone, cooldown is 1 s
        radar.update([track], "cam", float(t))
        if t >= 1:  # after min_history_points
            assert radar.get_current_risk_level(1) == "CRITICAL"
    # Notifications are throttled, state is not.
    assert len(radar.get_events()) == 1


def test_zone_disabled_emits_no_events():
    radar = SafetyRadar(min_history_points=1)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]}, enabled=True)
    radar.set_zones({"zone": [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]}, enabled=False)
    events = radar.update([_track(1, 0.35, 0.4, 0.45, 0.5)], "cam", 0.0)
    assert events == []
    assert radar.get_current_risk_level(1) == "UNKNOWN"


def test_zone_disable_invalidates_state(tmp_path):
    """Integration: turning zone_enabled off stops new WARNING/CRITICAL."""
    from app.services.runtime import Runtime

    class Provider:
        name = "p"

        def predict_sync(self, frame):
            return [Detection("person", 0.99, BoundingBox(0.35, 0.40, 0.45, 0.50))]

    settings = _settings(tmp_path)
    runtime = Runtime(settings, Provider())
    try:
        cam = next(iter(runtime.sessions.values()))
        cam.config.zone = [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]
        cam.config.zone_enabled = True
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for t in range(4):
            runtime.process(cam, frame, float(t))
        assert cam.radar.get_events()

        cam.config.zone_enabled = False
        for t in range(4, 8):
            latest = runtime.process(cam, frame, float(t))
        assert cam.radar.all_events() == []
        assert latest["radar"]["events"] == []
        assert latest["radar"]["track_risks"].get(1) in {"UNKNOWN", "NORMAL"}
    finally:
        runtime.close()


def test_multiple_events_share_failure_safe_evidence(tmp_path):
    """Two tracks in the zone must not crash if evidence writing fails."""
    from app.services.runtime import Runtime

    class TwoProvider:
        name = "two"

        def predict_sync(self, frame):
            return [
                Detection("person", 0.99, BoundingBox(0.35, 0.40, 0.45, 0.50)),
                Detection("person", 0.98, BoundingBox(0.70, 0.70, 0.80, 0.80)),
            ]

    settings = _settings(tmp_path)
    runtime = Runtime(settings, TwoProvider())
    try:
        cam = next(iter(runtime.sessions.values()))
        cam.config.zone = [(0.3, 0.3), (0.95, 0.3), (0.95, 0.95), (0.3, 0.95)]
        cam.config.zone_enabled = True
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Force writer failure: unwritable evidence dir
        (runtime.data / "evidence").chmod(0o500)
        try:
            for t in range(5):
                latest = runtime.process(cam, frame, float(t))
        finally:
            (runtime.data / "evidence").chmod(0o700)
        # No crash; events exist, evidence may be None.
        assert len(latest["radar"]["events"]) >= 1
        for ev in latest["radar"]["events"]:
            assert "evidence_url" in ev
    finally:
        runtime.close()


def test_evidence_requires_api_token(tmp_path):
    """Evidence video must be protected by X-API-Token, not a query string."""
    import time as _time

    from fastapi.testclient import TestClient

    from app.main import create_app

    provider = SequenceProvider([(0.10 + i * 0.03, 0.70, 0.30 + i * 0.03, 0.80) for i in range(18)])
    settings = _settings(tmp_path)
    settings.api_token = "secret-token"
    runtime = Runtime(settings, provider)
    try:
        with TestClient(create_app(settings, runtime)) as client:
            auth = {"X-API-Token": "secret-token"}
            cam = client.get("/api/v1/cameras", headers=auth).json()[0]
            cam = client.put(
                f"/api/v1/cameras/{cam['id']}",
                headers=auth,
                json={
                    **cam,
                    "zone": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
                    "zone_enabled": True,
                },
            ).json()
            started = client.post(
                f"/api/v1/cameras/{cam['id']}/start",
                headers=auth,
                json={"kind": "browser"},
            ).json()
            import cv2

            _, encoded = cv2.imencode(".jpg", np.full((240, 320, 3), 100, np.uint8))
            headers = {**auth, "X-Session-Id": started["session_id"]}
            url = None
            for _ in range(18):
                body = client.post(
                    f"/api/v1/cameras/{cam['id']}/frame",
                    content=encoded.tobytes(),
                    headers=headers,
                ).json()
                for ev in body.get("radar", {}).get("events", []):
                    if ev.get("evidence_url"):
                        url = ev["evidence_url"]
            assert url
            # Wait for the async writer.
            deadline = _time.monotonic() + 5.0
            while _time.monotonic() < deadline:
                if (runtime.data / "evidence" / url.rsplit("/", 1)[-1]).is_file():
                    break
                _time.sleep(0.1)
            assert client.get(url).status_code == 401
            # UI uses the header path only; token never goes into the media URL.
            assert client.get(url, headers=auth).status_code == 200
    finally:
        runtime.close()


def test_radar_critical_confirms_single_incident(tmp_path):
    """Radar CRITICAL feeds TemporalEngine; one incident, no duplicates."""
    from app.services.runtime import Runtime

    class Provider:
        name = "p"

        def predict_sync(self, frame):
            return [Detection("person", 0.99, BoundingBox(0.35, 0.40, 0.45, 0.50))]

    settings = _settings(tmp_path)
    runtime = Runtime(settings, Provider())
    try:
        cam = next(iter(runtime.sessions.values()))
        cam.config.zone = [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]
        cam.config.zone_enabled = True
        cam.config.confirm_seconds = 0.1
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for t in range(10):
            runtime.process(cam, frame, t / 30)
        assert runtime.storage.incidents()["total"] == 1
        # Continued presence must not create duplicates.
        for t in range(10, 40):
            runtime.process(cam, frame, t / 30)
        assert runtime.storage.incidents()["total"] == 1
    finally:
        runtime.close()


def test_prediction_intersects_zone_boundary():
    """Predicted path must hit the near boundary, not the zone centroid."""
    radar = SafetyRadar(min_history_points=3)
    radar.set_zones({"zone": [(0.60, 0.60), (0.90, 0.60), (0.90, 0.90), (0.60, 0.90)]})
    # Walk diagonally toward the lower-left corner of the zone
    for t, (x, y) in enumerate([(0.30, 0.86), (0.38, 0.84), (0.46, 0.82)]):
        radar.update([_track(1, x, y, x + 0.10, y + 0.10)], "cam", float(t))
    pred = radar.predict(1, 2.0, horizon_seconds=2.0)
    assert pred is not None
    assert pred["intersection"] is not None
    # Intersection lies on the near (left) boundary x≈0.60
    assert abs(pred["intersection"][0] - 0.60) < 0.08
    assert pred["confidence"] > 0


def test_prediction_absent_when_stationary():
    radar = SafetyRadar(min_history_points=2, )
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    for t in range(5):
        radar.update([_track(1, 0.30, 0.30, 0.40, 0.40)], "cam", float(t))
    assert radar.predict(1, 4.0) is None


def test_prediction_absent_with_stale_track():
    radar = SafetyRadar(min_history_points=2)
    radar.set_zones({"zone": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    for t in range(3):
        radar.update([_track(1, 0.1 + t * 0.05, 0.80, 0.2 + t * 0.05, 0.95)], "cam", float(t))
    assert radar.predict(1, 30.0) is None


def test_prediction_parallel_path_does_not_intersect():
    radar = SafetyRadar(min_history_points=2)
    radar.set_zones({"zone": [(0.6, 0.2), (0.9, 0.2), (0.9, 0.4), (0.6, 0.4)]})
    # Walk horizontally below the zone
    for t in range(5):
        x = 0.1 + t * 0.08
        radar.update([_track(1, x, 0.80, x + 0.10, 0.95)], "cam", float(t) * 0.2)
    pred = radar.predict(1, 0.8, horizon_seconds=2.0)
    assert pred is not None
    assert pred["intersection"] is None


def test_snapshot_contains_scene_geometry():
    radar = SafetyRadar(min_history_points=2)
    radar.set_zones({"Зона": [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]})
    for t in range(4):
        radar.update([_track(1, 0.1 + t * 0.05, 0.70, 0.2 + t * 0.05, 0.88)], "cam", float(t))
    scene = radar.snapshot(3.0)
    assert scene["zones"][0]["name"] == "Зона"
    assert scene["tracks"][0]["track_id"] == 1
    assert scene["tracks"][0]["trajectory"]


def test_id_switch_does_not_multiply_incidents(tmp_path):
    """One continuous zone violation must stay one incident across ID changes."""
    from app.services.runtime import Runtime

    class Provider:
        name = "id-switch"
        i = 0

        def predict_sync(self, frame):
            # Same physical person, deliberately alternating track identity.
            x = 0.35 + (self.i % 3) * 0.02
            self.i += 1
            return [Detection("person", 0.99, BoundingBox(x, 0.40, x + 0.10, 0.50))]

    settings = _settings(tmp_path)
    runtime = Runtime(settings, Provider())
    try:
        cam = next(iter(runtime.sessions.values()))
        cam.config.zone = [(0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9)]
        cam.config.zone_enabled = True
        cam.config.confirm_seconds = 0.1
        runtime.start(cam.id, "browser")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for t in range(30):
            runtime.process(cam, frame, t / 10)
        assert runtime.storage.incidents()["total"] == 1
    finally:
        runtime.close()
