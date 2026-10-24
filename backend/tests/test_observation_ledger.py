"""Persisted event provenance, camera isolation and actual aggregate counts."""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.runtime import Runtime
from app.services.storage import Storage


def test_timeline_links_only_same_camera_episode_and_actual_evidence(tmp_path):
    storage = Storage(tmp_path / "events.db")
    incident = storage.create_incident(
        "a", "A", (1, "danger_zone"), 0.9, 1, episode_id="episode", evidence="clip.mp4"
    )
    storage.record_event("a", "warning", episode_id="episode", source_seconds=1.25)
    storage.record_event("b", "warning", episode_id="episode")
    storage.record_event("a", "critical", episode_id="another")
    storage.record_event("a", "confirmed", incident_id=incident["id"])
    storage.record_event("a", "evidence_ready", evidence="clip.mp4")
    storage.record_event("a", "evidence_ready", evidence="other.mp4")
    rows = storage.timeline(incident)
    assert [r["kind"] for r in rows] == ["warning", "confirmed", "evidence_ready"]
    assert rows[0]["source_seconds"] == 1.25
    assert storage.timeline(storage.create_incident("a", "A", (2, "running"), 0.9, 1)) == []


def test_statistics_are_scoped_counts_with_time_and_zone(tmp_path):
    storage = Storage(tmp_path / "stats.db")
    a = storage.create_incident(
        "a", "A", (1, "danger_zone"), 0.9, 1, zone_name="Z", started_at="2026-10-24T00:00:00+00:00"
    )
    storage.create_incident("b", "B", (1, "danger_zone"), 0.9, 1, zone_name="Z")
    storage.update(a["id"], feedback="confirmed")
    storage.record_event("a", "warning", zone_name="Z", occurred_at="2026-10-24T00:00:00+00:00")
    storage.record_event("a", "critical", zone_name="Z", occurred_at="2026-10-24T00:00:01+00:00")
    result = storage.risk_statistics({"a"})
    assert result["incidents"][0]["count"] == result["incidents"][0]["confirmed"] == 1
    assert result["incidents"][0]["zone_name"] == "Z"
    assert len(result["warnings"]) == 2
    assert storage.risk_statistics(set())["incidents"] == []
    assert storage.risk_statistics({"a"}, from_time="2027-01-01")["warnings"] == []
    assert storage.risk_statistics({"a"}, to_time="2025-01-01")["incidents"] == []


def test_timeline_and_analytics_authorization(tmp_path):
    class Provider:
        def supported_classes(self):
            return ["person", "helmet", "vest", "no_helmet", "no_vest"]

    settings = Settings(
        _env_file=None,
        api_token="operator-test",
        data_dir=str(tmp_path),
        database_url=f"sqlite:///{tmp_path}/api.db",
        telegram_enabled=False,
    )
    runtime = Runtime(settings, Provider())
    camera = next(iter(runtime.sessions))
    own = runtime.storage.create_incident(camera, "Mine", (1, "running"), 0.9, 1)
    other = runtime.storage.create_incident("other", "Other", (1, "running"), 0.9, 1)
    runtime.storage.record_event(camera, "confirmed", incident_id=own["id"])
    try:
        with TestClient(create_app(settings, runtime)) as client:
            path = f"/api/v1/incidents/{own['id']}/timeline"
            assert client.get(path).status_code == 401
            token = runtime.pairing.exchange(runtime.pairing.create(camera)[0])[1].token
            phone = {"X-API-Token": token}
            assert client.get(path, headers=phone).status_code == 200
            assert (
                client.get(f"/api/v1/incidents/{other['id']}/timeline", headers=phone).status_code
                == 403
            )
            assert client.get("/api/v1/analytics/risk", headers=phone).status_code == 403
            master = {"X-API-Token": "operator-test"}
            assert client.get("/api/v1/analytics/risk", headers=master).status_code == 200
            client.post(
                f"/api/v1/incidents/{own['id']}/feedback",
                headers=phone,
                json={"verdict": "confirmed"},
            )
            timeline = client.get(path, headers=phone).json()["items"]
            assert timeline[-1]["kind"] == "operator_review"
            assert timeline[-1]["payload"]["verdict"] == "confirmed"
            assert (
                client.get(
                    "/api/v1/analytics/risk?from_time=2027-01-01T00:00:00Z&to_time=2026-01-01T00:00:00Z",
                    headers=master,
                ).status_code
                == 422
            )
    finally:
        runtime.close()
