"""Case slide 13 requirement 5: journal filters by type, camera, zone, time."""

from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.runtime import Runtime
from app.services.storage import Storage


def _seed(storage: Storage):
    return [
        storage.create_incident("cam1", "Цех 1", (1, "no_helmet"), 0.9, 1.0),
        storage.create_incident(
            "cam1", "Цех 1", (2, "danger_zone"), 0.8, 1.0, zone_name="Зона А"
        ),
        storage.create_incident("cam2", "Цех 2", (3, "running"), 0.7, 1.0),
    ]


def test_storage_filters_by_type_camera_zone_and_time(tmp_path):
    storage = Storage(tmp_path / "s.db")
    records = _seed(storage)

    assert storage.incidents(types=["running"])["total"] == 1
    assert storage.incidents(camera_ids={"cam1"})["total"] == 2
    assert storage.incidents(zone_name="Зона А")["total"] == 1
    assert (
        storage.incidents(types=["no_helmet", "danger_zone"], camera_ids={"cam1"})[
            "total"
        ]
        == 2
    )
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    assert storage.incidents(from_time=future)["total"] == 0
    assert storage.incidents(to_time=past)["total"] == 0
    assert storage.incidents(from_time=records[0]["started_at"])["total"] == 3
    page = storage.incidents(limit=1, offset=0)
    assert len(page["items"]) == 1 and page["total"] == 3


class _StubProvider:
    def supported_classes(self):
        return ["person", "helmet", "no_helmet", "vest", "no_vest"]


def _client(tmp_path):
    config = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/s.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
    )
    runtime = Runtime(config, provider=_StubProvider())
    _seed(runtime.storage)
    return runtime, TestClient(create_app(config, runtime=runtime))


def test_api_filters_and_rejects_unknown_type(tmp_path):
    runtime, client = _client(tmp_path)
    with client:
        assert client.get("/api/v1/incidents?type=running").json()["total"] == 1
        assert (
            client.get("/api/v1/incidents?camera_id=cam1").json()["total"] == 2
        )
        assert (
            client.get("/api/v1/incidents?zone=" + "Зона А").json()["total"] == 1
        )
        assert client.get("/api/v1/incidents?type=fall").status_code == 422
        start = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        end = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        assert (
            client.get(
                f"/api/v1/incidents?from_time={quote(start)}&to_time={quote(end)}"
            ).json()["total"]
            == 3
        )
        assert (
            client.get(
                f"/api/v1/incidents?from_time={quote(end)}&to_time={quote(start)}"
            ).status_code
            == 422
        )
    runtime.close()
