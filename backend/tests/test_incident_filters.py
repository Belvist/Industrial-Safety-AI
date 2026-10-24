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
        storage.create_incident("cam1", "Цех 1", (2, "danger_zone"), 0.8, 1.0, zone_name="Зона А"),
        storage.create_incident("cam2", "Цех 2", (3, "running"), 0.7, 1.0),
    ]


def test_storage_filters_by_type_camera_zone_and_time(tmp_path):
    storage = Storage(tmp_path / "s.db")
    records = _seed(storage)

    assert storage.incidents(types=["running"])["total"] == 1
    assert storage.incidents(camera_ids={"cam1"})["total"] == 2
    assert storage.incidents(zone_name="Зона А")["total"] == 1
    assert storage.incidents(types=["no_helmet", "danger_zone"], camera_ids={"cam1"})["total"] == 2
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
        assert client.get("/api/v1/incidents?camera_id=cam1").json()["total"] == 2
        assert client.get("/api/v1/incidents?zone=" + "Зона А").json()["total"] == 1
        assert client.get("/api/v1/incidents?type=fall").status_code == 422
        start = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        end = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        assert (
            client.get(f"/api/v1/incidents?from_time={quote(start)}&to_time={quote(end)}").json()[
                "total"
            ]
            == 3
        )
        assert (
            client.get(
                f"/api/v1/incidents?from_time={quote(end)}&to_time={quote(start)}"
            ).status_code
            == 422
        )
    runtime.close()


def test_review_filter_keeps_physical_status_and_counts(tmp_path):
    runtime, client = _client(tmp_path)
    records = runtime.storage.incidents()["items"]
    with client:
        response = client.post(
            f"/api/v1/incidents/{records[0]['id']}/feedback", json={"verdict": "confirmed"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "active"
        assert client.get("/api/v1/incidents?feedback=confirmed").json()["total"] == 1
        assert client.get("/api/v1/incidents?feedback=unreviewed").json()["total"] == 2
        assert client.get("/api/v1/incidents?feedback=wrong").status_code == 422
        assert runtime.storage.stats()["unreviewed"] == 2
        assert runtime.storage.stats(camera_ids=set())["unreviewed"] == 0
    runtime.close()


def test_export_includes_all_filtered_rows_and_neutralizes_formulas(tmp_path):
    import csv
    import io

    runtime, client = _client(tmp_path)
    for n in range(35):
        runtime.storage.create_incident(
            "csvcam", ' =HYPERLINK("https://test")', (n, "no_helmet"), 0.9, 1
        )
    with client:
        response = client.get("/api/v1/incidents/export.csv?camera_id=csvcam&feedback=unreviewed")
        assert response.status_code == 200
        assert response.content.startswith(b"\xef\xbb\xbf")
        assert response.headers["cache-control"] == "no-store"
        rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))
        assert len(rows) == 36  # not just the UI's first 30 records
        assert all(row[2].startswith("' =HYPERLINK") for row in rows[1:])
        assert client.get("/api/v1/incidents/export.csv?type=unknown").status_code == 422
        assert client.get("/api/v1/incidents/export.csv?feedback=wrong").status_code == 422
    runtime.close()


def test_export_requires_operator_credentials(tmp_path):
    config = Settings(
        _env_file=None,
        api_token="test-operator",
        data_dir=str(tmp_path),
        database_url=f"sqlite:///{tmp_path}/s.db",
        telegram_enabled=False,
    )
    runtime = Runtime(config, provider=_StubProvider())
    _seed(runtime.storage)
    with TestClient(create_app(config, runtime=runtime)) as client:
        assert client.get("/api/v1/incidents/export.csv").status_code == 401
        pairing = runtime.pairing.create("cam1")
        phone = runtime.pairing.exchange(pairing[0])[1]
        assert (
            client.get(
                "/api/v1/incidents/export.csv", headers={"X-API-Token": phone.token}
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/api/v1/incidents/export.csv", headers={"X-API-Token": "test-operator"}
            ).status_code
            == 200
        )
    runtime.close()
