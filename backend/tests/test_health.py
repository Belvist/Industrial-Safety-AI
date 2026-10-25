from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["inference_provider"] in {"local_onnx", "linq"}


def test_system_info_reports_available_providers() -> None:
    with TestClient(app) as live:
        body = live.get("/api/v1/system/info").json()
    assert "CPUExecutionProvider" in body["available_providers"]
