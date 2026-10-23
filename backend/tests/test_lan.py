"""LAN/phone mode: public URL exposure, allowed origin and host."""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

LAN = "https://192.168.1.194:8000"


def _config(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/s.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
        **overrides,
    )


def test_public_url_exposed_and_lan_origin_allowed(tmp_path):
    config = _config(tmp_path, public_url=LAN, extra_origins=LAN)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/v1/system/info").json()["public_url"] == LAN
        cid = client.get("/api/v1/cameras").json()[0]["id"]
        ok = client.post(f"/api/v1/cameras/{cid}/stop", headers={"Origin": LAN})
        assert ok.status_code == 200
        evil = client.post(
            f"/api/v1/cameras/{cid}/stop", headers={"Origin": "https://evil.example"}
        )
        assert evil.status_code == 403


def test_public_url_hidden_without_lan_mode(tmp_path):
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/api/v1/system/info").json()["public_url"] is None


def test_lan_host_allowed_only_when_configured(tmp_path):
    lan = _config(tmp_path, public_url=LAN, extra_origins=LAN)
    with TestClient(create_app(lan), base_url=LAN) as client:
        assert client.get("/health").status_code == 200
    plain = _config(tmp_path)
    with TestClient(create_app(plain), base_url=LAN) as client:
        assert client.get("/health").status_code == 400


def test_api_token_gates_api_but_not_health_or_info(tmp_path):
    config = _config(tmp_path, api_token="secret")
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/system/info").status_code == 200
        assert client.get("/api/v1/cameras").status_code == 401
        assert client.get("/api/v1/cameras?token=secret").status_code == 200
        assert client.get("/api/v1/incidents").status_code == 401
        cid = client.get("/api/v1/cameras?token=secret").json()[0]["id"]
        denied = client.post(f"/api/v1/cameras/{cid}/stop")
        assert denied.status_code == 401
        allowed = client.post(
            f"/api/v1/cameras/{cid}/stop", headers={"X-API-Token": "secret"}
        )
        assert allowed.status_code == 200


def test_system_info_never_leaks_api_token(tmp_path):
    """S1: even a misconfigured PUBLIC_URL with ?token= is stripped."""
    config = _config(
        tmp_path,
        api_token="secret",
        public_url=LAN + "?token=secret",
        extra_origins=LAN,
    )
    with TestClient(create_app(config)) as client:
        body = client.get("/api/v1/system/info").json()
        assert body["public_url"] == LAN
        assert "secret" not in body["public_url"]
