"""Phone pairing: one-time codes, scoped tokens, revocation."""

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.pairing import PairingStore
from app.services.runtime import Runtime


def _config(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/p.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
        api_token="master-secret",
        **overrides,
    )


class _Provider:
    name = "fake"

    def predict_sync(self, frame):
        return []


def _runtime(tmp_path):
    return Runtime(_config(tmp_path), _Provider())


def test_pairing_create_requires_operator(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            cid = client.get("/api/v1/cameras?token=master-secret").json()[0]["id"]
            assert client.post("/api/v1/pairing", json={"camera_id": cid}).status_code == 401
            created = client.post(
                "/api/v1/pairing?token=master-secret", json={"camera_id": cid}
            )
            assert created.status_code == 201
            body = created.json()
            assert body["camera_id"] == cid and body["expires_in"] > 0
    finally:
        runtime.close()


def test_exchange_single_use_and_expiry(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            cid = client.get("/api/v1/cameras?token=master-secret").json()[0]["id"]
            code = client.post(
                "/api/v1/pairing?token=master-secret", json={"camera_id": cid}
            ).json()["code"]
            first = client.post("/api/v1/pairing/exchange", json={"code": code})
            assert first.status_code == 200
            assert first.json()["camera_id"] == cid
            again = client.post("/api/v1/pairing/exchange", json={"code": code})
            assert again.status_code == 404
            assert client.post("/api/v1/pairing/exchange", json={"code": "nope"}).status_code == 404
            # Forced expiry -> 410.
            code2 = client.post(
                "/api/v1/pairing?token=master-secret", json={"camera_id": cid}
            ).json()["code"]
            runtime.pairing.codes[code2].expires_at = 0
            assert (
                client.post("/api/v1/pairing/exchange", json={"code": code2}).status_code
                == 410
            )
    finally:
        runtime.close()


def test_phone_token_scope_and_revoke(tmp_path):
    import cv2

    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            cams = client.get("/api/v1/cameras?token=master-secret").json()
            mine = cams[0]["id"]
            other = client.post(
                "/api/v1/cameras?token=master-secret", json={"name": "Other"}
            ).json()["id"]
            code = client.post(
                "/api/v1/pairing?token=master-secret", json={"camera_id": mine}
            ).json()["code"]
            phone = client.post("/api/v1/pairing/exchange", json={"code": code}).json()["token"]
            headers = {"X-API-Token": phone}
            # Own camera: start + frame work.
            started = client.post(f"/api/v1/cameras/{mine}/start", json={"kind": "browser"}, headers=headers)
            assert started.status_code == 200
            _, encoded = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))
            posted = client.post(
                f"/api/v1/cameras/{mine}/frame",
                content=encoded.tobytes(),
                headers={**headers, "X-Session-Id": started.json()["session_id"]},
            )
            assert posted.status_code == 200
            # Foreign camera, settings, pairing: forbidden; history filtered.
            assert (
                client.post(f"/api/v1/cameras/{other}/start", json={"kind": "browser"}, headers=headers).status_code
                == 403
            )
            assert client.get("/api/v1/incidents", headers=headers).json() == {
                "items": [],
                "total": 0,
            }
            assert (
                client.put(f"/api/v1/cameras/{mine}", json={"name": "X"}, headers=headers).status_code
                == 403
            )
            assert (
                client.post("/api/v1/pairing", json={"camera_id": mine}, headers=headers).status_code
                == 403
            )
            # Stop revokes the phone token.
            assert client.post(f"/api/v1/cameras/{mine}/stop", headers=headers).status_code == 200
            assert (
                client.post(f"/api/v1/cameras/{mine}/start", json={"kind": "browser"}, headers=headers).status_code
                == 401
            )
    finally:
        runtime.close()


def test_store_expiry_without_api():
    store = PairingStore(code_ttl_seconds=0)
    code, _ = store.create("cam")
    assert store.exchange(code)[0] == "expired"
    assert store.camera_for("whatever") is None



def test_scoped_phone_ui_contract_and_no_remote_sources(tmp_path):
    """The phone UI needs GET /cameras/{own}, not the full admin list."""
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            cams = client.get("/api/v1/cameras?token=master-secret").json()
            mine = cams[0]["id"]
            other = client.post("/api/v1/cameras?token=master-secret", json={"name": "Other"}).json()["id"]
            code = client.post("/api/v1/pairing?token=master-secret", json={"camera_id": mine}).json()["code"]
            token = client.post("/api/v1/pairing/exchange", json={"code": code}).json()["token"]
            headers = {"X-API-Token": token}
            mine_data = client.get(f"/api/v1/cameras/{mine}", headers=headers)
            assert mine_data.status_code == 200
            assert mine_data.json()["id"] == mine
            assert client.get(f"/api/v1/cameras/{other}", headers=headers).status_code == 403
            # Lists are filtered to the phone's own cameras, not forbidden.
            listed = client.get("/api/v1/cameras", headers=headers)
            assert listed.status_code == 200
            assert [c["id"] for c in listed.json()] == [mine]
            assert client.get("/api/v1/incidents", headers=headers).status_code == 200
            assert client.post(f"/api/v1/cameras/{mine}/start", headers=headers, json={"kind": "rtsp", "url": "rtsp://127.0.0.1/test"}).status_code == 403
            assert client.post(f"/api/v1/cameras/{mine}/video", headers=headers, content=b"test").status_code == 403
            start = client.post(f"/api/v1/cameras/{mine}/start", headers=headers, json={"kind": "browser"})
            assert start.status_code == 200
            assert client.post(f"/api/v1/cameras/{mine}/stop", headers=headers, params={"session_id": start.json()["session_id"]}).status_code == 200
            assert client.get(f"/api/v1/cameras/{mine}", headers=headers).status_code == 401
    finally:
        runtime.close()


def _phone_token(client, camera_id=None, name=None):
    master = {"X-API-Token": "master-secret"}
    if name is not None:
        code = client.post("/api/v1/pairing", json={"new_camera_name": name}, headers=master).json()["code"]
    else:
        code = client.post("/api/v1/pairing", json={"camera_id": camera_id}, headers=master).json()["code"]
    return client.post("/api/v1/pairing/exchange", json={"code": code}).json()


def test_isolated_phones_see_only_own_cameras_and_incidents(tmp_path):
    """Two phones via new_camera pairing: no shared state anywhere."""
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            master = {"X-API-Token": "master-secret"}
            op_cam = client.get("/api/v1/cameras", headers=master).json()[0]["id"]
            binding_a = _phone_token(client, name="Phone A")
            binding_b = _phone_token(client, name="Phone B")
            ha = {"X-API-Token": binding_a["token"]}
            hb = {"X-API-Token": binding_b["token"]}
            ids_a = [c["id"] for c in client.get("/api/v1/cameras", headers=ha).json()]
            ids_b = [c["id"] for c in client.get("/api/v1/cameras", headers=hb).json()]
            assert ids_a == [binding_a["camera_id"]]
            assert ids_b == [binding_b["camera_id"]]
            assert ids_a != ids_b
            # Cross access denied everywhere, including foreign incident ids.
            assert client.get(f"/api/v1/cameras/{ids_b[0]}", headers=ha).status_code == 403
            assert client.get("/api/v1/incidents", headers=ha).json() == {"items": [], "total": 0}
            master_ids = {c["id"] for c in client.get("/api/v1/cameras", headers=master).json()}
            assert {op_cam, ids_a[0], ids_b[0]} <= master_ids
            # Owner marker transferred from pairing code to phone token.
            assert runtime.storage.cameras()
            owners = {c["id"]: c.get("owner") for c in runtime.storage.cameras()}
            assert owners[ids_a[0]] and owners[ids_a[0]].startswith("phone:")
            assert owners[ids_a[0]] != owners[ids_b[0]]
    finally:
        runtime.close()


def test_pairing_new_camera_validation(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            master = {"X-API-Token": "master-secret"}
            bad = client.post("/api/v1/pairing", json={}, headers=master)
            assert bad.status_code == 422
            both = client.post(
                "/api/v1/pairing",
                json={"camera_id": "x", "new_camera_name": "y"},
                headers=master,
            )
            assert both.status_code == 422
    finally:
        runtime.close()


def test_revoked_isolated_phone_cannot_reclaim_owned_camera(tmp_path):
    """Hashed owner marker is not an access token after explicit revocation."""
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            binding = _phone_token(client, name="Phone isolated")
            camera = binding["camera_id"]
            headers = {"X-API-Token": binding["token"]}
            assert client.get(f"/api/v1/cameras/{camera}", headers=headers).status_code == 200
            assert client.post(f"/api/v1/cameras/{camera}/start", headers=headers, json={"kind": "browser"}).status_code == 200
            assert client.post(f"/api/v1/cameras/{camera}/stop", headers=headers).status_code == 200
            assert client.get(f"/api/v1/cameras/{camera}", headers=headers).status_code == 401
            assert client.get("/api/v1/cameras", headers=headers).status_code == 401
            assert client.get("/api/v1/system/info", headers=headers).json()["stats"]["total"] == 0
    finally:
        runtime.close()


def test_expired_isolated_phone_token_cannot_reclaim_workspace(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        with TestClient(create_app(runtime.settings, runtime)) as client:
            binding = _phone_token(client, name="Phone expires")
            camera = binding["camera_id"]
            token = binding["token"]
            headers = {"X-API-Token": token}
            assert client.get(f"/api/v1/cameras/{camera}", headers=headers).status_code == 200
            runtime.pairing.tokens[token].expires_at = 0
            assert client.get(f"/api/v1/cameras/{camera}", headers=headers).status_code == 401
            assert client.get("/api/v1/cameras", headers=headers).status_code == 401
    finally:
        runtime.close()
