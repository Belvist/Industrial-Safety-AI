"""Rule-vs-model gating: unsupported rules cannot be enabled.

Pure helper tests need no weights. API tests inject a stub provider, so they
run in clean checkouts too: mask:true on a 5-class model must 422, while
helmet/vest stay allowed and a full-class model allows mask.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.runtime import Runtime, unsupported_rules  # noqa: E402


class _StubProvider:
    def __init__(self, classes):
        self._classes = classes

    def supported_classes(self):
        return list(self._classes)


def _config(tmp_path, **overrides):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/s.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
        **overrides,
    )


def _client(tmp_path, classes):
    config = _config(tmp_path)
    service = Runtime(config, provider=_StubProvider(classes))
    return TestClient(create_app(config, runtime=service))


def test_unsupported_rules_pure():
    full = {"helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask",
            "smoking", "person"}
    five = {"helmet", "no_helmet", "vest", "no_vest", "person"}

    class Cfg:
        helmet = vest = mask = smoking = True

    assert unsupported_rules(Cfg(), full) == []
    # a rule needs its VIOLATION class: bare `helmet` is not enough
    assert unsupported_rules(Cfg(), five) == ["mask", "smoking"]
    assert unsupported_rules(Cfg(), {"helmet", "vest", "mask", "smoking",
                                     "person"}) == ["helmet", "vest", "mask"]
    assert unsupported_rules(Cfg(), {"person"}) == ["helmet", "vest", "mask", "smoking"]

    class Off:
        helmet = vest = mask = smoking = False

    assert unsupported_rules(Off(), {"person"}) == []


def test_mask_rejected_on_five_class_model(tmp_path):
    kinds = ["helmet", "no_helmet", "vest", "no_vest", "person"]
    with _client(tmp_path, kinds) as client:
        body = {"name": "K", "helmet": True, "vest": True, "mask": True,
                "smoking": False, "zone_enabled": False, "zone": []}
        created = client.post("/api/v1/cameras", json=body)
        assert created.status_code == 422, created.text
        body["mask"] = False
        ok = client.post("/api/v1/cameras", json=body)
        assert ok.status_code == 201, ok.text
        cid = ok.json()["id"]
        bad = client.put(f"/api/v1/cameras/{cid}", json={**body, "mask": True})
        assert bad.status_code == 422, bad.text


def test_start_refuses_saved_unsupported_rules(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.schemas import CameraConfig
    from app.services.storage import Storage

    kinds = ["helmet", "no_helmet", "vest", "no_vest", "person"]
    config = _config(tmp_path)
    # legacy saved config, as loaded on restart (bypasses the API gate)
    body = {"name": "K", "helmet": True, "vest": True, "mask": True,
            "smoking": False, "zone_enabled": False, "zone": []}
    storage = Storage(config.storage_path)
    cid = storage.save_camera(CameraConfig(**body))
    service = Runtime(config, provider=_StubProvider(kinds))
    assert service.session(cid).config.mask is True
    with TestClient(create_app(config, runtime=service)) as client:
        refused = client.post(f"/api/v1/cameras/{cid}/start",
                              json={"kind": "browser", "url": ""})
        assert refused.status_code == 422, refused.text
        assert "mask" in refused.text


def test_mask_allowed_on_full_class_model(tmp_path):
    kinds = ["helmet", "no_helmet", "vest", "no_vest",
             "mask", "no_mask", "smoking", "person"]
    with _client(tmp_path, kinds) as client:
        body = {"name": "K", "helmet": True, "vest": True, "mask": True,
                "smoking": False, "zone_enabled": False, "zone": []}
        created = client.post("/api/v1/cameras", json=body)
        assert created.status_code == 201, created.text
