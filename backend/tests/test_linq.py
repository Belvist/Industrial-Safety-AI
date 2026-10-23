"""LinQ adapter contract: normalized Detection output in every mode."""

import numpy as np

from app.domain.models import BoundingBox
from app.inference.linq import LinqProvider


def _frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_linq_remote_normalizes_response(monkeypatch):
    provider = LinqProvider(endpoint="https://linq.example.com", api_key="k")
    provider.load()
    assert provider.mode == "linq_remote"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):

        assert timeout == 8.0
        payload = {
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [0.1, 0.1, 0.5, 0.9],
                },
                {
                    "label": "no-helmet",
                    "confidence": 0.8,
                    "bbox": [0.2, 0.1, 0.4, 0.3],
                },
            ]
        }
        return FakeResponseWithData(payload)

    class FakeResponseWithData(FakeResponse):
        def __init__(self, payload):
            import io
            import json

            self._io = io.StringIO(json.dumps(payload))

        def __enter__(self):
            return self._io

        def __exit__(self, *args):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    detections = provider.predict_sync(_frame())
    assert [d.class_name for d in detections] == ["person", "no_helmet"]
    assert all(isinstance(d.bbox, BoundingBox) for d in detections)
    assert provider.stats.remote_calls == 1


def test_linq_remote_failure_is_explicit(monkeypatch):
    provider = LinqProvider(endpoint="https://linq.example.com")
    provider.load()

    import urllib.request

    def fail(request, timeout=None):
        raise ConnectionError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    try:
        provider.predict_sync(_frame())
    except RuntimeError as exc:
        assert "LinQ endpoint недоступен" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("remote failure must raise RuntimeError")


def test_linq_local_fallback_without_weights_keeps_contract(tmp_path):
    provider = LinqProvider(
        model_path=str(tmp_path / "missing.onnx"),
        person_model_path=str(tmp_path / "missing-person.onnx"),
    )
    provider.load()
    assert provider.mode == "local_unloaded"
    assert provider.status()["provider"] == "linq"


def test_parse_remote_rejects_garbage():
    import pytest

    with pytest.raises(RuntimeError):
        LinqProvider._parse_remote({"detections": [{"label": "person"}]})


def test_remote_output_filtered_by_per_class_threshold(monkeypatch):
    provider = LinqProvider(
        endpoint="https://linq.example.com",
        thresholds={"no_helmet": 0.60, "person": 0.45},
    )
    provider.load()

    import io
    import json

    payload = {
        "detections": [
            {"label": "person", "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.9]},
            {"label": "no-helmet", "confidence": 0.45, "bbox": [0.2, 0.1, 0.4, 0.3]},
            {"label": "no-helmet", "confidence": 0.8, "bbox": [0.6, 0.1, 0.8, 0.3]},
        ]
    }

    def fake_urlopen(request, timeout=None):
        return io.BytesIO(json.dumps(payload).encode())

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    detections = provider.predict_sync(_frame())
    # Weak no_helmet 0.45 is kept flagged weak (visible, harmless);
    # strong 0.80 passes like locally.
    assert [(d.class_name, d.confidence, d.weak) for d in detections] == [
        ("person", 0.9, False),
        ("no_helmet", 0.45, True),
        ("no_helmet", 0.8, False),
    ]
    # A completed detect round-trip confirms inference (not hardware).
    assert provider.readiness == "inference_verified"
    assert provider.accelerator_confirmed() is False


def test_provider_status_reports_real_execution():
    from app.inference.local_onnx import LocalOnnxProvider

    provider = LocalOnnxProvider("unused", providers=["CPUExecutionProvider"])
    status = provider.status()
    assert status["provider"] == "local_onnx"
    assert status["active_providers"] == []
    linq = LinqProvider(endpoint="https://linq.example.com")
    linq.load()
    assert linq.status()["mode"] == "linq_remote"


def test_readiness_lifecycle(monkeypatch, tmp_path):
    import contextlib
    import urllib.request

    provider = LinqProvider(endpoint="https://linq.example.com")
    provider.load()
    assert provider.readiness == "configured"
    assert provider.status()["readiness"] == "configured"
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: contextlib.nullcontext()
    )
    # Health probe proves reachability, never verified inference.
    assert provider.verify() == "reachable"
    assert provider.status()["readiness"] == "reachable"

    def fail(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert provider.verify() == "failed"

    missing = LinqProvider(
        model_path=str(tmp_path / "m.onnx"),
        person_model_path=str(tmp_path / "mp.onnx"),
    )
    missing.load()
    assert missing.mode == "local_unloaded"
    assert missing.readiness == "failed"


class _FakeSession:
    """Minimal onnxruntime session double: records providers, serves metadata."""

    instances: list = []

    def __init__(self, path, options, providers):
        self.requested = list(providers)
        _FakeSession.instances.append(self)

    def get_inputs(self):
        class _Input:
            shape = [1, 3, 640, 640]
            name = "images"

        return [_Input()]

    def get_providers(self):
        return list(self.requested)

    def get_modelmeta(self):
        class _Meta:
            custom_metadata_map = {"names": "{0: 'person'}"}

        return _Meta()


def _patch_ort(monkeypatch, available):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: list(available))
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    _FakeSession.instances.clear()


def test_local_session_receives_filtered_providers_with_cpu_fallback(
    monkeypatch, tmp_path
):
    from app.inference.local_onnx import LocalOnnxProvider

    _patch_ort(monkeypatch, ["CPUExecutionProvider"])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    provider = LocalOnnxProvider(
        str(model),
        person_model_path=None,
        required_classes={"person"},
        providers=["LinQExecutionProvider"],
    )
    provider.load()
    # Unknown EP filtered out, CPU appended — never passed blindly to ORT.
    assert _FakeSession.instances[0].requested == ["CPUExecutionProvider"]
    assert provider.active_providers == ["CPUExecutionProvider"]


def test_linq_local_mode_only_when_ep_really_in_session(monkeypatch, tmp_path):
    _patch_ort(monkeypatch, ["CPUExecutionProvider"])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    provider = LinqProvider(
        model_path=str(model),
        person_model_path=None,
        providers=["LinQExecutionProvider"],
    )
    # LocalOnnxProvider requires person in classes for the main model; the fake
    # metadata serves {0: 'person'} so relax the requirement for this wiring test.
    from app.inference.local_onnx import LocalOnnxProvider

    provider._local = LocalOnnxProvider(
        str(model),
        provider.confidence,
        provider.threads,
        person_model_path=None,
        required_classes={"person"},
        providers=provider.requested_providers,
    )
    provider._local.load()
    provider.stats.mode = provider._pick_local_mode()
    provider._loaded = True
    # EP not in the real session -> honest local_onnx, never fake linq_local.
    assert provider.mode == "local_onnx"
    assert provider.status()["active_providers"] == ["CPUExecutionProvider"]
    assert provider.status()["requested_providers"] == ["LinQExecutionProvider"]


def test_linq_local_mode_when_ep_in_session(monkeypatch, tmp_path):
    _patch_ort(monkeypatch, ["LinQExecutionProvider", "CPUExecutionProvider"])
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    provider = LinqProvider(
        model_path=str(model),
        person_model_path=None,
        providers=["LinQExecutionProvider"],
    )
    from app.inference.local_onnx import LocalOnnxProvider

    provider._local = LocalOnnxProvider(
        str(model),
        provider.confidence,
        provider.threads,
        person_model_path=None,
        required_classes={"person"},
        providers=provider.requested_providers,
    )
    provider._local.load()
    provider.stats.mode = provider._pick_local_mode()
    provider._loaded = True
    assert provider.mode == "linq_local"
    assert _FakeSession.instances[0].requested == [
        "LinQExecutionProvider",
        "CPUExecutionProvider",
    ]
    # Local session proof upgrades to hardware_verified.
    provider.readiness = (
        "hardware_verified" if provider.accelerator_confirmed() else "fallback"
    )
    assert provider.readiness == "hardware_verified"
    assert provider.accelerator_confirmed() is True
