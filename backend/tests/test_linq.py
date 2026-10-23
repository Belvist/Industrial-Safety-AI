"""LinQ adapter contract: normalized Detection output in every mode."""

import json

import numpy as np
import pytest

from app.domain.models import BoundingBox
from app.inference.linq import LinqProvider


def _frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)

class _FakeChannel:
    def __init__(self, output, worker):
        self.output = output
        self.worker = worker
        self.published = None
        self.reply_queue = ""
        self.bound = None

    def queue_declare(self, queue, auto_delete=True):
        self.reply_queue = queue
        assert auto_delete is True

    def queue_bind(self, exchange, queue, routing_key):
        self.bound = (exchange, queue, routing_key)

    def basic_publish(self, exchange, routing_key, body, properties):
        self.published = {
            "exchange": exchange,
            "routing_key": routing_key,
            "body": body,
            "headers": dict(properties.headers),
        }

    def basic_get(self, queue, auto_ack=True):
        assert queue == self.reply_queue
        assert auto_ack is True
        from app.inference.tpu_wire import PyTask, serialize

        task_id = self.published["headers"]["tpu_pycloud_message_id"]
        body = serialize(
            PyTask(
                {"output0": self.output},
                worker_pid=self.worker.get("worker_pid", "7"),
                inference_time=self.worker.get("inference_time", 0.0012),
                status_code=self.worker.get("status_code", 0),
                error_message=self.worker.get("error_message", ""),
            )
        )

        class _Method:
            pass

        class _Props:
            headers = {"tpu_pycloud_message_id": task_id}

        return _Method(), _Props(), body

class _FakeConnection:
    is_open = True

    def close(self):
        self.is_open = False

def _wire(monkeypatch, provider, output, worker=None):
    channel = _FakeChannel(output, worker or {})

    def fake_connect():
        provider._connection = _FakeConnection()
        provider._channel = channel
        provider._reply_queue = "reply-test"
        channel.reply_queue = "reply-test"
        return channel

    monkeypatch.setattr(provider, "_connect", fake_connect)
    return channel

def _output(score=0.9, label=11):
    output = np.zeros((1, 18, 8400), dtype=np.float32)
    output[0, 0:4, 0] = [100, 100, 40, 80]
    output[0, 4 + label, 0] = score
    return output

def test_linq_remote_uses_rabbitmq_protocol(monkeypatch):
    provider = LinqProvider(
        amqp_host="10.0.114.30",
        tpu_program="ppe.tpu",
        tpu_output_is_logits=False,
    )
    provider.load()
    assert provider.mode == "linq_remote"
    channel = _wire(monkeypatch, provider, _output(0.9))

    detections = provider.predict_sync(_frame())
    assert [d.class_name for d in detections] == ["person"]
    assert detections[0].confidence == pytest.approx(0.9, abs=1e-5)
    assert isinstance(detections[0].bbox, BoundingBox)
    assert channel.published["exchange"] == "tpu.direct"
    assert channel.published["routing_key"] == "input"
    assert channel.published["headers"]["uri"].startswith("file:///auto/cache/")
    assert channel.published["headers"]["uri"].endswith("ppe.tpu?raw=False")
    assert json.loads(channel.published["headers"]["response_routing_keys"]) == [
        "reply-test"
    ]
    assert provider.stats.remote_calls == 1
    assert provider.stats.worker_inference_ms == pytest.approx(1.2, abs=1e-3)
    assert provider.readiness == "inference_verified"
    assert provider.worker_verified() is True
    assert provider.accelerator_confirmed() is False

def test_linq_remote_applies_sigmoid_to_logits(monkeypatch):
    provider = LinqProvider(
        amqp_host="10.0.114.30", thresholds={"person": 0.5}, floor=0.25
    )
    provider.load()
    # Raw logit 2.0 -> sigmoid 0.881, not 2.0.
    _wire(monkeypatch, provider, _output(2.0))

    detections = provider.predict_sync(_frame())
    assert detections[0].confidence == pytest.approx(0.8808, abs=1e-3)
    assert detections[0].weak is False

def test_linq_remote_skips_sigmoid_when_program_outputs_scores(monkeypatch):
    provider = LinqProvider(
        amqp_host="10.0.114.30",
        tpu_output_is_logits=False,
        thresholds={"person": 0.5},
        floor=0.25,
    )
    provider.load()
    _wire(monkeypatch, provider, _output(0.8))
    detections = provider.predict_sync(_frame())
    assert detections[0].confidence == pytest.approx(0.8, abs=1e-6)

def test_linq_remote_failure_is_explicit(monkeypatch):
    provider = LinqProvider(amqp_host="10.0.114.30", timeout_seconds=1.0)
    provider.load()
    _wire(monkeypatch, provider, _output(), worker={"status_code": 2, "error_message": "boom"})

    with pytest.raises(RuntimeError, match="worker error 2"):
        provider.predict_sync(_frame())
    assert provider.readiness == "failed"
    assert provider.stats.remote_failures == 1

def test_linq_remote_timeout_is_explicit(monkeypatch):
    provider = LinqProvider(amqp_host="10.0.114.30", timeout_seconds=0.05)
    provider.load()

    class EmptyChannel(_FakeChannel):
        def basic_get(self, queue, auto_ack=True):
            return None, None, None

    channel = EmptyChannel(_output(), {})

    def fake_connect():
        provider._connection = _FakeConnection()
        provider._channel = channel
        provider._reply_queue = "reply-test"
        channel.reply_queue = "reply-test"
        return channel

    monkeypatch.setattr(provider, "_connect", fake_connect)
    with pytest.raises(RuntimeError, match="не ответил"):
        provider.predict_sync(_frame())
    assert provider.stats.remote_failures == 1

def test_linq_remote_filters_by_class_threshold(monkeypatch):
    provider = LinqProvider(
        amqp_host="10.0.114.30",
        tpu_output_is_logits=False,
        thresholds={"person": 0.45, "no_helmet": 0.60},
        floor=0.25,
    )
    provider.load()
    _wire(monkeypatch, provider, _output(0.45, label=8))

    detections = provider.predict_sync(_frame())
    assert detections[0].class_name == "no_helmet"
    assert detections[0].weak is True

def test_linq_local_fallback_without_weights_keeps_contract(tmp_path):
    provider = LinqProvider(
        model_path=str(tmp_path / "missing.onnx"),
        person_model_path=str(tmp_path / "missing-person.onnx"),
    )
    provider.load()
    assert provider.mode == "local_unloaded"
    assert provider.status()["provider"] == "linq"

def test_provider_status_reports_real_execution(monkeypatch):
    from app.inference.local_onnx import LocalOnnxProvider

    provider = LocalOnnxProvider("unused", providers=["CPUExecutionProvider"])
    status = provider.status()
    assert status["provider"] == "local_onnx"
    assert status["active_providers"] == []
    linq = LinqProvider(amqp_host="10.0.114.30")
    linq.load()
    assert linq.status()["mode"] == "linq_remote"
    assert linq.status()["protocol"] == "linq-rabbitmq-tpu-v1"

def test_readiness_lifecycle(monkeypatch, tmp_path):
    provider = LinqProvider(amqp_host="10.0.114.30")
    provider.load()
    assert provider.readiness == "configured"
    assert provider.status()["readiness"] == "configured"

    monkeypatch.setattr(provider, "_connect", lambda: _FakeConnection())
    assert provider.verify() == "reachable"
    assert provider.status()["readiness"] == "reachable"

    def fail():
        raise ConnectionError("down")

    monkeypatch.setattr(provider, "_connect", fail)
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

def test_linq_remote_status_exposes_normalized_program_classes():
    provider = LinqProvider(amqp_host="10.0.114.30")
    provider.load()
    supported = provider.status()["supported_classes"]
    assert "person" in supported
    assert "no_helmet" in supported
    assert "no_vest" in supported
    assert "no_mask" in supported
    assert "smoking" not in supported
    assert "no_gloves" not in supported
