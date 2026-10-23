"""LinQ inference adapter.

Architecture (hackathon case "LinQ", OOO "ХайТэк"):

- Downstream modules (tracking -> rules -> incidents -> SQLite/Telegram/UI)
  only consume normalized :class:`Detection` objects. Switching hardware
  never changes them.
- Three execution modes, selected by environment:
  1. ``linq_remote`` — the organizers' TPU service: frames are letterboxed
     and sent as an int8 tensor over RabbitMQ (``tpu.direct`` / ``input``).
     The worker runs a compiled ``.tpu`` program from the shared cache and
     answers on the reply queue named in ``response_routing_keys``.
  2. ``linq_local`` — ONNX Runtime with a LinQ execution provider when the
     vendor package is installed (``onnxruntime-linq`` exposing
     ``LinQExecutionProvider``). Falls back to CPU with a warning.
  3. ``local_onnx`` fallback — while organizers' access is not issued, the
     provider delegates to the dev ONNX weights so the whole product loop
     (cameras -> incidents -> Telegram) stays testable.

Only this file knows about LinQ wire details. Everything else is hardware
agnostic by design (see docs/ARCHITECTURE.md).
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.domain.models import Detection
from app.inference.base import Frame, InferenceProvider

PROTOCOL = "linq-rabbitmq-tpu-v1"

@dataclass
class LinqStats:
    mode: str
    last_latency_ms: float = 0.0
    frames: int = 0
    remote_calls: int = 0
    remote_failures: int = 0
    worker_inference_ms: float = 0.0
    worker_pid: str = ""
    worker_status: int = -1
    worker_error: str = ""

class LinqProvider(InferenceProvider):
    """Drop-in inference adapter for LinQ accelerators with dev fallback.

    Readiness is separate from configuration: a configured broker being set
    ("configured") never means the accelerator works. A passed health probe
    means "reachable"; a completed detect round-trip means
    "inference_verified"; only vendor telemetry or a local session EP check
    means "hardware_verified". A registered EP name alone is never proof.
    """

    # unconfigured: nothing to run on; configured: endpoint set, unprobed;
    # connecting: probe in flight; reachable: health answers; inference_verified:
    # a detect round-trip completed; hardware_verified: execution on the
    # required accelerator confirmed; failed: proven broken; fallback: local.
    READINESS = ("unconfigured", "configured", "connecting", "reachable", "inference_verified", "hardware_verified", "failed", "fallback")

    def __init__(
        self,
        model_path: str = "models/ppe.onnx",
        confidence: float = 0.4,
        threads: int = 2,
        person_model_path: str | None = None,
        endpoint: str = "",
        api_key: str = "",
        timeout_seconds: float = 8.0,
        model: str = "ppe-linq",
        providers: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        floor: float = 0.25,
        roi_mode: str = "auto",
        amqp_host: str = "",
        amqp_port: int = 5672,
        amqp_user: str = "guest",
        amqp_password: str = "guest",
        amqp_vhost: str = "/",
        amqp_exchange: str = "tpu.direct",
        amqp_input_queue: str = "input",
        tpu_program: str = "ppe_nosig_p999.tpu",
        tpu_cache_dir: str = "/auto/cache/hackathon93154",
        tpu_output_is_logits: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.threads = threads
        self.person_model_path = person_model_path
        # ``endpoint`` is kept for backward compatibility and means the AMQP
        # host when set (e.g. LINQ_ENDPOINT=10.0.114.30). It never triggers the
        # removed placeholder HTTP path.
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.model = model
        # ``endpoint`` is a deprecated alias; only a bare host is accepted.
        self.amqp_host = amqp_host or (endpoint if endpoint and "://" not in endpoint else "")
        self.amqp_port = amqp_port
        self.amqp_user = amqp_user
        self.amqp_password = amqp_password
        self.amqp_vhost = amqp_vhost
        self.amqp_exchange = amqp_exchange
        self.amqp_input_queue = amqp_input_queue
        self.tpu_program = tpu_program
        self.tpu_cache_dir = tpu_cache_dir.rstrip("/")
        self.tpu_output_is_logits = tpu_output_is_logits
        # Vendor EP name may differ from the guess — override via LINQ_PROVIDERS.
        self.requested_providers = list(providers) if providers else ["LinQExecutionProvider"]
        self.thresholds = dict(thresholds) if thresholds else {}
        self.floor = floor
        self.roi_mode = roi_mode
        self.readiness = "unconfigured"
        self.stats = LinqStats(mode="uninitialized")
        self._local = None
        self._loaded = False
        self._connection = None
        self._channel = None
        self._reply_queue = ""
        self._lock = Lock()

    @property
    def name(self) -> str:
        return "linq"

    @property
    def mode(self) -> str:
        return self.stats.mode

    @property
    def program_uri(self) -> str:
        return f"file://{self.tpu_cache_dir}/{self.tpu_program}?raw=False"

    def load(self) -> None:
        if self._loaded:
            return
        if self.amqp_host:
            # Remote mode needs no local weights; the broker is connected
            # lazily per frame so the UI can start without blocking.
            self.stats.mode = "linq_remote"
            self.readiness = "configured"
            self._loaded = True
            return
        # Local mode: prefer a LinQ execution provider, fall back to CPU.
        # Imported lazily so a missing vendor package never breaks `local_onnx`.
        try:
            from app.inference.local_onnx import LocalOnnxProvider
        except ImportError as exc:  # pragma: no cover - import-time failure
            raise RuntimeError(f"LinQ local fallback unavailable: {exc}") from exc
        self._local = LocalOnnxProvider(
            str(self.model_path),
            self.confidence,
            self.threads,
            person_model_path=self.person_model_path,
            providers=self.requested_providers,
            thresholds=self.thresholds,
            floor=self.floor,
            roi_mode=self.roi_mode,
        )
        try:
            self._local.load()
        except RuntimeError:
            # No dev weights yet (fresh clone): stay loadable so /system/info
            # reports a clear message instead of crashing the app. First real
            # predict() will surface the same message via ready().
            self.stats.mode = "local_unloaded"
            self.readiness = "failed"
            self._loaded = True
            return
        self.stats.mode = self._pick_local_mode()
        if self.stats.mode == "linq_local":
            # Local session proof: the requested EP really took the graph.
            self.readiness = "hardware_verified"
        else:
            self.readiness = "fallback"
        self._loaded = True

    def verify(self) -> str:
        """Probe the AMQP broker.

        Transitions configured/connecting -> reachable/failed. A passed probe
        proves the broker answers, never that inference ran on an accelerator.
        """
        self.load()
        if not self.amqp_host:
            self.readiness = "fallback" if self._local is not None else "failed"
            return self.readiness
        self.readiness = "connecting"
        try:
            with self._lock:
                self._connect()
            self.readiness = "reachable"
        except Exception:
            self.readiness = "failed"
        return self.readiness

    def _pick_local_mode(self) -> str:
        # Mode by FACT from the loaded session, not by global availability:
        # a requested EP may be installed yet refused for this graph/device.
        active = getattr(self._local, "active_providers", []) if self._local else []
        if any(p in active for p in self.requested_providers):
            return "linq_local"
        return "local_onnx"

    def accelerator_confirmed(self) -> bool:
        """True only with local session proof: a requested non-CPU execution
        provider is really in the session's active providers. Remote TPU
        execution is confirmed by worker telemetry (status + inference time),
        reported separately as ``worker_verified``."""
        active = getattr(self._local, "active_providers", []) if self._local else []
        wanted = [p for p in self.requested_providers if p != "CPUExecutionProvider"]
        return any(p in active for p in wanted)

    def worker_verified(self) -> bool:
        """A TPU worker answered and reported successful inference.

        This proves the remote program executed. It does not expose hardware
        counters; the vendor worker's ``status_code`` and ``inference_time``
        are the only telemetry the protocol provides.
        """
        return (
            self.stats.mode == "linq_remote"
            and self.stats.remote_calls > 0
            and self.stats.worker_status == 0
            and self.stats.worker_inference_ms > 0
        )

    def supported_classes(self) -> list[str]:
        """Classes the active backend can actually emit.

        Remote mode returns the compiled program's classes (normalized to the
        same names as the local model), so the UI enables exactly the rules
        that can fire.
        """
        if self.stats.mode == "linq_remote":
            return sorted(set(self._class_map().values()))
        if self._local is not None:
            return self._local.supported_classes()
        return []

    def threshold_for(self, class_name: str) -> float:
        """Same per-class contract as LocalOnnxProvider: a weak remote
        no_helmet must not reach downstream just because it came from LinQ."""
        return self.thresholds.get(class_name, self.confidence)

    def status(self) -> dict:
        local = self._local
        return {
            "provider": "linq",
            "mode": self.stats.mode,
            "readiness": self.readiness,
            "accelerator_confirmed": self.accelerator_confirmed(),
            "worker_verified": self.worker_verified(),
            "endpoint_configured": bool(self.amqp_host),
            "amqp": {
                "host": self.amqp_host,
                "port": self.amqp_port,
                "exchange": self.amqp_exchange,
                "input_queue": self.amqp_input_queue,
            },
            "program": self.tpu_program,
            "program_uri": self.program_uri,
            "requested_providers": self.requested_providers,
            "active_providers": (
                list(getattr(local, "active_providers", [])) if local else []
            ),
            "supported_classes": self.supported_classes(),
            "model": self.model,
            "protocol": PROTOCOL if self.stats.mode == "linq_remote" else None,
            "hardware_verified": self.readiness == "hardware_verified",
            "worker_pid": self.stats.worker_pid,
            "worker_inference_ms": round(self.stats.worker_inference_ms, 3),
            "last_latency_ms": round(self.stats.last_latency_ms, 1),
            "frames": self.stats.frames,
            "remote_calls": self.stats.remote_calls,
            "remote_failures": self.stats.remote_failures,
        }

    async def predict(self, frame: Frame) -> list[Detection]:
        return await asyncio.to_thread(self.predict_sync, frame)

    def predict_sync(self, frame: Frame) -> list[Detection]:
        self.load()
        started = time.perf_counter()
        if self.stats.mode == "linq_remote":
            detections = self._predict_remote(frame)
        elif self._local is not None:
            detections = self._local.predict_sync(frame)
        else:  # pragma: no cover - defensive; load() always sets a mode
            raise RuntimeError("LinQ провайдер не инициализирован.")
        self.stats.last_latency_ms = (time.perf_counter() - started) * 1000
        self.stats.frames += 1
        return detections

    # -- RabbitMQ protocol --------------------------------------------------

    def _connect(self):
        if self._channel is not None and getattr(self._connection, "is_open", False):
            return self._channel
        try:
            import pika
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Для LinQ TPU нужен пакет pika. Установите extra: pip install -e '.[linq]'."
            ) from exc
        credentials = pika.PlainCredentials(self.amqp_user, self.amqp_password)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.amqp_host,
                port=self.amqp_port,
                virtual_host=self.amqp_vhost,
                credentials=credentials,
                heartbeat=60,
                blocked_connection_timeout=self.timeout_seconds,
                socket_timeout=self.timeout_seconds,
            )
        )
        self._channel = self._connection.channel()
        self._reply_queue = f"linq_{uuid.uuid4().hex[:12]}"
        self._channel.queue_declare(queue=self._reply_queue, auto_delete=True)
        self._channel.queue_bind(
            exchange=self.amqp_exchange,
            queue=self._reply_queue,
            routing_key=self._reply_queue,
        )
        return self._channel

    def _close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass
        finally:
            self._connection = None
            self._channel = None
            self._reply_queue = ""

    def _predict_remote(self, frame: Frame) -> list[Detection]:
        """Letterbox, send the int8-ready float tensor, decode the TPU answer.

        The compiled program was exported without the final Sigmoid, so the
        class rows are logits. ``tpu_output_is_logits`` makes that explicit;
        the local ONNX decode path is reused unchanged after applying sigmoid.
        """
        import cv2

        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        nw, nh = round(width * scale), round(height * scale)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        import numpy as np

        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(frame, (nw, nh))
        tensor = (
            np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        output = self._send_tensor(tensor)
        return self._decode(output, width, height, scale, left, top)

    def _send_tensor(self, tensor):
        import pika

        from app.inference.tpu_wire import (
            MESSAGE_ID_ATTR_NAME,
            RESPONSE_ROUTING_KEYS_ATTR_NAME,
            TARGET_URI_ATTR_NAME,
            PyTask,
            deserialize,
            serialize,
        )

        with self._lock:
            channel = self._connect()
            task_id = uuid.uuid4().hex
            headers = {
                TARGET_URI_ATTR_NAME: self.program_uri,
                RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self._reply_queue]),
                MESSAGE_ID_ATTR_NAME: task_id,
            }
            channel.basic_publish(
                exchange=self.amqp_exchange,
                routing_key=self.amqp_input_queue,
                body=serialize(PyTask({"images": tensor})),
                properties=pika.BasicProperties(headers=headers),
            )
            deadline = time.monotonic() + self.timeout_seconds
            try:
                while time.monotonic() < deadline:
                    method, properties, body = channel.basic_get(
                        self._reply_queue, auto_ack=True
                    )
                    if method is None:
                        time.sleep(0.005)
                        continue
                    reply_headers = properties.headers or {}
                    if reply_headers.get(MESSAGE_ID_ATTR_NAME) != task_id:
                        continue
                    data, _metadata, worker = deserialize(body)
                    self.stats.worker_pid = str(worker.get("worker_pid", ""))
                    self.stats.worker_status = int(worker.get("status_code", -1))
                    self.stats.worker_error = str(worker.get("error_message", ""))
                    self.stats.worker_inference_ms = (
                        float(worker.get("inference_time", 0.0)) * 1000
                    )
                    self.stats.remote_calls += 1
                    if self.stats.worker_status != 0:
                        self.stats.remote_failures += 1
                        self.readiness = "failed"
                        raise RuntimeError(
                            f"LinQ TPU worker error {self.stats.worker_status}: "
                            f"{self.stats.worker_error or 'без описания'}"
                        )
                    self.readiness = "inference_verified"
                    if not isinstance(data, dict) or "output0" not in data:
                        raise RuntimeError("LinQ TPU вернул ответ без тензора output0.")
                    return data["output0"]
            except RuntimeError:
                raise
            except Exception as exc:
                self.stats.remote_failures += 1
                self.readiness = "failed"
                self._close()
                raise RuntimeError(f"LinQ TPU недоступен: {exc}") from exc
            self.stats.remote_failures += 1
            self.readiness = "failed"
            self._close()
            raise RuntimeError(
                f"LinQ TPU не ответил за {self.timeout_seconds:.1f} с "
                f"(program={self.tpu_program})."
            )

    def _decode(self, output, width, height, scale, left, top) -> list[Detection]:
        import numpy as np

        rows = output[0].T
        if rows.shape[1] < 5:
            raise RuntimeError("Неподдерживаемый формат выхода LinQ (ожидается YOLOv8).")
        if self.tpu_output_is_logits:
            rows = rows.copy()
            rows[:, 4:] = 1.0 / (1.0 + np.exp(-rows[:, 4:]))
        # Reuse the exact local decode contract (class map, thresholds, NMS,
        # letterbox projection) by delegating to a lightweight provider.
        from app.inference.local_onnx import LocalOnnxProvider

        decoder = LocalOnnxProvider(
            str(self.model_path),
            self.confidence,
            self.threads,
            person_model_path=None,
            required_classes={"person"},
            thresholds=self.thresholds,
            floor=self.floor,
            roi_mode="off",
        )
        decoder.classes = self._class_map()
        return decoder.decode(rows.T[None], width, height, scale, left, top)

    def _class_map(self) -> dict[int, str]:
        # Class order of models/ppe.onnx (ONNX metadata ``names``). Kept here
        # because the remote worker returns only the raw output tensor.
        # Normalize exactly like LocalOnnxProvider so rules and thresholds
        # match between the local and TPU paths.
        from app.inference.local_onnx import ALIASES

        return {
            index: ALIASES.get(
                name.lower().replace("_", "-"), ALIASES.get(name.lower(), name.lower())
            )
            for index, name in enumerate(TPU_CLASSES)
        }

TPU_CLASSES = (
    "fall-detected",
    "gloves",
    "goggles",
    "helmet",
    "ladder",
    "mask",
    "no-gloves",
    "no-goggles",
    "no_helmet",
    "no-mask",
    "no_vest",
    "person",
    "safety cone",
    "vest",
)
