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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.domain.models import Detection
from app.inference.base import Frame, InferenceProvider
from app.services.metrics import Metrics

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
    READINESS = (
        "unconfigured",
        "configured",
        "connecting",
        "reachable",
        "inference_verified",
        "hardware_verified",
        "failed",
        "fallback",
    )

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
        max_in_flight: int = 1,
        allow_spinning: bool = False,
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
        self.allow_spinning = allow_spinning
        self.last_ppe_route: dict = {}
        # Vendor EP name may differ from the guess — override via LINQ_PROVIDERS.
        self.requested_providers = list(providers) if providers else ["LinQExecutionProvider"]
        self.thresholds = dict(thresholds) if thresholds else {}
        self.floor = floor
        self.roi_mode = roi_mode
        self.readiness = "unconfigured"
        self.stats = LinqStats(mode="uninitialized")
        self._local = None
        self._person = None
        self._loaded = False
        self._connection = None
        self._channel = None
        self._reply_queue = ""
        self._lock = Lock()
        if not 1 <= max_in_flight <= 12:
            raise ValueError("max_in_flight must be in 1..12")
        self.max_in_flight = max_in_flight
        self.metrics = Metrics()
        self._transport = None
        self._person_pool = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix="person"
        )

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
            # Remote mode runs the PPE graph on the TPU. The PPE weights have
            # no usable `person` head, so worker tracking needs a separate
            # person detector. It runs locally on CPU and this split is
            # reported honestly in status().
            self.stats.mode = "linq_remote"
            self._load_remote_person()
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
            allow_spinning=self.allow_spinning,
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
            self._rpc().call(None)
            self.readiness = "reachable"
        except Exception:
            self.readiness = "failed"
        return self.readiness

    def _load_remote_person(self) -> None:
        if not self.person_model_path:
            return
        try:
            from app.inference.local_onnx import LocalOnnxProvider

            self._person = LocalOnnxProvider(
                str(self.person_model_path),
                self.confidence,
                self.threads,
                providers=["CPUExecutionProvider"],
                thresholds=self.thresholds,
                floor=self.floor,
                roi_mode="off",
                required_classes={"person"},
                allow_spinning=self.allow_spinning,
            )
            self._person.load()
        except Exception:  # noqa: BLE001 - PPE/TPU path still works without it
            self._person = None

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
            "active_providers": (list(getattr(local, "active_providers", [])) if local else []),
            "supported_classes": self.supported_classes(),
            "model": self.model,
            "protocol": PROTOCOL if self.stats.mode == "linq_remote" else None,
            "ppe_route": getattr(self, "last_ppe_route", {}),
            "person_detector": (
                "cpu" if self.stats.mode == "linq_remote" and self._person is not None else None
            ),
            "ppe_runtime": "tpu" if self.stats.mode == "linq_remote" else "cpu",
            "hardware_verified": self.readiness == "hardware_verified",
            "worker_pid": self.stats.worker_pid,
            "worker_inference_ms": round(self.stats.worker_inference_ms, 3),
            "last_latency_ms": round(self.stats.last_latency_ms, 1),
            "frames": self.stats.frames,
            "remote_calls": self.stats.remote_calls,
            "remote_failures": self.stats.remote_failures,
            "max_in_flight": self.max_in_flight,
            "metrics": self.metrics.snapshot(),
        }

    async def predict(self, frame: Frame) -> list[Detection]:
        return await asyncio.to_thread(self.predict_sync, frame)

    def predict_sync(self, frame: Frame, on_tpu_result=None) -> list[Detection]:
        self.load()
        started = time.perf_counter()
        if self.stats.mode == "linq_remote":
            # Both branches receive the exact same immutable source frame.
            person = self._person_pool.submit(self._predict_people, frame) if self._person else None
            try:
                detections = self._predict_remote(frame)
                if on_tpu_result is not None:
                    on_tpu_result()
                if person is not None:
                    detections = person.result() + detections
            except BaseException:
                if person is not None:
                    person.cancel()
                raise
        elif self._local is not None:
            detections = self._local.predict_sync(frame)
        else:  # pragma: no cover - defensive; load() always sets a mode
            raise RuntimeError("LinQ провайдер не инициализирован.")
        self.stats.last_latency_ms = (time.perf_counter() - started) * 1000
        self.stats.frames += 1
        return detections

    def motion_capable(self):
        return (
            self._person is not None
            if self.mode == "linq_remote"
            else bool(self._local and self._local.motion_capable())
        )

    def predict_people(self, frame):
        return (
            self._predict_people(frame)
            if self.mode == "linq_remote"
            else self._local.predict_people(frame)
        )

    def predict_ppe(self, frame, people=None):
        started = time.perf_counter()
        if self.mode == "linq_remote":
            detections = self._predict_ppe_remote(frame, people or [])
        else:
            detections = self._local.predict_ppe(frame, people)
        self.stats.last_latency_ms = (time.perf_counter() - started) * 1000
        self.stats.frames += 1
        return detections

    def _predict_ppe_remote(self, frame, people):
        """Mirror LocalOnnxProvider ROI crops so remote TPU sees the same PPE path.

        ROI is only used for 1-2 strong workers with boxes big enough to be
        informative; otherwise the full frame is sent. ``last_ppe_route``
        records which path actually ran so benchmarks cannot claim ROI when
        the fallback executed.
        """
        from app.inference.local_onnx import (
            deduplicate_ppe,
            person_roi_pixels,
            project_roi_detection,
        )

        height, width = frame.shape[:2]
        strong = [d for d in people if d.class_name == "person" and not d.weak]
        reason = ""
        rois: list[tuple[int, int, int, int]] = []
        if self.roi_mode != "auto":
            reason = f"roi_mode={self.roi_mode}"
        elif not 1 <= len(strong) <= 2:
            reason = f"people={len(strong)} (roi needs 1-2)"
        else:
            rois = [person_roi_pixels(d.bbox, width, height) for d in strong]
            if not all(
                x2 - x1 >= 72 and y2 - y1 >= 72
                and (x2 - x1) * (y2 - y1) <= 0.8 * width * height
                for x1, y1, x2, y2 in rois
            ):
                reason = "roi too small or too large"
                rois = []

        if rois:
            ppe = []
            for roi in rois:
                x1, y1, x2, y2 = roi
                for det in self._predict_remote(frame[y1:y2, x1:x2]):
                    if det.class_name != "person":
                        ppe.append(project_roi_detection(det, roi, width, height))
            self.last_ppe_route = {
                "roi_executed": True,
                "roi_count": len(rois),
                "fallback_reason": None,
            }
            return deduplicate_ppe(ppe)

        self.last_ppe_route = {
            "roi_executed": False,
            "roi_count": 0,
            "fallback_reason": reason or "disabled",
        }
        return [d for d in self._predict_remote(frame) if d.class_name != "person"]

    def _predict_people(self, frame):
        with self.metrics.stage("cpu_person_ms"):
            return [d for d in self._person.predict_sync(frame) if d.class_name == "person"]

    def _rpc(self):
        from app.inference.amqp_pipeline import AmqpPipeline

        with self._lock:
            if self._transport is None:
                self._transport = AmqpPipeline(self, self.max_in_flight)
            return self._transport

    def close(self):
        with self._lock:
            transport = self._transport
        if transport is not None:
            transport.close()
        self._person_pool.shutdown(wait=True, cancel_futures=True)

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
        self._channel.queue_declare(
            queue=self._reply_queue,
            auto_delete=True,
            exclusive=True,
            arguments={"x-expires": 120000, "x-max-length": self.max_in_flight * 2},
        )
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

        started = time.perf_counter()
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
        self.metrics.observe("preprocess_ms", (time.perf_counter() - started) * 1000)
        output = self._send_tensor(tensor)
        with self.metrics.stage("postprocess_ms"):
            return self._decode(output, width, height, scale, left, top)

    def _send_tensor(self, tensor):
        from app.inference.tpu_wire import PyTask, deserialize, serialize

        started = time.perf_counter()
        try:
            with self.metrics.stage("serialization_ms"):
                body = serialize(PyTask({"images": tensor}))
            self.metrics.observe("tensor_bytes", len(body))
            with self.metrics.stage("rpc_ms"):
                reply = self._rpc().call(body)
            if len(reply) > 4 * 1024 * 1024:
                raise RuntimeError("LinQ reply exceeds the tensor size limit.")
            with self.metrics.stage("deserialize_ms"):
                data, _metadata, worker = deserialize(reply)
            if not isinstance(data, dict) or "output0" not in data:
                raise RuntimeError("LinQ TPU вернул ответ без тензора output0.")
            with self._lock:
                self.stats.worker_pid = str(worker.get("worker_pid", ""))
                self.stats.worker_status = int(worker.get("status_code", -1))
                self.stats.worker_error = str(worker.get("error_message", ""))
                self.stats.worker_inference_ms = float(worker.get("inference_time", 0.0)) * 1000
                if self.stats.worker_status != 0:
                    raise RuntimeError(f"LinQ TPU worker error {self.stats.worker_status}")
                self.stats.remote_calls += 1
                self.readiness = "inference_verified"
                worker_ms = self.stats.worker_inference_ms
            self.metrics.observe("worker_inference_ms", worker_ms)
            self.metrics.mark("tpu")
            return data["output0"]
        except Exception:
            with self._lock:
                self.stats.remote_failures += 1
                self.readiness = "failed"
            self.metrics.mark("failures")
            raise
        finally:
            self.metrics.observe("remote_total_ms", (time.perf_counter() - started) * 1000)

    def _decode(self, output, width, height, scale, left, top) -> list[Detection]:
        import numpy as np

        if output.shape != (1, 18, 8400) or not np.isfinite(output).all():
            raise RuntimeError("Invalid LinQ output tensor shape or non-finite data.")
        rows = output[0].T
        if rows.shape[1] < 5:
            raise RuntimeError("Неподдерживаемый формат выхода LinQ (ожидается YOLOv8).")
        if self.tpu_output_is_logits:
            rows = rows.copy()
            rows[:, 4:] = 1.0 / (1.0 + np.exp(-np.clip(rows[:, 4:], -80, 80)))
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
