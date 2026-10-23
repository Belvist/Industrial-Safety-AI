"""LinQ inference adapter.

Architecture (hackathon case "LinQ", OOO "ХайТэк"):

- Downstream modules (tracking -> rules -> incidents -> SQLite/Telegram/UI)
  only consume normalized :class:`Detection` objects. Switching hardware
  never changes them.
- Three execution modes, selected by environment:
  1. ``linq_remote`` — online accelerator access provided by organizers:
     ``LINQ_ENDPOINT`` (+ optional ``LINQ_API_KEY``). Frames are JPEG-encoded
     and POSTed to ``{endpoint}/v1/detect``; the response is normalized to
     ``Detection``. Used at the hackathon.
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
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from app.domain.models import BoundingBox, Detection
from app.inference.base import Frame, InferenceProvider

REMOTE_PATH = "/v1/detect"


@dataclass
class LinqStats:
    mode: str
    last_latency_ms: float = 0.0
    frames: int = 0
    remote_calls: int = 0
    remote_failures: int = 0


class LinqProvider(InferenceProvider):
    """Drop-in inference adapter for LinQ accelerators with dev fallback.

    Readiness is separate from configuration: an endpoint being set
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
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.threads = threads
        self.person_model_path = person_model_path
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.model = model
        # Vendor EP name may differ from the guess — override via LINQ_PROVIDERS.
        self.requested_providers = list(providers) if providers else ["LinQExecutionProvider"]
        self.thresholds = dict(thresholds) if thresholds else {}
        self.floor = floor
        self.roi_mode = roi_mode
        self.readiness = "unconfigured"
        self.stats = LinqStats(mode="uninitialized")
        self._local = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "linq"

    @property
    def mode(self) -> str:
        return self.stats.mode

    def load(self) -> None:
        if self._loaded:
            return
        if self.endpoint:
            # Remote mode needs no local weights; connectivity is checked lazily
            # per frame so the UI can start without blocking on the network.
            # Configured proves nothing: health -> reachable, a completed
            # detect -> inference_verified, never active before that.
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
        """Probe the configured remote endpoint (placeholder health check).

        Transitions configured/connecting -> reachable/failed. A passed probe
        proves the endpoint answers, never that inference ran on an
        accelerator. Until organizers publish the real API this stays a
        placeholder probe.
        """
        self.load()
        if not self.endpoint:
            self.readiness = "fallback" if self._local is not None else "failed"
            return self.readiness
        self.readiness = "connecting"
        request = urllib.request.Request(
            self.endpoint + "/health", headers={"Accept": "application/json"}
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
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
        provider is really in the session's active providers. Remote
        execution cannot be confirmed without server telemetry (documented)."""
        active = getattr(self._local, "active_providers", []) if self._local else []
        wanted = [p for p in self.requested_providers if p != "CPUExecutionProvider"]
        return any(p in active for p in wanted)

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
            "endpoint_configured": bool(self.endpoint),
            "requested_providers": self.requested_providers,
            "active_providers": (
                list(getattr(local, "active_providers", [])) if local else []
            ),
            "supported_classes": (
                local.supported_classes() if local is not None else []
            ),
            "model": self.model,
            "last_latency_ms": round(self.stats.last_latency_ms, 1),
            "frames": self.stats.frames,
            "remote_failures": self.stats.remote_failures,
        }

    async def predict(self, frame: Frame) -> list[Detection]:
        return await asyncio.to_thread(self.predict_sync, frame)

    def predict_sync(self, frame: Frame) -> list[Detection]:
        self.load()
        started = time.perf_counter()
        try:
            if self.stats.mode == "linq_remote":
                detections = self._predict_remote(frame)
            elif self._local is not None:
                detections = self._local.predict_sync(frame)
            else:  # pragma: no cover - defensive; load() always sets a mode
                raise RuntimeError("LinQ провайдер не инициализирован.")
        finally:
            pass
        self.stats.last_latency_ms = (time.perf_counter() - started) * 1000
        self.stats.frames += 1
        return detections

    # -- remote protocol ----------------------------------------------------

    def _predict_remote(self, frame: Frame) -> list[Detection]:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:  # pragma: no cover - cv2 encode failure is environment specific
            raise RuntimeError("Не удалось закодировать кадр для LinQ.")
        payload = json.dumps(
            {"model": self.model, "confidence": self.confidence}
        ).encode()
        request = urllib.request.Request(
            self.endpoint + REMOTE_PATH,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # JPEG bytes travel as a second request in multipart form would; the MVP
        # protocol keeps it simple: JSON meta + octet-stream body override.
        # Organizers' final schema may differ — only _parse_remote changes then.
        request.add_header("X-LinQ-Image-Bytes", str(len(encoded)))
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        # NOTE: urllib sends one body; image bytes are appended after a newline
        # separator so a minimal server can split meta/frame without multipart.
        body = payload + b"\n" + encoded.tobytes()
        # Rebuild with the full body (headers preserved).
        request = urllib.request.Request(
            self.endpoint + REMOTE_PATH,
            data=body,
            headers=dict(request.header_items()),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
        except Exception as exc:
            self.stats.remote_failures += 1
            self.readiness = "failed"
            raise RuntimeError(f"LinQ endpoint недоступен: {exc}") from exc
        self.stats.remote_calls += 1
        # A completed detect round-trip confirms inference ran somewhere.
        # Whether that was the accelerator needs server telemetry the
        # placeholder protocol does not provide (documented limitation).
        self.readiness = "inference_verified"
        parsed = self._parse_remote(data)
        # Same display/violation split as locally (F2): below floor dropped,
        # below class threshold kept flagged weak, never downstream-usable.
        kept = []
        for det in parsed:
            if det.confidence < min(self.threshold_for(det.class_name), self.floor):
                continue
            if det.confidence < self.threshold_for(det.class_name):
                det = replace(det, weak=True)
            kept.append(det)
        return kept

    @staticmethod
    def _parse_remote(data: dict) -> list[Detection]:
        """Normalize ``{detections: [{label, confidence, bbox:[x1,y1,x2,y2]}]}``.

        Only this method changes when organizers publish the final schema.
        """
        from app.inference.local_onnx import ALIASES

        detections: list[Detection] = []
        items = data.get("detections", []) if isinstance(data, dict) else []
        for item in items:
            try:
                label = str(item.get("label", "")).strip().lower()
                class_name = ALIASES.get(label, label)
                confidence = float(item.get("confidence", 0))
                x1, y1, x2, y2 = (float(v) for v in item["bbox"])
                detections.append(
                    Detection(class_name, confidence, BoundingBox(x1, y1, x2, y2))
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Некорректный ответ LinQ: {exc}") from exc
        return detections
