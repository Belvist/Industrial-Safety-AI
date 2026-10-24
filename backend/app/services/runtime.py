import asyncio
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from app.inference.local_onnx import LocalOnnxProvider
from app.services.behavior import RunningDetector
from app.services.engine import (
    TemporalEngine,
    Tracker,
    assign_detections,
    cleared_assigned,
    evaluate_assigned,
)
from app.services.independent_ppe import IndependentPpe, observation_quality
from app.services.metrics import Metrics
from app.services.notifications import Notifier
from app.services.pairing import PairingStore
from app.services.replay import replay_frames
from app.services.safety_radar import SafetyRadar
from app.services.schemas import CameraConfig
from app.services.state_machine import CameraEvent, CameraMachine, CameraState
from app.services.storage import Storage, utcnow


class DiscardedFrame(RuntimeError):
    """A stale frame must not fail a newer camera session."""


class CameraSession:
    def __init__(self, camera_id, config):
        self.id, self.config = camera_id, config
        self.tracker, self.engine = Tracker(), TemporalEngine()
        self.behavior = RunningDetector()
        self.lock = threading.Lock()
        self.frame_gate = threading.BoundedSemaphore(1)
        self.condition = threading.Condition(self.lock)
        self.input_sequence = 0
        self.next_commit = 1
        self.source_frame_id = 0
        self.browser_origin = None
        self.generation = 0
        self.metrics = Metrics()
        self.source_wall = time.time()
        self.last_timestamp = None
        self.source_origin = None
        self.control = threading.RLock()
        self.machine = CameraMachine()
        self.session_id = None
        self.sequence = 0
        self.frames = deque(maxlen=150)  # ~5-30s ring buffer for Evidence Replay
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.kind = None
        self.jpeg = None
        self.latest = None
        self.last_frame = None
        self.last_tick = None
        self.upload_path = None
        self.capture = None
        # Display-only confidence smoothing (EMA per tracked object + class).
        # Raw detections keep flowing into the tracker/engine untouched; only
        # the numbers shown on the overlay and in session.latest are smoothed
        # so the UI stops flickering frame to frame.
        self.display_conf: dict[tuple[str, str], tuple[float, float]] = {}
        self.radar_events: list = []
        self.radar_evidence: str | None = None
        self.radar_evidence_by_track: dict[tuple[int, str, str], str] = {}
        # Last confirmed PPE state per track: avoids UI flicker when the
        # TPU misses a helmet/vest on an individual frame.
        self.ppe_memory: dict[int, tuple[dict, dict, float]] = {}
        self.independent_ppe = IndependentPpe()
        self.radar = SafetyRadar()

    @property
    def status(self):
        return self.machine.state

    def _snapshot(self):
        latest = self.latest
        if latest is not None:
            age = latest.get("data_age_ms", 0) + max(
                0, (time.monotonic() - (self.last_frame or time.monotonic())) * 1000
            )
            latest = {**latest, "data_age_ms": round(age, 1)}
        return {
            "id": self.id,
            "frame_capacity": getattr(self, "frame_capacity", 1),
            **self.config.model_dump(),
            "status": self.status,
            "error": self.error,
            "kind": self.kind,
            "latest": latest,
            "session_id": self.session_id,
            "state_version": self.machine.version,
        }

    def summary(self):
        with self.lock:
            return self._snapshot()


RULE_CLASSES = {
    # A rule needs its VIOLATION class present: merely detecting `helmet`
    # says nothing about `no_helmet` absence, so overlap is not enough.
    "helmet": {"no_helmet"},
    "vest": {"no_vest"},
    "mask": {"no_mask"},
    "smoking": {"smoking"},
}


def unsupported_rules(config, supported: set[str]) -> list[str]:
    """Rule keys enabled in config whose classes the model cannot emit.

    A rule needs its violation class present (e.g. helmet rule needs
    no_helmet); enabling it otherwise would pretend monitoring that can
    never fire. Pure helper, unit-tested without weights.
    """
    bad = []
    for rule, needed in RULE_CLASSES.items():
        if getattr(config, rule, False) and not (needed & set(supported)):
            bad.append(rule)
    return bad


class Runtime:
    def __init__(self, settings, provider=None):
        self.settings = settings
        self.metrics = Metrics()
        self._evidence_slots = threading.BoundedSemaphore(4)
        self.storage = Storage(settings.storage_path)
        self.data = settings.resolve(settings.data_dir)
        (self.data / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.data / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data / "evidence").mkdir(parents=True, exist_ok=True)
        self._evidence_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence")
        self.storage.purge_old_resolved(settings.snapshot_retention_days, self.data / "snapshots")
        self.notifier = Notifier(settings, self.storage)
        self.pairing = PairingStore(
            settings.pairing_code_ttl_seconds, settings.phone_token_ttl_hours * 3600
        )
        self.provider = provider
        self.model_error = None
        if self.provider is None:
            if settings.inference_provider == "linq":
                from app.inference.linq import LinqProvider

                self.provider = LinqProvider(
                    str(settings.resolve(settings.model_path)),
                    settings.confidence_threshold,
                    settings.inference_threads,
                    person_model_path=(
                        str(settings.resolve(settings.person_model_path))
                        if settings.use_separate_person_model
                        else None
                    ),
                    endpoint=settings.linq_endpoint,
                    api_key=settings.linq_api_key,
                    timeout_seconds=settings.linq_timeout_seconds,
                    model=settings.linq_model,
                    providers=settings.linq_provider_list,
                    thresholds=settings.class_thresholds,
                    floor=settings.display_floor,
                    roi_mode=settings.ppe_roi_mode,
                    amqp_host=settings.linq_amqp_host,
                    amqp_port=settings.linq_amqp_port,
                    amqp_user=settings.linq_amqp_user,
                    amqp_password=settings.linq_amqp_password,
                    amqp_vhost=settings.linq_amqp_vhost,
                    amqp_exchange=settings.linq_amqp_exchange,
                    amqp_input_queue=settings.linq_amqp_input_queue,
                    tpu_program=settings.linq_tpu_program,
                    tpu_cache_dir=settings.linq_tpu_cache_dir,
                    tpu_output_is_logits=settings.linq_tpu_output_is_logits,
                    max_in_flight=settings.linq_max_in_flight,
                )
                try:
                    self.provider.load()
                    if getattr(self.provider, "stats", None) is not None and (
                        self.provider.stats.mode == "local_unloaded"
                    ):
                        raise RuntimeError(
                            "Модель не найдена. Запустите python scripts/download_model.py."
                        )
                except Exception as exc:  # noqa: BLE001 - expose readiness without crashing UI
                    self.model_error = str(exc)
            else:
                self.provider = LocalOnnxProvider(
                    str(settings.resolve(settings.model_path)),
                    settings.confidence_threshold,
                    settings.inference_threads,
                    person_model_path=(
                        str(settings.resolve(settings.person_model_path))
                        if settings.use_separate_person_model
                        else None
                    ),
                    providers=settings.onnx_provider_list,
                    thresholds=settings.class_thresholds,
                    floor=settings.display_floor,
                    roi_mode=settings.ppe_roi_mode,
                )
                try:
                    self.provider.load()
                except Exception as exc:  # noqa: BLE001 - expose model readiness without crashing the UI
                    self.model_error = str(exc)
        self.independent_motion = bool(
            settings.independent_motion_enabled
            and callable(getattr(self.provider, "motion_capable", None))
            and self.provider.motion_capable()
        )
        self._ppe_pool = ThreadPoolExecutor(
            max_workers=max(1, min(2, settings.linq_max_in_flight)),
            thread_name_prefix="independent-ppe",
        )
        self.sessions = {}
        self.sessions_lock = threading.RLock()
        self.shutdown = threading.Event()
        cameras = self.storage.cameras()
        if not cameras:
            config = CameraConfig(
                confirm_seconds=settings.incident_confirm_seconds,
                clear_seconds=settings.incident_clear_seconds,
                temporal_window=settings.temporal_window,
                temporal_required=settings.temporal_required,
                helmet=settings.rule_helmet,
                vest=settings.rule_vest,
                mask=settings.rule_mask,
                smoking=settings.rule_smoking,
                zone_enabled=settings.rule_zone_enabled,
            )
            cameras = [{"id": self.storage.save_camera(config), **config.model_dump()}]
        for camera in cameras:
            self.sessions[camera["id"]] = CameraSession(camera["id"], CameraConfig(**camera))

        self.watchdog = threading.Thread(target=self._watch_sources, daemon=True)
        self.watchdog.start()

    def _watch_sources(self):
        while not self.shutdown.wait(0.5):
            with self.sessions_lock:
                sessions = list(self.sessions.values())
            for session in sessions:
                if not session.lock.acquire(blocking=False):
                    continue
                try:
                    if (
                        session.status not in {CameraState.RUNNING, CameraState.STALE}
                        or session.last_frame is None
                    ):
                        continue
                    silence = time.monotonic() - session.last_frame
                    if silence > 15:
                        session.stop.set()
                        session.machine.dispatch(CameraEvent.FAIL)
                        session.error = "Источник не передаёт кадры. Перезапустите мониторинг."
                        self._interrupt(session, "source_timeout")
                    elif silence > 5 and session.status == CameraState.RUNNING:
                        session.machine.dispatch(CameraEvent.STALE)
                finally:
                    session.lock.release()

    def session(self, camera_id):
        if camera_id not in self.sessions:
            raise KeyError(camera_id)
        return self.sessions[camera_id]

    def ready(self):
        if self.model_error or self.provider is None:
            raise RuntimeError(self.model_error or "Детектор не готов.")

    def reserve_upload(self, camera_id):
        self.ready()
        session = self.session(camera_id)
        with session.control:
            with session.lock:
                if session.status not in {
                    CameraState.STOPPED,
                    CameraState.ENDED,
                    CameraState.ERROR,
                }:
                    raise RuntimeError("Сначала остановите текущий источник.")
            self.stop(camera_id, revoke_tokens=False)
            with session.lock:
                session.machine.dispatch(CameraEvent.UPLOAD)
                session.session_id = uuid4().hex
                session.latest = None
                session.frames.clear()
                session.error = None
                return session.session_id

    def finish_upload(self, camera_id, upload_id, path):
        session = self.session(camera_id)
        with session.control:
            if session.session_id != upload_id or session.status != CameraState.UPLOADING:
                raise RuntimeError("Загрузка видео была отменена.")
            session.upload_path = path
            return self.start(camera_id, "video", str(path), expected_upload=upload_id)

    def cancel_upload(self, camera_id, upload_id):
        session = self.session(camera_id)
        with session.control:
            if session.session_id == upload_id and session.status == CameraState.UPLOADING:
                self.stop(camera_id, revoke_tokens=False)

    def check_rules_supported(self, config) -> list[str]:
        """Enabled rules the loaded model cannot serve (empty = all servable).

        Fail-open when the model failed to load: without weights we cannot
        judge support, and camera management must keep working (CI has no
        weights). With a loaded model the check is strict.
        """
        if self.model_error is not None:
            return []
        try:
            supported = set(self.provider.supported_classes())
        except Exception:
            return []
        return unsupported_rules(config, supported)

    def start(self, camera_id, kind, source=None, expected_upload=None):
        self.ready()
        session = self.session(camera_id)
        with session.control:
            if session.status == CameraState.UPLOADING and expected_upload != session.session_id:
                raise RuntimeError("Дождитесь загрузки видео или отмените её.")
            self.stop(camera_id, revoke_tokens=False)
            with session.lock:
                session.stop.clear()
                session.tracker, session.engine = Tracker(), TemporalEngine()
                session.behavior = RunningDetector()
                session.radar = SafetyRadar()
                session.radar_evidence_by_track.clear()
                session.machine.dispatch(CameraEvent.START)
                session.kind, session.error = kind, None
                session.session_id = uuid4().hex
                session.sequence = 0
                session.input_sequence, session.next_commit = 0, 1
                session.source_frame_id = 0
                session.browser_origin = None
                session.frame_capacity = (
                    self.settings.linq_max_in_flight
                    if not self.independent_motion and getattr(self.provider, "name", "") == "linq"
                    else 1
                )
                session.frame_gate = threading.BoundedSemaphore(session.frame_capacity)
                session.metrics = Metrics()
                session.source_wall = time.time()
                session.last_timestamp = None
                session.source_origin = None
                session.ppe_memory.clear()
                session.display_conf.clear()
                session.frames.clear()
                session.latest, session.jpeg, session.last_tick = None, None, None
                session.last_frame = time.monotonic()
                if kind == "browser":
                    session.machine.dispatch(CameraEvent.READY)
                    return session._snapshot()
                session.thread = threading.Thread(
                    target=self._source_loop, args=(session, source), daemon=True
                )
                session.thread.start()
                return session._snapshot()

    def stop(
        self, camera_id, reason="monitoring_stopped", expected_session=None, revoke_tokens=True
    ):
        session = self.session(camera_id)
        with session.control:
            with session.lock:
                if expected_session and expected_session != session.session_id:
                    raise RuntimeError("Сессия камеры уже изменилась. Обновите состояние.")
                session.stop.set()
                if session.status != CameraState.STOPPED and session.status != CameraState.STOPPING:
                    session.machine.dispatch(CameraEvent.STOP)
            if session.thread and session.thread is not threading.current_thread():
                session.thread.join(timeout=7)
                if session.thread.is_alive():
                    raise RuntimeError(
                        "Источник ещё завершает чтение. Повторите остановку через несколько секунд."
                    )
            with session.lock:
                self._interrupt(session, reason)
                if session.status == CameraState.STOPPING:
                    session.machine.dispatch(CameraEvent.STOPPED)
                session.latest = None
                session.jpeg = None
                session.frames.clear()
                session.error = None
                # Explicit stops end the session: phone tokens bound to this
                # camera die with it. Internal restarts pass revoke_tokens=False.
                if revoke_tokens:
                    self.pairing.revoke_camera(camera_id)
                return session._snapshot()

    @staticmethod
    def _zone_for_track(tracks, track_id):
        for track in tracks:
            if track.id == track_id and getattr(track, "zone_name", None):
                return track.zone_name
        return None

    def _finish(self, incident_id, outcome, reason, duration, session=None, timestamp=None):
        """Finish once. Two tracks may share one merged incident."""
        if not incident_id:
            return
        current = self.storage.get(incident_id)
        if not current or current["status"] != "active":
            return
        finished = self.storage.finish_incident(
            incident_id,
            outcome,
            reason,
            duration=duration,
            ended_at=self._source_datetime(session, timestamp)
            if session is not None and timestamp is not None
            else None,
        )
        self.storage.record_event(
            finished["camera_id"],
            outcome,
            track_id=finished["track_id"],
            zone_name=finished["zone_name"],
            incident_id=incident_id,
            episode_id=finished.get("episode_id"),
            payload={"reason": reason},
            source_seconds=max(0, timestamp - (session.source_origin or 0))
            if session is not None and timestamp is not None
            else None,
            occurred_at=self._source_datetime(session, timestamp)
            if session is not None and timestamp is not None
            else None,
        )

    def _save_evidence(self, session, events):
        """Queue an MP4 clip around a radar event (never blocks the pipeline).

        The clip contains buffered frames before the event plus frames that
        arrive during a short post-event window. Writing happens on a
        background thread; failures are swallowed and logged by the caller.
        """
        if not events or not session.frames:
            return None
        if not self._evidence_slots.acquire(blocking=False):
            self.metrics.mark("evidence_dropped")
            return None
        try:
            first_event = events[0]
            name = f"radar_{session.id}_{int(first_event.timestamp * 1000)}_{uuid4().hex[:6]}.mp4"
            snapshot = list(session.frames)
            self._evidence_pool.submit(
                self._evidence_job,
                name,
                snapshot,
                session,
                session.session_id,
                first_event.timestamp,
                session.source_origin,
            )
            return name
        except Exception:  # noqa: BLE001 - evidence must not break monitoring
            self._evidence_slots.release()
            return None

    def _evidence_job(self, *args):
        try:
            with self.metrics.stage("evidence_encoding_ms"):
                self._write_evidence(*args[:5])
            name, _, session, _, event_time, source_origin = args
            if (self.data / "evidence" / name).is_file():
                self.storage.record_event(
                    session.id,
                    "evidence_ready",
                    evidence=name,
                    source_seconds=max(0, event_time - (source_origin or 0)),
                )
        finally:
            self._evidence_slots.release()

    def _write_evidence(self, name, pre_frames, session, expected_session=None, event_time=None):
        """Write pre/post frames to MP4 on a worker thread."""
        import cv2

        final_path = self.data / "evidence" / name
        staging = self.data / "evidence" / ".pending"
        staging.mkdir(exist_ok=True)
        path = staging / name
        try:
            # Collect a short post-event window (bounded wait, never blocks frames)
            deadline = time.monotonic() + 1.0
            post = []
            while time.monotonic() < deadline and not self.shutdown.is_set():
                time.sleep(0.05)
                with session.lock:
                    if expected_session is not None and expected_session != session.session_id:
                        break
                    post = [f for f in session.frames if f[0] > pre_frames[-1][0]]
            frames = replay_frames(
                pre_frames + post, max(1.0, self.settings.target_fps), event_time
            )
            if not frames:
                return
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            if first is None:
                return
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"avc1"),
                max(1.0, float(self.settings.target_fps)),
                (w, h),
            )
            transcode = False
            if not writer.isOpened():
                transcode = True
                # Some OpenCV builds lack H.264; fall back to MPEG-4 Part 2.
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, float(self.settings.target_fps)),
                    (w, h),
                )
            if not writer.isOpened():
                path.unlink(missing_ok=True)
                return
            try:
                for entry in frames:
                    img = cv2.imdecode(np.frombuffer(entry[1], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        writer.write(img)
            finally:
                writer.release()
            if transcode and shutil.which("ffmpeg"):
                converted = path.with_name(path.stem + ".h264.mp4")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            str(path),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-threads",
                            "1",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(converted),
                        ],
                        check=True,
                        timeout=30,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    converted.replace(path)
                finally:
                    converted.unlink(missing_ok=True)
            path.replace(final_path)
            self._purge_evidence(keep_name=name)
        except Exception:  # noqa: BLE001 - evidence failure must not crash a worker
            self.metrics.mark("evidence_failures")
            path.unlink(missing_ok=True)

    def _purge_evidence(self, keep_name: str | None = None):
        """Retention by age and total size; never deletes the fresh clip.

        Age retention is only applied when snapshot_retention_days > 0
        (0 means "no age limit", consistent with snapshot retention).
        """
        folder = self.data / "evidence"
        now = time.time()
        max_age = self.settings.snapshot_retention_days * 86400
        files = sorted(
            (p for p in folder.glob("*.mp4") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = 0
        for path in reversed(files):
            size = path.stat().st_size
            too_old = max_age > 0 and (now - path.stat().st_mtime) > max_age
            too_big = total + size > 512 * 1024 * 1024
            if path.name != keep_name and (too_old or too_big):
                path.unlink(missing_ok=True)
            else:
                total += size

    def _smooth_for_display(self, session, detection, now, alpha: float = 0.4):
        """EMA over raw confidence, keyed by (track, class).

        Returns a display copy: confidence = smoothed value, weak recomputed
        from the smoothed value against the class threshold, raw_confidence =
        the untouched raw evidence. The pair (confidence, weak) is therefore
        always self-consistent, while the engine keeps consuming the RAW
        objects (temporal votes, incidents) — display never leaks into logic.
        """
        box = detection.bbox
        cx = round((box.x1 + box.x2) / 2, 1)
        cy = round((box.y1 + box.y2) / 2, 1)
        key = (detection.track_hint or f"{cx}:{cy}", detection.class_name)
        old, _ = session.display_conf.get(key, (detection.confidence, now))
        value = alpha * detection.confidence + (1 - alpha) * old
        session.display_conf[key] = (value, now)
        for stale in [
            k for k, (_, seen) in session.display_conf.items() if now - seen > 2.0 and k != key
        ]:
            del session.display_conf[stale]
        thr = self.settings.class_thresholds.get(detection.class_name, 0.4)
        shown = round(value, 4)
        return replace(
            detection,
            confidence=shown,
            weak=shown < thr,
            raw_confidence=round(detection.confidence, 4),
        )

    def fail_processing(self, session):
        if session.status in {CameraState.RUNNING, CameraState.STALE}:
            session.machine.dispatch(CameraEvent.FAIL)
            session.error = "Обработка кадра остановлена. Проверьте модель и доступ к папке данных."
            session.stop.set()
            self._interrupt(session, "processing_failed")

    def _interrupt(self, session, reason):
        session.generation += 1
        session.next_commit = session.input_sequence + 1
        session.condition.notify_all()
        session.ppe_memory.clear()
        session.display_conf.clear()
        session.radar_evidence_by_track.clear()
        for state in list(session.engine.states.values()) + list(
            session.independent_ppe.engine.states.values()
        ):
            if state.incident_id:
                self._finish(
                    state.incident_id,
                    "interrupted",
                    reason,
                    state.last - state.since,
                )
        session.engine = TemporalEngine()
        if session.independent_ppe.pending:
            session.metrics.mark("ppe_cancelled_on_source_end")
        session.independent_ppe.reset()
        session.behavior.reset()
        session.radar.reset()
        if session.latest:
            session.latest = {**session.latest, "rules": [], "active": 0}

    def _source_loop(self, session, source):
        """Capture owns OpenCV; scheduler owns ordered analytics; inference is bounded.

        Live sources keep at most one unscheduled frame (latest wins). Files
        use bounded backpressure and media timestamps, never pretend realtime.
        """
        import cv2

        inbox = queue.Queue(maxsize=1)
        eof = threading.Event()
        capture_cancel = threading.Event()
        failure = []
        limit = (
            self.settings.linq_max_in_flight
            if not self.independent_motion and getattr(self.provider, "name", "") == "linq"
            else 1
        )
        pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="frame-inference")
        pending = deque()

        def capture_frames():
            capture = None
            try:
                if session.kind == "rtsp":
                    capture = cv2.VideoCapture(
                        source,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_N_THREADS,
                            self.settings.capture_decode_threads,
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                            5000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                            5000,
                        ],
                    )
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    capture = cv2.VideoCapture(
                        str(source),
                        cv2.CAP_FFMPEG,
                        [cv2.CAP_PROP_N_THREADS, self.settings.capture_decode_threads],
                    )
                if not capture.isOpened():
                    raise RuntimeError("Could not open video source")
                session.capture = capture
                with session.lock:
                    if session.stop.is_set():
                        return
                    session.machine.dispatch(CameraEvent.READY)
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if 0 < fps < 240 else 25
                next_media = 0.0
                playback_started = time.monotonic()
                next_live = 0.0
                while not session.stop.is_set() and not capture_cancel.is_set():
                    with session.metrics.stage("capture_decode_ms"):
                        ok = capture.grab()
                    if not ok:
                        if session.kind == "video":
                            break
                        raise RuntimeError("Video source lost")
                    session.metrics.mark("capture")
                    captured = time.monotonic()
                    timestamp = (
                        max(0, capture.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps
                        if session.kind == "video"
                        else captured
                    )
                    # Decode live video continuously; don't accumulate decoder buffers.
                    if session.kind == "video":
                        if timestamp + 1e-6 < next_media:
                            session.metrics.mark("sampled_out")
                            continue
                        next_media += 1 / self.settings.target_fps
                        if self.independent_motion and self.settings.file_realtime_motion:
                            delay = playback_started + timestamp - time.monotonic()
                            if delay > 0 and session.stop.wait(delay):
                                break
                            captured = time.monotonic()
                    else:
                        if captured < next_live:
                            session.metrics.mark("sampled_out")
                            continue
                        next_live = captured + 1 / self.settings.target_fps
                    # Decode the stream continuously, but convert/copy to BGR only
                    # for scheduled frames. Never reduce model input or source time.
                    with session.metrics.stage("capture_retrieve_ms"):
                        ok, frame = capture.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError("Could not retrieve scheduled video frame")
                    with session.lock:
                        item = (frame, timestamp, captured, session.session_id, session.generation)
                    if session.kind == "video":
                        while not session.stop.is_set() and not capture_cancel.is_set():
                            try:
                                inbox.put(item, timeout=0.05)
                                break
                            except queue.Full:
                                continue
                    else:
                        try:
                            inbox.put_nowait(item)
                        except queue.Full:
                            try:
                                inbox.get_nowait()
                                session.metrics.mark("dropped")
                            except queue.Empty:
                                pass
                            try:
                                inbox.put_nowait(item)
                            except queue.Full:
                                session.metrics.mark("dropped")
            except Exception:
                failure.append(True)
            finally:
                if capture is not None:
                    capture.release()
                session.capture = None
                eof.set()

        capture_thread = threading.Thread(target=capture_frames, daemon=True, name="video-capture")
        capture_thread.start()
        try:
            while not session.stop.is_set():
                if len(pending) < limit:
                    try:
                        item = inbox.get(timeout=0.01)
                        frame, timestamp, captured, sid, generation = item
                        pending.append((pool.submit(self._infer, frame, session.metrics), item))
                    except queue.Empty:
                        pass
                if pending and pending[0][0].done():
                    future, item = pending.popleft()
                    frame, timestamp, captured, sid, generation = item
                    prepared, detections, inference_ms = future.result()
                    with session.lock:
                        if session.stop.is_set():
                            break
                        if sid != session.session_id or generation != session.generation:
                            session.metrics.mark("stale")
                            continue
                        self._commit(
                            session, prepared, detections, timestamp, captured, inference_ms
                        )
                elif pending:
                    session.stop.wait(0.002)
                if eof.is_set() and inbox.empty() and not pending:
                    if failure:
                        raise RuntimeError("Video capture failed")
                    with session.lock:
                        if not session.stop.is_set():
                            session.machine.dispatch(CameraEvent.EOF)
                    break
        except Exception:
            with session.lock:
                if not session.stop.is_set():
                    session.machine.dispatch(CameraEvent.FAIL)
                    session.error = "Не удалось обработать источник. Проверьте камеру и модель."
        finally:
            for future, _ in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            capture_cancel.set()
            capture_thread.join(timeout=5.5)
            with session.lock:
                if not session.stop.is_set():
                    self._interrupt(
                        session, "source_ended" if session.status == "ended" else "source_lost"
                    )
            if session.upload_path and not capture_thread.is_alive():
                session.upload_path.unlink(missing_ok=True)
                session.upload_path = None

    def _infer(self, frame, metrics):
        import cv2

        self.ready()
        started = time.monotonic()
        if frame.shape[0] > 1080 or frame.shape[1] > 1920:
            ratio = min(1080 / frame.shape[0], 1920 / frame.shape[1])
            frame = cv2.resize(
                frame, (round(frame.shape[1] * ratio), round(frame.shape[0] * ratio))
            )
        from app.inference.linq import LinqProvider

        if self.independent_motion:
            with metrics.stage("cpu_person_ms"):
                detections = self.provider.predict_people(frame)
            metrics.mark("person")
        elif isinstance(self.provider, LinqProvider):
            detections = self.provider.predict_sync(
                frame, on_tpu_result=lambda: metrics.mark("tpu")
            )
        elif hasattr(self.provider, "predict_sync"):
            detections = self.provider.predict_sync(frame)
        else:
            detections = asyncio.run(self.provider.predict(frame))
        elapsed = (time.monotonic() - started) * 1000
        metrics.observe("inference_ms", elapsed)
        return frame, detections, elapsed

    def process(
        self,
        session,
        frame,
        timestamp=None,
        expected_session=None,
        expected_generation=None,
        source_frame_id=None,
        source_elapsed=None,
    ):
        # Admission and commits hold the short state lock. I/O/inference never do.
        with session.condition:
            sid, generation = session.session_id, session.generation
            if expected_generation is not None and generation != expected_generation:
                raise DiscardedFrame("Устаревший кадр: правила камеры изменились.")
            if expected_session is not None and sid != expected_session:
                raise DiscardedFrame("Устаревший кадр: сессия камеры изменилась.")
            if session.stop.is_set() or session.status not in {
                CameraState.RUNNING,
                CameraState.STALE,
            }:
                raise DiscardedFrame("Устаревший кадр: камера остановлена.")
            captured = time.monotonic()
            if source_frame_id is not None:
                if source_frame_id <= session.source_frame_id:
                    session.metrics.mark("stale")
                    raise DiscardedFrame("Устаревший кадр: нарушен порядок источника.")
                session.source_frame_id = source_frame_id
                if session.browser_origin is None:
                    session.browser_origin = captured - source_elapsed
                timestamp = session.browser_origin + source_elapsed
                if session.last_timestamp is not None and timestamp <= session.last_timestamp:
                    raise DiscardedFrame("Устаревший кадр: время источника не возрастает.")
            session.input_sequence += 1
            ticket = session.input_sequence
        session.metrics.mark("capture")
        error = None
        try:
            prepared, detections, inference_ms = self._infer(frame, session.metrics)
        except Exception as exc:
            error = exc
        with session.condition:
            while (
                ticket != session.next_commit
                and sid == session.session_id
                and generation == session.generation
                and not session.stop.is_set()
            ):
                session.condition.wait(timeout=0.05)
            if (
                sid != session.session_id
                or generation != session.generation
                or session.stop.is_set()
            ):
                session.metrics.mark("stale")
                raise DiscardedFrame("Устаревший результат кадра отклонён.")
            try:
                if error is not None:
                    raise error
                return self._commit(
                    session, prepared, detections, timestamp, captured, inference_ms
                )
            finally:
                session.next_commit += 1
                session.condition.notify_all()

    @staticmethod
    def _source_datetime(session, timestamp):
        value = (
            session.source_wall + timestamp
            if session.kind == "video"
            else time.time() - (time.monotonic() - timestamp)
        )
        return datetime.fromtimestamp(value, UTC).isoformat()

    def _predict_ppe(self, frame, tracks, metrics):
        from app.domain.models import Detection

        people = [Detection("person", t.confidence, t.bbox) for t in tracks]
        with metrics.stage("ppe_inference_ms"):
            detections = self.provider.predict_ppe(frame, people)
        if getattr(self.provider, "mode", None) == "linq_remote":
            metrics.mark("tpu")
        metrics.mark("ppe")
        return detections

    def _update_ppe(self, session, frame, tracks, quality, now, captured):
        ppe = session.independent_ppe
        invalid = ppe.observe_tracks(tracks, now)
        for key, state in list(ppe.engine.states.items()):
            if key[0] in invalid:
                self._finish(
                    state.incident_id, "interrupted", "identity_uncertain", state.last - state.since
                )
                ppe.engine.states.pop(key, None)
                ppe.engine.votes.pop(key, None)
        events = []
        packet = ppe.take(session, now, self.settings.ppe_max_age_seconds)
        if packet is not None:
            cfg = session.config.model_copy(update={"zone_enabled": False, "zone": [], "zones": []})
            assigned, _ = assign_detections(packet.tracks, packet.detections, cfg)
            for track in packet.tracks:
                if not packet.quality["tracks"][track.id]["ppe_observable"]:
                    assigned[track.id] = {}
            observed = evaluate_assigned(packet.tracks, assigned, cfg)
            cleared = cleared_assigned(packet.tracks, assigned, cfg)
            events = ppe.engine.update(
                observed,
                packet.timestamp,
                cfg.confirm_seconds,
                cfg.clear_seconds,
                cfg.temporal_window,
                cfg.temporal_required,
                cleared,
            )
            for track in packet.tracks:
                ppe.values[track.id] = (
                    dict(track.ppe),
                    dict(track.ppe_confidence),
                    packet.timestamp,
                    packet.frame_id,
                )
        if (
            ppe.last_result is not None
            and now - ppe.last_result > self.settings.ppe_max_age_seconds
        ):
            for state in ppe.engine.states.values():
                self._finish(
                    state.incident_id, "interrupted", "ppe_stale", state.last - state.since
                )
            ppe.engine = TemporalEngine()
            ppe.values.clear()
            if ppe.status != "unavailable":
                ppe.status = "stale"
        for track in tracks:
            value = ppe.values.get(track.id)
            if value and quality["tracks"][track.id]["ppe_observable"]:
                track.ppe, track.ppe_confidence = value[0], value[1]
                track.ppe_age_seconds = round(max(0, now - value[2]), 3)
        if any(
            (
                session.config.helmet,
                session.config.vest,
                session.config.mask,
                session.config.smoking,
            )
        ):
            ppe.submit(
                session,
                frame,
                tracks,
                quality,
                now,
                captured,
                self._ppe_pool,
                lambda f, t: self._predict_ppe(f, t, session.metrics),
                self.settings.ppe_interval_seconds,
            )
        return events, packet

    @staticmethod
    def _ppe_snapshot(packet):
        import cv2

        image = packet.frame.copy()
        h, w = image.shape[:2]
        for detection in packet.detections:
            if detection.weak:
                continue
            b = detection.bbox
            cv2.rectangle(
                image,
                (int(b.x1 * w), int(b.y1 * h)),
                (int(b.x2 * w), int(b.y2 * h)),
                (40, 180, 240),
                2,
            )
            cv2.putText(
                image,
                f"{detection.class_name} {detection.confidence:.2f}",
                (int(b.x1 * w), max(18, int(b.y1 * h) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (40, 180, 240),
                1,
            )
        ok, jpg = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Could not encode PPE source snapshot")
        return jpg.tobytes()

    def _commit(self, session, frame, detections, timestamp, captured, inference_ms):
        import cv2

        started = time.monotonic()
        now = timestamp if timestamp is not None else captured
        if session.last_timestamp is not None and now <= session.last_timestamp:
            session.metrics.mark("stale")
            raise RuntimeError("Нарушен порядок кадров.")
        session.last_timestamp = now
        if session.source_origin is None:
            session.source_origin = 0.0 if session.kind == "video" else now
        stage = time.perf_counter()
        tracks = session.tracker.update(detections, now)
        assignments, owners = assign_detections(tracks, detections, session.config)
        session.metrics.observe("tracking_ms", (time.perf_counter() - stage) * 1000)
        stage = time.perf_counter()

        # Safety Radar: update trajectories and detect risks
        # Use normalized coordinates [0,1] consistently with track.bbox
        radar = session.radar
        # zone_enabled must be respected; a disabled/empty zone invalidates
        # prior risk state instead of continuing to warn.
        zones = session.config.effective_zones()
        if zones:
            radar.set_zones({z.name: z.points for z in zones}, enabled=True)
        else:
            radar.set_zones({}, enabled=False)
        # Radar works directly on confirmed person tracks from tracker
        # track.bbox is normalized [0,1], use frame timestamp 'now' for time
        radar_events = radar.update(tracks, session.id, now)
        session.metrics.observe("radar_ms", (time.perf_counter() - stage) * 1000)
        session.metrics.mark("radar")
        # Store radar events in session for API access
        session.radar_events = radar_events
        observed = evaluate_assigned(tracks, assignments, session.config)
        cleared = cleared_assigned(tracks, assignments, session.config)
        ppe_events, ppe_packet = [], None
        quality = observation_quality(frame, tracks) if self.independent_motion else None
        if self.independent_motion:
            ppe_events, ppe_packet = self._update_ppe(
                session, frame, tracks, quality, now, captured
            )
        else:
            # Display-only PPE memory: keep the last confirmed state for a few
            # seconds so a single missed detection does not flip the UI to
            # "unknown". Violations still use the raw per-frame assignments.
            for track in tracks:
                known = {k: v for k, v in track.ppe.items() if v != "unknown"}
                if known:
                    session.ppe_memory[track.id] = (
                        dict(track.ppe),
                        dict(track.ppe_confidence),
                        now,
                    )
                else:
                    remembered = session.ppe_memory.get(track.id)
                    if remembered and now - remembered[2] <= 5.0:
                        track.ppe = dict(remembered[0])
                        track.ppe_confidence = dict(remembered[1])
                        track.ppe_age_seconds = round(now - remembered[2], 3)
            stale = [tid for tid, (_, _, seen) in session.ppe_memory.items() if now - seen > 30.0]
            for tid in stale:
                del session.ppe_memory[tid]
        # Each named zone and each observed entry has its own temporal key.
        # A confirmed geometric exit closes that episode; missing data never does.
        observed = {key: value for key, value in observed.items() if key[1] != "danger_zone"}
        cleared = {key: value for key, value in cleared.items() if key[1] != "danger_zone"}
        zone_end_events = []
        present_ids = {track.id for track in tracks}
        for key, state in list(session.engine.states.items()):
            if (
                key[1] == "danger_zone"
                and len(key) > 2
                and key[0] in present_ids
                and radar.zone_level(key[0], key[2]) in {"NORMAL", "WARNING"}
            ):
                if state.incident_id:
                    zone_end_events.append(("resolved", key, state))
                session.engine.states.pop(key, None)
                session.engine.votes.pop(key, None)
        for track in tracks:
            for zone in zones:
                if radar.zone_level(track.id, zone.name) == "CRITICAL":
                    observed[
                        (track.id, "danger_zone", zone.name, radar.episode_id(track.id, zone.name))
                    ] = track.confidence
        if session.config.running_enabled:
            running_observed, running_cleared = session.behavior.update(
                tracks,
                now,
                session.config.running_speed,
                session.config.running_min_seconds,
            )
            for track in tracks:
                if (track.id, "running") in running_observed:
                    track.violations.append("running")
            observed.update(running_observed)
            for key, confidence in running_cleared.items():
                cleared.setdefault(key, confidence)
        else:
            session.behavior.reset()
        detections = [
            replace(d, track_hint=str(owners[index]) if owners[index] is not None else None)
            for index, d in enumerate(detections)
        ]
        display_detections = [self._smooth_for_display(session, d, now) for d in detections]
        events = session.engine.update(
            observed,
            now,
            session.config.confirm_seconds,
            session.config.clear_seconds,
            session.config.temporal_window,
            session.config.temporal_required,
            cleared,
        )
        events.extend(zone_end_events)
        events.extend(ppe_events)
        ppe_state_ids = {id(state) for _, _, state in ppe_events}
        stage = time.perf_counter()
        annotated = frame.copy()
        h, w = frame.shape[:2]
        for zone in session.config.effective_zones():
            polygon = np.array([(int(x * w), int(y * h)) for x, y in zone.points], np.int32)
            cv2.polylines(annotated, [polygon], True, (0, 160, 255), 2)
        for detection in display_detections:
            if detection.weak:
                # Evidence overlay shows confirmed-relevant detections only;
                # weak diagnostics stay in the frame API output and raw UI.
                continue
            box = detection.bbox
            color = (60, 180, 255) if detection.class_name.startswith("no_") else (190, 140, 80)
            x1, y1, x2, y2 = int(box.x1 * w), int(box.y1 * h), int(box.x2 * w), int(box.y2 * h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                annotated,
                f"{detection.class_name} {detection.confidence:.0%}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        for track in tracks:
            box = track.bbox
            x1, y1, x2, y2 = int(box.x1 * w), int(box.y1 * h), int(box.x2 * w), int(box.y2 * h)
            color = (70, 80, 240) if track.violations else (150, 200, 40)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            level = radar.get_current_risk_level(track.id, now)
            label = f"Worker #{track.id} {level} " + ",".join(track.violations)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        for path in radar.snapshot(now)["tracks"]:
            if path["predicted"]:
                points = np.array(
                    [(int(x * w), int(y * h)) for x, y in path["predicted"]], np.int32
                )
                cv2.polylines(annotated, [points], False, (0, 190, 255), 2)
            if path["intersection"] and path["level"] == "WARNING":
                x, y = path["intersection"]
                cv2.circle(annotated, (int(x * w), int(y * h)), 8, (0, 190, 255), 2)
        cv2.putText(
            annotated,
            f"Source: {max(0, now - session.source_origin):.2f}s | motion CPU",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated, utcnow(), (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Не удалось закодировать кадр.")
        jpeg = encoded.tobytes()
        session.metrics.observe("jpeg_annotated_ms", (time.perf_counter() - stage) * 1000)
        stage = time.perf_counter()
        raw_ok, raw_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not raw_ok:
            raise RuntimeError("Не удалось закодировать исходный кадр.")
        session.metrics.observe("jpeg_raw_ms", (time.perf_counter() - stage) * 1000)
        # The event frame must already be in the immutable pre-event snapshot,
        # even if the camera is stopped immediately after the incident.
        session.frames.append((session.sequence + 1, raw_encoded.tobytes(), jpeg, now))
        # Evidence Replay: save a short clip for the first new event
        evidence_name = None
        if radar_events:
            evidence_name = self._save_evidence(session, radar_events)
            if evidence_name:
                for event in radar_events:
                    event.evidence = evidence_name
        session.radar_evidence = evidence_name
        # Map track -> evidence for this frame so the incident row can link
        # the clip that actually contains its CRITICAL moment.
        for event in radar_events:
            if event.evidence:
                session.radar_evidence_by_track[
                    (event.track_id, event.zone_name, event.episode_id)
                ] = event.evidence
                if len(session.radar_evidence_by_track) > 256:
                    session.radar_evidence_by_track.pop(next(iter(session.radar_evidence_by_track)))
            self.storage.record_event(
                session.id,
                event.level.lower(),
                track_id=event.track_id,
                zone_name=event.zone_name,
                source_seconds=max(0, event.timestamp - session.source_origin),
                occurred_at=self._source_datetime(session, event.timestamp),
                episode_id=event.episode_id,
                evidence=event.evidence,
                payload={
                    "lead_time_seconds": event.lead_time_seconds,
                    "predicted_entry_seconds": event.predicted_entry_seconds,
                    "event_delay_ms": round((time.monotonic() - captured) * 1000, 3),
                },
            )
            session.metrics.observe("event_delay_ms", (time.monotonic() - captured) * 1000)
        for action, key, state in events:
            if action == "active":
                # Never merge different workers solely by camera/zone. Without
                # identity evidence, separate episodes are safer than lost incidents.
                incident_id = uuid4().hex
                path = self.data / "snapshots" / f"{incident_id}.jpg"
                temporary = path.with_suffix(".tmp")
                try:
                    event_jpeg = (
                        self._ppe_snapshot(ppe_packet)
                        if id(state) in ppe_state_ids and ppe_packet is not None
                        else jpeg
                    )
                    temporary.write_bytes(event_jpeg)
                    temporary.replace(path)
                    record = self.storage.create_incident(
                        session.id,
                        session.config.name,
                        key,
                        state.confidence,
                        state.last - state.since,
                        incident_id=incident_id,
                        snapshot=path.name,
                        zone_name=(
                            (key[2] if len(key) > 2 else self._zone_for_track(tracks, key[0]))
                            or session.config.zone_name
                            if key[1] == "danger_zone"
                            else None
                        ),
                        evidence=(
                            self._save_evidence(
                                session, [type("Event", (), {"timestamp": state.last})()]
                            )
                            if id(state) in ppe_state_ids
                            else session.radar_evidence_by_track.get((key[0], key[2], key[3]))
                            if len(key) > 3
                            else None
                        )
                        if key[1] in {"danger_zone", "no_helmet", "no_vest", "no_mask", "smoking"}
                        else None,
                        episode_id=key[3] if len(key) > 3 else None,
                        source_seconds=max(0, state.since - session.source_origin),
                        started_at=datetime.fromtimestamp(
                            session.source_wall + state.since
                            if session.kind == "video"
                            else time.time() - (time.monotonic() - state.since),
                            UTC,
                        ).isoformat(),
                    )
                except (OSError, ValueError, sqlite3.Error):
                    path.unlink(missing_ok=True)
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
                state.incident_id = record["id"]
                self.storage.record_event(
                    session.id,
                    "confirmed",
                    incident_id=record["id"],
                    track_id=key[0],
                    zone_name=record["zone_name"],
                    episode_id=record.get("episode_id"),
                    source_seconds=max(0, state.last - session.source_origin),
                    occurred_at=self._source_datetime(session, state.last),
                    evidence=record["evidence"],
                    payload={"type": key[1]},
                )
                self.notifier.submit(record, event_jpeg)
                delay_ms = (
                    time.monotonic()
                    - (
                        ppe_packet.captured
                        if id(state) in ppe_state_ids and ppe_packet
                        else captured
                    )
                ) * 1000
                session.metrics.observe("event_delay_ms", delay_ms)
            elif action == "resolved":
                self._finish(
                    state.incident_id,
                    "resolved",
                    "violation_cleared",
                    state.last - state.since,
                    session,
                    now,
                )
            elif action == "interrupted":
                self._finish(
                    state.incident_id,
                    "interrupted",
                    "frame_gap",
                    state.last - state.since,
                )
            elif action == "lost":
                # Prolonged unknown: never claim the violation cleared.
                self._finish(
                    state.incident_id,
                    "interrupted",
                    "observation_lost",
                    state.last - state.since,
                )
        for state in session.engine.states.values():
            if state.incident_id:
                self.storage.update(
                    state.incident_id,
                    duration=state.last - state.since,
                    confidence=state.confidence,
                )
        ended = time.monotonic()
        interval = ended - session.last_tick if session.last_tick is not None else 0
        session.sequence += 1
        # Safety Radar: get current risk levels for tracks
        radar = session.radar
        track_risks = {}
        for track in tracks:
            track_risks[track.id] = radar.get_current_risk_level(track.id, now)
        track_risks.update(radar.current_risks(now))

        # Collect recent radar events (not only this frame) for UI/evidence
        radar_events_data = []
        for ev in radar.get_events(since=now - 10.0):
            radar_events_data.append(
                {
                    "track_id": ev.track_id,
                    "camera_id": ev.camera_id,
                    "risk_type": ev.risk_type,
                    "level": ev.level,
                    "timestamp": ev.timestamp,
                    "source_seconds": max(0, ev.timestamp - session.source_origin),
                    "message": ev.message,
                    "zone_name": ev.zone_name,
                    "episode_id": ev.episode_id,
                    "lead_time_seconds": ev.lead_time_seconds,
                    "predicted_entry_seconds": ev.predicted_entry_seconds,
                    "evidence_url": (
                        f"/api/v1/cameras/{session.id}/evidence/{getattr(ev, 'evidence', '')}"
                        if getattr(ev, "evidence", None)
                        else None
                    ),
                }
            )

        session.metrics.mark("analytics")
        session.metrics.observe("analytics_ms", (ended - started) * 1000)
        session.metrics.observe("end_to_end_ms", (ended - captured) * 1000)
        measured = session.metrics.snapshot()
        session.latest = {
            "metrics": measured,
            "motion_history": session.behavior.observation_diagnostics(),
            "person_fps": measured["rates"].get("person", {}).get("fps"),
            "radar_fps": measured["rates"].get("radar", {}).get("fps"),
            "tpu_fps": measured["rates"].get("tpu", {}).get("fps"),
            "ppe": {
                **session.independent_ppe.describe(now, self.settings.ppe_max_age_seconds),
                "runtime": "tpu"
                if getattr(self.provider, "mode", None) == "linq_remote"
                else "cpu",
            }
            if self.independent_motion
            else None,
            "observation_quality": quality,
            "source_timestamp": now,
            "data_age_ms": round((ended - captured) * 1000, 1),
            "session_id": session.session_id,
            "sequence": session.sequence,
            "frame_url": f"/api/v1/cameras/{session.id}/frame.jpg?raw=true&session_id={session.session_id}&sequence={session.sequence}",
            "rules": session.engine.describe(now, session.config.confirm_seconds)
            + session.independent_ppe.engine.describe(
                session.independent_ppe.last_result or now, session.config.confirm_seconds
            ),
            "tracks": [asdict(t) for t in tracks],
            "detections": [asdict(d) for d in display_detections],
            "latency_ms": round((ended - captured) * 1000, 1),
            "fps": round(1 / interval, 1) if interval > 0 else 0,
            "timestamp": utcnow(),
            "width": w,
            "height": h,
            "active": sum(
                s.incident_id is not None
                for s in list(session.engine.states.values())
                + list(session.independent_ppe.engine.states.values())
            ),
            "radar": {
                "track_risks": track_risks,
                "events": radar_events_data,
                "scene": radar.snapshot(now),
            },
        }
        session.last_tick = ended
        session.last_frame = ended
        session.jpeg = jpeg
        session.machine.dispatch(CameraEvent.FRAME)
        return session.latest

    def close(self):
        self.shutdown.set()
        self.watchdog.join(timeout=1)
        for camera_id in list(self.sessions):
            try:
                self.stop(camera_id, "server_stopped")
            except RuntimeError:
                pass
        self._ppe_pool.shutdown(wait=True, cancel_futures=True)
        self._evidence_pool.shutdown(wait=True, cancel_futures=True)
        self.notifier.close()
        if hasattr(self.provider, "close"):
            self.provider.close()
