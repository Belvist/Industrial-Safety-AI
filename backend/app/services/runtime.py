import asyncio
import sqlite3
import threading
import time
from collections import deque
from dataclasses import asdict, replace
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
from app.services.notifications import Notifier
from app.services.pairing import PairingStore
from app.services.safety_radar import SafetyRadar
from app.services.schemas import CameraConfig
from app.services.state_machine import CameraEvent, CameraMachine, CameraState
from app.services.storage import Storage, utcnow


class CameraSession:
    def __init__(self, camera_id, config):
        self.id, self.config = camera_id, config
        self.tracker, self.engine = Tracker(), TemporalEngine()
        self.behavior = RunningDetector()
        self.lock = threading.Lock()
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
        self.radar = SafetyRadar()

    @property
    def status(self):
        return self.machine.state

    def _snapshot(self):
        return {
            "id": self.id,
            **self.config.model_dump(),
            "status": self.status,
            "error": self.error,
            "kind": self.kind,
            "latest": self.latest,
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
        self.storage = Storage(settings.storage_path)
        self.data = settings.resolve(settings.data_dir)
        (self.data / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.data / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data / "evidence").mkdir(parents=True, exist_ok=True)
        self.storage.purge_old_resolved(
            settings.snapshot_retention_days, self.data / "snapshots"
        )
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
                session.machine.dispatch(CameraEvent.START)
                session.kind, session.error = kind, None
                session.session_id = uuid4().hex
                session.sequence = 0
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

    def stop(self, camera_id, reason="monitoring_stopped", expected_session=None, revoke_tokens=True):
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

    def _save_evidence(self, session, events):
        """Save a short MP4 around a radar event from the ring buffer.

        Frames are already JPEG bytes; decode, write at target fps.
        Returns filename or None. Never raises into the pipeline.
        """
        import cv2

        if not events or not session.frames:
            return None
        try:
            frames = list(session.frames)
            # Keep the last N frames (up to ~5 s at target_fps)
            limit = max(10, int(self.settings.target_fps * 5))
            frames = frames[-limit:]
            first_event = events[0]
            name = f"radar_{session.id}_{int(first_event.timestamp * 1000)}.mp4"
            path = self.data / "evidence" / name
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            if first is None:
                return None
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, float(self.settings.target_fps)), (w, h),
            )
            if not writer.isOpened():
                return None
            try:
                for _, raw, _ in frames:
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        writer.write(img)
            finally:
                writer.release()
            return name
        except Exception:  # noqa: BLE001 - evidence must not break monitoring
            return None

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
        key = (detection.track_hint or f'{cx}:{cy}', detection.class_name)
        old, _ = session.display_conf.get(key, (detection.confidence, now))
        value = alpha * detection.confidence + (1 - alpha) * old
        session.display_conf[key] = (value, now)
        for stale in [k for k, (_, seen) in session.display_conf.items()
                      if now - seen > 2.0 and k != key]:
            del session.display_conf[stale]
        thr = self.settings.class_thresholds.get(detection.class_name, 0.4)
        shown = round(value, 4)
        return replace(detection, confidence=shown, weak=shown < thr,
                       raw_confidence=round(detection.confidence, 4))

    def fail_processing(self, session):
        if session.status in {CameraState.RUNNING, CameraState.STALE}:
            session.machine.dispatch(CameraEvent.FAIL)
            session.error = "Обработка кадра остановлена. Проверьте модель и доступ к папке данных."
            session.stop.set()
            self._interrupt(session, "processing_failed")

    def _interrupt(self, session, reason):
        for state in session.engine.states.values():
            if state.incident_id:
                self.storage.finish_incident(
                    state.incident_id, "interrupted", reason,
                    duration=state.last - state.since,
                )
        session.engine = TemporalEngine()
        session.behavior.reset()
        session.radar.reset()
        if session.latest:
            session.latest = {**session.latest, "rules": [], "active": 0}

    def _source_loop(self, session, source):
        import cv2

        capture = None
        try:
            if session.kind == "rtsp":
                capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
                )
                # Latest-frame wins: never process a queue of stale RTSP frames.
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError("Не удалось открыть источник. Проверьте адрес/формат видео.")
            session.capture = capture
            with session.lock:
                if session.stop.is_set():
                    return
                session.machine.dispatch(CameraEvent.READY)
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if 0 < fps < 240 else 25
            started = time.monotonic()
            next_position = 0.0
            while not session.stop.is_set():
                before = time.monotonic()
                if session.kind == "video":
                    # Skip frames when inference is slower than the source; timestamps remain media time.
                    current = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    target = max(current, int(next_position * fps))
                    if target > current:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = capture.read()
                if not ok:
                    if session.kind == "video":
                        with session.lock:
                            if not session.stop.is_set():
                                session.machine.dispatch(CameraEvent.EOF)
                        break
                    raise RuntimeError("Видеопоток прерван.")
                timestamp = (
                    capture.get(cv2.CAP_PROP_POS_FRAMES) / fps
                    if session.kind == "video"
                    else time.monotonic()
                )
                with session.lock:
                    if not session.stop.is_set():
                        self.process(session, frame, timestamp)
                elapsed = time.monotonic() - before
                next_position = max(
                    timestamp + 1 / self.settings.target_fps, time.monotonic() - started
                )
                session.stop.wait(max(0, 1 / self.settings.target_fps - elapsed))
        except Exception:  # noqa: BLE001 - isolate camera workers and redact source credentials
            with session.lock:
                if not session.stop.is_set():
                    session.machine.dispatch(CameraEvent.FAIL)
                    session.error = "Не удалось обработать источник. Проверьте доступность камеры, кодек и модель."
        finally:
            if capture is not None:
                capture.release()
            session.capture = None
            with session.lock:
                if not session.stop.is_set():
                    self._interrupt(
                        session, "source_ended" if session.status == "ended" else "source_lost"
                    )
            if session.upload_path:
                session.upload_path.unlink(missing_ok=True)
                session.upload_path = None

    def process(self, session, frame, timestamp=None):
        import cv2

        self.ready()
        if session.stop.is_set() or session.status not in {CameraState.RUNNING, CameraState.STALE}:
            raise RuntimeError("Камера не принимает кадры в текущем состоянии.")
        started = time.monotonic()
        if frame.shape[0] > 1080 or frame.shape[1] > 1920:
            ratio = min(1080 / frame.shape[0], 1920 / frame.shape[1])
            frame = cv2.resize(
                frame, (round(frame.shape[1] * ratio), round(frame.shape[0] * ratio))
            )
        now = timestamp if timestamp is not None else started
        if hasattr(self.provider, "predict_sync"):
            detections = self.provider.predict_sync(frame)
        else:
            detections = asyncio.run(self.provider.predict(frame))
        tracks = session.tracker.update(detections, now)
        assignments, owners = assign_detections(tracks, detections, session.config)
        
        # Safety Radar: update trajectories and detect risks
        # Use normalized coordinates [0,1] consistently with track.bbox
        radar = session.radar
        if session.config.zone:
            # Zone points are already normalized [0,1] in config
            radar.set_zones({session.config.zone_name or "Опасная зона": session.config.zone})
        # Radar works directly on confirmed person tracks from tracker
        # track.bbox is normalized [0,1], use frame timestamp 'now' for time
        radar_events = radar.update(tracks, session.id, now)
        # Store radar events in session for API access
        session.radar_events = radar_events
        # Evidence Replay: save a short clip for the first new event
        evidence_name = None
        if radar_events:
            evidence_name = self._save_evidence(session, radar_events)
            if evidence_name:
                radar_events[0].evidence = evidence_name
        session.radar_evidence = evidence_name
        observed = evaluate_assigned(tracks, assignments, session.config)
        cleared = cleared_assigned(tracks, assignments, session.config)
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
        annotated = frame.copy()
        h, w = frame.shape[:2]
        if session.config.zone:
            polygon = np.array([(int(x * w), int(y * h)) for x, y in session.config.zone], np.int32)
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
            label = f"Worker #{track.id} " + ",".join(track.violations)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        cv2.putText(
            annotated, utcnow(), (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Не удалось закодировать кадр.")
        jpeg = encoded.tobytes()
        for action, key, state in events:
            if action == "active":
                incident_id = uuid4().hex
                path = self.data / "snapshots" / f"{incident_id}.jpg"
                temporary = path.with_suffix(".tmp")
                try:
                    temporary.write_bytes(jpeg)
                    temporary.replace(path)
                    record = self.storage.create_incident(
                        session.id,
                        session.config.name,
                        key,
                        state.confidence,
                        now - state.since,
                        incident_id=incident_id,
                        snapshot=path.name,
                        zone_name=(
                            session.config.zone_name
                            if key[1] == "danger_zone"
                            else None
                        ),
                    )
                except (OSError, ValueError, sqlite3.Error):
                    path.unlink(missing_ok=True)
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
                state.incident_id = record["id"]
                self.notifier.submit(record, jpeg)
            elif action == "resolved":
                self.storage.finish_incident(
                    state.incident_id,
                    "resolved",
                    "violation_cleared",
                    duration=state.last - state.since,
                )
            elif action == "interrupted":
                self.storage.finish_incident(
                    state.incident_id, "interrupted", "frame_gap",
                    duration=state.last - state.since,
                )
            elif action == "lost":
                # Prolonged unknown: never claim the violation cleared.
                self.storage.finish_incident(
                    state.incident_id,
                    "interrupted",
                    "observation_lost",
                    duration=state.last - state.since,
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
        _, raw_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        session.frames.append((session.sequence, raw_encoded.tobytes(), jpeg))
        # Safety Radar: get current risk levels for tracks
        radar = session.radar
        track_risks = {}
        for track in tracks:
            track_risks[track.id] = radar.get_current_risk_level(track.id, now)
        
        # Collect recent radar events (not only this frame) for UI/evidence
        radar_events_data = []
        for ev in radar.get_events(since=now - 10.0):
            radar_events_data.append({
                "track_id": ev.track_id,
                "camera_id": ev.camera_id,
                "risk_type": ev.risk_type,
                "level": ev.level,
                "timestamp": ev.timestamp,
                "message": ev.message,
                "zone_name": ev.zone_name,
                "evidence_url": (
                    f"/api/v1/cameras/{session.id}/evidence/{ev.evidence}"
                    if ev.evidence else None
                ),
            })

        session.latest = {
            "session_id": session.session_id,
            "sequence": session.sequence,
            "frame_url": f"/api/v1/cameras/{session.id}/frame.jpg?raw=true&session_id={session.session_id}&sequence={session.sequence}",
            "rules": session.engine.describe(now, session.config.confirm_seconds),
            "tracks": [asdict(t) for t in tracks],
            "detections": [asdict(d) for d in display_detections],
            "latency_ms": round((ended - started) * 1000, 1),
            "fps": round(1 / interval, 1) if interval > 0 else 0,
            "timestamp": utcnow(),
            "width": w,
            "height": h,
            "active": sum(s.incident_id is not None for s in session.engine.states.values()),
            "radar": {
                "track_risks": track_risks,
                "events": radar_events_data,
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
        self.notifier.close()
