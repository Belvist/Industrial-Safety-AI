import math
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import ROOT, Settings, settings
from app.services.pairing import token_owner_id
from app.services.runtime import CameraSession, DiscardedFrame, Runtime
from app.services.schemas import (
    CameraConfig,
    CameraView,
    Feedback,
    FrameResult,
    PairingExchange,
    PairingRequest,
    StartSource,
)


def create_app(config: Settings = settings, runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime or await run_in_threadpool(Runtime, config)
        yield
        await run_in_threadpool(app.state.runtime.close)

    app = FastAPI(title=config.app_name, version="0.2.0", lifespan=lifespan)
    origins = set(config.origin_list)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.host_list)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Content-Type",
            "X-Session-Id",
            "X-API-Token",
            "Authorization",
            "X-Frame-Id",
            "X-Capture-Time",
        ],
    )

    def caller(request):
        """Resolve (master: bool, scope: set[str] | None).

        Open local mode (no api_token): everything is master. Otherwise the
        master token grants full access; a live phone token grants its paired
        camera plus cameras it owns. Returns (False, None) when unauthenticated.
        """
        token = request.query_params.get("token") or request.headers.get("x-api-token")
        if not token:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
        if not config.api_token:
            return True, None
        if token and secrets.compare_digest(token.encode(), config.api_token.encode()):
            return True, None
        try:
            service = request.app.state.runtime
            paired = service.pairing.camera_for(token or "")
            # IMPORTANT: ownership markers are not credentials. An expired or
            # revoked token must not resurrect access via its SHA owner marker.
            if paired is None:
                return False, None
            mark = token_owner_id(token or "")
            owned = {c["id"] for c in service.storage.cameras() if c.get("owner") == mark}
            scope = owned | ({paired} if paired else set())
        except AttributeError:
            return False, None
        if scope:
            return False, scope
        return False, None

    @app.middleware("http")
    async def local_only(request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and origin not in origins:
                return JSONResponse(
                    {"detail": "Этот сервер доступен только локальному интерфейсу."},
                    status_code=403,
                )
        path = request.url.path
        if path.startswith("/api/v1/"):
            # The exchange endpoint is open by design: the one-time code,
            # shown only on the operator screen, is the credential.
            # system/info stays open: it carries no secrets (see S1).
            if (
                not (request.method == "POST" and path == "/api/v1/pairing/exchange")
                and path != "/api/v1/system/info"
            ):
                master, scope = caller(request)
                if not master and scope is None:
                    if config.api_token:
                        return JSONResponse({"detail": "Нужен токен доступа."}, status_code=401)
                elif not master:
                    parts = path.split("/")
                    # Phone scope: frames and transport of its own cameras only.
                    # Settings, pairing management and everything else stay
                    # operator-only. List endpoints pass through and filter
                    # inside their handlers.
                    list_endpoint = path in {
                        "/api/v1/cameras",
                        "/api/v1/incidents",
                    }
                    incident_item = (
                        len(parts) == 6
                        and parts[3] == "incidents"
                        and parts[5] in {"snapshot", "feedback", "timeline"}
                        and request.method in {"GET", "POST"}
                    )
                    allowed = (
                        list_endpoint
                        or incident_item
                        or (
                            request.method in {"GET", "POST"}
                            and len(parts) >= 5
                            and parts[3] == "cameras"
                            and parts[4] in scope
                            and (
                                (len(parts) == 5 and request.method == "GET")
                                or (
                                    len(parts) == 6
                                    and (
                                        (
                                            parts[5] in {"start", "stop", "frame"}
                                            and request.method == "POST"
                                        )
                                        or (parts[5] == "frame.jpg" and request.method == "GET")
                                    )
                                )
                            )
                        )
                    )
                    if not allowed:
                        return JSONResponse(
                            {"detail": "Токен телефона ограничен своими камерами."},
                            status_code=403,
                        )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        service = getattr(request.app.state, "runtime", None)
        if request.url.path.startswith("/api/") and hasattr(service, "metrics"):
            service.metrics.observe(
                "api_response_headers_ms", (time.perf_counter() - started) * 1000
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def rt(request):
        return request.app.state.runtime

    def session(request, camera_id):
        try:
            return rt(request).session(camera_id)
        except KeyError:
            raise HTTPException(404, "Камера не найдена.") from None

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": config.app_name,
            "inference_provider": config.inference_provider,
        }

    @app.get("/api/v1/system/info")
    def system_info(request: Request):
        service = rt(request)
        provider_status = None
        provider = getattr(service, "provider", None)
        if provider is not None and hasattr(provider, "status"):
            try:
                provider_status = provider.status()
            except Exception:  # noqa: BLE001 - info endpoint must not fail
                provider_status = None
        try:
            import onnxruntime as ort

            available_providers = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001 - optional dependency state
            available_providers = ["CPUExecutionProvider"]
        master, scope = caller(request)
        scoped_stats = service.storage.stats(camera_ids=(scope or set()) if not master else None)
        if not master:
            provider_status = None
        return {
            "environment": config.app_env,
            "inference_provider": config.inference_provider,
            "incident_confirm_seconds": config.incident_confirm_seconds,
            "model_ready": service.model_error is None,
            "model_error": service.model_error if master else None,
            "inference": provider_status,
            "available_providers": available_providers,
            # Never expose the demo token: strip any query (?token=) defensively
            # even if PUBLIC_URL was misconfigured with one. The operator
            # browser appends its own token to the QR (see api.ts withToken).
            "public_url": (config.public_url or "").split("?")[0] or None,
            "telegram_ready": service.notifier.ready,
            "target_fps": config.target_fps,
            "stats": scoped_stats,
            "api_metrics": service.metrics.snapshot()
            if master and hasattr(service, "metrics")
            else None,
        }

    @app.post("/api/v1/pairing", status_code=201)
    def create_pairing(body: PairingRequest, request: Request):
        master, _ = caller(request)
        if config.api_token and not master:
            raise HTTPException(401, "Только оператор может создавать pairing-коды.")
        service = rt(request)
        if body.new_camera_name:
            # Isolated camera owned by whoever exchanges this code: each
            # phone gets its own device, history and settings namespace.
            with service.sessions_lock:
                if len(service.sessions) >= 8:
                    raise HTTPException(
                        400, "Локальная версия поддерживает до 8 конфигураций камер."
                    )
                code, ttl = service.pairing.create("__pending__")
                mark = f"pair:{code}"
                camera_id = service.storage.save_camera(
                    CameraConfig(name=body.new_camera_name, owner=mark)
                )
                live = CameraSession(camera_id, CameraConfig(name=body.new_camera_name, owner=mark))
                service.sessions[camera_id] = live
                service.pairing.bind(code, camera_id)
                return {"code": code, "camera_id": camera_id, "expires_in": ttl}
        service.session(body.camera_id)
        code, ttl = service.pairing.create(body.camera_id)
        return {"code": code, "camera_id": body.camera_id, "expires_in": ttl}

    @app.post("/api/v1/pairing/exchange")
    def exchange_pairing(body: PairingExchange, request: Request):
        service = rt(request)
        status, token = service.pairing.exchange(body.code)
        if status == "expired":
            raise HTTPException(410, "Pairing-код истёк. Создайте новый.")
        if token is None:
            raise HTTPException(404, "Неверный или уже использованный код.")
        # Transfer phone-created cameras from the code marker to the token owner.
        with service.sessions_lock:
            new_mark = token_owner_id(token.token)
            service.storage.reassign_camera_owner(f"pair:{body.code}", new_mark)
            for live in service.sessions.values():
                if live.config.owner == f"pair:{body.code}":
                    live.config.owner = new_mark
        return {
            "token": token.token,
            "camera_id": token.camera_id,
            "expires_in": max(0, int(token.expires_at - time.monotonic())),
        }

    @app.delete("/api/v1/pairing/{code}")
    def cancel_pairing(code: str, request: Request):
        master, _ = caller(request)
        if config.api_token and not master:
            raise HTTPException(401, "Только оператор может отменять pairing-коды.")
        if not rt(request).pairing.cancel(code):
            raise HTTPException(404, "Код не найден.")
        return {"cancelled": True}

    @app.get("/api/v1/cameras", response_model=list[CameraView])
    def cameras(request: Request):
        master, scope = caller(request)
        with rt(request).sessions_lock:
            sessions = list(rt(request).sessions.values())
        if master or scope is None:
            return [s.summary() for s in sessions]
        # Phone sees only its paired and owned cameras: no cross-user sync.
        return [s.summary() for s in sessions if s.id in scope]

    @app.get("/api/v1/cameras/{camera_id}", response_model=CameraView)
    def camera_details(camera_id: str, request: Request):
        # Scoped tokens can read only the camera they are paired to.
        return session(request, camera_id).summary()

    @app.post("/api/v1/cameras", status_code=201, response_model=CameraView)
    def add_camera(config: CameraConfig, request: Request):
        service = rt(request)
        config = config.model_copy(update={"owner": None})
        with service.sessions_lock:
            if len(service.sessions) >= 8:
                raise HTTPException(400, "Локальная версия поддерживает до 8 конфигураций камер.")
            bad = service.check_rules_supported(config)
            if bad:
                raise HTTPException(
                    422,
                    f"Модель не поддерживает правила: {', '.join(bad)}. "
                    f"Отключите их или выберите модель с нужными классами.",
                )
            camera_id = service.storage.save_camera(config)
            service.sessions[camera_id] = CameraSession(camera_id, config)
            return service.sessions[camera_id].summary()

    @app.put("/api/v1/cameras/{camera_id}", response_model=CameraView)
    def update_camera(camera_id: str, config: CameraConfig, request: Request):
        current = session(request, camera_id)
        with current.lock:
            config = config.model_copy(update={"owner": current.config.owner})
            bad = rt(request).check_rules_supported(config)
            if bad:
                raise HTTPException(
                    422,
                    f"Модель не поддерживает правила: {', '.join(bad)}. "
                    f"Отключите их или выберите модель с нужными классами.",
                )
            rt(request)._interrupt(current, "rules_changed")
            current.config = config
            rt(request).storage.save_camera(config, camera_id)
        return current.summary()

    @app.post("/api/v1/cameras/{camera_id}/start", response_model=CameraView)
    def start(camera_id: str, source: StartSource, request: Request):
        cfg = session(request, camera_id).config
        bad = rt(request).check_rules_supported(cfg)
        if bad:
            raise HTTPException(
                422,
                f"Модель не поддерживает включённые правила: {', '.join(bad)}. "
                f"Отключите их в настройках камеры или выберите модель с нужными классами.",
            )
        master, scope = caller(request)
        if not master and (scope is None or camera_id not in scope or source.kind != "browser"):
            raise HTTPException(
                403, "Телефон может запускать только собственную браузерную камеру."
            )
        if source.kind == "rtsp":
            parsed = urlsplit(source.url)
            if parsed.scheme not in {"rtsp", "rtsps", "http", "https"} or not parsed.hostname:
                raise HTTPException(422, "Нужен корректный RTSP или HTTP(S) URL камеры.")
        return rt(request).start(camera_id, source.kind, source.url)

    @app.post("/api/v1/cameras/{camera_id}/stop", response_model=CameraView)
    def stop(
        camera_id: str,
        request: Request,
        session_id: str | None = None,
        revoke: bool | None = None,
    ):
        """Stop capture. Token revoke is explicit:

        - Phone caller, revoke omitted/true → unpair (need a new QR).
        - Phone caller, revoke=false → pause only; trusted session kept.
        - Operator caller → keep phone pairing unless revoke=true.
        """
        session(request, camera_id)
        master, _scope = caller(request)
        if revoke is None:
            revoke = not master
        return rt(request).stop(
            camera_id, expected_session=session_id, revoke_tokens=revoke
        )

    @app.post("/api/v1/cameras/{camera_id}/video", response_model=CameraView)
    async def upload(camera_id: str, request: Request):
        session(request, camera_id)
        service = rt(request)
        service.ready()
        upload_id = await run_in_threadpool(service.reserve_upload, camera_id)
        path = service.data / "uploads" / f"{uuid4().hex}.mp4"
        size = 0
        try:
            with path.open("wb") as file:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 200 * 1024 * 1024:
                        raise HTTPException(413, "Максимальный размер видео — 200 МБ.")
                    await run_in_threadpool(file.write, chunk)
            if size == 0:
                raise HTTPException(400, "Пустой видеофайл.")
            return await run_in_threadpool(service.finish_upload, camera_id, upload_id, path)
        except BaseException:
            path.unlink(missing_ok=True)
            await run_in_threadpool(service.cancel_upload, camera_id, upload_id)
            raise

    @app.post("/api/v1/cameras/{camera_id}/frame", response_model=FrameResult)
    async def frame(camera_id: str, request: Request):
        import cv2
        import numpy as np

        current = session(request, camera_id)
        if current.kind != "browser" or current.status not in {"running", "stale"}:
            raise HTTPException(409, "Сначала запустите браузерную камеру.")
        gate = current.frame_gate
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Предыдущий кадр ещё обрабатывается.")
        try:
            if request.headers.get("X-Session-Id") != current.session_id:
                raise HTTPException(409, "Кадр относится к другой сессии камеры.")
            expected_session = current.session_id
            expected_generation = current.generation
            source_id, elapsed = None, None
            if request.headers.get("X-Frame-Id") or request.headers.get("X-Capture-Time"):
                try:
                    source_id = int(request.headers.get("X-Frame-Id", ""))
                    elapsed = float(request.headers.get("X-Capture-Time", ""))
                    if (
                        not 0 < source_id < 2**53
                        or not math.isfinite(elapsed)
                        or not 0 <= elapsed <= 1e9
                    ):
                        raise ValueError
                except ValueError:
                    raise HTTPException(422, "Некорректный ID или время кадра.") from None
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 5 * 1024 * 1024:
                    raise HTTPException(413, "Кадр слишком большой (максимум 5 МБ).")

            def process():
                from app.services.images import frame_dimensions

                try:
                    width, height = frame_dimensions(raw)
                except ValueError:
                    raise HTTPException(400, "Нужен корректный JPEG/PNG кадр.") from None
                if width <= 0 or height <= 0 or width * height > 16_777_216:
                    raise HTTPException(413, "Слишком большое разрешение кадра (максимум 16 Мп).")
                try:
                    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                except cv2.error:
                    raise HTTPException(400, "Не удалось прочитать JPEG/PNG кадр.") from None
                if image is None:
                    raise HTTPException(400, "Не удалось прочитать JPEG/PNG кадр.")
                if current.stop.is_set():
                    raise HTTPException(409, "Камера остановлена.")
                try:
                    return rt(request).process(
                        current,
                        image,
                        expected_session=expected_session,
                        expected_generation=expected_generation,
                        source_frame_id=source_id,
                        source_elapsed=elapsed,
                    )
                except DiscardedFrame as exc:
                    raise HTTPException(409, str(exc)) from None
                except Exception as exc:  # noqa: BLE001 - camera boundary must fail closed
                    with current.lock:
                        if (
                            current.session_id == expected_session
                            and current.generation == expected_generation
                        ):
                            rt(request).fail_processing(current)
                    raise RuntimeError(
                        "Обработка кадра остановлена. Проверьте модель и доступ к папке данных."
                    ) from exc

            return await run_in_threadpool(process)
        finally:
            gate.release()

    @app.get("/api/v1/cameras/{camera_id}/frame.jpg")
    def preview(
        camera_id: str,
        request: Request,
        raw: bool = False,
        sequence: int | None = None,
        session_id: str | None = None,
    ):
        current = session(request, camera_id)
        with current.lock:
            if session_id and current.session_id != session_id:
                raise HTTPException(409, "Сессия кадра устарела.")
            match = next(
                (
                    frame
                    for frame in reversed(current.frames)
                    if sequence is None or frame[0] == sequence
                ),
                None,
            )
            if match is None:
                raise HTTPException(404, "Кадр ещё не получен или уже устарел.")
            jpeg = match[1 if raw else 2]
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/cameras/{camera_id}/evidence/{name}")
    def evidence(camera_id: str, name: str, request: Request):
        from fastapi.responses import FileResponse

        session(request, camera_id)  # 404 if the camera is unknown
        # Path traversal guard: only a bare filename from our evidence dir.
        if (
            "/" in name
            or "\\" in name
            or not name.endswith(".mp4")
            or not name.startswith(f"radar_{camera_id}_")
        ):
            raise HTTPException(404, "Доказательство не найдено.")
        folder = (rt(request).data / "evidence").resolve()
        path = folder / name
        if path.resolve().parent != folder or not path.is_file():
            raise HTTPException(404, "Доказательство не найдено.")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    incident_types = {
        "no_helmet",
        "no_vest",
        "no_mask",
        "smoking",
        "danger_zone",
        "running",
    }

    def _utc(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()

    @app.get("/api/v1/incidents")
    def incidents(
        request: Request,
        status: str | None = None,
        feedback: str | None = None,
        types: Annotated[list[str] | None, Query(alias="type")] = None,
        camera_id: Annotated[list[str] | None, Query()] = None,
        zone: Annotated[str | None, Query(max_length=80)] = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        if status not in {None, "active", "resolved", "interrupted"}:
            raise HTTPException(422, "Неизвестный статус.")
        if feedback not in {None, "unreviewed", "confirmed", "false_positive"}:
            raise HTTPException(422, "Неизвестная оценка оператора.")
        if types:
            unknown = set(types) - incident_types
            if unknown:
                raise HTTPException(
                    422, f"Неизвестный тип нарушения: {', '.join(sorted(unknown))}."
                )
        start, end = _utc(from_time), _utc(to_time)
        if start and end and start > end:
            raise HTTPException(422, "Начало интервала позже его окончания.")
        master, scope = caller(request)
        allowed = set(camera_id) if camera_id else None
        if not master and scope is not None:
            allowed = scope if allowed is None else allowed & scope
        return rt(request).storage.incidents(
            status,
            limit,
            offset,
            camera_ids=allowed,
            types=sorted(set(types)) if types else None,
            zone_name=zone.strip() if zone else None,
            from_time=start,
            to_time=end,
            feedback=feedback,
        )

    @app.get("/api/v1/incidents/export.csv")
    def export_incidents(
        request: Request,
        status: str | None = None,
        feedback: str | None = None,
        types: Annotated[list[str] | None, Query(alias="type")] = None,
        camera_id: Annotated[list[str] | None, Query()] = None,
        zone: Annotated[str | None, Query(max_length=80)] = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ):
        import csv
        import io

        master, _ = caller(request)
        if not master:
            raise HTTPException(403, "Экспорт доступен только оператору.")
        # One SQLite snapshot, same validation and filters as the journal.
        result = incidents(
            request,
            status,
            feedback,
            types,
            camera_id,
            zone,
            from_time,
            to_time,
            10001,
            0,
        )
        if result["total"] > 10000:
            raise HTTPException(413, "Больше 10 000 событий. Ограничьте период экспорта.")
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter=";")
        writer.writerow(
            [
                "ID события",
                "Событие",
                "Камера",
                "Зона",
                "Трек",
                "Время UTC",
                "Длительность, сек.",
                "Состояние наблюдения",
                "Решение оператора",
            ]
        )
        titles = {
            "no_helmet": "Без каски",
            "no_vest": "Без жилета",
            "no_mask": "Без респиратора",
            "smoking": "Курение",
            "danger_zone": "Вход в опасную зону",
            "running": "Быстрое перемещение",
        }
        states = {
            "active": "Продолжается",
            "resolved": "Завершено",
            "interrupted": "Наблюдение прервано",
        }
        reviews = {
            "unreviewed": "На проверке",
            "confirmed": "Подтверждено",
            "false_positive": "Ложное срабатывание",
        }

        def safe_cell(value):
            text = str(value if value is not None else "")
            # Quoting alone does not prevent spreadsheet formula execution.
            return (
                "'" + text
                if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r"))
                else text
            )

        for item in result["items"]:
            writer.writerow(
                [
                    safe_cell(value)
                    for value in [
                        item["id"],
                        titles.get(item["type"], item["type"]),
                        item["camera_name"],
                        item["zone_name"],
                        item["track_id"],
                        item["started_at"],
                        item["duration"],
                        states[item["status"]],
                        reviews[item["feedback"]],
                    ]
                ]
            )
        return Response(
            ("\ufeff" + output.getvalue()).encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="industrial-safety.csv"',
                "Cache-Control": "no-store",
            },
        )

    def _own_incident(request: Request, incident_id: str) -> dict:
        service = rt(request)
        record = service.storage.get(incident_id)
        if not record:
            raise HTTPException(404, "Инцидент не найден.")
        master, scope = caller(request)
        if not master and scope is not None and record["camera_id"] not in scope:
            raise HTTPException(403, "Чужой инцидент недоступен.")
        return record

    @app.get("/api/v1/analytics/risk")
    def risk_analytics(
        request: Request,
        camera_id: Annotated[list[str] | None, Query()] = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ):
        master, scope = caller(request)
        allowed = set(camera_id) if camera_id else None
        if not master:
            allowed = (scope or set()) if allowed is None else allowed & (scope or set())
        start, end = _utc(from_time), _utc(to_time)
        if start and end and start > end:
            raise HTTPException(422, "Начало интервала позже окончания.")
        return rt(request).storage.risk_statistics(allowed, start, end)

    @app.get("/api/v1/incidents/{incident_id}/timeline")
    def incident_timeline(incident_id: str, request: Request):
        incident = _own_incident(request, incident_id)
        return {"items": rt(request).storage.timeline(incident)}

    @app.get("/api/v1/incidents/{incident_id}/snapshot")
    def snapshot(incident_id: str, request: Request):
        service = rt(request)
        record = _own_incident(request, incident_id)
        if not record["snapshot"]:
            raise HTTPException(404, "Снимок не найден.")
        path = service.data / "snapshots" / Path(record["snapshot"]).name
        if not path.is_file():
            raise HTTPException(404, "Файл снимка недоступен.")
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"}
        )

    @app.post("/api/v1/incidents/{incident_id}/feedback")
    def feedback(incident_id: str, body: Feedback, request: Request):
        service = rt(request)
        _own_incident(request, incident_id)
        service.storage.update(incident_id, feedback=body.verdict)
        record = service.storage.get(incident_id)
        service.storage.record_event(
            record["camera_id"],
            "operator_review",
            incident_id=incident_id,
            track_id=record["track_id"],
            zone_name=record["zone_name"],
            episode_id=record.get("episode_id"),
            payload={"verdict": body.verdict},
        )
        return record

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        # Phone pairing page is part of the SPA bundle.
        @app.get("/connect")
        def connect_page():
            return FileResponse(dist / "index.html")

        @app.get("/phone")
        def phone_capture_page():
            return FileResponse(dist / "index.html")

        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:

        @app.get("/")
        def frontend_missing():
            return JSONResponse(
                {
                    "message": "Соберите интерфейс: cd frontend && npm ci && npm run build. Затем перезапустите backend."
                },
                status_code=503,
            )

    return app


app = create_app()
