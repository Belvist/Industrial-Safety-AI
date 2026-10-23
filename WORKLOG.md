# WORKLOG — бортовой журнал проекта

Живой лог: каждый запрос владельца → что сделано (с файлами) → что осталось.
Обновляется после каждого запроса. Факты только проверенные (тесты/код),
не планы из головы.

- Репозиторий: `github.com/judogaaa/Industrial-Safety-AI` (приватный)
- Локально: `Industrial-Safety-AI/` внутри `DBI/` (у родителя свой git — вложенность помнить!)
- Рабочая ветка бэкенда: `feature/linq-backend` (НЕ закоммичена, изменения в worktree)
- Трек хакатона: **LinQ** (кейс «Открой Моспром»), инференс — российские ускорители LinQ
- Главный документ требований: `Industrial_Safety_AI_Project_Description.pdf` (корень репо)

---

## Запрос 1 — «Скопируй всё что есть тут github.com/judogaaa/Industrial-Safety-AI»

**Сделано:**
- `git clone https://github.com/judogaaa/Industrial-Safety-AI.git` во временную папку,
  перенос содержимого (включая `.git`) в пустую `Industrial-Safety-AI/`
- Веб-страница репозитория отдавала 404, но `git ls-remote` показал HEAD `3383a35` —
  репозиторий существует, клонирование пошло напрямую через git (веб-404 проигнорирован)
- Забраны все ветки: `main`, `chore/bootstrap`, `docs/add-project-description-pdf`,
  `feature/local-safety-mvp` (+ локальные трекинги созданы)

**Осталось:** ничего, папка = полная копия репозитория с историей.

**Зачем/куда:** пустая папка `DBI/Industrial-Safety-AI/` стала рабочей копией проекта.
Нюанс: `DBI/` сам git-репозиторий → внутри теперь nested-репозиторий, в родитель не коммитить.

---

## Запрос 2 — «Делай бэкенд логику полностью, направление LinQ»

**Сделано (ветка `feature/linq-backend`, создана от `main` + влит `feature/local-safety-mvp`):**
- Домен: `backend/app/domain/models.py` — добавлены `NO_MASK`, `SMOKING`
  (+ канонические семейства меток `MASK_LABELS`, `SMOKING_LABELS`)
- `backend/app/inference/linq.py` — **переписан с нуля**: режимы `linq_remote`
  (POST JPEG → `{endpoint}/v1/detect`, парсер только в `_parse_remote`),
  `linq_local` (LinQ EP в ONNX Runtime), dev-fallback на локальные веса;
  `status()` для `/system/info`
- `backend/app/inference/local_onnx.py` — алиасы `mask/no_mask/smoking/...`,
  `KNOWN_CLASSES` расширен, мягкий проход неизвестных классов
- `backend/app/services/engine.py` — `evaluate()`: маска по зоне лица (band 0–0.5),
  курение по близости к лицу, далёкие сигареты игнорятся
- `backend/app/services/schemas.py` — `CameraConfig.mask/smoking`, `RuleView` на 5 типов
- `backend/app/services/runtime.py` — фабрика `LinqProvider` вместо заглушки-ошибки,
  дефолты новой камеры из `Settings.rule_*`
- `backend/app/services/storage.py` — `smoking` → `critical`
- `backend/app/services/notifications.py` — подписи `no_mask/smoking`
- `backend/app/main.py` — `system_info.inference` с режимом LinQ
- `.env.example` — `LINQ_ENDPOINT/KEY/TIMEOUT/MODEL`, `RULE_*`
- `backend/tests/test_linq.py` — **новый**: remote-нормализация, remote-ошибка, fallback без весов
- `backend/tests/test_engine.py` — **новый** `test_mask_and_smoking_rules`
- Проверки (venv python3.12): `ruff` чисто, `pytest` **44 passed**, E2E через TestClient

**Осталось на тот момент:** сверить дефолты с PDF (см. запрос 4).

**Зачем/куда:** весь даунстрим (трекинг → правила → инциденты → SQLite/Telegram/UI)
не знает про железо; переключение на LinQ = одна env-строка `INFERENCE_PROVIDER=linq`.

---

## Запрос 3 — «Что сделано-то» (отчёт)

**Сделано:** сверено `git status/diff`, дан краткий итог (ветка, 13 файлов +452/−24,
44 теста). Без изменений в коде.

**Осталось:** н/д — запрос был про отчёт.

---

## Запрос 4 — PDF с описанием проекта («тут всё написано»)

**Сделано — сверка с PDF, найдено и исправлено 1 расхождение:**
- Дефолты возвращены строго под PDF §11 (MVP = каска + жилет + зона):
  `backend/app/services/schemas.py`, `backend/app/core/config.py`,
  `.env.example` → `helmet=true, vest=true, mask=false, smoking=false`
- `mask/smoking` оставлены как выключенные по умолчанию пост-MVP расширения
  (выбор владельца из запроса 2 сохранён как опция, PDF §11 «десятки СИЗ — не делаем» соблюдён)
- `getattr(config, "smoking", False)` — безопасный фолбэк для старых строк камер в SQLite
- Telegram-подпись переписана под формат PDF §10:
  `backend/app/services/notifications.py` → заголовок + Тип / Камера / Worker /
  Время / Длительность / Уверенность + фото
- `docs/ARCHITECTURE.md` — раздел LinQ-трека, скоуп под §11
- Тест `test_mask_and_smoking_rules` переведён на явные флаги + проверка дефолтов
- Проверки: `ruff` чисто, `pytest` **44 passed**, дефолты подтверждены через API

**Осталось:** н/д — сверка закрыта.

**Зачем/куда:** PDF — главный источник правды; живой код теперь соответствует
ему дословно (пример настроек камеры, формат Telegram, live-demo §12 уже покрыт:
FPS/latency в каждом `FrameResult`).

---

## Запрос 5 — «Веди MD-файл: что сделано / что осталось, постоянно»

**Сделано:** создан этот `WORKLOG.md`, правило — обновлять после каждого запроса.

**Осталось:** поддерживать (обновление — часть каждого ответа, не отдельная задача).

---

## Запрос 6 — «Гит пуш делай»

**Сделано:** закоммичена и запушена в `origin` ветка `feature/linq-backend`.
Коммит: LinQ-бэкенд (провайдер, mask/smoking-расширения,
дефолты и Telegram по PDF, тесты 44 passed) + этот журнал.

**Осталось:** открыть PR `feature/linq-backend` → `main` на GitHub (по процессу PDF §14).

---

## Запрос 7 — ревью: «linq_local врёт, инференс всё равно на CPU» + план железа

**Сделано — исправлен честный выбор ONNX execution providers:**
- `backend/app/inference/local_onnx.py` — `LocalOnnxProvider(providers=[...])`:
  запрошенные EP фильтруются по `ort.get_available_providers()`, CPU всегда
  дописывается последним fallback'ом, факт пишется в `active_providers`
  из `session.get_providers()`; person-детектор наследует те же провайдеры
- `backend/app/inference/linq.py` — `_pick_local_mode()` смотрит только на
  `active_providers` реальной сессии (а не на глобальную доступность);
  `status()` отдаёт `requested_providers` + `active_providers` в `/system/info`
- `backend/app/core/config.py` — `LINQ_PROVIDERS` (через запятую, дефолт
  `LinQExecutionProvider`), `backend/app/services/runtime.py` прокидывает в провайдер,
  `.env.example` документирован
- `backend/tests/test_linq.py` — 3 новых теста: фильтрация + CPU-fallback,
  `local_onnx` когда EP нет в сессии, `linq_local` когда есть (фейк-сессия ORT)
- Проверки: `ruff` чисто, `pytest` **47 passed** (было 44)
- Закоммичено и запушено в `origin/feature/linq-backend`

**Осталось (железо, за людьми, не за агентом):**
- MacBook Даниила: `git checkout feature/linq-backend && git pull && python3.11 run.py` →
  встроенная камера → боксы едут → polygon → 1 incident → выход → RESOLVED
- Настоящий Telegram со snapshot
- Тот же commit на Windows + A4Tech (после возвращения Егора)
- PR #4 → merge в `main`, затем отдельный PR `feature/linq-backend` → `main`
- mask/smoking НЕ развивать (запас уже есть, выключены по умолчанию)

---

## Запрос 8 — «Проверь как работает ИИ, отчёту не верю» (независимая проверка)

**Сделано — каждое утверждение отчёта сверено с кодом ветки `feature/linq-backend` @ `7098df4`:**
- Ветка/коммит — ПРАВДА: `git log` = `7098df4` поверх `b94db2c`
- Модели не в git, тянет `download_model.py` — ПРАВДА: `git ls-files models` пуст,
  `.gitignore:16-18,42-44`, скрипт с 2 HF-URL + SHA256; оба URL живые (HEAD → 302 → 200)
- Классы person/helmet/no_helmet/vest/no_vest — ПРАВДА, и даже больше:
  `THIRD_PARTY.md:36` — в исходных классах есть и Mask/NO-Mask
- Ядро — `local_onnx.py` — ПРАВДА; цепочка `runtime.py:348,351,352`
  (`predict_sync → tracker.update → evaluate`) — дословно ПРАВДА
- Дефолты `.env` — ПРАВДА; фабрика `runtime.py:72` (`== "linq"` → `LinqProvider`) — ПРАВДА
- EP-фикс — ПРАВДА: фильтрация + `active_providers` из `get_providers()` + 3 теста
- Поправки к отчёту: тестов уже **47** (не 44 — отчёт писан на `b94db2c`);
  `models/` локально отсутствуют — живой инференс здесь не гнался,
  доказательство опирается на `docs/VALIDATION.md` + мокированные тесты
- AGPL-3.0 в метаданных весов — подтверждено (`THIRD_PARTY.md:34`)

**Осталось:** живой прогон на камере (за людьми) — единственный недоказанный пункт.

---

## Запрос 9 — «Как потестить» (инструкция)

**Сделано:** выдана лесенка из 4 уровней (живое демо §12 → 1 кадр → pytest/vitest →
Telegram); факты сверены: `run.py` требует Python 3.11/3.12, фронт-тесты — `npm test`
(vitest), бэк-тесты — pytest в venv.

**Осталось:** выполнить уровни 1 и 4 на железе (за людьми).

---

## Запрос 10 — «Запусти всё + QR чтобы телефон коннектился как камера»

**Сделано (ветка `feature/linq-backend`, НЕ закоммичено):**
- Бэкенд: `PUBLIC_URL` / `EXTRA_ORIGINS` (`config.py`), origins+hosts собираются
  динамически (`main.py`), `system_info.public_url`, `.env.example`
- `run.py --lan`: определяет LAN IP, выпускает self-signed серт
  (`data/certs/`, gitignored), поднимает uvicorn 0.0.0.0:8000 по HTTPS,
  прокидывает env; `--host/--port/--public-url/--https-cert/--https-key` для ручного режима
- Исправлен выбор IP: `lan_candidates()` предпочитает приватные 192.168/16 —
  старый UDP-трюк возвращал IP VPN-туннеля (`utun4` default route)
- Фронт: `components/PhoneConnect.tsx` (QR через `qrcode` + шаги), панель
  в «Камерах» при `info.public_url`, `types.ts`, стили, `qrcode`/`@types/qrcode` в зависимостях
- `backend/tests/test_lan.py` (3 теста: public_url, LAN origin 200/evil 403, host 200/400)
- Проверки: `ruff` чисто, `pytest` **50 passed**, фронт `vitest` 4 passed, `vite build` ок
- Живой сервер ПОДНЯТ: `https://192.168.1.194:8000`, `model_ready: true`,
  реальный ONNX-кадр через API (latency 483 мс), LAN origin 200 / чужой 403
- Важно: VPN на Маке уводит default route в туннель — для телефона нужен именно
  Wi-Fi IP из QR (сервер его уже определяет сам)

**Осталось:** проверить связку с реального телефона (QR → серт → камера → кадры);
закоммитить/запушить LAN-слой.

---

## Запрос 11 — спринт Accuracy + Performance + Product (план из 19 пунктов)

**Сделано — ветка `feature/accuracy-performance-product`, PR #5 (без merge):**
- §0 аудит: baseline 50+4, `docs/ACCURACY_PERFORMANCE_REPORT.md` (только реальные замеры)
- §1 per-class thresholds (absence 0.60) + тесты слабого no_helmet
- §2/§3: PPE требует пересечения с боксом работника; `assign_detections` + `track_hint`;
  UI: сырые боксы скрыты, toggle «Сырые детекции»
- §4 temporal consensus 3/4 + `TEMPORAL_WINDOW/REQUIRED` в env
- §5 `TrackerProvider` + 2-stage IoU-трекер, ByteTrack не понадобился (без новых зависимостей)
- §6 benchmark: dual 448 мс / single 390 мс, но single теряет людей 0 vs 3 → dual остаётся
- §7 аудит: RTSP `BUFFERSIZE=1`, остальное уже корректно
- §8 CoreML на Mac: **16 мс vs 437 мс CPU** (~27x); `ONNX_PROVIDERS` env
- §9-11 UI: worker-карточки (зона+статус), бейдж реального провайдера, `uploading` в типах
- §12 `evaluate_folder.py`, §13 `docs/CAMERA_VALIDATION_PLAN.md`
- §14 не тронуто: mask/smoking выключены; §15 LinQ-адаптер не переписан
- Коммиты: 5 штук (accuracy core / tracking / perf / worker-centric UI / tests+docs)
- Проверки: `pytest` **66 passed** (было 50), фронт 4 passed + build, `ruff` чисто
- Push + PR #5 → `feature/linq-backend` (НЕ merge, ждёт железо и CI)

**Осталось:** footage-проверка A-O (за людьми, Mac + Windows/A4Tech); CI в PR #5;
ретаргет PR на main после мержа PR #4 и linq-backend.

---

## Запрос 12 — «Реальными тестами всё закрываешь?» (аудит покрытия)

**Сделано — честная карта (проверено чтением всех 9 файлов + прогоном):**
- Реальные: engine (21), state_machine (8), runtime-интеграция (13), decode-математика (4),
  lan/origin/host (3), notifications multipart+токен (2), contracts (1), health (1),
  linq remote-парсинг + EP-проводка (8), фронт-машины (4)
- Моки только там, где иначе нельзя: сеть Telegram/ORT-сессия/HF-веса
- Закрыты найденные дыры настоящими тестами (коммит выше):
  слабый no_helmet 0.45 end-to-end → 0 инцидентов; `track_hint` в API фреймов;
  подпись сверена с PDF §10 дословно; `inferenceLabel` выделен и тестирован
  (падение LinQ→CPU показывает правду, а не «LinQ»)
- Честные непокрытия (не скрываю): живые веса в CI (gitignored — только ручной
  smoke + evaluate_folder); реальная отправка Telegram (нужен токен);
  footage A-O (нужна камера); canvas-QR в jsdom (нет canvas)
- Проверки: `pytest` **69 passed**, фронт **8 passed** + build, `ruff` чисто; запушено

**Осталось:** то же footage + CI (см. выше).

---

## Запрос 13 — hardening pass (8 пунктов ревьюера)

**Сделано — ветка `feature/accuracy-performance-product` @ `26c70a7`, PR #5 открыт:**

---

## Запрос 14 — спринт validation-reliability (7 задач)

**Сделано — ветка `feature/validation-reliability` @ `fa52651`, запушена (PR не открывал):**
1. Unknown-семантика: `cleared_assigned()` + RESOLVED только по `present`;
   долгое unknown → `lost` → `interrupted/observation_lost`, никогда ложный RESOLVED
2. Event-level scoring в `evaluate_folder.py --events` (matched/missed/FP,
   latency в кадрах, ID-stability proxy); проверено на 6 кадрах bus: 3 ID, 0 FP
3. Потолки с транспортом: CPU 1.4 FPS / CoreML 35.9 FPS; `TARGET_FPS=5` не тронут
4. Readiness `unconfigured/configured/connecting/verified/failed/fallback`;
   UI `LinQ Active` только при verified; `available_providers` в API + хинт
5. `API_TOKEN` гейтит /api (health/info открыты); `--lan` генерит токен в QR
6. `docs/REGRESSION_SET.md` + раннер; клипов пока нет (честно помечено)
7. Проверки: `pytest` **78 passed**, фронт **12 passed** + build, `ruff` чисто везде

**Осталось:** footage A-O и клипы regression set (за людьми); CI смотреть после PR;
оргвопросы (регламент/команда 3–5/ИИ-маркировка) — вне кода, требуют людей.

---

## Запрос 20 — 401 у пользователя + изоляция «у каждого своё»

**Сделано — ветка `fix/per-user-workspaces`, запушена (PR не открывал):**
- Причина 401: вкладка без токена + токен менялся каждый рестарт
- `run.py --lan`: токен стабильный (`data/api_token`, 600, gitignored) —
  рестарты больше не ломают вкладки; сразу отдан pairing-URL
- UI: 401 показывает «открой pairing-ссылку / попроси QR»
- Изоляция: `POST /pairing {new_camera_name}` → камера с owner;
  exchange переносит `pair:code` → `phone:sha`; списки камер/истории
  фильтруются; чужое — 403; master видит всё
- PhoneConnect: чекбокс «отдельная камера» (default ON)
- Проверки: `pytest` **107 passed**, фронт **20 passed** + build, `ruff` чисто
- Прод перезапущен на новом коде, stable token, health 200

**Осталось:** ручная проверка двух телефонов (A не видит B); PR по желанию.

---

## Запрос 19 — FINAL FIELD TEST (ветка PR #11, 15–20 мин, без функций)

**Сделано — `fix/field-test-rc-20261008` @ `2d7299b`, PR #11:**
- CI: было 5/5 на `a0048e2`; после доки снова 4/5 + macos pending
- Smoke на весах: SHA записаны, 7 классов supported, bus даёт 3 persons +
  weak no_helmet/no_vest, 0 инцидентов — как спроектировано
- Сценарий 8/8 через реальный API: ACTIVE → snapshot/SQLite → RESOLVED
  (`violation_cleared`) → новый ACTIVE → stop → INTERRUPTED + `ended_at`
- MP4: upload → ended без падений; polygon PUT round-trip ok
- `FIELD_TEST.md`: запуск Mac/Windows/QR, ожидания, 10-минутный протокол,
  таблица PASS/FAIL/UNTESTED
- UNTESTED честно: физическая камера, телефон, Telegram, RTSP, fine-tuning, LinQ

**Осталось:** дождаться macos CI; отдать тестировщику 4 ситуации + видео/скрины.

---

## Запрос 18 — CV-спринт: E2E + ML-трубы (база PR #9 @ 9404808)

**Сделано — ветка `fix/ml-pipeline-e2e`, PR #10 (НЕ merge):**
- База проверена: SHA `9404808`, CI PR #9 green, локально 96+19
- E2E Playwright: QR → /connect → /phone (mock MediaStream) → кадры →
  running у оператора → stop → revoke/401; CI job e2e; README
- `training/metrics.py` + тесты с точными значениями (mAP50=0.75 синтетика)
- `evaluate.py`: yaml-root, class-map check, `--person-model` dual, F1/confusion/mAP
- `prepare_dataset`: валидация пикселей/labels/id, guard сплитов, проверка YAML
- `export_onnx`: letterbox как в проде, числовое сравнение (gate 5px), smoke
- Error-table template (пустая); fine-tuning НЕ запускался (нет данных)
- Проверки: `pytest` **102 passed**, фронт **19 passed** + build, `ruff` чисто

**Осталось:** дождаться CI (особенно новый e2e job); footage; physical phone; LinQ SDK.

---

## Запрос 17 — спринт 1 Runtime Correctness (план техрука)

**Сделано — ветка `fix/runtime-correctness`, PR #8 (НЕ merge):**
- Pairing: коды TTL 5 мин → scoped токены (только своя камера), revoke on stop,
  QR `/connect?pair=`, blob-картинки, pagehide с токеном; chain-тесты
- `finish_incident`: атомарно, `ended_at` + миграция/backfill, без double-finish;
  retention по реальному концу; ACTIVE не удаляется
- LinQ: reachable ≠ verified; `hardware_verified` + `accelerator_confirmed`;
  UI Active только при подтверждении
- Evidence: weak вне snapshot-аннотации
- Проверки: `pytest` **94 passed**, фронт **19 passed** + build, `ruff` чисто;
  CI: macos/ubuntu/frontend pass, windows pending
- Открыто (без тестового подтверждения): настоящий browser-E2E, pixel-тест
  snapshot, footage, LinQ SDK

---

## Запрос 15 — фикс регрессий видимости (ревью: пропали каски + 401 на кадрах)

**Сделано — ветка `feature/fix-visibility-regressions`, отдельный PR #6 (НЕ merge):**
1. 401: `withToken()` для `<img>` (frame.jpg + snapshot); тесты 401/200
2. `DISPLAY_FLOOR=0.25` + `weak`: слабые видны, не нарушают/не закрывают/не тречат
3. Все классы в summary, маска в карточке, бейдж «слабая» в raw
4. Этапы отказа: weak / track_hint / rules; таблица этапов в отчёте
5. Engine untouched; веса и violation thresholds untouched
6. Old-vs-new на тех же весах: `no_helmet 0.38` был невидим → теперь weak;
   `no_vest 0.52` шёл в нарушения → теперь weak; person 0.44 не растит трек
7. Чекбоксы mask/smoking в форме (раунд-трип через configOf)
8. Проверки: `pytest` **82 passed**, фронт **12 passed** + build, `ruff` чисто;
   CI PR #6: macos/ubuntu/frontend pass, windows pending

**Потеряно → восстановлено:** invisible sub-threshold → weak-visible; кадры при
токене (401) → грузятся; маски в UI → строка + настройки.

**Осталось:** ручная проверка на живой камере; скриншот для ревьюера; PR #6 не merge.

---

## Запрос 16 — спринт Security, Model Accuracy & Reliability (7 этапов)

**Сделано — ветка `feature/security-model-quality` @ `981b4d3`, PR #7 (НЕ merge):**
- S1: токен убран из `system/info` (strip query + тест); QR из браузера оператора;
  токена нет в консоли и access-логах (`--no-access-log` при гейте)
- S2: аудит таблицей (401/409/413/422 + тесты); RTSP allowlist; retention +
  orphan sweep (dotfiles exempt — чуть не съел `.gitkeep`, поймано тестом);
  токены не логируются; SQLite локально
- Классы: все 7 проходят; **smoking честно отсутствует** (веса) → UI disable
  через живой `supported_classes()`; уровни RAW/WORKER/INCIDENT разделены
- Training: `datasets/README` + 4 скрипта (split проверен); обучения не было
- Tracking: кросс 10 кадров — 0 switches; ByteTrack отложен (нет показаний)
- FPS: CPU 1.4 / CoreML 35.9 потолки; дефолт не тронут
- UI: причины завершения в диалоге; state machines покрыты
- Доки: SECURITY_AUDIT / MODEL_ACCURACY / PERFORMANCE / REGRESSION reports
- Проверки: `pytest` **86 passed**, фронт **12 passed** + build, `ruff` чисто;
  CI PR #7 запущен (pending на момент записи)

**Дообучение:** нет (нужны размеченные кадры A-O от людей). Веса: Hexmon PPE
(CC BY 4.0 card / AGPL-3.0 meta) + webml yolov8n (AGPL-3.0 meta) — см. THIRD_PARTY.

**Осталось (приоритеты):** P0 footage A-O; P1 обучение при данных; P2 Windows-CI
дождаться; оргвопросы регламента/команды — людям.

---
1. Пороги и в `linq_remote`: `_predict_remote` режет ниже `threshold_for` + тест
2. Окно буквально rolling: голос пишется до переходов, ключи из `votes` тоже
   голосуют; тесты `T,F,F,F,T,T,T` / `T,F,T,F,T` / `T,T,F,T` (ожидания посчитаны
   вручную, не подогнаны)
3. Wording: `LinQ Active` только при `linq_local/remote`; иначе
   `Local ONNX fallback` + `LinQ configured, accelerator inactive`
4. Фронт `Config` + `configOf` несут `temporal_*`/`mask`/`smoking` (бэкенд уже отдавал)
5. `system_info.available_providers` + UI-хинт про idle-ускорение
6. Throughput живого pipeline: CPU 2.2 FPS / CoreML **51 FPS**; дефолт
   `TARGET_FPS=5` не тронут (гейтит только video/RTSP)
7. Таблица под реальные кадры в отчёте; заявления без данных запрещены
8. Проверки: `pytest` **74 passed**, фронт **10 passed** + build, `ruff` чисто;
   CI: macos/ubuntu/frontend pass, windows pending; запушено

**Осталось:** footage A-O; дождаться windows CI; PR #5 не merge.

---

## Запрос 21 — V4 цикл, шаг 1: воспроизводимый sweep порогов V3/V3-clean

**Сделано (НЕ закоммичено, production-веса не тронуты):**
- `training/threshold_sweep.py` — **новый**: один ONNX-проход на (веса × ROI),
  затем sweep `no_helmet/no_vest` 0.35–0.60 @ IoU 0.5, dual pipeline с
  `models/person.onnx`, остальные классы на прод-порогах; temporal-proxy
  (Tracker + assign + TemporalEngine) с честной пометкой-заглушкой;
  `results.json` + `summary.csv` + списки ошибочных кадров + P-R графики
  (matplotlib брался из `.venv-train`)
- `training/sweep_v3_v3clean/` — артефакты: `REPORT.md`, `summary.csv`,
  `results.json`, `error_frames.json`, `pr_no_helmet.png`, `pr_no_vest.png`,
  `recall_vs_fp.png`, `review/fn_no_helmet/` (11 кадров), `review/fp_no_vest/`
  (12 кадров), кэш детекций для `--reuse-cache`
- Подтверждены цифры из GitHub на прод-пороге 0.60: V3 10/31 no_helmet,
  V3-clean 8/31; no_vest V3 25–26/61, V3-clean 18–20/61 (off/auto ±1–2)
- Рекомендации (V3-clean, ROI off): `no_helmet=0.35` (R 0.26→0.55, FP +1,
  P 0.90, F1 max 0.68); `no_vest=0.45` (R 0.33→0.62, FP 6→17, F1 0.66 при
  −9 FP против 0.35). ROI auto: ±1 TP, +50 мс, пороги не пересекает
- Temporal-proxy дал 0 ACTIVE во всех 8 комбо — доказано, что статика из
  124 кадров не скорирует события (ID новый каждый кадр); нужны видео +
  `events.json` и `evaluate_folder.py --events`
- `training/train.py`: добавлены `--patience/--seed/--workers/--device`
  под спецификацию V4 (60–80 эпох, patience 15), `--help` проверен
- Проверки: синтаксис обоих скриптов ok; прогон ~7 мин CPU (410/464 мс
  v3 off/auto, 407/467 мс v3-clean)

**Осталось (по плану V4):** §2 прицельный датасет 300–600 кадров из НОВЫХ
видео, клип-сплит, QA (`prepare_dataset.py` + `clean_v3_leaks.py`); §3 одна
V4 (YOLO11m, 5 классов, 640, 60–80 эпох, patience 15; 960 — отдельный опыт);
§4 двухэтапный head/torso-классификатор только при провале V4 на
видео-критериях (≥80% events, ≤1 FP/час, ≤5 с, ≥5 FPS). Пороги выше — только
экспериментальные оверрайды, не прод-дефолты.

---

## Запрос 22 — V4 dataset pipeline до рабочего состояния (ревью 24a8dce, 3 недоработки)

**Сделано (ветка `fix/ppe-person-roi-inference`, НЕ закоммичено, V4 не обучалась):**
- `training/build_ppe_v4.py` — **новый**, полный путь MP4→датасет: `footage_v4/<video_id>/`
  (ровно одно видео + `LICENSE.json` с consent + опциональные ручные `labels/`) →
  extract (cv2, 1 кадр/2 с, `frames.jsonl` с sha) → autolabel-черновик (опционален,
  ручные метки побеждают, черновики в билд только с явным `--allow-drafts`) →
  QA (валидация + 0 конфликтов opposite-IoU>0.5) → сплит СТРОГО по `video_id`
  (pHash — только warning-аудит, точные пиксель-дубли через сплиты — fail) →
  forbid-guard попиксельным SHA против v1/v3/v3-clean и старых test/val →
  гейты (≥3 видео, `--min-per-class`, пустые выходы) → `data.yaml` (рукописный,
  без PyYAML) + `audit.json` И `audit_leaks.json` (байт-идентичны) + `manifest.json`
  (video_id/sha/draft/license на кадр) + `CREDITS.md` + `contact_sheet.jpg` +
  holdout с `HOLDOUT_README.md`. Импорт только на stdlib+cv2+numpy (CI-safe).
- `--self-test` зелёный: 3 синтетических видео → train/val/holdout по video_id,
  оба audit-файла, manifest; `near_pairs=3` как warning (синтетика похожа).
- `ppe_public_v3/TRAIN_V3.py`: гейт `resolve_gate()` — `AUDIT_VERIFIED.json`
  (legacy) → `audit_leaks.json` → `audit.json`; bootstrap весов — явные
  `--weights` > V2 `best.pt` > `yolo11m.pt` (копия в корне или авто-скачивание
  Ultralytics). Чистый чекаут больше не падает на отсутствующем V2.
- `backend/tests/test_v4_dataset_pipeline.py` — **новый**, 4 теста в CI:
  self-test, оба audit-файла, отказ сплита на 1 видео, forbid-guard на копии
  пикселей. Без футажа/весов/сети.
- `training/V4_DATASET_PLAN.md`: статус `PIPELINE IMPLEMENTED, SELF-TEST GREEN,
  FOOTAGE MISSING` (преждевременное `PIPELINE READY` убрано); стадии сборки с
  командами вместо неверного `--rb-dir footage_v4`; схема `LICENSE.json`;
  фикс гейта/весов; честная сноска про sweep CLI (`ROI=off`, `--grid`,
  `--reuse-cache` не существует).
- Проверки: `ruff` чисто, `pytest` **127 passed** (было 123), оба `--help` ok,
  `data.yaml` парсится yaml. Production-веса не тронуты.

**Осталось:** снять ≥3 видео по протоколу (§Collection), разметить
`no_helmet/no_vest` правилом двух глаз, прогнать `build_ppe_v4.py`,
затем ОДНА V4 по плану; PR #13 остаётся черновиком.

---

## Запрос 23 — push пайплайна + защита по scene_id (ревью: GitHub пуст)

**Сделано (ветка `fix/ppe-person-roi-inference`, V4 не обучалась, веса не тронуты):**
- `training/build_ppe_v4.py`: split key `video_id` → `scene_id`. Поле
  `scene_id` в `LICENSE.json` (default = имя папки, валидация без слэшей);
  все клипы одной съёмки делят сцену и уходят строго в один сплит.
  Audit: `split_key: scene_id` + `scenes`-маппинг; manifest: `scene_id` на кадр;
  гейт ≥3 сцен; HOLDOUT_README — «whole scenes».
- `--self-test`: 4 видео / 3 сцены (два клипа делят `shoot_day`); оба ушли в
  holdout вместе — изоляция доказана логом. min-классы стабильны при любом
  шаффле (6 кадров/видео, `min-per-class=2`).
- `backend/tests/test_v4_dataset_pipeline.py`: проверка сцен (3 сцены —
  3 разных сплита, day-клипы вместе), остальное без изменений.
- `training/V4_DATASET_PLAN.md`: ключ сплита, схема `LICENSE.json` + правило
  «одна съёмка — один scene_id».
- Проверки: self-test PASS, новые тесты 4/4, `ruff` чисто, полный `pytest`
  **127 passed**. Push `0a023cd` (`24a8dce..0a023cd`, `ls-remote` подтверждает);
  CI на GitHub из этой среды не виден (приватный репозиторий, API 401) —
  смотреть Actions/PR #13 владельцу.

**Осталось:** футаж ≥3 сцен по протоколу; одна V4 после зелёного CI.

---

## Запрос 24 — 5 точечных фиксов сборщика + регрессии (аудит fef7450, БЕЗ коммита)

**Сделано (worktree, V4 не обучалась, веса не тронуты):**
- `training/build_ppe_v4.py`: `audit_cross_split()` — декартово сравнение хешей
  вместо позиционного `zip()`; `OPPOSITES` в обе стороны + `find_conflicts()`;
  `forbid_pixel_shas()` падает при пустом хешировании (opt-out — голый
  `--forbid-dirs`); фикстура self-test — разные сцены/движущаяся точка
  (старая давала 37 точных хеш-совпадений через сплиты — новый аудит
  справедливо её забраковал).
- `ppe_public_v3/TRAIN_V3.py`: гейт валидирует содержимое (`dict` с `counts`,
  покрывает и legacy `AUDIT_VERIFIED.json`); `resolve_weights()` без torch +
  громкий фол `YOLO()` с инструкцией для чистой машины.
- `backend/tests/test_v4_dataset_pipeline.py`: 9 тестов (+5 регрессий, тест
  гейта на `{}` переписан — `{}` и мусор теперь отклоняются).
- `training/V4_DATASET_PLAN.md`: одна строка про vacuous-forbid и оба порядка меток.
- Проверки: self-test PASS (day-клипы вместе в holdout), pytest **132 passed**,
  `ruff` чисто. Коммит/push НЕ деланы — нужна явная просьба.

**Осталось:** коммит+push по просьбе владельца; затем заморозка сборщика и данные V4.

---

## Запрос 25 — финал сборщика: pixel-SHA hard fail, веса на чистой машине, строгий гейт

**Сделано (V4 не обучалась, веса не тронуты):**
- `training/build_ppe_v4.py`: `audit_cross_split()` на `(dhash, psha)` —
  равенство pixel SHA через сплиты = hard fail, совпадение только dHash =
  warning (коллизии хешей больше не блокируют); all-pairs сохранён; в audit
  добавлен ключ `duplicate_check`. Граница: psha ловит точные копии под одним
  декодированием, не реэнкод — near-дубликаты покрыты dHash-warning.
- `ppe_public_v3/TRAIN_V3.py`: `KNOWN_PRETRAINED` + явный `--weights yolo11m.pt`
  без файла = идентификатор для авто-скачивания (команда из плана работает на
  чистой машине при сети); путь к несуществующему файлу — громкий фол.
  Гейт: новые аудиты требуют `label_conflict_count/exact_duplicates/draft_labels`
  == 0 и наличие ключей, иначе «пересоберите build_ppe_v4.py»; legacy
  grandfathered (его `errors: []` проверен); near-пары — warning, не гейт.
- Тесты: 11 в файле (+2: dHash-коллизия не фейлит, bare-имя качает/пути падают;
  строгие verdicts гейта). `pytest` **134 passed**, `ruff` чисто, self-test PASS
  (day-клипы вместе в holdout, near_pairs=3).
- Заморозка сборщика после зелёного CI. Дальше — только данные V4.

---

## Запрос 26 — аудит train V3-clean + survey источников под V4 (без скачиваний)

**Факт-аудит `datasets/ppe_v3_clean/train` (2039 img, все с метками):**
- По источникам: rf100 980 img / coco 297 / head 762.
- Боксы: helmet 2964, person 2384, vest 1107, no_helmet 898, no_vest 753,
  mask 1, no_mask 36. По источникам: head = helmet 585 + no_helmet 714
  (БЕЗ person/vest!); rf100 = vest 1063 + no_vest 734 + person 2342 +
  helmet 2097 + no_helmet 94; coco = остатки + mask 37.
- Классы V4 (5): mask/no_mask дропаются (37 боксов — не учится).

**Лицензионный вердикт (проверено API HF, не памятью):**
- `hf-vision/hardhat`: в метаданных НЕТ license-тега, карточка «More
  Information needed» → 762 img / 714 no_helmet (80% train-no_helmet)
  ЗАБЛОКИРОВАНЫ до прояснения. Без них: train 1277 img, no_helmet лишь 184.
  Вывод: новый футаж обязан нести no_helmet (минимум 150 → рекомендую 200+).
- Кандидаты: `keremberke/hard-hat-detection` (19.7k img, hardhat/no-hardhat,
  семейство Roboflow Universe CC BY 4.0, без person/vest — только добавка);
  `roboflow-universe-projects/safety-vests` (CC BY 4.0, 3897 img,
  Vest vs No-Vest ~4200 боксов — закрывает no_vest); `ConstructionNoVest`
  (Kaggle, MIT, 10.7k img / 29.5k no_vest, один класс, требует чистки);
  `51ddhesh/PPE_Detection` (HF cc-by-4.0, Vest 4418/Helmet 2703, без
  no-классов — только негативы/позитивы). Ultralytics Construction-PPE —
  мимо (AGPL-3.0 + нет no_vest).
- Риск пересечений (тот же автор keremberke) закрывается механически:
  forbid-guard + pixel-SHA cross-split аудит билдера при сборке.

**Готовность пре-лейблинга:** доказана smoke-тестом — `ppe-v3-clean.onnx` +
  `person.onnx` грузятся, инференс идёт по всем 7 классам (3 кадра → плотные
  детекты). QA-выход билдера (audit + contact_sheet) генерируется при сборке.

**Следующий шаг:** утверждение shortlist + футаж ≥3 сцен от владельца
  (камеры/условия хакатона); старый test/val не трогаем, holdout — только
  новые сцены. Скачиваний не делал (гигабайты + лицензии — после аппрува).

---

## Запрос 27 — V4 TRAIN собран (инфра заморожена, не тронута)

**Готово в `datasets/v4_train/` (gitignored), обучение НЕ запускалось:**
- Verified pool (только ручные метки): **1150 train + 127 val** (rf100+coco,
  7→5 классов, mask дропнут). Боксы: helmet 2379, person 2384, vest 1107,
  no_vest 753, **no_helmet 184**. Строгий TRAIN-гейт ПРОХОДИТ.
- `head_*` (762) исключены — лицензия hf-vision/hardhat отсутствует в метаданных
  (проверено API). Остаток V3-clean: rf100 CC BY 4.0 (тег API), coco CC BY 4.0
  (README.dataset.txt) — оба подтверждены машинно, не памятью.
- hard-hat-detection (CC BY 4.0, README+лоадер): train.zip 779МБ выкачан,
  13782 img / 28996 hardhat + 9705 no-hardhat; детерминированная выборка
  600+600+60; семантика боксов проверена визуально (head-боксы = как V3).
- Drafts на ревью (НЕ в TRAIN): **1018** (source + person/vest-детектор);
  карантин 242 с причинами; near-пар 14. no_vest никогда не драфтится.
- Артефакты: data.yaml (5 классов), audit.json+audit_leaks.json, manifest.json,
  SOURCES.md (версии/sha/ссылки/лицензии), QA_REPORT.md, 3 контакт-листа.
- Новый файл `training/assemble_v4_train.py` (data-скрипт, frozen-файлы не менялись).
- Safety Vests (CC BY 4.0, 3897 img): недоступен без Roboflow API-ключа —
  точная команда в SOURCES.md, действие за владельцем (либо Kaggle creds).
- Окружение: в .venv доустановлен `datasets` (pip работал, venv-pip сломан).

**Следующее:** ревью 1018 драфтов человеком + футаж ≥3 сцен (no_helmet ≥200!) +
  Safety Vests по ключу → финальная сборка → одно обучение V4.

---

## Запрос 28 — финал V4 TRAIN: 347 принято после 100% визуального QA

**Готово в `datasets/v4_train/` (1448 train + 160 val, gate PASS, V4 не обучалась):**
- Ревью: все 317 t2 + 53 bg просмотрены (24 листа) → accept 297+50, reject 20+3.
  Паттерн отказов подтверждает ценность ревью: 13 жилетов под no_vest, 2 красные
  каски под no-helmet (подтверждено зумом — шум источника), ребёнок, руки в кадре.
- no_helmet: 184 → **1706**; no_vest: 753 → 2273 (1520 derived-torso с подтверждением).
- Пустые coco (175, техника без людей): фильтр person-детектором → 159 фон, 16 дроп.
- Лицензии машинно проверены: rf100/coco/hardhat — CC BY 4.0; head_* (762) исключены.
- Safety Vests недоступен без ключа — команда в SOURCES.md, действие за владельцем.
- Артефакты: accepted/rejected.json, manifest, audit (safety-нули), 4 контакт-листа,
  QA_REPORT.md + `training/V4_TRAIN_QA.md` (в push), decisions (370, в push).
- Запушено: assemble_v4_train.py, v4_review_decisions.json, V4_TRAIN_QA.md, WORKLOG.

**Следующее:** футаж ≥3 сцен + Safety Vests → финальная сборка → одна V4 → ONNX → A/B.

---

## Запрос 29 — спот-чек пройден, V4 обучается (PID 73965)

**Спот-чек (51 кадр: 48 derived + 3 комбо no_helmet&vest):** критических ошибок
нет — каски/боксы/derived-торсы на месте, unworn-каски на столе без боксов.
Датасет заморожен. Оранжевые спецовки/робы = no_vest по конвенции (как раньше).

**Обучение запущено (точной командой из плана, .venv-train, MPS M5):**
`TRAIN_V3.py --dataset datasets/v4_train/train_verified --classes helmet
no_helmet vest no_vest person --weights yolo11m.pt --epochs 80 --patience 15
--name industrial-ppe-v4` (yolo11m.pt взят из корня, гейт пройден).
Фон: `nohup ... > training/runs/v4_console.log`, PID 73965.
Эпоха 1: P 0.489 / R 0.576 / mAP50 0.578, лоссы падают, ~4.5 мин/эпоха
(80 эпох ≈ 6 ч worst case, early stopping раньше). Prod-веса не тронуты.

**Следующее (когда досчитает):** экспорт `models/ppe-v4.onnx` (не трогая
`ppe.onnx`), A/B на старом тесте vs V3-clean, затем живые видео + FPS.

---

## Запрос 30 — V4 обучена, ONNX, A/B против V3-clean (веса прод не тронуты)

**Обучение:** все 80 эпох за 6.35 ч (early stopping не сработал — рост до конца),
MPS M5. Финал на внутр. val (best.pt): all P 0.876 R 0.878 mAP50 **0.928**;
по классам: no_helmet 0.918/0.896/**0.956**, no_vest 0.868/0.832/0.934,
helmet 0.93, vest 0.853, person 0.968. Кривые + confusion matrix в
`training/runs/industrial-ppe-v4/`.

**Экспорт:** `models/ppe-v4.onnx` (80МБ) + `.sha256`, имена классов ок.
Паритет torch/ONNX: 6/6 боксов, diff 0.000px. `ppe.onnx` не тронут (дата 8.10).
Замечание: встроенная проверка `export_onnx.py` падает на этой версии
ultralytics (`predict(tensor CHW)`); сам экспорт корректен, паритет проверен
отдельно — frozen-скрипт не менял.

**A/B на старом тесте (124 кадра, прод-пороги 0.60, dual pipeline):**
- no_helmet: V3-clean 8/31 (R 0.26) → **V4 14/31 (R 0.45)**, F1 0.40→0.58 (FP 1→3)
- no_vest: V3-clean 20/61 (R 0.33) → V4 19/61 (R 0.31), FP 6→11 — плоско/хуже
- helmet R 0.75→0.78, vest плоско, person идентичен (та же person-модель — ок)
- Латентность CPU ONNX: ~400/423 мс (v3-clean/v4). Итог в `training/runs/ab_v3clean_v4.json`.

**Вывод честно:** каски — большой шаг (+75% recall на рабочих порогах);
жилеты — не закрыты (нужны футаж + Safety Vests + sweep порога 0.45).
В прод НЕ ставить. Следующее: живые видео (recall/FP-час/FPS) + решение.

---

## Запрос 31 — независимая видео-оценка V4 (6 YouTube-роликов, 9.5 мин)

**Данные:** 6 роликов буровых (20–244 с, 480p, 2869 кадров @5fps) в
gitignored `footage_v4/yt_eval/` + `MANIFEST.md` (5× Standard → только eval,
1× CC BY). В обучение НЕ идут. Полный просмотр всех 6: **0 явных эпизодов
нарушений** — все в касках; жилеты неоднозначны (полосы на спецовках).
Следствие: recall на видео НЕИЗМЕРИМ; измерены FP-инциденты, FPS, латентность.
`score_incidents.py` — тот же Tracker+TemporalEngine (2.0/2.0с, 4/3).

**Итог (прод-пороги, dual pipeline):**
- FP-инциденты: V3-clean 1, V4 1 (оба — no_vest на полосах спецовок UVh,
  подтверждено кадрами). Остальные 5 клипов — 0/0. Итого ~6 FP/час на
  «рабочей» одежде у ОБЕИХ — критерий ≤1/час НЕ выполнен ни одной (оговорка:
  кейсы пограничные; на обычной одежде не меряли).
- Паровой всплеск no_helmet V3 (6 кадров) в инцидент не вырос ни у кого —
  temporal-фильтр отработал; у V4 таких вспышек вообще не было.
- FPS CPU ONNX на этом Mac: ~2.0–2.1 (467–522 мс/кадр обе модели) —
  критерий ≥5 FPS НЕ выполнен на CPU-пути (нужен CoreML/GPU/меньшая модель).
- Латентность ≤5 с неизмерима (нет онсетов нарушений).
- Recall событий: неизмерим; прокси — статика (no_helmet 14/31, no_vest 19/61).

**Вердикт честно:** V4 не регрессирует vs V3-clean на видео (1:1 FP, чище
покадрово), но критерии приёмки для продакшна НЕ закрыты: нужны видео С
нарушениями (recall), замер FP на обычной одежде и быстрый инференс-путь.
В прод НЕ ставить. Артефакты: `footage_v4/yt_eval/{dets_*,incidents.json}`.

---

## Запрос 32 — воспроизводимый A/B + исправление фактов (ответ на 3 замечания)

**Признано и исправлено:**
- «184 примера»: неряшливая формулировка — V4 училась на **1706** no_helmet
  (184 ручных + 1522 принятых). Таблица расхождений — в отчёте ниже.
- 14/31: подтверждено воспроизводимым прогоном — это **TP=14 при пороге 0.60**
  (прод, dual pipeline). Совпадение со sweep-FN=14 при 0.35 — разные эксперименты.
- GitHub был пуст по V4 → пуш `training/eval_ab_v4.py` + `training/eval_ab_v4_final/`
  (REPORT/results.csv/json, error_frames.json, 4 листа ошибок). Веса — только SHA
  (80+40МБ бинарников в git не кладу; рецепт: сид 17, датасет c1a62db, команда из плана).

**A/B (124 кадра, прод-пороги, commit+SHA в REPORT.md):**
no_helmet: база 8/1/23 → V4 **14/3/17**; no_vest: 20/6/41 → 19/11/42;
helmet 171/177 R, vest плоско, person 1:1. Латентность CPU ~455 мс обе.
Видео-часть встроена в REPORT (1:1 FP, recall неизмерим — 0 эпизодов).

**Структурные пометки (в REPORT.md явно):** same-distribution тест (не CCTV-обобщение);
1520 derived-torso disclosed; CI покрывает код, не зрение.

---

## Запрос 33 — диагностика V4: причины, матрица, track-aware видео-скоринг

**`training/diag_objects.py` (новый) + `training/diag_v4/`:** матрица 2×2×2
на 124 кадрах + пообъектные причины всех FN/FP (assert сверки сумм) + кроп
на каждый ошибочный объект нарушений (144 шт).
- CoreML работает на обеих моделях: **~36 мс p50 → 23–27 FPS** (CPU ~2 FPS).
  Полный app-путь ещё не замерен. ROI auto: 0 прироста recall, p95 CPU 852 мс.
- V4 no_helmet FN=17: bad_iou 5, below 3, no_detection 4, taken_by_other 4,
  no_person 1. V4 no_vest FN=42: taken_by_other 17, below 10, bad_iou 8,
  wrong 3, no_detection 3, no_assign 1. FP: фон доминирует.
- **Вывод: блокирует модель (порог+локализация+теснота сцен), НЕ person/трекинг
  (≤2).** Детали и лимиты — `training/diag_v4/DIAGNOSTIC_REPORT.md`.
- Правки скрипта после прогона — косметика (diff проверен), цифры валидны.

**`backend/scripts/evaluate_folder.py`:** track-aware `match_incidents()` —
GT с worker+боксами, тревога на чужом человеке = FP (`worker_mismatch`),
без боксов — legacy time-only с флагом `worker_unverified`. Тесты
`test_evaluate_folder_events.py`: 6/6. Всего pytest **140 passed**, ruff чисто.

---

## Запрос 34 — ночная смена V4b: данные ×3, обучение идёт

**Данные (merge: 4554 train + 505 val = 5059; gate PASS):**
- Ревью: 1144 кадра (t1/t2/bg), вердиктов 1144: accept 957 / reject 187.
  Флипы сверялись зумом (3 исправления моих ошибок зафиксированы).
  Конвенция no_vest: только явный жилет (рукава/полосы/фасон); робы/куртки — no_vest.
- Новое: ok-пул + viol batch-2 (+800) + hb1204 (2832, CC BY 4.0, классы совпали,
  семантика боксов проверена: torso/голова ок) + coco-full (+10 новых; 91 — дубли
  нашего test/val, выкинуты forbid-ом; 297 — дубли train).
- no_helmet: 184 → **4143**; no_vest: 753 → **5980** (1645 ручных hb!).
- Отложено: vgourbin (429-throttling, клон умер на LFS) — на вторую волну.
- Safety Vests по-прежнему без ключа.

**Обучение V4b:** batch 8 (влезло, 9.8G MPS), PID 3609, ~13 мин/эпоха.
 classes/metrics/экспорт/A-B — по готовности. Prod-веса не тронуты.

---

## Запрос 35 — обучение на арендованном A10 запущено

**Сервер 185.182.108.54 (A10 23GB, Ubuntu 24.04):** venv + torch cu126
(CUDA True) + ultralytics 8.4.175. Датасет 272МБ залит rsync, data.yaml path
починен, gate пройден. Команда: TRAIN_V3, 80 эпох, patience 15, batch 24,
run industrial-ppe-v4cloud. GPU 99%, ~1 мин/эпоха (против 18 мин на M5!).
Эпоха 1: mAP50 0.597 (норма для старта). Локальный прогон M5 ОСТАНОВЛЕН на
эпохе 11/80 (ран industrial-ppe-v4b сохранён, возможен --resume). Работает
только сервер.
Лог: /root/train_cloud.log. ВАЖНО: root-пароль светился в чате — сменить!

---

## Запрос 36 — V4-cloud готова: A/B трёх моделей (веса стянуты с сервера)

**Обучение на A10:** все 80 эпох (~1.5 ч), финал mAP50 **0.939**.
`best.pt` (40.5МБ, sha `ded90261…`) + results.csv стянуты;
`models/ppe-v4cloud.onnx` (80МБ, sha `f02ceb50…`), паритет torch/ONNX 6/6 0.000px.
`ppe.onnx` не тронут.

**A/B на старом тесте (прод-пороги):**
- no_helmet R: 0.258 (8) → 0.452 (14, V4) → 0.387 (**12, V4-cloud, P 0.923**)
- no_vest R: 0.328 (20) → 0.311 (19) → **0.443 (27, V4-cloud)** — жилеты поехали!
- helmet R: 0.753 → 0.780 → **0.841**; vest R: 0.672 → 0.679 → **0.718**
- person 1:1 (та же person-модель). Итог в `training/runs/ab_3way.json`.

**Вывод:** V4-cloud — лучшая overall (жилеты +42% к V4, чистейшая precision касок),
V4 держит recall касок 14 vs 12. До критериев прод (≥80% событий) всё ещё далеко —
нужны видео с нарушениями. В прод НЕ ставить.

---

## Запрос 37 — публикация V4 + V4-cloud (без касанья продакшна)

**Запушено:** `training/eval_ab_v4cloud_final/` (REPORT + 3-way таблица +
листы ошибок V4 vs V4-cloud), `assemble_v4_train.py` (ночные расширения:
batch-2, ok-triage, integrate hb/coco-full, bgfilter), `v4_review_decisions.json`
(1144 вердикта), `V4_TRAIN_QA.md` (пометка superseded), `*.onnx.sha256`,
WORKLOG. Бинарники весов в git НЕ кладутся (80+80+40МБ) — только SHA +
рецепт + реестр в REPORT. `ppe.onnx` (sha `b99fed37…`) не тронут, NO-GO в силе.

---

## Запрос 38 — part-прототип: честный отрицательный результат

**`training/part_proto.py` (новый, прод не тронут):** гео-сплит person→голова/торс,
V4-cloud на кропах (CoreML), правило unknown, пороги прод. Замер на 124:
- no_helmet: baseline 12/31 → part **1/31** (при crop-thr 0.3: 2/31 при 5 FP)
- no_vest: 28/61 → **4/61** (при 0.3: 12/61 при 57 FP); helmet/vest тоже хуже
- unknown: 133 голов + 105 торсов молчат; +196 мс/кадр к латентности
- Баг по дороге (unfiltered persons → 9568 FP) пойман и исправлен до замера

**Вывод:** детектор, ученный на полных кадрах, разваливается на тайтовых кропах
(domain shift масштаба/контекста) — дело не в порогах. Переиспользование той же
модели на кропах — тупик. Рабочие part-варианты: (а) отдельные классификаторы
на авто-кропах (P1, метки бесплатны), (б) пороги/temporal/ансамбль + футаж
с нарушениями. Тупиковую ветку закрываю, время не жгу.

---

## Запрос 39 — один эксперимент без обучения: итог (пуш)

**`training/sweep_recall.py` (новый):** 3 модели × 0.25–0.60, floor 0.05.
Максимум: nh 19/31 (v4cloud@0.25, FP 3), nv 40/61 (v4cloud@0.25, FP 43).
**Невидимых даже на 0.25:** nh 12–14, nv 16–25 — пороги их не вернут.
Split-pick не нужен: один V4-cloud бьёт комбо (дешевле в 2 раза по FPS).

**`training/sweep_temporal.py` (новый):** FP/час на 9.6 мин harsh-видео:
0.25→31–50/h, 0.35→19–38/h, 0.45→19–25/h, 0.60→0–6/h. Окна 6/4, 8/5 НЕ
помогают против persistent-FP. Вывод: порогами критерии недостижимы.

**`evaluate_folder.py`:** --window/--required/--confirm/--clear + печать params.
**`part_proto.py`:** PPE-only `_predict_single` на кропах; сплит: person 7.3 /
PPE-кадр 44.6 / PPE-кроп 38.9 мс (CoreML). Ранее +196 мс включали ~19% waste.
**Тест-флейк:** CWD-зависимость resolve_weights-теста → monkeypatch.chdir;
140/140 с обеих директорий, ruff чисто.
**Итог:** `training/RECALL_REPORT.md` + предложение ровно одного
head/torso-эксперимента (дизайн, НЕ запущен). Без обучения — потолок доказан.

---

## Запрос 40 — unified model: унификация уже внутри YOLO, замерено

**`training/eval_person.py` (новый) + `training/eval_person/person.json`:**
intrinsic person бьёт dedicated: P 0.78–0.81 vs 0.715 (FP 46–53 vs 76),
recall равный (188–193 vs 191), минус ~80 мс/кадр (второй проход не нужен).
**`training/UNIFIED_REPORT.md`:** UVh single vs dual — 0 FP у обоих,
треков 46 vs 50 (меньше призраков), −150 мс. Отдельный person.onnx —
самая шумная часть пайплайна на статике, не стабилизатор.
**Вывод:** ppe-v4cloud.onnx уже есть один ONNX на всех классах; мержить веса
не надо и нельзя; кастомные головы отложены (нужны видео-доказательства потерь
трекинга); дистилляция не запускалась. В прод single-режим НЕ включён.

---

## Запрос 41 — single/dual + полный Runtime + инструмент видео (задача ревьюера)

**A/B (`training/eval_single_dual.py`):** PPE-выходы single vs dual идентичны
(assert). Person: single 188/53/30 (P 0.78) vs dual 191/76/27. Привязка:
no_helmet single 23/31 vs dual 29/31 — честная цена single (−6 покрытий);
vest наоборот лучше (124 vs 117). Экономия ~60–75 мс (не 150 — там сравнивались
разные модели, исправляю свою ошибку).
**Runtime 12+ мин (CoreML, single V4-cloud):** paced 4.81 FPS при 38/48 мс
p50/p95 на кадр (потолок — пейсинг, не вычисления); unpaced-догон в фоне.
CPU ~106%, RSS ~350МБ, провайдеры CoreML+CPU, инциденты/SQLite/JPEG — в отчёте.
**Инструмент:** `record_clip.py` + `label_events.py` (тестированы, headless-fallback
есть). **Безопасность:** SSH-ключ создан; пароль меняется отдельно (см. ниже).

---

## Запрос 42 — замена прод-весов: замерен старый прод, поставлен V4-cloud

**Замер старого прод ppe.onnx на 124 (впервые!):** no_helmet 6/31 (R 0.19),
no_vest **0/61 (R 0.00!)**, helmet 65/227, vest 5/131. Прод был слепым —
хуже ВСЕХ кандидатов по всем PPE-классам. Решение о замене — на цифрах.
**Замена:** бэкап `models/ppe.onnx.bak` (sha b99fed37 ✓), записан V4-cloud
(sha f02ceb50, +39 TP нарушений против 6, те же 14 FP). Откат: одна команда cp.
Сервер перезапущен, модель новая (5 классов в info). Пороги/настройки не тронуты.
**Проверка живьём:** старая camera-сессия умерла при рестарте — жду переподключения
камеры владельцем, затем кадр + инциденты в БД.
**Открытый пункт:** жалоба «рука/нога считывается как каска» — проверю постановкой
(поднять руку к голове по команде) + листами helmet-FP.

---

## Запрос 43 — дрожание цифр в UI: EMA-сглаживание отображения

**Проблема:** уверенности в UI прыгали покадрово (44→51→38) — смотреть невозможно.
**Фикс (только отображение, `backend/app/services/runtime.py`):** EMA α=0.4
по ключу (трек, класс), fallback — квантованный центр; прунинг >2с; сырые
детекты в трекер/движок идут как раньше — поведение детекта НЕ менялось.
Проверка: прыжки 0.13 → 0.035 на synthetic-последовательности; ruff + 142 теста.
Сервер перезапущен (сессию камеры переподключить). Пороги/веса не тронуты.

---

## Запрос 44 — найдена причина «прод не обновился»: ROI=off, сглаживание живо

**Диагностика:** smoothing работает (API-дрейф 0.42→0.36 мелкими шагами).
Но live no_helmet висел ~0.35–0.49: офлайн-замер того же кадра дал
**ROI off 0.76 STRONG vs ROI auto 0.45 weak** — прод-дефолт auto душил детект.
Матрица это предсказывала (0 прироста recall, p95 852 мс).
**Фикс:** локальный `.env` (в репо, веса/пороги/код не тронуты):
`PPE_ROI_MODE=off`. Сервер перезапущен, info подтверждает roi=off.
По пути: разборки с зомби-процессами (pkill не брал — убиваю по порту),
.env сначала лёг не в ту папку (workdir) — исправлено, проверено чтением Settings.
**Жду:** переподключения камеры + 1 мин в кадре без каски → проверка инцидента.

---

## Запрос 45 — три P0 аудита в работе (CI/test, веса, бенч)

**P0-1 (CI):** тесты больше не зависят от локального mp4 — синтетический
fixture через OpenCV в tmp_path. Пройдено локально, ruff чисто.
**P0-2 (веса):** реестр SHA в `download_model.py` (standard + V4/V4-cloud с
provenance); verify-by-filename; mismatch/unregistered/custom-missing — громкие
отказы (проверено); `--rollback` с SHA-проверкой бэкапа (проверено в tmp).
Канон восстановлен: `ppe.onnx`=Hexmon, выбор V4 — явно через
`MODEL_PATH=models/ppe-v4.onnx` в локальном `.env`; сервер подтверждает.
Тесты `test_download_model.py`: 7/7 (без сети). Пороги не тронуты.
**P0-3 (бенч):** `incidents["total"]` + сверка каждого JPEG с файлом
(`snapshots_verified`); найден и признан мой баг: bench хардкодил v4cloud —
теперь модель берётся из активных настроек, в отчёте SHA/пороги/ROI.
Повторный 12-мин прогон идёт (prod V4 dual, CoreML).
**По дороге:** CWD-флейк весов-теста вылечен окончательно.

---

## Запрос 46 — полный аудит: 3 этапа закрыты (без обучения)

**Этап 1 (дефекты):** CI-fixture синтетический (2 passed); реестр SHA +
select/rollback через журнал (11 тестов, без сети); single без person.onnx;
bench: total + snapshots_verified (перепроверен мини-прогоном 529=529) +
12-мин повтор (V4 dual, 25.62 FPS, 0 инцидентов — честно).
**Этап 2 (видео-инструменты):** record — метки времени без дрейфа + sidecar;
label — ленивые кадры (RAM bounded) + bbox мышью + worker_unverified-fallback
уже был в скорере; confidence: raw/display разделены (`raw_confidence`,
weak от отображаемого — пара всегда консистентна, движок ест только raw).
**Этап 3 (чистка):** PDF-дубликат удалён (SHA совпал); research/-архив
(part_crops, history, diag_crops, sweep_review) через git mv; удалены только
base.py/opencv_source.py (импортов нет, тесты зелёные); README (выбор модели,
single) + THIRD_PARTY (локальные V4/V4-cloud, AGPL-оговорка). PR-ветки не тронуты.
**Не закрыто мной:** ротация root-пароля (SSH банится — владельцу через панель);
физическая камера/Telegram (инструменты готовы, NOT TESTED).

---

## Запрос 47 — аудит стейт-машин: покрыто, с оговорками

**Явные машины (образцово):** CameraMachine (8 состояний, 9 событий, строгий
dispatch с RuntimeError на нелегальном переходе); ViolationPhase
(PENDING/ACTIVE/CLEARING) + терминалы resolved/lost/interrupted с
документированными правилами; upload идёт через UPLOADING + session_id-check.
**Без машин (обоснованно):** треки (эфемерные данные, одно правило max_age);
токены pairing (TTL-maps, стандарт); stop-Event/потоки (машина не join'ит потоки);
session.error (носитель сообщения при ERROR); notifier fire-and-forget
(документировано в README); display_conf (кэш 2с, только отображение).
**Минорные зазоры (не баги):** статусы инцидентов в SQLite не guard'ятся на
уровне хранилища (единственный писатель — движок, приемлемо); у треков нет
явного LOST (экспайр молчаливый — «потерянные треки» для A/B считать внешне:
был → пропал > max_age без события); спецом ничего не менял, только прочитал.

---

## Запрос 48 — исправляю 2 свои неточности + двухворкерный сценарий

**Коррекция к запросу 47 (проверено по коду, ревьюер прав):**
- SQLite-статусы ЗАЩИЩЕНЫ: `finish_incident()` пишет только через
  `WHERE id=? AND status='active'` (повторный финиш — ошибка), `update()`
  ограничивает значения/переходы. Мой «никак не защищены» — неверно.
  Остаток-нюанс (не чиню, вне скоупа): `update()` позволяет терминальный
  статус напрямую без `ended_at`/`resolution_reason` — зазор API хранения.
- `ViolationPhase` — 3 значения (PENDING→ACTIVE⇄CLEARING); resolved/lost/
  interrupted — исходы TemporalEngine (в SQLite lost хранится как interrupted).

**`backend/tests/test_two_worker_scenario.py` (новый, 4 passed + 1 xfail):**
синтетика → настоящие Tracker+TemporalEngine (dt 0.2, 2.0/2.0с, 4/3).
Доказано: исчезновение → ровно один `lost`, ноль `resolved`; возврат → ровно
один новый ACTIVE (frames 8–14 и 80–95); compliant-работник — ноль инцидентов.
Пересечение: ID-switch с ложным RESOLVED + дублированным ACTIVE измерен
(f57/f70/f87/f89) — xfail-тест как честная метка известного ограничения
(строгий re-ID отсутствует by design). Синтетика ≠ реальный трекинг: настоящий
ID-switch rate только по размеченному видео (NOT TESTED). Без рефакторинга
машин, без обучения.

---

## Запрос 49 — независимая проверка всего от и до (без новых фич)

**Проверено:** backend pytest 157+1xfail, ruff всего backend чисто;
frontend vitest 23/23 + `vite build` ок; все 12 скриптов `--help` ок;
реестр SHA verify (V4+person); сайт жив (ppe-v4.onnx, roi off, ready);
12 unused-стилей нет (36 ruff-ошибок — только старые training-файлы, не трогал).
**Найдено и исправлено по ходу:** ничего нового (дерево было чистым).
**NOT TESTED локально:** e2e playwright (нет браузеров; покрыт CI);
живая камера (отключена); Telegram; веса V4-cloud на видео с нарушениями.
**Git:** HEAD == origin (ffa5f71), пушить нечего кроме этой записи.

---

## Запрос 50 — полный аудит: трекер починён, инструменты, CI, чистка (push)

**Трекер:** velocity-prediction + строгое one-to-one (без re-ID, без трогания
машин). Кросс-сценарий: было RESOLVED+дубль ACTIVE → стало 2 ACTIVE + 1 lost,
ID {1,2,3} держатся. xfail удалён по честному PASS. По дороге пойман краш
прогноза за границами кадра на реальном видео → clamp + регрессионный тест.
Видео-дампы: инциденты без изменений (1:1). 164 backend + 12 training зелёных.
**Оценка:** time-only vs worker-verified разделены в отчёте скорера.
**Видео-инструменты:** record без дрейфа + sidecar; label ленивый + bbox мышью;
confidence raw/display разделены (`raw_confidence`, weak от отображаемого,
движок ест raw).
**Модели:** single без person.onnx; select идемпотентен (фикс symlink-нормализации);
rollback в предыдущую версию через журнал. **CI:** новый job training-tests
(3 ОС, только pytest+pillow). **Чистка:** research/-архив, PDF-дубликат и мёртвые
модули удалены, README/THIRD_PARTY точечно. PR-ветки не тронуты.
**Не закрыто:** пароль сервера (бан, владельцу через панель); камера/Telegram;
event recall без размеченных нарушений.

---

## Запрос 51 — добивка lint в training (только безопасное)

**Исправлено (17):** автофикс ruff (9: imports, f-strings, re-алиас) + вручную:
B023 (fail() теперь биндит src явно — был корректен по вызову, стал корректен
по построению), RUF059×5, F841, SIM115×2, ISC004, PLC0206.
**Сознательно оставлено (19):** BLE001×9 + S112×5 (tolerance к грязным данным —
ослабление роняло бы сборку), PLW1510×2 (fail-open git-хелпера), SIM102,
FURB161 (frozen dhash-ядро, математически эквивалентно, но не трогаю).
**Доказательство не-регресса:** triage rerun → triage.json побайтово идентичен
(1946 записей). pytest 164 + ruff backend чисто. Без обучения.

---

## Запрос 52 — полный аудит-2: трекер, гейты, видео-инструменты, CI (push)

**Трекер:** velocity-prediction + строгое one-to-one (без re-ID/машин).
Кросс: было RESOLVED+дубль → стало 2 ACTIVE + 1 lost, ID {1,2,3} держатся;
xfail удалён честным PASS. По дороге: краш прогноза за краями на реальном
видео → робастный clamp (все 7 краевых кейсов) + регрессии. Видео-дампы:
инциденты без изменений (1:1). pytest 187, ruff чисто, frontend build ок.
**Оценка:** verified vs time-only разделены + recall-строки; worker_mismatch был.
**Видео:** record без дрейфа (sidecar stamps); label ленивый + bbox мышью
(sx/sy раздельно, 25→7 даёт ровно 7.0); confidence raw/display (`raw_confidence`).
**Модели:** single без person.onnx; select идемпотентен (фикс symlink-нормализации
после пойманного кейса /tmp→/private/tmp); rollback в предыдущую версию.
**Правила:** violation-класс обязателен; 422 на create/update/start; UI зеркалит.
**CI:** job training-tests (3 ОС, pytest+pillow+pyyaml). **Чистка:** research/-архив,
PDF-дубликат, мёртвые модули; README/THIRD_PARTY точечно; PR-ветки не тронуты.
**Не закрыто:** пароль сервера; камера/Telegram; event recall без видео.

---

## Запрос 53 — сверка: аудит c4ed5896 vs факт c553137 (код не менялся)

**Итог проверки:** все 5 требований аудита уже закрыты в HEAD —
pyyaml в CI, clamp_predicted, violation-классы + проверка при start,
timestamps/sidecar, timebase 25→7, sx/sy, recall-строки. Подтверждено чтением
кода, не памятью. Полный прогон: pytest **187 passed**, ruff backend чисто,
дерево чистое. Аудитный CI был снят со старого SHA (до фиксов) — текущий SHA
нуждается в свежем прогоне Actions (проверяет владелец).
**Новых изменений не вносил** — чинить было нечего.

---

## Запрос 54 — калибровка fps камеры (последний пункт аудита)

**Проблема:** writer получал запрошенный `--fps`, хотя камера могла отдавать
10 fps → MP4 ускорялся в 2.5 раза при корректном sidecar.
**Фикс (`record_clip.py`):** калибровка реальной частоты (первые ~90 кадров),
writer fps = min(requested, measured), sidecar пишет requested/measured/fps +
флаг calibrated; ~3 с lead-in задокументирован. Падение на requested при
пустой калибровке — с варнингом, не молча.
**Тесты:** `measure_rate` на скриптованных нерегулярных метках (точное значение
со сталлом); `calibrate_camera` на фейк-камере (моя ошибка в границах теста —
функция была права, 20/2.25=8.89); writer следует устройству (10→25 даёт ~10).
pytest **190 passed**, ruff чисто. Без ML/трекера/машин.

---

## Запрос 55 — перестройка панели: поведение, не стили

**Что бесило и что сделано (фронт only, бэкенд не тронут):**
- Мелькающий мусор → панель объектов показывает только PPE-классы,
  weak — только привязанные к треку; нарушения всегда сверху.
- Нет ответа «что делать» → баннер: красное НАРУШЕНИЕ (кто/что/секунды),
  янтарная ПРОВЕРКА с прогрессом, зелёное «Под контролем · N чел.».
- Нарушители всегда первые в «Людях в кадре» (стабильный порядок).
- Опрос разделён: кадры 1с, инциденты 5с in-place (список не прыгает).
- Устройства перечисляются до старта, а не после.
- `frontend/src/monitor.ts` (чистые функции) + 7 тестов; vitest 30/30,
  `vite build` ок, backend 178 зелёных (не трогал). Перезапуск сервера
  не нужен — обновите страницу. Стили те же.

---

## Запрос 56 — прод обновлён (перезапуск, без смены весов/настроек)

Сервер перезапущен чисто: отдаёт новый бандл `index-DhiOQrMJ.js` (умная панель),
модель `ppe-v4.onnx`, roi off, ready True. Камера переподключается владельцем.
Веса/пороги/settings — без изменений.

---

## Запрос 57 — многокамерность: сетка, общая тревога, стоп-всех

**Боль:** одна камера за раз, тревогу на другой легко пропустить, группового
управления нет, из инцидента нельзя прыгнуть на камеру. Стили не тронуты.
**Сделано (фронт only, API не менялось):**
- Сетка камер (переключатель): плитки с живым кадром (auth-blob, без токена
  в URL), статусом и бейджем тревоги; клик выбирает камеру.
- Общая тревожная полоса: худшая тревога НЕвыбранных камер + кнопка «Смотреть».
- Строки камер показывают тревогу вместо статуса, когда она есть.
- «Стоп все» с подсчётом запущенных (bulk-старт невозможен честно: у каждой
  камеры свой источник — задокументировано решением не делать).
- Клик по камере в журнале инцидентов → эта камера на мониторе.
- `cameraAlert`/`globalAlert` + 2 теста; vitest 32/32, build ок, backend 178
  зелёных (не трогал). Перезапуск не нужен — обновите страницу.
- Честно про модель: другую нейросеть подключить мне нечем — думал сам.

---

## Сквозной остаток (обновляется)

### Бэкенд
- [ ] Реальная PPE-модель с классами mask/smoking (текущие веса их могут не содержать —
  правила тогда молчат, это by design, не баг)
- [ ] `LINQ_ENDPOINT`/`LINQ_API_KEY` от организаторов → в `.env`, проверить `_parse_remote`
  под финальную схему ответа (меняется только он)
- [ ] Скачать веса: `python backend/scripts/download_model.py`, прогнать
  `backend/scripts/smoke_model.py` на своём кадре
- [ ] Реальная отправка в Telegram (нужны токен/чат), сейчас проверено только
  с управляемым HTTP-транспортом в тестах
- [ ] Live-demo §12 end-to-end на камере (снять/надеть каску, вход в зону)
- [x] Закоммитить `feature/linq-backend` → запушено в `origin` (см. запрос 6)

### Фронтенд (не трогали; по коду на сегодня)
- [ ] `frontend/src/App.tsx:169,643,712,880`, `frontend/src/types.ts:37,85-88,121` —
  знают только helmet/vest/zone: нет переключателей mask/smoking и их подписей.
  Нужно, только если включаем расширения на демо
- [ ] Экран с реальными FPS/latency LinQ (§12 п.8) — данные уже есть в API

### Процесс
- [ ] Проверка одного коммита на двух машинах (Windows + A4Tech / MacBook + FaceTime)
- [ ] `THIRD_PARTY.md` — дописать при смене/дообучении модели
