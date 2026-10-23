# Accuracy + Performance report (sprint `feature/accuracy-performance-product`)

Все цифры — реальные замеры, не оценки. Пустые ячейки = ещё не замерено.

## Baseline (ветка `feature/linq-backend` @ `2b91cfd`, Mac, venv python3.12)

- Backend tests: **50 passed**; frontend tests: **4 passed**; `vite build`: ok; `ruff`: clean
- Inference mode (default): `local_onnx`, dual-model (PPE `ppe.onnx` + person `person.onnx`)
- Thresholds: единый `confidence_threshold = 0.4` для всех классов
- Tracking: жадный IoU-трекинг (`Tracker`, IoU ≥ 0.2, expire 2 c) — `services/engine.py`
- PPE→Worker: анатомические band'ы (каска верх / жилет torso / маска лицо), `evaluate()`
- Incident: `TemporalEngine` PENDING (confirm) → ACTIVE ↔ CLEARING → RESOLVED/INTERRUPTED
- ORT providers на Mac: `CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider`
- Live API frame (noise 640×480, dual-model CPU): **latency 483 мс**

## §1 Per-class thresholds

| Класс | Было | Стало | Env |
|---|---|---|---|
| person | 0.4 (общий) | 0.45 | `PERSON_THRESHOLD` |
| helmet | 0.4 | 0.50 | `HELMET_THRESHOLD` |
| no_helmet | 0.4 | **0.60** | `NO_HELMET_THRESHOLD` |
| vest | 0.4 | 0.50 | `VEST_THRESHOLD` |
| no_vest | 0.4 | **0.60** | `NO_VEST_THRESHOLD` |

Пороги конфигурируемы, единый `threshold_for()` с fallback на глобальный.
Слабый no_helmet 0.45 при пороге 0.60 отрезается (тест).

## §4 Temporal consensus

- Было: один кадр с нарушением сразу идёт в PENDING-отсчёт
- Стало: окно `TEMPORAL_WINDOW=4`, требуется `TEMPORAL_REQUIRED=3` позитивных наблюдений
- Реализация: `TemporalEngine.votes` (bounded deque), пропуск кадра = голос «нет»,
  история чистится при gap/resolved; конфиг в `CameraConfig` + env
- Проверено тестами: 1 кадр → тихо; 2/4 → тихо; пропуск внутри серии лишь откладывает;
  custom window=2/required=2 работает

## §6 Single vs dual model (одинаковые кадры, CPU)

Замер: `backend/scripts/benchmark_models.py --runs 3`, bus.jpg 1080×810 (публичный
сэмпл Ultralytics, он же в VALIDATION.md), CPU ONNX, venv python3.12, этот Mac.

| Режим | latency/кадр | FPS | person proxy/кадр |
|---|---|---|---|
| A dual (PPE+person) | 448 мс (p95 443) | 2.2 | 3.00 |
| B single (только PPE) | 390 мс (p95 388) | 2.6 | **0.00** |

Вывод: выигрыш single всего 1.15x (person-модель лёгкая), а людей PPE-модель
на уличной сцене не находит вовсе (0 vs 3). Решение: **dual остаётся
рекомендуемым** (`USE_SEPARATE_PERSON_MODEL=true`); пересмотреть только
на индустриальных кадрах с людьми в СИЗ.

## §7 Inference pipeline audit

Найдено / исправлено:

- ONNX threading: `intra_op_num_threads` выставляется (§8: см. ниже) — ok
- Session создаётся один раз (`_session` guard в `load()`) — ok
- Двойной JPEG encode (annotated + raw) — оба нужны (evidence + точный кадр), не трогаем
- Двойной resize/letterbox — неустраним в dual-режиме (две модели), аргумент за single в будущем
- Двойного инференса одного кадра нет — ok
- Video: пропуск кадров по таймлайну уже есть (latest wins) — ok
- **RTSP: буфер OpenCV не ограничивался → отставание от realtime. Исправлено:
  `CAP_PROP_BUFFERSIZE=1` (latest-frame wins)**
- Browser: кадров очереди нет, обработка по запросу — ok

## §8 CoreML vs CPU (Mac, настоящий бенчмарк)

bus.jpg, dual-model, этот Mac, venv python3.12. CoreML берёт 230/233 и 293/297
узлов графа, остальное CPU. Детекции совпадают с CPU с точностью до округления
(те же 5 боксов: 4×person, 1×no_vest).

| Провайдер | чистый inference | полный pipeline (`Runtime.process`) |
|---|---|---|
| CPUExecutionProvider | 437 мс / 2.3 FPS | 456 мс / 2.2 FPS |
| CoreMLExecutionProvider | **16 мс / 62.6 FPS** | **20 мс / 51 FPS** |

Вывод: на Mac для демо ставить `ONNX_PROVIDERS=CoreMLExecutionProvider`
(ускорение ~25x end-to-end). Дефолт остаётся CPU (детерминизм CI).
`TARGET_FPS=5` ограничивает только video/RTSP-воркеры (browser paced клиентом);
поднимать до 10–15 только после live-проверки. LinQ-код не тронут.

## T3 Transport ceilings (потолок с JPEG-транспортом, полный `process()`)

bus.jpg 1080×810 ×8 кадров, первые 2 прогревочных отброшены:

| Провайдер | потолок FPS | мс/кадр | тянет 5/10/15? |
|---|---|---|---|
| CPU | 1.4 | 693 | нет даже 5 (совпадает с наблюдаемыми ~1.8) |
| CoreML | 35.9 | 28 | да, все три с запасом ~2x на 15 |

Дефолт `TARGET_FPS=5` не меняется: на CPU это выше потолка, на CoreML требует
live-подтверждения (сеть + SQLite + Telegram под нагрузкой не замерялись).

## F6 Old vs new on identical frames and weights (fix-visibility)

bus.jpg, real `ppe.onnx`+`person.onnx`, CPU. OLD = global 0.40 (linq-backend);
NEW = per-class thresholds + floor 0.25.

| Detection | OLD | NEW |
|---|---|---|
| no_helmet 0.38 | dropped, invisible | kept, flagged weak, no violation |
| no_vest 0.52 | passed as full detection (could violate!) | kept weak, no violation |
| person 0.89/0.88/0.88 | tracks | tracks (unchanged) |
| person 0.44 | track (below person 0.45) | no track, raw only |
| noise frame | [] | [] |

Nothing disappeared: every OLD detection is present in NEW (strong or weak);
weak persons no longer spawn Worker tracks. Violation thresholds unchanged.

## §H7 Real-footage comparison (заполнить после съёмки A–O)

| Метрика | До (main) | После (эта ветка) |
|---|---|---|
| no_helmet найден из N снятий | | |
| helmet ложные missing | | |
| vest на полу как PPE / инцидентов | | |
| Worker ID switches за ролик | | |
| FP-инцидентов всего | | |

Заявление «модель стала точнее» запрещено без этих цифр: веса не менялись,
улучшен только системный слой (пороги + привязка + консенсус).

## Acceptance gates (§18)

- [x] G1: vest без человека — не PPE Worker, не incident (тест)
- [x] G2: no_helmet ниже threshold — не violation (тест, порог 0.60)
- [x] G3: один случайный кадр — не incident (тест)
- [x] G4: подтверждённое нарушение — ровно один incident (тесты + дедупликация engine)
- [x] G5: устранение → RESOLVED (тест)
- [x] G6: потеря источника → INTERRUPTED (тесты; `_interrupt` меняет движок целиком)
- [ ] G7: Worker ID стабильнее baseline — код (2-stage tracker + тесты) готов,
  нужна footage-проверка по `docs/CAMERA_VALIDATION_PLAN.md` (K, M)
- [x] G8: single vs dual benchmarked (dual остаётся: 448 мс vs 390 мс, persons 3 vs 0)
- [x] G9: FPS/latency записаны (baseline 483 мс live-noise; bus CPU 437–448 мс;
  CoreML 16 мс; single 390 мс)
- [x] G10: UI показывает настоящий provider (бейдж + `system_info.inference`)
- [x] G11: backend tests зелёные (см. ниже)
- [x] G12: frontend tests/build зелёные (см. ниже)
- [ ] G13: CI — смотреть в GitHub Actions после PR
