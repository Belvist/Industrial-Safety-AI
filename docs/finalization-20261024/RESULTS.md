# Критическая финализация — 24 октября 2026

**Архитектурное окно PPE исправлено. CPU motion ≥5 FPS подтверждён. Аппаратная приёмка LinQ остаётся BLOCKED; качество PPE не объявляется исправленным.**

Исходная стабильная версия `f27f3912bb5c9cbac736a548e7265b1edf2b60f2` и прежние A/B материалы не изменены: тег `demo-radar-local-20261024`, [предыдущая приёмка](../acceptance/RESULTS.md). Ветка `feature/linq-pipeline-app`, PR #16. UX не переделывался, main не изменялся, force-push отсутствует.

## Что исправлено

- Несколько PPE-запросов на камеру: `PPE_MAX_IN_FLIGHT_PER_CAMERA`, по умолчанию 1. Фактическое окно ограничено также `LINQ_MAX_IN_FLIGHT` (глобальные выполняющиеся задачи). Для окна 4 нужны **обе** настройки =4; включать на реальном TPU только в разрешённой квоте.
- Каждый запрос хранит UUID, source frame ID, timestamp, session generation и копию версий/боксов треков. Ответы буферизуются и применяются по исходному времени, максимум один пакет на motion commit. Истёкший head не блокирует готовые новые ответы; работающий старый запрос помещается в карантин и никогда не применяется позднее.
- Общий FIFO: ≤N исполняющихся задач и ≤8 ожидающих. Очередь физически ограничена, в том числе при частых отменах/рестартах. Камеры не отбирают друг у друга каждый освобождающийся слот. `in_flight`, `queued`, `completed_waiting`, `retired_count`, применённые/отброшенные ответы считаются отдельно.
- Старые кадры после рестарта не увеличивают счётчики новой сессии. PPE использует метрики своей исходной сессии. Смена/неоднозначность ID сбрасывает связь с PPE, неизвестность не подтверждает нарушение.
- Подтверждённый выход сразу в WARNING теперь начинает новый episode_id. Раньше повторный вход мог попасть в старую группу Timeline.
- ONNX worker threads больше не расходуют CPU quota холостым ожиданием. Вход 640×640, веса и confidence thresholds сохранены. На 6 реальных кадрах сравнение обеих моделей с/без spinning дало численную разницу 0: [паритет](measurements/spinning-parity.json); это проверка неизменности вычислений, не recall.
- Replay кодируется один раз из исходных JPEG в H.264, без промежуточного MPEG-4. FFmpeg decoder/encoder ограничены одним потоком; атомарная публикация сохранена. Настройки кодирования доказательств меняются, вход нейросетей не меняется.

## Измерения: одинаковые источник и CPU quota

Intel worker-zone-detection, 1920×1080, 59.94 исходных FPS; CPU quota 2 ядра (`200000 100000`), 2 inference threads. Каждый результат ниже — **3 последовательных повтора**, прогрев 5 с + реальное окно 60 с, включая вычисление person, JPEG, Radar, SQLite и реальное кодирование Replay. Во время замеров не запускались другие инференсы/benchmark. Все 33 основных отчёта имеют `git_dirty=false`. Полные p50/p95/p99, stage sums, CPU и события: [measurements](measurements).

`baseline` — f27f391, `after` — c5b0ecd. Одинаковая подача 5 сравнивается с 5, подача 6 — с 6. Новый default TARGET_FPS=6 выбран после измерения запаса производительности; число на экране не заменяет подсчёт кадров.

| Версия / режим | Подача FPS | PPE | Реальный motion FPS (диапазон) | PPE replies/s, среднее | p95 возраста кадра, мс | max отставание файла, с |
|---|---:|---|---:|---:|---:|---:|
| baseline / motion-one | 5 | выключен | 4.732–4.750 | 0.000 | 868–996 | 4.77 |
| baseline / motion-two | 5 | выключен | 2.933–3.083 | 0.000 | 1576–1699 | 25.97 |
| baseline / ppe-one | 5 | CPU PPE | 3.216–3.350 | 0.461 | 1300–1358 | 22.27 |
| baseline6 / motion-one | 6 | выключен | 4.932–5.432 | 0.000 | 858–912 | 10.81 |
| baseline6 / ppe-one | 6 | CPU PPE | 3.066–3.616 | 0.467 | 1166–1295 | 30.15 |
| after5 / motion-one | 5 | выключен | 4.999–5.000 | 0.000 | 199–516 | 0.71 |
| after5 / motion-two | 5 | выключен | 3.949–4.149 | 0.000 | 1133–1174 | 13.57 |
| after5 / ppe-one | 5 | CPU PPE | 4.333–4.867 | 0.578 | 763–1078 | 8.90 |
| after6 / motion-one | 6 | выключен | 5.999–6.000 | 0.000 | 449–506 | 1.03 |
| after6 / motion-two | 6 | выключен | 4.100–4.183 | 0.000 | 1196–1312 | 19.78 |
| after6 / ppe-one | 6 | CPU PPE | 4.967–5.000 | 0.606 | 843–909 | 10.88 |

Финальная проверка кода `84ee9473d6d7a3a64c0bb97d685482ede1df64de` после FIFO/изоляции метрик: **360 кадров / 60.006411 с = 5.999359 FPS**, PPE выключен, git_dirty=false. Точный SHA сохранён в [release-motion6.json](measurements/release-motion6.json).

При включённом **локальном CPU PPE** стабильные ≥5 motion FPS не доказаны: повторы дали 4.967–5.000, PPE около 0.606 ответа/с. Это не TPU и не 5 полностью проверенных PPE-кадров/с. Дополнительный запуск двух камер с PPE дал около **2.533 motion FPS и 0.300 PPE результата/с на каждую**, один повтор: [two-with-ppe](measurements/two-with-ppe.json). FIFO обеспечивает результаты обеим камерам, но не добавляет вычислительную ёмкость.

Возраст кадра считается от захвата сервером до backend commit; это не sensor-to-browser latency. Для файлов отдельно указан media lag: backpressure замедляет воспроизведение. Две камеры при подаче 6 отставали до 20 с; запуск с CPU PPE — до 11 с. Нулевой dropped для файла не означает realtime. Физический RTSP не принимался. Результаты, включая неудобные, сохранены целиком.

## Реальные сценарии и события

- A/B при прежних TARGET_FPS=5: A — 0 предупреждений/инцидентов; B — WARNING 5.8058, CRITICAL 6.6066, подтверждение 7.2072, один инцидент; прежний lead **0.8008 с** сохранён.
- Новый реальный browser E2E при 6 FPS: lead **0.8342 с по источнику**, один инцидент, H.264, Timeline, решение оператора. [Запись приложения](assets/operator-demo.mp4), [скриншот](assets/critical.png), [результат](measurements/real-browser.json).
- Один трек #1 прошёл U-образную зону дважды: входы 3.6703 и 6.5065 с, два разных episode_id, два инцидента. [Первый Replay](assets/reentry-1.mp4), [второй](assets/reentry-2.mp4), [исходные события](measurements/reentry.json). Геометрия выбрана по известной траектории; это тест свойства, не holdout точности.
- Две именованные зоны: один человек создал по одному отдельному инциденту в каждой; [результат](measurements/two-zones.json).
- С включённым PPE на 60 с исходного промышленного видео зарегистрированы пять отдельных входов, включая треки 3/4 разных работников; [весь результат](measurements/real-ppe-trace.json). Не заявляется пять распознанных видов нарушений.
- На дополнительном лицензированном видео с обычной ходьбой положительных running-событий нет. Положительный реальный бег **не подтверждён**, сценарий остаётся PARTIAL. Порог/минимум наблюдений не ослаблялись.

У промышленных треков этого ролика не встретилось body IoU >0.2. Поэтому реальная устойчивость PPE при тесном перекрытии не доказана. [Аудит наблюдений](overlap-audit.json): 328 повторных отображений треков, каска UNKNOWN в 130, жилет UNKNOWN в 324. Это не 328 независимых инференсов. Чужие нарушения не объединяются по камере/зоне; произвольные физические ID switches не распознаны по независимой разметке и дедупликация для них не гарантируется.

## Качество PPE и Blind Spot Guard

[Таблица настоящих кадров, размеров головы/корпуса, качества, FP/FN](PPE_VISIBILITY.md), [CSV](ppe-visibility.csv), [реальные выходы модели](ppe-detections.json).

17 предварительно размеченных примеров, 3 видео. Для класса отсутствия каски — 5 FN, для отсутствия жилета — 10 FN в обоих режимах full-frame/ROI. FP этих классов — 0, поскольку UNKNOWN не считается успешным ответом. Все размеры людей в этой выборке выше 72 px; нижняя граница не калибрована. Порог 72 — консервативный эвристический запрет, **не доказательство невозможности распознавания**. Причина ошибок модели остаётся UNKNOWN. В этой работе PPE-модель не переобучена и её качество не исправлено обещанием FPS.

## Проверки и границы доказательства

**272 Python tests passed, 3 skipped**, ruff clean, **38 frontend tests**, production build. Phone/workspace/review browser regressions и отдельный real-video E2E прошли. Первые три используют mock media или помеченные fixtures; они не являются доказательством recall.

[Controlled AMQP proof](measurements/controlled-amqp-proof.txt): одна камера реально публикует 1/4 запросов до получения ответов; проходят полный preprocessing 1×3×640×640, сериализация, AMQP correlation и postprocessing. Peer искусственный и явно помечен. В 60-секундном stress-test получено 1856 успешных ответов, 0 failures, peak pending=4, pending at end=0; [машинный отчёт](measurements/amqp-soak.json). Его synthetic replies/sec не являются скоростью LinQ. В тестовом peer устранено чтение peak до окончания публикации; производственный транспорт этим исправлением не менялся. Out-of-order, старый head, timeout, stop/restart, смена/неоднозначность ID, FIFO/backpressure, потеря данных, геометрия и изоляция счётчиков проверены controlled tests. Это не проверка физического TPU/camera outage.

**LinQ BLOCKED:** в окружении нет VPN, TCP grants, endpoint/credentials и разрешённой квоты. [Машинный результат](measurements/linq-blocked.json): worker_verified=false, CPU fallback не запускался. Ни одного нового настоящего worker request эта работа не подтверждает; TPU FPS/latency/failures/positive-negative-UNKNOWN не измерены. Значения concurrency по умолчанию остались 1. Shared TPU не нагружался.

Покрытие кейса: 01 PARTIAL (нет положительного бега); 02/03/04/06/07/08 не реализованы либо не проверены; 05 PARTIAL — реально показаны заданные зоны, но нет распознавания погрузчика/конвейера/груза. Установка на предприятии, независимый event-level recall/FP в час, реальные ID switches и физический RTSP остаются непроверенными. Полная приёмка не объявляется закрытой.

## Повторение

Веса те же, SHA — в [предыдущем source.json](../acceptance/source.json). Источники и их CC BY 4.0 лицензия/точные URL/SHA — в ppe-visibility-labels.json; лицензия сохранена в ../acceptance/INTEL_SOURCE_LICENSE.txt. Клипы/скриншоты добавляют результаты обработки и демонстрационные границы; Intel не одобрял и не подтверждал эти результаты.

```bash
# Каждый output должен быть новым. Требуются существующие модели и FFmpeg/libx264.
python backend/scripts/profile_motion.py --repo "$PWD" --video /tmp/worker-zone-detection.mp4 --output /tmp/motion-run-1 --target-fps 6 --motion-only
python backend/scripts/profile_motion.py --repo "$PWD" --video /tmp/worker-zone-detection.mp4 --output /tmp/ppe-run-1 --target-fps 6
python backend/scripts/profile_motion.py --repo "$PWD" --video /tmp/worker-zone-detection.mp4 --output /tmp/two-run-1 --target-fps 6 --cameras 2
python backend/scripts/profile_motion.py --repo "$PWD" --video /tmp/worker-zone-detection.mp4 --output /tmp/motion-only-run-1 --target-fps 6 --motion-only
# Run both profiles; do not compare motion-only FPS as full PPE performance.
# с теми же моделями (weights runtime assets, в Git не хранятся).
python backend/scripts/evaluate_ppe_visibility.py --repo "$PWD" --video-dir /tmp/industrial-real-footage --labels docs/finalization-20261024/ppe-visibility-labels.json --output /tmp/visibility-run-1
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/finalization-20261024/configs/reentry.json --output /tmp/reentry-run-1 --seconds 12
python -m pytest backend/tests/test_ppe_window.py backend/tests/test_independent_motion.py backend/tests/test_concurrent_pipeline.py -q
```

Для аппаратной приёмки сначала настройте разрешённое подключение и квоту; затем
повторите сценарий C и profile_motion с --provider linq. Только после
разрешения настройте оба лимита PPE и выполните benchmark_linq с --approved-max.
Ни batch, ни новая TPU-компиляция в этой работе не заявляются совместимыми.

## PPE accuracy follow-up (24 Oct 2026)

- Isolated feature work started from `0905ebe` on `fix/ppe-always-on-accuracy`; no production thresholds or user `.env` changed.
- `backend/scripts/profile_motion.py` now enables helmet + vest by default. `--motion-only` is the explicit diagnostic opt-out; `--model` selects an ONNX weight without reading user `.env`. Reports include model SHA, active provider, enabled PPE flags and aggregate person/Radar/PPE rates. Finite-source runs that end during the window correctly report `INCOMPLETE`.
- Four-model same-decoder diagnostic sweep on `datasets/ppe_v3_clean` test set (124 images; IoU 0.5; CPU; floor 0.05; thresholds 0.25–0.60) is in session temp output `/var/folders/54/hs33sq6j50s5jstt9q5062680000gn/T/opencode/ppe-ab-0905` (not checked in). Same-distribution evidence only: `ppe-v3-clean` at no-v* 0.30 got no_helmet 17/31 TP, 4 FP; no_vest 45/61 TP, 28 FP. `ppe-v4` at 0.25 got no_helmet 18/31 TP, 11 FP; no_vest 36/61 TP, 37 FP. `ppe-v4cloud` at 0.25 got no_helmet 19/31 TP, 3 FP; no_vest 40/61 TP, 43 FP. These are detection-box results, not event recall or operating safety metrics; no thresholds were changed from them.
- Dataset `audit_leaks.json` lists 30 near-duplicate cross-split image pairs and two label conflicts. This test set is not a frozen independent holdout and must not be used for model selection or generalization claims. No decoder defect was established; do not change postprocessing based on this sweep.
- Isolated 10-second profile using `models/ppe-v4.onnx` on the previously licensed CC BY running clip completed with `MEASURED`, PPE enabled, CPUExecutionProvider, model SHA `688bbf4d68217d1ff722aac06b2bae0a0296b6e5938d4b29f68623edd2d09f88`: person/Radar 3.997/3.997 FPS, PPE requested/applied 0.700/0.700 per second, CPU 108.7% of one core, source age p95 86.1 ms. Exploratory throughput only; not accuracy acceptance or sustained-camera performance.
- Backend Ruff clean; pytest 279 passed, 1 skipped. Independent holdout A–H videos, verified labels, classifier experiment, real browser E2E and LinQ remain outstanding/blocked.
- Exploratory person-model check on the 17 existing manually labeled person samples: full-frame `person.onnx` matched 17/17 at IoU >= 0.4, with 19 strong detections total; a naive 640x640 overlapping-tile pass matched 13/17 and emitted 45 strong detections. A broader two-window crop pass matched 17/17 but emitted 25 detections versus 19 and increased mean latency from 54.3 ms to 75.2 ms. These small repeated samples show no safe recall gain; no tiling/crop rescue was added.
- Current provisional 17-sample PPE rerun with the real person model and `ppe.onnx` reports no person misses. Full-frame: helmet present 5, missing 0, unknown 12; vest present 0, missing 0, unknown 17. ROI-auto: helmet present 6, missing 1, unknown 10; vest present 3, missing 0, unknown 14. Against known visual labels this still includes 5/4 helmet violation false negatives and 10/10 vest violation false negatives respectively; this is a small diagnostic set, not independent accuracy acceptance. Full-frame and ROI-auto each use one detection frame per label, not a production temporal event pipeline.
- The PPE warning was moved from normal document flow into the video surface and its CSS specificity was corrected so the overlay positioning/background/padding win over generic warning styles. Monitor/focus video sizing no longer derives width from viewport height, aiming to avoid layout reflow. Frontend tests/build pass, but no browser geometry or screenshot comparison was run; visual stability is not yet confirmed.
- Live system-info check: port 8000 reports LinQ remote readiness `failed`, no confirmed accelerator, zero successful TPU results in the last metrics window, 93 remote failures/246 calls since process start, and PPE full-frame fallback reason `roi too small or too large`; PPE work was not skipped. The independent CPU person detector remains active. Isolated port 8011 reports `ppe-v4.onnx` on CPU and no recorded incidents. These states are point-in-time diagnostics, not camera detection acceptance.
- Atharion v1.0 archive was checksum-verified (`f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5`); its 791/170/169 split and 2,002 Person / 1,776 Hardhat / 1,577 Safety Vest box totals match its build metadata. `NOTICE.md` identifies 1,026 Ultralytics Construction-PPE images (AGPL-3.0) and 104 Construction Site Safety images (CC BY 4.0), with per-image attribution. It is positive-only; absent PPE boxes are not verified absences.
- Audit against 33,343 local dataset images found 3 exact SHA matches among the 169 Atharion test images, all in css-kaggle-v28; 110 test images had pHash distance <=5 (102 Ultralytics-source and 8 css-kaggle-v28), which may include exact matches. pHash is only a candidate-overlap signal, not scene-level adjudication. The split is contaminated and was not used for model scoring; do not call it an independent holdout.
- Three exploratory Pexels/Pixabay clips were downloaded and SHA-256 verified, but no visual ground-truth adjudication was possible; no A–H or PPE-state labels are established. Keep them out of TP/FP/FN and event-rate metrics until qualified annotation is available. No accuracy fix or independent PPE acceptance is established.
