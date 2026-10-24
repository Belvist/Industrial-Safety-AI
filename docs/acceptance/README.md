# Критическая приёмка: реальное видео и Predictive Safety Radar

Проверяется реальный путь видео → person ONNX на CPU → tracking → Radar →
SQLite → MP4 → браузер. LinQ PPE работает отдельной ограниченной ветвью,
с исходным кадром, временем и проверкой непрерывности ID. CPU person назван
явно; его скорость не является скоростью TPU. Доступ к LinQ в этом окружении
не настроен; аппаратная приёмка не закрыта.

Исходник: Intel IoT DevKit, `sample-videos/worker-zone-detection.mp4`, CC BY 4.0.
Точная ревизия, URL, SHA256 видео и моделей: [source.json](source.json).
[Текст лицензии](INTEL_SOURCE_LICENSE.txt). Скриншоты и клипы изменены:
добавлены демонстрационные границы, реальные результаты распознавания,
траектории и отметки времени. Intel не утверждал и не одобрял эти результаты.
Границы не обозначают подтверждённые опасные участки исходного предприятия.

## Повторение

Установите зависимости по README_HACKATHON.md и загрузите существующие модели.
Модели не обучались и пороги распознавания не снижались.

```bash
python - <<'PY'
import hashlib, json, pathlib, urllib.request
m = json.loads(pathlib.Path('docs/acceptance/source.json').read_text())
p = pathlib.Path('/tmp/worker-zone-detection.mp4')
urllib.request.urlretrieve(m['source_url'], p)
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['video_sha256']
PY
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-zone-a.json --output /tmp/safety-a --seconds 12
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-zone-b.json --output /tmp/safety-b --seconds 12
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-ppe-c.json --output /tmp/safety-c --seconds 12 --provider linq
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-zone-motion.json --output /tmp/safety-soak --warmup 5 --window-seconds 60
# Два независимых файловых потока, не две физические RTSP-камеры:
python backend/scripts/validate_real_video.py --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-zone-motion.json --output /tmp/safety-two --warmup 5 --window-seconds 60 --cameras 2
```

Каталог output должен быть новым для каждого запуска. Для C настройте
разрешённый VPN, endpoint и credentials в окружении; без endpoint скрипт
возвращает BLOCKED/exit 2 и не запускает подмену TPU локальным инференсом.
Локальная диагностика C: замените `--provider linq` на `--provider local_onnx`.
Её результат не является аппаратной приёмкой.

`report.json` сохраняет SHA/dirty, фактическую длительность окна, число кадров,
скорости независимых контуров, p50/p95/p99, события, журнал и ffprobe Replay.
`*-frames.json` содержит фактические наблюдения. Отсутствие `dropped` для файла
означает backpressure: воспроизведение может отставать. Это не доказательство
realtime RTSP. Проверка камеры/сети отдельно обязательна.

Браузерная проверка на изолированном запущенном backend:

```bash
python e2e/test_real_video.py --base http://127.0.0.1:8002 --master "$ACCEPTANCE_OPERATOR_TOKEN" --video /tmp/worker-zone-detection.mp4 --config docs/acceptance/configs/worker-zone-b.json --artifacts /tmp/safety-browser
```

Тест загружает видео через API и использует настоящий детектор. Он не создаёт
bbox/риски/инциденты вручную. Другие E2E с искусственной камерой или помеченными
fixtures проверяют интерфейс и не доказывают качество распознавания.

## Критерии и ограничения

- A: один проход мимо заданной зоны; ноль WARNING, CRITICAL, инцидентов.
- B: WARNING до входа → CRITICAL → один инцидент → воспроизводимый H.264.
- C: положительное распознавание каски/жилета **реальным LinQ worker**.
- Бег: наличие достаточной истории проверяется отдельно от достоверности
  классификации. Реального положительного примера бега в этом ролике нет.
- Смена ID в плотной толпе не решена гарантированно. Скоростная ассоциация
  сохраняет ID в проверенных тестах; неоднозначное PPE сбрасывается в UNKNOWN.
- Запоздалые ответы, смена геометрии, остановка и перезапуск изолированы
  поколением сессии. При завершении источника ожидающий PPE отменяется;
  `ppe_cancelled_on_source_end` считает такие задачи. Последний PPE может
  остаться без применения; это не скрывается как успешная аналитика.
- Эвристики Blind Spot Guard (72 пикселя высоты в масштабе входа модели,
  яркость/резкость) консервативны и не являются калиброванной оценкой recall.
- Предсказание использует постоянную скорость на коротком горизонте.
  Человек может изменить направление; предупреждение не гарантирует вход.
  Начальная диагностическая конфигурация A выдала два предупреждения при
  последующем повороте. Показательная геометрия выбрана после просмотра этого
  же видео: это демонстрация, а не независимый holdout или оценка FP/час.
- Периодическое PPE на CPU может занимать ядра у person. На LinQ этот вывод
  необходимо проверить заново; никаких новых обещаний TPU FPS нет.

Итоговые измерения, матрица 8 сценариев и непроверенные пункты находятся в
[RESULTS.md](RESULTS.md). До аппаратного C и независимой проверки на площадке
продукт не объявляется полностью готовым к промышленному внедрению.
