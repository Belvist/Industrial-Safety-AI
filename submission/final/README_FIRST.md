# Industrial Safety AI × LinQ — комплект для защиты

## Проверяемая версия

Репозиторий: `https://github.com/Belvist/Industrial-Safety-AI`
Сдаваемая ветка: `fix/ppe-always-on-accuracy`.
Закреплённый релизный тег: `linq-defense-20261025-final` (указатель на
проверенный финальный commit; см. `git rev-parse`). `main` намеренно не используется.

Проверенные локальные изменения перед публикацией: backend Ruff clean,
281 passed / 1 skipped; frontend Vitest 39 passed; TypeScript и Vite build
успешны. Эти результаты относятся к сдаваемой ревизии только после того,
как соответствующий commit появится в публичном репозитории.

## Что это

Система промышленной безопасности: видеопоток → детекция людей → трекинг →
Safety Radar → подтверждённый инцидент → видеодоказательство → журнал.

Рабочий режим зависит от конфигурации. На последней проверенной изолированной
версии Radar и person-детектор работали на CPU; текущая аппаратная приёмка
этой версии на LinQ **не выполнена**. Исторические LinQ-измерения и ответы
worker приведены отдельно в `ACTUAL_RESULTS.md` и `LINQ_PROOF.md`; они не
доказывают доступность LinQ на защите.

## Чистая установка (CPU baseline)

```bash
git clone https://github.com/Belvist/Industrial-Safety-AI.git
cd Industrial-Safety-AI
git checkout linq-defense-20261025-final

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "./backend[dev,vision,linq]" -c backend/constraints.txt
cd frontend && npm ci && npm run build && cd ..
python backend/scripts/download_model.py
```

Эта команда **только проверяет наличие и SHA-256 моделей**, она не скачивает
отсутствующие файлы. Для чистой установки используйте `python run.py`, который
настраивает окружение и собирает UI; если моделей нет, предварительно
разместите registry-веса по процедуре в `backend/scripts/download_model.py` /
`THIRD_PARTY.md`. В текущем чистом клоне загрузку модели отдельно подтвердить
не удалось. `models/ppe-v4.onnx` и `ppe-v4cloud.onnx` — локальные
экспериментальные веса, не baseline для чистого клона. После размещения и верификации `ppe.onnx` и
`person.onnx` можно запустить API вручную:

```bash
python -m uvicorn app.main:app --app-dir backend --port 8000
```

Откройте `http://127.0.0.1:8000`. Не подменяйте baseline локальными
экспериментальными весами.

## LinQ режим (только при наличии доступа)

```bash
export INFERENCE_PROVIDER=linq
export LINQ_AMQP_HOST="broker-host"
export LINQ_TPU_PROGRAM=ppe_nosig_p999.tpu
export LINQ_TPU_CACHE_DIR="shared-program-cache"
python -m uvicorn app.main:app --app-dir backend --port 8000
```

До демонстрации откройте `/api/v1/system/info` и подтвердите текущие
`readiness`, `worker_verified`, `ppe_runtime`, `person_detector` и счётчик
ответов. Если worker не отвечает, переключитесь на CPU-демо и прямо назовите
его CPU. Не показывайте сохранённые исторические ответы как live LinQ.
`hardware_verified=false` означает отсутствие независимых аппаратных счётчиков.

## Проверенный сценарий показа

1. Подключить видеофайл и показать человека/трек.
2. Заранее размеченная именованная зона: показать переход Radar
   NORMAL → WARNING → CRITICAL.
3. Показать ровно один подтверждённый инцидент, временную метку и камеру.
4. Открыть H.264 Replay и решение оператора.
5. Отдельно показать фактический runtime provider и ограничения PPE.

Положительное распознавание бега не подтверждено и не является частью демо.
Физическая IP-камера и новая LinQ-приёмка не подтверждены.

## Материалы

- `ACTUAL_RESULTS.md` — метрики с разделением исторической LinQ-приёмки и текущих ограничений.
- `LINQ_PROOF.md` — исторические аппаратные доказательства и границы.
- `REQUIREMENTS_MATRIX.md` — обязательные требования и статус доказательств.
- `DEMO_SCRIPT.md`, `JURY_QA.md` — сценарий и ответы без неподтверждённых обещаний.
- `Industrial_Safety_AI_LinQ.pptx` / `.pdf` — презентационные материалы; актуальность каждого слайда требует ручной сверки.
