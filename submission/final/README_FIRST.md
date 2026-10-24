# Industrial Safety AI × LinQ — финальный комплект

## Что это

Система промышленной безопасности: камеры → детекция → трекинг →
Safety Radar → инцидент → видеодоказательство → журнал.

PPE-сеть исполняется на **LinQ TPU**. Person-детектор, трекинг, Radar,
правила и UI — на CPU.

## Как запустить

```bash
git clone https://github.com/judogaaa/Industrial-Safety-AI.git
cd Industrial-Safety-AI
git checkout feature/linq-pipeline-app

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "./backend[dev,vision,linq]" -c backend/constraints.txt
cd frontend && npm ci && npm run build && cd ..
python backend/scripts/download_model.py

# локальный режим (CPU, без TPU)
python -m uvicorn app.main:app --app-dir backend --port 8000

# режим LinQ TPU
export INFERENCE_PROVIDER=linq
export LINQ_AMQP_HOST=<broker-host>
export LINQ_TPU_PROGRAM=ppe_nosig_p999.tpu
export LINQ_TPU_CACHE_DIR=/auto/cache/hackathon93154
python -m uvicorn app.main:app --app-dir backend --port 8000
```

Открыть `http://127.0.0.1:8000`.

## Что смотреть

1. **Наблюдение** → выбрать камеру.
2. **Подключить источник** → видеофайл.
3. **Опасные зоны** → нарисовать 1–3 именованные зоны.
4. Смотреть **Радар**: NORMAL → WARNING → CRITICAL.
5. **События** → инцидент → **Replay** (H.264).
6. **Система** → `ppe_runtime=tpu`, `worker_verified`, `remote_calls`.

## Документы

| Файл | Что внутри |
|---|---|
| `metrics/PRESENTATION_METRICS.md` | все замеры с методологией |
| `docs/LINQ_CASE_TRACEABILITY.md` | требование → код → тест → статус |
| `docs/finalization-20261024/RESULTS.md` | независимая приёмка на реальном видео |
| `docs/LINQ_HARDWARE_REPORT.md` | TPU-программа, parity, латентность |
| `Industrial_Safety_AI_LinQ.pptx` / `.pdf` | презентация, 13 слайдов |

## Честные ограничения

- `hardware_verified=false` — протокол не отдаёт аппаратные счётчики.
- PPE слабый: 5 FN каски, 10 FN жилета на 17 примерах.
- Person на CPU, не на TPU.
- Сценарии 02–04, 06–08 не реализованы; 05 — только нарисованные зоны.
- Положительный бег на реальном видео не подтверждён.
