import {
  Activity,
  ArrowRight,
  Check,
  CircleHelp,
  HardDrive,
  Radio,
  ShieldCheck,
  Video,
  AlertCircle,
} from "lucide-react";
import type { Camera, Info, Result } from "../types";
import { clientProfile, uiRate } from "../clientMetrics";
import { inferenceLabel } from "../inferenceLabel";
export default function SystemPage({
  info,
  cameras,
  camera,
  online,
  result,
  onHelp,
  onCameras,
}: {
  info: Info | null;
  cameras: Camera[];
  camera?: Camera;
  online: boolean;
  result?: Result | null;
  onHelp: () => void;
  onCameras: () => void;
}) {
  const remote = info?.inference?.mode === "linq_remote",
    worker = !!info?.inference?.worker_verified;
  const checks = [
    {
      icon: Activity,
      title: "Приложение",
      state: online ? "Соединение установлено" : "Сервер недоступен",
      ok: online,
      description: online
        ? "Рабочее пространство подключено к локальному серверу."
        : "Запустите приложение и дождитесь подключения.",
    },
    {
      icon: ShieldCheck,
      title: "Распознавание",
      state: !info?.model_ready
        ? "Требуется настройка"
        : remote && !worker
          ? "LinQ ожидает проверки"
          : "Готово к работе",
      ok: !!info?.model_ready && (!remote || worker),
      description: !info?.model_ready
        ? "Модуль распознавания не загружен. Перезапустите приложение штатным скриптом запуска."
        : remote
          ? worker
            ? "Рабочий LinQ ответил на запрос. PPE обрабатывается удалённо, люди — на CPU."
            : "Подключение настроено. Реальная работа ускорителя подтвердится после успешной обработки кадра."
          : "Модели обрабатывают кадры на этом компьютере.",
    },
    {
      icon: Video,
      title: "Камеры",
      state: `В работе: ${cameras.filter((c) => c.status === "running").length} из ${cameras.length}`,
      ok: cameras.some((c) => c.status === "running"),
      description:
        "Для проверки источника откройте его наблюдение и дождитесь свежего кадра.",
    },
    {
      icon: HardDrive,
      title: "Журнал",
      state: online ? "История доступна" : "Нет связи",
      ok: online,
      description:
        "Подтверждённые события и решения оператора сохраняются на этом устройстве.",
    },
    {
      icon: Radio,
      title: "Уведомления Telegram",
      state: info?.telegram_ready ? "Настроены" : "Не подключены",
      ok: !!info?.telegram_ready,
      description: info?.telegram_ready
        ? "Снимки подтверждённых событий могут отправляться в настроенный чат."
        : "Локальное наблюдение и журнал работают без Telegram.",
    },
  ];
  return (
    <section className="p-system-page">
      <div className="p-page-heading">
        <div>
          <span className="p-kicker">ГОТОВНОСТЬ К РАБОТЕ</span>
          <h1>Система</h1>
          <p>Проверьте подключение и распознавание перед началом наблюдения.</p>
        </div>
        <button className="p-button" onClick={onHelp}>
          <CircleHelp size={16} />
          Как пользоваться
        </button>
      </div>
      <div className="p-system-checks">
        {checks.map(({ icon: Icon, title, state, ok, description }) => (
          <div key={title}>
            <Icon size={22} />
            <div>
              <strong>{title}</strong>
              <p>{description}</p>
            </div>
            <span className={ok ? "ready" : "attention"}>
              {ok ? <Check size={15} /> : <AlertCircle size={15} />} {state}
            </span>
          </div>
        ))}
      </div>
      <div className="p-system-actions">
        <button className="p-button p-primary" onClick={onCameras}>
          Перейти к камерам
          <ArrowRight size={16} />
        </button>
        <span>
          Достаточно одной камеры, чтобы проверить весь рабочий сценарий.
        </span>
      </div>
      <details className="p-advanced p-diagnostics">
        <summary>Техническая диагностика</summary>
        <p>
          {camera?.name ?? "Камера не выбрана"} · {inferenceLabel(info)}
        </p>
        {info?.model_error && (
          <p className="p-diagnostic-error">{info.model_error}</p>
        )}
        <div className="p-diagnostic-grid">
          {[
            ["Захват", result?.metrics?.rates.capture?.fps],
            ["Люди · CPU", result?.metrics?.rates.person?.fps],
            ["Safety Radar", result?.metrics?.rates.radar?.fps],
            ["LinQ TPU", result?.metrics?.rates.tpu?.fps],
            ["Аналитика", result?.metrics?.rates.analytics?.fps],
            ["Экран", uiRate(camera?.id)?.fps],
          ].map(([title, value]) => (
            <div key={String(title)}>
              <span>{String(title)}</span>
              <strong>
                {typeof value === "number"
                  ? `${value.toFixed(1)} FPS`
                  : "Нет замера"}
              </strong>
            </div>
          ))}
        </div>
        <p>
          Это разные измерения. Частота обновления интерфейса и целевые 5 FPS не
          доказывают скорость обработки LinQ.
        </p>
        <dl className="p-evidence-facts">
          <div>
            <dt>Пропущено при перегрузке</dt>
            <dd>{result?.metrics?.rates.dropped?.total ?? 0}</dd>
          </div>
          <div>
            <dt>Отклонено устаревших результатов</dt>
            <dd>{result?.metrics?.rates.stale?.total ?? 0}</dd>
          </div>
          <div>
            <dt>Возраст последнего результата</dt>
            <dd>
              {result?.data_age_ms !== undefined
                ? `${(result.data_age_ms / 1000).toFixed(1)} с`
                : "Нет замера"}
            </dd>
          </div>
          <div>
            <dt>Окно аналитики</dt>
            <dd>
              {result?.metrics
                ? `${result.metrics.window_seconds.toFixed(1)} с · ${result.metrics.rates.analytics?.window_count ?? 0} кадров`
                : "Нет замера"}
            </dd>
          </div>
          {Object.entries(clientProfile()).map(([name, stage]) => (
            <div key={name}>
              <dt>
                {name === "api_roundtrip_ms"
                  ? "API в браузере, p50 / p95"
                  : "Загрузка кадра, p50 / p95"}
              </dt>
              <dd>
                {stage.p50} / {stage.p95} мс
              </dd>
            </div>
          ))}
        </dl>
        {result?.metrics && (
          <div className="p-stage-profile">
            <h3>Задержки стадий, мс</h3>
            <p>
              p50 / p95 / p99. Время worker входит в удалённое ожидание; стадии
              нельзя механически складывать.
            </p>
            {Object.entries(result.metrics.stages)
              .filter(([key]) => key.endsWith("_ms"))
              .map(([key, stage]) => (
                <div key={key}>
                  <span>
                    {(
                      {
                        inference_ms: "Инференс",
                        analytics_ms: "Аналитика",
                        tracking_ms: "Трекинг",
                        radar_ms: "Safety Radar",
                        jpeg_ms: "JPEG",
                        person_ms: "CPU Person",
                        end_to_end_ms: "Кадр до результата",
                        capture_decode_ms: "Декодирование",
                      } as Record<string, string>
                    )[key] ?? key}
                  </span>
                  <span>
                    {stage.p50} / {stage.p95} / {stage.p99}
                  </span>
                </div>
              ))}
          </div>
        )}
      </details>
    </section>
  );
}
