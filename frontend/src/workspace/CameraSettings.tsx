import { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  Loader2,
  Plus,
  ScanLine,
  Video,
} from "lucide-react";
import type { Camera, Config, Info } from "../types";
import { configOf, isActive } from "../types";
import { friendlyError, sourceNames, supports, zonesOf } from "./model";
import PhoneConnect from "../components/PhoneConnect";

export type SettingsControls = { save: () => Promise<boolean> };
function CameraForm({
  camera,
  info,
  busy,
  onSave,
  onConnect,
  onMonitor,
  onZones,
  onDirty,
}: {
  camera: Camera;
  info: Info | null;
  busy: boolean;
  onSave: (config: Config) => Promise<Camera>;
  onConnect: () => void;
  onMonitor: () => void;
  onZones: () => void;
  onDirty: (dirty: boolean, controls: SettingsControls) => void;
}) {
  const [draft, setDraft] = useState(() => configOf(camera)),
    [saved, setSaved] = useState(() => configOf(camera)),
    [tab, setTab] = useState("rules"),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);
  async function save() {
    if (!draft.name.trim()) {
      setError("Укажите название камеры.");
      return false;
    }
    if (
      !Number.isFinite(draft.confirm_seconds) ||
      draft.confirm_seconds < 0.2 ||
      draft.confirm_seconds > 60 ||
      !Number.isFinite(draft.clear_seconds) ||
      draft.clear_seconds < 0.2 ||
      draft.clear_seconds > 60
    ) {
      setError(
        "Время подтверждения и завершения должно быть от 0,2 до 60 секунд.",
      );
      return false;
    }
    if (
      !Number.isInteger(draft.temporal_required) ||
      !Number.isInteger(draft.temporal_window) ||
      draft.temporal_required < 1 ||
      draft.temporal_window > 20 ||
      draft.temporal_required > draft.temporal_window
    ) {
      setError(
        "Число подтверждений должно быть от 1 до размера окна, окно — не больше 20 кадров.",
      );
      return false;
    }
    setSaving(true);
    setError("");
    try {
      const next = await onSave({ ...draft, name: draft.name.trim() });
      setDraft(configOf(next));
      setSaved(configOf(next));
      return true;
    } catch (e) {
      setError(friendlyError(e));
      return false;
    } finally {
      setSaving(false);
    }
  }
  useEffect(() => {
    const actual = configOf(camera);
    if (!dirty && JSON.stringify(actual) !== JSON.stringify(saved)) {
      setDraft(actual);
      setSaved(actual);
    }
  }, [camera]);
  useEffect(() => {
    onDirty(dirty, { save });
  }, [dirty, draft]);
  useEffect(() => () => onDirty(false, { save: async () => true }), []);
  useEffect(() => {
    if (!dirty) return;
    const leave = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", leave);
    return () => window.removeEventListener("beforeunload", leave);
  }, [dirty]);
  return (
    <div className="p-camera-settings">
      <header>
        <div>
          <h2>{camera.name}</h2>
          <p>
            {sourceNames[camera.kind ?? ""] ?? "Источник не подключён"} ·{" "}
            {camera.status === "running"
              ? "Наблюдение работает"
              : isActive(camera)
                ? "Подключение выполняется"
                : "Наблюдение остановлено"}
          </p>
        </div>
        <button className="p-link" onClick={onMonitor}>
          Открыть наблюдение
          <ArrowRight size={16} />
        </button>
      </header>
      <div className="p-settings-tabs" aria-label="Настройки камеры">
        {[
          ["rules", "Основное и проверки"],
          ["zones", "Опасные зоны"],
          ["source", "Подключение"],
        ].map(([key, title]) => (
          <button
            className={tab === key ? "selected" : ""}
            aria-pressed={tab === key}
            key={key}
            onClick={() => setTab(key)}
          >
            {title}
          </button>
        ))}
      </div>
      {tab === "rules" ? (
        <div className="p-settings-content">
          <div className="p-field">
            <label htmlFor="edit-camera-name">Название камеры</label>
            <input
              id="edit-camera-name"
              maxLength={80}
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
            <span className="p-hint">
              Используйте место установки: так оператору будет проще найти
              событие.
            </span>
          </div>
          <div className="p-setting-section">
            <h3>Средства защиты</h3>
            <p>
              Подтверждённое отсутствие средства защиты создаёт инцидент. Если
              оно не видно, система показывает «не определено».
            </p>
            {(
              [
                ["helmet", "Защитная каска"],
                ["vest", "Сигнальный жилет"],
              ] as const
            ).map(([key, title]) => (
              <label className="p-rule-choice" key={key}>
                <div>
                  <strong>{title}</strong>
                  <small>
                    {supports(info, key)
                      ? "Проверять у каждого обнаруженного человека"
                      : "Недоступно в текущей модели"}
                  </small>
                </div>
                <input
                  type="checkbox"
                  checked={draft[key]}
                  disabled={!supports(info, key)}
                  onChange={(e) =>
                    setDraft({ ...draft, [key]: e.target.checked })
                  }
                />
              </label>
            ))}
          </div>
          <details className="p-advanced">
            <summary>Время подтверждения и дополнительные проверки</summary>
            <div className="p-two-fields">
              <div className="p-field">
                <label htmlFor="confirm-seconds">Подтверждение, сек.</label>
                <input
                  id="confirm-seconds"
                  type="number"
                  min="0.2"
                  max="60"
                  step="0.1"
                  value={draft.confirm_seconds}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      confirm_seconds: Number(e.target.value),
                    })
                  }
                />
                <small>Сколько наблюдать нарушение до создания события.</small>
              </div>
              <div className="p-field">
                <label htmlFor="clear-seconds">Завершение, сек.</label>
                <input
                  id="clear-seconds"
                  type="number"
                  min="0.2"
                  max="60"
                  step="0.1"
                  value={draft.clear_seconds}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      clear_seconds: Number(e.target.value),
                    })
                  }
                />
                <small>
                  Сколько наблюдать отсутствие нарушения до завершения.
                </small>
              </div>
            </div>
            {(
              [
                ["mask", "Респиратор"],
                ["smoking", "Курение"],
              ] as const
            )
              .filter(([key]) => supports(info, key))
              .map(([key, title]) => (
                <label className="p-rule-choice" key={key}>
                  <strong>{title}</strong>
                  <input
                    type="checkbox"
                    checked={draft[key]}
                    onChange={(e) =>
                      setDraft({ ...draft, [key]: e.target.checked })
                    }
                  />
                </label>
              ))}
            <label className="p-rule-choice">
              <div>
                <strong>Быстрое перемещение</strong>
                <small>
                  Экспериментальная проверка по движению трека в изображении.
                  Требует проверки на вашей камере.
                </small>
              </div>
              <input
                type="checkbox"
                checked={draft.running_enabled ?? false}
                onChange={(e) =>
                  setDraft({ ...draft, running_enabled: e.target.checked })
                }
              />
            </label>
            <div className="p-two-fields">
              <div className="p-field">
                <label htmlFor="temporal-window">Окно проверки, кадров</label>
                <input
                  id="temporal-window"
                  type="number"
                  min="1"
                  max="20"
                  value={draft.temporal_window}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      temporal_window: Number(e.target.value),
                    })
                  }
                />
              </div>
              <div className="p-field">
                <label htmlFor="temporal-required">Подтверждений в окне</label>
                <input
                  id="temporal-required"
                  type="number"
                  min="1"
                  max={draft.temporal_window}
                  value={draft.temporal_required}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      temporal_required: Number(e.target.value),
                    })
                  }
                />
              </div>
            </div>
            {draft.running_enabled && (
              <div className="p-two-fields">
                <div className="p-field">
                  <label htmlFor="running-speed">
                    Порог движения, высот тела/с
                  </label>
                  <input
                    id="running-speed"
                    type="number"
                    min="0.5"
                    max="10"
                    step="0.1"
                    value={draft.running_speed ?? 1.8}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        running_speed: Number(e.target.value),
                      })
                    }
                  />
                </div>
                <div className="p-field">
                  <label htmlFor="running-time">
                    Длительность движения, сек.
                  </label>
                  <input
                    id="running-time"
                    type="number"
                    min="0.2"
                    max="10"
                    step="0.1"
                    value={draft.running_min_seconds ?? 0.8}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        running_min_seconds: Number(e.target.value),
                      })
                    }
                  />
                </div>
              </div>
            )}
          </details>
        </div>
      ) : tab === "zones" ? (
        <div className="p-settings-content">
          <div className="p-setting-section">
            <ScanLine size={28} />
            <h3>Укажите, куда нельзя заходить</h3>
            <p>
              Отметьте границы прямо на изображении. Система будет предупреждать
              о приближении и подтверждать вход человека в зону.
            </p>
            {zonesOf(camera).length ? (
              <div className="p-saved-zones">
                {zonesOf(camera).map((zone) => (
                  <div key={zone.id || zone.name}>
                    <span>
                      <i />
                      {zone.name}
                    </span>
                    <small>
                      {zone.enabled ? "Контроль включён" : "Контроль выключен"}
                    </small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="p-hint">На этой камере пока нет опасных зон.</p>
            )}
            <button className="p-button p-primary" onClick={onZones}>
              <ScanLine size={16} />
              {zonesOf(camera).length
                ? "Редактировать зоны"
                : "Обозначить первую зону"}
            </button>
            {!isActive(camera) && (
              <p className="p-hint">
                Для разметки нужен свежий кадр — сначала подключите камеру.
              </p>
            )}
          </div>
        </div>
      ) : (
        <div className="p-settings-content">
          <div className="p-setting-section">
            <Video size={28} />
            <h3>{sourceNames[camera.kind ?? ""] ?? "Подключите источник"}</h3>
            <p>
              {camera.kind === "rtsp"
                ? "Адрес сетевой камеры используется только в текущем подключении. При повторном подключении введите его снова."
                : camera.kind === "video"
                  ? "Для следующего анализа выберите видеозапись. Журнал предыдущих событий сохранится."
                  : "Веб-камера передаёт изображение, пока открыто это приложение в браузере."}
            </p>
            <button
              className="p-button"
              disabled={isActive(camera)}
              onClick={onConnect}
            >
              Подключить источник
              <ArrowRight size={16} />
            </button>
            {isActive(camera) && (
              <p className="p-hint">
                Чтобы сменить источник, остановите наблюдение на экране камеры.
              </p>
            )}
          </div>
          {info?.public_url && (
            <PhoneConnect url={info.public_url} cameraId={camera.id} />
          )}
        </div>
      )}
      {error && (
        <div className="p-inline-error" role="alert">
          {error}
        </div>
      )}
      {dirty && (
        <div className="p-save-bar">
          <span>
            Есть несохранённые изменения
            {isActive(camera) && (
              <small>
                Новые правила прервут текущие наблюдения о нарушениях.
              </small>
            )}
          </span>
          <div>
            <button
              className="p-button"
              disabled={saving}
              onClick={() => {
                setDraft(saved);
                setError("");
              }}
            >
              Отменить
            </button>
            <button
              className="p-button p-primary"
              disabled={saving || busy}
              onClick={() => void save()}
            >
              {saving ? (
                <Loader2 className="spin" size={15} />
              ) : (
                <Check size={15} />
              )}
              Сохранить
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
export default function CameraSettings({
  cameras,
  selected,
  info,
  busy,
  onSelect,
  onSave,
  onConnect,
  onMonitor,
  onZones,
  onDirty,
}: {
  cameras: Camera[];
  selected: string;
  info: Info | null;
  busy: boolean;
  onSelect: (id: string) => void;
  onSave: (config: Config) => Promise<Camera>;
  onConnect: (camera?: Camera) => void;
  onMonitor: () => void;
  onZones: () => void;
  onDirty: (dirty: boolean, controls: SettingsControls) => void;
}) {
  const camera = cameras.find((c) => c.id === selected) ?? cameras[0];
  return (
    <section>
      <div className="p-page-heading">
        <div>
          <span className="p-kicker">УПРАВЛЕНИЕ</span>
          <h1>Камеры</h1>
          <p>Подключения, проверки и опасные зоны для каждой рабочей зоны.</p>
        </div>
        <button
          className="p-button p-primary"
          disabled={busy}
          onClick={() => onConnect()}
        >
          <Plus size={16} />
          Новая камера
        </button>
      </div>
      <div className="p-settings-layout">
        <aside className="p-settings-cameras">
          <span className="p-list-label">ВАШИ КАМЕРЫ</span>
          {cameras.map((c) => (
            <button
              key={c.id}
              className={camera?.id === c.id ? "selected" : ""}
              aria-pressed={camera?.id === c.id}
              onClick={() => onSelect(c.id)}
            >
              <Video size={18} />
              <div>
                <strong>{c.name}</strong>
                <small>
                  {c.status === "running"
                    ? "В работе"
                    : c.status === "error"
                      ? "Ошибка подключения"
                      : "Остановлена"}
                </small>
              </div>
              <i className={c.status === "running" ? "live" : ""} />
            </button>
          ))}
        </aside>
        {camera ? (
          <CameraForm
            key={camera.id}
            camera={camera}
            info={info}
            busy={busy}
            onSave={onSave}
            onConnect={() => onConnect(camera)}
            onMonitor={onMonitor}
            onZones={onZones}
            onDirty={onDirty}
          />
        ) : (
          <div className="p-empty">
            <Video size={30} />
            <h2>Подключите первую камеру</h2>
            <button className="p-button p-primary" onClick={() => onConnect()}>
              Подключить камеру
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
