import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FileVideo,
  Loader2,
  Network,
  Upload,
  Video,
} from "lucide-react";
import Dialog from "./Dialog";
import type { Camera, Info } from "../types";
import type { ConnectionDraft, SourceChoice } from "./model";
import { friendlyError, supports } from "./model";

export default function ConnectCamera({
  camera,
  info,
  onClose,
  onConnect,
}: {
  camera?: Camera;
  info: Info | null;
  onClose: () => void;
  onConnect: (draft: ConnectionDraft) => Promise<void>;
}) {
  const [step, setStep] = useState(1);
  const [draft, setDraft] = useState<ConnectionDraft>({
    name: camera?.name ?? "Новая камера",
    source:
      camera?.kind === "video" || camera?.kind === "rtsp"
        ? camera.kind
        : "browser",
    device: "",
    file: null,
    url: "",
    helmet: camera?.helmet ?? supports(info, "helmet"),
    vest: camera?.vest ?? supports(info, "vest"),
  });
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function discover() {
    try {
      const all = await navigator.mediaDevices?.enumerateDevices();
      if (!all)
        throw new Error(
          "Откройте приложение на localhost или через HTTPS для доступа к камере.",
        );
      setDevices(all.filter((d) => d.kind === "videoinput"));
    } catch (e) {
      setError(friendlyError(e));
    }
  }
  function validate() {
    if (!draft.name.trim())
      return "Дайте камере название — например, «Сборочный цех».";
    if (draft.source === "video" && !draft.file)
      return "Выберите видеофайл для анализа.";
    if (draft.file && draft.file.size > 200 * 1024 * 1024)
      return "Видео должно быть не больше 200 МБ.";
    if (draft.source === "rtsp") {
      try {
        const url = new URL(draft.url);
        if (
          !["rtsp:", "rtsps:", "http:", "https:"].includes(url.protocol) ||
          !url.hostname
        )
          throw new Error();
      } catch {
        return "Укажите полный адрес камеры: rtsp://… или https://…";
      }
    }
    return "";
  }
  async function next() {
    const invalid = validate();
    setError(invalid);
    if (invalid) return;
    if (step === 1) {
      setStep(2);
      return;
    }
    setBusy(true);
    try {
      await onConnect(draft);
    } catch (e) {
      setError(friendlyError(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog
      title={camera ? "Подключить камеру" : "Новая камера"}
      description="Источник → правила → наблюдение"
      onClose={onClose}
      className="p-connect-dialog"
      footer={
        <>
          <button
            className="p-button"
            onClick={() =>
              busy ? onClose() : step === 2 ? setStep(1) : onClose()
            }
          >
            {step === 2 && <ArrowLeft size={16} />}{" "}
            {busy ? "Отменить подключение" : step === 2 ? "Назад" : "Отмена"}
          </button>
          <button
            className="p-button p-primary"
            disabled={busy || !info?.model_ready}
            onClick={() => void next()}
          >
            {busy ? (
              <Loader2 className="spin" size={16} />
            ) : step === 2 ? (
              <Video size={16} />
            ) : null}
            {busy
              ? "Подключаем…"
              : step === 1
                ? "Продолжить"
                : "Начать наблюдение"}
            {step === 1 && <ArrowRight size={16} />}
          </button>
        </>
      }
    >
      <div className="p-steps">
        <span className={step === 1 ? "current" : "complete"}>
          <i>{step > 1 ? <Check size={12} /> : "1"}</i>Источник
        </span>
        <b />
        <span className={step === 2 ? "current" : ""}>
          <i>2</i>Правила
        </span>
      </div>
      {step === 1 ? (
        <>
          <div className="p-field">
            <label htmlFor="camera-name">Как назвать камеру</label>
            <input
              id="camera-name"
              value={draft.name}
              maxLength={80}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              placeholder="Например, сборочный цех"
              autoFocus
            />
          </div>
          <fieldset className="p-source-choices">
            <legend>Откуда получать изображение</legend>
            {(
              [
                ["browser", "Веб-камера", "На этом компьютере", Video],
                ["video", "Видеофайл", "Для анализа записи", FileVideo],
                ["rtsp", "IP-камера", "Сетевой видеопоток", Network],
              ] as const
            ).map(([key, title, sub, Icon]) => (
              <button
                key={key}
                aria-pressed={draft.source === key}
                className={draft.source === key ? "selected" : ""}
                onClick={() => {
                  setDraft({ ...draft, source: key as SourceChoice });
                  setError("");
                }}
              >
                <Icon size={22} />
                <strong>{title}</strong>
                <small>{sub}</small>
              </button>
            ))}
          </fieldset>
          <div className="p-source-input" key={draft.source}>
            {draft.source === "browser" ? (
              <>
                <div className="p-field">
                  <label htmlFor="camera-device">Устройство</label>
                  <select
                    id="camera-device"
                    value={draft.device}
                    onChange={(e) =>
                      setDraft({ ...draft, device: e.target.value })
                    }
                  >
                    <option value="">Камера по умолчанию</option>
                    {devices.map((d, i) => (
                      <option value={d.deviceId} key={d.deviceId || i}>
                        {d.label || `Камера ${i + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
                <button className="p-link" onClick={() => void discover()}>
                  Обновить список устройств
                </button>
                <p className="p-hint">
                  Доступ к камере браузер запросит при запуске. Закройте
                  программы, которые могут её использовать.
                </p>
              </>
            ) : draft.source === "video" ? (
              <>
                <label className="p-upload">
                  <Upload size={24} />
                  <strong>
                    {draft.file?.name ?? "Выберите видео или перетащите сюда"}
                  </strong>
                  <span>MP4, MOV, WebM · до 200 МБ</span>
                  <input
                    type="file"
                    aria-label="Выбрать видео"
                    accept="video/mp4,video/webm,video/quicktime"
                    onChange={(e) =>
                      setDraft({ ...draft, file: e.target.files?.[0] ?? null })
                    }
                  />
                </label>
                <p className="p-hint">
                  Анализируется настоящая запись. Скорость обработки может
                  отличаться от скорости воспроизведения.
                </p>
              </>
            ) : (
              <>
                <div className="p-field">
                  <label htmlFor="stream-url">Адрес видеопотока</label>
                  <input
                    id="stream-url"
                    type="password"
                    autoComplete="off"
                    value={draft.url}
                    onChange={(e) =>
                      setDraft({ ...draft, url: e.target.value })
                    }
                    placeholder="rtsp://адрес-камеры/stream"
                  />
                </div>
                <p className="p-hint">
                  Камера должна быть доступна этому компьютеру. Адрес
                  используется только для текущего подключения.
                </p>
              </>
            )}
          </div>
        </>
      ) : (
        <>
          <div className="p-step-intro">
            <span className="p-kicker">ШАГ 2 ИЗ 2</span>
            <h3>Что нужно контролировать?</h3>
            <p>
              Начните со средств защиты. Опасные зоны можно будет отметить прямо
              на изображении после подключения.
            </p>
          </div>
          {(
            [
              [
                "helmet",
                "Защитная каска",
                "Событие создаётся, если отсутствие каски подтверждено во времени.",
              ],
              [
                "vest",
                "Сигнальный жилет",
                "Система проверяет наличие жилета у обнаруженного человека.",
              ],
            ] as const
          ).map(([key, title, sub]) => (
            <label
              className={`p-rule-choice ${!supports(info, key) ? "unavailable" : ""}`}
              key={key}
            >
              <div>
                <strong>{title}</strong>
                <small>
                  {supports(info, key)
                    ? sub
                    : "Эта проверка недоступна в текущей модели."}
                </small>
              </div>
              <input
                type="checkbox"
                checked={draft[key] && supports(info, key)}
                disabled={!supports(info, key)}
                onChange={(e) =>
                  setDraft({ ...draft, [key]: e.target.checked })
                }
              />
            </label>
          ))}
          <div className="p-inform">
            <Check size={17} />
            <p>
              Кратковременная ошибка не создаёт инцидент. Подтверждённое событие
              сохраняется со снимком и временем.
            </p>
          </div>
        </>
      )}
      {!info?.model_ready && (
        <div className="p-inline-error">
          Модуль распознавания пока недоступен. Проверьте его состояние в
          разделе «Система».
        </div>
      )}
      {error && (
        <div className="p-inline-error" role="alert">
          {error}
        </div>
      )}
    </Dialog>
  );
}
