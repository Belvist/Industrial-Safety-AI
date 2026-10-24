import { useEffect, useState } from "react";
import {
  ArrowLeft,
  Radar,
  ArrowRight,
  AlertTriangle,
  Check,
  Eye,
  EyeOff,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  ScanLine,
  Settings2,
  Square,
  Users,
  Video,
  X,
} from "lucide-react";
import RadarInspector from "./RadarInspector";
import CameraViewport from "../components/CameraViewport";
import type { Camera, Incident, Result } from "../types";
import { date, isActive, labels, percentage } from "../types";
import { observation, reviewNames, ruleTitle, zonesOf } from "./model";

export default function Monitor({
  camera,
  result,
  online,
  busy,
  incidents,
  onBack,
  onConnect,
  onStop,
  onSettings,
  onZones,
  onIncident,
  onHistory,
}: {
  camera: Camera;
  result?: Result | null;
  online: boolean;
  busy: boolean;
  incidents: Incident[];
  onBack: () => void;
  onConnect: () => void;
  onStop: () => void;
  onSettings: () => void;
  onZones: () => void;
  onIncident: (incident: Incident) => void;
  onHistory: () => void;
}) {
  const [inspector, setInspector] = useState<
      "people" | "events" | "radar" | null
    >(null),
    [focus, setFocus] = useState(false),
    [boxes, setBoxes] = useState(true),
    [raw, setRaw] = useState(false);
  const state = observation(camera, result, online);
  const fresh =
    online &&
    camera.status === "running" &&
    !!result &&
    (result.data_age_ms ?? 0) <= 5000;
  const active = isActive(camera),
    pending = ["starting", "stopping", "uploading"].includes(camera.status);
  const events = incidents.filter((i) => i.camera_id === camera.id);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        document.querySelector('[role="dialog"]') ||
        (e.target instanceof HTMLElement &&
          e.target.matches("input,textarea,select"))
      )
        return;
      if (e.key === "Escape") {
        setFocus(false);
        setInspector(null);
      }
      if (e.key.toLowerCase() === "f") {
        e.preventDefault();
        setFocus((v) => !v);
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, []);
  useEffect(() => {
    setInspector(null);
    setFocus(false);
  }, [camera.id]);
  const qualityReasons = [
    ...new Set(
      Object.values(result?.observation_quality?.tracks ?? {}).flatMap(
        (t) => t.reasons,
      ),
    ),
  ];
  const qualityText: Record<string, string> = {
    person_too_small: "человек слишком мал в кадре",
    blurred_frame: "изображение размыто",
    dark_frame: "недостаточно света",
  };
  const ppeEnabled =
    camera.helmet || camera.vest || camera.mask || camera.smoking;
  const ppeUnknown =
    ppeEnabled &&
    fresh &&
    (result?.tracks.length ?? 0) > 0 &&
    (qualityReasons.length > 0 ||
      result?.ppe?.state !== "current" ||
      result.tracks.some((t) =>
        [
          camera.helmet && "helmet",
          camera.vest && "vest",
          camera.mask && "mask",
        ].some((k) => k && t.ppe[k] === "unknown"),
      ));
  return (
    <section className={`p-monitor ${focus ? "p-focus-mode" : ""}`}>
      <header className="p-monitor-heading">
        <div>
          <button className="p-link p-back" onClick={onBack}>
            <ArrowLeft size={16} />
            Все камеры
          </button>
          <h1>{camera.name}</h1>
        </div>
        <div className="p-monitor-actions">
          {active ? (
            <button
              className="p-button"
              disabled={busy || pending}
              onClick={onStop}
            >
              {pending ? (
                <Loader2 className="spin" size={15} />
              ) : (
                <Square size={14} />
              )}
              Остановить
            </button>
          ) : (
            <button
              className="p-button p-primary"
              disabled={busy || !online}
              onClick={onConnect}
            >
              <Play size={15} />
              {camera.kind ? "Подключить снова" : "Подключить источник"}
            </button>
          )}
          <button
            className="p-icon"
            title="Настройки камеры"
            aria-label="Настройки камеры"
            onClick={onSettings}
          >
            <Settings2 size={19} />
          </button>
          <button
            className="p-icon"
            title={focus ? "Вернуться · Esc" : "Режим наблюдения · F"}
            aria-label={focus ? "Выйти из режима наблюдения" : "На весь экран"}
            onClick={() => setFocus(!focus)}
          >
            {focus ? <Minimize2 size={19} /> : <Maximize2 size={19} />}
          </button>
        </div>
      </header>
      <div className={`p-observation-status ${state.tone}`} role="status">
        <i />
        <div>
          <strong>{state.title}</strong>
          <span>{state.description}</span>
        </div>
        {state.tone === "danger" && (
          <button className="p-link" onClick={() => setInspector("events")}>
            Посмотреть события
            <ArrowRight size={15} />
          </button>
        )}
        {camera.status === "ended" && (
          <button className="p-link" onClick={onHistory}>
            Журнал камеры
            <ArrowRight size={15} />
          </button>
        )}
      </div>
      {ppeUnknown && (
        <div className="p-inform p-warning" role="status">
          <AlertTriangle size={18} />
          <p>
            <strong>СИЗ: недостаточно достоверных данных.</strong>{" "}
            {qualityReasons.length
              ? qualityReasons.map((r) => qualityText[r] ?? r).join("; ")
              : result?.ppe?.state === "unavailable"
                ? "Модель СИЗ недоступна. Контроль движения продолжается."
                : result?.ppe?.state === "stale"
                  ? "Ответ СИЗ устарел. Старый статус не подтверждает новое нарушение."
                  : "Ожидаем распознавание каски и жилета. Отсутствие результата не означает нарушение."}
          </p>
        </div>
      )}
      <div
        className={`p-observation-layout ${inspector ? "with-inspector" : ""}`}
      >
        <div className="p-observation-video">
          <div className="p-video-surface">
            {result ? (
              <CameraViewport
                camera={camera}
                result={result}
                editing={false}
                points={[]}
                onPoint={() => {}}
                showBoxes={boxes}
                setShowBoxes={setBoxes}
                showRaw={raw}
                setShowRaw={setRaw}
                dataStale={!fresh}
                minimal
              />
            ) : (
              <div className="p-video-waiting">
                {active ? (
                  <Loader2 className="spin" size={32} />
                ) : (
                  <Video size={38} />
                )}
                <h2>
                  {pending || active
                    ? "Подключаем источник"
                    : camera.status === "error"
                      ? "Нет изображения"
                      : "Камера готова к подключению"}
                </h2>
                <p>
                  {active
                    ? "Дождитесь первого обработанного кадра."
                    : "Выберите веб-камеру, сетевой поток или видеозапись."}
                </p>
                {!active && (
                  <button
                    className="p-button p-light"
                    disabled={busy || !online}
                    onClick={onConnect}
                  >
                    <Play size={15} />
                    Подключить источник
                  </button>
                )}
              </div>
            )}
          </div>
          <div className="p-monitor-tools">
            <div>
              <button
                className={boxes ? "selected" : ""}
                aria-pressed={boxes}
                onClick={() => setBoxes(!boxes)}
              >
                {boxes ? <Eye size={16} /> : <EyeOff size={16} />}Разметка
              </button>
              <button
                className={inspector === "people" ? "selected" : ""}
                aria-pressed={inspector === "people"}
                onClick={() =>
                  setInspector(inspector === "people" ? null : "people")
                }
              >
                <Users size={16} />
                Люди <span>{fresh ? (result?.tracks.length ?? 0) : "—"}</span>
              </button>
              <button
                className={inspector === "radar" ? "selected" : ""}
                aria-pressed={inspector === "radar"}
                onClick={() =>
                  setInspector(inspector === "radar" ? null : "radar")
                }
              >
                <Radar size={16} />
                Радар
              </button>
              <button
                className={inspector === "events" ? "selected" : ""}
                aria-pressed={inspector === "events"}
                onClick={() =>
                  setInspector(inspector === "events" ? null : "events")
                }
              >
                <AlertTriangle size={16} />
                События
              </button>
            </div>
            <button onClick={onZones} disabled={!result || !active || !fresh}>
              <ScanLine size={16} />
              {zonesOf(camera).length ? "Опасные зоны" : "Обозначить зону"}
            </button>
          </div>
          <div className="p-camera-context">
            <span>
              {camera.kind === "video"
                ? "Анализ видеозаписи"
                : camera.kind === "rtsp"
                  ? "Сетевая камера"
                  : "Камера этого компьютера"}
            </span>
            <span>
              {[
                camera.helmet && "Каска",
                camera.vest && "Жилет",
                zonesOf(camera).some((z) => z.enabled) && "Зоны",
              ]
                .filter(Boolean)
                .join(" · ") || "Проверки не включены"}
            </span>
            {focus && <span>Esc — вернуться</span>}
          </div>
        </div>
        {inspector && (
          <aside className="p-inspector">
            <header>
              <h2>
                {inspector === "people"
                  ? "Люди в кадре"
                  : inspector === "radar"
                    ? "Safety Radar"
                    : "События камеры"}
              </h2>
              <button
                className="p-icon"
                onClick={() => setInspector(null)}
                aria-label="Закрыть панель"
              >
                <X size={18} />
              </button>
            </header>
            {inspector === "radar" ? (
              <RadarInspector radar={fresh ? result?.radar : null} />
            ) : inspector === "people" ? (
              fresh && result?.tracks.length ? (
                <div className="p-worker-list">
                  {result.tracks.map((track) => (
                    <div key={track.id}>
                      <strong>Сотрудник #{track.id}</strong>
                      <dl>
                        {(
                          [
                            ["helmet", "Каска"],
                            ["vest", "Жилет"],
                          ] as const
                        ).map(([key, title]) => (
                          <div key={key}>
                            <dt>{title}</dt>
                            <dd className={track.ppe[key] ?? "unknown"}>
                              {track.ppe[key] === "present" ? (
                                <>
                                  <Check size={13} />
                                  Обнаружена
                                </>
                              ) : track.ppe[key] === "missing" ? (
                                <>
                                  <AlertTriangle size={13} />
                                  Не обнаружена
                                </>
                              ) : (
                                "Не определено"
                              )}
                            </dd>
                          </div>
                        ))}
                      </dl>
                      {(track.ppe_age_seconds ?? 0) > 0 && (
                        <small>
                          Последнее наблюдение СИЗ:{" "}
                          {(track.ppe_age_seconds ?? 0).toFixed(1)} с назад
                        </small>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-inspector-empty">
                  <Users size={26} />
                  <p>
                    {fresh
                      ? "Люди пока не обнаружены."
                      : "Дождитесь свежих кадров."}
                  </p>
                  <small>
                    Идентификатор относится к треку в текущем наблюдении, а не к
                    личности работника.
                  </small>
                </div>
              )
            ) : (
              <>
                {events.length ? (
                  <div className="p-inspector-events">
                    {events.map((incident) => (
                      <button
                        key={incident.id}
                        onClick={() => onIncident(incident)}
                      >
                        <span className={`p-review-dot ${incident.feedback}`} />
                        <div>
                          <strong>{ruleTitle(incident.type)}</strong>
                          <small>
                            {date(incident.started_at)} · #{incident.track_id}
                          </small>
                          <span>
                            {reviewNames[incident.feedback] ?? "На проверке"}
                          </span>
                        </div>
                        <ArrowRight size={15} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="p-inspector-empty">
                    <AlertTriangle size={25} />
                    <p>Подтверждённых событий пока нет.</p>
                    <small>
                      Возможное нарушение сначала проходит проверку во времени.
                    </small>
                  </div>
                )}
                <button
                  className="p-link p-inspector-footer"
                  onClick={onHistory}
                >
                  Весь журнал камеры
                  <ArrowRight size={15} />
                </button>
              </>
            )}
            {inspector === "people" && (
              <details className="p-advanced p-model-objects">
                <summary>Объекты и уверенность модели</summary>
                <label className="p-rule-choice">
                  <strong>Все детекции на видео</strong>
                  <input
                    type="checkbox"
                    checked={raw}
                    onChange={(e) => {
                      setRaw(e.target.checked);
                      if (e.target.checked) setBoxes(true);
                    }}
                  />
                </label>
                {fresh &&
                  result?.detections.map((d, index) => (
                    <div className="p-model-object" key={index}>
                      <span>{labels[d.class_name] ?? d.class_name}</span>
                      <span>
                        {percentage(d.confidence)}
                        {d.weak ? " · слабое распознавание" : ""}
                      </span>
                    </div>
                  ))}
                <p>
                  Дополнительные объекты помогают проверить распознавание и сами
                  по себе не подтверждают нарушение.
                </p>
              </details>
            )}
          </aside>
        )}
      </div>
    </section>
  );
}
