import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  ShieldCheck,
  Video,
  LayoutDashboard,
  BellRing,
  Settings2,
  Play,
  Square,
  Upload,
  AlertTriangle,
  Check,
  X,
  Plus,
  ScanLine,
  ExternalLink,
  Activity,
  ChevronRight,
  Loader2,
  ArrowRight,
  Radio,
  SlidersHorizontal,
} from "lucide-react";
import { api, json, saveOperatorToken, stopPath } from "./api";
import { useSafety } from "./hooks/useSafety";
import { useBrowserCamera } from "./hooks/useBrowserCamera";
import CameraViewport from "./components/CameraViewport";
import SafetyRadarPanel from "./components/SafetyRadarPanel";
import CameraTile from "./components/CameraTile";
import IncidentDialog from "./components/IncidentDialog";
import PhoneConnect from "./components/PhoneConnect";
import ConnectPage from "./ConnectPage";
import type { Camera, Config, Incident, Page, Point, Result } from "./types";
import { configOf, date, isActive, labels, percentage, typeLabels } from "./types";
import { accelerationHint, inferenceLabel, inferenceSub } from "./inferenceLabel";
import {
  cameraAlert,
  globalAlert,
  screenStatus,
  sortDetections,
  sortTracks,
  visibleDetections,
} from "./monitor";

const pages: Record<Page, string> = {
  monitor: "Мониторинг",
  dashboard: "Обзор",
  incidents: "Инциденты",
  cameras: "Камеры и правила",
};
export default function App() {
  const safety = useSafety();
  const [page, setPage] = useState<Page>("monitor");
  const [selected, setSelected] = useState("");
  const [source, setSource] = useState("browser");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [device, setDevice] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [operatorKey, setOperatorKey] = useState("");
  const [form, setForm] = useState<Config | null>(null);
  const [formCamera, setFormCamera] = useState("");
  const [editing, setEditing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showRaw, setShowRaw] = useState(false);
  const [detail, setDetail] = useState<Incident | null>(null);
  const [grid, setGrid] = useState(false);
  const browser = useBrowserCamera(setError);
  const camera = safety.cameras.find((c) => c.id === selected);
  const active = isActive(camera);
  const busy =
    safety.busy ||
    ["permission", "connecting", "stopping"].includes(browser.state.phase);
  const result: Result | null | undefined =
    browser.owner === selected &&
    browser.result?.session_id === camera?.session_id
      ? browser.result
      : camera?.latest;
  const shownDetections = result
    ? sortDetections(visibleDetections(result.detections))
    : [];
  const shownTracks = result ? sortTracks(result.tracks, result.rules) : [];
  const screen = screenStatus(
    active,
    result?.tracks ?? [],
    result?.rules ?? [],
  );
  const otherAlert =
    page === "monitor" ? globalAlert(safety.cameras, selected) : null;
  const runningCount = safety.cameras.filter((c) =>
    ["running", "stale"].includes(c.status),
  ).length;
  useEffect(() => {
    if (!selected && safety.cameras[0]) setSelected(safety.cameras[0].id);
  }, [safety.cameras, selected]);
  // Enumerate video devices on mount so the operator picks a camera BEFORE
  // the first start (previously the list appeared only after starting).
  useEffect(() => {
    let alive = true;
    navigator.mediaDevices
      ?.enumerateDevices()
      .then((all) => {
        if (!alive) return;
        const video = all.filter((d) => d.kind === "videoinput");
        setDevices(video);
        if (video.length > 0) setDevice((d) => d || video[0].deviceId);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);
  useEffect(() => {
    if (camera && formCamera !== camera.id) {
      setForm(configOf(camera));
      setFormCamera(camera.id);
      setPoints(camera.zone);
      setEditing(false);
    }
  }, [camera, formCamera]);
  useEffect(() => {
    setDetail((old) =>
      old ? safety.incidents.find((i) => i.id === old.id) || old : null,
    );
  }, [safety.incidents]);
  async function act(work: () => Promise<void>) {
    setError("");
    await safety.act(work);
  }
  async function start() {
    await act(async () => {
      if (!camera) return;
      if (source === "browser") {
        await browser.start(camera.id, device, safety.info?.target_fps || 5);
        setDevices(
          (await navigator.mediaDevices.enumerateDevices()).filter(
            (d) => d.kind === "videoinput",
          ),
        );
      } else if (source === "video") {
        if (!file) throw new Error("Выберите видеофайл.");
        if (file.size > 200 * 1024 * 1024) throw new Error("Максимум 200 МБ.");
        await api(`/cameras/${camera.id}/video`, {
          method: "POST",
          headers: { "Content-Type": "video/mp4" },
          body: file,
        });
      } else
        await api(`/cameras/${camera.id}/start`, json({ kind: "rtsp", url }));
    });
  }
  async function stop() {
    if (
      ["permission", "connecting"].includes(browser.state.phase) &&
      browser.state.cameraId === selected
    ) {
      try {
        await browser.stop();
        await safety.refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
      return;
    }
    await act(async () => {
      if (browser.owner === selected || browser.state.cameraId === selected)
        await browser.stop();
      else if (camera)
        await api(stopPath(camera.id, camera.session_id), { method: "POST" });
    });
  }
  async function enumerate() {
    await act(async () => {
      if (!navigator.mediaDevices)
        throw new Error("Для доступа к камере откройте localhost.");
      const media = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      try {
        setDevices(
          (await navigator.mediaDevices.enumerateDevices()).filter(
            (d) => d.kind === "videoinput",
          ),
        );
      } finally {
        media.getTracks().forEach((t) => t.stop());
      }
    });
  }
  async function save(config: Config) {
    await act(async () => {
      const saved = await api<Camera>(
        `/cameras/${selected}`,
        json(config, "PUT"),
      );
      setForm(configOf(saved));
      setPoints(saved.zone);
      setEditing(false);
    });
  }
  function editZone() {
    setPage("monitor");
    setPoints(camera?.zone || []);
    setEditing(true);
  }
  const errorMessage = error || safety.command.error;
  const canStop =
    active ||
    (browser.state.cameraId === selected &&
      browser.state.phase !== "idle" &&
      browser.state.phase !== "failed");
  const groups = [
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "mask",
    "no_mask",
    "smoking",
  ].map((name) => ({
    name,
    items: result?.detections.filter((d) => d.class_name === name) || [],
  }));

  return (
    <div className="app-shell">
      {location.pathname === "/connect" ? (
        <ConnectPage />
      ) : (
      <>
      <header className="site-header">
        <a
          className="brand"
          href="#monitor"
          onClick={(e) => {
            e.preventDefault();
            setPage("monitor");
          }}
        >
          <span className="brand-mark">
            <i />
            <i />
          </span>
          <span>
            safety<span className="brand-ai">ai</span>
          </span>
        </a>
        <nav aria-label="Основная навигация">
          {(Object.keys(pages) as Page[]).map((p) => (
            <button
              key={p}
              className={page === p ? "active" : ""}
              onClick={() => setPage(p)}
            >
              {p === "monitor" ? (
                <Video size={15} />
              ) : p === "dashboard" ? (
                <LayoutDashboard size={15} />
              ) : p === "incidents" ? (
                <BellRing size={15} />
              ) : (
                <Settings2 size={15} />
              )}
              <span>{pages[p]}</span>
              {p === "incidents" && !!safety.info?.stats.active && (
                <b>{safety.info.stats.active}</b>
              )}
            </button>
          ))}
        </nav>
        <div className="header-tools">
          <span
            className={
              "connection " + (safety.connection === "online" ? "online" : "")
            }
          >
            <i />
            {safety.connection === "online"
              ? "Локально"
              : safety.connection === "connecting"
                ? "Подключение"
                : "Нет связи"}
          </span>
          <button
            className="header-button"
            onClick={() => {
              setPage("monitor");
              document
                .getElementById("source-controls")
                ?.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
          >
            Подключить камеру
            <ArrowUpRight size={15} />
          </button>
        </div>
      </header>
      <main>
        <section className="page-heading">
          <div>
            <span className="eyebrow">
              INDUSTRIAL SAFETY AI / {pages[page].toUpperCase()}
            </span>
            <h1>
              {page === "monitor" ? (
                <>
                  Безопасность.
                  <br />
                  <span>В поле зрения.</span>
                </>
              ) : page === "dashboard" ? (
                <>
                  Вся картина.
                  <br />
                  <span>В одном месте.</span>
                </>
              ) : page === "incidents" ? (
                <>
                  Каждое событие.
                  <br />
                  <span>С доказательством.</span>
                </>
              ) : (
                <>
                  Ваша камера.
                  <br />
                  <span>Ваши правила.</span>
                </>
              )}
            </h1>
          </div>
          <div className="page-intro">
            <p>
              {page === "monitor"
                ? "Люди, средства защиты и опасные зоны. Наблюдайте за происходящим и реагируйте на подтверждённые нарушения."
                : page === "dashboard"
                  ? "Контролируйте состояние камер, активные нарушения и историю безопасности на вашем устройстве."
                  : page === "incidents"
                    ? "Проверяйте снимки, отслеживайте статусы и отмечайте ложные срабатывания."
                    : "Выберите источник видео. Настройте правила и время подтверждения для каждой рабочей зоны."}
            </p>
            <span
              className={
                "model-status " + (safety.info?.model_ready ? "ready" : "")
              }
            >
              <i />
              {safety.info?.model_ready
                ? "Детектор готов к работе"
                : "Детектор не готов"}
              <span>ONNX / CPU</span>
            </span>
          </div>
        </section>
        {safety.accessDenied ? (
          <div className="notice danger">
            <AlertTriangle size={18} />
            <span>
              Доступ к серверу закрыт (401). Камеры и инциденты защищены.
              <small>
                На компьютере оператора ключ находится в файле data/api_token. Вставьте его ниже.
                Для независимого тестирования на другом компьютере запускайте собственную копию приложения.
              </small>
              <form
                className="inline"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!operatorKey.trim()) return;
                  saveOperatorToken(operatorKey);
                  location.reload();
                }}
              >
                <input
                  type="password"
                  autoComplete="off"
                  aria-label="Ключ оператора"
                  placeholder="Ключ оператора"
                  value={operatorKey}
                  onChange={(event) => setOperatorKey(event.target.value)}
                />
                <button className="button primary" type="submit" disabled={!operatorKey.trim()}>
                  Войти
                </button>
              </form>
            </span>
          </div>
        ) : (
          safety.connection === "offline" && (
            <div className="notice danger">
              <AlertTriangle size={18} />
              Нет связи с backend. Запустите сервер и дождитесь загрузки модели.
            </div>
          )
        )}
        {safety.info?.model_error && (
          <div className="notice amber">
            <AlertTriangle size={18} />
            <span>
              {safety.info.model_error}
              <small>После установки моделей перезапустите приложение.</small>
            </span>
          </div>
        )}
        {errorMessage && (
          <div className="notice danger" role="alert">
            <AlertTriangle size={18} />
            <span>{errorMessage}</span>
            <button
              className="icon-button"
              aria-label="Закрыть ошибку"
              onClick={() => {
                setError("");
                safety.clearError();
              }}
            >
              <X size={18} />
            </button>
          </div>
        )}
        <section className="stats-grid">
          {[
            [
              Video,
              "Камеры в эфире",
              safety.cameras.filter((c) => c.status === "running").length,
              `из ${safety.cameras.length} настроенных`,
            ],
            [
              AlertTriangle,
              "Активные нарушения",
              safety.info?.stats.active ?? 0,
              "требуют внимания",
            ],
            [
              Activity,
              "События сегодня",
              safety.info?.stats.today ?? 0,
              "по времени UTC",
            ],
            [
              ShieldCheck,
              "Всего инцидентов",
              safety.info?.stats.total ?? 0,
              "сохранено в истории",
            ],
          ].map(([Icon, title, value, sub], i) => {
            const Glyph = Icon as typeof Video;
            return (
              <div className="stat" key={String(title)}>
                <span className="stat-number">0{i + 1}</span>
                <div>
                  <span className="stat-title">{String(title)}</span>
                  <strong>{String(value)}</strong>
                  <small>{String(sub)}</small>
                </div>
                <Glyph size={19} />
              </div>
            );
          })}
        </section>
        {(page === "monitor" || page === "dashboard") && (
          <section className="monitor-layout">
            {otherAlert && (
              <div
                className={"alert-banner " + (otherAlert.level === "active" ? "firing" : "checking")}
                role="alert"
              >
                <strong>
                  {otherAlert.level === "active" ? "НАРУШЕНИЕ" : "ПРОВЕРКА"} на{" "}
                  {otherAlert.cameraName}: {labels[otherAlert.type] || otherAlert.type}
                </strong>
                <span>
                  Работник #{otherAlert.trackId} · {otherAlert.elapsed.toFixed(1)} сек.
                </span>
                <button
                  className="text-button"
                  onClick={() => {
                    setSelected(otherAlert.cameraId);
                    setGrid(false);
                  }}
                >
                  Смотреть
                  <ArrowRight size={15} />
                </button>
              </div>
            )}
            <div className="monitor-main">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">01 / LIVE MONITORING</span>
                  <h2>Рабочая зона</h2>
                  <small className="inference-badge">
                    {inferenceLabel(safety.info)}
                    {inferenceSub(safety.info) && (
                      <>
                        <br />
                        {inferenceSub(safety.info)}
                      </>
                    )}
                    {accelerationHint(safety.info) && (
                      <>
                        <br />
                        {accelerationHint(safety.info)}
                      </>
                    )}
                  </small>
                </div>
                <div className="inline">
                  <button
                    className="icon-button outlined"
                    aria-label="Сетка камер"
                    title="Все камеры сразу"
                    onClick={() => setGrid((g) => !g)}
                  >
                    <LayoutDashboard size={17} />
                  </button>
                  <select
                    aria-label="Выбор камеры"
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                  >
                    {safety.cameras.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                  <button
                    className="icon-button outlined"
                    aria-label="Настройки камеры"
                    onClick={() => setPage("cameras")}
                  >
                    <SlidersHorizontal size={17} />
                  </button>
                </div>
              </div>
              <div className="camera-panel">
                {grid ? (
                  <div className="camera-grid" role="group" aria-label="Все камеры">
                    {safety.cameras.map((c) => (
                      <CameraTile
                        key={c.id}
                        camera={c}
                        selected={c.id === selected}
                        onSelect={() => setSelected(c.id)}
                      />
                    ))}
                    {!safety.cameras.length && (
                      <div className="compact-empty">
                        Добавьте камеру на вкладке «Камеры и правила».
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                {screen.kind === "alert" && (
                  <div
                    className={
                      "alert-banner " + (screen.level === "active" ? "firing" : "checking")
                    }
                    role="alert"
                  >
                    <strong>
                      {screen.level === "active" ? "НАРУШЕНИЕ" : "ПРОВЕРКА"}:{" "}
                      {typeLabels[screen.type] || labels[screen.type] || screen.type}
                    </strong>
                    <span>
                      Работник #{screen.trackId} · {screen.elapsed.toFixed(1)} /{" "}
                      {screen.confirm} сек.
                    </span>
                  </div>
                )}
                {screen.kind === "calm" && (
                  <div className="calm-chip" role="status">
                    Под контролем · {screen.workers}{" "}
                    {screen.workers === 1 ? "человек" : "чел."}
                  </div>
                )}
                <SafetyRadarPanel
                  radar={result?.radar}
                  workers={result?.tracks.length ?? 0}
                />
                <CameraViewport
                  camera={camera}
                  result={result}
                  editing={editing}
                  points={points}
                  onPoint={(p) => setPoints((old) => [...old, p])}
                  showBoxes={showBoxes}
                  setShowBoxes={setShowBoxes}
                  showRaw={showRaw}
                  setShowRaw={setShowRaw}
                />
                  </>
                )}
              </div>
              {editing ? (
                <div className="zone-actions">
                  <span>
                    Нормализованные координаты · {points.length} точек
                  </span>
                  <div className="inline">
                    <button
                      className="button small"
                      disabled={!points.length}
                      onClick={() => setPoints(points.slice(0, -1))}
                    >
                      Убрать точку
                    </button>
                    <button
                      className="button small"
                      onClick={() => setEditing(false)}
                    >
                      Отмена
                    </button>
                    <button
                      className="button primary small"
                      disabled={points.length < 3 || busy}
                      onClick={() =>
                        form &&
                        void save({ ...form, zone: points, zone_enabled: true })
                      }
                    >
                      <Check size={14} />
                      Сохранить зону
                    </button>
                  </div>
                </div>
              ) : (
                <div className="detection-summary">
                  <span className="summary-label">В кадре</span>
                  {groups.map((g) => (
                    <div
                      className={
                        "detected-chip " +
                        (g.name.startsWith("no_") ? "warning" : "")
                      }
                      key={g.name}
                    >
                      <i />
                      <span>{labels[g.name]}</span>
                      <b>{g.items.length}</b>
                      {g.items.length > 0 && (
                        <small>
                          {percentage(
                            Math.max(...g.items.map((d) => d.confidence)),
                          )}
                        </small>
                      )}
                    </div>
                  ))}
                </div>
              )}
              <div className="source-controls" id="source-controls">
                <div className="source-tabs">
                  {[
                    ["browser", "Веб-камера"],
                    ["video", "Видеофайл"],
                    ["rtsp", "RTSP / IP"],
                  ].map(([key, label]) => (
                    <button
                      disabled={active || busy}
                      className={source === key ? "chosen" : ""}
                      key={key}
                      onClick={() => setSource(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="source-row">
                  {source === "browser" ? (
                    <>
                      <select
                        aria-label="Устройство камеры"
                        disabled={active || busy}
                        value={device}
                        onChange={(e) => setDevice(e.target.value)}
                      >
                        <option value="">Камера по умолчанию</option>
                        {devices.map((d, i) => (
                          <option key={d.deviceId} value={d.deviceId}>
                            {d.label || `Камера ${i + 1}`}
                          </option>
                        ))}
                      </select>
                      <button
                        className="button"
                        disabled={active || busy}
                        onClick={() => void enumerate()}
                      >
                        Найти камеры
                      </button>
                    </>
                  ) : source === "video" ? (
                    <label className="file-picker">
                      <Upload size={17} />
                      <span>{file?.name || "Выбрать видео · до 200 МБ"}</span>
                      <input
                        aria-label="Выбрать видео"
                        type="file"
                        accept="video/mp4,video/webm,video/quicktime"
                        disabled={active || busy}
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                      />
                    </label>
                  ) : (
                    <input
                      type="password"
                      autoComplete="off"
                      aria-label="Адрес видеопотока"
                      placeholder="rtsp://192.168.1.10:554/stream"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      disabled={active || busy}
                    />
                  )}
                  {canStop ? (
                    <button
                      className="button stop"
                      disabled={
                        browser.state.phase === "stopping" ||
                        camera?.status === "stopping" ||
                        (busy &&
                          !["permission", "connecting"].includes(
                            browser.state.phase,
                          ))
                      }
                      onClick={() => void stop()}
                    >
                      <Square size={14} />
                      Остановить
                    </button>
                  ) : (
                    <button
                      className="button primary"
                      disabled={
                        busy ||
                        !selected ||
                        !safety.info?.model_ready ||
                        safety.connection !== "online"
                      }
                      onClick={() => void start()}
                    >
                      {busy ? (
                        <Loader2 className="spin" size={15} />
                      ) : (
                        <Play size={15} />
                      )}
                      Запустить мониторинг
                    </button>
                  )}
                </div>
                <p className="source-note">
                  <ShieldCheck size={12} />
                  Обработка на вашем устройстве. Камера работает, пока вкладка
                  открыта.
                </p>
              </div>
              {camera?.error && (
                <div className="notice danger">
                  <AlertTriangle size={16} />
                  {camera.error}
                </div>
              )}
              <section className="objects-panel">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">РЕЗУЛЬТАТЫ ТЕКУЩЕГО КАДРА</span>
                    <h2>Что определила модель</h2>
                  </div>
                  <span className="count">
                    {shownDetections.length || 0} объектов
                  </span>
                </div>
                {shownDetections.length ? (
                  <div className="object-list">
                    {shownDetections.map((d, i) => (
                      <div
                        className={
                          "object-card " +
                          (d.class_name.startsWith("no_") ? "negative" : "")
                        }
                        key={i}
                      >
                        <span className="object-index">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <div>
                          <strong>
                            {labels[d.class_name] || d.class_name}
                          </strong>
                          <span>Уверенность распознавания</span>
                        </div>
                        <div className="confidence-meter">
                          <strong>{percentage(d.confidence)}</strong>
                          <div>
                            <i style={{ width: percentage(d.confidence) }} />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="compact-empty">
                    Распознанные объекты появятся после запуска камеры.
                  </div>
                )}
                <p className="confidence-note">
                  Проценты отражают уверенность модели в классе объекта. Это не
                  оценка безопасности человека.
                </p>
              </section>
            </div>
            <aside className="monitor-side">
              <section className="rules-panel">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">02 / SAFETY RULES</span>
                    <h2>Контроль безопасности</h2>
                  </div>
                  <ShieldCheck size={20} />
                </div>
                {[
                  ["helmet", "Защитная каска", "Явная детекция «без каски»"],
                  ["vest", "Сигнальный жилет", "Явная детекция «без жилета»"],
                  [
                    "zone_enabled",
                    "Опасная зона",
                    camera?.zone.length
                      ? "Контроль точки под ногами"
                      : "Обозначьте границы на кадре",
                  ],
                ].map(([key, title, subtitle]) => (
                  <div className="rule" key={key}>
                    <div>
                      <strong>{title}</strong>
                      <small>{subtitle}</small>
                    </div>
                    <button
                      className={
                        "toggle " + (camera?.[key as keyof Config] ? "on" : "")
                      }
                      aria-label={title}
                      aria-pressed={Boolean(camera?.[key as keyof Config])}
                      disabled={
                        busy ||
                        !form ||
                        (key === "zone_enabled" && !camera?.zone.length)
                      }
                      onClick={() =>
                        form &&
                        void save({
                          ...form,
                          [key]: !camera?.[key as keyof Config],
                        })
                      }
                    >
                      <span />
                    </button>
                  </div>
                ))}
                <button
                  className="zone-button"
                  onClick={editZone}
                  disabled={!camera || busy}
                >
                  <ScanLine size={17} />
                  Настроить опасную зону
                  <ArrowUpRight size={16} />
                </button>
                <div className="confirmation">
                  <span>Подтверждение нарушения</span>
                  <strong>
                    {camera?.confirm_seconds || 2}
                    <small> сек.</small>
                  </strong>
                  <p>Кратковременная ошибка не создаёт инцидент.</p>
                </div>
              </section>
              <section className="workers-panel">
                <div className="section-heading">
                  <h2>Люди в кадре</h2>
                  <span className="count">{result?.tracks.length || 0}</span>
                </div>
                {shownTracks.length ? (
                  shownTracks.map((t) => (
                    <div className="worker-card" key={t.id}>
                      <div className="worker-top">
                        <span className="worker-avatar">#{t.id}</span>
                        <strong>Worker #{t.id}</strong>
                        <b>{percentage(t.confidence)}</b>
                      </div>
                      <div className="worker-ppe">
                        {[
                          ["helmet", "Каска"],
                          ["vest", "Жилет"],
                          ["mask", "Респиратор"],
                        ].map(([p, title]) => (
                          <span key={p} className={t.ppe[p] || "unknown"}>
                            {title}
                            <strong>
                              {t.ppe[p] === "present"
                                ? "Есть"
                                : t.ppe[p] === "missing"
                                  ? "Нет"
                                  : "Не определено"}
                              {t.ppe[p] !== "unknown" &&
                                !!t.ppe_confidence?.[p] &&
                                ` · ${percentage(t.ppe_confidence[p])}`}
                            </strong>
                          </span>
                        ))}
                      </div>
                      {(() => {
                        const inZone = (result?.rules || []).some(
                          (r) => r.track_id === t.id && r.type === "danger_zone",
                        );
                        return (
                          <div className="worker-status">
                            <span className={inZone ? "in-zone" : ""}>
                              {inZone ? "В опасной зоне" : "Вне зоны"}
                            </span>
                            <strong>
                              {t.violations.length
                                ? `Нарушение: ${t.violations.map((v) => labels[v] || v).join(", ")}`
                                : "Норма"}
                            </strong>
                          </div>
                        );
                      })()}
                      {(result?.rules ?? [])
                        .filter((r) => r.track_id === t.id)
                        .map((r) => (
                          <div
                            className={"rule-progress " + r.phase}
                            key={r.type}
                          >
                            <div>
                              <span>{labels[r.type]}</span>
                              <b>
                                {r.phase === "pending"
                                  ? `${r.elapsed_seconds.toFixed(1)} / ${r.confirm_seconds} сек.`
                                  : labels[r.phase]}
                              </b>
                            </div>
                            <i>
                              <span
                                style={{
                                  width: `${Math.min(100, (r.elapsed_seconds / r.confirm_seconds) * 100)}%`,
                                }}
                              />
                            </i>
                          </div>
                        ))}
                    </div>
                  ))
                ) : (
                  <div className="compact-empty">
                    Здесь появятся ID и результаты проверки СИЗ.
                  </div>
                )}
              </section>
              <section className="telegram-card">
                <Radio size={21} />
                <div>
                  <h3>Всегда на связи.</h3>
                  <p>
                    {safety.info?.telegram_ready
                      ? "Подтверждённые инциденты отправляются в Telegram вместе со снимком."
                      : "Подключите Telegram, чтобы получать подтверждённые нарушения со снимками."}
                  </p>
                  <span
                    className={
                      "status-badge " +
                      (safety.info?.telegram_ready ? "resolved" : "")
                    }
                  >
                    {safety.info?.telegram_ready
                      ? "Telegram подключён"
                      : "Настройка через .env"}
                  </span>
                </div>
              </section>
            </aside>
          </section>
        )}
        {page === "cameras" && (
          <section className="camera-layout">
            <div className="camera-list-panel">
              <div className="section-heading">
                <h2>Ваши камеры</h2>
                <div className="inline">
                  {runningCount > 0 && (
                    <button
                      className="button small"
                      disabled={busy}
                      onClick={() =>
                        void act(async () => {
                          for (const c of safety.cameras) {
                            try {
                              await api(`/cameras/${c.id}/stop`, {
                                method: "POST",
                              });
                            } catch {
                              // keep stopping the rest; refresh shows the outcome
                            }
                          }
                        })
                      }
                    >
                    <Square size={14} />
                    Стоп все ({runningCount})
                  </button>
                  )}
                  <button
                    className="button small"
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        const c = await api<Camera>(
                          "/cameras",
                          json({ name: `Камера ${safety.cameras.length + 1}` }),
                        );
                        setSelected(c.id);
                      })
                    }
                  >
                    <Plus size={14} />
                    Добавить
                  </button>
                </div>
              </div>
              {safety.cameras.map((c) => {
                const alert = cameraAlert(c);
                return (
                <button
                  className={
                    "camera-row " + (selected === c.id ? "selected" : "") +
                    (alert ? " alarming" : "")
                  }
                  key={c.id}
                  onClick={() => setSelected(c.id)}
                >
                  <Video size={20} />
                  <div>
                    <strong>{c.name}</strong>
                    <small>
                      {alert
                        ? `${labels[alert.type] || alert.type} · #${alert.trackId}`
                        : labels[c.status]}
                    </small>
                  </div>
                  <ChevronRight size={17} />
                </button>
                );
              })}
              {safety.info?.public_url && (
                <PhoneConnect url={safety.info.public_url} cameraId={selected} />
              )}
            </div>
            {form && (
              <section className="config-form">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">CONFIGURATION</span>
                    <h2>Настройки рабочей зоны</h2>
                  </div>
                  <Settings2 size={20} />
                </div>
                <label>
                  Название камеры
                  <input
                    value={form.name}
                    maxLength={80}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                  />
                </label>
                <div className="form-columns">
                  <label>
                    Подтверждение, сек.
                    <input
                      type="number"
                      min="0.2"
                      max="60"
                      step="0.1"
                      value={form.confirm_seconds}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          confirm_seconds: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <label>
                    Завершение, сек.
                    <input
                      type="number"
                      min="0.2"
                      max="60"
                      step="0.1"
                      value={form.clear_seconds}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          clear_seconds: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                </div>
                <p>
                  Подтверждение требует наблюдаемого нарушения. Завершение — его
                  отсутствия в течение указанного времени.
                </p>
                {[
                  ["helmet", "Контроль каски"],
                  ["vest", "Контроль жилета"],
                  ["mask", "Контроль респиратора"],
                  ["smoking", "Контроль курения"],
                  ["zone_enabled", "Контроль опасной зоны"],
                  ["running_enabled", "Контроль бега по цеху"],
                ].map(([key, title]) => {
                  // Rule needs at least one of its VIOLATION classes in the model.
                  // Unknown backend info preserves the legacy default
                  // (everything but smoking enabled).
                  const known = safety.info?.inference?.supported_classes;
                  const need: Record<string, string[]> = {
                    helmet: ["no_helmet"],
                    vest: ["no_vest"],
                    mask: ["no_mask"],
                    smoking: ["smoking"],
                    zone_enabled: ["person"],
                    running_enabled: ["person"],
                  };
                  const supported =
                    known === undefined
                      ? key !== "smoking"
                      : (need[key] ?? []).some((c) => known.includes(c));
                  return (
                    <label className="check-label" key={key}>
                      <input
                        type="checkbox"
                        checked={Boolean(form[key as keyof Config])}
                        disabled={!supported}
                        onChange={(e) =>
                          setForm({ ...form, [key]: e.target.checked })
                        }
                      />
                      {title}
                      {!supported && " — не поддерживается моделью"}
                    </label>
                  );
                })}
                <div className="inline">
                  <button className="button" onClick={editZone}>
                    <ScanLine size={16} />
                    Редактор зоны
                  </button>
                  <button
                    className="button"
                    onClick={() =>
                      setForm({ ...form, zone: [], zone_enabled: false })
                    }
                  >
                    Удалить зону
                  </button>
                </div>
                <div className="form-footer">
                  <span>Изменение правил прерывает текущие инциденты.</span>
                  <button
                    className="button primary"
                    disabled={busy || !form.name.trim()}
                    onClick={() => void save(form)}
                  >
                    <Check size={15} />
                    Сохранить
                  </button>
                </div>
              </section>
            )}
          </section>
        )}
        {page !== "cameras" && (
          <section className="incidents-panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  {page === "incidents" ? "EVENT HISTORY" : "03 / INCIDENTS"}
                </span>
                <h2>
                  {page === "incidents"
                    ? "Журнал инцидентов"
                    : "Последние события"}
                  <span className="count">{safety.total}</span>
                </h2>
              </div>
              <div className="inline">
                <select
                  aria-label="Фильтр статуса"
                  value={safety.filter}
                  onChange={(e) => safety.setFilter(e.target.value)}
                >
                  <option value="">Все статусы</option>
                  <option value="active">Активные</option>
                  <option value="resolved">Завершённые</option>
                  <option value="interrupted">Прерванные</option>
                </select>
                <select
                  aria-label="Фильтр типа нарушения"
                  value={safety.incidentType}
                  onChange={(e) => safety.setIncidentType(e.target.value)}
                >
                  <option value="">Все типы</option>
                  {[
                    "no_helmet",
                    "no_vest",
                    "no_mask",
                    "smoking",
                    "danger_zone",
                    "running",
                  ].map((kind) => (
                    <option key={kind} value={kind}>
                      {typeLabels[kind] || labels[kind] || kind}
                    </option>
                  ))}
                </select>
                <select
                  aria-label="Фильтр камеры"
                  value={safety.incidentCamera}
                  onChange={(e) => safety.setIncidentCamera(e.target.value)}
                >
                  <option value="">Все камеры</option>
                  {safety.cameras.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
                {page !== "incidents" && (
                  <button
                    className="text-button"
                    onClick={() => setPage("incidents")}
                  >
                    Весь журнал
                    <ArrowUpRight size={15} />
                  </button>
                )}
              </div>
            </div>
            {safety.incidents.length ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Событие</th>
                      <th>Камера / сотрудник</th>
                      <th>Время</th>
                      <th>Уверенность</th>
                      <th>Статус</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {safety.incidents.map((i) => (
                      <tr key={i.id}>
                        <td>
                          <button
                            className="event-button"
                            onClick={() => setDetail(i)}
                          >
                            <span className={"event-icon " + i.type}>
                              <AlertTriangle size={16} />
                            </span>
                            {typeLabels[i.type] || labels[i.type] || i.type}
                          </button>
                        </td>
                        <td>
                          <button
                            className="text-button"
                            title="Открыть камеру"
                            onClick={() => {
                              setSelected(i.camera_id);
                              setPage("monitor");
                            }}
                          >
                            {i.camera_name}
                          </button>
                          <small>Worker #{i.track_id}</small>
                        </td>
                        <td>
                          {date(i.started_at)}
                          <small>{i.duration.toFixed(1)} сек.</small>
                        </td>
                        <td>
                          <span className="table-confidence">
                            {percentage(i.confidence)}
                          </span>
                        </td>
                        <td>
                          <span className={"status-badge " + i.status}>
                            {labels[i.status]}
                          </span>
                        </td>
                        <td>
                          <button
                            className="icon-button"
                            aria-label="Открыть инцидент"
                            onClick={() => setDetail(i)}
                          >
                            <ExternalLink size={15} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="incidents-empty">
                <ShieldCheck size={27} />
                <div>
                  <h3>Пока всё спокойно.</h3>
                  <p>
                    {safety.filter
                      ? "Инцидентов с выбранным статусом нет."
                      : "Подтверждённые нарушения появятся здесь с фотографией и временем."}
                  </p>
                </div>
              </div>
            )}
            {safety.total > 30 && (
              <div className="pagination">
                <button
                  className="button small"
                  disabled={!safety.offset}
                  onClick={() =>
                    safety.setOffset(Math.max(0, safety.offset - 30))
                  }
                >
                  Назад
                </button>
                <span>
                  {safety.offset + 1}–
                  {Math.min(safety.offset + 30, safety.total)} из {safety.total}
                </span>
                <button
                  className="button small"
                  disabled={safety.offset + 30 >= safety.total}
                  onClick={() => safety.setOffset(safety.offset + 30)}
                >
                  Далее
                </button>
              </div>
            )}
          </section>
        )}
        <footer>
          <a
            className="footer-brand"
            href="#"
            onClick={(e) => e.preventDefault()}
          >
            safety ai
            <ArrowRight size={14} />
          </a>
          <span>
            Кадры и история хранятся на этом устройстве.
            {safety.info?.telegram_ready
              ? " Снимки отправляются в Telegram."
              : ""}
          </span>
          <span>Industrial Safety / 2026</span>
        </footer>
      </main>
      {detail && (
        <IncidentDialog
          incident={detail}
          onClose={() => setDetail(null)}
          busy={busy}
          onFeedback={(verdict) =>
            void act(async () => {
              setDetail(
                await api<Incident>(
                  `/incidents/${detail.id}/feedback`,
                  json({ verdict }),
                ),
              );
            })
          }
        />
      )}
      </>
      )}
    </div>
  );
}
