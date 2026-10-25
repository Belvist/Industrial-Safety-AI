import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  CircleHelp,
  Loader2,
  LogIn,
  ShieldCheck,
  Video,
  X,
} from "lucide-react";
import { api, json, saveOperatorToken, stopPath } from "./api";
import { useSafety } from "./hooks/useSafety";
import { useBrowserCamera } from "./hooks/useBrowserCamera";
import type { Camera, Config, Incident, Zone } from "./types";
import { configOf, isActive } from "./types";
import WorkspaceShell from "./workspace/WorkspaceShell";
import ObservationHome from "./workspace/ObservationHome";
import Monitor from "./workspace/Monitor";
import ConnectCamera from "./workspace/ConnectCamera";
import CameraSettings from "./workspace/CameraSettings";
import type { SettingsControls } from "./workspace/CameraSettings";
import Journal from "./workspace/Journal";
import EvidenceDrawer from "./workspace/EvidenceDrawer";
import ZoneEditor from "./workspace/ZoneEditor";
import SystemPage from "./workspace/SystemPage";
import Dialog from "./workspace/Dialog";
import { friendlyError, pageNames } from "./workspace/model";
import type { ConnectionDraft, WorkspacePage } from "./workspace/model";

type Route = { page: WorkspacePage; cameraId: string };
function readRoute(): Route {
  const [page, cameraId = ""] = location.hash.slice(1).split("/");
  return {
    page: Object.hasOwn(pageNames, page)
      ? (page as WorkspacePage)
      : "dashboard",
    cameraId,
  };
}
export default function App() {
  const safety = useSafety();
  const [route, setRoute] = useState(readRoute),
    [connectOpen, setConnectOpen] = useState(false),
    [connectInitial, setConnectInitial] = useState<Camera | undefined>(),
    [zoneOpen, setZoneOpen] = useState(false),
    [detail, setDetail] = useState<Incident | null>(null),
    [help, setHelp] = useState(false),
    [error, setError] = useState("");
  const [working, setWorking] = useState(false),
    [toast, setToast] = useState(""),
    [operatorKey, setOperatorKey] = useState("");
  const [seen, setSeen] = useState(
    () => localStorage.getItem("safety.workspace-configured") === "1",
  );
  const [leave, setLeave] = useState<(() => void) | null>(null);
  const dirty = useRef(false),
    settingsControls = useRef<SettingsControls | null>(null),
    executing = useRef(false),
    operation = useRef(0),
    connectionRun = useRef(0),
    connectionCamera = useRef<Camera | undefined>(undefined),
    connectionAbort = useRef<AbortController | null>(null);
  const browser = useBrowserCamera((message) =>
    setError(friendlyError(new Error(message))),
  );
  const camera =
    safety.cameras.find((c) => c.id === route.cameraId) ?? safety.cameras[0];
  const owned =
    browser.owner === camera?.id &&
    browser.result?.session_id === camera?.session_id
      ? browser.result
      : null;
  const latest = camera?.latest;
  const result =
    owned && owned.sequence > (latest?.sequence ?? 0) ? owned : latest;
  const busy =
    working ||
    ["permission", "connecting", "stopping"].includes(browser.state.phase);
  const online = safety.connection === "online";
  const firstRun =
    !seen &&
    safety.cameras.length <= 1 &&
    !safety.cameras.some((c) => c.kind) &&
    !safety.info?.stats.total;
  function navigate(page: WorkspacePage, cameraId = "") {
    const hash = `#${page}${cameraId ? "/" + cameraId : ""}`;
    if (location.hash !== hash) history.pushState(null, "", hash);
    setRoute({ page, cameraId });
    setError("");
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  function request(action: () => void) {
    if (dirty.current) setLeave(() => action);
    else action();
  }
  const onDirty = useCallback((value: boolean, controls: SettingsControls) => {
    dirty.current = value;
    settingsControls.current = controls;
  }, []);
  useEffect(() => {
    const restore = () => {
      if (dirty.current) {
        const target = readRoute();
        history.pushState(
          null,
          "",
          `#${route.page}${route.cameraId ? "/" + route.cameraId : ""}`,
        );
        setLeave(() => () => {
          dirty.current = false;
          navigate(target.page, target.cameraId);
        });
        return;
      }
      setRoute(readRoute());
    };
    window.addEventListener("hashchange", restore);
    return () => {
      window.removeEventListener("hashchange", restore);
    };
  }, [route]);
  useEffect(() => {
    document.title = `${pageNames[route.page]} · Industrial Safety AI`;
  }, [route.page]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 4500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    setDetail((previous) =>
      previous
        ? (safety.incidents.find((i) => i.id === previous.id) ??
          safety.recent.find((i) => i.id === previous.id) ??
          safety.activeIncidents.find((i) => i.id === previous.id) ??
          previous)
        : null,
    );
  }, [safety.incidents, safety.recent, safety.activeIncidents]);
  async function execute<T>(work: () => Promise<T>): Promise<T> {
    if (executing.current)
      throw new Error("Дождитесь завершения текущего действия.");
    const id = ++operation.current;
    executing.current = true;
    setWorking(true);
    try {
      const value = await work();
      // A failed background refresh must not report a successful save as failed.
      await safety.refresh().catch(() => {});
      return value;
    } finally {
      if (operation.current === id) {
        executing.current = false;
        setWorking(false);
      }
    }
  }
  function openConnection(target?: Camera) {
    request(() => {
      connectionCamera.current = target;
      setConnectInitial(target);
      connectionRun.current++;
      setConnectOpen(true);
      setError("");
    });
  }
  function closeConnection() {
    connectionRun.current++;
    connectionAbort.current?.abort();
    if (["permission", "connecting"].includes(browser.state.phase))
      void browser.stop().catch(() => {});
    operation.current++;
    executing.current = false;
    setWorking(false);
    setConnectOpen(false);
    void safety.refresh().catch(() => {});
  }
  async function connect(draft: ConnectionDraft) {
    const run = connectionRun.current,
      controller = new AbortController();
    connectionAbort.current = controller;
    const current = () => {
      if (run !== connectionRun.current || controller.signal.aborted)
        throw new Error("Подключение отменено.");
    };
    await execute(async () => {
      let target = connectionCamera.current;
      if (target && isActive(target))
        throw new Error(
          "Эта камера уже работает. Сначала остановите её наблюдение.",
        );
      const config = {
        ...(target ? configOf(target) : {}),
        name: draft.name.trim(),
        helmet: draft.helmet,
        vest: draft.vest,
      };
      target = await api<Camera>(
        target ? `/cameras/${target.id}` : "/cameras",
        { ...json(config, target ? "PUT" : "POST"), signal: controller.signal },
      );
      current();
      connectionCamera.current = target;
      if (draft.source === "browser")
        await browser.start(
          target.id,
          draft.device,
          safety.info?.target_fps ?? 5,
        );
      else if (draft.source === "video") {
        if (!draft.file) throw new Error("Выберите видеофайл.");
        await api(`/cameras/${target.id}/video`, {
          method: "POST",
          headers: { "Content-Type": "video/mp4" },
          body: draft.file,
          signal: controller.signal,
        });
      } else {
        // Read the startup acknowledgement even after cancelling the dialog,
        // so cleanup stops that exact session without stopping a newer retry.
        const started = await api<Camera>(
          `/cameras/${target.id}/start`,
          json({ kind: "rtsp", url: draft.url }),
        );
        if (run !== connectionRun.current || controller.signal.aborted)
          await api(stopPath(target.id, started.session_id), {
            method: "POST",
          });
      }
      current();
      localStorage.setItem("safety.workspace-configured", "1");
      setSeen(true);
      setConnectOpen(false);
      navigate("monitor", target.id);
      setToast("Источник подключается. Дождитесь первого кадра.");
    });
  }
  async function stop() {
    if (!camera) return;
    try {
      await execute(async () => {
        if (browser.owner === camera.id || browser.state.cameraId === camera.id)
          await browser.stop();
        else
          await api(stopPath(camera.id, camera.session_id), { method: "POST" });
      });
      setToast("Наблюдение остановлено. История событий сохранена.");
    } catch (e) {
      setError(friendlyError(e));
    }
  }
  async function saveConfig(config: Config) {
    if (!camera) throw new Error("Выберите камеру.");
    const saved = await execute(() =>
      api<Camera>(`/cameras/${camera.id}`, json(config, "PUT")),
    );
    setToast("Настройки камеры сохранены.");
    return saved;
  }
  function openZones() {
    request(() => {
      if (!camera) return;
      if (
        !result ||
        camera.status !== "running" ||
        (result.data_age_ms ?? 0) > 5000
      ) {
        navigate("monitor", camera.id);
        setError(
          "Для разметки нужен свежий кадр. Подключите камеру и дождитесь изображения.",
        );
        return;
      }
      setZoneOpen(true);
    });
  }
  async function saveZones(zones: Zone[]) {
    if (!camera) return;
    await saveConfig({
      ...configOf(camera),
      zones,
      zone: [],
      zone_enabled: false,
    });
    setToast("Опасные зоны сохранены. Контроль обновлён.");
  }
  function openJournal(cameraId = "", feedback = "unreviewed") {
    request(() => {
      safety.setIncidentCamera(cameraId);
      safety.setFeedback(feedback);
      safety.setFilter("");
      safety.setIncidentType("");
      safety.setFromDate("");
      safety.setToDate("");
      navigate("incidents");
    });
  }
  async function feedback(verdict: string) {
    if (!detail) return;
    const saved = await execute(() =>
      api<Incident>(`/incidents/${detail.id}/feedback`, json({ verdict })),
    );
    setDetail(saved);
    setToast("Решение оператора сохранено.");
  }
  const nextIncident = safety.incidents.find(
    (i) => i.id !== detail?.id && i.feedback === "unreviewed",
  );
  if (safety.accessDenied)
    return (
      <div className="product-app p-auth">
        <div className="p-auth-card">
          <span className="p-kicker">INDUSTRIAL SAFETY AI</span>
          <ShieldCheck size={35} />
          <h1>Вход в рабочее пространство</h1>
          <p>
            Введите ключ оператора. Камеры и история доступны только после
            входа.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (operatorKey.trim()) {
                saveOperatorToken(operatorKey);
                location.reload();
              }
            }}
          >
            <label className="p-field">
              Ключ оператора
              <input
                type="password"
                autoComplete="off"
                value={operatorKey}
                onChange={(e) => setOperatorKey(e.target.value)}
                autoFocus
              />
            </label>
            <button
              className="p-button p-primary"
              disabled={!operatorKey.trim()}
            >
              <LogIn size={17} />
              Войти
            </button>
          </form>
          <details className="p-advanced">
            <summary>Где взять ключ</summary>
            <p>
              Используйте ссылку штатного запуска приложения или ключ из файла
              data/api_token на компьютере оператора. Для телефона нужен
              отдельный QR-код.
            </p>
          </details>
        </div>
      </div>
    );
  return (
    <WorkspaceShell
      route={route}
      cameras={safety.cameras}
      info={safety.info}
      online={online}
      connecting={safety.connection === "connecting"}
      firstRun={firstRun}
      onNavigate={(page, id) => request(() => navigate(page, id))}
      onHelp={() => setHelp(true)}
    >
      {!online && safety.connection === "offline" && (
        <div className="p-connection-banner" role="status">
          <Activity size={17} />
          <span>Нет связи с приложением. Пытаемся переподключиться.</span>
          <button
            className="p-link"
            onClick={() => request(() => navigate("system"))}
          >
            Проверить систему
            <ArrowRight size={15} />
          </button>
        </div>
      )}
      {online &&
        safety.info &&
        !safety.info.model_ready &&
        route.page !== "system" && (
          <div className="p-connection-banner">
            <ShieldCheck size={17} />
            <span>Распознавание пока недоступно.</span>
            <button
              className="p-link"
              onClick={() => request(() => navigate("system"))}
            >
              Что нужно настроить
              <ArrowRight size={15} />
            </button>
          </div>
        )}
      {error && (
        <div className="p-inline-error p-global-error" role="alert">
          <span>{error}</span>
          <button
            className="p-icon"
            aria-label="Закрыть сообщение"
            onClick={() => setError("")}
          >
            <X size={16} />
          </button>
        </div>
      )}
      <div className="p-page-content">
        {!safety.info && safety.connection === "connecting" ? (
          <div className="p-loading-screen">
            <Loader2 className="spin" size={30} />
            <h1>Открываем рабочее пространство</h1>
            <p>Проверяем подключение и состояние камер.</p>
          </div>
        ) : route.page === "dashboard" ? (
          <ObservationHome
            cameras={safety.cameras}
            info={safety.info}
            recent={safety.recent}
            firstRun={firstRun}
            onConnect={() =>
              openConnection(firstRun ? safety.cameras[0] : undefined)
            }
            onOpen={(id) => navigate("monitor", id)}
            onIncident={setDetail}
            onJournal={() => openJournal("", "")}
            onHelp={() => setHelp(true)}
          />
        ) : route.page === "monitor" && camera ? (
          <Monitor
            camera={camera}
            result={result}
            online={online}
            busy={busy}
            incidents={[
              ...safety.activeIncidents,
              ...safety.recent.filter(
                (i) => !safety.activeIncidents.some((a) => a.id === i.id),
              ),
            ]}
            onBack={() => navigate("dashboard")}
            onConnect={() => openConnection(camera)}
            onStop={() => void stop()}
            onSettings={() => request(() => navigate("cameras", camera.id))}
            onZones={openZones}
            onIncident={setDetail}
            onHistory={() => openJournal(camera.id, "")}
          />
        ) : route.page === "cameras" ? (
          <CameraSettings
            cameras={safety.cameras}
            selected={camera?.id ?? ""}
            info={safety.info}
            busy={busy}
            onSelect={(id) => request(() => navigate("cameras", id))}
            onSave={saveConfig}
            onConnect={openConnection}
            onMonitor={() => request(() => navigate("monitor", camera?.id))}
            onZones={openZones}
            onDirty={onDirty}
          />
        ) : route.page === "incidents" ? (
          <Journal
            items={safety.incidents}
            total={safety.total}
            info={safety.info}
            cameras={safety.cameras}
            loading={safety.incidentLoading}
            error={safety.journalError}
            feedback={safety.feedback}
            status={safety.filter}
            type={safety.incidentType}
            cameraId={safety.incidentCamera}
            fromDate={safety.fromDate}
            toDate={safety.toDate}
            offset={safety.offset}
            onFeedback={safety.setFeedback}
            onStatus={safety.setFilter}
            onType={safety.setIncidentType}
            onCamera={safety.setIncidentCamera}
            onFrom={safety.setFromDate}
            onTo={safety.setToDate}
            onOffset={safety.setOffset}
            params={safety.incidentParams}
            onOpen={setDetail}
            onConnect={() => navigate("dashboard")}
          />
        ) : route.page === "system" ? (
          <SystemPage
            camera={camera}
            info={safety.info}
            cameras={safety.cameras}
            online={online}
            result={result}
            onHelp={() => setHelp(true)}
            onCameras={() => navigate("cameras", camera?.id)}
          />
        ) : (
          <div className="p-empty">
            <Video size={30} />
            <h1>Выберите камеру</h1>
            <button
              className="p-button p-primary"
              onClick={() => navigate("dashboard")}
            >
              Все камеры
            </button>
          </div>
        )}
      </div>
      {toast && !connectOpen && !zoneOpen && !detail && !leave && !help && (
        <div className="p-toast" role="status">
          <Check size={18} />
          <span>{toast}</span>
          <button
            className="p-icon"
            onClick={() => setToast("")}
            aria-label="Закрыть уведомление"
          >
            <X size={15} />
          </button>
        </div>
      )}
      {connectOpen && (
        <ConnectCamera
          camera={connectInitial}
          info={safety.info}
          onClose={closeConnection}
          onConnect={connect}
        />
      )}
      {zoneOpen && camera && result && (
        <ZoneEditor
          camera={camera}
          result={result}
          onClose={() => setZoneOpen(false)}
          onSave={saveZones}
        />
      )}
      {detail && (
        <EvidenceDrawer
          key={detail.id}
          incident={detail}
          busy={working}
          onClose={() => setDetail(null)}
          onFeedback={feedback}
          onCamera={() => {
            const id = detail.camera_id;
            setDetail(null);
            request(() => navigate("monitor", id));
          }}
          onNext={nextIncident ? () => setDetail(nextIncident) : undefined}
        />
      )}
      {leave && (
        <Dialog
          title="Сохранить изменения?"
          description="Настройки камеры ещё не сохранены."
          onClose={() => setLeave(null)}
          className="p-small-dialog"
          footer={
            <>
              <button
                className="p-button"
                onClick={() => {
                  const next = leave;
                  setLeave(null);
                  dirty.current = false;
                  next();
                }}
              >
                Не сохранять
              </button>
              <button
                className="p-button p-primary"
                disabled={working}
                onClick={() =>
                  void (async () => {
                    const ok = await settingsControls.current?.save();
                    if (ok) {
                      const next = leave;
                      setLeave(null);
                      dirty.current = false;
                      next();
                    }
                  })()
                }
              >
                Сохранить и перейти
              </button>
            </>
          }
        >
          <p>
            Можно сохранить настройки перед переходом или вернуться к
            редактированию.
          </p>
          <button className="p-link" onClick={() => setLeave(null)}>
            <ArrowLeft size={15} />
            Продолжить редактирование
          </button>
        </Dialog>
      )}
      {help && (
        <Dialog
          title="От камеры до решения оператора"
          description="Четыре действия, с которых начинается работа."
          onClose={() => setHelp(false)}
          className="p-help-dialog"
          footer={
            <button
              className="p-button p-primary"
              onClick={() => {
                setHelp(false);
                openConnection(firstRun ? safety.cameras[0] : undefined);
              }}
            >
              Подключить камеру
              <ArrowRight size={16} />
            </button>
          }
        >
          <ol className="p-help-steps">
            <li>
              <span>1</span>
              <div>
                <h3>Подключите источник</h3>
                <p>
                  Выберите веб-камеру, IP-камеру или видео. Укажите понятное
                  название и нужные проверки.
                </p>
              </div>
            </li>
            <li>
              <span>2</span>
              <div>
                <h3>Откройте наблюдение</h3>
                <p>
                  Дождитесь изображения. При необходимости отметьте опасную
                  зону. Панели людей и событий открываются по запросу.
                </p>
              </div>
            </li>
            <li>
              <span>3</span>
              <div>
                <h3>Дождитесь подтверждённого события</h3>
                <p>
                  Система проверяет нарушение во времени. Нет изображения или
                  средство защиты не видно — это не подтверждение безопасности.
                </p>
              </div>
            </li>
            <li>
              <span>4</span>
              <div>
                <h3>Проверьте доказательство</h3>
                <p>
                  Откройте инцидент в журнале. Посмотрите снимок или ролик,
                  затем подтвердите нарушение либо отметьте ложное срабатывание.
                </p>
              </div>
            </li>
          </ol>
          <div className="p-help-shortcuts">
            <span>
              <kbd>F</kbd>Режим наблюдения
            </span>
            <span>
              <kbd>Esc</kbd>Закрыть панель
            </span>
            <span>
              <kbd>Ctrl / ⌘ Z</kbd>Отменить точку зоны
            </span>
          </div>
        </Dialog>
      )}
    </WorkspaceShell>
  );
}
