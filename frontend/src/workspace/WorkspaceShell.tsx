import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  CircleHelp,
  FileCheck2,
  LayoutGrid,
  Moon,
  PanelLeft,
  Plus,
  Settings2,
  ShieldCheck,
  Sun,
  Video,
} from "lucide-react";
import type { Camera, Info, Result } from "../types";
import { pageNames } from "./model";
import type { WorkspacePage } from "./model";

const states: Record<string, string> = {
  running: "Наблюдение",
  stale: "Нет свежих кадров",
  starting: "Подключается",
  uploading: "Загрузка записи",
  stopping: "Останавливается",
  stopped: "Остановлена",
  error: "Ошибка подключения",
  ended: "Запись обработана",
};
const nav = [
  { page: "dashboard" as const, title: "Наблюдение", icon: LayoutGrid },
  { page: "incidents" as const, title: "Инциденты", icon: FileCheck2 },
  { page: "cameras" as const, title: "Камеры", icon: Video },
];
export default function WorkspaceShell({
  route,
  cameras,
  info,
  result,
  online,
  connecting,
  firstRun,
  onNavigate,
  onConnect,
  onHelp,
  children,
}: {
  route: { page: WorkspacePage; cameraId: string };
  cameras: Camera[];
  info: Info | null;
  result?: Result | null;
  online: boolean;
  connecting: boolean;
  firstRun: boolean;
  onNavigate: (page: WorkspacePage, cameraId?: string) => void;
  onConnect: () => void;
  onHelp: () => void;
  children: ReactNode;
}) {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("safety.theme") === "light" ? "light" : "dark",
  );
  const [explorer, setExplorer] = useState(() => window.innerWidth > 900),
    [query, setQuery] = useState("");
  const selected =
      cameras.find((c) => c.id === route.cameraId) ??
      (["cameras", "monitor"].includes(route.page) ? cameras[0] : undefined),
    unreviewed = info?.stats.unreviewed ?? 0;
  useEffect(() => {
    const media = window.matchMedia("(max-width:900px)");
    const resize = () => {
      if (media.matches) setExplorer(false);
    };
    media.addEventListener("change", resize);
    return () => media.removeEventListener("change", resize);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key.toLowerCase() === "b" &&
        !document.querySelector('[role="dialog"]')
      ) {
        e.preventDefault();
        setExplorer((value) => !value);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  useEffect(() => {
    localStorage.setItem("safety.theme", theme);
    document.documentElement.style.colorScheme = theme;
  }, [theme]);
  function go(page: WorkspacePage, id?: string) {
    onNavigate(page, id ?? (page === "system" ? route.cameraId : undefined));
    if (window.innerWidth <= 900) setExplorer(false);
  }
  function toggleTheme() {
    setTheme(theme === "dark" ? "light" : "dark");
  }
  return (
    <div
      className={`product-app p-editor-shell ${explorer ? "p-explorer-open" : ""}`}
      data-theme={theme}
    >
      <aside className="p-activitybar">
        <button
          className="p-app-mark"
          aria-label="Industrial Safety AI — наблюдение"
          title="Industrial Safety AI"
          onClick={() => go("dashboard")}
        >
          <ShieldCheck size={24} />
        </button>
        <nav aria-label="Основная навигация">
          {nav.map(({ page, title, icon: Icon }) => (
            <button
              key={page}
              title={title}
              aria-label={title}
              aria-current={
                route.page === page ||
                (page === "dashboard" && route.page === "monitor")
                  ? "page"
                  : undefined
              }
              onClick={() =>
                go(page, page === "cameras" ? route.cameraId : undefined)
              }
            >
              <Icon size={21} />
              <span>{title}</span>
              {page === "incidents" && unreviewed > 0 && <b>{unreviewed}</b>}
            </button>
          ))}
        </nav>
        <div className="p-activity-bottom">
          <button
            title="Как пользоваться"
            aria-label="Как пользоваться"
            onClick={onHelp}
          >
            <CircleHelp size={20} />
          </button>
          <button
            title="Система"
            aria-label="Система"
            className={route.page === "system" ? "selected" : ""}
            onClick={() => go("system")}
          >
            <Settings2 size={20} />
          </button>
        </div>
      </aside>
      {explorer && (
        <button
          className="p-explorer-scrim"
          aria-label="Скрыть список камер"
          onClick={() => setExplorer(false)}
        />
      )}
      <aside className="p-sidebar p-camera-explorer" aria-label="Список камер">
        <header className="p-project-title">
          <ShieldCheck size={16} />
          <strong>Industrial Safety AI</strong>
          <button
            className="p-icon"
            aria-label="Скрыть список камер"
            title="Скрыть список камер · Ctrl B"
            onClick={() => setExplorer(false)}
          >
            <PanelLeft size={16} />
          </button>
        </header>
        <div className="p-explorer-heading">
          <span>КАМЕРЫ</span>
          <button
            className="p-icon"
            title="Подключить камеру"
            aria-label="Подключить камеру в списке"
            onClick={onConnect}
          >
            <Plus size={16} />
          </button>
        </div>
        {cameras.length > 4 && (
          <div className="p-explorer-search">
            <input
              aria-label="Найти камеру"
              placeholder="Найти камеру…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        )}
        <div className="p-explorer-cameras">
          {cameras
            .filter((c) =>
              c.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()),
            )
            .map((c) => (
              <button
                key={c.id}
                className={selected?.id === c.id ? "selected" : ""}
                aria-label={`Открыть камеру ${c.name}`}
                title={`${c.name} · ${states[c.status]}`}
                onClick={() =>
                  go(route.page === "cameras" ? "cameras" : "monitor", c.id)
                }
              >
                <Video size={15} />
                <div>
                  <strong>{c.name}</strong>
                  <small>
                    {firstRun && !c.kind
                      ? "Источник не подключён"
                      : states[c.status]}
                  </small>
                </div>
                <i
                  className={
                    c.status === "running"
                      ? "live"
                      : c.status === "error" || c.status === "stale"
                        ? "attention"
                        : ""
                  }
                />
              </button>
            ))}
        </div>
        <button className="p-explorer-add" onClick={onConnect}>
          <Plus size={14} />
          Подключить камеру
        </button>
        <div className="p-explorer-footer">
          <span>
            <ShieldCheck size={13} />
            Локальное пространство
          </span>
          <p>
            Камеры, события и доказательства
            <br />
            на компьютере оператора.
          </p>
        </div>
      </aside>
      <main className="p-workspace">
        <header className="p-workspace-header">
          <div className="p-editor-tabs">
            <button
              className="p-icon p-toggle-explorer"
              aria-label={
                explorer ? "Скрыть список камер" : "Показать список камер"
              }
              title="Список камер · Ctrl B"
              onClick={() => setExplorer(!explorer)}
            >
              <PanelLeft size={17} />
            </button>
            <div className="p-editor-tab">
              {route.page === "incidents" ? (
                <FileCheck2 size={15} />
              ) : route.page === "cameras" ? (
                <Settings2 size={15} />
              ) : route.page === "system" ? (
                <Activity size={15} />
              ) : (
                <Video size={15} />
              )}
              <span>{pageNames[route.page]}</span>
              {selected && (
                <>
                  <span className="p-tab-divider">/</span>
                  <strong>{selected.name}</strong>
                </>
              )}
            </div>
          </div>
          <div className="p-header-tools">
            <button
              className="p-icon"
              aria-label={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
              title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
              onClick={toggleTheme}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              className="p-system-top"
              aria-label="Система"
              onClick={() => go("system")}
            >
              <Settings2 size={17} />
            </button>
            <button className="p-help-top" onClick={onHelp}>
              <CircleHelp size={16} />
              <span>Помощь</span>
            </button>
          </div>
        </header>
        {children}
      </main>
      <div className="p-statusbar">
        <span className="p-local-mark">
          <i className={online ? "live" : ""} />
          {online
            ? "Приложение подключено"
            : connecting
              ? "Подключение…"
              : "Нет связи с сервером"}
        </span>
        <button onClick={() => go("system")}>
          {info?.model_ready
            ? info.inference?.mode === "linq_remote"
              ? "LinQ PPE · CPU Person"
              : "Локальная модель"
            : "Распознавание недоступно"}
        </button>
        <span className="p-status-spacer" />
        {result && (
          <>
            <span>
              Аналитика{" "}
              {result.metrics?.rates.analytics?.fps?.toFixed(1) ?? "—"} FPS
            </span>
            {info?.inference?.mode === "linq_remote" && (
              <span>
                TPU {result.metrics?.rates.tpu?.fps?.toFixed(1) ?? "—"} FPS
              </span>
            )}
          </>
        )}
        <button onClick={() => go("incidents")}>
          {unreviewed ? `На проверке: ${unreviewed}` : "Инциденты"}
        </button>
      </div>
    </div>
  );
}
