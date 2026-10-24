import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  ChevronDown,
  ChevronRight,
  CircleHelp,
  FileCheck2,
  LayoutGrid,
  Menu,
  Moon,
  Settings2,
  ShieldCheck,
  Sun,
  Video,
  X,
} from "lucide-react";
import type { Camera, Info } from "../types";
import { pageNames } from "./model";
import type { WorkspacePage } from "./model";

const states: Record<string, string> = {
  running: "Наблюдение работает",
  stale: "Нет свежих кадров",
  starting: "Подключается",
  uploading: "Загрузка записи",
  stopping: "Останавливается",
  stopped: "Остановлена",
  error: "Ошибка подключения",
  ended: "Запись обработана",
};
const navigation = [
  { page: "dashboard" as const, title: "Наблюдение", icon: LayoutGrid },
  { page: "incidents" as const, title: "Инциденты", icon: FileCheck2 },
  { page: "cameras" as const, title: "Камеры", icon: Video },
];

export default function WorkspaceShell({
  route,
  cameras,
  info,
  online,
  connecting,
  firstRun,
  onNavigate,
  onHelp,
  children,
}: {
  route: { page: WorkspacePage; cameraId: string };
  cameras: Camera[];
  info: Info | null;
  online: boolean;
  connecting: boolean;
  firstRun: boolean;
  onNavigate: (page: WorkspacePage, cameraId?: string) => void;
  onHelp: () => void;
  children: ReactNode;
}) {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("safety.theme") === "light" ? "light" : "dark",
  );
  const [open, setOpen] = useState(() => window.innerWidth > 900),
    [quick, setQuick] = useState(false);
  const selected =
      cameras.find((c) => c.id === route.cameraId) ??
      (["cameras", "monitor"].includes(route.page) ? cameras[0] : undefined),
    unreviewed = info?.stats.unreviewed ?? 0;
  useEffect(() => {
    const media = window.matchMedia("(max-width:900px)");
    const resize = () => {
      if (media.matches) setOpen(false);
    };
    media.addEventListener("change", resize);
    return () => media.removeEventListener("change", resize);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key.toLowerCase() === "b" &&
        !e.repeat &&
        !document.querySelector('[role="dialog"]')
      ) {
        e.preventDefault();
        setOpen((value) => !value);
      }
      if (e.key === "Escape" && window.innerWidth <= 900) setOpen(false);
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
    if (window.innerWidth <= 900) setOpen(false);
  }
  function nav(className: string) {
    return (
      <nav className={className} aria-label="Основная навигация">
        {navigation.map(({ page, title, icon: Icon }) => (
          <button
            key={page}
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
            <Icon size={19} aria-hidden="true" />
            <span>{title}</span>
            {page === "incidents" && unreviewed > 0 && <b>{unreviewed}</b>}
          </button>
        ))}
      </nav>
    );
  }
  return (
    <div
      className={`product-app p-operator-shell ${open ? "p-navigation-open" : "p-navigation-closed"} ${quick ? "p-quick-open" : ""}`}
      data-theme={theme}
    >
      {open && (
        <button
          className="p-navigation-scrim"
          aria-label="Скрыть навигацию"
          onClick={() => setOpen(false)}
        />
      )}
      <aside className="p-app-sidebar" aria-label="Навигация приложения">
        <button
          className="p-app-brand"
          aria-label="Industrial Safety AI — наблюдение"
          onClick={() => go("dashboard")}
        >
          <span>
            <ShieldCheck size={25} />
          </span>
          <div>
            <strong>
              safety<span>ai</span>
            </strong>
            <small>Безопасность на производстве</small>
          </div>
        </button>
        <button
          className="p-icon p-sidebar-close"
          aria-label="Закрыть меню"
          onClick={() => setOpen(false)}
        >
          <X size={18} />
        </button>
        <span className="p-app-nav-label">РАБОЧЕЕ ПРОСТРАНСТВО</span>
        {nav("p-main-navigation")}
        <div className="p-camera-shortcuts">
          <button
            className="p-camera-shortcuts-heading"
            aria-expanded={quick}
            onClick={() => setQuick(!quick)}
          >
            <span>Быстрый доступ к камерам</span>
            {quick ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          </button>
          {quick && (
            <div className="p-quick-camera-list">
              {cameras.map((c) => (
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
                  <span>
                    <strong>{c.name}</strong>
                    <small>
                      {firstRun && !c.kind
                        ? "Источник не подключён"
                        : states[c.status]}
                    </small>
                  </span>
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
          )}
        </div>
        <div className="p-app-sidebar-bottom">
          <button
            className={route.page === "system" ? "selected" : ""}
            aria-label="Система"
            onClick={() => go("system")}
          >
            <Settings2 size={18} />
            <span>Система</span>
          </button>
          <button aria-label="Как пользоваться" onClick={onHelp}>
            <CircleHelp size={18} />
            <span>Как пользоваться</span>
          </button>
          <span className="p-local-mark">
            <i className={online ? "live" : ""} />
            {online
              ? "Приложение подключено"
              : connecting
                ? "Подключение…"
                : "Нет связи с сервером"}
          </span>
        </div>
      </aside>
      <main className="p-workspace">
        <header className="p-workspace-header">
          <div className="p-app-breadcrumb">
            <button
              className="p-icon p-navigation-toggle"
              aria-label={open ? "Скрыть навигацию" : "Показать навигацию"}
              title="Навигация · Ctrl B"
              onClick={() => setOpen(!open)}
            >
              <Menu size={18} />
            </button>
            <span>{pageNames[route.page]}</span>
            {selected && (
              <>
                <ChevronRight size={13} />
                <strong>{selected.name}</strong>
              </>
            )}
          </div>
          <div className="p-app-header-actions">
            <span className="p-workspace-label">Industrial Safety AI</span>
            <button
              className="p-icon"
              aria-label={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
              title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button
              className="p-icon p-mobile-system"
              aria-label="Система"
              onClick={() => go("system")}
            >
              <Settings2 size={18} />
            </button>
          </div>
        </header>
        {children}
      </main>
      {nav("p-mobile-navigation")}
    </div>
  );
}
