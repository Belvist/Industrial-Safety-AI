import {
  ArrowUpRight,
  ShieldCheck,
  Video,
  Radio,
  AlertTriangle,
} from "lucide-react";
import CameraTile from "./CameraTile";
import type { Camera, Info } from "../types";

export default function Overview({
  cameras,
  online,
  info,
  onSelect,
  onConfigure,
}: {
  cameras: Camera[];
  online: boolean;
  info: Info | null;
  onSelect: (id: string) => void;
  onConfigure: () => void;
}) {
  const attention = cameras.filter((c) =>
    ["error", "stale"].includes(c.status),
  );
  return (
    <div className="overview-layout">
      <section className="overview-cameras">
        <div className="section-heading">
          <div>
            <span className="eyebrow">ОБЪЕКТ</span>
            <h2>
              Все камеры <span className="count">{cameras.length}</span>
            </h2>
          </div>
          <button className="text-button" onClick={onConfigure}>
            Управление <ArrowUpRight size={15} />
          </button>
        </div>
        {cameras.length ? (
          <div className="overview-grid">
            {cameras.map((camera) => (
              <CameraTile
                key={camera.id}
                camera={camera}
                selected={false}
                onSelect={() => onSelect(camera.id)}
              />
            ))}
          </div>
        ) : (
          <div className="overview-empty">
            <Video size={30} />
            <h3>Подключите первую камеру</h3>
            <p>Выберите источник, затем настройте правила рабочей зоны.</p>
            <button className="button primary" onClick={onConfigure}>
              Добавить камеру
            </button>
          </div>
        )}
      </section>
      <aside className="system-panel">
        <span className="eyebrow">СОСТОЯНИЕ СИСТЕМЫ</span>
        <h2>Готовность к работе</h2>
        <div className="system-row">
          <Video size={18} />
          <div>
            <strong>Сервер приложения</strong>
            <small>{online ? "Подключён" : "Нет связи"}</small>
          </div>
          <i className={online ? "ok" : ""} />
        </div>
        <div className="system-row">
          <ShieldCheck size={18} />
          <div>
            <strong>Распознавание</strong>
            <small>
              {info?.model_ready ? "Детектор готов" : "Ожидает настройки"}
            </small>
          </div>
          <i className={info?.model_ready ? "ok" : ""} />
        </div>
        <div className="system-row">
          <Radio size={18} />
          <div>
            <strong>Telegram</strong>
            <small>
              {info?.telegram_ready ? "Уведомления включены" : "Не подключён"}
            </small>
          </div>
          <i className={info?.telegram_ready ? "ok" : ""} />
        </div>
        <div className="overview-note">
          <ShieldCheck size={18} />
          <p>
            Видеопоток обрабатывается на вашем устройстве. Снимки и история
            сохраняются локально.
          </p>
        </div>
        {attention.length > 0 && (
          <div className="attention-list">
            <h3>
              <AlertTriangle size={16} />
              Требуют внимания
            </h3>
            {attention.map((camera) => (
              <button key={camera.id} onClick={() => onSelect(camera.id)}>
                {camera.name}
                <ArrowUpRight size={15} />
              </button>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}
