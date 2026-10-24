import {
  ArrowRight,
  CircleHelp,
  Plus,
  ShieldCheck,
  Video,
  Camera as CameraIcon,
  AlertTriangle,
  Eye,
  FileCheck2,
} from "lucide-react";
import { useAuthedImageUrl } from "../useAuthedImage";
import type { Camera, Incident, Info } from "../types";
import { date } from "../types";
import { lifecycleNames, ruleTitle, sourceNames } from "./model";

function Preview({ camera, onOpen }: { camera: Camera; onOpen: () => void }) {
  const image = useAuthedImageUrl(camera.latest?.frame_url ?? null);
  const live = camera.status === "running";
  return (
    <button
      className="p-camera-card"
      aria-label={`Камера ${camera.name}`}
      onClick={onOpen}
    >
      <span className="p-camera-picture">
        {image ? (
          <img src={image} alt="" />
        ) : (
          <span className="p-camera-placeholder">
            <Video size={30} />
            <span>
              {camera.status === "ended"
                ? "Анализ записи завершён"
                : "Изображение появится после подключения"}
            </span>
          </span>
        )}
        <span
          className={`p-camera-badge ${live ? "live" : camera.status === "error" || camera.status === "stale" ? "attention" : ""}`}
        >
          <i />
          {live
            ? "В работе"
            : camera.status === "error"
              ? "Ошибка подключения"
              : camera.status === "stale"
                ? "Нет свежих кадров"
                : camera.status === "starting"
                  ? "Подключается"
                  : camera.status === "ended"
                    ? "Запись обработана"
                    : "Остановлена"}
        </span>
      </span>
      <span className="p-camera-card-bottom">
        <span>
          <strong>{camera.name}</strong>
          <small>
            {sourceNames[camera.kind ?? ""] ?? "Источник ещё не подключён"}
          </small>
        </span>
        <ArrowRight size={18} />
      </span>
    </button>
  );
}
export default function ObservationHome({
  cameras,
  info,
  recent,
  firstRun,
  onConnect,
  onOpen,
  onIncident,
  onJournal,
  onHelp,
}: {
  cameras: Camera[];
  info: Info | null;
  recent: Incident[];
  firstRun: boolean;
  onConnect: () => void;
  onOpen: (id: string) => void;
  onIncident: (incident: Incident) => void;
  onJournal: () => void;
  onHelp: () => void;
}) {
  const running = cameras.filter((c) => c.status === "running").length;
  if (firstRun)
    return (
      <section className="p-welcome">
        <div className="p-welcome-copy">
          <span className="p-kicker">ВАШЕ РАБОЧЕЕ ПРОСТРАНСТВО</span>
          <h1>
            Начните
            <br />с одной камеры.
          </h1>
          <p>
            Наблюдайте за рабочей зоной. Система проверит средства защиты и
            сохранит подтверждённые нарушения — чтобы вы могли вовремя
            отреагировать.
          </p>
          <div className="p-welcome-actions">
            <button className="p-button p-primary p-large" onClick={onConnect}>
              <Plus size={18} />
              Подключить первую камеру
            </button>
            <button className="p-link" onClick={onHelp}>
              <CircleHelp size={16} />
              Как это работает
            </button>
          </div>
          <span className="p-privacy-line">
            <ShieldCheck size={14} />
            Снимки и история сохраняются на вашем устройстве.
          </span>
        </div>
        <div className="p-welcome-visual" aria-hidden="true">
          <span className="p-visual-label">РАБОЧАЯ ЗОНА / НАБЛЮДЕНИЕ</span>
          <div className="p-visual-camera">
            <CameraIcon size={36} />
            <i />
            <i />
            <i />
            <i />
          </div>
          <div className="p-visual-line" />
          <div className="p-visual-result">
            <ShieldCheck size={23} />
            <span>
              Заметить.
              <br />
              <strong>Проверить. Отреагировать.</strong>
            </span>
          </div>
          <span className="p-visual-caption">
            Внимание — тому, что действительно важно.
          </span>
        </div>
        <div className="p-welcome-steps">
          {[
            [
              Video,
              "01",
              "Подключите источник",
              "Веб-камера, IP-камера или видеозапись.",
            ],
            [
              Eye,
              "02",
              "Выберите проверки",
              "Каска, жилет и опасные зоны на изображении.",
            ],
            [
              FileCheck2,
              "03",
              "Разберите событие",
              "Снимок, время и решение оператора в одном месте.",
            ],
          ].map(([Icon, n, title, text]) => {
            const Glyph = Icon as typeof Video;
            return (
              <div key={String(n)}>
                <span>{String(n)}</span>
                <Glyph size={21} />
                <h2>{String(title)}</h2>
                <p>{String(text)}</p>
              </div>
            );
          })}
        </div>
      </section>
    );
  return (
    <>
      <div className="p-page-heading">
        <div>
          <span className="p-kicker">РАБОЧЕЕ ПРОСТРАНСТВО</span>
          <h1>Наблюдение</h1>
          <p>
            {running
              ? `В работе: ${running} из ${cameras.length} камер.`
              : "Запустите камеру, чтобы наблюдать за рабочей зоной."}
          </p>
        </div>
        <button className="p-button p-primary" onClick={onConnect}>
          <Plus size={16} />
          Подключить камеру
        </button>
      </div>
      {!!info?.stats.active && (
        <button className="p-attention-banner" onClick={onJournal}>
          <AlertTriangle size={20} />
          <div>
            <strong>Активных инцидентов: {info.stats.active}</strong>
            <span>
              Откройте журнал, чтобы посмотреть доказательства и проверить
              события.
            </span>
          </div>
          <ArrowRight size={19} />
        </button>
      )}
      <section className="p-camera-overview" aria-label="Камеры наблюдения">
        <div className="p-section-title">
          <h2>
            Ваши камеры <span>{cameras.length}</span>
          </h2>
          <span>Выберите камеру, чтобы открыть наблюдение</span>
        </div>
        <div className="p-camera-cards">
          {cameras.map((camera) => (
            <Preview
              key={camera.id}
              camera={camera}
              onOpen={() => onOpen(camera.id)}
            />
          ))}
        </div>
      </section>
      <section className="p-recent">
        <div className="p-section-title">
          <h2>Последние события</h2>
          <button className="p-link" onClick={onJournal}>
            Открыть журнал
            <ArrowRight size={15} />
          </button>
        </div>
        {recent.length ? (
          <div className="p-recent-list">
            {recent.slice(0, 3).map((incident) => (
              <button key={incident.id} onClick={() => onIncident(incident)}>
                <span
                  className={`p-event-mark ${incident.type === "danger_zone" ? "danger" : ""}`}
                >
                  <AlertTriangle size={18} />
                </span>
                <span>
                  <strong>{ruleTitle(incident.type)}</strong>
                  <small>
                    {incident.camera_name} · {date(incident.started_at)}
                  </small>
                </span>
                <span className={`p-lifecycle ${incident.status}`}>
                  {lifecycleNames[incident.status]}
                </span>
                <ArrowRight size={16} />
              </button>
            ))}
          </div>
        ) : (
          <div className="p-quiet">
            <FileCheck2 size={22} />
            <div>
              <strong>Здесь появятся подтверждённые события</strong>
              <p>
                Для каждого сохранится снимок, камера и время. События не
                создаются без наблюдаемого нарушения.
              </p>
            </div>
          </div>
        )}
      </section>
    </>
  );
}
