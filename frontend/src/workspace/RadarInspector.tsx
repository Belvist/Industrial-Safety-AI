import { useState } from "react";
import {
  AlertTriangle,
  Download,
  Film,
  Loader2,
  Radar as RadarIcon,
  RefreshCw,
} from "lucide-react";
import type { Radar, RadarEvent } from "../types";
import { useEvidence } from "../hooks/useEvidence";
import Dialog from "./Dialog";
const levels: Record<string, string> = {
  NORMAL: "Наблюдение есть",
  WARNING: "Приближается к зоне",
  CRITICAL: "В опасной зоне",
  UNKNOWN: "Недостаточно данных",
};
function Replay({
  event,
  onClose,
}: {
  event: RadarEvent;
  onClose: () => void;
}) {
  const video = useEvidence(event.evidence_url ?? null, true);
  return (
    <Dialog
      title="Видеодоказательство"
      description={`Сотрудник #${event.track_id} · ${event.zone_name ?? "Опасная зона"} · ${event.timestamp.toFixed(1)} с от начала источника`}
      onClose={onClose}
      footer={
        <>
          <span className="p-hint">
            Время относится к исходному наблюдению.
          </span>
          {video.url && (
            <a
              className="p-button"
              href={video.url}
              download={`radar-${event.track_id}-${event.timestamp}.mp4`}
            >
              <Download size={15} />
              Скачать
            </a>
          )}
        </>
      }
    >
      <div className="p-evidence-media">
        {video.loading ? (
          <div>
            <Loader2 className="spin" size={26} />
            <p>Подготавливаем видео…</p>
          </div>
        ) : video.url ? (
          <video src={video.url} controls playsInline />
        ) : (
          <div>
            <AlertTriangle size={25} />
            <p>Видео сейчас недоступно</p>
            <button className="p-button" onClick={video.retry}>
              <RefreshCw size={15} />
              Повторить
            </button>
          </div>
        )}
      </div>
    </Dialog>
  );
}
export default function RadarInspector({
  radar,
}: {
  radar: Radar | null | undefined;
}) {
  const [event, setEvent] = useState<RadarEvent | null>(null);
  const tracks = Object.entries(radar?.track_risks ?? {}),
    events = radar?.events ?? [];
  return (
    <div className="p-radar-details">
      <div className="p-radar-explainer">
        <RadarIcon size={22} />
        <p>
          Траектория и прогноз видны на видео. Оценка зоны относится к точке под
          ногами человека.
        </p>
      </div>
      {tracks.length ? (
        <div className="p-radar-tracks">
          {tracks.map(([id, level]) => (
            <div key={id}>
              <span>Сотрудник #{id}</span>
              <strong className={level.toLowerCase()}>{levels[level]}</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="p-hint">Пока нет свежих данных для прогноза движения.</p>
      )}
      <h3>Предупреждения и входы</h3>
      {events.length ? (
        <div className="p-radar-events">
          {events.map((ev, index) => (
            <div key={`${ev.track_id}-${ev.timestamp}-${index}`}>
              <strong className={ev.level.toLowerCase()}>
                {levels[ev.level]}
              </strong>
              <span>
                Сотрудник #{ev.track_id}
                {ev.zone_name ? ` · ${ev.zone_name}` : ""}
              </span>
              <small>{ev.timestamp.toFixed(1)} с от начала источника</small>
              {ev.evidence_url && (
                <button className="p-link" onClick={() => setEvent(ev)}>
                  <Film size={14} />
                  Посмотреть видео
                </button>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="p-hint">
          Предупреждения появятся при приближении к зоне. Подтверждённые
          инциденты сохраняются в журнале.
        </p>
      )}
      {event && <Replay event={event} onClose={() => setEvent(null)} />}
    </div>
  );
}
