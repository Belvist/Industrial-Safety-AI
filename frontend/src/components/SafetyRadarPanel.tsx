import { AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";
import type { Radar, RadarEvent } from "../types";
import { useAuthedVideoUrl } from "../useAuthedVideo";

type Props = {
  radar: Radar | null | undefined;
  workers: number;
};

const levelConfig = {
  NORMAL: { label: "Норма", color: "normal", icon: CheckCircle },
  WARNING: { label: "Риск", color: "warning", icon: AlertTriangle },
  CRITICAL: { label: "Опасность", color: "critical", icon: AlertTriangle },
  UNKNOWN: { label: "Нет данных", color: "unknown", icon: HelpCircle },
};

function riskSummary(radar: Radar | null | undefined): keyof typeof levelConfig {
  if (!radar) return "UNKNOWN";
  const levels = Object.values(radar.track_risks);
  if (levels.includes("CRITICAL")) return "CRITICAL";
  if (levels.includes("WARNING")) return "WARNING";
  if (levels.includes("UNKNOWN") || levels.length === 0) return "UNKNOWN";
  return "NORMAL";
}

export default function SafetyRadarPanel({ radar, workers }: Props) {
  const summary = riskSummary(radar);
  const evidenceSrc = radar?.events.find((e) => e.evidence_url)?.evidence_url ?? null;
  const videoUrl = useAuthedVideoUrl(evidenceSrc);
  const cfg = levelConfig[summary];
  const Icon = cfg.icon;
  const events = radar?.events ?? [];

  return (
    <section className="radar-panel" aria-label="Safety Radar">
      <div className={`radar-status radar-${cfg.color}`}>
        <Icon size={18} />
        <div>
          <strong>Safety Radar: {cfg.label}</strong>
          <small>
            {workers > 0
              ? `${workers} работник(ов) в кадре`
              : "Нет подтверждённых работников"}
          </small>
        </div>
      </div>
      {events.length > 0 && (
        <ul className="radar-events">
          {events.map((ev: RadarEvent, idx) => (
            <li key={`${ev.track_id}-${ev.timestamp}-${idx}`} className={`radar-event ${ev.level.toLowerCase()}`}>
              <span className="radar-event-level">{ev.level}</span>
              <span className="radar-event-message">{ev.message}</span>
              {ev.evidence_url && videoUrl && (
                <a className="radar-replay" href={videoUrl} target="_blank" rel="noreferrer">
                  Replay
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
      {videoUrl && <video className="radar-video" controls src={videoUrl} />}
      {events.length === 0 && summary !== "UNKNOWN" && (
        <p className="radar-empty">Опасных сближений не обнаружено.</p>
      )}
    </section>
  );
}
