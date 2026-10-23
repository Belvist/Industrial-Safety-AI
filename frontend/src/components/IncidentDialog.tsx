import { useEffect } from "react";
import { Check, X } from "lucide-react";
import type { Incident } from "../types";
import { date, labels, percentage } from "../types";
import { useAuthedImageUrl } from "../useAuthedImage";

function SnapshotImage({ incidentId }: { incidentId: string }) {
  const src = useAuthedImageUrl(`/api/v1/incidents/${incidentId}/snapshot`);
  if (!src) return null;
  return <img src={src} alt="Снимок подтверждённого нарушения" />;
}
export default function IncidentDialog({
  incident,
  onClose,
  onFeedback,
  busy,
}: {
  incident: Incident;
  onClose: () => void;
  onFeedback: (verdict: string) => void;
  busy: boolean;
}) {
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", listener);
    return () => document.removeEventListener("keydown", listener);
  }, [onClose]);
  const notification: Record<string, string> = {
    sent: "Фото отправлено в Telegram",
    pending: "Telegram: отправляется",
    failed: "Telegram: ошибка доставки",
    not_configured: "Telegram не настроен",
    disabled: "Telegram выключен",
  };
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Инцидент"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-heading">
          <div>
            <span className="eyebrow">ДОКАЗАТЕЛЬСТВО / ИНЦИДЕНТ</span>
            <h2>{labels[incident.type]}</h2>
          </div>
          <button
            className="icon-button"
            aria-label="Закрыть"
            onClick={onClose}
          >
            <X size={22} />
          </button>
        </div>
        {incident.snapshot && (
          <SnapshotImage incidentId={incident.id} />
        )}
        <div className="modal-content">
          <span className={"status-badge " + incident.status}>
            {labels[incident.status]}
          </span>
          <h3>
            {incident.camera_name} · Worker #{incident.track_id}
          </h3>
          <div className="incident-metrics">
            <div>
              <span>Время</span>
              <strong>{date(incident.started_at)}</strong>
            </div>
            <div>
              <span>Длительность</span>
              <strong>{incident.duration.toFixed(1)} сек.</strong>
            </div>
            <div>
              <span>Уверенность</span>
              <strong>{percentage(incident.confidence)}</strong>
            </div>
          </div>
          <p>{notification[incident.notification] || incident.notification}</p>
          {incident.resolution_reason && (
            <p className="notice-text">
              Причина: {labels[incident.resolution_reason] || incident.resolution_reason}
            </p>
          )}
          {incident.status === "interrupted" && (
            <p className="notice-text">
              Наблюдение прервано. Устранение нарушения не подтверждено.
            </p>
          )}
          <div className="feedback">
            <strong>Проверка оператором</strong>
            <p>Ваш вердикт сохраняется в истории инцидента.</p>
            <div className="inline">
              <button
                className={
                  "button " +
                  (incident.feedback === "confirmed" ? "primary" : "")
                }
                disabled={busy}
                onClick={() => onFeedback("confirmed")}
              >
                <Check size={16} />
                Подтвердить
              </button>
              <button
                className={
                  "button " +
                  (incident.feedback === "false_positive" ? "stop" : "")
                }
                disabled={busy}
                onClick={() => onFeedback("false_positive")}
              >
                <X size={16} />
                Ложное срабатывание
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
