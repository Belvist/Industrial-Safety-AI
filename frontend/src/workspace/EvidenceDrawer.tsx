import { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  Download,
  Film,
  Image as ImageIcon,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";
import Dialog from "./Dialog";
import { useEvidence } from "../hooks/useEvidence";
import type { Incident } from "../types";
import { date, percentage } from "../types";
import { friendlyError, lifecycleNames, reviewNames, ruleTitle } from "./model";

export default function EvidenceDrawer({
  incident,
  busy,
  onClose,
  onFeedback,
  onCamera,
  onNext,
}: {
  incident: Incident;
  busy: boolean;
  onClose: () => void;
  onFeedback: (verdict: string) => Promise<void>;
  onCamera: () => void;
  onNext?: () => void;
}) {
  const [tab, setTab] = useState<"snapshot" | "video">(
      incident.evidence ? "video" : "snapshot",
    ),
    [error, setError] = useState("");
  const path =
    tab === "video" && incident.evidence
      ? `/api/v1/cameras/${incident.camera_id}/evidence/${incident.evidence}`
      : incident.snapshot
        ? `/api/v1/incidents/${incident.id}/snapshot`
        : null;
  const asset = useEvidence(path, tab === "video");
  async function review(verdict: string) {
    setError("");
    try {
      await onFeedback(verdict);
    } catch (e) {
      setError(friendlyError(e));
    }
  }
  return (
    <Dialog
      title={ruleTitle(incident.type)}
      description={`${incident.camera_name} · ${date(incident.started_at)}`}
      onClose={onClose}
      className="p-evidence-drawer"
      footer={
        <div className="p-verdict-footer">
          <span>Решение оператора</span>
          <p>
            Оценка сохраняется в журнале. Состояние самого нарушения
            определяется по видео.
          </p>
          <div>
            <button
              className={`p-button ${incident.feedback === "false_positive" ? "p-selected" : ""}`}
              disabled={busy}
              onClick={() => void review("false_positive")}
            >
              <X size={16} />
              Ложное срабатывание
            </button>
            <button
              className={`p-button p-primary ${incident.feedback === "confirmed" ? "p-selected" : ""}`}
              disabled={busy}
              onClick={() => void review("confirmed")}
            >
              {busy ? (
                <Loader2 className="spin" size={16} />
              ) : (
                <Check size={16} />
              )}
              Подтвердить нарушение
            </button>
          </div>
          {incident.feedback !== "unreviewed" && (
            <span className="p-review-saved">
              <Check size={14} />
              {reviewNames[incident.feedback]} · сохранено
              {onNext && (
                <button className="p-link" onClick={onNext}>
                  Следующее событие
                  <ArrowRight size={14} />
                </button>
              )}
            </span>
          )}
          {error && (
            <div className="p-inline-error" role="alert">
              {error}
            </div>
          )}
        </div>
      }
    >
      <div className="p-evidence-tabs">
        <button
          className={tab === "snapshot" ? "selected" : ""}
          aria-pressed={tab === "snapshot"}
          onClick={() => setTab("snapshot")}
        >
          <ImageIcon size={16} />
          Снимок
        </button>
        {incident.evidence && (
          <button
            className={tab === "video" ? "selected" : ""}
            aria-pressed={tab === "video"}
            onClick={() => setTab("video")}
          >
            <Film size={16} />
            Видеодоказательство
          </button>
        )}
        {asset.url && (
          <a
            href={asset.url}
            className="p-link"
            download={`incident-${incident.id}.${tab === "video" ? "mp4" : "jpg"}`}
          >
            <Download size={15} />
            Скачать
          </a>
        )}
      </div>
      <div className="p-evidence-media">
        {asset.loading ? (
          <div>
            <Loader2 size={28} className="spin" />
            <p>
              {tab === "video"
                ? "Подготавливаем видеодоказательство…"
                : "Загружаем снимок…"}
            </p>
            {tab === "video" && (
              <small>
                Ролик формируется после события. Снимок доступен отдельно.
              </small>
            )}
          </div>
        ) : asset.url ? (
          tab === "video" ? (
            <video
              controls
              playsInline
              src={asset.url}
              onError={() =>
                setError(
                  "Браузер не смог воспроизвести этот ролик. Скачайте видео или откройте снимок события.",
                )
              }
            />
          ) : (
            <img
              src={asset.url}
              alt={`Снимок: ${ruleTitle(incident.type)}, ${incident.camera_name}`}
            />
          )
        ) : (
          <div>
            <ImageIcon size={28} />
            <p>
              {path
                ? "Доказательство сейчас недоступно"
                : "Для этого события файл не сохранён"}
            </p>
            {path && (
              <button className="p-button" onClick={asset.retry}>
                <RefreshCw size={15} />
                Попробовать ещё раз
              </button>
            )}
          </div>
        )}
      </div>
      <div className="p-evidence-summary">
        <span className={`p-review-state ${incident.feedback}`}>
          {reviewNames[incident.feedback] ?? "На проверке"}
        </span>
        <span className={`p-lifecycle ${incident.status}`}>
          {lifecycleNames[incident.status]}
        </span>
      </div>
      {incident.status === "interrupted" && (
        <div className="p-inform p-warning">
          <AlertTriangle size={18} />
          <p>
            Наблюдение было прервано. Это не подтверждает устранение нарушения.
          </p>
        </div>
      )}
      <dl className="p-evidence-facts">
        <div>
          <dt>Камера</dt>
          <dd>
            <button className="p-link" onClick={onCamera}>
              {incident.camera_name}
              <ArrowRight size={14} />
            </button>
          </dd>
        </div>
        {incident.zone_name && (
          <div>
            <dt>Опасная зона</dt>
            <dd>{incident.zone_name}</dd>
          </div>
        )}
        <div>
          <dt>Сотрудник в наблюдении</dt>
          <dd>#{incident.track_id}</dd>
        </div>
        <div>
          <dt>Время события</dt>
          <dd>{date(incident.started_at)}</dd>
        </div>
        <div>
          <dt>Длительность наблюдения</dt>
          <dd>{incident.duration.toFixed(1)} сек.</dd>
        </div>
        <div>
          <dt>Уверенность распознавания</dt>
          <dd>{percentage(incident.confidence)}</dd>
        </div>
      </dl>
      <details className="p-advanced">
        <summary>Как интерпретировать событие</summary>
        <p>
          Процент показывает уверенность модели в распознанном объекте, а не
          вероятность безопасности. Номер сотрудника обозначает трек в текущем
          сеансе и не устанавливает личность. Проверьте изображение, прежде чем
          сохранить оценку.
        </p>
      </details>
    </Dialog>
  );
}
