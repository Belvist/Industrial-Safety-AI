import { useState } from "react";
import RiskStatistics from "./RiskStatistics";
import {
  AlertTriangle,
  ArrowRight,
  CheckCheck,
  ChevronLeft,
  ChevronRight,
  Download,
  Filter,
  Loader2,
  X,
} from "lucide-react";
import type { Camera, Incident } from "../types";
import { date, typeLabels } from "../types";
import { friendlyError, lifecycleNames, reviewNames, ruleTitle } from "./model";
import { authHeaders } from "../api";

type JournalProps = {
  items: Incident[];
  total: number;
  cameras: Camera[];
  loading: boolean;
  error: string;
  feedback: string;
  status: string;
  type: string;
  cameraId: string;
  fromDate: string;
  toDate: string;
  offset: number;
  onFeedback: (value: string) => void;
  onStatus: (value: string) => void;
  onType: (value: string) => void;
  onCamera: (value: string) => void;
  onFrom: (value: string) => void;
  onTo: (value: string) => void;
  onOffset: (value: number) => void;
  params: () => URLSearchParams;
  onOpen: (incident: Incident) => void;
  onConnect: () => void;
};
export default function Journal(props: JournalProps) {
  const [filters, setFilters] = useState(false),
    [exporting, setExporting] = useState(false),
    [exportError, setExportError] = useState("");
  const activeFilters = [
    props.status,
    props.type,
    props.cameraId,
    props.fromDate,
    props.toDate,
  ].filter(Boolean).length;
  async function download() {
    setExporting(true);
    setExportError("");
    try {
      const response = await fetch(
        `/api/v1/incidents/export.csv?${props.params()}`,
        { headers: authHeaders() },
      );
      if (!response.ok) {
        const data = await response
          .json()
          .catch(() => ({ detail: "Не удалось выгрузить журнал." }));
        throw new Error(data.detail);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `industrial-safety-${new Date().toISOString().slice(0, 10)}.csv`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setExportError(friendlyError(e));
    } finally {
      setExporting(false);
    }
  }
  return (
    <section className="p-journal">
      <div className="p-page-heading">
        <div>
          <span className="p-kicker">ИСТОРИЯ И ДОКАЗАТЕЛЬСТВА</span>
          <h1>Инциденты</h1>
          <p>Проверьте событие по снимку или видео и сохраните решение.</p>
        </div>
        <button
          className="p-button"
          disabled={exporting || props.loading || !props.total}
          onClick={() => void download()}
        >
          {exporting ? (
            <Loader2 size={16} className="spin" />
          ) : (
            <Download size={16} />
          )}
          Скачать CSV
        </button>
      </div>
      <div className="p-journal-toolbar">
        <div className="p-review-tabs" aria-label="Оценка оператора">
          {[
            ["unreviewed", "На проверке"],
            ["", "Все события"],
            ["confirmed", "Подтверждены"],
            ["false_positive", "Ложные"],
          ].map(([value, title]) => (
            <button
              key={value}
              className={props.feedback === value ? "selected" : ""}
              aria-pressed={props.feedback === value}
              onClick={() => props.onFeedback(value)}
            >
              {title}
            </button>
          ))}
        </div>
        <button
          className={`p-button ${filters ? "p-selected" : ""}`}
          aria-expanded={filters}
          onClick={() => setFilters(!filters)}
        >
          <Filter size={16} />
          Фильтры
          {!!activeFilters && <span className="p-count">{activeFilters}</span>}
        </button>
      </div>
      {filters && (
        <div className="p-journal-filters">
          <div className="p-field">
            <label htmlFor="incident-camera">Камера</label>
            <select
              id="incident-camera"
              value={props.cameraId}
              onChange={(e) => props.onCamera(e.target.value)}
            >
              <option value="">Все камеры</option>
              {props.cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="p-field">
            <label htmlFor="incident-type">Событие</label>
            <select
              id="incident-type"
              value={props.type}
              onChange={(e) => props.onType(e.target.value)}
            >
              <option value="">Все события</option>
              {Object.entries(typeLabels).map(([key, title]) => (
                <option value={key} key={key}>
                  {title}
                </option>
              ))}
            </select>
          </div>
          <div className="p-field">
            <label htmlFor="incident-status">Состояние наблюдения</label>
            <select
              id="incident-status"
              value={props.status}
              onChange={(e) => props.onStatus(e.target.value)}
            >
              <option value="">Любое состояние</option>
              {Object.entries(lifecycleNames).map(([key, title]) => (
                <option value={key} key={key}>
                  {title}
                </option>
              ))}
            </select>
          </div>
          <div className="p-field">
            <label htmlFor="date-from">С даты</label>
            <input
              id="date-from"
              type="date"
              value={props.fromDate}
              max={props.toDate || undefined}
              onChange={(e) => props.onFrom(e.target.value)}
            />
          </div>
          <div className="p-field">
            <label htmlFor="date-to">По дату</label>
            <input
              id="date-to"
              type="date"
              value={props.toDate}
              min={props.fromDate || undefined}
              onChange={(e) => props.onTo(e.target.value)}
            />
          </div>
          {!!activeFilters && (
            <button
              className="p-link"
              onClick={() => {
                props.onCamera("");
                props.onType("");
                props.onStatus("");
                props.onFrom("");
                props.onTo("");
              }}
            >
              <X size={14} />
              Сбросить
            </button>
          )}
        </div>
      )}
      {(props.error || exportError) && (
        <div className="p-inline-error" role="alert">
          {props.error || exportError}
        </div>
      )}
      <RiskStatistics
        cameras={props.cameras}
        cameraId={props.cameraId}
        fromDate={props.fromDate}
        toDate={props.toDate}
      />
      <div className="p-journal-results" aria-busy={props.loading}>
        {props.loading ? (
          <div className="p-empty">
            <Loader2 size={26} className="spin" />
            <h2>Загружаем журнал</h2>
          </div>
        ) : props.error ? (
          <div className="p-empty">
            <AlertTriangle size={28} />
            <h2>Журнал временно недоступен</h2>
            <p>
              Проверяем соединение. События появятся после восстановления связи.
            </p>
          </div>
        ) : props.items.length ? (
          <>
            <div className="p-result-count">
              Найдено событий: {props.total}
              <span>Выберите событие, чтобы посмотреть доказательство</span>
            </div>
            <div className="p-incident-table">
              <table>
                <thead>
                  <tr>
                    <th>Событие</th>
                    <th>Камера</th>
                    <th>Когда</th>
                    <th>Проверка оператором</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {props.items.map((item) => (
                    <tr key={item.id} onClick={() => props.onOpen(item)}>
                      <td>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            props.onOpen(item);
                          }}
                          className="p-event-title"
                        >
                          <span
                            className={`p-event-mark ${item.type === "danger_zone" ? "danger" : ""}`}
                          >
                            <AlertTriangle size={17} />
                          </span>
                          <span>
                            <strong>{ruleTitle(item.type)}</strong>
                            <small>{lifecycleNames[item.status]}</small>
                          </span>
                        </button>
                      </td>
                      <td>
                        {item.camera_name}
                        <small>
                          {item.zone_name ? `${item.zone_name} · ` : ""}
                          Сотрудник #{item.track_id}
                        </small>
                      </td>
                      <td>
                        {date(item.started_at)}
                        <small>{item.duration.toFixed(1)} сек.</small>
                      </td>
                      <td>
                        <span className={`p-review-state ${item.feedback}`}>
                          {reviewNames[item.feedback] ?? "На проверке"}
                        </span>
                      </td>
                      <td>
                        <ArrowRight size={16} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="p-empty">
            <CheckCheck size={31} />
            <h2>
              {activeFilters
                ? "Событий по этим условиям нет"
                : props.feedback === "unreviewed"
                  ? "Нет событий для проверки"
                  : "Пока нет подтверждённых событий"}
            </h2>
            <p>
              {activeFilters
                ? "Измените фильтры или выберите другой период."
                : "Когда система подтвердит нарушение, здесь появятся доказательство и время события."}
            </p>
            {!activeFilters && (
              <button className="p-button" onClick={props.onConnect}>
                Перейти к наблюдению
                <ArrowRight size={16} />
              </button>
            )}
          </div>
        )}
      </div>
      {props.total > 30 && (
        <div className="p-pagination">
          <span>
            {props.offset + 1}–{Math.min(props.offset + 30, props.total)} из{" "}
            {props.total}
          </span>
          <div>
            <button
              className="p-button"
              disabled={!props.offset || props.loading}
              onClick={() => props.onOffset(Math.max(0, props.offset - 30))}
            >
              <ChevronLeft size={15} />
              Назад
            </button>
            <button
              className="p-button"
              disabled={props.offset + 30 >= props.total || props.loading}
              onClick={() => props.onOffset(props.offset + 30)}
            >
              Далее
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
