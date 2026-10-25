import { useEffect, useState } from "react";
import { api } from "../api";
import type { Camera } from "../types";
import { friendlyError, ruleTitle } from "./model";
type Count = {
  camera_id: string;
  zone_name: string;
  day: string;
  count: number;
  type?: string;
  kind?: string;
  confirmed?: number;
  false_positive?: number;
};
export default function RiskStatistics({
  cameras,
  cameraId,
  fromDate,
  toDate,
}: {
  cameras: Camera[];
  cameraId: string;
  fromDate: string;
  toDate: string;
}) {
  const [open, setOpen] = useState(false),
    [data, setData] = useState<{
      incidents: Count[];
      warnings: Count[];
    } | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setData(null);
    setError("");
    const params = new URLSearchParams();
    if (cameraId) params.set("camera_id", cameraId);
    if (fromDate)
      params.set("from_time", new Date(`${fromDate}T00:00:00`).toISOString());
    if (toDate)
      params.set("to_time", new Date(`${toDate}T23:59:59.999`).toISOString());
    api<{ incidents: Count[]; warnings: Count[] }>(
      `/analytics/risk?${params}`,
      { signal: controller.signal },
    )
      .then(setData)
      .catch((e) => {
        if (!controller.signal.aborted) setError(friendlyError(e));
      });
    return () => controller.abort();
  }, [open, cameraId, fromDate, toDate]);
  const name = (id: string) => cameras.find((c) => c.id === id)?.name ?? id;
  return (
    <details
      className="p-advanced"
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>Статистика по камерам и зонам</summary>
      <p>
        Сводка по выбранной камере и периоду, включая все типы и решения. Это
        количество записей в журнале, а не вероятность аварии.
      </p>
      {error ? (
        <p role="alert">{error}</p>
      ) : !data ? (
        <p>Загружаем статистику…</p>
      ) : (
        <>
          <div className="p-incident-table">
            <table>
              <thead>
                <tr>
                  <th>Камера / зона</th>
                  <th>Дата / событие</th>
                  <th>Инциденты</th>
                  <th>Оценка оператора</th>
                </tr>
              </thead>
              <tbody>
                {data.incidents.map((r, i) => (
                  <tr key={i}>
                    <td>
                      {name(r.camera_id)}
                      <small>{r.zone_name || "Без зоны"}</small>
                    </td>
                    <td>
                      {r.day}
                      <small>{ruleTitle(r.type ?? "")}</small>
                    </td>
                    <td>{r.count}</td>
                    <td>
                      Подтверждено: {r.confirmed} · ложных: {r.false_positive}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!data.incidents.length && <p>Инцидентов за этот период нет.</p>}
          {data.warnings.length > 0 && (
            <ul>
              {data.warnings.map((r, i) => (
                <li key={i}>
                  {name(r.camera_id)} · {r.zone_name} · {r.day}:{" "}
                  {r.kind === "warning"
                    ? "ранних предупреждений"
                    : "входов в зону"}{" "}
                  — {r.count}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </details>
  );
}
