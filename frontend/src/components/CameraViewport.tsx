import { useEffect, useState } from "react";
import { ScanLine, Video, Eye, EyeOff, Activity } from "lucide-react";
import type { Camera, Point, Result } from "../types";
import { labels, percentage } from "../types";
import { useAuthedImageUrl } from "../useAuthedImage";
export default function CameraViewport({
  camera,
  result,
  zones,
  activeIndex = 0,
  editing,
  points,
  onPoint,
  showBoxes,
  setShowBoxes,
  showRaw,
  setShowRaw,
}: {
  camera?: Camera;
  result?: Result | null;
  zones?: { id: string; name: string; enabled: boolean; points: Point[] }[];
  activeIndex?: number;
  editing: boolean;
  points: Point[];
  onPoint: (point: Point) => void;
  showBoxes: boolean;
  setShowBoxes: (value: boolean) => void;
  showRaw: boolean;
  setShowRaw: (value: boolean) => void;
}) {
  const [displayed, setDisplayed] = useState<Result | null>(null);
  useEffect(() => {
    setDisplayed(null);
  }, [camera?.id, camera?.session_id]);
  // When editing, show every draft zone; the active one is highlighted.
  const draftZones = editing ? zones ?? [] : [];
  const activeZone = editing ? points : camera?.zone || [];
  // Evidence frames load with auth headers into a Blob URL: the bearer
  // token never lands in an <img> URL, logs or Referer query strings.
  const frameSrc = useAuthedImageUrl(result ? result.frame_url : null);
  // By default only worker-owned objects: person boxes and PPE attached to a
  // worker (track_hint). Unassigned raw detections need the debug toggle.
  const ratio = result ? result.width / result.height : 16 / 9;
  const fittedWidth = Math.min(100, (ratio / (16 / 9)) * 100);
  const fittedHeight = Math.min(100, (16 / 9 / ratio) * 100);
  return (
    <>
      <div
        className="video-stage"
        style={{
          aspectRatio: "16/9",
        }}
      >
        <div
          className="frame-surface"
          style={{
            width: `${fittedWidth}%`,
            height: `${fittedHeight}%`,
            left: `${(100 - fittedWidth) / 2}%`,
            top: `${(100 - fittedHeight) / 2}%`,
          }}
        >
          {result ? (
            <img
              className="inference-frame"
              src={frameSrc || undefined}
              alt="Кадр камеры с распознанными объектами"
              onLoad={(e) => {
                if (frameSrc && e.currentTarget.src === frameSrc)
                  setDisplayed(result);
              }}
            />
          ) : (
            <div className="video-empty">
              <div className="camera-reticle">
                <ScanLine size={38} />
              </div>
              <span className="mono-label">ВИДЕОПОТОК / ОЖИДАНИЕ</span>
              <h3>Начните с вашей камеры.</h3>
              <p>
                Подключите устройство или загрузите видео.
                <br />
                Система покажет объекты и уверенность распознавания.
              </p>
              <div className="empty-source">
                <Video size={14} /> Веб-камера · MP4 · RTSP
              </div>
            </div>
          )}
          <svg
            className={"zone-overlay " + (editing ? "editing" : "")}
            viewBox="0 0 1000 1000"
            preserveAspectRatio="none"
            onClick={(e) => {
              if (!editing || points.length >= 30) return;
              const rect = e.currentTarget.getBoundingClientRect();
              onPoint([
                (e.clientX - rect.left) / rect.width,
                (e.clientY - rect.top) / rect.height,
              ]);
            }}
          >
            {editing
              ? draftZones.map((z, zi) => (
                  <g key={z.id || zi}>
                    {z.points.length > 1 && (
                      <polygon
                        points={z.points
                          .map(([x, y]) => `${x * 1000},${y * 1000}`)
                          .join(" ")}
                        fill={
                          z.enabled
                            ? zi === activeIndex
                              ? "rgba(251,191,36,.22)"
                              : "rgba(251,191,36,.08)"
                            : "rgba(148,163,184,.10)"
                        }
                        stroke={z.enabled ? "#eac36b" : "#94a3b8"}
                        strokeWidth={zi === activeIndex ? 3 : 1.5}
                        strokeDasharray={z.enabled ? "6 5" : "3 4"}
                      />
                    )}
                    {z.points.length > 1 && (
                      <text
                        x={z.points[0][0] * 1000 + 8}
                        y={z.points[0][1] * 1000 + 22}
                        fill={z.enabled ? "#8a6a1f" : "#64748b"}
                        fontSize="26"
                      >
                        {z.name}
                      </text>
                    )}
                  </g>
                ))
              : activeZone.length > 1 && (
                  <polygon
                    points={activeZone
                      .map(([x, y]) => `${x * 1000},${y * 1000}`)
                      .join(" ")}
                    fill="rgba(251,191,36,.13)"
                    stroke="#eac36b"
                    strokeWidth="2"
                    strokeDasharray="6 5"
                  />
                )}
            {editing &&
              activeZone.map(([x, y], i) => (
                <circle
                  key={i}
                  cx={x * 1000}
                  cy={y * 1000}
                  r="5"
                  fill="#eac36b"
                />
              ))}
          </svg>
          {showBoxes && !editing && displayed && (
            <div className="detection-overlay" aria-label="Работники, экипировка и прогноз">
              {/* Safety Radar: zone fill, trajectories, predicted path */}
              <svg
                className="radar-overlay"
                viewBox="0 0 1000 1000"
                preserveAspectRatio="none"
              >
                {(displayed.radar?.scene?.zones ?? []).map((z) => (
                  <polygon
                    key={z.name}
                    points={z.points.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ")}
                    fill="rgba(220,38,38,.10)"
                    stroke="rgba(220,38,38,.55)"
                    strokeWidth="2"
                  />
                ))}
                {(displayed.radar?.scene?.tracks ?? []).map((t) => {
                  const color =
                    t.level === "CRITICAL" ? "#ef4444"
                    : t.level === "WARNING" ? "#f59e0b"
                    : t.level === "UNKNOWN" ? "#94a3b8"
                    : "#22c55e";
                  const path = t.trajectory.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ");
                  const last = t.trajectory[t.trajectory.length - 1];
                  const end = t.predicted[t.predicted.length - 1];
                  if (!last) return null;
                  return (
                    <g key={t.track_id}>
                      {t.trajectory.length > 1 && (
                        <polyline points={path} fill="none" stroke={color} strokeWidth="3" opacity="0.75" />
                      )}
                      {end && (
                        <>
                          <line
                            x1={last[0] * 1000} y1={last[1] * 1000}
                            x2={end[0] * 1000} y2={end[1] * 1000}
                            stroke={color} strokeWidth="3" strokeDasharray="8 6" opacity="0.9"
                          />
                          <circle cx={end[0] * 1000} cy={end[1] * 1000} r="6" fill="none" stroke={color} strokeWidth="3" />
                        </>
                      )}
                      {t.intersection && (
                        <g>
                          <circle cx={t.intersection[0] * 1000} cy={t.intersection[1] * 1000} r="9" fill="none" stroke="#f59e0b" strokeWidth="4" />
                          <circle cx={t.intersection[0] * 1000} cy={t.intersection[1] * 1000} r="3" fill="#f59e0b" />
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Confirmed tracks with ID and PPE status */}
              {(displayed.tracks ?? []).map((t) => {
                const level = displayed.radar?.track_risks?.[t.id] ?? "UNKNOWN";
                const color =
                  level === "CRITICAL" ? "critical"
                  : level === "WARNING" ? "warning"
                  : level === "UNKNOWN" ? "unknown"
                  : "normal";
                const helmet = t.ppe?.helmet ?? "unknown";
                const vest = t.ppe?.vest ?? "unknown";
                const label = (v: string) =>
                  v === "present" ? "есть" : v === "missing" ? "НЕТ" : "неизвестно";
                return (
                  <div
                    key={t.id}
                    className={"track-box " + color}
                    style={{
                      left: `${t.bbox.x1 * 100}%`,
                      top: `${t.bbox.y1 * 100}%`,
                      width: `${(t.bbox.x2 - t.bbox.x1) * 100}%`,
                      height: `${(t.bbox.y2 - t.bbox.y1) * 100}%`,
                    }}
                  >
                    <span className="track-label">
                      <b>#{t.id}</b> · {level}
                      <br />
                      каска: {label(helmet)}
                      {" · "}жилет: {label(vest)}
                    </span>
                  </div>
                );
              })}

              {/* Diagnostics: raw/unassigned detections only in raw mode */}
              {showRaw && (displayed.detections ?? [])
                .filter((d) => d.track_hint == null)
                .map((d, index) => (
                  <div
                    key={`raw-${index}`}
                    className="detection-box raw"
                    style={{
                      left: `${d.bbox.x1 * 100}%`,
                      top: `${d.bbox.y1 * 100}%`,
                      width: `${(d.bbox.x2 - d.bbox.x1) * 100}%`,
                      height: `${(d.bbox.y2 - d.bbox.y1) * 100}%`,
                    }}
                  >
                    <span className="detection-label">
                      {labels[d.class_name] || d.class_name}
                      {d.weak ? " · слабая" : ""}
                      <b>{percentage(d.confidence)}</b>
                    </span>
                  </div>
                ))}
            </div>
          )}
        </div>
        <div className="video-topline">
          <span
            className={
              "video-live " + (camera?.status === "running" ? "on" : "")
            }
          >
            <i />
            {labels[camera?.status || "stopped"]}
          </span>
          {displayed && (
            <span className="frame-counter">
              КАДР {displayed.sequence.toString().padStart(5, "0")}
            </span>
          )}
        </div>
        {editing && (
          <div className="zone-hint">
            Отметьте границы опасной зоны · {points.length} точек
          </div>
        )}
      </div>
      <div className="video-toolbar">
        <div className="telemetry">
          <span>
            <Activity size={14} />
            {displayed?.fps || "—"} FPS
          </span>
          <span>{displayed?.latency_ms || "—"} мс</span>
          <span>{displayed?.tracks.length ?? 0} чел.</span>
        </div>
        <button
          className="text-button"
          onClick={() => setShowBoxes(!showBoxes)}
          aria-pressed={showBoxes}
        >
          {showBoxes ? <Eye size={15} /> : <EyeOff size={15} />}Объекты и
          проценты
        </button>
        <button
          className="text-button"
          onClick={() => setShowRaw(!showRaw)}
          aria-pressed={showRaw}
          title="Показать все сырые детекции, включая непривязанные к работникам"
        >
          {showRaw ? <Eye size={15} /> : <EyeOff size={15} />}Сырые детекции
        </button>
      </div>
    </>
  );
}
