import { useEffect, useState } from "react";
import { ScanLine, Video, Eye, EyeOff, Activity } from "lucide-react";
import type { Camera, Detection, Point, Result } from "../types";
import { labels, percentage } from "../types";
import { useAuthedImageUrl } from "../useAuthedImage";
export default function CameraViewport({
  camera,
  result,
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
  const zone = editing ? points : camera?.zone || [];
  const boxes = displayed?.detections || [];
  // Evidence frames load with auth headers into a Blob URL: the bearer
  // token never lands in an <img> URL, logs or Referer query strings.
  const frameSrc = useAuthedImageUrl(result ? result.frame_url : null);
  // By default only worker-owned objects: person boxes and PPE attached to a
  // worker (track_hint). Unassigned raw detections need the debug toggle.
  const visible = showRaw
    ? boxes
    : boxes.filter(
        (d) => d.class_name === "person" || d.track_hint != null,
      );
  const ratio = result ? result.width / result.height : 16 / 9;
  const fittedWidth = Math.min(100, (ratio / (16 / 9)) * 100);
  const fittedHeight = Math.min(100, (16 / 9 / ratio) * 100);
  const identify = (d: Detection) =>
    d.class_name === "person"
      ? "person"
      : d.class_name.startsWith("no_")
        ? "violation"
        : "ppe";
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
            {zone.length > 1 && (
              <polygon
                points={zone
                  .map(([x, y]) => `${x * 1000},${y * 1000}`)
                  .join(" ")}
                fill="rgba(251,191,36,.13)"
                stroke="#eac36b"
                strokeWidth="2"
                strokeDasharray="6 5"
              />
            )}
            {editing &&
              zone.map(([x, y], i) => (
                <circle
                  key={i}
                  cx={x * 1000}
                  cy={y * 1000}
                  r="5"
                  fill="#eac36b"
                />
              ))}
          </svg>
          {showBoxes && !editing && (
            <div
              className="detection-overlay"
              aria-label="Распознанные объекты"
            >
              {visible.map((d, index) => {
                const worker =
                  d.class_name === "person"
                    ? displayed?.tracks.find(
                        (t) =>
                          Math.abs(t.bbox.x1 - d.bbox.x1) < 0.0001 &&
                          Math.abs(t.bbox.y1 - d.bbox.y1) < 0.0001,
                      )
                    : null;
                return (
                  <div
                    key={index}
                    className={"detection-box " + identify(d)}
                    style={{
                      left: `${d.bbox.x1 * 100}%`,
                      top: `${d.bbox.y1 * 100}%`,
                      width: `${(d.bbox.x2 - d.bbox.x1) * 100}%`,
                      height: `${(d.bbox.y2 - d.bbox.y1) * 100}%`,
                    }}
                  >
                    <span
                      className="detection-label"
                      style={
                        d.bbox.x1 > 0.72
                          ? { left: "auto", right: -1 }
                          : undefined
                      }
                    >
                      {labels[d.class_name] || d.class_name}
                      {worker ? ` #${worker.id}` : ""}
                      {d.weak ? " · слабая" : ""}
                      <b>{percentage(d.confidence)}</b>
                    </span>
                  </div>
                );
              })}
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
