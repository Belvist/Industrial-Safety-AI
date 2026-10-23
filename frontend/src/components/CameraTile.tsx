import { useAuthedImageUrl } from "../useAuthedImage";
import { cameraAlert } from "../monitor";
import { labels } from "../types";
import type { Camera } from "../types";

export default function CameraTile({
  camera,
  selected,
  onSelect,
}: {
  camera: Camera;
  selected: boolean;
  onSelect: () => void;
}) {
  const img = useAuthedImageUrl(camera.latest?.frame_url ?? null);
  const alert = cameraAlert(camera);
  const live = camera.status === "running" || camera.status === "stale";
  return (
    <button
      className={
        "camera-tile" + (selected ? " selected" : "") + (alert ? " alarming" : "")
      }
      onClick={onSelect}
      aria-label={`Камера ${camera.name}`}
    >
      <span className="tile-shot">
        {img ? (
          <img src={img} alt="" />
        ) : (
          <span className="tile-empty">{live ? "…" : "стоп"}</span>
        )}
        {alert && (
          <span className={"tile-badge " + alert.level}>
            {labels[alert.type] || alert.type} · #{alert.trackId}
          </span>
        )}
      </span>
      <span className="tile-meta">
        <strong>{camera.name}</strong>
        <small className={"status-dot " + camera.status} />
      </span>
    </button>
  );
}
