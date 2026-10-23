export type Point = [number, number];
export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Detection = {
  class_name: string;
  confidence: number;
  bbox: Box;
  track_hint: string | null;
  weak: boolean;
};
export type Track = {
  id: number;
  bbox: Box;
  confidence: number;
  violations: string[];
  ppe: Record<string, string>;
  ppe_confidence: Record<string, number>;
};
export type Rule = {
  track_id: number;
  type: string;
  phase: "pending" | "active" | "clearing";
  confidence: number;
  elapsed_seconds: number;
  confirm_seconds: number;
};
export type Result = {
  session_id: string;
  sequence: number;
  frame_url: string;
  tracks: Track[];
  detections: Detection[];
  rules: Rule[];
  fps: number;
  latency_ms: number;
  active: number;
  width: number;
  height: number;
  timestamp: string;
};
export type Config = {
  name: string;
  helmet: boolean;
  vest: boolean;
  mask: boolean;
  smoking: boolean;
  zone_enabled: boolean;
  zone: Point[];
  confirm_seconds: number;
  clear_seconds: number;
  temporal_window: number;
  temporal_required: number;
};
export type CameraState =
  | "stopped"
  | "uploading"
  | "starting"
  | "running"
  | "stale"
  | "stopping"
  | "ended"
  | "error";
export type Camera = Config & {
  id: string;
  status: CameraState;
  session_id: string | null;
  state_version: number;
  error: string | null;
  kind: string | null;
  latest: Result | null;
};
export type Incident = {
  id: string;
  camera_name: string;
  camera_id: string;
  track_id: number;
  type: string;
  status: "active" | "resolved" | "interrupted";
  started_at: string;
  duration: number;
  confidence: number;
  snapshot: string | null;
  feedback: string;
  notification: string;
  resolution_reason: string | null;
};
export type InferenceInfo = {
  provider: string;
  mode: string;
  readiness?: string;
  accelerator_confirmed?: boolean;
  requested_providers?: string[];
  active_providers?: string[];
  supported_classes?: string[];
  model?: string;
  last_latency_ms?: number;
  frames?: number;
} | null;
export type Info = {
  model_ready: boolean;
  model_error: string | null;
  telegram_ready: boolean;
  target_fps: number;
  inference_provider: string;
  inference: InferenceInfo;
  available_providers: string[];
  public_url: string | null;
  stats: {
    total: number;
    active: number;
    today: number;
    false_positive: number;
  };
};
export type Page = "monitor" | "dashboard" | "incidents" | "cameras";
export const labels: Record<string, string> = {
  person: "Человек",
  helmet: "Каска",
  no_helmet: "Без каски",
  vest: "Жилет",
  no_vest: "Без жилета",
  mask: "Респиратор",
  no_mask: "Без респиратора",
  smoking: "Курение",
  danger_zone: "Опасная зона",
  active: "Активен",
  resolved: "Завершён",
  interrupted: "Прерван",
  running: "В эфире",
  stopped: "Остановлена",
  uploading: "Загрузка видео",
  starting: "Подключение",
  stopping: "Остановка",
  stale: "Нет новых кадров",
  ended: "Видео завершено",
  error: "Ошибка источника",
  pending: "Подтверждение",
  clearing: "Проверка завершения",
  violation_cleared: "Нарушение устранено",
  observation_lost: "Наблюдение потеряно",
  frame_gap: "Потеря источника",
  rules_changed: "Правила изменены",
  server_restarted: "Перезапуск сервера",
  processing_failed: "Ошибка обработки",
};
export const date = (v: string) =>
  new Date(v).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
export const percentage = (value: number) => `${Math.round(value * 100)}%`;
export const isActive = (camera?: Camera) =>
  !!camera &&
  ["running", "uploading", "starting", "stale", "stopping"].includes(
    camera.status,
  );
export function configOf(camera: Camera): Config {
  return {
    name: camera.name,
    helmet: camera.helmet,
    vest: camera.vest,
    mask: camera.mask,
    smoking: camera.smoking,
    zone_enabled: camera.zone_enabled,
    zone: camera.zone,
    confirm_seconds: camera.confirm_seconds,
    clear_seconds: camera.clear_seconds,
    temporal_window: camera.temporal_window,
    temporal_required: camera.temporal_required,
  };
}
