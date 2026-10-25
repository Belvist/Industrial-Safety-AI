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
  ppe_age_seconds?: number;
};
export type Rule = {
  track_id: number;
  type: string;
  phase: "pending" | "active" | "clearing";
  confidence: number;
  elapsed_seconds: number;
  confirm_seconds: number;
};
export type RadarEvent = {
  track_id: number;
  camera_id: string;
  risk_type: "zone_approach" | "zone_cross" | "worker_cross";
  level: "WARNING" | "CRITICAL" | "UNKNOWN";
  timestamp: number;
  source_seconds?: number;
  lead_time_seconds?: number | null;
  predicted_entry_seconds?: number | null;
  message: string;
  zone_name?: string | null;
  other_track_id?: number | null;
  evidence_url?: string | null;
};
export type RadarTrack = {
  track_id: number;
  level: "NORMAL" | "WARNING" | "CRITICAL" | "UNKNOWN";
  trajectory: [number, number][];
  predicted: [number, number][];
  direction: [number, number] | null;
  intersection: [number, number] | null;
  confidence: number;
};
export type RadarScene = {
  tracks: RadarTrack[];
  zones: { name: string; points: [number, number][] }[];
};
export type Radar = {
  track_risks: Record<number, "NORMAL" | "WARNING" | "CRITICAL" | "UNKNOWN">;
  events: RadarEvent[];
  scene?: RadarScene | null;
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
  radar?: Radar | null;
  metrics?: {
    window_seconds: number;
    rates: Record<string, { fps: number; total: number; window_count: number }>;
    stages: Record<string, { p50: number; p95: number; p99: number }>;
  };
  data_age_ms?: number;
  person_fps?: number;
  radar_fps?: number;
  tpu_fps?: number;
  ppe?: {
    state: string;
    age_seconds: number | null;
    pending: boolean;
    runtime?: "cpu" | "tpu";
    error?: string | null;
  } | null;
  observation_quality?: {
    reasons: string[];
    tracks: Record<string, { ppe_observable: boolean; reasons: string[] }>;
  };
};
export type Zone = {
  id: string;
  name: string;
  enabled: boolean;
  points: Point[];
};
export type Config = {
  name: string;
  helmet: boolean;
  vest: boolean;
  mask: boolean;
  smoking: boolean;
  zone_enabled: boolean;
  zone_name?: string;
  running_enabled?: boolean;
  running_speed?: number;
  running_min_seconds?: number;
  zone: Point[];
  zones?: Zone[];
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
  frame_capacity?: number;
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
  zone_name?: string | null;
  evidence?: string | null;
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
  protocol?: string | null;
  program?: string | null;
  ppe_runtime?: string | null;
  person_detector?: string | null;
  worker_verified?: boolean;
  hardware_verified?: boolean;
  worker_pid?: string;
  remote_calls?: number;
  remote_failures?: number;
  max_in_flight?: number;
  worker_inference_ms?: number;
  metrics?: {
    window_seconds?: number;
    rates?: Record<string, { fps?: number; total?: number; window_count?: number }>;
    stages?: Record<string, { p50?: number; p95?: number; p99?: number }>;
  };
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
    unreviewed?: number;
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
export const typeLabels: Record<string, string> = {
  no_helmet: "Без каски",
  no_vest: "Без жилета",
  no_mask: "Без респиратора",
  smoking: "Курение",
  danger_zone: "Опасная зона",
  running: "Бег по цеху",
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
    zone_name: camera.zone_name ?? "Опасная зона",
    running_enabled: camera.running_enabled ?? false,
    running_speed: camera.running_speed ?? 1.8,
    running_min_seconds: camera.running_min_seconds ?? 0.8,
    zone: camera.zone,
    zones: camera.zones ?? [],
    confirm_seconds: camera.confirm_seconds,
    clear_seconds: camera.clear_seconds,
    temporal_window: camera.temporal_window,
    temporal_required: camera.temporal_required,
  };
}

export type TimelineEvent = {
  id: number;
  kind: string;
  source_seconds: number | null;
  occurred_at: string;
  zone_name: string | null;
  payload: {
    lead_time_seconds?: number | null;
    verdict?: string;
    reason?: string;
  };
};
