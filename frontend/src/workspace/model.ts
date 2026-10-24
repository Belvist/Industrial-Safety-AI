import { typeLabels } from "../types";
import type { Camera, Info, Result, Zone } from "../types";

export const pageNames = {
  dashboard: "Наблюдение",
  monitor: "Наблюдение",
  incidents: "Инциденты",
  cameras: "Камеры",
  system: "Система",
};
export type WorkspacePage = keyof typeof pageNames;
export type SourceChoice = "browser" | "video" | "rtsp";
export type ConnectionDraft = {
  name: string;
  source: SourceChoice;
  device: string;
  file: File | null;
  url: string;
  helmet: boolean;
  vest: boolean;
};
export const sourceNames: Record<string, string> = {
  browser: "Веб-камера",
  video: "Видеофайл",
  rtsp: "IP-камера",
};
export const reviewNames: Record<string, string> = {
  unreviewed: "На проверке",
  confirmed: "Подтверждено",
  false_positive: "Ложное срабатывание",
};
export const lifecycleNames: Record<string, string> = {
  active: "Продолжается",
  resolved: "Завершён",
  interrupted: "Наблюдение прервано",
};
export function ruleTitle(type: string) {
  return typeLabels[type] || type;
}
export function zonesOf(camera: Camera): Zone[] {
  return camera.zones?.length
    ? camera.zones
    : camera.zone.length >= 3
      ? [
          {
            id: "legacy",
            name: camera.zone_name || "Опасная зона",
            enabled: camera.zone_enabled,
            points: camera.zone,
          },
        ]
      : [];
}
export function supports(
  info: Info | null,
  key: "helmet" | "vest" | "mask" | "smoking",
) {
  const required = {
    helmet: "no_helmet",
    vest: "no_vest",
    mask: "no_mask",
    smoking: "smoking",
  };
  const classes = info?.inference?.supported_classes;
  return classes
    ? classes.includes(required[key])
    : key === "helmet" || key === "vest";
}
export function observation(
  camera: Camera,
  result: Result | null | undefined,
  online: boolean,
) {
  if (!online)
    return {
      tone: "neutral",
      title: "Нет связи с приложением",
      description: "Переподключаемся. Текущее состояние камеры неизвестно.",
    };
  if (camera.status === "error")
    return {
      tone: "warning",
      title: "Камеру нужно переподключить",
      description:
        "Источник или распознавание недоступны. Проверьте подключение.",
    };
  if (camera.status === "ended")
    return {
      tone: "neutral",
      title: "Анализ файла завершён",
      description: "Подтверждённые события сохранены в журнале этой камеры.",
    };
  if (camera.status === "stopped")
    return {
      tone: "neutral",
      title: "Наблюдение остановлено",
      description: "Подключите источник, чтобы продолжить работу.",
    };
  if (camera.status === "uploading")
    return {
      tone: "neutral",
      title: "Передаём видео для анализа",
      description: "Дождитесь завершения загрузки.",
    };
  if (camera.status === "stopping")
    return {
      tone: "neutral",
      title: "Останавливаем наблюдение",
      description: "Завершаем текущую обработку.",
    };
  if (camera.status === "starting" || !result)
    return {
      tone: "neutral",
      title: "Ожидаем первый кадр",
      description: "Камера подключается. Распознавание начнётся автоматически.",
    };
  if (camera.status === "stale" || (result.data_age_ms ?? 0) > 5000)
    return {
      tone: "warning",
      title: "Нет свежих кадров",
      description:
        "Последний кадр не отражает текущее состояние. Проверьте источник.",
    };
  const confirmed = result.rules.filter(
    (r) => r.phase === "active" || r.phase === "clearing",
  );
  if (confirmed.length)
    return {
      tone: "danger",
      title:
        confirmed.length === 1
          ? "Нарушение подтверждено"
          : `Подтверждено нарушений: ${confirmed.length}`,
      description: [...new Set(confirmed.map((r) => ruleTitle(r.type)))].join(
        " · ",
      ),
    };
  const risks = Object.values(result.radar?.track_risks ?? {});
  if (risks.includes("CRITICAL"))
    return {
      tone: "danger",
      title: "Человек в опасной зоне",
      description:
        "Радар зафиксировал вход. Событие проходит подтверждение во времени.",
    };
  if (risks.includes("WARNING"))
    return {
      tone: "warning",
      title: "Человек приближается к опасной зоне",
      description:
        "Прогноз движения указывает на возможное пересечение границы.",
    };
  if (result.rules.some((r) => r.phase === "pending"))
    return {
      tone: "warning",
      title: "Проверяем возможное нарушение",
      description:
        "Система проверяет событие во времени, прежде чем записать инцидент.",
    };
  if (risks.includes("UNKNOWN"))
    return {
      tone: "warning",
      title: "Оценка риска пока неполная",
      description:
        "Для части треков недостаточно свежих наблюдений. Это не подтверждение безопасности.",
    };
  return {
    tone: "live",
    title: "Наблюдение работает",
    description: result.tracks.length
      ? `Людей в кадре: ${result.tracks.length}. Подтверждённых нарушений нет.`
      : "Людей в кадре пока не обнаружено.",
  };
}
export function friendlyError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const translations: [string, string][] = [
    [
      "Zone edges must not intersect",
      "Граница зоны пересекает сама себя. Переместите точки и сохраните ещё раз.",
    ],
    [
      "Zone area must be greater",
      "Зона слишком мала. Расставьте точки так, чтобы они ограничивали область.",
    ],
    ["Zone names must be unique", "У каждой зоны должно быть своё название."],
    [
      "duplicate adjacent points",
      "Две соседние точки совпадают. Удалите одну из них.",
    ],
    [
      "Failed to fetch",
      "Не удалось связаться с приложением. Проверьте, что оно запущено.",
    ],
    [
      "Permission denied",
      "Доступ к камере закрыт. Разрешите камеру в настройках браузера и повторите подключение.",
    ],
    [
      "NotAllowedError",
      "Разрешите доступ к камере в браузере и попробуйте ещё раз.",
    ],
    [
      "Could not start video source",
      "Камера занята другой программой. Освободите её и повторите подключение.",
    ],
  ];
  return (
    translations.find(([fragment]) => message.includes(fragment))?.[1] ??
    message
  );
}
