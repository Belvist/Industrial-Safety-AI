import { useEffect, useState } from "react";
import {
  Check,
  Loader2,
  Plus,
  RotateCcw,
  ScanLine,
  Trash2,
  Undo2,
} from "lucide-react";
import Dialog from "./Dialog";
import CameraViewport from "../components/CameraViewport";
import type { Camera, Point, Result, Zone } from "../types";
import { friendlyError, zonesOf } from "./model";

function newZone(zones: Zone[]): Zone {
  let n = zones.length + 1;
  while (zones.some((z) => z.name === `Опасная зона ${n}`)) n++;
  return {
    id: crypto.randomUUID().replaceAll("-", "").slice(0, 32),
    name: `Опасная зона ${n}`,
    enabled: true,
    points: [],
  };
}
export default function ZoneEditor({
  camera,
  result,
  onClose,
  onSave,
}: {
  camera: Camera;
  result: Result;
  onClose: () => void;
  onSave: (zones: Zone[]) => Promise<void>;
}) {
  const [zones, setZones] = useState<Zone[]>(() => {
    const existing = zonesOf(camera).map((z) => ({
      ...z,
      points: [...z.points],
    }));
    return existing.length ? existing : [newZone([])];
  });
  const [index, setIndex] = useState(0),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const current = zones[index];
  function update(patch: Partial<Zone>) {
    setZones(zones.map((z, i) => (i === index ? { ...z, ...patch } : z)));
    setError("");
  }
  function point(value: Point) {
    if (current && current.points.length < 30)
      update({ points: [...current.points, value] });
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.matches("input,textarea"))
        return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (current) update({ points: current.points.slice(0, -1) });
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [zones, index]);
  async function save() {
    if (zones.some((z) => z.points.length < 3)) {
      setError(
        "У каждой зоны должно быть не меньше трёх точек. Незавершённую зону можно удалить.",
      );
      return;
    }
    if (zones.some((z) => !z.name.trim())) {
      setError("Укажите название каждой зоны.");
      return;
    }
    if (new Set(zones.map((z) => z.name.trim())).size !== zones.length) {
      setError("У каждой зоны должно быть своё название.");
      return;
    }
    setSaving(true);
    try {
      await onSave(zones.map((z) => ({ ...z, name: z.name.trim() })));
      onClose();
    } catch (e) {
      setError(friendlyError(e));
    } finally {
      setSaving(false);
    }
  }
  return (
    <Dialog
      title="Опасные зоны"
      description={camera.name}
      onClose={() => {
        if (!saving) onClose();
      }}
      className="p-zone-dialog"
      footer={
        <>
          <span className="p-hint">
            Изменения применятся только после сохранения.
          </span>
          <div>
            <button className="p-button" disabled={saving} onClick={onClose}>
              Отмена
            </button>
            <button
              className="p-button p-primary"
              disabled={saving}
              onClick={() => void save()}
            >
              {saving ? (
                <Loader2 className="spin" size={16} />
              ) : (
                <Check size={16} />
              )}
              Сохранить зоны
            </button>
          </div>
        </>
      }
    >
      <div className="p-zone-layout">
        <div className="p-zone-canvas">
          <div className="p-zone-instruction">
            <ScanLine size={18} />
            <span>
              {current
                ? current.points.length < 3
                  ? "Нажмите на изображение минимум в трёх местах, чтобы очертить границу."
                  : "Перетаскивайте точки, чтобы уточнить границу. Можно добавить ещё точки."
                : "Создайте зону, чтобы отметить её границы."}
            </span>
          </div>
          <div className="p-video-surface">
            <CameraViewport
              camera={camera}
              result={result}
              zones={zones}
              activeIndex={index}
              editing={true}
              points={current?.points ?? []}
              onPoint={point}
              onMovePoint={(pointIndex, value) =>
                current &&
                update({
                  points: current.points.map((p, i) =>
                    i === pointIndex ? value : p,
                  ),
                })
              }
              showBoxes={false}
              setShowBoxes={() => {}}
              showRaw={false}
              setShowRaw={() => {}}
              minimal
            />
          </div>
          <div className="p-zone-drawing-tools">
            <span>{current?.points.length ?? 0} / 30 точек</span>
            <button
              className="p-link"
              disabled={!current?.points.length}
              onClick={() =>
                current && update({ points: current.points.slice(0, -1) })
              }
            >
              <Undo2 size={16} />
              Убрать точку
            </button>
            <button
              className="p-link"
              disabled={!current?.points.length}
              onClick={() => update({ points: [] })}
            >
              <RotateCcw size={15} />
              Начать заново
            </button>
          </div>
        </div>
        <aside className="p-zone-sidebar">
          <div className="p-section-title">
            <h3>Зоны камеры</h3>
            <span>{zones.length}/10</span>
          </div>
          <div className="p-zone-list">
            {zones.map((z, i) => (
              <button
                className={i === index ? "selected" : ""}
                key={z.id}
                onClick={() => setIndex(i)}
              >
                <ScanLine size={16} />
                <span>
                  {z.name || "Без названия"}
                  <small>
                    {z.points.length < 3
                      ? "Нужно нарисовать границу"
                      : z.enabled
                        ? "Контроль включён"
                        : "Контроль выключен"}
                  </small>
                </span>
              </button>
            ))}
          </div>
          <button
            className="p-button"
            disabled={zones.length >= 10}
            onClick={() => {
              setZones([...zones, newZone(zones)]);
              setIndex(zones.length);
            }}
          >
            <Plus size={16} />
            Новая зона
          </button>
          {current && (
            <div className="p-zone-properties">
              <div className="p-field">
                <label htmlFor="zone-name">Название выбранной зоны</label>
                <input
                  id="zone-name"
                  maxLength={80}
                  value={current.name}
                  onChange={(e) => update({ name: e.target.value })}
                />
              </div>
              <label className="p-rule-choice">
                <strong>Контролировать вход</strong>
                <input
                  type="checkbox"
                  checked={current.enabled}
                  onChange={(e) => update({ enabled: e.target.checked })}
                />
              </label>
              <button
                className="p-link p-danger-text"
                onClick={() => {
                  setZones(zones.filter((_, i) => i !== index));
                  setIndex(Math.max(0, index - 1));
                }}
              >
                <Trash2 size={15} />
                Удалить зону
              </button>
            </div>
          )}
          <p className="p-hint">
            Проверяется точка под ногами человека. Зона должна соответствовать
            реальному участку на изображении.
          </p>
        </aside>
      </div>
      {error && (
        <div className="p-inline-error" role="alert">
          {error}
        </div>
      )}
    </Dialog>
  );
}
