"""
Safety Radar — раннее предупреждение об опасных сближениях.

Работает поверх tracking, не зависит от аппаратного инференса (LinQ/CPU).
Использует реальные timestamps и координаты bbox center / foot point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from app.services.engine import Track


@dataclass
class TrajectoryPoint:
    """Точка траектории с timestamp."""
    x: float
    y: float
    t: float
    track_id: int
    class_name: str = "person"


@dataclass
class RiskEvent:
    """Событие риска для Evidence Replay."""
    track_id: int
    camera_id: str
    risk_type: str  # "zone_approach", "zone_cross", "worker_cross"
    level: str      # "WARNING" | "CRITICAL" | "UNKNOWN"
    timestamp: float
    message: str
    trajectory: list[tuple[float, float]]
    zone_name: str | None = None
    other_track_id: int | None = None


class SafetyRadar:
    """
    Анализ траекторий и раннее предупреждение.
    
    Уровни:
    - NORMAL: подтверждённой опасности нет
    - WARNING: наблюдается тенденция сближения с опасной зоной
    - CRITICAL: обнаружено подтверждённое нарушение (пересечение зоны)
    - UNKNOWN: трек потерян или данных недостаточно
    """
    
    def __init__(
        self,
        history_seconds: float = 5.0,
        zone_threshold_norm: float = 0.02,
        speed_threshold_norm_s: float = 0.5,
        min_history_points: int = 3,
        cooldown_seconds: float = 3.0,
    ):
        self.history_seconds = history_seconds
        self.zone_threshold_norm = zone_threshold_norm
        self.speed_threshold_norm_s = speed_threshold_norm_s
        self.min_history_points = min_history_points
        self.cooldown_seconds = cooldown_seconds
        
        # track_id -> deque of TrajectoryPoint
        self._trajectories: dict[int, deque] = {}
        # track_id -> last risk event time (for cooldown)
        self._last_risk_time: dict[int, float] = {}
        # track_id -> last risk level (escalation must not be suppressed)
        self._last_level: dict[int, str] = {}
        # Список событий для Evidence Replay
        self._events: list[RiskEvent] = []
        self._max_events: int = 1000  # Ограничение памяти
        # Зоны: zone_name -> Polygon (в пикселях кадра)
        self._zones: dict[str, Polygon] = {}
        
    def reset(self):
        """Сброс состояния: новые сессии, остановка камеры, изменение правил."""
        self._trajectories.clear()
        self._last_risk_time.clear()
        self._last_level.clear()
        self._events.clear()
        self._zones.clear()
    
    def set_zones(self, zones: dict[str, list[tuple[float, float]]]):
        """Установить зоны: zone_name -> list of (x, y) points."""
        self._zones = {name: Polygon(points) for name, points in zones.items()}
        
    def _foot_point(self, track: Track) -> tuple[float, float]:
        """Точка контакта ног с полом: центр по X, низ по Y (нормализованные [0,1])."""
        bbox = track.bbox
        x = (bbox.x1 + bbox.x2) / 2.0
        y = bbox.y2  # низ bbox, нормализованный [0,1]
        return x, y
        
    def update(self, tracks: list[Track], camera_id: str, frame_time: float) -> list[RiskEvent]:
        """Обновить траектории и вернуть новые события риска.
        
        tracks — уже подтверждённые person-треки от трекера.
        bbox нормализован [0,1], frame_time — временная метка кадра (time.monotonic или видео timestamp).
        """
        now = frame_time
        new_events = []
        active_ids = set()
        
        for track in tracks:
            # Radar работает только с person-треками (трекер уже фильтрует)
            active_ids.add(track.id)
            x, y = self._foot_point(track)  # normalized [0,1]
            
            # Обновить траекторию
            traj = self._trajectories.setdefault(track.id, deque())
            traj.append(TrajectoryPoint(x, y, now, track.id, "person"))
            
            # Удалить старые точки
            cutoff = now - self.history_seconds
            while traj and traj[0].t < cutoff:
                traj.popleft()
                
            # Анализ риска только если достаточно истории
            if len(traj) >= self.min_history_points:
                risk_events = self._analyze_track(track, traj, camera_id, now)
                for ev in risk_events:
                    # Cooldown: first event always passes; escalation to
                    # CRITICAL is never suppressed (a warning must be able
                    # to become a confirmed violation immediately).
                    last = self._last_risk_time.get(track.id)
                    escalate = ev.level == "CRITICAL" and self._last_level.get(track.id) != "CRITICAL"
                    if last is None or escalate or now - last >= self.cooldown_seconds:
                        self._last_risk_time[track.id] = now
                        self._last_level[track.id] = ev.level
                        self._events.append(ev)
                        new_events.append(ev)
                        # Ограничение размера списка событий
                        if len(self._events) > self._max_events:
                            self._events = self._events[-self._max_events:]
        
        # Удалить траектории потерянных треков
        lost_ids = set(self._trajectories.keys()) - active_ids
        for lost_id in lost_ids:
            del self._trajectories[lost_id]
            
        return new_events
    
    def _analyze_track(
        self, track: Track, traj: deque, camera_id: str, now: float
    ) -> list[RiskEvent]:
        events = []
        
        # 1. Скорость (нормализованные единицы/сек)
        speed = self._estimate_speed(traj)
        
        # 2. Проверка зон (координаты нормализованы [0,1])
        last_pt = traj[-1]
        pt = Point(last_pt.x, last_pt.y)
        
        for zone_name, zone_poly in self._zones.items():
            dist = pt.distance(zone_poly)
            inside = zone_poly.contains(pt)
            
            if inside:
                # CRITICAL: пересечение зоны
                events.append(RiskEvent(
                    track_id=track.id,
                    camera_id=camera_id,
                    risk_type="zone_cross",
                    level="CRITICAL",
                    timestamp=now,
                    message=f"Работник #{track.id} вошёл в зону «{zone_name}»",
                    trajectory=[(p.x, p.y) for p in traj],
                    zone_name=zone_name,
                ))
            elif dist < self.zone_threshold_norm:
                # WARNING: приближение к зоне
                # Проверяем направление движения
                if self._moving_towards_zone(traj, zone_poly):
                    events.append(RiskEvent(
                        track_id=track.id,
                        camera_id=camera_id,
                        risk_type="zone_approach",
                        level="WARNING",
                        timestamp=now,
                        message=(
                            f"Работник #{track.id} приближается к зоне «{zone_name}» "
                            f"(дистанция {dist:.3f} норм., скорость {speed:.3f} норм./с)"
                        ),
                        trajectory=[(p.x, p.y) for p in traj],
                        zone_name=zone_name,
                    ))
        
        # 3. Пересечение траекторий двух людей (опционально)
        # Можно добавить проверку сближения треков
        
        return events
    
    def _estimate_speed(self, traj: deque) -> float:
        """Скорость в нормализованных единицах/сек по последним 3 точкам."""
        if len(traj) < 3:
            return 0.0
        p1, p3 = traj[-3], traj[-1]
        dt = p3.t - p1.t
        if dt <= 0:
            return 0.0
        dx = p3.x - p1.x
        dy = p3.y - p1.y
        return np.hypot(dx, dy) / dt
    
    def _moving_towards_zone(self, traj: deque, zone: Polygon) -> bool:
        """Проверка, что вектор движения указывает к зоне (нормализованные координаты)."""
        if len(traj) < 2:
            return False
        p1 = traj[-2]
        p2 = traj[-1]
        # Вектор движения
        vx = p2.x - p1.x
        vy = p2.y - p1.y
        if vx == 0 and vy == 0:
            return False
        # Вектор от последней точки к центру зоны
        zone_center = zone.centroid
        ux = zone_center.x - p2.x
        uy = zone_center.y - p2.y
        # Косинус угла
        dot = vx * ux + vy * uy
        norm_v = np.hypot(vx, vy)
        norm_u = np.hypot(ux, uy)
        if norm_v == 0 or norm_u == 0:
            return False
        cos_angle = dot / (norm_v * norm_u)
        return cos_angle > 0.3  # ~70 градусов
    
    def get_events(self, since: float = 0) -> list[RiskEvent]:
        """Получить последние события, по одному на (track, risk_type).

        Полная история остаётся в self._events для аналитики; API отдаёт
        только актуальные, чтобы UI не дублировал одну и ту же тревогу.
        """
        latest: dict[tuple[int, str], RiskEvent] = {}
        for ev in self._events:
            if ev.timestamp >= since:
                latest[(ev.track_id, ev.risk_type)] = ev
        return sorted(latest.values(), key=lambda e: e.timestamp)
    
    def get_current_risk_level(self, track_id: int, frame_time: float = 0.0) -> str:
        """Текущий уровень риска для трека (для UI).
        
        frame_time — текущая временная метка кадра (time.monotonic или видео timestamp).
        Если не передан, использует timestamp последнего события.
        """
        if track_id not in self._trajectories:
            return "UNKNOWN"
        traj = self._trajectories[track_id]
        if len(traj) < self.min_history_points:
            return "NORMAL"
        # Проверить недавние события на той же временной шкале
        now = frame_time if frame_time > 0 else (self._events[-1].timestamp if self._events else 0)
        for ev in reversed(self._events):
            if ev.track_id == track_id and now - ev.timestamp < self.cooldown_seconds:
                return ev.level
        return "NORMAL"

