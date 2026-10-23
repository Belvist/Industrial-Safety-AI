"""
Safety Radar — раннее предупреждение об опасных сближениях.

Работает поверх tracking, не зависит от аппаратного инференса (LinQ/CPU).
Использует реальные timestamps и координаты bbox center / foot point.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

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
    trajectory: List[Tuple[float, float]]
    zone_name: Optional[str] = None
    other_track_id: Optional[int] = None


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
        zone_threshold_px: float = 50.0,
        speed_threshold_px_s: float = 80.0,
        min_history_points: int = 3,
        cooldown_seconds: float = 3.0,
    ):
        self.history_seconds = history_seconds
        self.zone_threshold_px = zone_threshold_px
        self.speed_threshold_px_s = speed_threshold_px_s
        self.min_history_points = min_history_points
        self.cooldown_seconds = cooldown_seconds
        
        # track_id -> deque of TrajectoryPoint
        self._trajectories: Dict[int, deque] = {}
        # track_id -> last risk event time (for cooldown)
        self._last_risk_time: Dict[int, float] = {}
        # Список событий для Evidence Replay
        self._events: List[RiskEvent] = []
        # Зоны: zone_name -> Polygon (в пикселях кадра)
        self._zones: Dict[str, Polygon] = {}
        
    def set_zones(self, zones: Dict[str, List[Tuple[float, float]]]):
        """Установить зоны: zone_name -> list of (x, y) points."""
        self._zones = {name: Polygon(points) for name, points in zones.items()}
        
    def _foot_point(self, track: Track) -> Tuple[float, float]:
        """Точка контакта ног с полом: центр по X, низ по Y."""
        bbox = track.bbox
        x = (bbox.x1 + bbox.x2) / 2.0
        y = bbox.y2  # низ bbox
        return x, y
        
    def update(self, tracks: List[Track], camera_id: str, frame_time: float, 
                 track_classes: Optional[Dict[int, str]] = None) -> List[RiskEvent]:
        """Обновить траектории и вернуть новые события риска."""
        now = frame_time
        new_events = []
        active_ids = set()
        
        for track in tracks:
            cls_name = track_classes.get(track.id, "") if track_classes else ""
            if cls_name != "person":
                continue
            active_ids.add(track.id)
            x, y = self._foot_point(track)
            
            # Обновить траекторию
            traj = self._trajectories.setdefault(track.id, deque())
            traj.append(TrajectoryPoint(x, y, now, track.id, cls_name))
            
            # Удалить старые точки
            cutoff = now - self.history_seconds
            while traj and traj[0].t < cutoff:
                traj.popleft()
                
            # Анализ риска только если достаточно истории
            if len(traj) >= self.min_history_points:
                risk_events = self._analyze_track(track, traj, camera_id, now)
                for ev in risk_events:
                    # Cooldown
                    last = self._last_risk_time.get(track.id, 0)
                    if now - last >= self.cooldown_seconds:
                        self._last_risk_time[track.id] = now
                        self._events.append(ev)
                        new_events.append(ev)
        
        # Удалить траектории потерянных треков (пометить UNKNOWN)
        lost_ids = set(self._trajectories.keys()) - active_ids
        for lost_id in lost_ids:
            # Можно оставить траекторию на некоторое время или удалить сразу
            # Здесь удаляем сразу, но событие UNKNOWN не генерируем (нет данных)
            del self._trajectories[lost_id]
            
        return new_events
    
    def _analyze_track(
        self, track: Track, traj: deque, camera_id: str, now: float
    ) -> List[RiskEvent]:
        events = []
        
        # 1. Скорость (по последним 3 точкам)
        speed = self._estimate_speed(traj)
        
        # 2. Проверка зон
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
            elif dist < self.zone_threshold_px:
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
                            f"(дистанция {dist:.0f} px, скорость {speed:.1f} px/s)"
                        ),
                        trajectory=[(p.x, p.y) for p in traj],
                        zone_name=zone_name,
                    ))
        
        # 3. Пересечение траекторий двух людей (опционально)
        # Можно добавить проверку сближения треков
        
        return events
    
    def _estimate_speed(self, traj: deque) -> float:
        """Скорость в px/s по последним 3 точкам."""
        if len(traj) < 3:
            return 0.0
        p1, p2, p3 = traj[-3], traj[-2], traj[-1]
        dt = p3.t - p1.t
        if dt <= 0:
            return 0.0
        dx = p3.x - p1.x
        dy = p3.y - p1.y
        return np.hypot(dx, dy) / dt
    
    def _moving_towards_zone(self, traj: deque, zone: Polygon) -> bool:
        """Проверка, что вектор движения указывает к зоне."""
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
    
    def get_events(self, since: float = 0) -> List[RiskEvent]:
        """Получить события для Evidence Replay."""
        return [e for e in self._events if e.timestamp >= since]
    
    def get_current_risk_level(self, track_id: int) -> str:
        """Текущий уровень риска для трека (для UI)."""
        if track_id not in self._trajectories:
            return "UNKNOWN"
        traj = self._trajectories[track_id]
        if len(traj) < self.min_history_points:
            return "NORMAL"
        # Проверить недавние события
        now = time.time()
        for ev in reversed(self._events):
            if ev.track_id == track_id and now - ev.timestamp < self.cooldown_seconds:
                return ev.level
        return "NORMAL"


# Глобальный экземпляр для интеграции в runtime (один на камеру)
_radars: Dict[str, SafetyRadar] = {}

def get_radar(camera_id: str) -> SafetyRadar:
    if camera_id not in _radars:
        _radars[camera_id] = SafetyRadar()
    return _radars[camera_id]
