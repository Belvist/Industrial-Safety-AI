import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4


def utcnow():
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS cameras (id TEXT PRIMARY KEY, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, camera_name TEXT NOT NULL,
                    track_id INTEGER NOT NULL, type TEXT NOT NULL, severity TEXT NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, resolved_at TEXT,
                    confidence REAL NOT NULL, duration REAL NOT NULL, snapshot TEXT,
                    feedback TEXT NOT NULL DEFAULT 'unreviewed',
                    resolution_reason TEXT, notification TEXT NOT NULL DEFAULT 'disabled',
                    zone_name TEXT, evidence TEXT
                );
                CREATE INDEX IF NOT EXISTS incident_time ON incidents(started_at DESC);
                CREATE INDEX IF NOT EXISTS incident_type ON incidents(type, started_at DESC);
                CREATE INDEX IF NOT EXISTS incident_camera ON incidents(camera_id, started_at DESC);
            """)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(incidents)")}
            if "ended_at" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN ended_at TEXT")
            if "zone_name" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN zone_name TEXT")
            if "evidence" not in columns:
                db.execute("ALTER TABLE incidents ADD COLUMN evidence TEXT")
            for name, kind in [("episode_id", "TEXT"), ("source_seconds", "REAL")]:
                if name not in columns:
                    db.execute(f"ALTER TABLE incidents ADD COLUMN {name} {kind}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS observation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL,
                    track_id INTEGER, zone_name TEXT, kind TEXT NOT NULL,
                    source_seconds REAL, occurred_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    episode_id TEXT, incident_id TEXT, evidence TEXT,
                    payload TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS event_camera_time ON observation_events(camera_id, occurred_at);
                CREATE INDEX IF NOT EXISTS event_episode ON observation_events(episode_id);
                CREATE INDEX IF NOT EXISTS event_incident ON observation_events(incident_id);
            """)
            db.execute("CREATE INDEX IF NOT EXISTS incident_zone ON incidents(zone_name)")
            # Legacy rows finished before ended_at existed: end time falls
            # back to resolved_at, then started_at (documented, never invented
            # as a fresh observation — it only bounds retention).
            db.execute(
                "UPDATE incidents SET ended_at=COALESCE(resolved_at, started_at) "
                "WHERE status!='active' AND ended_at IS NULL"
            )
            # A process restart cannot prove that a safety violation has cleared.
            # ended_at is recorded (terminal time); resolved_at stays NULL
            # because nothing was proven cleared.
            db.execute(
                "UPDATE incidents SET status='interrupted', resolution_reason='server_restarted', ended_at=COALESCE(ended_at, ?) WHERE status='active'",
                (utcnow(),),
            )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def cameras(self):
        with self.connection() as db:
            return [
                {"id": r["id"], **json.loads(r["config"])}
                for r in db.execute("SELECT * FROM cameras")
            ]

    def save_camera(self, config, camera_id=None):
        camera_id = camera_id or uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO cameras VALUES (?,?)",
                (camera_id, json.dumps(config.model_dump())),
            )
        return camera_id

    def reassign_camera_owner(self, old_owner: str, new_owner: str) -> int:
        """Move cameras from one owner marker to another. Returns count."""
        with self.connection() as db:
            rows = db.execute("SELECT id, config FROM cameras").fetchall()
            moved = 0
            for row in rows:
                config = json.loads(row["config"])
                if config.get("owner") == old_owner:
                    config["owner"] = new_owner
                    db.execute(
                        "UPDATE cameras SET config=? WHERE id=?",
                        (json.dumps(config), row["id"]),
                    )
                    moved += 1
            return moved

    def create_incident(
        self,
        camera_id,
        name,
        key,
        confidence,
        duration,
        incident_id=None,
        snapshot=None,
        zone_name=None,
        evidence=None,
        started_at=None,
        episode_id=None,
        source_seconds=None,
    ):
        record = {
            "id": incident_id or uuid4().hex,
            "camera_id": camera_id,
            "camera_name": name,
            "track_id": key[0],
            "type": key[1],
            "severity": "critical" if key[1] in {"danger_zone", "smoking"} else "warning",
            "status": "active",
            "started_at": started_at or utcnow(),
            "confidence": confidence,
            "duration": duration,
            "snapshot": snapshot,
            "zone_name": zone_name,
            "evidence": evidence,
            "episode_id": episode_id,
            "source_seconds": source_seconds,
        }
        with self.connection() as db:
            db.execute(
                f"INSERT INTO incidents ({','.join(record)}) VALUES ({','.join('?' for _ in record)})",
                tuple(record.values()),
            )
        return record

    def update(self, incident_id, **values):
        allowed = {
            "status",
            "resolved_at",
            "ended_at",
            "confidence",
            "duration",
            "snapshot",
            "evidence",
            "feedback",
            "resolution_reason",
            "notification",
        }
        if not values.keys() <= allowed:
            raise ValueError("Unsupported incident fields")
        with self.connection() as db:
            if "status" in values and values["status"] not in {"resolved", "interrupted"}:
                raise ValueError("Only ACTIVE incidents can transition to RESOLVED or INTERRUPTED.")
            guard = " AND status='active'" if "status" in values else ""
            cursor = db.execute(
                f"UPDATE incidents SET {','.join(k + '=?' for k in values)} WHERE id=?{guard}",
                (*values.values(), incident_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Incident not found or its terminal state cannot change.")

    def finish_incident(self, incident_id, outcome, reason, ended_at=None, duration=None):
        """Atomically finish an ACTIVE incident exactly once.

        outcome is "resolved" (violation proven cleared) or "interrupted".
        Records the terminal time (ended_at, default now; resolved_at mirrors
        it for compatibility), the reason, and keeps snapshot/feedback rows
        untouched. Raises ValueError on double-finish or unknown id.
        """
        if outcome not in {"resolved", "interrupted"}:
            raise ValueError("Outcome must be RESOLVED or INTERRUPTED.")
        stamp = ended_at or utcnow()
        values = {
            "status": outcome,
            "resolved_at": stamp if outcome == "resolved" else None,
            "ended_at": stamp,
            "resolution_reason": reason,
        }
        if duration is not None:
            values["duration"] = duration
        with self.connection() as db:
            cursor = db.execute(
                f"UPDATE incidents SET {','.join(k + '=?' for k in values)} "
                "WHERE id=? AND status='active'",
                (*values.values(), incident_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Incident not found or already finished.")
        return self.get(incident_id)

    def incident_for_episode(self, camera_id, kind, zone_name, episode_id):
        if not episode_id:
            return None
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM incidents WHERE camera_id=? AND type=? "
                "AND COALESCE(zone_name,'')=? AND episode_id=? "
                "AND status IN ('active','interrupted') ORDER BY started_at LIMIT 1",
                (camera_id, kind, zone_name or "", episode_id),
            ).fetchone()
        return dict(row) if row else None

    def reopenable_incident(self, camera_id, kind, zone_name=None, within_seconds=60):
        """A recently finished incident for the same camera/type/zone.

        Used to merge a continuous violation across Worker ID switches and
        short tracking gaps instead of creating a new row every time.
        """
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM incidents WHERE camera_id=? AND type=? "
                "AND COALESCE(zone_name,'')=? ORDER BY started_at DESC LIMIT 5",
                (camera_id, kind, zone_name or ""),
            ).fetchall()
        if not rows:
            return None
        now = datetime.now(UTC)
        for row in rows:
            record = dict(row)
            if record["status"] == "active":
                return record
            stamp = record.get("ended_at") or record.get("resolved_at") or record["started_at"]
            try:
                ended = datetime.fromisoformat(stamp)
            except (TypeError, ValueError):
                continue
            if (now - ended).total_seconds() <= within_seconds:
                return record
        return None

    def reopen(self, incident_id):
        """Return an incident to active state for a merged continuation."""
        with self.connection() as db:
            db.execute(
                "UPDATE incidents SET status='active', ended_at=NULL, "
                "resolved_at=NULL, resolution_reason=NULL WHERE id=?",
                (incident_id,),
            )
        return self.get(incident_id)

    def active_incident(self, camera_id, kind, zone_name=None):
        """An active incident for the same camera/type/zone, if any.

        Used to merge a continuous violation across Worker ID switches.
        """
        with self.connection() as db:
            if zone_name:
                row = db.execute(
                    "SELECT * FROM incidents WHERE camera_id=? AND type=? "
                    "AND status='active' AND COALESCE(zone_name,'')=? "
                    "ORDER BY started_at DESC LIMIT 1",
                    (camera_id, kind, zone_name),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM incidents WHERE camera_id=? AND type=? "
                    "AND status='active' ORDER BY started_at DESC LIMIT 1",
                    (camera_id, kind),
                ).fetchone()
        return dict(row) if row else None

    def get(self, incident_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        return dict(row) if row else None

    def incidents(
        self,
        status=None,
        limit=100,
        offset=0,
        camera_ids=None,
        types=None,
        zone_name=None,
        from_time=None,
        to_time=None,
        feedback=None,
    ):
        with self.connection() as db:
            # Freeze count and rows together while live cameras append incidents.
            db.execute("BEGIN")
            clauses, params = [], []
            if feedback:
                clauses.append("feedback=?")
                params.append(feedback)
            if status:
                clauses.append("status=?")
                params.append(status)
            if camera_ids is not None:
                ids = sorted(camera_ids)
                if not ids:
                    return {"items": [], "total": 0}
                clauses.append(f"camera_id IN ({','.join('?' * len(ids))})")
                params.extend(ids)
            if types:
                kinds = sorted(set(types))
                clauses.append(f"type IN ({','.join('?' * len(kinds))})")
                params.extend(kinds)
            if zone_name:
                clauses.append("zone_name=?")
                params.append(zone_name)
            if from_time:
                clauses.append("started_at>=?")
                params.append(from_time)
            if to_time:
                clauses.append("started_at<=?")
                params.append(to_time)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            total = db.execute("SELECT count(*) FROM incidents" + where, params).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM incidents" + where + " ORDER BY started_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            )
            return {"items": [dict(r) for r in rows], "total": total}

    def purge_old_resolved(self, days: int, snapshots_dir: Path) -> int:
        """Delete terminal incidents older than `days` with their snapshots.

        Plus always sweep orphan snapshot files (no DB row references them).
        days <= 0 disables age purging (orphan sweep still runs).
        """
        removed = 0
        if days > 0:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            with self.connection() as db:
                # Real end time with documented legacy fallback; ACTIVE never matches.
                rows = db.execute(
                    "SELECT id, snapshot FROM incidents "
                    "WHERE status IN ('resolved','interrupted') "
                    "AND COALESCE(ended_at, resolved_at, started_at) < ?",
                    (cutoff,),
                ).fetchall()
                for row in rows:
                    if row["snapshot"]:
                        (snapshots_dir / Path(row["snapshot"]).name).unlink(missing_ok=True)
                    db.execute("DELETE FROM incidents WHERE id=?", (row["id"],))
                    removed += 1
                db.execute(
                    "DELETE FROM observation_events WHERE recorded_at < ? AND NOT EXISTS (SELECT 1 FROM incidents i WHERE i.status='active' AND i.camera_id=observation_events.camera_id AND (i.id=observation_events.incident_id OR i.episode_id=observation_events.episode_id))",
                    (cutoff,),
                )
        with self.connection() as db:
            known = {r["snapshot"] for r in db.execute("SELECT snapshot FROM incidents")}
        if snapshots_dir.is_dir():
            for path in snapshots_dir.iterdir():
                if path.is_file() and not path.name.startswith(".") and path.name not in known:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed

    def stats(self, camera_ids=None):
        if camera_ids is not None and not camera_ids:
            return {"total": 0, "active": 0, "today": 0, "false_positive": 0, "unreviewed": 0}
        sql = (
            "SELECT count(*) AS total, coalesce(sum(status='active'),0) AS active, "
            "coalesce(sum(started_at >= date('now')),0) AS today, "
            "coalesce(sum(feedback='false_positive'),0) AS false_positive, "
            "coalesce(sum(feedback='unreviewed'),0) AS unreviewed "
            "FROM incidents"
        )
        params = []
        if camera_ids is not None:
            ids = sorted(camera_ids)
            sql += " WHERE camera_id IN (" + ",".join("?" for _ in ids) + ")"
            params = ids
        with self.connection() as db:
            return dict(db.execute(sql, params).fetchone())

    def record_event(
        self,
        camera_id,
        kind,
        *,
        track_id=None,
        zone_name=None,
        source_seconds=None,
        occurred_at=None,
        episode_id=None,
        incident_id=None,
        evidence=None,
        payload=None,
    ):
        if kind not in {
            "warning",
            "critical",
            "confirmed",
            "resolved",
            "interrupted",
            "evidence_ready",
            "operator_review",
        }:
            raise ValueError("Unknown observation event")
        with self.connection() as db:
            db.execute(
                "INSERT INTO observation_events(camera_id,track_id,zone_name,kind,source_seconds,occurred_at,recorded_at,episode_id,incident_id,evidence,payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    camera_id,
                    track_id,
                    zone_name,
                    kind,
                    source_seconds,
                    occurred_at or utcnow(),
                    utcnow(),
                    episode_id,
                    incident_id,
                    evidence,
                    json.dumps(payload or {}),
                ),
            )

    def timeline(self, incident):
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM observation_events WHERE camera_id=? AND (incident_id=? OR (episode_id IS NOT NULL AND episode_id=?) OR (kind='evidence_ready' AND evidence IS NOT NULL AND evidence=?)) ORDER BY recorded_at,id LIMIT 500",
                (
                    incident["camera_id"],
                    incident["id"],
                    incident.get("episode_id"),
                    incident.get("evidence"),
                ),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def risk_statistics(self, camera_ids=None, from_time=None, to_time=None):
        clauses, params = [], []
        if camera_ids is not None:
            ids = sorted(camera_ids)
            if not ids:
                return {
                    "incidents": [],
                    "warnings": [],
                    "note": "Counts of recorded observations, not accident probabilities.",
                }
            clauses.append("camera_id IN (" + ",".join("?" for _ in ids) + ")")
            params.extend(ids)
        if from_time:
            clauses.append("{time}>=?")
            params.append(from_time)
        if to_time:
            clauses.append("{time}<=?")
            params.append(to_time)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as db:
            db.execute("BEGIN")
            incidents = db.execute(
                "SELECT camera_id,type,coalesce(zone_name,'') AS zone_name,substr(started_at,1,10) AS day,count(*) AS count,sum(feedback='confirmed') AS confirmed,sum(feedback='false_positive') AS false_positive FROM incidents"
                + where.format(time="started_at")
                + " GROUP BY camera_id,type,zone_name,day ORDER BY day DESC,camera_id,zone_name",
                params,
            ).fetchall()
            event_where = (
                where.format(time="occurred_at")
                + (" AND " if clauses else " WHERE ")
                + "kind IN ('warning','critical')"
            )
            warnings = db.execute(
                "SELECT camera_id,coalesce(zone_name,'') AS zone_name,kind,substr(occurred_at,1,10) AS day,count(*) AS count FROM observation_events"
                + event_where
                + " GROUP BY camera_id,zone_name,kind,day ORDER BY day DESC",
                params,
            ).fetchall()
        return {
            "incidents": [dict(row) for row in incidents],
            "warnings": [dict(row) for row in warnings],
            "note": "Counts of recorded observations, not accident probabilities.",
        }
