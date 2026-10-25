# Camera validation plan (physical footage, not mocks)

Shoot each scenario as a separate 10–20 s clip (or live), then check the
expected result in dashboard + SQLite + Telegram. Defaults assumed:
helmet/vest on, zone drawn, confirm 2 s, consensus 3/4, thresholds from `.env`.

## PPE scenarios

| # | Scenario | Expected |
|---|---|---|
| A | Person, no PPE at all | Worker appears; `no_helmet` + `no_vest` confirm ~2 s → 2 ACTIVE incidents, 1 snapshot each |
| B | Person in helmet + vest | Statuses present, zero incidents, no Telegram |
| C | Person removes helmet on camera | One frame changes nothing; ~2 s later exactly one `no_helmet` ACTIVE |
| D | Person puts helmet back on | Same incident → RESOLVED, no new incident |
| E | Person in vest, camera close | `vest: present`, no incident |
| F | Person removes/hides vest | One `no_vest` ACTIVE after confirmation |
| G | Person far from camera (~5 m+) | Worker tracked; PPE states mostly `unknown`, **no** missing/incidents |
| H | Person close (~1–2 m) | Stable ID, PPE present/missing with confidence |
| I | Person sideways / back to camera | No false `missing`: face-hidden PPE must stay `unknown` unless the negative head fires in-band |
| J | Poor light | Either `unknown` or confirmed violations — but no incident from a single flicker frame |
| K | Partial occlusion (door/equipment) | Worker ID survives short occlusion (stage-2 tracking); no duplicate incidents |
| L | Vest/clothing item on the floor, no person | Zero workers, zero incidents; raw box visible only with «Сырые детекции» toggle |
| M | Two people, one without helmet | Two stable IDs; violation attaches to the correct Worker only |

## Zone scenarios

| # | Scenario | Expected |
|---|---|---|
| N | Walk into drawn polygon | Separate CRITICAL `danger_zone` ACTIVE after confirmation |
| O | Walk out of polygon | Same incident → RESOLVED |

## Cross-cutting checks (every scenario)

- Exactly one incident per continuous violation (no per-frame spam)
- Snapshot attached to every ACTIVE incident
- Telegram message matches PDF §10 format with photo
- Source loss mid-violation → INTERRUPTED, never silent RESOLVED
- Monitor header shows the real provider (Local ONNX · CPU / CoreML, or LinQ mode)
- `backend/scripts/evaluate_folder.py` on saved frames matches live counts
