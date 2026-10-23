# Frontend

React + Vite + TypeScript. Russian-language local dashboard: monitoring, browser camera selection, MP4/RTSP sources, normalized polygon editor, per-camera rules, incident history/evidence and operator feedback.

```bash
npm ci
npm run dev
npm run build
```

Development: `http://127.0.0.1:5173`, API proxy to `127.0.0.1:8000`. Production build is served by FastAPI from `frontend/dist` after backend restart. No CDN scripts/fonts are needed.

Browser camera capture uses `getUserMedia`, sends one JPEG at a time and waits for the result before sending the next frame. Device IDs/resolution are not hard-coded. Polygons and detections use normalized `0..1` coordinates. Camera access requires localhost or HTTPS; this app binds only to localhost.


Frontend state machines are in `src/machines/`; camera resource handling is in `src/hooks/useBrowserCamera.ts`. Server polling/commands are separate in `src/hooks/useSafety.ts`, and preview/evidence rendering is in `src/components/`. `npm run test` verifies lifecycle transitions. Design follows Scale's white workspace, restrained typography and a prominent monitoring surface. Detection labels include percentages; confidence is never presented as a safety probability.
