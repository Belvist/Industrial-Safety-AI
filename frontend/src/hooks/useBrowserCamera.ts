import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { api, authHeaders, errorText, json, stopPath } from "../api";
import { CaptureMachine } from "../machines/capture";
import type { Camera, Result } from "../types";

export function useBrowserCamera(onError: (message: string) => void) {
  const [machine] = useState(() => new CaptureMachine());
  const state = useSyncExternalStore(machine.subscribe, machine.getSnapshot);
  const [result, setResult] = useState<Result | null>(null);
  const [owner, setOwner] = useState<string | null>(null);
  const resource = useRef<{
    stream: MediaStream;
    video: HTMLVideoElement;
    cameraId: string;
    sessionId: string;
  } | null>(null);
  const epoch = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const request = useRef<AbortController | null>(null);
  const pending = useRef<Promise<void> | null>(null);
  const errorHandler = useRef(onError);
  errorHandler.current = onError;
  const dispose = () => {
    if (timer.current) clearTimeout(timer.current);
    request.current?.abort();
    resource.current?.stream.getTracks().forEach((t) => t.stop());
    resource.current = null;
    setOwner(null);
    setResult(null);
  };
  async function stop() {
    if (machine.getSnapshot().phase === "idle") return;
    if (machine.getSnapshot().phase === "stopping") {
      await pending.current;
      return;
    }
    machine.send("STOP");
    epoch.current++;
    const old = resource.current;
    dispose();
    // Permission/start may still be in flight. Its completion checks epoch and closes that exact session.
    await pending.current;
    try {
      if (old)
        await api(stopPath(old.cameraId, old.sessionId), { method: "POST" });
      machine.send("STOPPED");
    } catch (e) {
      machine.send("FAIL", undefined, errorText(e));
      throw e;
    }
  }
  async function start(cameraId: string, deviceId: string, targetFps: number) {
    await stop();
    machine.send("START", cameraId);
    setResult(null);
    const run = ++epoch.current;
    const setup = async () => {
      let media: MediaStream | null = null;
      let server: Camera | null = null;
      try {
        if (!navigator.mediaDevices)
          throw new Error(
            "Откройте приложение на localhost для доступа к камере.",
          );
        media = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 } }
            : { width: { ideal: 1280 } },
          audio: false,
        });
        if (run !== epoch.current) {
          media.getTracks().forEach((t) => t.stop());
          return;
        }
        machine.send("PERMITTED");
        server = await api<Camera>(
          `/cameras/${cameraId}/start`,
          json({ kind: "browser" }),
        );
        if (run !== epoch.current) {
          media.getTracks().forEach((t) => t.stop());
          await api(stopPath(cameraId, server.session_id), { method: "POST" });
          return;
        }
        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.srcObject = media;
        await video.play();
        if (run !== epoch.current) {
          media.getTracks().forEach((t) => t.stop());
          await api(stopPath(cameraId, server.session_id), { method: "POST" });
          return;
        }
        resource.current = {
          stream: media,
          video,
          cameraId,
          sessionId: server.session_id!,
        };
        setOwner(cameraId);
        machine.send("CONNECTED");
        const canvas = document.createElement("canvas");
        const loop = async () => {
          if (run !== epoch.current || !resource.current) return;
          const before = performance.now();
          try {
            if (!video.videoWidth || !video.videoHeight)
              throw new Error("Камера не передаёт изображение.");
            canvas.width = Math.min(video.videoWidth, 1280);
            canvas.height = Math.round(
              (video.videoHeight * canvas.width) / video.videoWidth,
            );
            canvas
              .getContext("2d")!
              .drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise<Blob>((resolve, reject) =>
              canvas.toBlob(
                (b) =>
                  b
                    ? resolve(b)
                    : reject(new Error("Не удалось получить кадр")),
                "image/jpeg",
                0.85,
              ),
            );
            if (run !== epoch.current) return;
            machine.send("CAPTURED");
            request.current = new AbortController();
            const next = await api<Result>(`/cameras/${cameraId}/frame`, {
              method: "POST",
              headers: {
                "Content-Type": "image/jpeg",
                "X-Session-Id": server!.session_id!,
              },
              body: blob,
              signal: request.current.signal,
            });
            if (run !== epoch.current) return;
            if (next.session_id !== server!.session_id)
              throw new Error(
                "Сессия камеры изменилась. Перезапустите мониторинг.",
              );
            setResult(next);
            machine.send("RESULT");
            timer.current = setTimeout(
              () => {
                if (run === epoch.current) {
                  machine.send("NEXT");
                  void loop();
                }
              },
              Math.max(0, 1000 / targetFps - (performance.now() - before)),
            );
          } catch (e) {
            if (run !== epoch.current) return;
            const old = resource.current;
            epoch.current++;
            dispose();
            machine.send("FAIL", undefined, errorText(e));
            errorHandler.current(errorText(e));
            if (old)
              void api(stopPath(old.cameraId, old.sessionId), {
                method: "POST",
              }).catch(() => {});
          }
        };
        media.getVideoTracks()[0].onended = () => {
          if (run === epoch.current) {
            errorHandler.current("Доступ к камере прерван.");
            void stop().catch(() => {});
          }
        };
        void loop();
      } catch (e) {
        media?.getTracks().forEach((t) => t.stop());
        if (server)
          await api(stopPath(cameraId, server.session_id), {
            method: "POST",
          }).catch(() => {});
        if (run === epoch.current) {
          machine.send("FAIL", undefined, errorText(e));
          throw e;
        }
      }
    };
    pending.current = setup();
    try {
      await pending.current;
    } finally {
      pending.current = null;
    }
  }
  useEffect(() => {
    const leave = () => {
      epoch.current++;
      if (timer.current) clearTimeout(timer.current);
      request.current?.abort();
      const old = resource.current;
      old?.stream.getTracks().forEach((t) => t.stop());
      if (old)
        void fetch("/api/v1" + stopPath(old.cameraId, old.sessionId), {
          method: "POST",
          headers: authHeaders(),
          keepalive: true,
        });
    };
    window.addEventListener("pagehide", leave);
    return () => {
      window.removeEventListener("pagehide", leave);
      leave();
    };
  }, []);
  return { state, result, owner, start, stop };
}
