import { useEffect, useState } from "react";
import { api, errorText } from "./api";
import { useBrowserCamera } from "./hooks/useBrowserCamera";
import { useAuthedImageUrl } from "./useAuthedImage";
import type { Camera } from "./types";

function PhonePreview({ src }: { src: string | null }) {
  const url = useAuthedImageUrl(src);
  return url ? (
    <img
      src={url}
      alt="Анализируемый кадр"
      style={{ width: "100%", maxWidth: 640, borderRadius: 12 }}
    />
  ) : (
    <p>Ожидаем первый обработанный кадр…</p>
  );
}
export default function PhoneCapturePage() {
  const [cameraId] = useState(() => localStorage.getItem("phone_camera") || "");
  const [camera, setCamera] = useState<Camera | null>(null);
  const [message, setMessage] = useState("");
  const browser = useBrowserCamera(setMessage);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!cameraId) {
      setMessage("Телефон не привязан. Отсканируйте новый QR у оператора.");
      return;
    }
    let live = true;
    void api<Camera>(`/cameras/${encodeURIComponent(cameraId)}`)
      .then((c) => {
        if (live) setCamera(c);
      })
      .catch((e: unknown) => {
        if (live)
          setMessage(
            "Нет доступа к камере: " +
              errorText(e) +
              ". Подключитесь повторно.",
          );
      });
    return () => {
      live = false;
    };
  }, [cameraId]);

  async function start() {
    if (!cameraId || busy) return;
    setBusy(true);
    setMessage("");
    try {
      await browser.start(cameraId, "", 5);
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  }
  async function stop() {
    setBusy(true);
    try {
      await browser.stop();
      localStorage.removeItem("phone_token");
      localStorage.removeItem("phone_camera");
      setMessage(
        "Камера остановлена. Для следующего сеанса потребуется новый QR.",
      );
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  }
  const capturing =
    browser.state.phase !== "idle" && browser.state.phase !== "failed";
  return (
    <main className="connect-page">
      <h2>Камера телефона</h2>
      <p>
        {camera?.name || "Подключённая камера"} · {browser.state.phase}
      </p>
      {message && <p role="alert">{message}</p>}
      {!capturing && (
        <button
          className="button primary"
          disabled={busy || !camera}
          onClick={() => void start()}
        >
          Начать передачу
        </button>
      )}
      {capturing && (
        <button className="button" disabled={busy} onClick={() => void stop()}>
          Остановить
        </button>
      )}
      {capturing && <PhonePreview src={browser.result?.frame_url || null} />}
      {browser.result && (
        <p>
          {browser.result.tracks.length} работников ·{" "}
          {browser.result.metrics?.rates.analytics?.fps.toFixed(1) ?? "—"} FPS
          аналитики · {browser.result.latency_ms} мс
        </p>
      )}
      <p>
        Этот экран может передавать кадры только назначенной камеры. Настройки,
        история и другие камеры недоступны.
      </p>
    </main>
  );
}
