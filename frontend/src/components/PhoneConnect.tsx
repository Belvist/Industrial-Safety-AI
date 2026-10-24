import { useEffect, useRef, useState } from "react";
import QRCode from "qrcode";
import { Smartphone } from "lucide-react";
import { api, errorText, pairingUrl } from "../api";

/** Operator side: mint a one-time pairing code, show it as a QR. */
export default function PhoneConnect({
  url,
  cameraId,
}: {
  url: string;
  cameraId: string;
}) {
  const [code, setCode] = useState<string | null>(null);
  const [expiresIn, setExpiresIn] = useState(0);
  const [deadline, setDeadline] = useState(0);
  const [isolated, setIsolated] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const pairing = code ? pairingUrl(url, code) : null;

  useEffect(() => {
    if (!pairing) return;
    const node = canvas.current;
    if (!node) return;
    QRCode.toCanvas(node, pairing, { width: 200, margin: 1 }).catch(() => {});
  }, [pairing]);
  useEffect(() => {
    setCode(null);
    setExpiresIn(0);
    setDeadline(0);
  }, [cameraId, url]);
  useEffect(() => {
    if (!code || !deadline) return;
    const timer = setInterval(() => {
      const left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      setExpiresIn(left);
      if (!left) {
        setCode(null);
        setError("Время действия кода истекло. Создайте новый код.");
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [code, deadline]);

  async function mint() {
    setBusy(true);
    setError(null);
    try {
      const created = await api<{ code: string; expires_in: number }>(
        "/pairing",
        {
          method: "POST",
          ...json(
            isolated ? { new_camera_name: "Телефон" } : { camera_id: cameraId },
          ),
        },
      );
      setCode(created.code);
      setExpiresIn(created.expires_in);
      setDeadline(Date.now() + created.expires_in * 1000);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }
  async function cancel() {
    if (!code) return;
    try {
      await api(`/pairing/${code}`, { method: "DELETE" });
    } catch {
      /* already gone */
    }
    setCode(null);
  }
  return (
    <section className="phone-panel">
      <div className="section-heading">
        <h2>
          <Smartphone size={16} /> Телефон как камера
        </h2>
      </div>
      {!code ? (
        <>
          <p>Одноразовый код на 5 минут.</p>
          <label className="check-label">
            <input
              type="checkbox"
              checked={isolated}
              onChange={(e) => setIsolated(e.target.checked)}
            />
            Отдельная камера телефона (свои кадры и история)
          </label>
          <button
            className="button"
            disabled={busy}
            onClick={() => void mint()}
          >
            Создать код для телефона
          </button>
          {error && <p className="notice-text">{error}</p>}
        </>
      ) : (
        <>
          <canvas
            ref={canvas}
            className="qr"
            aria-label="QR-код подключения телефона"
          />
          <p className="qr-url">{pairing}</p>
          <p>
            Одноразовый код · осталось {Math.floor(expiresIn / 60)}:
            {String(expiresIn % 60).padStart(2, "0")}
          </p>
          <button className="button" onClick={() => void cancel()}>
            Отменить код
          </button>
        </>
      )}
      <ol>
        <li>Телефон в том же Wi-Fi, что этот компьютер</li>
        <li>Отсканируйте QR камерой телефона</li>
        <li>Откройте страницу через HTTPS и разрешите доступ к камере</li>
        <li>Нажмите «Открыть камеру телефона», затем «Начать передачу»</li>
      </ol>
    </section>
  );
}

function json(body: unknown): RequestInit {
  return {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
