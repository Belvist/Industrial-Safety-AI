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
  }, [cameraId, url]);

  async function mint() {
    setBusy(true);
    setError(null);
    try {
      const created = await api<{ code: string; expires_in: number }>(
        "/pairing",
        {
          method: "POST",
          ...json(
            isolated
              ? { new_camera_name: "Телефон" }
              : { camera_id: cameraId },
          ),
        },
      );
      setCode(created.code);
      setExpiresIn(created.expires_in);
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
          <button className="button" disabled={busy} onClick={() => void mint()}>
            Создать код для телефона
          </button>
          {error && <p className="notice-text">{error}</p>}
        </>
      ) : (
        <>
          <canvas ref={canvas} className="qr" aria-label={`QR: ${pairing}`} />
          <p className="qr-url">{pairing}</p>
          <p>Код сгорит при использовании или через {expiresIn} сек.</p>
          <button className="button" onClick={() => void cancel()}>
            Отменить код
          </button>
        </>
      )}
      <ol>
        <li>Телефон в том же Wi-Fi, что этот компьютер</li>
        <li>Отсканируй QR камерой телефона</li>
        <li>Прими предупреждение о сертификате и разреши доступ к камере</li>
        <li>Нажми «Запустить мониторинг» — кадры пойдут с телефона</li>
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
