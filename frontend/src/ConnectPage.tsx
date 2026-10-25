import { useEffect, useState } from "react";
import { api, errorText } from "./api";

type PhoneBinding = { token: string; camera_id: string; expires_in: number };
// React StrictMode remounts effects; exchange the single-use code only once.
const exchanges = new Map<string, Promise<PhoneBinding>>();
function exchangeOnce(code: string): Promise<PhoneBinding> {
  const prior = exchanges.get(code);
  if (prior) return prior;
  const promise = api<PhoneBinding>("/pairing/exchange", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
  exchanges.set(code, promise);
  return promise;
}

export default function ConnectPage() {
  const [status, setStatus] = useState("Подключаем телефон…");
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const code = new URLSearchParams(location.search).get("pair");
    if (!code) {
      setStatus("Нет кода подключения. Создайте новый QR на компьютере.");
      return;
    }
    let mounted = true;
    void exchangeOnce(code).then((binding) => {
      localStorage.setItem("phone_token", binding.token);
      localStorage.setItem("phone_camera", binding.camera_id);
      history.replaceState(null, "", "/connect");
      if (mounted) {
        setReady(true);
        setStatus("Телефон подключён. Доступна только назначенная камера.");
      }
    }).catch((err: unknown) => {
      history.replaceState(null, "", "/connect");
      if (mounted) setStatus("Ошибка подключения: " + errorText(err) + ". Запросите новый QR.");
    });
    return () => { mounted = false; };
  }, []);
  return <main className="connect-page">
    <h2>Подключение телефона</h2>
    <p>{status}</p>
    {ready && <button className="button primary" onClick={() => { location.href = "/phone"; }}>Открыть камеру телефона</button>}
  </main>;
}
