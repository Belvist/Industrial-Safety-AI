import { useEffect, useState } from "react";
import { authHeaders } from "./api";

/** Evidence is encoded asynchronously; retry 404 briefly without leaking a token. */
export function useAuthedVideoUrl(src: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    setUrl(null);
    if (!src) return;
    const controller = new AbortController();
    let objectUrl: string | null = null;
    let timer: ReturnType<typeof setTimeout>;
    const load = async (attempt: number) => {
      try {
        const response = await fetch(src, {
          headers: authHeaders(),
          signal: controller.signal,
        });
        if (response.status === 404 && attempt < 10) {
          timer = setTimeout(() => void load(attempt + 1), 1000);
          return;
        }
        if (!response.ok) throw new Error(`video ${response.status}`);
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        if (!controller.signal.aborted) setUrl(null);
      }
    };
    void load(0);
    return () => {
      controller.abort();
      clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);
  return url;
}
