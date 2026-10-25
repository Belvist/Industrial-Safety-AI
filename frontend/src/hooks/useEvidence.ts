import { useEffect, useState } from "react";
import { authHeaders } from "../api";

export function useEvidence(src: string | null, retry404: boolean) {
  const [version, setVersion] = useState(0),
    [state, setState] = useState<{
      url: string | null;
      loading: boolean;
      error: boolean;
    }>({ url: null, loading: !!src, error: false });
  useEffect(() => {
    setState({ url: null, loading: !!src, error: false });
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
        if (response.status === 404 && retry404 && attempt < 20) {
          timer = setTimeout(() => void load(attempt + 1), 1000);
          return;
        }
        if (!response.ok) throw new Error();
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ url: objectUrl, loading: false, error: false });
      } catch {
        if (!controller.signal.aborted)
          setState({ url: null, loading: false, error: true });
      }
    };
    void load(0);
    return () => {
      controller.abort();
      clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src, retry404, version]);
  return { ...state, retry: () => setVersion((v) => v + 1) };
}
