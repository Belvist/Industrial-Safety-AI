import { useEffect, useState } from "react";
import { authHeaders } from "./api";

/** Keep the source payload with the pixels; polling must never relabel old pixels. */
export function useAuthedImage<T>(src: string | null, payload: T) {
  const [image, setImage] = useState<{
    url: string;
    payload: T;
    startedAt: number;
  } | null>(null);
  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }
    const startedAt = performance.now();
    const controller = new AbortController();
    let objectUrl: string | null = null;
    fetch(src, { headers: authHeaders(), signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`image ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setImage({ url: objectUrl, payload, startedAt });
      })
      .catch(() => {
        if (!controller.signal.aborted) setImage(null);
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
    // payload belongs to this exact source URL, not subsequent polling results.
  }, [src]);
  return image;
}

export function useAuthedImageUrl(src: string | null): string | null {
  return useAuthedImage(src, null)?.url ?? null;
}
