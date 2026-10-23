import { useEffect, useState } from "react";
import { authHeaders } from "./api";

/** Load a video with auth headers into a Blob URL (no token in the URL). */
export function useAuthedVideoUrl(src: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!src) {
      setUrl(null);
      return;
    }
    let live = true;
    let objectUrl: string | null = null;
    fetch(src, { headers: authHeaders() })
      .then((response) => {
        if (!response.ok) throw new Error(`video ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (!live) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (live) setUrl(null);
      });
    return () => {
      live = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);
  return url;
}
