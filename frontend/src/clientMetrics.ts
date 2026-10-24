type Stage = {
  samples: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
};
const samples = new Map<string, number[]>();

/** Per-browser bounded timing samples. Never records URLs, payloads or credentials. */
export function observeClient(name: string, milliseconds: number) {
  const values = samples.get(name) ?? [];
  values.push(milliseconds);
  if (values.length > 4096) values.shift();
  samples.set(name, values);
}

export function clientProfile(): Record<string, Stage> {
  return Object.fromEntries(
    [...samples].map(([name, values]) => {
      const sorted = [...values].sort((a, b) => a - b);
      const percentile = (p: number) => {
        const index = (sorted.length - 1) * p;
        const low = Math.floor(index),
          high = Math.ceil(index);
        return (
          Math.round(
            (sorted[low] + (sorted[high] - sorted[low]) * (index - low)) * 10,
          ) / 10
        );
      };
      return [
        name,
        {
          samples: sorted.length,
          p50: percentile(0.5),
          p95: percentile(0.95),
          p99: percentile(0.99),
          max: sorted[sorted.length - 1],
        },
      ];
    }),
  );
}

const viewports = new Map<
  string,
  {
    session: string;
    started: number;
    last: number;
    frames: number[];
    total: number;
  }
>();
/** Counts only unique inference images whose pixels reached the browser viewport. */
export function markUiFrame(
  cameraId: string,
  session: string,
  sequence: number,
) {
  const now = performance.now();
  let view = viewports.get(cameraId);
  if (!view || view.session !== session) {
    view = { session, started: now, last: -1, frames: [], total: 0 };
    viewports.set(cameraId, view);
  }
  if (sequence <= view.last) return;
  view.last = sequence;
  view.total++;
  view.frames.push(now);
  view.frames = view.frames.filter((time) => time >= now - 60000).slice(-18000);
}
export function uiRate(cameraId?: string) {
  const view = cameraId ? viewports.get(cameraId) : undefined;
  if (!view) return null;
  const now = performance.now(),
    seconds = Math.min(60, (now - view.started) / 1000);
  const count = view.frames.filter((time) => time >= now - 60000).length;
  return {
    fps: seconds >= 1 ? count / seconds : null,
    window_seconds: seconds,
    window_count: count,
    total: view.total,
  };
}
