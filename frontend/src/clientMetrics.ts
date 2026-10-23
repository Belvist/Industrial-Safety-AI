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
