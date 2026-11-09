// Keep this algorithm in sync with ink2latex/features.py; fixtures verify float32 parity.
export const MAX_POINTS = 1600;
const STEP = 0.05;

function percentile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lo = Math.floor(index), hi = Math.ceil(index);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}

function resample(xy) {
  const positions = [xy[0]], distances = [0];
  let length = 0;
  for (let i = 1; i < xy.length; i++) {
    const delta = Math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]);
    length += delta;
    if (delta > 0) {
      positions.push(xy[i]);
      distances.push(length);
    }
  }
  if (length < STEP) return [xy[0]];
  const targets = [];
  for (let i = 0; i * STEP < length; i++) targets.push(i * STEP);
  targets.push(length);
  let segment = 1;
  return targets.map(target => {
    while (segment < distances.length - 1 && distances[segment] < target) segment++;
    const ratio = (target - distances[segment - 1]) / (distances[segment] - distances[segment - 1]);
    return [0, 1].map(k => positions[segment - 1][k] + ratio * (positions[segment][k] - positions[segment - 1][k]));
  });
}

export function features(strokes) {
  strokes = strokes.filter(s => s.length).map(s => s.map(p => p.slice(0, 2)));
  if (!strokes.length) return [];
  const all = strokes.flat();
  if (all.some(p => p.length !== 2 || p.some(v => !Number.isFinite(v)))) throw new Error("Ink coordinates must be finite");
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const minX = xs.reduce((a, b) => Math.min(a, b), Infinity);
  const minY = ys.reduce((a, b) => Math.min(a, b), Infinity);
  const height = percentile(ys, .95) - percentile(ys, .05);
  const width = percentile(xs, .95) - percentile(xs, .05);
  const scale = Math.max(height >= 1e-6 ? height : width, 1e-6);
  let rows = [];
  for (const stroke of strokes) {
    const xy = resample(stroke.map(([x, y]) => [(x - minX) / scale, (y - minY) / scale]));
    const center = xy.reduce((sum, p) => [sum[0] + p[0], sum[1] + p[1]], [0, 0]).map(v => v / xy.length);
    for (let i = 0; i < xy.length; i++) {
      const [x, y] = xy[i], next = xy[i + 1] || xy[i];
      rows.push([x, y, next[0] - x, next[1] - y, +(i === 0), +(i === xy.length - 1), center[0] - x, center[1] - y].map(Math.fround));
    }
  }
  if (rows.length > MAX_POINTS) {
    const original = rows;
    rows = Array.from({length: MAX_POINTS}, (_, i) => original[Math.floor(i * (original.length - 1) / (MAX_POINTS - 1) + .5)]);
  }
  return rows;
}
