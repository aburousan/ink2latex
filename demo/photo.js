// Keep integer rounding, neighbor order and tie-breaking in sync with ink2latex/photo.py.
function resize(gray) {
  const h = gray.length, w = gray[0].length, scale = Math.min(1, 1200 / Math.max(h, w));
  const nh = Math.max(1, Math.floor(h * scale)), nw = Math.max(1, Math.floor(w * scale));
  if (nh === h && nw === w) return {gray: Uint8Array.from(gray.flatMap(row => Array.from(row, v => Math.floor(v + .5)))), w, h, scale: [1, 1]};
  const horizontal = new Float64Array(h * nw), output = new Uint8Array(nh * nw);
  for (let x = 0; x < nw; x++) {
    const a = x * w, b = (x + 1) * w;
    for (let j = Math.floor(a / nw); j <= Math.floor((b - 1) / nw); j++) {
      const weight = Math.min(b, (j + 1) * nw) - Math.max(a, j * nw);
      for (let y = 0; y < h; y++) horizontal[y * nw + x] += Math.floor(gray[y][j] + .5) * weight;
    }
  }
  for (let y = 0; y < nh; y++) {
    const a = y * h, b = (y + 1) * h, sums = new Float64Array(nw);
    for (let j = Math.floor(a / nh); j <= Math.floor((b - 1) / nh); j++) {
      const weight = Math.min(b, (j + 1) * nh) - Math.max(a, j * nh);
      for (let x = 0; x < nw; x++) sums[x] += weight * horizontal[j * nw + x];
    }
    for (let x = 0; x < nw; x++) output[y * nw + x] = Math.floor((sums[x] + Math.floor(h * w / 2)) / (h * w));
  }
  return {gray: output, w: nw, h: nh, scale: [w / nw, h / nh]};
}

function threshold(gray, w, h) {
  const r = Math.max(1, Math.floor(Math.min(h, w) / 20)), stride = w + 1;
  const integral = new Float64Array((h + 1) * stride), flat = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += gray[y * w + x];
      integral[(y + 1) * stride + x + 1] = row + integral[y * stride + x + 1];
    }
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const ya = Math.max(0, y - r), yb = Math.min(h, y + r + 1), xa = Math.max(0, x - r), xb = Math.min(w, x + r + 1);
    const total = integral[yb * stride + xb] - integral[ya * stride + xb] - integral[yb * stride + xa] + integral[ya * stride + xa];
    const value = Math.min(255, Math.floor((gray[y * w + x] * 255 * (yb - ya) * (xb - xa) + Math.floor(total / 2)) / Math.max(1, total)));
    flat[y * w + x] = value;
  }
  return otsu(flat);
}

function otsu(gray) {
  const hist = new Float64Array(256), size = gray.length;
  for (const v of gray) hist[v]++;
  let full = 0, count = 0, weighted = 0, best = -1, cut = 0;
  for (let i = 0; i < 256; i++) full += i * hist[i];
  for (let i = 0; i < 256; i++) {
    count += hist[i]; weighted += i * hist[i];
    if (!count || count === size) continue;
    const score = (full * count - weighted * size) ** 2 / (count * (size - count));
    if (score > best) { best = score; cut = i; }
  }
  return gray.map(v => +(best >= 0 && v <= cut));
}

function neighbors(p, w, h) {
  const x = p % w, y = Math.floor(p / w), result = [];
  for (let yy = Math.max(0, y - 1); yy < Math.min(h, y + 2); yy++)
    for (let xx = Math.max(0, x - 1); xx < Math.min(w, x + 2); xx++)
      if (xx !== x || yy !== y) result.push(yy * w + xx);
  return result;
}

function components(mask, w, h) {
  const seen = new Uint8Array(mask.length), groups = [];
  for (let first = 0; first < mask.length; first++) {
    if (!mask[first] || seen[first]) continue;
    const group = [first]; seen[first] = 1;
    for (let i = 0; i < group.length; i++) for (const q of neighbors(group[i], w, h)) {
      if (mask[q] && !seen[q]) { seen[q] = 1; group.push(q); }
    }
    groups.push(group);
  }
  return groups;
}

function thin(mask, w, h) {
  const stride = w + 2, a = new Uint8Array((h + 2) * stride);
  for (let y = 0; y < h; y++) a.set(mask.subarray(y * w, (y + 1) * w), (y + 1) * stride + 1);
  const offsets = [-stride, -stride + 1, 1, stride + 1, stride, stride - 1, -1, -stride - 1];
  while (true) {
    let changed = false;
    for (let phase = 0; phase < 2; phase++) {
      const remove = [];
      for (let y = 1; y <= h; y++) for (let x = 1; x <= w; x++) {
        const i = y * stride + x;
        if (!a[i]) continue;
        const p = offsets.map(o => a[i + o]), count = p.reduce((s, v) => s + v, 0);
        let transitions = 0;
        for (let j = 0; j < 8; j++) if (!p[j] && p[(j + 1) % 8]) transitions++;
        const products = phase === 0 ? [p[0] * p[2] * p[4], p[2] * p[4] * p[6]] : [p[0] * p[2] * p[6], p[0] * p[4] * p[6]];
        if (count >= 2 && count <= 6 && transitions === 1 && products.every(v => v === 0)) remove.push(i);
      }
      for (const i of remove) a[i] = 0;
      if (remove.length) changed = true;
    }
    if (!changed) break;
  }
  const result = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) result.set(a.subarray((y + 1) * stride + 1, (y + 1) * stride + w + 1), y * w);
  return result;
}

function length(line) {
  let total = 0;
  for (let i = 1; i < line.length; i++) total += Math.hypot(line[i][0] - line[i-1][0], line[i][1] - line[i-1][1]);
  return total;
}

function trace(skeleton, w, h, width) {
  const pixels = [], adjacent = new Map(), visited = new Set(), lines = [];
  for (let p = 0; p < skeleton.length; p++) if (skeleton[p]) pixels.push(p);
  for (const p of pixels) adjacent.set(p, neighbors(p, w, h).filter(q => skeleton[q]));
  for (const group of components(skeleton, w, h)) if (group.length <= 1.5 * width) {
    const first = group.reduce((a, b) => Math.min(a, b));
    lines.push([[first % w, Math.floor(first / w)]]);
    for (const p of group) visited.add(p);
  }
  for (const start of [...pixels.filter(p => adjacent.get(p).length === 1), ...pixels]) {
    if (visited.has(start)) continue;
    const path = [];
    let p = start, previous = null;
    while (true) {
      path.push([p % w, Math.floor(p / w)]); visited.add(p);
      const choices = adjacent.get(p).filter(q => !visited.has(q));
      if (!choices.length) break; // through junctions, as photo.py
      if (previous !== null) {
        const dx = p % w - previous % w, dy = Math.floor(p / w) - Math.floor(previous / w);
        const straight = q => {
          const qx = q % w - p % w, qy = Math.floor(q / w) - Math.floor(p / w);
          return (dx * qx + dy * qy) / Math.hypot(qx, qy);
        };
        choices.sort((a, b) => straight(b) - straight(a));
      }
      previous = p; p = choices[0];
    }
    if (length(path) >= 1.5 * width) lines.push(path);
  }
  return lines;
}

function merge(lines, width) {
  const cosine = Math.cos(40 * Math.PI / 180);
  while (true) {
    let best = null;
    for (let i = 0; i < lines.length; i++) {
      const a = lines[i];
      if (a.length < 2) continue;
      for (let j = i + 1; j < lines.length; j++) {
        const b = lines[j];
        if (b.length < 2) continue;
        for (const ra of [false, true]) for (const rb of [false, true]) {
          const aa = ra ? [...a].reverse() : a, bb = rb ? [...b].reverse() : b;
          const distance = Math.hypot(aa.at(-1)[0] - bb[0][0], aa.at(-1)[1] - bb[0][1]);
          if (distance > 1.5 * width) continue;
          const k = Math.min(aa.length - 1, Math.max(1, Math.floor(width))), l = Math.min(bb.length - 1, Math.max(1, Math.floor(width)));
          const u = [aa.at(-1)[0] - aa.at(-1-k)[0], aa.at(-1)[1] - aa.at(-1-k)[1]];
          const v = [bb[l][0] - bb[0][0], bb[l][1] - bb[0][1]], norm = Math.hypot(...u) * Math.hypot(...v);
          if (norm && (u[0] * v[0] + u[1] * v[1]) / norm > cosine && (!best || distance < best[0])) best = [distance, i, j, [...aa, ...bb]];
        }
      }
    }
    if (!best) return lines;
    lines[best[1]] = best[3]; lines.splice(best[2], 1);
  }
}

function simplify(line) {
  if (line.length <= 2) return line;
  const keep = new Set([0, line.length - 1]), stack = [[0, line.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop(), dx = line[b][0] - line[a][0], dy = line[b][1] - line[a][1], norm = dx * dx + dy * dy;
    let maximum = .25, index = null;
    for (let i = a + 1; i < b; i++) {
      const x = line[i][0] - line[a][0], y = line[i][1] - line[a][1];
      const t = norm ? Math.max(0, Math.min(1, (x * dx + y * dy) / norm)) : 0;
      const distance = (x - t * dx) ** 2 + (y - t * dy) ** 2;
      if (distance > maximum) { maximum = distance; index = i; }
    }
    if (index !== null) { keep.add(index); stack.push([a, index], [index, b]); }
  }
  return [...keep].sort((a, b) => a - b).map(i => line[i]);
}

// Groups of ids whose boxes overlap along an axis (1: rows by y; 0: columns by x), in order along it. Columns need a
// substantial overlap (half the narrower), so a superscript beside its base stays apart. Mirrors photo.py _bands.
function bands(ids, boxes, axis) {
  const lo = axis ? 1 : 0, hi = axis ? 3 : 2;
  let groups = [];
  const sorted = [...ids].sort((a, b) => boxes[a][lo] - boxes[b][lo] || boxes[a][1 - axis] - boxes[b][1 - axis] || a - b);
  for (const i of sorted) {
    const a0 = boxes[i][lo], a1 = boxes[i][hi];
    const joins = g => axis ? a0 <= g[1] && a1 >= g[0] : Math.min(a1, g[1]) - Math.max(a0, g[0]) > .5 * Math.min(a1 - a0, g[1] - g[0]);
    const group = [a0, a1, [i]];
    for (const g of groups.filter(joins)) { group[0] = Math.min(group[0], g[0]); group[1] = Math.max(group[1], g[1]); group[2].push(...g[2]); }
    groups = groups.filter(g => !joins(g)); groups.push(group);
  }
  return groups.sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(g => g[2]);
}

// Writing order: rows, then columns inside a row, then stacked parts inside a column, and so on (photo.py _arrange).
function arrange(ids, boxes, dots, rows = true, depth = 0) {
  if (ids.length <= 1 || depth > 12) return [...ids];
  let parts = bands(ids, boxes, rows ? 1 : 0);
  if (parts.length === 1) {
    rows = !rows;
    parts = bands(ids, boxes, rows ? 1 : 0);
    if (parts.length === 1) return [...ids].sort((a, b) => boxes[a][0] - boxes[b][0] || boxes[a][1] - boxes[b][1] || a - b);
  }
  if (rows) for (let k = 0; k < parts.length - 1; k++) {
    if (parts[k].every(i => dots.has(i))) { [parts[k], parts[k + 1]] = [parts[k + 1], parts[k]]; k++; }
  }
  return parts.flatMap(part => arrange(part, boxes, dots, !rows, depth + 1));
}

function order(lines, width) {
  const boxes = lines.map(s => s.reduce((b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)], [Infinity, Infinity, -Infinity, -Infinity]));
  const dots = new Set(lines.flatMap((s, i) => s.length === 1 ? [i] : []));
  // A dot far from all other ink is a speck, not an i's dot or a \cdot.
  const heights = boxes.flatMap((b, i) => dots.has(i) ? [] : [b[3] - b[1]]).sort((a, b) => a - b);
  const specks = new Set();
  if (heights.length) {
    const reach = Math.max(3 * width, .6 * heights[Math.floor(heights.length / 2)]);
    const gap = (i, j) => Math.max(boxes[j][0] - boxes[i][2], boxes[i][0] - boxes[j][2], boxes[j][1] - boxes[i][3], boxes[i][1] - boxes[j][3], 0);
    for (const i of dots) if (boxes.every((_, j) => dots.has(j) || gap(i, j) > reach)) specks.add(i);
  }
  const keep = lines.map((_, i) => i).filter(i => !specks.has(i));
  return arrange(keep, boxes, dots).map(i => {
    const line = lines[i], dx = line.at(-1)[0] - line[0][0], dy = line.at(-1)[1] - line[0][1];
    const vertical = Math.abs(dx) <= width && Math.abs(dy) > Math.abs(dx);
    return (vertical && dy < 0) || (!vertical && dx < -width) ? [...line].reverse() : line;
  });
}

export function image_to_strokes(input) {
  if (!input.length) return [];
  const columns = input[0].length;
  if (input.some(row => row.length !== columns || Array.from(row).some(v => !Number.isFinite(v) || v < 0 || v > 255))) throw new Error('Expected a finite 2D gray image in 0..255');
  if (!columns) return [];
  const resized = resize(input), {w, h, scale} = resized;
  let gray = resized.gray;
  if (otsu(gray).reduce((a, b) => a + b, 0) > gray.length / 2) gray = gray.map(v => 255 - v);
  let ink = threshold(gray, w, h);
  if (ink.reduce((a, b) => a + b, 0) > ink.length / 2) ink = threshold(gray.map(v => 255 - v), w, h);
  for (const group of components(ink, w, h)) if (group.length < ink.length * .0002) for (const p of group) ink[p] = 0;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let p = 0; p < ink.length; p++) if (ink[p]) {
    x0 = Math.min(x0, p % w); x1 = Math.max(x1, p % w); y0 = Math.min(y0, Math.floor(p / w)); y1 = Math.max(y1, Math.floor(p / w));
  }
  if (x1 < 0) return [];
  x0 = Math.max(0, x0 - 4); y0 = Math.max(0, y0 - 4); x1 = Math.min(w, x1 + 5); y1 = Math.min(h, y1 + 5);
  const cw = x1 - x0, ch = y1 - y0, crop = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) crop.set(ink.subarray((y + y0) * w + x0, (y + y0) * w + x1), y * cw);
  const skeleton = thin(crop, cw, ch), count = skeleton.reduce((a, b) => a + b, 0);
  if (!count) return [];
  const width = 2 * crop.reduce((a, b) => a + b, 0) / count;
  let time = 0;
  return order(merge(trace(skeleton, cw, ch, width), width).map(simplify), width)
    .map(line => line.map(([x, y]) => [(x + x0) * scale[0], (y + y0) * scale[1], time++]));
}

// Separate lines of maths (a system of equations): rows with nothing joining them, each of 3 strokes or more, wide (half
// the ink's width), of similar height (0.6 of the tallest) with a clear gap (0.15 of the tallest) between. A fraction's bar, lim over
// x->3, or an i's dot are never lines of their own. Strokes keep their order inside a line. Mirrors photo.py split_lines.
export function split_lines(strokes) {
  if (strokes.length < 2) return [strokes];
  const boxes = strokes.map(s => s.reduce((b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)], [Infinity, Infinity, -Infinity, -Infinity]));
  const rows = bands(strokes.map((_, i) => i), boxes, 1);
  if (rows.length < 2) return [strokes];
  const box = ids => ids.reduce((b, i) => [Math.min(b[0], boxes[i][0]), Math.min(b[1], boxes[i][1]), Math.max(b[2], boxes[i][2]), Math.max(b[3], boxes[i][3])], [Infinity, Infinity, -Infinity, -Infinity]);
  const all = box(strokes.map((_, i) => i)), width = all[2] - all[0];
  // A long flat stroke is a fraction bar (touching its denominator, it looked like a line break): one formula.
  if (boxes.some(b => b[2] - b[0] >= .4 * width && b[3] - b[1] <= .15 * (b[2] - b[0]))) return [strokes];
  const rb = rows.map(box), tallest = Math.max(...rb.map(b => b[3] - b[1]));
  const lines = rb.every((b, k) => rows[k].length >= 3 && b[2] - b[0] >= .5 * width && b[3] - b[1] >= .6 * tallest) &&
    rb.slice(1).every((b, k) => b[1] - rb[k][3] >= .15 * tallest);
  return lines ? rows.map(ids => [...ids].sort((a, b) => a - b).map(i => strokes[i])) : [strokes];
}
