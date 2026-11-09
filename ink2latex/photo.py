"""Deterministic photo tracing, mirrored in demo/photo.js; NumPy is sufficient."""
import math
import numpy as np


def _resize(gray):
    h, w = gray.shape
    scale = min(1., 1200 / max(h, w))
    nh, nw = max(1, math.floor(h * scale)), max(1, math.floor(w * scale))
    if (nh, nw) == (h, w):
        return gray.astype(np.uint8), (1., 1.)
    # Integer overlap weights give exact area averages, rounding only at the end.
    horizontal = np.empty((h, nw), np.int64)
    for x in range(nw):
        a, b = x * w, (x + 1) * w
        ids = np.arange(a // nw, (b - 1) // nw + 1)
        weights = np.minimum(b, (ids + 1) * nw) - np.maximum(a, ids * nw)
        horizontal[:, x] = gray[:, ids] @ weights
    result = np.empty((nh, nw), np.uint8)
    for y in range(nh):
        a, b = y * h, (y + 1) * h
        ids = np.arange(a // nh, (b - 1) // nh + 1)
        weights = np.minimum(b, (ids + 1) * nh) - np.maximum(a, ids * nh)
        result[y] = (weights @ horizontal[ids] + h * w // 2) // (h * w)
    return result, (w / nw, h / nh)


def _threshold(gray):
    h, w = gray.shape
    r = max(1, min(h, w) // 20)
    integral = np.pad(gray.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y, x = np.arange(h), np.arange(w)
    ya, yb = np.maximum(0, y - r), np.minimum(h, y + r + 1)
    xa, xb = np.maximum(0, x - r), np.minimum(w, x + r + 1)
    total = (integral[yb[:, None], xb] - integral[ya[:, None], xb]
             - integral[yb[:, None], xa] + integral[ya[:, None], xa])
    area = (yb - ya)[:, None] * (xb - xa)
    flat = np.minimum(255, (gray.astype(np.int64) * 255 * area + total // 2) // np.maximum(1, total))
    return _otsu(flat)


def _otsu(gray):
    h, w = gray.shape
    hist = np.bincount(gray.ravel(), minlength=256)
    count, weighted = 0, 0
    full = int(sum(i * int(n) for i, n in enumerate(hist)))
    best, threshold = -1., 0
    for i in range(256):
        count += int(hist[i])
        weighted += i * int(hist[i])
        if count == 0 or count == h * w:
            continue
        score = (full * count - weighted * h * w) ** 2 / (count * (h * w - count))
        if score > best:
            best, threshold = score, i
    return gray <= threshold if best >= 0 else np.zeros_like(gray, dtype=bool)


def _neighbors(p, w, h):
    y, x = divmod(p, w)
    return [yy * w + xx for yy in range(max(0, y - 1), min(h, y + 2))
            for xx in range(max(0, x - 1), min(w, x + 2)) if xx != x or yy != y]


def _components(mask):
    h, w = mask.shape
    remaining = set(map(int, np.flatnonzero(mask)))
    groups = []
    for first in sorted(remaining):
        if first not in remaining:
            continue
        remaining.remove(first)
        group = [first]
        for p in group:
            for q in _neighbors(p, w, h):
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
        groups.append(group)
    return groups


def _thin(mask):
    a = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in range(2):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] != 0)).astype(np.uint8) for i in range(8))
            if phase == 0:
                clear = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                clear = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (a[1:-1, 1:-1] != 0) & (count >= 2) & (count <= 6) & (transitions == 1) & clear
            if remove.any():
                a[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1].astype(bool)


def _length(line):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line, line[1:]))


def _trace(skeleton, width):
    h, w = skeleton.shape
    pixels = sorted(map(int, np.flatnonzero(skeleton)))
    present = set(pixels)
    adjacent = {p: [q for q in _neighbors(p, w, h) if q in present] for p in pixels}
    visited, lines = set(), []
    # A short whole connected component is an isolated dot, never a junction spur.
    dots = set()
    for group in _components(skeleton):
        if len(group) <= 1.5 * width:
            first = min(group)
            lines.append([(first % w, first // w)])
            dots.update(group)
    visited.update(dots)
    for start in [p for p in pixels if len(adjacent[p]) == 1] + pixels:
        if start in visited:
            continue
        path, p, previous = [], start, None
        while True:
            path.append((p % w, p // w))
            visited.add(p)
            choices = [q for q in adjacent[p] if q not in visited]
            # Through junctions along the straightest branch: cut at every junction, a letter became several
            # dashes and the reader saw "1 1 1" (0% on synthetic photos; 10% with this on the owner's lines).
            if not choices:
                break
            if previous is not None:
                dx, dy = p % w - previous % w, p // w - previous // w
                def straight(q):
                    qx, qy = q % w - p % w, q // w - p // w
                    return (dx * qx + dy * qy) / math.hypot(qx, qy)
                choices.sort(key=straight, reverse=True)
            previous, p = p, choices[0]
        if _length(path) >= 1.5 * width:
            lines.append(path)
    return lines


def _merge(lines, width):
    cosine = math.cos(math.radians(40))
    while True:
        best = None
        for i, a in enumerate(lines):
            if len(a) < 2:
                continue
            for j in range(i + 1, len(lines)):
                b = lines[j]
                if len(b) < 2:
                    continue
                for ra in (False, True):
                    aa = a[::-1] if ra else a
                    for rb in (False, True):
                        bb = b[::-1] if rb else b
                        distance = math.hypot(aa[-1][0] - bb[0][0], aa[-1][1] - bb[0][1])
                        if distance > 1.5 * width:
                            continue
                        # A short tangent window ignores the last pixel's staircase direction.
                        k = min(len(aa) - 1, max(1, math.floor(width)))
                        l = min(len(bb) - 1, max(1, math.floor(width)))
                        u = (aa[-1][0] - aa[-1-k][0], aa[-1][1] - aa[-1-k][1])
                        v = (bb[l][0] - bb[0][0], bb[l][1] - bb[0][1])
                        norm = math.hypot(*u) * math.hypot(*v)
                        if norm and (u[0] * v[0] + u[1] * v[1]) / norm > cosine:
                            if best is None or distance < best[0]:
                                best = (distance, i, j, aa + bb)
        if best is None:
            return lines
        _, i, j, line = best
        lines[i] = line
        del lines[j]


def _simplify(line):
    if len(line) <= 2:
        return line
    keep, stack = {0, len(line) - 1}, [(0, len(line) - 1)]
    while stack:
        a, b = stack.pop()
        dx, dy = line[b][0] - line[a][0], line[b][1] - line[a][1]
        norm = dx * dx + dy * dy
        maximum, index = .25, None
        for i in range(a + 1, b):
            x, y = line[i][0] - line[a][0], line[i][1] - line[a][1]
            t = max(0, min(1, (x * dx + y * dy) / norm)) if norm else 0
            distance = (x - t * dx) ** 2 + (y - t * dy) ** 2
            if distance > maximum:
                maximum, index = distance, i
        if index is not None:
            keep.add(index)
            stack.extend([(a, index), (index, b)])
    return [line[i] for i in sorted(keep)]


def _bands(ids, boxes, axis):
    """Groups of `ids` whose boxes overlap along `axis` (1: rows, by y; 0: columns, by x), in order along it.
    Columns need a substantial overlap (half the narrower), so a superscript beside its base stays apart."""
    lo, hi = (1, 3) if axis else (0, 2)
    groups = []
    for i in sorted(ids, key=lambda i: (boxes[i][lo], boxes[i][1 - axis], i)):
        a0, a1 = boxes[i][lo], boxes[i][hi]
        def joins(g):
            if axis:
                return a0 <= g[1] and a1 >= g[0]
            return min(a1, g[1]) - max(a0, g[0]) > .5 * min(a1 - a0, g[1] - g[0])
        group = [a0, a1, [i]]
        for g in [g for g in groups if joins(g)]:
            group[0], group[1] = min(group[0], g[0]), max(group[1], g[1])
            group[2].extend(g[2])
            groups.remove(g)
        groups.append(group)
    return [g[2] for g in sorted(groups, key=lambda g: (g[0], g[1]))]


def _arrange(ids, boxes, dots, rows=True, depth=0):
    """Writing order: rows top to bottom, inside a row columns left to right, inside a column its stacked parts
    (lim over x->3, numerator over bar over denominator) top to bottom, and so on. A row of dots alone goes after
    the row under it (an i is written stem first). Sorting all by x read the x of x->3 before the l of lim."""
    if len(ids) <= 1 or depth > 12:
        return list(ids)
    parts = _bands(ids, boxes, 1 if rows else 0)
    if len(parts) == 1:
        rows = not rows
        parts = _bands(ids, boxes, 1 if rows else 0)
        if len(parts) == 1:
            return sorted(ids, key=lambda i: (boxes[i][0], boxes[i][1], i))
    if rows:
        k = 0
        while k < len(parts) - 1:
            if all(i in dots for i in parts[k]):
                parts[k], parts[k + 1] = parts[k + 1], parts[k]
                k += 1
            k += 1
    return [i for part in parts for i in _arrange(part, boxes, dots, not rows, depth + 1)]


def _order(lines, width):
    boxes = [(min(x for x, y in s), min(y for x, y in s), max(x for x, y in s), max(y for x, y in s)) for s in lines]
    dots = {i for i, s in enumerate(lines) if len(s) == 1}
    # A dot far from all other ink is a speck (dust, a shadow's edge), not an i's dot or a \cdot.
    heights = sorted(boxes[i][3] - boxes[i][1] for i in range(len(lines)) if i not in dots)
    if heights:
        reach = max(3 * width, .6 * heights[len(heights) // 2])
        def gap(i, j):
            return max(boxes[j][0] - boxes[i][2], boxes[i][0] - boxes[j][2], boxes[j][1] - boxes[i][3], boxes[i][1] - boxes[j][3], 0)
        specks = {i for i in dots if all(gap(i, j) > reach for j in range(len(lines)) if j not in dots)}
    else:
        specks = set()
    keep = [i for i in range(len(lines)) if i not in specks]
    result = []
    for i in _arrange(keep, boxes, dots):
        line = lines[i]
        dx, dy = line[-1][0] - line[0][0], line[-1][1] - line[0][1]
        vertical = abs(dx) <= width and abs(dy) > abs(dx)
        if (vertical and dy < 0) or (not vertical and dx < -width):
            line = line[::-1]
        result.append(line)
    return result


def image_to_strokes(gray):
    """Return (x, y, cumulative index) in original image coordinates, y down."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or not np.isfinite(gray).all() or (gray < 0).any() or (gray > 255).any():
        raise ValueError("Expected a finite 2D gray image in 0..255")
    if not gray.size:
        return []
    gray, scale = _resize(np.floor(gray + .5).astype(np.int64))
    # Determine polarity before division can turn a dark board into a light background.
    if _otsu(gray).sum() > gray.size / 2:
        gray = 255 - gray
    ink = _threshold(gray)
    if ink.sum() > ink.size / 2:
        ink = _threshold(255 - gray)
    for group in _components(ink):
        if len(group) < ink.size * .0002:
            ink.flat[group] = False
    y, x = np.nonzero(ink)
    if not len(x):
        return []
    x0, y0 = max(0, int(x.min()) - 4), max(0, int(y.min()) - 4)
    ink = ink[y0:int(y.max()) + 5, x0:int(x.max()) + 5]
    skeleton = _thin(ink)
    if not skeleton.any():
        return []
    width = 2 * int(ink.sum()) / int(skeleton.sum())
    lines = _order([_simplify(s) for s in _merge(_trace(skeleton, width), width)], width)
    result, time = [], 0
    for line in lines:
        stroke = []
        for x, y in line:
            stroke.append(((x + x0) * scale[0], (y + y0) * scale[1], time))
            time += 1
        result.append(stroke)
    return result


def split_lines(strokes):
    """Separate lines of maths (a system of equations): rows with nothing joining them, each of 3 strokes or more,
    wide (half the ink's width), of similar height (0.6 of the tallest) with a clear gap (0.15 of the tallest) between. A fraction's
    bar, lim over x->3, or an i's dot are never lines of their own. Strokes keep their order inside a line."""
    if len(strokes) < 2:
        return [strokes]
    boxes = [(min(p[0] for p in s), min(p[1] for p in s), max(p[0] for p in s), max(p[1] for p in s)) for s in strokes]
    rows = _bands(range(len(strokes)), boxes, 1)
    if len(rows) < 2:
        return [strokes]
    def box(ids):
        return (min(boxes[i][0] for i in ids), min(boxes[i][1] for i in ids), max(boxes[i][2] for i in ids), max(boxes[i][3] for i in ids))
    whole = box(range(len(strokes))); width = whole[2] - whole[0]
    # A long flat stroke is a fraction bar (touching its denominator, it looked like a line break): one formula.
    if any(b[2] - b[0] >= .4 * width and b[3] - b[1] <= .15 * (b[2] - b[0]) for b in boxes):
        return [strokes]
    rb = [box(ids) for ids in rows]; tallest = max(b[3] - b[1] for b in rb)
    lines = all(len(ids) >= 3 and b[2] - b[0] >= .5 * width and b[3] - b[1] >= .6 * tallest for ids, b in zip(rows, rb)) and \
        all(rb[k + 1][1] - rb[k][3] >= .15 * tallest for k in range(len(rb) - 1))
    return [[strokes[i] for i in sorted(ids)] for ids in rows] if lines else [strokes]
