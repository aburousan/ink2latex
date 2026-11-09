"""The deterministic eight-channel feature contract shared with demo/features.js."""

import numpy as np

STEP = 0.05
MAX_POINTS = 1600


def resample(stroke, step=STEP):
    xy = np.asarray(stroke, dtype=np.float64)[:, :2]
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    length = distances[-1]
    if length < step:
        return xy[:1].copy()
    # Remove repeated positions before interpolation, preserving the endpoints.
    keep = np.r_[True, np.diff(distances) > 0]
    distances, xy = distances[keep], xy[keep]
    targets = np.r_[np.arange(0, length, step), length]
    return np.stack([np.interp(targets, distances, xy[:, k]) for k in range(2)], axis=1)


def channels(strokes):
    rows = []
    for xy in strokes:
        if not len(xy):
            continue
        f = np.zeros((len(xy), 8), dtype=np.float64)
        f[:, :2] = xy
        f[:-1, 2:4] = np.diff(xy, axis=0)
        f[0, 4], f[-1, 5] = 1, 1
        f[:, 6:8] = xy.mean(axis=0) - xy
        rows.append(f)
    return np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 8), np.float32)


def features(strokes, *, return_capped=False):
    strokes = [np.asarray(s, dtype=np.float64)[:, :2] for s in strokes if len(s)]
    if not strokes:
        result = np.empty((0, 8), dtype=np.float32)
        return (result, False) if return_capped else result
    all_xy = np.concatenate(strokes)
    if not np.isfinite(all_xy).all():
        raise ValueError("Ink coordinates must be finite")
    origin = all_xy.min(axis=0)
    spans = np.percentile(all_xy, 95, axis=0) - np.percentile(all_xy, 5, axis=0)
    scale = max(spans[1] if spans[1] >= 1e-6 else spans[0], 1e-6)
    result = channels([resample((s - origin) / scale) for s in strokes])
    capped = len(result) > MAX_POINTS
    if capped:
        indices = np.floor(np.linspace(0, len(result) - 1, MAX_POINTS) + 0.5).astype(int)
        result = result[indices]
    return (result, capped) if return_capped else result


def augment(points, rng):
    """Augment resampled ink, rebuilding geometric channels after perturbation."""
    boundaries = [0] + [i for i in range(1, len(points))
                        if points[i, 4] or points[i - 1, 5]] + [len(points)]
    angle = rng.uniform(-5, 5) * np.pi / 180
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    affine = rotation @ np.array([[rng.uniform(.85, 1.15), rng.uniform(-.25, .25)],
                                  [0, rng.uniform(.85, 1.15)]])
    strokes = []
    for a, b in zip(boundaries, boundaries[1:]):
        xy = points[a:b, :2].astype(np.float64) @ affine.T
        if not len(xy):
            continue
        knots = rng.normal(0, .01, (max(2, (len(xy) + 7) // 8), 2))
        position = np.linspace(0, len(knots) - 1, len(xy))
        xy += np.stack([np.interp(position, np.arange(len(knots)), knots[:, k])
                        for k in range(2)], axis=1)
        keep = rng.random(len(xy)) >= rng.uniform(0, .1)
        keep[0] = True  # Even a dot survives.
        strokes.append(xy[keep])
    for i in range(0, len(strokes) - 1, 2):
        if rng.random() < .1:
            strokes[i], strokes[i + 1] = strokes[i + 1], strokes[i]
    return channels(strokes)
