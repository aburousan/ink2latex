"""Image I/O and synthetic rendering; Pillow is optional for the PNG-only fallback."""
import struct
import zlib
from pathlib import Path

import numpy as np


def read_gray(path):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return read_png(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        rgb = np.asarray(background)[:, :, :3].astype(np.int64)
        return ((299 * rgb[:, :, 0] + 587 * rgb[:, :, 1] + 114 * rgb[:, :, 2] + 500) // 1000).astype(np.uint8)


def read_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Without Pillow, use an 8-bit non-interlaced gray/RGB PNG")
    offset, chunks, header = 8, [], None
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        kind, body = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        if kind == b"IDAT":
            chunks.append(body)
        offset += size + 12
    if header is None:
        raise ValueError("Missing PNG header")
    w, h, depth, color, compression, filtering, interlace = header
    if depth != 8 or color not in (0, 2) or compression or filtering or interlace:
        raise ValueError("Without Pillow, use an 8-bit non-interlaced gray/RGB PNG")
    channels = 1 if color == 0 else 3
    raw = zlib.decompress(b"".join(chunks))
    stride = w * channels
    if len(raw) != h * (stride + 1):
        raise ValueError("Invalid PNG data length")
    rows = np.zeros((h, stride), np.uint8)
    for y in range(h):
        mode = raw[y * (stride + 1)]
        if mode > 4:
            raise ValueError("Invalid PNG filter")
        for x in range(stride):
            a = int(rows[y, x - channels]) if x >= channels else 0
            b = int(rows[y - 1, x]) if y else 0
            c = int(rows[y - 1, x - channels]) if y and x >= channels else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            value = (0, a, b, (a + b) // 2, predictor)[mode]
            rows[y, x] = (raw[y * (stride + 1) + x + 1] + value) % 256
    if channels == 1:
        return rows
    rgb = rows.reshape(h, w, 3).astype(np.int64)
    return ((299 * rgb[:, :, 0] + 587 * rgb[:, :, 1] + 114 * rgb[:, :, 2] + 500) // 1000).astype(np.uint8)


def write_png(path, gray):
    gray = np.asarray(gray, dtype=np.uint8)
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    h, w = gray.shape
    raw = b"".join(b"\0" + row.tobytes() for row in gray)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def draw_strokes(strokes, size, width=3):
    """Round pen, including isolated dots. size is (width, height)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
    if Image is not None:
        image = Image.new("L", size, 255)
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            xy = [tuple(p[:2]) for p in stroke]
            if len(xy) > 1:
                draw.line(xy, fill=0, width=width, joint="curve")
            for x, y in xy:
                r = width / 2
                draw.ellipse((x-r, y-r, x+r, y+r), fill=0)
        return np.array(image)
    image = np.full((size[1], size[0]), 255, np.uint8)
    r = width / 2
    for stroke in strokes:
        for a, b in zip(stroke, stroke[1:] or stroke):
            x0, y0 = np.maximum(0, np.floor(np.minimum(a[:2], b[:2]) - r)).astype(int)
            x1, y1 = np.minimum(size, np.ceil(np.maximum(a[:2], b[:2]) + r + 1)).astype(int)
            y, x = np.mgrid[y0:y1, x0:x1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = dx * dx + dy * dy
            t = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / norm, 0, 1) if norm else 0
            image[y0:y1, x0:x1][(x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2 <= r * r] = 0
    return image


def fit_strokes(strokes, height=120, margin=12):
    strokes = [s for s in strokes if len(s)]
    if not strokes:
        return [], (2 * margin + 1, 2 * margin + 1)
    xy = np.concatenate([np.asarray(s)[:, :2] for s in strokes])
    origin, span = xy.min(0), np.ptp(xy, axis=0)
    scale = height / max(float(span[1]), float(span[0]) if span[1] < 1e-6 else 0, 1)
    transformed = [((np.asarray(s)[:, :2] - origin) * scale + margin).tolist() for s in strokes]
    size = tuple((np.ceil(span * scale).astype(int) + 2 * margin + 1).tolist())
    return transformed, size


def show_strokes(strokes, path):
    strokes, size = fit_strokes(strokes, height=240, margin=24)
    gray = draw_strokes(strokes, size)
    # A tiny bitmap font keeps numbered debugging PNGs available without Pillow.
    digits = ['111101101101111', '010110010010111', '111001111100111', '111001111001111',
              '101101111001001', '111100111001111', '111100111101111', '111001001001001',
              '111101111101111', '111101111001111']
    for number, stroke in enumerate(strokes, 1):
        x, y = map(lambda v: max(0, int(v) - 10), stroke[0])
        for digit in str(number):
            for j, bit in enumerate(digits[int(digit)]):
                yy, xx = y + j // 3, x + j % 3
                if bit == '1' and yy < gray.shape[0] and xx < gray.shape[1]:
                    gray[yy, xx] = 80
            x += 4
    write_png(path, gray)


def blur_noise(gray, rng):
    kernel = np.exp(-np.arange(-3, 4, dtype=float) ** 2 / (2 * .8 ** 2))
    kernel /= kernel.sum()
    result = gray.astype(float)
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (3, 3)
        padded = np.pad(result, padding, mode="edge")
        result = sum(weight * np.take(padded, np.arange(result.shape[axis]) + i, axis=axis)
                     for i, weight in enumerate(kernel))
    return np.clip(np.floor(result + rng.normal(0, 6, result.shape) + .5), 0, 255).astype(np.uint8)
