#!/usr/bin/env python3
"""Compare stored ink and a synthetic photo of the same rows, on CPU."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink2latex.data import ShardDataset
from ink2latex.onnx_reader import OnnxReader
from ink2latex.photo import image_to_strokes
from ink2latex.photo_io import blur_noise, draw_strokes, fit_strokes
from ink2latex.tokenizer import detokenize


def reconstruct_strokes(points):
    strokes = []
    for i, p in enumerate(points):
        if not strokes or p[4] or (i and points[i - 1, 5]):
            strokes.append([])
        strokes[-1].append([float(p[0]), float(p[1])])
    return strokes


def metrics(references, predictions):
    errors = 0
    for a, b in zip(references, predictions):
        row = list(range(len(b) + 1))
        for i, token in enumerate(a, 1):
            new = [i]
            for j, other in enumerate(b, 1):
                new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (token != other)))
            row = new
        errors += row[-1]
    return {"samples": len(references),
            "exact_match": sum(a == b for a, b in zip(references, predictions)) / len(references) if references else None,
            "token_error_rate": errors / max(1, sum(map(len, references))) if references else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Per-row JSONL (default: photo-SPLIT.jsonl in the current directory)")
    args = parser.parse_args()
    if args.rows is not None and args.rows < 0:
        parser.error("--rows must be nonnegative")
    started = time.perf_counter()
    data = ShardDataset(args.data, args.split, max_cached=2)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(data))
    if args.rows is not None and args.rows < len(data):
        indices = np.sort(rng.choice(indices, args.rows, replace=False))
    reader = OnnxReader(args.models)
    references, originals, photos, empty = [], [], [], 0
    output = Path(args.out or f"photo-{args.split}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index in indices:
            points, _, metadata = data[int(index)]
            strokes, size = fit_strokes(reconstruct_strokes(points))
            photo = blur_noise(draw_strokes(strokes, size, width=3), rng)
            traced = image_to_strokes(photo)
            original, prediction = reader.predict_points(points), reader.predict(traced)
            references.append(metadata["tokens"])
            originals.append(original)
            photos.append(prediction)
            empty += not traced
            stream.write(json.dumps({**metadata, "row": int(index), "original_tokens": original,
                                     "original": detokenize(original), "photo_tokens": prediction,
                                     "photo": detokenize(prediction), "photo_empty": not traced}, ensure_ascii=False) + "\n")
    report = {"split": args.split, "seed": args.seed, "original": metrics(references, originals),
              "photo": metrics(references, photos), "photo_empty": empty,
              "photo_empty_rate": empty / len(indices) if len(indices) else None,
              "seconds": time.perf_counter() - started, "predictions": str(output)}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
