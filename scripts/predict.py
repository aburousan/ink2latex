#!/usr/bin/env python3
"""Read a local photo or pen ink with exported ONNX models."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink2latex.onnx_reader import OnnxReader
from ink2latex.photo import image_to_strokes, split_lines
from ink2latex.photo_io import read_gray, show_strokes
from ink2latex.tokenizer import tidy, to_latex


def read_ink(path):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("--ink expects one ink JSONL row or one strokes array")
        value = rows[0]
    return value["strokes"] if isinstance(value, dict) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--ink")
    parser.add_argument("--beam", type=int, choices=(1, 5), default=1)
    parser.add_argument("--show-strokes", metavar="OUT.png")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        strokes = image_to_strokes(read_gray(args.image)) if args.image else read_ink(args.ink)
        if args.show_strokes:
            show_strokes(strokes, args.show_strokes)
        reader = OnnxReader(args.models)
        # The model reads one line of maths: separate lines (a system of equations) are read one by one.
        parts = [to_latex(tidy(reader.predict(line, args.beam))) for line in split_lines(strokes)]
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(parts[0] if len(parts) == 1 else "\\begin{gathered}" + "\\\\".join(parts) + "\\end{gathered}")
    print(f"{(time.perf_counter() - started) * 1000:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
