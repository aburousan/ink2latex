"""Regenerate the committed Python reference after deliberate feature-contract changes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink2latex.features import features

cases = [
    ("empty", [[], []]),
    ("dot", [[[20, 40, 0]]]),
    ("horizontal", [[[10, 5, 0], [30, 5, 1], [50, 5, 2]]]),
    ("vertical", [[[8, -3, 0], [8, 2, 1], [8, 12, 2]]]),
    ("multistroke", [[[5, 5, 0], [8, 11, 2], [10, 5, 3]], [], [[13, 8, 1], [17, 8, 9]], [[15, 4, 3]]]),
    ("duplicates", [[[0, 0, 0], [0, 0, 1], [1, 1, 2], [1, 1, 3], [2, 0, 4]]]),
    ("tiny_stroke", [[[0, 0], [1, 2]], [[3, 1], [3.001, 1.001]]]),
    ("capped", [[[0, 0, 0], [100, 1, 1]]]),
    ("tiny_scale", [[[0, 0], [1e-10, 1e-10]]]),
]
Path(__file__).with_name("fixtures").joinpath("features_case.json").write_text(json.dumps([
    {"name": name, "strokes": ink, "expected": features(ink).tolist()} for name, ink in cases
], separators=(",", ":")) + "\n")
