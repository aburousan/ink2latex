"""Generate small photo PNGs and Python reference strokes (Pillow when available)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink2latex.photo import image_to_strokes
from ink2latex.photo_io import draw_strokes, write_png

ROOT = Path(__file__).parent / "fixtures"
CASES = {
    "fraction": [[(40, 18), (70, 40)], [(70, 18), (40, 40)], [(25, 57), (95, 57)], [(45, 77), (70, 77), (45, 106), (72, 106)]],
    "x_squared": [[(22, 55), (65, 105)], [(65, 55), (22, 105)], [(80, 25), (92, 18), (103, 25), (80, 47), (105, 47)]],
    "crossed_t": [[(65, 18), (65, 108)], [(27, 53), (105, 53)]],
    "dotted_i": [[(62, 23)], [(62, 48), (62, 106)]],
}


def main():
    directory = ROOT / "photo_cases"
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for name, strokes in CASES.items():
        gray = draw_strokes(strokes, (132, 128), width=5)
        write_png(directory / f"{name}.png", gray)
        fixtures.append({"file": f"photo_cases/{name}.png", "strokes": image_to_strokes(gray)})
    (ROOT / "photo_strokes.json").write_text(json.dumps(fixtures, indent=2) + "\n")


if __name__ == "__main__":
    main()
