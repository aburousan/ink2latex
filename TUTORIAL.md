# ink2latex tutorial: read, run and train

ink2latex turns handwritten mathematics into LaTeX. It reads the pen's strokes, the
way a tablet records them, and it can also read a photo of a formula by first
turning the photo into strokes. Everything runs on your own computer.

This tutorial has four parts:

1. [Run the local app](#1-run-the-local-app): write with a mouse, pen or finger, or open a photo.
2. [Use it from the command line or Python](#2-use-it-from-the-command-line-or-python).
3. [Fine-tune it on your own handwriting](#3-fine-tune-it-on-your-own-handwriting).
4. [Train a model from scratch](#4-train-a-model-from-scratch).

Before you start, know what it can and cannot do:

- It reads one formula at a time, up to about 1,600 pen points and 160 LaTeX tokens.
- Written ink is read best. A photo has no pen strokes, so the app guesses them
  from the picture, and photos are read less well (see the numbers in the README).
- It knows the 243 LaTeX tokens in `demo/models/vocab.json`. Anything outside them
  (for example `\slashed`) it cannot write.
- Licence: the code is under the PolyForm Noncommercial License 1.0.0 and the
  trained weights under CC BY-NC-SA 4.0. Both mean: give credit, and no commercial
  use. See [LICENSE](LICENSE) and [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md).

## 1. Run the local app

You need Python 3 (only for its small built-in web server) and a modern browser.

```sh
git clone https://github.com/aburousan/ink2latex.git
cd ink2latex/demo
python3 -m http.server 8000
```

Open <http://localhost:8000> in Chrome, Edge, Firefox or Safari.

- **Write:** draw the formula in the big box. The LaTeX and the typeset formula
  appear about half a second after you stop. *Undo stroke* removes the last stroke,
  *Eraser* removes the strokes you touch and *Pen* writes again (a stylus eraser end
  works too), *Clear*
  starts again, *Copy* copies the LaTeX.
- **Photo:** press *Photo* and choose a picture (on a phone or tablet it can open
  the camera), paste a picture or screenshot with Ctrl+V (⌘V on a Mac) or the
  *Paste* button, or drag an image file onto the box. The app shows the strokes it found in the box, so you can see what
  it will read, and then reads them. You can *Undo* stray strokes before it reads.
  For the best result: one formula, dark pen on plain white paper, the formula
  filling most of the picture, taken straight on, in even light.
- **Offline:** after the first load the app works without a network. Nothing you
  write or photograph leaves your computer.

The app uses model r3-photo, whose two files (`encoder.onnx`, `decoder.onnx`,
about 40 MB together) are in `demo/models/`. To use r3 instead, unzip
`r3-onnx.zip` from the GitHub release into `demo/models/` and bump the cache
version in `demo/sw.js`. If you downloaded only the release, unzip it and run the same
command inside its `demo` folder.

## 2. Use it from the command line or Python

Make a virtual environment once:

```sh
cd ink2latex
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy onnxruntime pillow
```

### Command line

```sh
# A photo; also save a picture of the strokes it found, numbered in the order read
python3 scripts/predict.py --models demo/models --image formula.jpg --show-strokes traced.png

# Ink saved as JSON: [[[x, y, t], [x, y, t], ...], [...]]  (one list per stroke, y grows downwards)
python3 scripts/predict.py --models demo/models --ink formula.json

# Beam search (width 5) is a little more accurate and slower
python3 scripts/predict.py --models demo/models --ink formula.json --beam 5
```

The LaTeX is printed on standard output and the time taken on standard error.

### In your own Python code (ONNX, no PyTorch)

```python
from ink2latex.onnx_reader import OnnxReader
from ink2latex.photo import image_to_strokes
from ink2latex.photo_io import read_gray
from ink2latex.tokenizer import to_latex

reader = OnnxReader("demo/models")

# Pen strokes: a list of strokes, each a list of (x, y) or (x, y, t), y downwards.
strokes = [[(0, 0), (10, 12), (20, 0)], [(25, 5), (35, 5)]]
print(to_latex(reader.predict(strokes)))

# A photo
gray = read_gray("formula.jpg")          # 2D array, 0..255
print(to_latex(reader.predict(image_to_strokes(gray), beam=5)))
```

Units do not matter: the ink is scaled to a fixed height before reading. The
order of the strokes matters a little (write order is best).

### With the PyTorch checkpoint

The GitHub release also has `r3.pt` and `r3-photo.pt`, the PyTorch models. Use it to evaluate,
fine-tune or export again (`pip install torch` first):

```python
import torch
from ink2latex.training import load_checkpoint
from ink2latex.features import features
from ink2latex.inference import greedy
from ink2latex.tokenizer import to_latex

model, tokenizer, _ = load_checkpoint("r3.pt", "cpu")
model.eval()
points = torch.tensor(features(strokes))[None]
mask = torch.ones(points.shape[:2], dtype=torch.bool)
with torch.no_grad():
    ids = greedy(model, points, mask)[0]
print(to_latex(tokenizer.decode(ids)))
```

## 3. Fine-tune it on your own handwriting

Fine-tuning teaches the model your hand without forgetting what it knows. A few
hundred of your own formulas already help (the author's 59 lines raised exact
reading of his held-out lines from 20% to 35%). A GPU makes it fast; a CPU works
but is slow.

**Step 1. Collect your handwriting** as JSON lines, one formula per line:

```json
{"latex": "E = m c^{2}", "strokes": [[[12.0, 30.5, 0], [12.4, 31.0, 8]], [[40.1, 22.0, 120]]]}
```

`strokes` is a list of strokes, each a list of `[x, y, t]` points (t in
milliseconds, y downwards). Write the LaTeX the way the model does: braces on
every script and fraction part (`x^{2}`, `\frac{a}{b}`). Keep a tenth of your
lines aside for testing: give those rows `"meta": {"kind": "physics"}` and they
go to the held-out `user_physics_test` split instead of training.

**Step 2. Prepare the data.** Fine-tuning still needs the model's original
data mixed in, or it forgets other writers: download MathWriting (see
[DATA.md](DATA.md)), then

```sh
python3 scripts/prep.py --mathwriting data/mathwriting-2024 --out data/prepared \
  --user-jsonl data/my_handwriting.jsonl --vocab demo/models/vocab.json
```

`--vocab` matters: it encodes everything with the model's own 243 tokens, which
the fine-tuned model must keep (training refuses a mismatch). Rows of yours that
use a token outside it are dropped from training; prep reports how many.

**Step 3. Fine-tune from r3** (or from `r3-photo.pt` if you also want photos):

```sh
python3 scripts/train.py --data data/prepared --out runs/mine --device cuda \
  --init r3.pt --lr 5e-5 --warmup 200 --steps 4000 \
  --source-weight user=50 --workers 4
```

- `--init r3.pt` starts from r3's weights with a fresh optimizer.
- `--lr 5e-5 --warmup 200`: a small learning rate, so it adjusts rather than relearns.
- `--source-weight user=50` shows each of your lines 50 times per pass through the
  data; raise it if you have few lines, lower it if you have thousands.
- `runs/mine/best.pt` is the checkpoint that read MathWriting validation best.

**Step 4. Check it.** Compare on your held-out lines and on other writers:

```sh
python3 scripts/eval.py --checkpoint runs/mine/best.pt --data data/prepared --split user_physics_test --out runs/mine/mine.jsonl
python3 scripts/eval.py --checkpoint runs/mine/best.pt --data data/prepared --split test --out runs/mine/test.jsonl
```

Run the same two commands with `r3.pt` to see what changed. Expect your hand to
improve a lot and other writers to drop a little.

**Step 5. Use it in the app:**

```sh
python3 scripts/export_onnx.py --checkpoint runs/mine/best.pt --out demo/models
```

Then bump the cache version in `demo/sw.js`, reload the page, and the app uses
your model. Weights fine-tuned from r3 are an adaptation of it, so they keep its
licence (CC BY-NC-SA 4.0: credit, no commercial use, same licence when shared).

## 4. Train a model from scratch

This is how r3 was made. It needs a CUDA GPU (r3 took about two hours on one
NVIDIA A30, 80,000 steps) and MathWriting.

```sh
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements.txt
python3 scripts/prep.py --mathwriting data/mathwriting-2024 --out data/prepared \
  --user-jsonl data/my_handwriting.jsonl
python3 scripts/train.py --data data/prepared --out runs/new --device cuda \
  --steps 80000 --workers 4
```

The README describes the model, every training option, the source weights and
the data format in detail. To make a model that may be used commercially, train
only on data whose licence allows it: MathWriting does not (it is CC BY-NC-SA).

## Credit

If you use ink2latex or its weights, please credit:

- **ink2latex** by Kazi Abu Rousan, <https://github.com/aburousan/ink2latex>
- **MathWriting**: Gervais et al., "MathWriting: A Dataset For Handwritten
  Mathematical Expression Recognition", 2024, Google Research.
