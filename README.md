# ink2latex

Handwritten mathematics to LaTeX, on your own computer. Write a formula with a pen,
finger or mouse, or open a photo of one, and get its LaTeX. The model reads pen
strokes directly (a relative 2D transformer encoder and an attention-refinement
decoder); a photo is first turned into strokes.

**Try it now:** <https://aburousan.github.io/ink2latex/> (nothing to install).

**New here?** [GETTING_STARTED.md](GETTING_STARTED.md) shows step by step, for
beginners, how to run and use it. [TUTORIAL.md](TUTORIAL.md) covers using it from
Python, fine-tuning it on your own handwriting, and training a model.

```sh
git clone https://github.com/aburousan/ink2latex.git
cd ink2latex/demo && python3 -m http.server 8000      # then open http://localhost:8000
```

## Models

Both models have d 256, 8 heads, 6 encoder and 4 decoder layers, up to 1,600 points
and 160 tokens, and the same 243 LaTeX tokens. [DATA.md](DATA.md) lists everything
they were trained on.

- **r3** was trained from scratch for 80,000 steps on one A30 (about two hours).
- **r3-photo** is r3 fine-tuned for 8,000 more steps with 80,000 photo-style copies
  of its training formulas mixed in (each drawn as a picture with a random pen,
  size, slant, blur, noise and light, then traced back into strokes). It reads
  photos about three times better and pen strokes as well as r3. **The app in
  `demo/` uses r3-photo.**

Fully right (every LaTeX token matches) with beam 5, and the token error rate
(inserted, deleted and wrong tokens per reference token) in brackets:

| Held-out set, pen strokes | r3 | r3-photo |
| --- | --- | --- |
| MathWriting test (7,644 rows, other writers) | 59.2% (6.2%) | 59.4% (6.3%) |
| The author's notes (119 rows) | 37.0% (7.9%) | 37.0% (7.9%) |
| The author's Samsung Notes lines (20 rows) | 40.0% (16.6%) | 30.0% (13.1%) |
| A physics notebook (47 rows) | 57.4% (10.2%) | 68.1% (9.6%) |
| Physics test (10 rows) | 50.0% (3.7%) | 60.0% (2.7%) |
| Legacy notes (16 rows) | 62.5% (3.8%) | 68.8% (4.2%) |

Photos, greedy decoding. These are synthetic photos: each held-out ink drawn as a
picture (3 px pen, light blur and noise) and read back from the picture. Real
camera photos (shadows, paper texture, perspective) have not been measured and
will likely be read worse.

| Held-out set, synthetic photos | r3 | r3-photo |
| --- | --- | --- |
| MathWriting test (300 rows) | 7.7% (29.4%) | 26.7% (16.6%) |
| A physics notebook (47 rows) | 4.3% (39.4%) | 27.7% (17.3%) |
| The author's notes (119 rows) | 4.2% (49.7%) | 15.1% (23.5%) |
| The author's Samsung Notes lines (20 rows) | 10.0% (52.5%) | 15.0% (37.0%) |

The GitHub release has, for each model: the PyTorch checkpoint (`r3.pt`,
`r3-photo.pt`, weights only), the ONNX files for the app (`r3-onnx.zip`,
`r3-photo-onnx.zip`) and a compact half-precision weights file (`*-ink.bin`).

## Licence and credit

- **Code:** PolyForm Noncommercial License 1.0.0 ([LICENSE](LICENSE)). Use, change
  and share it for any noncommercial purpose; keep the `Required Notice` line.
- **Trained weights:** CC BY-NC-SA 4.0 ([WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md)),
  because they were trained on MathWriting: credit, no commercial use, and share
  adaptations under the same licence.
- Please credit **ink2latex by Kazi Abu Rousan** and **MathWriting** (Gervais et
  al., 2024, Google Research).

## Technical reference

The rest of this file documents the code in detail.

## Local CPU smoke test

Use Python 3 in a project virtual environment. Nothing needs a global install.
On macOS, the PyPI torch wheel supports CPU execution:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install torch
python3 -m unittest discover -s tests -v
node tests/test_features_js.mjs
```

On Linux without CUDA, replace the torch install command with
`python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
The tests explicitly use CPU, one thread, generated inks and a 64-dimensional
2+2-layer model. They prepare all source formats, train 30 steps, evaluate
greedy/beam and legacy data, export ONNX, check dynamic-shape numerical parity,
verify a resumed run reproduces identical parameters, and check random-prefix
parallel/incremental equivalence, relative coverage and source sampling.
All artifacts are temporary. The suite is designed to finish in under two minutes; dependency
installation is outside that budget.

Node is needed only for the JS parity check. `tests/fixtures/features_case.json`
contains nine Python reference inks; regenerate deliberately with
`python3 tests/generate_features_fixture.py` after changing both implementations.
An optional browser integration check, after exporting models, is
`node tests/test_demo.mjs demo` (Node 22+ and Chrome; set `CHROME_PATH` outside
macOS). It creates an isolated browser profile and local server, exercises
recognition/undo/clear, stops the server, disables the network and verifies
offline reload and recognition. It installs no npm packages.

## Prepare data

Obtain the data separately. See [DATA.md](DATA.md) for licences and split rules.
Preparation itself only needs NumPy, not PyTorch.

```sh
python3 scripts/prep.py --mathwriting data/mathwriting-2024 --out data/prepared \
  --user-jsonl data/notes.jsonl --extra-pkl old=data/old-project.pkl \
  --test-pkl legacy=data/legacy-16.pkl
```

The optional flags each accept multiple files. Use a fresh output directory.
MathWriting directories are `train`, `synthetic`, `symbols`, `valid`, and `test`.
`normalizedLabel` takes precedence over `label`. User pressure/time channels
are ignored. User rows accept either `latex` or `label` (studio single symbols);
`latex` takes precedence if both are present, and `meta.kind == "physics"`
always goes to `user_physics_test`.

Extra pickle labels may be token lists (joined with spaces) or strings, with
strokes supplied as lists or NumPy `(n,2)` / `(n,3)` arrays. `--extra-pkl PATH`
uses source `extra/<file stem>`; `--extra-pkl NAME=PATH` uses `extra/NAME`.
`--test-pkl NAME=PATH` creates a separate evaluation split named `NAME`, excluded
from training and vocabulary construction. Names use letters, digits, `_` or
`-` and must not collide with existing split names. Evaluate it with `--split NAME`.

Preparation makes two streaming passes: vocabulary construction, then feature
extraction and bounded 512-row shard writes. The default minimum token count
is 20; `--min-count` and `--shard-size` are configurable for experiments.
Vocabulary counts come from all raw training-source labels, before filtering
and before 50× user sampling. Validation, physics and test labels never affect
the vocabulary. Training rows with rare tokens are dropped. Held-out unknown
tokens are preserved verbatim in metadata and stored as `-1` token IDs so those
rows remain in exact-match/error-rate denominators; they never enter training.
All splits drop empty ink, empty labels and labels longer than 160 tokens.

Each `.npz` contains float16 `points [N,8]`, int16 `tokens [M]`, and int64
`point_offsets` / `token_offsets`, including a terminal offset. Labels exclude
BOS/EOS in storage. JSONL sidecars retain source, ID and reference tokens.
`manifest.json` records split shard names and per-row lengths and sources.
Runtime loading keeps at most two decompressed shards cached. `prep_report.json` records counts,
drops, capping, median stored points and longest kept/observed label lengths.

## Train on the CUDA A30 server

Use the server's PyTorch 2.7 / CUDA 12.8 environment. If creating one there:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements.txt
python3 scripts/train.py --data data/prepared --out runs/a30 \
  --device cuda --steps 100000 --seed 42
```

Defaults: d=256, 8 heads, FF=1024, six encoder layers, four decoder layers,
dropout .1; AdamW 3e-4, betas (.9,.98), weight decay .05; 2,000 warmup steps,
cosine decay to 1e-5; CUDA bf16; clip norm 1; label smoothing .1. Length buckets
respect a 24,000 **padded input-point** budget, conservatively bounding memory.
Mixture defaults per epoch are:

| Source prefix | Weight |
| --- | ---: |
| `mathwriting/train` | 1 |
| `mathwriting/synthetic` | 0.6 |
| `mathwriting/symbols` | 1 |
| `extra` | 0.5 |
| `user` | 50 |

For weight W, each row appears floor(W) times plus one more with probability
frac(W). Thus weights below 1 retain each row with probability W. Sampling is
seeded by epoch. Override defaults with repeatable `--source-weight PREFIX=W`;
the longest matching prefix wins, unmatched sources have weight 1, and zero
excludes a source. For example:

```sh
python3 scripts/train.py --data data/prepared --out runs/custom --device cuda \
  --source-weight mathwriting/synthetic=0.3 --source-weight extra/old=2.5 \
  --source-weight user=20
```

`source_counts.jsonl` records effective counts for every sampled epoch, including
zero counts and skipped empty epochs. Counts also appear on the first training
record of each epoch (and on resume). All-zero mixtures are rejected. Prepared
v1 shards remain readable; old `pickle:` source tags are mapped to `extra/` for
sampling. Mixture weights do not duplicate stored data or affect evaluation.
Train-only affine, smooth jitter, point dropping and neighboring stroke swaps run before padding.

`train.jsonl` contains step, epoch, total/CE/two coverage losses, the fraction of
valid ink tokens with relative coverage below .25, LR, gradient norm,
output tokens/s, input points/s and validation metrics. Every 5,000 steps saves
a checkpoint and evaluates a seeded, fixed subset of up to 2,000 valid rows
with greedy decoding. Final-step evaluation also runs. `best.pt` uses exact
match; `last.pt` saves the final state. `config.json` and `vocab.json` accompany
the checkpoints.

```sh
python3 scripts/train.py --data data/prepared --out runs/a30 --device cuda \
  --steps 100000 --seed 42 --resume runs/a30/step-0005000.pt
```

Resume saves the optimizer, sampler epoch/cursor, augmentation and torch RNGs,
fixed valid subset and best score. Keep the data, seed, token budget, validation
settings, `--source-weight` overrides and total `--steps` unchanged for exact
continuation. Checkpoints record the effective weights and reject mismatches.
Changing total steps intentionally changes the remaining cosine schedule. `--steps` is the
total target, not additional steps. JSON `--config` overrides model defaults;
`--tiny` selects the CPU smoke architecture. Don't run full training on the
8-GB laptop. V1 model checkpoints need retraining: v2 removes the coverage
context projection and adds per-head refinement convolutions.

## Evaluate and export

```sh
python3 scripts/eval.py --data data/prepared --checkpoint runs/a30/best.pt \
  --split test --beam 1 --out runs/a30/test-greedy.jsonl
python3 scripts/eval.py --data data/prepared --checkpoint runs/a30/best.pt \
  --split test --beam 5 --out runs/a30/test-beam5.jsonl
python3 scripts/eval.py --data data/prepared --checkpoint runs/a30/best.pt \
  --split user_physics_test --out runs/a30/physics.jsonl
python3 scripts/eval.py --checkpoint runs/a30/best.pt \
  --legacy-pkl data/legacy-16.pkl --out runs/a30/legacy.jsonl
python3 scripts/export_onnx.py --checkpoint runs/a30/best.pt \
  --data data/prepared --out demo/models
```

Evaluation also accepts `--split valid`, any named held-out split, and
`--device cuda`. Prediction JSONL contains reference and predicted tokens, LaTeX, source and ID; the adjacent
`.metrics.json` gives exact match, corpus token edit distance divided by total
reference tokens, and loose exact match ignoring braces and `\left`/`\right`.
Beam search ranks cumulative log probabilities without length normalization.
`--max-decode-tokens` exists for smoke tests; normal decoding is capped at 160.

Export uses opset 17 and checks ONNX Runtime CPU against PyTorch across several
batch/ink/prefix shapes, including a dot and a padded batch. It also checks the
incremental PyTorch decoder against full-prefix recomputation. The report saves
maximum absolute differences and fp32/int8 candidate sizes. Dynamic int8
quantizes linear weights, leaving convolutions float32. It is published only
if exact match drops by at most 0.5 percentage points on a seeded 1,000-row valid
subset. Without 1,000 rows, the candidate is measured then discarded and the
report states why; no accuracy result is fabricated. `--no-quantize` skips it.

ONNX tensor contract (batch B, points P, reduced ink S=ceil(P/4), prefix T):

| Graph | Inputs | Outputs |
| --- | --- | --- |
| Encoder | float32 `points [B,P,8]`, bool `point_mask [B,P]` | float32 `memory [B,S,d]`, bool `memory_mask [B,S]` |
| Decoder | int64 `previous_tokens [B,T]`, memory, memory mask | float32 `logits [B,V]` (last prefix step) |

Start the prefix with BOS=1, append each chosen token, and stop at EOS=2.
Coverage is recomputed inside the parallel decoder on every growing-prefix call;
there is no coverage input or output. PAD=0 and BOS are suppressed at generation.
The demo uses fp32 by default. Replace both graphs together when upgrading v1
assets; the service-worker cache version is now `ink2latex-v9`.

## Static demo and offline use

Photos are converted into estimated pen strokes and are read less well than
written ink. Use one formula, dark pen on plain paper, filling the picture.
The **Photo** button, pasting a picture (Ctrl+V / ⌘V anywhere, or the **Paste**
button) and dropping an image file on the box all replace the drawing with the
traced strokes, centered in the canvas. Undo removes individual strokes; automatic recognition still applies.
Clear removes the photo note. Images are processed locally and never uploaded.

Read a photo or a single ink JSONL row (or a raw strokes array):

```sh
./.venv/bin/python scripts/predict.py --models demo/models --image formula.png --show-strokes traced.png
./.venv/bin/python scripts/predict.py --models demo/models --ink formula.json --beam 5
```

LaTeX goes to stdout and elapsed milliseconds to stderr. The numbered PNG shows
the estimated stroke order. Inference uses NumPy and ONNX Runtime on CPU, with
the same features and greedy prefix decoding as the demo; beam 5 ranks cumulative
log probabilities. Empty photos produce empty LaTeX.

Photo tracing uses an integral box blur, Otsu threshold, component filtering,
NumPy Zhang–Suen thinning, tracing that continues through junctions along the
straightest branch, crossing joins and RDP simplification.
It needs no scikit-image. Coordinates are restored to the original image scale;
rounding and traversal order match the browser implementation. Pillow supports
common image formats and EXIF orientation when installed. Without Pillow, a
standard-library fallback reads 8-bit non-interlaced gray/RGB PNGs and generates
the fixtures, synthetic photos and numbered debugging PNGs. No package is
installed automatically.

Compare original stored features against synthetic photographs of the same rows:

```sh
./.venv/bin/python scripts/eval_photo.py --models demo/models --data data/prepared --split test --rows 200 --seed 0 --out photo-test.jsonl
```

The evaluator reconstructs strokes from feature start/end flags, renders at
about 120 px ink height with a round 3 px pen, applies sigma-0.8 Gaussian blur
and sigma-6 gray-level noise, and traces the result. It reports exact match,
token error rate and empty-photo frequency. Sampling and noise are seeded.
Per-row JSONL contains both predictions and the reference; adjacent
`.metrics.json` stores the aggregate report. The default output is
`photo-SPLIT.jsonl` in the current directory. Both paths use CPU greedy decoding.
These synthetic results do not measure real-camera accuracy.

Run the photo checks with `./.venv/bin/python -m unittest discover -s tests -p
test_photo.py` and `node tests/test_photo_js.mjs`. Regenerate the four small
fixtures deliberately with `./.venv/bin/python tests/generate_photo_fixture.py`.
The full smoke test also runs prediction and photo evaluation against its
temporary tiny ONNX export, leaving published models untouched.

```sh
python3 -m http.server 8000 --directory demo
```

Open `http://localhost:8000`. Export trained models first to enable recognition.
For GitHub Pages, serve the contents of `demo/` at the site root or a subpath;
all local references are relative. No build tool or backend is required.

The canvas supports mouse, touch and pen, undo, clear, explicit recognition and
600-ms idle recognition. It displays copyable LaTeX, KaTeX and elapsed inference
time. Pressure is ignored. Stale inference results are discarded after edits.
ONNX Runtime Web **1.22.0** and KaTeX **0.16.22** are pinned on jsDelivr. The
[runtime JS and WASM builds must match](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html);
the service worker caches those exact builds, KaTeX fonts, shell and loaded
models. Only local static assets and those CDN assets are requested; no ink is
uploaded, no analytics are used, and no external fonts are loaded.

Offline use requires HTTPS or localhost, service workers, available cache
storage, and one successful initial online load with models installed. Wait for
“Available offline” before disconnecting. Clearing browser storage removes the
offline copy. Increment `CACHE` in `demo/sw.js` when releasing changed assets or
weights. Update source credits in DATA.md and the About box for actual training
data, and distribute the weights licence with exported assets.

## Implementation choices

- The feature count is **eight**, resolving the spec's “7 channels” typo. Y
  increases downward, percentiles use linear interpolation, and a short stroke
  becomes its first point. Stroke centers are means of resampled points.
  Features are computed before uniform, endpoint-preserving capping; capped
  rows keep their original channels. Time and pressure are discarded.
- Convolutions may mix across strokes; start/end flags are retained. Padding
  masks are applied at every reduction/layer. Reduced centers use masked local
  averages. Relative offsets are clipped to ±8 and quantized in .25 increments;
  one shared 2→32→heads MLP evaluates this grid for all encoder layers.
- Training decodes all target positions in one pass with causal self-attention.
  Each layer computes masked first attention `a`, then exclusive cumulative
  coverage `c_t = sum(a_u, u < t)` per head. A depthwise kernel-5 convolution
  along ink produces bias `-softplus(w_h) * relu(conv(c_t))`; the final softmax
  uses this bias. Convolutions start as local averages and penalty weights at 1.
  Refined attention supplies the values passed to the next layer and the last
  layer's head-mean coverage used by the loss. Padding contributes zero.
- Incremental greedy/beam inference caches self-attention keys/values and each
  layer's running **first-attention** coverage `[B,H,S]`, as required to match
  the parallel exclusive sum. The next layer computes its first attention from
  hidden states updated with the preceding layer's refined values. Export
  recomputes this same calculation from the full prefix, with no state contract.
- Relative coverage is `chat = cov_T * S_valid / T_valid`, including the
  EOS-prediction step and excluding padded targets. Loss is CE plus
  `.05 * mean(relu(.25 - chat)) + .05 * mean(relu(chat - 4))`, averaging only
  valid ink tokens. Uniform coverage has zero penalties even when T < S.
- Adjacent stroke swaps use disjoint two-stroke windows. Augmentation rebuilds
  geometric channels and never removes an entire represented stroke. For the
  rare capped sample, surviving start/end flags delimit represented fragments.
- Token detokenization inserts spaces to preserve command boundaries
  (`\alpha x` versus `\alphax`). Tokenization ignores those spaces.

The architecture is inspired by the public description of MyScript iink 3.0
math recognition; no MyScript code or weights are included.

## Full-size speed check

Use the existing environment for a small CPU run:

```sh
.venv/bin/python scripts/bench.py --device cpu --token-budget 512 --points 128 \
  --target-tokens 32 --iterations 5 --warmup 1 --threads 1
```

Measured locally: **78.18 samples/s**, **51.17 ms per forward+backward** iteration
(batch 4, fp32, one CPU thread, 9,822,248 parameters, vocabulary 256). This uses
the full d=256, six-encoder/four-decoder architecture with dropout .1. Five timed
iterations took .2558 seconds after one warmup. Synthetic inputs are preallocated;
loading, augmentation, optimizer updates and validation are outside the timing.
This is a small-budget throughput check, not a real-data accuracy result or a
measured speedup over v1.

For server measurements use `--device cuda --token-budget 24000 --points 400`
(and choose representative `--target-tokens` / `--vocab-size`). CUDA uses bf16
and synchronization around timing; the report includes peak allocated memory.
Batch size is floor(budget / points), and actual input-point usage is reported.
