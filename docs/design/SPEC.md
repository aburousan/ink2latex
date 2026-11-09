> Historical build specification (October 2026). The licences have changed since: code is now under the PolyForm
> Noncommercial License 1.0.0 and weights under CC BY-NC-SA 4.0; see LICENSE and WEIGHTS_LICENSE.md.

# ink2latex: build spec (for the implementer)

Online handwritten maths (pen strokes) to LaTeX, with an in-browser demo. Code under a noncommercial licence (see LICENSE); trained weights
under CC BY-NC-SA 4.0 (they are trained on MathWriting). Write everything in this folder. Do not download
data or train here: this laptop has no GPU and 8 GB of RAM. Training runs later on a CUDA A30 server
(PyTorch 2.7, CUDA 12.8). Locally, a tiny CPU smoke test on fake data must pass.

## 1. Model: stroke sequence encoder + coverage decoder

Inspired by the public description of MyScript iink 3.0 maths (no code copied): read the strokes directly,
transformer with relative 2D positions, down-sampling, and decoder attention fed back so each part of the
ink is read once.

**Input features** (`ink2latex/features.py`, mirrored exactly in `demo/features.js`):
1. Strokes: list of lists of (x, y, t). y grows downwards. Drop strokes with < 1 point.
2. Normalise: translate to the ink's top-left; scale so that the robust ink height (95th - 5th percentile of
   all y) is 1. If that is tiny (a single horizontal stroke), use the robust width instead; never divide by
   < 1e-6. Keep aspect ratio.
3. Resample each stroke by arc length at step 0.05 (in normalised units), always keep the first and last
   point; a dot (length < step) becomes 1 point.
4. Per point, 7 channels: x, y, dx, dy (to the next point in the same stroke, 0 at stroke end), stroke-start
   flag, stroke-end flag, and (x, y) absolute of the point's stroke centre minus the point (2 channels:
   cx - x, cy - y) => 8 channels total. No time channel (MathWriting synthetic has unreliable time).
5. Concatenate strokes in the given order; cap at 1,600 points (subsample uniformly if longer, and count
   how often this happens in prep logs).

**Encoder** (`ink2latex/model.py`): Linear 8 -> 256; two Conv1D blocks (kernel 5, stride 2, GELU, LayerNorm)
reducing the sequence by 4 (never mix points across a stroke boundary: mask with zero padding between
strokes, or accept the mix but keep the start/end flags; state which you chose); then 6 transformer
encoder layers, d=256, 8 heads, FF 1024, dropout 0.1, pre-LN. Positional information: **2D relative
attention bias** from each token pair's (dx, dy) of their (x, y) centres, through a small shared MLP
(2 -> 32 -> heads) evaluated on clipped/quantised offsets, added to the attention logits. Also add a 1D
learned position embedding for the reduced sequence index (ink order still matters a little).

**Decoder**: 4 transformer decoder layers, d=256, 8 heads, FF 1024, tied input/output embedding. Coverage:
keep `cov_t = sum_{u<t} mean_heads(cross_attention_u)` per encoder token. Feed it back two ways:
(a) add `-w_h * cov_t` (learned non-negative w per head, init 1.0) to cross-attention logits in every
decoder layer; (b) concatenate `[cov_t-weighted encoder mean, 1 - mean(cov_t)]` projected to d into the
decoder input at step t. **Coverage loss**: `mean(relu(cov_T - 1.2))` (read twice) + at EOS
`mean(relu(0.8 - cov_T) * ink_mask)` (never read), weight 0.1 each; log both separately.

**Vocabulary** (`ink2latex/tokenizer.py`): LaTeX tokens: `\command`, escaped characters, `{ } ^ _`, and
single characters; spaces dropped. Build from the training labels: keep tokens seen >= 20 times; a label
with any rarer token is dropped from training (count and log them). Special: <pad> <s> </s>. Max label
length 160 tokens (drop longer, log count). Save `vocab.json`.

## 2. Data (`ink2latex/data.py`, `scripts/prep.py`)

Write `scripts/prep.py --mathwriting DIR --out DIR [--user-jsonl FILE ...] [--extra-pkl FILE ...]` that writes
compact shards (numpy .npz of float16 points + int16 token ids + offsets) for train / valid / test.

- **MathWriting 2024** (`DIR/{train,valid,test,synthetic,symbols}/*.inkml`): parse each InkML; label =
  `<annotation type="normalizedLabel">` (fall back to `label`); traces are "x y t, x y t, ...".
  train = train + synthetic + symbols; valid = valid; test = test. Never mix splits.
- **User samples** (`--user-jsonl`, handwriting-samples JSONL): one JSON per line with `latex`,
  `strokes` = [[[x, y, t, p], ...], ...]. Rows whose `meta.kind == "physics"` go to a separate
  `user_physics_test` split; the rest to `user_train`, which is up-sampled 50x in the training mixture.
- **Extra pickles** (`--extra-pkl`): list of (strokes, tokens) from the older project (strokes as lists of
  (x, y) or (x, y, t)); add to train, tagged by source.
- Write `prep_report.json`: counts per source and split, dropped (rare token, too long, empty ink), median
  points per sample, longest label.

## 3. Training (`scripts/train.py`)

AdamW lr 3e-4, betas (0.9, 0.98), weight decay 0.05, warmup 2k steps, cosine to 1e-5; batch by token budget
(~24k input points per batch, bucketed by length); bf16 autocast on CUDA; grad clip 1.0; label smoothing 0.1.
Augmentation on the fly (train only): random affine (rotation ±5°, shear ±0.25, x/y scale 0.85-1.15),
small smooth jitter (sigma 0.01), random dropping of 0-10% of points (never a whole stroke), random
stroke-order perturbation inside a 2-stroke window with p=0.1.
Checkpoints every 5k steps + `best.pt` by valid exact-match on a fixed 2,000-row valid subset (greedy).
`--resume`, `--steps`, `--out`, deterministic seed. Log JSON lines: step, loss parts, lr, tokens/s.

## 4. Evaluation (`scripts/eval.py`)

Greedy and beam (width 5) decoding. Metrics on a split: exact match (token sequence equal), token error rate
(edit distance / reference tokens), and a "loose" exact match ignoring `{ }` and `\left \right`. Write
per-sample predictions to JSONL. Splits: MathWriting test (headline), valid, user_physics_test, and an
optional pickle of the user's 16 legacy notes.

## 5. Export and demo

`scripts/export_onnx.py`: `encoder.onnx` (points -> memory) and `decoder.onnx` (one step: previous
tokens + memory + coverage -> logits + new coverage; recomputing the prefix each step is acceptable, keep it
simple and correct). Opset 17. Also a dynamic int8 or fp16 version if it stays within 0.5 pp exact of fp32
on 1,000 valid rows (report both sizes). Verify ONNX Runtime outputs match PyTorch (max abs diff report).

`demo/` is a static site (GitHub Pages): `index.html`, `app.js`, `features.js`, `style.css`, `models/`.
- A large canvas: draw with mouse, touch or pen (Pointer Events, pressure ignored), undo stroke, clear.
- "Recognise" button and auto-recognise 600 ms after the last stroke.
- Runs onnxruntime-web (pinned exact version from cdn.jsdelivr.net) on WASM; greedy decode, max 160 tokens.
- Shows the LaTeX (copyable) and the typeset result with KaTeX (pinned exact version from cdn.jsdelivr.net).
- Shows inference time. Works offline after first load. No analytics, no network calls except the CDN files.
- A small "About" box: licence split (code noncommercial, weights CC BY-NC-SA 4.0, credits to MathWriting by Google
  Research and any other data used).
- `demo/features.js` must equal `features.py`: add `tests/fixtures/features_case.json` (a few inks and the
  expected features from Python) and `tests/test_features_js.mjs` (node) that checks max abs diff < 1e-4.

## 6. Repository

README.md (what it is, demo link placeholder, results table placeholder, how to train, how to run the demo
locally with `python -m http.server`), LICENSE (see the repository LICENSE),
WEIGHTS_LICENSE.md (CC BY-NC-SA 4.0 summary + link to the full text, says the weights are trained on
MathWriting and therefore carry its licence), DATA.md (every data source, its licence, and how it is used;
MathWriting citation: Gervais et al., "MathWriting: A Dataset For Handwritten Mathematical Expression
Recognition", 2024). `requirements.txt` (torch is required only for training). `.gitignore` (data, runs,
checkpoints; but NOT demo/models).

## 7. Tests (must pass locally, CPU, < 2 minutes)

`tests/test_smoke.py`: build fake inks, prep -> shards, train 30 steps on CPU with a tiny config
(d=64, 2+2 layers), eval runs, export ONNX and compare with PyTorch. Tokenizer round-trip tests. Coverage
loss shape test. Feature function determinism test.

Keep the code compact and readable; comment only where the reason is not obvious.
