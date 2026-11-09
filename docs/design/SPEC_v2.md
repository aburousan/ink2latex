> Historical design note (October 2026). The licences have changed since: code is now under the PolyForm
> Noncommercial License 1.0.0 and weights under CC BY-NC-SA 4.0; see LICENSE and WEIGHTS_LICENSE.md.

# ink2latex: review changes (v2)

The v1 code is good; change these points. Keep all tests passing and add the new ones below.

## 1. Parallel coverage (CoMER-style attention refinement) instead of step-by-step training

v1 trains the decoder one token at a time (a Python loop over T with KV caches), because coverage is updated
per step. On a GPU that is ~T sequential passes per batch. Replace it with the attention refinement of
CoMER (Zhao & Gao, ECCV 2022), written independently:

- In every decoder layer, cross-attention is computed for all target steps at once (teacher forcing).
  Raw scores `e[b,h,t,s]`; first attention `a = softmax(e)` (masked); per-layer coverage
  `c[b,h,t,s] = sum_{u<t} a[b,h,u,s]` = exclusive cumulative sum over t (padding steps contribute 0).
- Refinement bias `r = conv1d_over_s(c)`: a small depthwise conv along the ink axis (kernel 5, per head),
  then `- softplus(w_h) * relu(...)`; final attention `softmax(e + r)`. The refined attention is what is
  used for the values and is what the next layer's coverage and the coverage loss use.
- Causal self-attention as usual. The whole decoder is then one parallel pass in training.
- Incremental inference must give the same result: keep per layer a running coverage `[B,H,S]` and add
  each step's first attention after the step. Add a test: parallel teacher-forced logits == incremental
  step-by-step logits (max abs diff < 1e-4) on random data, for a few prefix lengths.
- Remove the v1 coverage-weighted context input and the coverage-history export contract.

## 2. A coverage loss that is possible to satisfy

Each step's attention sums to 1 over S ink tokens, so the total coverage is T (output length), not S. Asking
every ink token for coverage >= 0.8 is impossible when T < S and pushes attention to smear. Use relative
coverage of the last layer, head-mean: `chat = cov_T * S_valid / T_valid` (mean 1 per sample). Losses:
- unread: `mean_over_ink( relu(0.25 - chat) )` (an ink region the reader never looked at),
- twice: `mean_over_ink( relu(chat - 4.0) )` (looked at far more than its share),
weights 0.05 each. Log both and also the fraction of ink tokens with chat < 0.25.

## 3. Data mixture weights

Add `--source-weight PREFIX=W` (repeatable) to train.py: per epoch, each sample of a source whose name starts
with PREFIX is kept with probability W if W < 1, or repeated floor(W) times plus one more with probability
frac(W). Defaults: `mathwriting/train=1`, `mathwriting/synthetic=0.6`, `mathwriting/symbols=1`,
`extra=0.5`, `user=50` (replaces the hard-coded 50x). Log the effective count per source per epoch.

## 4. Inputs

- `--user-jsonl` rows may have `label` instead of `latex` (single symbols from the studio's
  `samples.jsonl`): accept either; physics rows still go to `user_physics_test`.
- `--extra-pkl` may also be given as `NAME=PATH`; the source name becomes `extra/NAME`. A tuple's second item
  may be a token list (join with spaces) or a string. Strokes may be (n,2) or (n,3) arrays.
- `--test-pkl NAME=PATH` adds a held-out evaluation split named NAME (never trained on).

## 5. ONNX

Export `encoder.onnx` and `decoder.onnx` where decoder takes the full token prefix `[B,T]`, memory and mask,
runs the parallel decoder and returns the last step's logits. No coverage input or output (coverage is
recomputed inside from the prefix). The demo's greedy loop calls it with the growing prefix. Update demo/app.js
accordingly and keep the browser test passing. Parity test as before.

## 6. Speed check

Add `scripts/bench.py`: time forward+backward of the full-size model (d=256, 6+4) on a batch of the given
token budget, CPU or CUDA, and print samples/s. Run it on CPU here once with a small budget and report.
