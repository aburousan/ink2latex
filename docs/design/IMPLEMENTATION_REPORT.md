> Historical design note (October 2026). The licences have changed since: code is now under the PolyForm
> Noncommercial License 1.0.0 and weights under CC BY-NC-SA 4.0; see LICENSE and WEIGHTS_LICENSE.md.

# Implementation report

Applied SPEC_v2.md on top of the original SPEC.md implementation using the
existing `.venv`. Neither spec was modified. No dataset was downloaded and no
real model was trained; all checks used generated ink or random tensors.

## Changes

- Replaced the decoder's Python target-step training loop with parallel causal
  teacher forcing. Every layer computes first attention, its exclusive
  cumulative sum, and a per-head depthwise kernel-5 convolution with negative
  softplus-weighted ReLU refinement. Refined attention supplies the values and
  the final layer's coverage loss. Removed the coverage-weighted context input.
- Incremental greedy and beam decoding maintain each layer's first-attention
  coverage `[B,H,S]` and self-attention caches. This matches the explicit v2
  cumulative-sum formula; the next layer computes its attention from hidden
  states updated by the preceding layer's refined values. Padding contributes
  zero, and EOS-prediction steps count toward coverage.
- Coverage losses now use `chat = cov_T * S_valid / T_valid`, thresholds .25
  and 4, and weights .05 each. Training logs both penalties and the fraction
  of valid ink tokens below .25. Uniform coverage incurs zero penalties even
  when output length is shorter than ink length.
- Added repeatable `--source-weight PREFIX=W`, with all five specified defaults,
  longest-prefix matching, deterministic fractional retention/repetition, and
  checkpoint validation of weights on resume. `source_counts.jsonl` records
  effective counts for each epoch; training records also include counts at
  epoch starts. Empty sampled epochs are logged and skipped. All-zero mixtures
  and negative/nonfinite weights are rejected.
- Added studio `label` support, named extra pickles (`extra/NAME`), NumPy 2D/3D
  strokes and string/token-list labels. `--test-pkl NAME=PATH` creates held-out
  splits excluded from vocabulary building and training; evaluation accepts
  these names. Reserved/duplicate split names are rejected. Older prepared
  shards remain readable, including source-tag translation for sampling.
- ONNX decoder now takes only the complete token prefix, memory and memory
  mask, recomputes coverage internally and returns last-step logits. Updated
  the browser greedy loop and bumped its service-worker cache to v2.
- Added `scripts/bench.py` for full-size CPU/CUDA forward/backward timing with
  configurable padded input budget, sequence lengths, vocabulary, warmup and
  iteration counts. CUDA uses bf16 and synchronized timing.
- Updated README, DATA and demo model instructions. V1 checkpoints require
  retraining, and v1 ONNX assets must be replaced together with v2 exports.

## Verification

Existing tokenizer, feature, augmentation, masking, metrics, training, evaluation,
export and deterministic-resume coverage remains. Assertions tied to superseded
v1 loss thresholds, sampling counts and coverage-history APIs now check v2.
New checks cover random prefix lengths 1, 2, 7 and 31, padded ink and targets,
causality, finite refinement gradients, uniform relative coverage, fractional
sampling, named held-out data, CLI weight overrides, empty-epoch handling and
resume weight validation. A test also rejects calls to incremental `step`
during training, ensuring the training forward path stays parallel.

- `.venv/bin/python -m unittest discover -s tests -v`: **11 tests passed in 12.369 seconds**.
  Includes 30 CPU training steps with d=64 / 2+2 layers; greedy and beam-5
  evaluation; named held-out evaluation; ONNX export; the int8 validation gate;
  and 15 resumed steps with parameters exactly matching uninterrupted training.
- `node tests/test_features_js.mjs`: **9 cases passed**, max absolute difference
  **0**, including the 1,600-point cap.
- `node tests/test_demo.mjs <temporary demo>`: **passed** in isolated headless
  Chrome using temporary synthetic model exports. Verified WASM inference,
  automatic recognition, KaTeX, undo, offline reload with the origin stopped,
  manual recognition and clear. No page errors or unexpected network origins.
  Tiny-model timings were 46 ms online / 3 ms offline (functional checks only).
  The local test server required execution outside the filesystem/network sandbox.
- Python compile checks, JS syntax checks and `git diff --check`: **passed**.

Maximum absolute differences:

| Comparison | Difference |
| --- | ---: |
| ONNX encoder memory versus PyTorch | 1.90735e-6 |
| ONNX decoder logits versus PyTorch | 1.04308e-7 |
| Cached decoder versus prefix recomputation (export check) | 8.94070e-8 |
| Parallel versus incremental logits on random prefixes | 2.08616e-7 |

All parity checks satisfy the requested **< 1e-4** limit. Coverage is no longer
an ONNX input/output, so its old export parity row has been removed.

## Full-size CPU benchmark

```sh
.venv/bin/python scripts/bench.py --device cpu --token-budget 512 --points 128 \
  --target-tokens 32 --iterations 5 --warmup 1 --threads 1
```

| Setting / measurement | Value |
| --- | ---: |
| Architecture | d=256, 8 heads, FF=1024, 6 encoder + 4 decoder layers |
| Parameters / vocabulary | 9,822,248 / 256 |
| Precision / CPU threads | fp32 / 1 |
| Batch size / padded input-point budget | 4 / 512 |
| Input points / output steps per sample | 128 / 32 |
| Warmup / timed iterations | 1 / 5 |
| Total timed duration | 0.255826 s |
| Forward + backward per iteration | 51.1653 ms |
| Throughput | **78.1780 samples/s** |

Inputs are synthetic and preallocated. Timing includes loss and backward,
excludes data preparation, augmentation, optimizer updates and validation.
This is one small-budget CPU measurement, not an accuracy result, GPU throughput
claim or comparison against a measured v1 baseline.

Environment: Python 3.13.2, macOS ARM64, torch 2.14.1, onnx 1.23.2,
onnxruntime 1.31.0, NumPy 2.5.4, Node 26.5.0. Used the existing `.venv`
without installing dependencies. CUDA A30 / PyTorch 2.7 execution and real-data
accuracy remain unmeasured. Int8 publication still requires the original
1,000-row validation gate; no acceptance claim is made from synthetic data.
No synthetic checkpoints or ONNX model assets were added to the repository.
