# Data sources and provenance

| Source | Licence | Use |
| --- | --- | --- |
| MathWriting 2024, Google Research / Google LLC | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | `train`, `synthetic`, and `symbols` form training; `valid` remains validation; `test` remains the headline test set. |
| User handwriting samples (JSONL) | User-owned; no blanket licence supplied. Record the owner's permission and attribution before including these samples in distributed weights. | Non-physics rows form `user_train`, sampled 50× by default (configurable); `meta.kind == "physics"` rows go only to `user_physics_test`. |
| Older-project extra pickle files | Source-dependent; no licence supplied. Record each original source and its permitted uses before adding it. | Optional additions to training, separately tagged as `extra/NAME` (or `extra/<file stem>` for unnamed files). |
| User's 16 legacy notes | User-owned; no licence supplied. | Optional evaluation-only pickle passed to `scripts/eval.py --legacy-pkl` or prepared as `--test-pkl NAME=PATH`; excluded from vocabulary construction and training. |
| Synthetic test fixtures in this repository | PolyForm Noncommercial 1.0.0, Kazi Abu Rousan, 2026 | Generated geometric ink for software tests only; no real handwriting or downloaded examples. |

No dataset is downloaded by this repository. No real training data or trained
weights are bundled. The optional sources above are supported formats, not
claims that their contents were used. For an actual release, update this file
with the filenames, authors, licences, permissions and credits of every added
source, and update the demo's About box. Do not infer licences for private data.

MathWriting citation: Gervais et al., “MathWriting: A Dataset For Handwritten
Mathematical Expression Recognition”, 2024.
[Paper](https://arxiv.org/abs/2404.10690) ·
[Official dataset and licence notice](https://github.com/google-research/google-research/tree/master/mathwriting).

Preparation records source and original sample identifiers in shard sidecars
and counts kept/dropped rows by source and split in `prep_report.json`.
User sampling does not affect validation/test or duplicate the stored records.
Pickles and PyTorch checkpoints should only be loaded from trusted sources,
because their serialization formats can execute Python code.

## What model r3 was trained on

Model r3 (the weights in `demo/models/` and in the GitHub release) was trained from
scratch for 80,000 steps on one A30. Rows seen per epoch:

| Source | Rows per epoch | Licence and credit |
| --- | --- | --- |
| MathWriting `train` | 229,820 | CC BY-NC-SA 4.0, Google Research / Google LLC, Gervais et al. 2024 |
| MathWriting `synthetic` (weight 0.6) | about 237,000 | as above |
| MathWriting `symbols` | 6,423 | as above |
| `extra/oldsynth`: generated formulas (weight 0.5 of 300,000) | about 150,000 | symbols from HWRT (ODbL 1.0, Martin Thoma), laid out with the Hershey fonts and the EMS fonts (SIL Open Font License); generator by Kazi Abu Rousan |
| `extra/real2`: the owner's handwritten notes (368 rows, weight 5) | about 1,840 | Kazi Abu Rousan, own handwriting, used with permission |
| `extra/sn`: the owner's Samsung Notes lines (59 rows, weight 50) | 2,950 | Kazi Abu Rousan, own handwriting, used with permission |
| `extra/focus`: training rows with a dagger (272) and the owner's rows with a slash (25), weight 5 | 1,485 | copies of rows above, same licences |
| Author's handwriting samples (`written.jsonl`, `samples.jsonl`) | 10,750 | Kazi Abu Rousan, own handwriting, used with permission |

MathWriting `valid` chose the checkpoint and MathWriting `test` was only
evaluated. Held out and never trained on: the owner's notes test (119 rows), 20
Samsung Notes lines, a 47-row physics notebook test and the 10-row physics test.
The owner's raw handwriting is not published; only the trained weights are.

## What model r3-photo was trained on

r3-photo (the weights the app in `demo/` uses) is r3 fine-tuned for 8,000 steps
(peak learning rate 1e-4) on the same sources and weights as r3, plus 80,000
photo-style copies of training rows (`photo/...`, weight 3): about 56,000
MathWriting `train` rows, about 20,000 generated `extra/oldsynth` rows and every
row of the owner's own handwriting, each drawn as a picture with a random pen
width (2 to 6 px), height (70 to 200 px), slant (±4°), blur, noise and uneven
light, then traced back into strokes with `ink2latex/photo.py`. These copies
carry the licences of the rows they were made from. No held-out row was used.
