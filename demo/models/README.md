Export trained models into this directory with `scripts/export_onnx.py`.

The demo uses `encoder.onnx`, `decoder.onnx`, and `vocab.json`. The ones here
are model r3-photo (trained weights, CC BY-NC-SA 4.0: see WEIGHTS_LICENSE.md and
DATA.md in the repository root); no random smoke-test weights are published.
`export_report.json` records parity, model sizes and the int8 accuracy gate.
V2 decoder inputs are `previous_tokens`, `memory` and `memory_mask`; its only
output is last-step `logits`. Export both graphs again when replacing v1 assets.
Accepted int8 models are additional assets; the demo defaults to verified fp32.

When publishing trained assets, include WEIGHTS_LICENSE.md and DATA.md alongside
them, credit any additional training sources, and increment the cache version
in `demo/sw.js` when replacing models.
