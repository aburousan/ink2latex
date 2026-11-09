import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, previous_tokens, memory, memory_mask):
        return self.model.prefix_step(previous_tokens, memory, memory_mask)


def ort_session(path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def numpy(tensor):
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def verify(model, encoder, decoder):
    report = {"memory": 0., "logits": 0., "cached_vs_prefix": 0.}
    for batch, length in ((1, 13), (2, 29), (1, 1)):
        points = torch.linspace(0, 1, batch * length * 8).reshape(batch, length, 8)
        mask = torch.ones(batch, length, dtype=torch.bool)
        if batch == 2:
            mask[1, length // 2:] = False
            points[1, length // 2:] = 0
        memory, valid = model.encoder(points, mask)
        actual, actual_mask = encoder.run(None, {"points": numpy(points), "point_mask": numpy(mask)})
        report["memory"] = max(report["memory"], float(np.max(np.abs(actual - numpy(memory)))))
        np.testing.assert_array_equal(actual_mask, numpy(valid))
        tokens = torch.ones(batch, 1, dtype=torch.long)
        coverage = model.initial_coverage(memory)
        cache = None
        for step in range(5):
            logits = model.prefix_step(tokens, memory, valid)
            cached, coverage, cache = model.step(tokens[:, -1], memory, valid, coverage, step, cache)
            report["cached_vs_prefix"] = max(report["cached_vs_prefix"], float((cached - logits).abs().max()))
            out_logits, = decoder.run(None, {"previous_tokens": numpy(tokens), "memory": actual,
                                             "memory_mask": actual_mask})
            report["logits"] = max(report["logits"], float(np.max(np.abs(out_logits - numpy(logits)))))
            next_token = torch.full((batch, 1), 3 + step % max(1, model.embedding.num_embeddings - 3), dtype=torch.long)
            tokens = torch.cat([tokens, next_token], 1)
    if max(report.values()) > 1e-4:
        raise AssertionError(f"ONNX parity failed: {report}")
    return report


def ort_greedy(encoder, decoder, points, max_tokens=160):
    memory, mask = encoder.run(None, {"points": points[None].astype(np.float32),
                                     "point_mask": np.ones((1, len(points)), bool)})
    tokens = np.ones((1, 1), np.int64)
    result = []
    for _ in range(max_tokens):
        logits, = decoder.run(None, {"previous_tokens": tokens, "memory": memory, "memory_mask": mask})
        logits[:, :2] = -np.inf
        token = int(logits.argmax(-1)[0])
        if token == 2:
            break
        result.append(token)
        tokens = np.concatenate([tokens, [[token]]], 1)
    return result


def export_models(model, tokenizer, out, validation=None, quantize=True):
    import onnx
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    points = torch.zeros(1, 17, 8)
    mask = torch.ones(1, 17, dtype=torch.bool)
    with torch.no_grad():
        memory, valid = model.encoder(points, mask)
    torch.onnx.export(model.encoder, (points, mask), str(out / "encoder.onnx"), opset_version=17,
                      input_names=["points", "point_mask"], output_names=["memory", "memory_mask"],
                      dynamic_axes={"points": {0: "batch", 1: "points"}, "point_mask": {0: "batch", 1: "points"},
                                    "memory": {0: "batch", 1: "ink"}, "memory_mask": {0: "batch", 1: "ink"}}, dynamo=False)
    torch.onnx.export(DecoderExport(model).eval(), (torch.ones(1, 2, dtype=torch.long), memory, valid), str(out / "decoder.onnx"),
                      opset_version=17, input_names=["previous_tokens", "memory", "memory_mask"],
                      output_names=["logits"], dynamic_axes={
                          "previous_tokens": {0: "batch", 1: "prefix"}, "memory": {0: "batch", 1: "ink"},
                          "memory_mask": {0: "batch", 1: "ink"}, "logits": {0: "batch"}}, dynamo=False)
    for name in ("encoder", "decoder"):
        onnx.checker.check_model(str(out / f"{name}.onnx"))
    encoder, decoder = [ort_session(out / f"{name}.onnx") for name in ("encoder", "decoder")]
    report = {"opset": 17, "max_abs_diff": verify(model, encoder, decoder),
              "fp32_bytes": {name: (out / f"{name}.onnx").stat().st_size for name in ("encoder", "decoder")}}
    report["int8"] = {"accepted": False, "reason": "disabled"}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"{name}.int8.onnx" for name in ("encoder", "decoder")]
            for name, path in zip(("encoder", "decoder"), paths):
                quantize_dynamic(str(out / f"{name}.onnx"), str(path), weight_type=QuantType.QInt8,
                                 op_types_to_quantize=["MatMul", "Gemm"])
            info = {"accepted": False, "bytes": {name: path.stat().st_size for name, path in zip(("encoder", "decoder"), paths)}}
            if validation is None or len(validation) < 1000:
                info["reason"] = "Requires 1,000 valid rows; candidate not published"
            else:
                quant_encoder, quant_decoder = [ort_session(path) for path in paths]
                indices = np.random.default_rng(42).permutation(len(validation))[:1000]
                fp_correct, quant_correct = 0, 0
                for index in indices:
                    points, _, meta = validation[index]
                    reference = meta["tokens"]
                    fp_correct += tokenizer.decode(ort_greedy(encoder, decoder, points)) == reference
                    quant_correct += tokenizer.decode(ort_greedy(quant_encoder, quant_decoder, points)) == reference
                info.update({"rows": 1000, "fp32_exact_match": fp_correct / 1000,
                             "int8_exact_match": quant_correct / 1000, "drop_pp": (fp_correct - quant_correct) / 10,
                             "accepted": fp_correct - quant_correct <= 5})
                if info["accepted"]:
                    for path in paths:
                        shutil.copyfile(path, out / path.name)
                else:
                    info["reason"] = "Accuracy drop exceeds 0.5 percentage points"
            report["int8"] = info
    if not report["int8"]["accepted"]:
        for name in ("encoder.int8.onnx", "decoder.int8.onnx"):
            (out / name).unlink(missing_ok=True)
    tokenizer.save(out / "vocab.json")
    (out / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
