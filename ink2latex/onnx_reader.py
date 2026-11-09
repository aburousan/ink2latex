"""CPU ONNX decoding with the demo's full-prefix tensor contract (no torch)."""
from pathlib import Path
import numpy as np

from .features import features
from .tokenizer import Tokenizer


def decode(encoder, decoder, points, beam=1, max_tokens=160):
    if beam not in (1, 5):
        raise ValueError("beam must be 1 or 5")
    points = np.asarray(points, np.float32)
    if not len(points):
        return []
    memory, mask = encoder.run(["memory", "memory_mask"], {
        "points": points[None], "point_mask": np.ones((1, len(points)), bool)})
    beams = [(0., [1], False)]
    for _ in range(max_tokens):
        candidates = []
        for score, sequence, done in beams:
            if done:
                candidates.append((score, sequence, True))
                continue
            logits = decoder.run(["logits"], {"previous_tokens": np.array([sequence], np.int64),
                                 "memory": memory, "memory_mask": mask})[0][0].astype(np.float64)
            logits[:2] = -np.inf
            logits -= np.max(logits)
            log_prob = logits - np.log(np.exp(logits).sum())
            for token in np.argsort(-log_prob, kind="stable")[:min(beam, len(logits) - 2)]:
                candidates.append((score + float(log_prob[token]), sequence + [int(token)], token == 2))
        beams = sorted(candidates, key=lambda b: b[0], reverse=True)[:beam]
        if all(b[2] for b in beams):
            break
    result = beams[0][1][1:]
    return result[:-1] if result and result[-1] == 2 else result


class OnnxReader:
    def __init__(self, models):
        import onnxruntime as ort
        models = Path(models)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(models / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(models / "decoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.load(models / "vocab.json")

    def predict_points(self, points, beam=1):
        return self.tokenizer.decode(decode(self.encoder, self.decoder, points, beam))

    def predict(self, strokes, beam=1):
        return self.predict_points(features(strokes), beam)
