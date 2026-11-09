import numpy as np
import torch

from .tokenizer import detokenize


def collate(rows, device="cpu", augmentation_rng=None):
    from .features import augment
    points = [augment(row[0], augmentation_rng) if augmentation_rng is not None else row[0] for row in rows]
    size = max(len(p) for p in points)
    inputs = np.zeros((len(rows), size, 8), np.float32)
    mask = np.zeros((len(rows), size), bool)
    length = max(len(row[1]) for row in rows) + 1
    decoder = np.zeros((len(rows), length), np.int64)
    targets = np.zeros_like(decoder)
    for i, (p, row) in enumerate(zip(points, rows)):
        inputs[i, :len(p)], mask[i, :len(p)] = p, True
        decoder[i, :len(row[1]) + 1] = [1, *row[1]]
        targets[i, :len(row[1]) + 1] = [*row[1], 2]
    return tuple(torch.from_numpy(a).to(device) for a in (inputs, mask, decoder, targets))


@torch.no_grad()
def greedy(model, points, point_mask, max_tokens=None):
    memory, mask = model.encoder(points, point_mask)
    kv = model.project_memory(memory)
    token = torch.ones(points.shape[0], dtype=torch.long, device=points.device)
    coverage = model.initial_coverage(memory)
    done = torch.zeros_like(token, dtype=torch.bool)
    sequences, caches = [[] for _ in range(len(token))], None
    for t in range(max_tokens or model.config.max_tokens):
        logits, coverage, caches = model.step(token, memory, mask, coverage, t, caches, kv)
        logits[:, :2] = -float("inf")
        token = logits.argmax(-1)
        for i, value in enumerate(token.tolist()):
            if not done[i] and value != 2:
                sequences[i].append(value)
        done |= token.eq(2)
        if done.all():
            break
    return sequences


@torch.no_grad()
def beam_search(model, points, point_mask, width=5, max_tokens=None):
    memory, mask = model.encoder(points, point_mask)
    results = []
    for i in range(len(points)):
        mem, valid = memory[i:i + 1], mask[i:i + 1]
        kv = model.project_memory(mem)
        beams = [(0., [1], model.initial_coverage(mem), None, False)]
        for t in range(max_tokens or model.config.max_tokens):
            candidates = []
            for score, sequence, coverage, cache, done in beams:
                if done:
                    candidates.append((score, sequence, coverage, cache, True))
                    continue
                token = torch.tensor([sequence[-1]], device=points.device)
                logits, updated, new_cache = model.step(token, mem, valid, coverage, t, cache, kv)
                logits[:, :2] = -float("inf")
                values, ids = logits.log_softmax(-1).topk(min(width, logits.shape[-1] - 2))
                for value, idx in zip(values[0].tolist(), ids[0].tolist()):
                    candidates.append((score + value, sequence + [idx], updated, new_cache, idx == 2))
            beams = sorted(candidates, key=lambda b: b[0], reverse=True)[:width]
            if all(b[4] for b in beams):
                break
        best = max(beams, key=lambda b: b[0])[1][1:]
        results.append(best[:-1] if best and best[-1] == 2 else best)
    return results


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def score_predictions(references, predictions):
    loose = lambda tokens: [t for t in tokens if t not in ("{", "}", "\\left", "\\right")]
    count = len(references)
    if not count:
        return {"samples": 0, "exact_match": None, "token_error_rate": None, "loose_exact_match": None}
    return {"samples": count,
            "exact_match": sum(a == b for a, b in zip(references, predictions)) / count,
            "token_error_rate": sum(edit_distance(a, b) for a, b in zip(references, predictions)) / max(1, sum(map(len, references))),
            "loose_exact_match": sum(loose(a) == loose(b) for a, b in zip(references, predictions)) / count}


@torch.no_grad()
def evaluate(model, dataset, tokenizer, *, indices=None, beam=1, batch_size=16,
             max_tokens=None, prediction_path=None):
    import json
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    indices = list(range(len(dataset))) if indices is None else list(indices)
    references, predictions = [], []
    output = open(prediction_path, "w") if prediction_path else None
    try:
        for start in range(0, len(indices), batch_size):
            rows = [dataset[i] for i in indices[start:start + batch_size]]
            points, mask, _, _ = collate(rows, device)
            ids = (greedy(model, points, mask, max_tokens) if beam == 1 else
                   beam_search(model, points, mask, beam, max_tokens))
            for row, prediction in zip(rows, ids):
                tokens = tokenizer.decode(prediction)
                references.append(row[2]["tokens"])
                predictions.append(tokens)
                if output:
                    output.write(json.dumps({**row[2], "prediction_tokens": tokens,
                                              "prediction": detokenize(tokens)}, ensure_ascii=False) + "\n")
    finally:
        if output:
            output.close()
        model.train(was_training)
    return score_predictions(references, predictions)
