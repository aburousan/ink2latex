"""Streaming preparation and bounded shard loading; no torch dependency here."""

import json
import math
import re
import pickle
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .features import features
from .tokenizer import Tokenizer, tokenize


def read_inkml(path):
    root = ET.parse(path).getroot()
    labels, strokes = {}, []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "annotation":
            labels[node.get("type")] = node.text or ""
        elif tag == "trace":
            stroke = [list(map(float, point.split())) for point in (node.text or "").split(",")
                      if point.strip()]
            if any(len(point) < 2 for point in stroke):
                raise ValueError(f"Invalid trace in {path}")
            strokes.append(stroke)
    if "normalizedLabel" not in labels and "label" not in labels:
        raise ValueError(f"No label in {path}")
    return strokes, labels.get("normalizedLabel", labels.get("label"))


DEFAULT_SOURCE_WEIGHTS = {"mathwriting/train": 1., "mathwriting/synthetic": .6,
                          "mathwriting/symbols": 1., "extra": .5, "user": 50.}


def source_weights(overrides=()):
    weights = dict(DEFAULT_SOURCE_WEIGHTS)
    for entry in overrides:
        prefix, separator, value = entry.rpartition("=")
        if not separator or not prefix:
            raise ValueError("Source weight must be PREFIX=W")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Source weights must be finite and nonnegative")
        weights[prefix] = weight
    return weights


def source_weight(source, weights):
    matches = [prefix for prefix in weights if source.startswith(prefix)]
    return weights[max(matches, key=len)] if matches else 1.


def named_pickle(value, required=False):
    name, separator, filename = str(value).partition("=")
    if not separator:
        if required:
            raise ValueError("Held-out pickle must be NAME=PATH")
        return Path(value).stem, str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not filename:
        raise ValueError("Pickle name must contain only letters, digits, underscores or hyphens")
    return name, filename


def iter_sources(mathwriting, user_jsonl=(), extra_pkl=(), test_pkl=()):
    for directory in ("train", "synthetic", "symbols", "valid", "test"):
        split = directory if directory in ("valid", "test") else "train"
        for path in sorted((Path(mathwriting) / directory).glob("*.inkml")):
            strokes, label = read_inkml(path)
            yield split, f"mathwriting/{directory}", path.name, strokes, label
    for filename in user_jsonl:
        with open(filename) as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                split = "user_physics_test" if row.get("meta", {}).get("kind") == "physics" else "user_train"
                yield split, f"user:{Path(filename).name}", str(line_no), row["strokes"], row["latex"] if "latex" in row else row["label"]
    for split, entry in [("train", e) for e in extra_pkl] + [(named_pickle(e, True)[0], e) for e in test_pkl]:
        name, filename = named_pickle(entry)
        # Pickle executes code: this opt-in input must come from a trusted project.
        with open(filename, "rb") as stream:
            rows = pickle.load(stream)
        for i, (strokes, tokens) in enumerate(rows):
            label = tokens if isinstance(tokens, str) else " ".join(tokens)
            yield split, f"extra/{name}", str(i), strokes, label


def prepare(mathwriting, out, user_jsonl=(), extra_pkl=(), min_count=20, shard_size=512, test_pkl=(), vocab=None):
    """`vocab`: a trained model's vocab.json to encode with (fine-tuning), instead of building one from the labels."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Preparation output already contains a dataset; choose a fresh directory")
    splits = ["train", "user_train", "valid", "test", "user_physics_test"]
    for entry in test_pkl:
        name, _ = named_pickle(entry, True)
        if name in splits:
            raise ValueError(f"Held-out split name is reserved or duplicated: {name}")
        splits.append(name)
    sources = lambda: iter_sources(mathwriting, user_jsonl, extra_pkl, test_pkl)
    vocab = Tokenizer.load(vocab) if vocab else Tokenizer.build(
        (label for split, _, _, _, label in sources() if split in ("train", "user_train")), min_count)
    vocab.save(out / "vocab.json")
    pending = {s: [] for s in splits}
    manifest = {"version": 2, "splits": {s: [] for s in splits}}
    report = {"sources": {}, "splits": {}, "dropped": Counter(), "capped": 0,
              "median_points": 0, "longest_label": 0, "longest_label_seen": 0}
    lengths = []

    def flush(split):
        rows = pending[split]
        if not rows:
            return
        name = f"{split}-{len(manifest['splits'][split]):05d}"
        point_lengths = [len(r[0]) for r in rows]
        token_lengths = [len(r[1]) for r in rows]
        np.savez_compressed(out / (name + ".npz"),
                            points=np.concatenate([r[0] for r in rows]).astype(np.float16),
                            tokens=np.concatenate([r[1] for r in rows]).astype(np.int16),
                            point_offsets=np.r_[0, np.cumsum(point_lengths)].astype(np.int64),
                            token_offsets=np.r_[0, np.cumsum(token_lengths)].astype(np.int64))
        (out / (name + ".jsonl")).write_text("".join(json.dumps(r[2], ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][split].append({"file": name + ".npz", "count": len(rows), "lengths": point_lengths,
                                           "sources": [r[2]["source"] for r in rows]})
        pending[split] = []

    for split, source, sample_id, strokes, label in sources():
        stats = report["sources"].setdefault(source, {})
        stats = stats.setdefault(split, {"seen": 0, "kept": 0, "capped": 0, "dropped": Counter()})
        stats["seen"] += 1
        tokens = tokenize(label)
        report["longest_label_seen"] = max(report["longest_label_seen"], len(tokens))
        reason = "too_long" if len(tokens) > 160 else None
        if not tokens:
            reason = "empty_label"
        if reason is None and split in ("train", "user_train") and any(t not in vocab.ids for t in tokens):
            reason = "rare_token"
        points, capped = features(strokes, return_capped=True)
        if not len(points):
            reason = "empty_ink"
        if reason:
            stats["dropped"][reason] += 1
            report["dropped"][reason] += 1
            continue
        stats["kept"] += 1
        stats["capped"] += int(capped)
        report["capped"] += int(capped)
        report["splits"][split] = report["splits"].get(split, 0) + 1
        lengths.append(len(points))
        report["longest_label"] = max(report["longest_label"], len(tokens))
        pending[split].append((points, vocab.encode(tokens, allow_unknown=True),
                               {"id": f"{source}:{sample_id}", "source": source, "tokens": tokens}))
        if len(pending[split]) >= shard_size:
            flush(split)
    for split in splits:
        flush(split)
        report["splits"].setdefault(split, 0)
    report["median_points"] = float(np.median(lengths)) if lengths else 0
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "prep_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class ShardDataset:
    def __init__(self, directory, splits, max_cached=None):
        self.directory = Path(directory)
        self.max_cached = max_cached
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.rows, self.lengths, self.split_names, self.sources = [], [], [], []
        self.cache = OrderedDict()
        for split in ([splits] if isinstance(splits, str) else splits):
            for shard in self.manifest["splits"][split]:
                sources = shard.get("sources")
                if sources is None:
                    sources = [json.loads(line)["source"] for line in
                               (self.directory / shard["file"]).with_suffix(".jsonl").read_text().splitlines()]
                    # Original prepared shards used pickle:filename source tags.
                    sources = [f"extra/{Path(s[7:]).stem}" if s.startswith("pickle:") else s for s in sources]
                for i, length in enumerate(shard["lengths"]):
                    self.rows.append((shard["file"], i))
                    self.lengths.append(length)
                    self.split_names.append(split)
                    self.sources.append(sources[i])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, row = self.rows[index]
        if name not in self.cache:
            with np.load(self.directory / name, allow_pickle=False) as shard:
                arrays = {k: shard[k] for k in shard.files}
            metadata = [json.loads(line) for line in (self.directory / name).with_suffix(".jsonl").read_text().splitlines()]
            self.cache[name] = arrays, metadata
            # Kept for the whole run when the data fits (max_cached=None): a random batch touches
            # hundreds of shards, and evicting after two reloaded each one from disk on every step.
            if self.max_cached is not None and len(self.cache) > self.max_cached:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        arrays, metadata = self.cache[name]
        p, t = arrays["point_offsets"], arrays["token_offsets"]
        return (arrays["points"][p[row]:p[row + 1]].astype(np.float32),
                arrays["tokens"][t[row]:t[row + 1]].astype(np.int64), metadata[row])


def token_batches(dataset, budget, seed, epoch=0, weights=None, effective_counts=None):
    rng = np.random.default_rng(seed + epoch)
    weights = DEFAULT_SOURCE_WEIGHTS if weights is None else weights
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Source weights must be finite and nonnegative")
    repeats = []
    counts = Counter(dict.fromkeys(dataset.sources, 0))
    for source in dataset.sources:
        weight = source_weight(source, weights)
        repeat = math.floor(weight)
        if weight % 1:
            repeat += int(rng.random() < weight % 1)
        repeats.append(repeat)
        counts[source] += repeat
    if effective_counts is not None:
        effective_counts.update(counts)
    indices = np.repeat(np.arange(len(dataset)), repeats)
    rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), 1024):
        bucket = sorted(indices[start:start + 1024], key=lambda i: dataset.lengths[i])
        batch, longest = [], 0
        for index in bucket:
            length = dataset.lengths[index]
            if batch and max(longest, length) * (len(batch) + 1) > budget:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(int(index))
            longest = max(longest, length)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    return batches
