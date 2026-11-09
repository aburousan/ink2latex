#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ink2latex.data import ShardDataset
from ink2latex.features import features
from ink2latex.inference import evaluate
from ink2latex.tokenizer import tokenize
from ink2latex.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate token accuracy; MathWriting test is the headline split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--split", default="test", help="Prepared evaluation split (including named --test-pkl splits)")
    parser.add_argument("--legacy-pkl", help="Trusted pickle containing the 16 legacy notes")
    parser.add_argument("--out", required=True, help="Per-sample predictions JSONL")
    parser.add_argument("--beam", type=int, choices=[1, 5], default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decode-tokens", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    if args.legacy_pkl:
        with open(args.legacy_pkl, "rb") as stream:
            raw = pickle.load(stream)
        dataset = []
        for i, (strokes, label) in enumerate(raw):
            tokens = tokenize(label) if isinstance(label, str) else list(label)
            points = features(strokes)
            if not len(points):
                raise ValueError(f"Empty legacy ink at row {i}")
            dataset.append((points, tokenizer.encode(tokens, allow_unknown=True),
                            {"id": str(i), "source": Path(args.legacy_pkl).name, "tokens": tokens}))
    else:
        if not args.data:
            parser.error("--data is required without --legacy-pkl")
        dataset = ShardDataset(args.data, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, dataset, tokenizer, beam=args.beam,
                       max_tokens=args.max_decode_tokens, prediction_path=args.out)
    metrics["split"] = "legacy_notes" if args.legacy_pkl else args.split
    metrics["beam"] = args.beam
    Path(args.out).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
