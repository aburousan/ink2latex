#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ink2latex.data import ShardDataset
from ink2latex.export import export_models
from ink2latex.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export opset-17 models and gate int8 on 1,000 valid rows")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="demo/models")
    parser.add_argument("--data", help="Prepared data for the quantization accuracy gate")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    valid = ShardDataset(args.data, "valid") if args.data else None
    print(json.dumps(export_models(model, tokenizer, args.out, valid, not args.no_quantize), indent=2))


if __name__ == "__main__":
    main()
