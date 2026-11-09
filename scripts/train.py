#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ink2latex.model import Config
from ink2latex.training import train


def main():
    parser = argparse.ArgumentParser(description="Train ink2latex; CPU --tiny is for smoke tests")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--resume", help="Continue an interrupted run exactly (same data, seed and weights)")
    parser.add_argument("--init", help="Fine-tune: start from this checkpoint's weights with a fresh optimizer and schedule")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (fine-tuning: try 5e-5 to 1e-4)")
    parser.add_argument("--warmup", type=int, default=2000, help="Warm-up steps (fine-tuning: try 200)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=24000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--max-decode-tokens", type=int)
    parser.add_argument("--source-weight", action="append", default=[], metavar="PREFIX=W")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="Batch-preparing worker processes (0: in the main process, bit-reproducible)")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--config", help="JSON Config overrides")
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop("threads"))
    tiny, filename = args.pop("tiny"), args.pop("config")
    config = Config(d=64, ff=256, encoder_layers=2, decoder_layers=2) if tiny else Config()
    if filename:
        config = Config(**{**config.to_dict(), **json.loads(Path(filename).read_text())})
    train(config=config, **args)


if __name__ == "__main__":
    main()
