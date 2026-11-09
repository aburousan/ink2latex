#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink2latex.data import prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare MathWriting and optional trusted user data")
    parser.add_argument("--mathwriting", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--user-jsonl", nargs="+", action="extend", default=[])
    parser.add_argument("--extra-pkl", nargs="+", action="extend", default=[])
    parser.add_argument("--test-pkl", nargs="+", action="extend", default=[], metavar="NAME=PATH")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--vocab", help="Encode with this vocab.json (a trained model's, for fine-tuning) instead of building one")
    args = parser.parse_args()
    print(json.dumps(prepare(args.mathwriting, args.out, args.user_jsonl, args.extra_pkl,
                             args.min_count, args.shard_size, args.test_pkl, args.vocab), indent=2))


if __name__ == "__main__":
    main()
