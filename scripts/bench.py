#!/usr/bin/env python3
"""Synthetic full-size forward/backward throughput; no data download or checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ink2latex.model import Config, Ink2Latex, training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--token-budget", type=int, default=512, help="Padded input points per batch")
    parser.add_argument("--points", type=int, default=128, help="Input points per sample")
    parser.add_argument("--target-tokens", type=int, default=32, help="Output steps including EOS")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if (min(args.token_budget, args.points, args.target_tokens, args.iterations, args.threads) < 1
            or args.warmup < 0 or args.vocab_size < 4 or args.points > 1600
            or args.target_tokens > 161 or args.token_budget < args.points):
        parser.error("Use positive sizes within model limits and a budget >= points")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = Config()
    model = Ink2Latex(args.vocab_size, config).to(args.device).train()
    batch = args.token_budget // args.points
    points = torch.randn(batch, args.points, 8, device=args.device)
    mask = torch.ones(batch, args.points, dtype=torch.bool, device=args.device)
    targets = torch.randint(3, args.vocab_size, (batch, args.target_tokens), device=args.device)
    targets[:, -1] = 2
    tokens = torch.cat([torch.ones_like(targets[:, :1]), targets[:, :-1]], 1)

    def iteration():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
            logits, coverage, ink_mask = model(points, mask, tokens, targets)
            loss, _ = training_loss(logits, targets, coverage, ink_mask)
        loss.backward()
        return loss

    for _ in range(args.warmup):
        iteration()
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.iterations):
        loss = iteration()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite benchmark loss")
    report = {**vars(args), "config": config.to_dict(), "batch_size": batch,
              "actual_input_points": batch * args.points, "parameters": sum(p.numel() for p in model.parameters()),
              "precision": "bf16" if args.device == "cuda" else "fp32", "seconds": elapsed,
              "seconds/iteration": elapsed / args.iterations, "samples/s": batch * args.iterations / elapsed}
    if args.device == "cuda":
        report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
