import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import ShardDataset, source_weight as weight_for_source, source_weights, token_batches
from .inference import collate, evaluate
from .model import Config, Ink2Latex, training_loss
from .tokenizer import Tokenizer


def learning_rate(step, total, warmup=2000, peak=3e-4):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    floor = min(1e-5, peak / 10)
    return floor + .5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Ink2Latex(len(checkpoint["vocab"]), Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), Tokenizer(checkpoint["vocab"]), checkpoint


def train(data, out, *, steps=100000, resume=None, config=None, device="cpu", seed=42,
          token_budget=24000, checkpoint_every=5000, eval_every=5000, max_decode_tokens=None, source_weight=(),
          workers=0, init=None, lr=3e-4, warmup=2000):
    """`init`: start from a trained checkpoint's weights (fine-tuning) with a fresh optimizer and schedule; `lr` is
    the peak learning rate and `warmup` its warm-up steps. `resume` instead continues a run exactly."""
    if steps < 1 or token_budget < 1 or checkpoint_every < 1 or eval_every < 1:
        raise ValueError("Steps, token budget and intervals must be positive")
    weights = source_weights(source_weight)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.load(Path(data) / "vocab.json")
    dataset = ShardDataset(data, ["train", "user_train"])
    valid = ShardDataset(data, "valid")
    if not len(dataset) or not len(valid):
        raise ValueError("Training requires nonempty train/user_train and valid splits")
    subset = np.random.default_rng(seed).permutation(len(valid))[:2000].tolist()
    if init and resume:
        raise ValueError("Use either --init (fine-tune) or --resume (continue), not both")
    if init:
        model, init_vocab, _ = load_checkpoint(init, device)
        if init_vocab.tokens != tokenizer.tokens:
            raise ValueError("The --init model's vocabulary differs from the prepared dataset's vocab.json")
    else:
        model = Ink2Latex(len(tokenizer), config).to(device)
    peak_lr = lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.05)
    start, epoch, cursor, best = 0, 0, 0, -1.
    if resume:
        loaded, saved_vocab, checkpoint = load_checkpoint(resume, device)
        if saved_vocab.tokens != tokenizer.tokens:
            raise ValueError("Resume vocabulary differs from prepared dataset")
        if checkpoint["seed"] != seed or checkpoint["token_budget"] != token_budget:
            raise ValueError("Resume must use the original seed and token budget")
        if checkpoint.get("source_weights") != weights:
            raise ValueError("Resume must use the original source weights")
        model = loaded
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.05)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, epoch, cursor, best = (checkpoint[k] for k in ("step", "epoch", "cursor", "best"))
        subset = checkpoint["valid_subset"]
        rng.bit_generator.state = checkpoint["augmentation_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if str(device).startswith("cuda") and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    tokenizer.save(out / "vocab.json")
    (out / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n")

    def save(name, step):
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": model.config.to_dict(), "vocab": tokenizer.tokens,
                      "step": step, "epoch": epoch, "cursor": cursor, "best": best,
                      "seed": seed, "token_budget": token_budget, "schedule_steps": steps,
                      "source_weights": weights,
                      "valid_subset": subset, "augmentation_rng": rng.bit_generator.state,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None}
        temporary = out / (name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(out / name)

    if not any(weight_for_source(s, weights) > 0 for s in set(dataset.sources)):
        raise ValueError("At least one training source must have a positive weight")
    counts_path = out / "source_counts.jsonl"
    if not resume:
        counts_path.write_text("")

    def epoch_batches():
        nonlocal epoch
        while True:
            counts = {}
            batches = token_batches(dataset, token_budget, seed, epoch, weights, counts)
            with counts_path.open("a") as counts_log:
                counts_log.write(json.dumps({"epoch": epoch, "effective_source_counts": counts}) + "\n")
            if batches:
                return batches, counts
            # A fractional mixture can legitimately retain no rows this epoch.
            epoch += 1

    batches, effective_counts = epoch_batches()
    loader, loader_epoch = None, None
    with (out / "train.jsonl").open("a" if resume else "w") as log:
        for step in range(start, steps):
            if cursor >= len(batches):
                epoch, cursor = epoch + 1, 0
                batches, effective_counts = epoch_batches()
            if workers:
                # Batches are read and augmented by worker processes ahead of the GPU (each worker its own
                # augmentation stream, so a resumed run is not bit-identical to an uninterrupted one).
                if loader is None or loader_epoch != (epoch, len(batches)):
                    loader = iter(torch.utils.data.DataLoader(
                        _BatchList(dataset, batches[cursor:], seed + 1000 * epoch), batch_size=None, shuffle=False,
                        num_workers=workers, prefetch_factor=4))
                    loader_epoch = (epoch, len(batches))
                points, mask, decoder, targets = (t.to(device, non_blocking=True) for t in next(loader))
            else:
                rows = [dataset[i] for i in batches[cursor]]
                points, mask, decoder, targets = collate(rows, device, rng)
            cursor += 1
            lr = learning_rate(step, steps, warmup, peak_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                logits, coverage, ink_mask = model(points, mask, decoder, targets)
                loss, parts = training_loss(logits, targets, coverage, ink_mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            record = {"step": step + 1, "epoch": epoch, "loss": float(loss.detach()), "lr": lr,
                      "tokens/s": int(targets.ne(0).sum()) / elapsed,
                      "input_points/s": int(mask.sum()) / elapsed, "grad_norm": float(grad_norm),
                      **{key: float(value.detach()) for key, value in parts.items()}}
            if cursor == 1 or step == start:
                record["effective_source_counts"] = effective_counts
            if (step + 1) % eval_every == 0 or step + 1 == steps:
                metrics = evaluate(model, valid, tokenizer, indices=subset, max_tokens=max_decode_tokens)
                record["valid"] = metrics
                if metrics["exact_match"] > best:
                    best = metrics["exact_match"]
                    save("best.pt", step + 1)
            if (step + 1) % checkpoint_every == 0:
                save(f"step-{step + 1:07d}.pt", step + 1)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        save("last.pt", max(start, steps))
    return model


class _BatchList(torch.utils.data.Dataset):
    """One epoch's batches, collated and augmented on the CPU inside a worker process."""

    def __init__(self, dataset, batches, seed):
        self.dataset, self.batches, self.seed = dataset, batches, seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, index))
        return collate([self.dataset[i] for i in self.batches[index]], "cpu", rng)
