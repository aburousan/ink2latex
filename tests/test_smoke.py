"""CPU-only synthetic integration tests. Run with python3 -m unittest discover -s tests -v."""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from ink2latex.data import ShardDataset, prepare, read_inkml, source_weights, token_batches
from ink2latex.features import augment, features
from ink2latex.inference import collate, score_predictions
from ink2latex.model import Config, Ink2Latex, coverage_loss, training_loss
from ink2latex.tokenizer import Tokenizer, detokenize, tokenize
from ink2latex.training import learning_rate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def command(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    if result.returncode:
        raise AssertionError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def fake_data(directory):
    directory.mkdir()
    ink = '<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="label">bad</annotation><annotation type="normalizedLabel">{label}</annotation>{trace}</ink>'
    trace = '<trace>0 0 0, 0.4 1 1, 0.8 0 2</trace><trace>1 0.4 0, 1.2 0.6 1</trace>'
    for split, count in (("train", 24), ("synthetic", 2), ("symbols", 2), ("valid", 4), ("test", 3)):
        (directory / split).mkdir()
        for i in range(count):
            label = "x+y" if not (split == "valid" and i == 3) else r"\unseen{x}"
            (directory / split / f"{i:03}.inkml").write_text(ink.format(label=label, trace=trace))
    for name, label, traces in (("rare", r"\rare", trace), ("long", "x" * 161, trace), ("empty", "x", "")):
        (directory / "train" / f"{name}.inkml").write_text(ink.format(label=label, trace=traces))
    (directory / "test" / "capped.inkml").write_text(ink.format(label="x", trace='<trace>0 0 0, 100 1 1</trace>'))
    user = directory.parent / "user.jsonl"
    strokes = [[[0, 0, 0, .5], [1, 1, 1, .8]]]
    user.write_text(json.dumps({"latex": "x+y", "strokes": strokes}) + "\n" +
                    json.dumps({"latex": r"\physics", "strokes": strokes, "meta": {"kind": "physics"}}) + "\n")
    extra = directory.parent / "extra.pkl"
    with extra.open("wb") as stream:
        pickle.dump([([[[0, 0], [1, 1]]], ["x", "+", "y"])], stream)
    return user, extra


class UnitTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        labels = [r"\frac{x_1^2}{\alpha x}+\{\%\}", r"\left( x+y \right)", "a b"]
        tokenizer = Tokenizer.build(labels * 20)
        for label in labels:
            tokens = tokenize(label)
            self.assertEqual(tokenize(detokenize(tokens)), tokens)
            self.assertEqual(tokenizer.decode([1, *tokenizer.encode(label), 2, 0]), tokens)
        self.assertNotIn(r"\rare", Tokenizer.build([r"\rare"] + ["x"] * 20).ids)

    def test_features(self):
        fixtures = json.loads((ROOT / "tests/fixtures/features_case.json").read_text())
        for fixture in fixtures:
            actual = features(fixture["strokes"])
            expected = np.array(fixture["expected"], np.float32).reshape(-1, 8)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, features(fixture["strokes"]))
        dot = features([[], [[20, 40, 0]]])
        np.testing.assert_array_equal(dot, [[0, 0, 0, 0, 1, 1, 0, 0]])
        original = [[[0, 0], [1, 2]], [[2, 1]]]
        shifted = [[[x * 3 + 100, y * 3 - 20] for x, y in s] for s in original]
        np.testing.assert_allclose(features(original), features(shifted), atol=1e-6)
        with self.assertRaises(ValueError):
            features([[[float("nan"), 0]]])

    def test_augmentation(self):
        ink = features([[[0, 0], [1, 1]], [[2, 2]], [[3, 0], [3, 1]]])
        a = augment(ink, np.random.default_rng(9))
        np.testing.assert_array_equal(a, augment(ink, np.random.default_rng(9)))
        self.assertEqual(int(a[:, 4].sum()), 3)
        self.assertEqual(int(a[:, 5].sum()), 3)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a[a[:, 5] == 1, 2:4] == 0).all())

    def test_coverage_loss_and_gradients(self):
        cov = torch.tensor([[5., 0., 0., 0., 0., 99.]], requires_grad=True)
        twice, unread = coverage_loss(cov, torch.tensor([[True] * 5 + [False]]), torch.tensor([5]))
        self.assertEqual(twice.shape, torch.Size([]))
        self.assertAlmostEqual(twice.item(), .2, places=6)
        self.assertAlmostEqual(unread.item(), .2, places=6)
        (twice + unread).backward()
        self.assertEqual(cov.grad[0, -1], 0)
        model = Ink2Latex(8, Config(d=64, ff=128, encoder_layers=2, decoder_layers=2, dropout=0))
        points = torch.randn(2, 13, 8)
        targets = torch.tensor([[3, 4, 2], [3, 2, 0]])
        with patch.object(model, "step", side_effect=AssertionError("Training must be parallel")):
            logits, coverage, mask = model(points, torch.ones(2, 13, dtype=torch.bool), torch.tensor([[1, 3, 4], [1, 3, 0]]), targets)
        self.assertEqual(logits.shape, (2, 3, 8))
        np.testing.assert_allclose(coverage.sum(1).detach().numpy(), [3, 2], atol=1e-5)
        training_loss(logits, targets, coverage, mask)[0].backward()
        for layer in model.layers:
            self.assertTrue(torch.isfinite(layer.coverage_weight.grad).all())
            self.assertGreater(layer.coverage_weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(layer.coverage_conv.weight.grad).all())
            self.assertGreater(layer.coverage_conv.weight.grad.abs().sum().item(), 0)

    def test_masks_and_causality(self):
        model = Ink2Latex(8, Config(d=64, ff=128, encoder_layers=2, decoder_layers=2, dropout=0)).eval()
        points = torch.randn(1, 9, 8)
        with torch.no_grad():
            for layer in model.layers:
                layer.coverage_conv.weight.zero_()
                layer.coverage_conv.weight[:, 0, 2] = 1.
            memory, mask = model.encoder(points, torch.ones(1, 9, dtype=torch.bool))
            padded = torch.cat([points, torch.zeros(1, 8, 8)], 1)
            padded_memory, _ = model.encoder(padded, torch.tensor([[True] * 9 + [False] * 8]))
            torch.testing.assert_close(memory, padded_memory[:, :3], atol=2e-6, rtol=2e-5)
            coverage = model.initial_coverage(memory)
            tokens, cache = torch.ones(1, 1, dtype=torch.long), None
            for t in range(4):
                logits = model.prefix_step(tokens, memory, mask)
                cached, cached_cov, cache = model.step(tokens[:, -1], memory, mask, coverage, t, cache)
                torch.testing.assert_close(logits, cached, atol=2e-6, rtol=2e-5)
                tokens = torch.cat([tokens, torch.tensor([[3 + t]])], 1)
                coverage = cached_cov
            zero_logits, _, _ = model.step(tokens[:, -1], memory, mask, torch.zeros_like(coverage), 4, cache)
            fed_logits, _, _ = model.step(tokens[:, -1], memory, mask, coverage, 4, cache)
            self.assertGreater((zero_logits - fed_logits).abs().max().item(), 1e-5)

    def test_metrics_and_schedule(self):
        metrics = score_predictions([["{", "x", "}"], ["a"]], [["x"], ["b"]])
        self.assertEqual(metrics["exact_match"], 0)
        self.assertEqual(metrics["loose_exact_match"], .5)
        self.assertEqual(metrics["token_error_rate"], .75)
        self.assertAlmostEqual(learning_rate(1999, 10000), 3e-4)
        self.assertAlmostEqual(learning_rate(9999, 10000), 1e-5)
        self.assertAlmostEqual(learning_rate(199, 1000, warmup=200, peak=5e-5), 5e-5)
        self.assertAlmostEqual(learning_rate(999, 1000, warmup=200, peak=5e-5), 5e-6)

    def test_parallel_incremental_random_prefixes(self):
        torch.manual_seed(123)
        model = Ink2Latex(19, Config(d=64, ff=128, encoder_layers=2, decoder_layers=2, dropout=0)).eval()
        maximum = 0.
        with torch.no_grad():
            for length in (1, 2, 7, 31):
                memory = torch.randn(3, 11, 64)
                mask = torch.arange(11)[None] < torch.tensor([11, 7, 1])[:, None]
                tokens = torch.randint(3, 19, (3, length))
                tokens[:, 0] = 1
                tokens[1, max(1, length // 2):] = 0
                if length > 2:
                    tokens[2, 1] = 0  # Padding must not contribute even inside a prefix.
                logits, cov = model.decode(tokens, memory, mask)
                coverage, cache, incremental = model.initial_coverage(memory), None, []
                kv = model.project_memory(memory)
                for t in range(length):
                    step, coverage, cache = model.step(tokens[:, t], memory, mask, coverage, t, cache, kv)
                    incremental.append(step)
                error = (logits - torch.stack(incremental, 1)).abs().max().item()
                maximum = max(maximum, error)
                self.assertLess(error, 1e-4)
                torch.testing.assert_close(cov.sum(1), tokens.ne(0).sum(1).float())
                self.assertEqual(cov[~mask].abs().sum().item(), 0.)
                for layer_cov in coverage:
                    torch.testing.assert_close(layer_cov.sum(-1), tokens.ne(0).sum(1)[:, None].expand(-1, 8).float())
                changed = tokens.clone()
                changed[:, -1] = 4
                torch.testing.assert_close(model.decode(changed, memory, mask)[0][:, :-1], logits[:, :-1])
        print(f"\nRandom parallel/incremental maximum difference: {maximum:.9g}")

    def test_relative_coverage_lengths_and_padding(self):
        # Uniform attention has zero penalties even when T is much smaller than S.
        for ink_length, target_length in ((40, 2), (3, 20)):
            cov = torch.full((2, ink_length + 3), target_length / ink_length, requires_grad=True)
            mask = torch.arange(ink_length + 3)[None].expand(2, -1) < ink_length
            twice, unread = coverage_loss(cov, mask, torch.full((2,), target_length))
            self.assertEqual(twice.item(), 0.)
            self.assertEqual(unread.item(), 0.)
            (twice + unread).backward()
            self.assertTrue(torch.isfinite(cov.grad).all())
            self.assertEqual(cov.grad[~mask].abs().sum().item(), 0.)
        targets = torch.tensor([[3, 2, 0]])
        cov = torch.tensor([[2., 0., 0., 0., 0., 99.]])
        mask = torch.tensor([[True] * 5 + [False]])
        loss, parts = training_loss(torch.zeros(1, 3, 5), targets, cov, mask)
        self.assertAlmostEqual(parts["coverage_unread_fraction"].item(), .8)
        self.assertAlmostEqual(parts["coverage_twice"].item(), .2)
        self.assertAlmostEqual(parts["coverage_unread"].item(), .2)
        self.assertAlmostEqual(loss.item(), parts["cross_entropy"].item() + .02, places=6)

    def test_source_sampling(self):
        class Dataset:
            sources = ["mathwriting/train", "mathwriting/synthetic", "extra/notes", "user:ink"] * 1000
            lengths = [10] * len(sources)

            def __len__(self):
                return len(self.sources)

        data = Dataset()
        counts = {}
        weights = source_weights(["extra=0", "extra/notes=2.5", "user=3"])
        batches = token_batches(data, 100, 17, weights=weights, effective_counts=counts)
        self.assertEqual(batches, token_batches(data, 100, 17, weights=weights))
        self.assertNotEqual(batches, token_batches(data, 100, 17, epoch=1, weights=weights))
        self.assertEqual(counts["mathwriting/train"], 1000)
        self.assertEqual(counts["user:ink"], 3000)
        self.assertTrue(540 < counts["mathwriting/synthetic"] < 660)
        self.assertTrue(2440 < counts["extra/notes"] < 2560)
        repeats = np.bincount([i for batch in batches for i in batch], minlength=len(data))
        self.assertTrue(np.isin(repeats[2::4], [2, 3]).all())
        self.assertTrue(np.isin(repeats[1::4], [0, 1]).all())
        self.assertEqual(token_batches(data, 100, 17, weights={"": 0}), [])
        for entry in ("user=-1", "extra=nan", "extra=inf", "=2", "user"):
            with self.assertRaises(ValueError):
                source_weights([entry])

    def test_new_input_formats_and_heldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mathwriting = root / "mathwriting"
            user, extra = fake_data(mathwriting)
            user.write_text(json.dumps({"label": "x", "strokes": [[[0, 0], [1, 1]]]}) + "\n" +
                            json.dumps({"label": r"\physics", "strokes": [[[0, 0]]], "meta": {"kind": "physics"}}) + "\n")
            with extra.open("wb") as stream:
                pickle.dump([([np.array([[0, 0], [1, 1]])], ["x", "+", "y"]),
                             ([np.array([[0, 0, 0], [1, 1, 1]])], "x+y")], stream)
            heldout = root / "notes.pkl"
            with heldout.open("wb") as stream:
                pickle.dump([([np.array([[0, 0], [1, 1]])], r"\heldout")], stream)
            prepared = root / "prepared"
            command("scripts/prep.py", "--mathwriting", mathwriting, "--out", prepared,
                    "--user-jsonl", user, "--extra-pkl", f"notes={extra}",
                    "--test-pkl", f"legacy={heldout}", "--min-count", 1)
            dataset = ShardDataset(prepared, ["train", "user_train"])
            self.assertEqual(dataset.sources.count("extra/notes"), 2)
            self.assertNotIn(r"\heldout", Tokenizer.load(prepared / "vocab.json").ids)
            self.assertNotIn(r"\physics", Tokenizer.load(prepared / "vocab.json").ids)
            self.assertEqual(ShardDataset(prepared, "user_train")[0][2]["tokens"], ["x"])
            self.assertEqual(len(ShardDataset(prepared, "user_physics_test")), 1)
            self.assertEqual(ShardDataset(prepared, "legacy")[0][1].tolist(), [-1])
            for reserved in ("train", "user_train", "valid", "test", "../escape"):
                with self.assertRaises(ValueError):
                    prepare(mathwriting, root / "bad", test_pkl=[f"{reserved}={heldout}"])
            # Fine-tuning data keeps a given vocabulary: same tokens, labels encoded with it, rare tokens dropped.
            given = Tokenizer(Tokenizer.load(prepared / "vocab.json").tokens[:-1])
            given.save(root / "given_vocab.json")
            prepare(mathwriting, root / "fixed", user_jsonl=[user], vocab=root / "given_vocab.json")
            self.assertEqual(Tokenizer.load(root / "fixed" / "vocab.json").tokens, given.tokens)
            fixed = ShardDataset(root / "fixed", ["train", "user_train"])
            self.assertGreater(len(fixed), 0)
            for k in range(len(fixed)):
                _, ids, meta = fixed[k]
                self.assertEqual(given.encode(meta["tokens"]), ids.tolist())

            options = ["--source-weight", "mathwriting/train=0", "--source-weight", "mathwriting/synthetic=0",
                       "--source-weight", "mathwriting/symbols=0", "--source-weight", "extra=0",
                       "--source-weight", "user=0.5"]
            run = root / "weighted"
            command("scripts/train.py", "--data", prepared, "--out", run, "--tiny", "--steps", 1,
                    "--device", "cpu", "--max-decode-tokens", 2, *options)
            counts = [json.loads(line) for line in (run / "source_counts.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in counts], [0, 1, 2])
            self.assertEqual([sum(row["effective_source_counts"].values()) for row in counts], [0, 0, 1])
            command("scripts/train.py", "--data", prepared, "--out", root / "resume",
                    "--resume", run / "last.pt", "--steps", 2, "--device", "cpu",
                    "--max-decode-tokens", 2, *options)
            with self.assertRaisesRegex(AssertionError, "Resume must use the original source weights"):
                command("scripts/train.py", "--data", prepared, "--out", root / "bad_resume",
                        "--resume", run / "last.pt", "--steps", 2, "--device", "cpu")


class PipelineSmoke(unittest.TestCase):
    def test_prep_train_eval_export_resume(self):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mathwriting, prepared, run = root / "mathwriting", root / "data", root / "run"
            user, extra = fake_data(mathwriting)
            command("scripts/prep.py", "--mathwriting", mathwriting, "--out", prepared,
                    "--user-jsonl", user, "--extra-pkl", extra, "--test-pkl", f"legacy={extra}", "--shard-size", 8)
            report = json.loads((prepared / "prep_report.json").read_text())
            self.assertEqual(report["dropped"], {"empty_ink": 1, "too_long": 1, "rare_token": 1})
            self.assertEqual(report["splits"], {"train": 29, "user_train": 1, "valid": 4, "test": 4, "user_physics_test": 1, "legacy": 1})
            self.assertEqual(report["capped"], 1)
            self.assertEqual(read_inkml(mathwriting / "train/000.inkml")[1], "x+y")
            vocab = Tokenizer.load(prepared / "vocab.json")
            self.assertNotIn(r"\unseen", vocab.ids)
            valid = ShardDataset(prepared, "valid")
            self.assertIn(-1, valid[3][1])
            data = ShardDataset(prepared, ["train", "user_train"])
            batches = token_batches(data, 128, 42, weights=source_weights(["mathwriting/synthetic=1", "extra=1"]))
            self.assertEqual(sum(i == 29 for batch in batches for i in batch), 50)
            self.assertEqual(sum(len(b) for b in batches), 79)
            self.assertTrue(all(max(data.lengths[i] for i in b) * len(b) <= 128 for b in batches))
            self.assertEqual(data[0][0].dtype, np.float32)
            command("scripts/train.py", "--data", prepared, "--out", run, "--steps", 30,
                    "--tiny", "--device", "cpu", "--token-budget", 128,
                    "--checkpoint-every", 15, "--eval-every", 15, "--max-decode-tokens", 8)
            records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 30)
            self.assertTrue(all(np.isfinite(r["loss"]) and r["tokens/s"] > 0 for r in records))
            self.assertIn("coverage_unread", records[-1])
            self.assertIn("coverage_unread_fraction", records[-1])
            counts = {}
            token_batches(data, 128, 42, effective_counts=counts)
            self.assertEqual(records[0]["effective_source_counts"], counts)
            epoch_counts = [json.loads(line) for line in (run / "source_counts.jsonl").read_text().splitlines()]
            for row in epoch_counts:
                expected = {}
                token_batches(data, 128, 42, row["epoch"], effective_counts=expected)
                self.assertEqual(row["effective_source_counts"], expected)
            self.assertTrue((run / "best.pt").exists())
            for beam, split in ((1, "test"), (5, "valid"), (1, "user_physics_test"), (1, "legacy")):
                prediction = root / f"{split}.jsonl"
                command("scripts/eval.py", "--data", prepared, "--checkpoint", run / "last.pt",
                        "--out", prediction, "--beam", beam, "--split", split, "--max-decode-tokens", 8)
                self.assertEqual(len(prediction.read_text().splitlines()), report["splits"][split])
            command("scripts/eval.py", "--legacy-pkl", extra, "--checkpoint", run / "last.pt",
                    "--out", root / "legacy.jsonl", "--max-decode-tokens", 8)
            exported = root / "export"
            command("scripts/export_onnx.py", "--checkpoint", run / "last.pt", "--out", exported, "--data", prepared)
            export_report = json.loads((exported / "export_report.json").read_text())
            self.assertLess(max(export_report["max_abs_diff"].values()), 1e-4)
            self.assertFalse(export_report["int8"]["accepted"])
            self.assertIn("bytes", export_report["int8"])
            self.assertFalse((exported / "encoder.int8.onnx").exists())
            from test_photo import check_exported_photo
            check_exported_photo(self, exported, prepared)
            resumed = root / "resume"
            command("scripts/train.py", "--data", prepared, "--out", resumed, "--steps", 30,
                    "--resume", run / "step-0000015.pt", "--device", "cpu", "--token-budget", 128,
                    "--checkpoint-every", 15, "--eval-every", 15, "--max-decode-tokens", 8)
            original, _, _ = load_checkpoint(run / "last.pt")
            restored, _, _ = load_checkpoint(resumed / "last.pt")
            for name, tensor in original.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
            # Fine-tuning: start from the trained weights with a fresh, smaller schedule.
            tuned = root / "tune"
            command("scripts/train.py", "--data", prepared, "--out", tuned, "--steps", 6, "--init", run / "last.pt",
                    "--lr", 5e-5, "--warmup", 2, "--device", "cpu", "--token-budget", 128,
                    "--checkpoint-every", 6, "--eval-every", 6, "--max-decode-tokens", 8)
            log = [json.loads(line) for line in (tuned / "train.jsonl").read_text().splitlines() if '"loss"' in line]
            self.assertEqual(log[0]["step"], 1)
            self.assertAlmostEqual(log[1]["lr"], 5e-5)
            self.assertLessEqual(max(r["lr"] for r in log), 5e-5 + 1e-12)
            changed, _, _ = load_checkpoint(tuned / "last.pt")
            self.assertTrue(any(not torch.equal(t, changed.state_dict()[n]) for n, t in original.state_dict().items()))
            print("\nSmoke export parity:", json.dumps(export_report["max_abs_diff"]))
        elapsed = time.perf_counter() - started
        print(f"Synthetic CPU pipeline (30 steps + deterministic resume): {elapsed:.2f}s")
        self.assertLess(elapsed, 120, "CPU smoke pipeline exceeded the two-minute budget")


if __name__ == "__main__":
    unittest.main()
