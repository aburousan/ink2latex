import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ink2latex.features import features
from ink2latex.onnx_reader import decode
from ink2latex.photo import image_to_strokes, _thin, _order, _resize
from ink2latex.photo_io import read_gray, read_png, write_png, draw_strokes, blur_noise
from scripts.eval_photo import reconstruct_strokes, metrics
from scripts.predict import read_ink

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'


def check_exported_photo(test, exported, prepared):
    """Called by the existing smoke test while its temporary export still exists."""
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'strokes.png'
        for beam in (1, 5):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/predict.py'), '--models', str(exported),
                                     '--image', str(FIXTURES / 'photo_cases/crossed_t.png'), '--beam', str(beam),
                                     '--show-strokes', str(output)], text=True, capture_output=True, timeout=90)
            test.assertEqual(result.returncode, 0, result.stderr)
            test.assertTrue(result.stdout.endswith('\n'))
            test.assertRegex(result.stderr, r'\d+\.\d+ ms')
            test.assertTrue((read_png(output) < 255).any())
        report = Path(temporary) / 'evaluation.jsonl'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/eval_photo.py'), '--models', str(exported),
                                 '--data', str(prepared), '--split', 'valid', '--rows', '2', '--seed', '0',
                                 '--out', str(report)], text=True, capture_output=True, timeout=90)
        test.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        test.assertEqual(len(rows), 2)
        summary = json.loads(result.stdout)
        test.assertEqual(summary['original']['samples'], 2)
        test.assertEqual(summary['photo']['samples'], 2)
        test.assertEqual(summary['photo_empty'], sum(row['photo_empty'] for row in rows))


class PhotoTests(unittest.TestCase):
    def test_fixtures_and_inversion(self):
        for case in json.loads((FIXTURES / 'photo_strokes.json').read_text()):
            gray = read_gray(FIXTURES / case['file'])
            expected = case['strokes']
            for image in (gray, 255 - gray, 30 + (255 - gray) // 2):
                strokes = image_to_strokes(image)
                self.assertEqual(json.loads(json.dumps(strokes)), expected)
                self.assertEqual([p[2] for s in strokes for p in s], list(range(sum(map(len, strokes)))))

    def test_empty_and_validation(self):
        for value in (0, 127, 255):
            self.assertEqual(image_to_strokes(np.full((35, 40), value)), [])
        self.assertEqual(image_to_strokes(np.empty((0, 0))), [])
        for gray in (np.zeros((2, 2, 3)), [[-1]], [[256]], [[float('nan')]]):
            with self.assertRaises(ValueError):
                image_to_strokes(gray)

    def test_dot_and_crossing(self):
        dot = image_to_strokes(read_gray(FIXTURES / 'photo_cases/dotted_i.png'))
        self.assertEqual(sorted(map(len, dot)), [1, 2])
        # Written order: the stem, then its dot (which sits above it).
        self.assertEqual(len(dot[1]), 1)
        self.assertLess(dot[1][0][1], dot[0][0][1])
        cross = image_to_strokes(read_gray(FIXTURES / 'photo_cases/crossed_t.png'))
        self.assertEqual(len(cross), 2)
        spans = [np.ptp(np.array(s)[:, :2], axis=0) for s in cross]
        self.assertTrue(any(x > 70 and y < 3 for x, y in spans))
        self.assertTrue(any(y > 80 and x < 3 for x, y in spans))

    def test_order_and_direction(self):
        lines = [[(60, 45), (60, 10)], [(40, 30), (10, 30)], [(10, 90), (40, 90)]]
        ordered = _order(lines, 5)
        self.assertEqual(ordered, [[(10, 30), (40, 30)], [(60, 10), (60, 45)], [(10, 90), (40, 90)]])
        gray = draw_strokes([[(20, 20), (20, 100)], [(90, 20), (90, 100)]], (130, 130), 3)
        strokes = image_to_strokes(gray)
        self.assertEqual(len(strokes), 2)
        self.assertLess(strokes[0][0][0], strokes[1][0][0])
        self.assertTrue(all(s[0][1] < s[-1][1] for s in strokes))

    def test_downscale_and_specks(self):
        gray = np.full((80, 2400), 255, np.uint8)
        gray[30:50, 300:2100] = 0
        resized, scale = _resize(gray)
        self.assertEqual(resized.shape, (40, 1200))
        self.assertEqual(scale, (2, 2))
        strokes = image_to_strokes(gray)
        self.assertEqual(len(strokes), 1)
        self.assertGreater(strokes[0][-1][0], 2000)
        image = np.full((150, 150), 255, np.uint8)
        image[20, 20] = 0
        self.assertEqual(image_to_strokes(image), [])

    def test_loop_and_thinning(self):
        gray = draw_strokes([[(20, 20), (100, 20), (100, 100), (20, 100), (20, 20)]], (128, 128), 3)
        strokes = image_to_strokes(gray)
        self.assertTrue(strokes)
        points = [(x, y) for s in strokes for x, y, _ in s]
        self.assertEqual(len(points), len(set(points)))
        skeleton = _thin(gray == 0)
        np.testing.assert_array_equal(_thin(skeleton), skeleton)

    def test_io_and_evaluation_helpers(self):
        strokes = [[[0, 0], [1, 1]], [[2, 2]]]
        points = features(strokes)
        recovered = reconstruct_strokes(points)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(len(recovered[1]), 1)
        self.assertEqual(metrics([['x', 'y'], ['z']], [['x'], ['q']])['token_error_rate'], 2 / 3)
        self.assertIsNone(metrics([], [])['exact_match'])
        gray = np.full((20, 30), 255, np.uint8)
        a = blur_noise(gray, np.random.default_rng(0))
        np.testing.assert_array_equal(a, blur_noise(gray, np.random.default_rng(0)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            write_png(path, a)
            with patch.dict(sys.modules, {'PIL': None}):
                np.testing.assert_array_equal(read_gray(path), a)
            ink = Path(temporary) / 'ink.json'
            for row in (strokes, {'strokes': strokes, 'latex': 'x'}):
                ink.write_text(json.dumps(row) + '\n')
                self.assertEqual(read_ink(ink), strokes)

    def test_decoding_helpers(self):
        class Encoder:
            def run(self, names, inputs):
                self.inputs = inputs
                return np.zeros((1, 1, 4), np.float32), np.ones((1, 1), bool)

        class Decoder:
            def run(self, names, inputs):
                prefix = inputs['previous_tokens'][0].tolist()
                # Huge PAD/BOS scores must be suppressed. Beam finds the better complete path.
                logits = [100, 100, -10, 2, 1.9] if len(prefix) == 1 else (
                    [100, 100, 0, 0, 0] if prefix[-1] == 3 else [100, 100, 10, -10, -10])
                return [np.array([logits], np.float32)]

        encoder, decoder = Encoder(), Decoder()
        points = np.zeros((1, 8), np.float32)
        self.assertEqual(decode(encoder, decoder, points), [3])
        self.assertEqual(decode(encoder, decoder, points, beam=5), [4])
        self.assertEqual(decode(encoder, decoder, points, max_tokens=1), [3])
        self.assertEqual(decode(encoder, decoder, []), [])
        self.assertEqual(encoder.inputs['points'].shape, (1, 1, 8))
        self.assertEqual(encoder.inputs['point_mask'].dtype, bool)


if __name__ == '__main__':
    unittest.main()


class SplitLinesTest(unittest.TestCase):
    """Separate lines are read one by one; a fraction, an = sign or lim over x->3 stay one formula."""
    @staticmethod
    def row(y, x0=0, n=4):
        return [[(x0 + 30 * k, y), (x0 + 30 * k + 20, y + 40)] for k in range(n)]

    def test_system_splits_and_formulas_do_not(self):
        from ink2latex.photo import split_lines
        system = self.row(0) + self.row(80)
        self.assertEqual(split_lines(system), [self.row(0), self.row(80)])
        fraction = self.row(0) + [[(0, 50), (110, 50)]] + self.row(60)
        self.assertEqual(split_lines(fraction), [fraction])
        equals = [[(0, 0), (40, 0)], [(0, 20), (40, 20)]]
        self.assertEqual(split_lines(equals), [equals])
        lim = self.row(0, n=3) + [[(0, 60), (10, 75)], [(30, 60), (60, 75)]]     # two strokes below: not a line
        self.assertEqual(split_lines(lim), [lim])
