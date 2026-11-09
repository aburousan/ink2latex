// No PNG or image packages: 8-bit non-interlaced gray/RGB with all five PNG filters.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {image_to_strokes} from '../demo/photo.js';

function png(data) {
  assert.equal(data.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
  let w, h, channels;
  const chunks = [];
  for (let offset = 8; offset < data.length;) {
    const size = data.readUInt32BE(offset), type = data.toString('ascii', offset + 4, offset + 8), body = data.subarray(offset + 8, offset + 8 + size);
    if (type === 'IHDR') {
      w = body.readUInt32BE(0); h = body.readUInt32BE(4);
      assert.equal(body[8], 8); assert.ok([0, 2].includes(body[9]));
      assert.deepEqual([...body.subarray(10)], [0, 0, 0]);
      channels = body[9] === 0 ? 1 : 3;
    }
    if (type === 'IDAT') chunks.push(body);
    offset += size + 12;
  }
  const raw = inflateSync(Buffer.concat(chunks)), stride = w * channels, rows = [];
  assert.equal(raw.length, h * (stride + 1));
  for (let y = 0; y < h; y++) {
    const row = new Uint8Array(stride), filter = raw[y * (stride + 1)];
    assert.ok(filter <= 4);
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? row[x - channels] : 0, b = y ? rows[y - 1][x] : 0, c = y && x >= channels ? rows[y - 1][x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      row[x] = raw[y * (stride + 1) + x + 1] + [0, a, b, Math.floor((a + b) / 2), predictor][filter];
    }
    rows.push(row);
  }
  return rows.map(row => channels === 1 ? row : Uint8Array.from({length: w}, (_, x) => Math.floor((299 * row[3*x] + 587 * row[3*x+1] + 114 * row[3*x+2] + 500) / 1000)));
}

function compare(actual, expected, name) {
  assert.equal(actual.length, expected.length, `${name}: stroke count`);
  actual.forEach((stroke, i) => {
    assert.equal(stroke.length, expected[i].length, `${name}: point count in stroke ${i}`);
    stroke.forEach((point, j) => point.forEach((v, k) => assert.ok(Math.abs(v - expected[i][j][k]) <= 1e-3, `${name}: ${i}/${j}/${k}: ${v} != ${expected[i][j][k]}`)));
  });
}
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/photo_strokes.json', import.meta.url)));
for (const fixture of fixtures) {
  const gray = png(readFileSync(new URL('./fixtures/' + fixture.file, import.meta.url)));
  compare(image_to_strokes(gray), fixture.strokes, fixture.file);
  compare(image_to_strokes(gray.map(row => row.map(v => 255 - v))), fixture.strokes, fixture.file + ' inverted');
  compare(image_to_strokes(gray.map(row => row.map(v => 30 + Math.floor((255 - v) / 2)))), fixture.strokes, fixture.file + ' gray blackboard');
}
assert.deepEqual(image_to_strokes(Array.from({length: 30}, () => new Uint8Array(40).fill(255))), []);
assert.deepEqual(image_to_strokes([]), []);
console.log(`Photo parity passed: ${fixtures.length} PNGs, inverted copies, and empty images.`);

// Line splitting, the same cases as tests/test_photo.py SplitLinesTest.
{
  const {split_lines} = await import('../demo/photo.js');
  const row = (y, x0 = 0, n = 4) => Array.from({length: n}, (_, k) => [[x0 + 30 * k, y], [x0 + 30 * k + 20, y + 40]]);
  assert.deepEqual(split_lines([...row(0), ...row(80)]), [row(0), row(80)]);
  const fraction = [...row(0), [[0, 50], [110, 50]], ...row(60)];
  assert.deepEqual(split_lines(fraction), [fraction]);
  const equals = [[[0, 0], [40, 0]], [[0, 20], [40, 20]]];
  assert.deepEqual(split_lines(equals), [equals]);
  const lim = [...row(0, 0, 3), [[0, 60], [10, 75]], [[30, 60], [60, 75]]];
  assert.deepEqual(split_lines(lim), [lim]);
  console.log('Line splitting matches: a system splits; a fraction, = and lim stay whole.');
}
