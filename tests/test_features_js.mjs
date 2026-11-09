import assert from 'node:assert/strict';
import fs from 'node:fs';
import {features} from '../demo/features.js';

const fixtures = JSON.parse(fs.readFileSync(new URL('./fixtures/features_case.json', import.meta.url)));
let maximum = 0;
for (const fixture of fixtures) {
  const actual = features(fixture.strokes);
  assert.equal(actual.length, fixture.expected.length, fixture.name);
  for (let i = 0; i < actual.length; i++) {
    assert.equal(actual[i].length, 8);
    for (let j = 0; j < 8; j++) {
      const difference = Math.abs(actual[i][j] - fixture.expected[i][j]);
      maximum = Math.max(maximum, difference);
      assert.ok(difference < 1e-4, `${fixture.name}: row ${i}, channel ${j}: ${difference}`);
    }
  }
}
console.log(`${fixtures.length} Python/JavaScript feature cases passed; max abs diff ${maximum}`);
