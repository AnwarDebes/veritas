// The solver and the proof checker against brute force on small random formulas.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { solve, satisfies } from '../cdcl.js';
import { checkRUP } from '../rup.js';

let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);

test('agrees with brute force, and every UNSAT answer has a proof that checks', () => {
  let unsat = 0;
  for (let t = 0; t < 3000; t++) {
    const n = 4 + Math.floor(rnd() * 13), m = Math.round(n * (3.6 + rnd() * 1.4));
    const cl = [];
    for (let i = 0; i < m; i++) {
      const k = rnd() < 0.85 ? 3 : 2, c = [];
      for (let j = 0; j < k; j++) c.push((1 + Math.floor(rnd() * n)) * (rnd() < 0.5 ? 1 : -1));
      cl.push(c);
    }
    let brute = false;
    for (let a = 0; a < 1 << n && !brute; a++) {
      const mdl = [0];
      for (let v = 1; v <= n; v++) mdl.push((a >> (v - 1)) & 1 ? 1 : -1);
      if (satisfies(mdl, cl)) brute = true;
    }
    const r = solve(n, cl);
    assert.equal(r.sat, brute, `formula ${t}`);
    if (r.sat) assert.ok(satisfies(r.model, cl), `model ${t}`);
    else { unsat++; assert.ok(checkRUP(n, cl, r.learnts).ok, `proof ${t}`); }
  }
  assert.ok(unsat > 500);
});

test('a fake proof for a satisfiable formula is rejected', () => {
  assert.equal(checkRUP(2, [[1, 2], [-1, 2], [1, -2]], [[-2], []]).ok, false);
});
