// Constructive repair. A bypass exists, so add one clause that fires exactly
// when every required word is present and votes "unsafe" with weight w. The
// smallest w that makes the property provable is found by binary search: each
// probe is one solver call over every possible input, a satisfiable probe
// hands back the next bypass, and the last unsatisfiable one is the proof,
// which rup.js re-checks without trusting the solver.
import { Guard } from './guard.js';
import { GuardLogic } from './logic.js';
import { solve } from './cdcl.js';
import { now } from './timer.js';

export const fixClause = (required, weight) => ({ id: 'fix', unsafe: true, lits: required.slice(), weight, fix: true });
export const repaired = (model, required, weight) => new Guard(model, [fixClause(required, weight)]);

// is "every input with these words is blocked" provable at this weight?
export function probe(model, required, weight, opts = {}) {
  const g = repaired(model, required, weight);
  const lg = new GuardLogic(g);
  const cnf = lg.mustBlock(required);
  const t0 = now();
  const r = solve(cnf.n, cnf.clauses, opts);
  const out = { weight, g, lg, cnf, r, ms: now() - t0 };
  if (r.sat) { out.x = lg.witness(r.model); out.vote = g.voteOf(out.x).vote; }
  return out;
}

// smallest weight in [1, safe + 1] with a proof. safe + 1 always works: the
// worst case is every safe clause voting, and the fix outvotes all of them.
export function minimalRepair(model, required, { onProbe = null, opts = {} } = {}) {
  const base = new Guard(model);
  const safe = base.clauses.filter((c) => !c.unsafe).length;
  let lo = 1, hi = safe + 1, best = null;
  const probes = [];
  const run = (w) => { const p = probe(model, required, w, opts); probes.push(p); if (onProbe) onProbe(p, { lo, hi }); return p; };
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const p = run(mid);
    if (p.r.sat === false) { hi = mid; best = p; } else lo = mid + 1;
  }
  if (!best || best.weight !== lo) best = run(lo);
  return { weight: lo, best, probes, safe };
}

// the deepest a padded prompt can push the vote while keeping the words: -(weight - 1)
export const floorVote = (weight) => 1 - weight;
