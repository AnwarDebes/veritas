// The guard as propositional logic. One variable per vocabulary bit, one per
// clause (its AND, Tseitin style), and a sequential counter for the vote.
// A query is satisfiable exactly when an input breaking the property exists.
//
// Proofs are made over every one of the 2^256 bit patterns. Each of them is
// some real prompt: the tokenizer keeps underscores, so even a lone word pair
// can be typed as "previous_instructions". A proof therefore covers every
// string. The optional "natural" constraint (a pair only appears with both of
// its words) is used only to look for bypasses that read like plain text.

export class CNF {
  constructor(nvars) { this.n = nvars; this.clauses = []; }
  fresh() { return ++this.n; }
  add(...lits) { this.clauses.push(lits); }
}

// at most k of `lits` are true (Sinz 2005)
export function atMost(cnf, lits, k) {
  const n = lits.length;
  if (k >= n) return;
  if (k < 0) { cnf.add(); return; }
  if (k === 0) { for (const l of lits) cnf.add(-l); return; }
  const s = [];
  for (let i = 0; i < n - 1; i++) { s.push([0]); for (let j = 1; j <= k; j++) s[i].push(cnf.fresh()); }
  cnf.add(-lits[0], s[0][1]);
  for (let j = 2; j <= k; j++) cnf.add(-s[0][j]);
  for (let i = 1; i < n - 1; i++) {
    cnf.add(-lits[i], s[i][1]);
    cnf.add(-s[i - 1][1], s[i][1]);
    for (let j = 2; j <= k; j++) { cnf.add(-lits[i], -s[i - 1][j - 1], s[i][j]); cnf.add(-s[i - 1][j], s[i][j]); }
    cnf.add(-lits[i], -s[i - 1][k]);
  }
  cnf.add(-lits[n - 1], -s[n - 2][k]);
}
export const atLeast = (cnf, lits, k) => atMost(cnf, lits.map((l) => -l), lits.length - k);

export class GuardLogic {
  constructor(guard) {
    this.g = guard;
    this.F = guard.F;
  }

  // feature i is variable i + 1
  v(i) { return i + 1; }

  // clause output variables; weighted clauses count `weight` times in the vote
  circuit(cnf) {
    const P = [], N = [];
    for (const c of this.g.clauses) {
      const out = cnf.fresh();
      const lits = c.lits.map((k) => (k < this.F ? this.v(k) : -this.v(k - this.F)));
      for (const L of lits) cnf.add(-out, L);
      cnf.add(out, ...lits.map((L) => -L));
      const w = c.weight || 1;
      for (let i = 0; i < w; i++) (c.unsafe ? P : N).push(out);
    }
    return { P, N };
  }

  // a word pair only appears together with both of its words
  natural(cnf) {
    for (const [t, j] of this.g.index) {
      if (!t.includes('_')) continue;
      for (const w of t.split('_')) { const u = this.g.index.get(w); if (u !== undefined) cnf.add(-this.v(j), this.v(u)); }
    }
  }

  // "every input containing these bits is blocked". SAT = an input that contains them and passes.
  mustBlock(required, { natural = false } = {}) {
    const cnf = new CNF(this.F);
    const { P, N } = this.circuit(cnf);
    atMost(cnf, [...P, ...N.map((c) => -c)], N.length); // vote <= 0
    for (const i of required) cnf.add(this.v(i));
    if (natural) this.natural(cnf);
    return cnf;
  }

  // "nothing is blocked unless one of these words is present". SAT = an input without them that is blocked.
  needsCue(cues, { natural = false } = {}) {
    const cnf = new CNF(this.F);
    const { P, N } = this.circuit(cnf);
    atLeast(cnf, [...P, ...N.map((c) => -c)], N.length + 1); // vote > 0
    const cueSet = new Set(cues);
    for (const [t, j] of this.g.index) {
      // a cue word, or any pair that contains one, must be absent
      if (cueSet.has(j) || (t.includes('_') && t.split('_').some((w) => cueSet.has(this.g.index.get(w))))) cnf.add(-this.v(j));
    }
    if (natural) this.natural(cnf);
    return cnf;
  }

  // read a satisfying assignment back as a feature vector
  witness(model) {
    const x = new Uint8Array(this.F);
    for (let i = 0; i < this.F; i++) x[i] = model[i + 1] === 1 ? 1 : 0;
    return x;
  }
}
