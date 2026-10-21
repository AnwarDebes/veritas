// The TRACE content guard (github.com/AnwarDebes/TRACE), inference only:
// 256 word and word-pair bits, 200 clauses, the first 100 vote "unsafe" and
// the last 100 vote "safe". A prompt is blocked when the vote is above zero.
// Extra clauses (a repair) may carry a weight: they count that many votes.
export class Guard {
  constructor(model, extra = []) {
    this.model = model;
    this.vocab = model.vocab;
    this.F = model.n_features;
    this.C = model.n_clauses;
    this.half = this.C / 2;
    this.index = new Map(model.vocab.map((t, i) => [t, i]));
    // every non-empty clause as { unsafe, lits }, lits are k < F (word present) or k >= F (word absent)
    this.clauses = [];
    model.clauses.forEach((inc, c) => { if (inc.length) this.clauses.push({ id: c, unsafe: c < this.half, lits: inc.slice() }); });
    for (const e of extra) this.clauses.push(e);
  }

  // lowercase, split on anything that is not [a-z0-9_], add adjacent word pairs
  terms(text) {
    const toks = text.toLowerCase().match(/[a-z0-9_]+/g) || [];
    const set = new Set(toks);
    for (let i = 0; i + 1 < toks.length; i++) set.add(toks[i] + '_' + toks[i + 1]);
    return set;
  }

  features(text) {
    const x = new Uint8Array(this.F);
    for (const t of this.terms(text)) { const j = this.index.get(t); if (j !== undefined) x[j] = 1; }
    return x;
  }

  lit(x, k) { return k < this.F ? x[k] : 1 - x[k - this.F]; }

  voteOf(x) {
    let v = 0;
    const fired = [];
    for (const c of this.clauses) {
      let ok = true;
      for (const k of c.lits) if (!this.lit(x, k)) { ok = false; break; }
      if (ok) { v += (c.unsafe ? 1 : -1) * (c.weight || 1); fired.push(c); }
    }
    return { vote: v, blocked: v > 0, fired };
  }

  verdict(text) { const x = this.features(text); return { ...this.voteOf(x), x }; }

  describe(c) {
    const present = [], absent = [];
    for (const k of c.lits) (k < this.F ? present : absent).push(this.vocab[k < this.F ? k : k - this.F]);
    return { present, absent, unsafe: c.unsafe };
  }
}
