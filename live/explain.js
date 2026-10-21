// Turning solver answers into prompts a person can read, and checking them on the guard.
import { solve } from './cdcl.js';
import { GuardLogic } from './logic.js';
import { now } from './timer.js';

// at most m of the given variables are true (small m, sequential counter)
export function capAtMost(cnf, vars, m) {
  const n = vars.length;
  if (m >= n) return;
  if (m === 0) { for (const v of vars) cnf.add(-v); return; }
  const s = [];
  for (let i = 0; i < n - 1; i++) { s.push([0]); for (let j = 1; j <= m; j++) s[i].push(cnf.fresh()); }
  cnf.add(-vars[0], s[0][1]);
  for (let j = 2; j <= m; j++) cnf.add(-s[0][j]);
  for (let i = 1; i < n - 1; i++) {
    cnf.add(-vars[i], s[i][1]); cnf.add(-s[i - 1][1], s[i][1]);
    for (let j = 2; j <= m; j++) { cnf.add(-vars[i], -s[i - 1][j - 1], s[i][j]); cnf.add(-s[i - 1][j], s[i][j]); }
    cnf.add(-vars[i], -s[i - 1][m]);
  }
  cnf.add(-vars[n - 1], -s[n - 2][m]);
}

// the smallest witness: add "at most m optional bits are on" for m = 0, 1, 2, ... until satisfiable
export function smallest(build, optional, maxM = 12, opts = {}) {
  let calls = 0, ms = 0;
  for (let m = 0; m <= maxM; m++) {
    const cnf = build();
    capAtMost(cnf, optional.map((i) => i + 1), m);
    const t0 = now();
    const r = solve(cnf.n, cnf.clauses, opts);
    ms += now() - t0; calls++;
    if (r.sat) return { m, r, cnf, calls, ms };
  }
  return null;
}

const FILLERS = ['um', 'hm', 'uh', 'mm', 'erm', 'hmm', 'ok', 'so'];

// bits -> a sentence that produces exactly those bits. The lead phrase is kept
// intact when all of its bits are on. A pair whose words are both on is written
// as two words, otherwise as the literal token "a_b" (the tokenizer keeps "_").
export function render(guard, x, lead = '') {
  const on = new Set();
  for (let i = 0; i < guard.F; i++) if (x[i]) on.add(i);
  const done = new Set();
  const segs = [];
  let leadUsed = false;
  if (lead) {
    const lf = guard.features(lead);
    let ok = true;
    for (let i = 0; i < guard.F; i++) if (lf[i] && !on.has(i)) { ok = false; break; }
    if (ok) { leadUsed = true; segs.push(lead.trim()); for (let i = 0; i < guard.F; i++) if (lf[i]) done.add(i); }
  }
  const word = (w) => guard.index.get(w);
  // writing "a b" also turns on a and b, so it is fine when each is on or not a feature at all
  const free = (w) => { const j = word(w); return j === undefined || on.has(j); };
  for (const i of on) {
    const t = guard.vocab[i];
    if (done.has(i) || !t.includes('_')) continue;
    const [a, b] = t.split('_');
    if (free(a) && free(b)) { segs.push(`${a} ${b}`); done.add(i); for (const w of [a, b]) if (word(w) !== undefined) done.add(word(w)); }
    else { segs.push(t); done.add(i); }
  }
  for (const i of on) if (!done.has(i)) segs.push(guard.vocab[i]);
  const fillers = FILLERS.filter((w) => !guard.index.has(w));
  const exact = (text) => { const f = guard.features(text); for (let i = 0; i < guard.F; i++) if (f[i] !== x[i]) return false; return true; };
  const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  // the lead goes last, as typed; extra words come first
  // a sentence-initial capital on the typed phrase drops once words go in front of it
  if (leadUsed && segs.length > 1 && /^[A-Z][a-z]/.test(segs[0])) segs[0] = segs[0].charAt(0).toLowerCase() + segs[0].slice(1);
  const parts = leadUsed ? [...segs.slice(1), segs[0]] : segs;
  if (exact(parts.join(' '))) return cap(parts.join(' '));
  for (const fl of fillers) {
    const text = parts.flatMap((p, i) => (i ? [fl, p] : [p])).join(' ');
    if (exact(text)) return cap(text);
  }
  return null;
}

const bitsOf = (g, s) => [...g.features(s)].map((b, i) => (b ? i : -1)).filter((i) => i >= 0);

// the shortest prompt that contains the required words and still passes the guard.
// Tries, in order: keep the typed phrase intact and read naturally; read naturally;
// anything at all. Returns null when the property is proved instead.
export function shortestBypass(g, required, phrase = '', { maxM = 10, opts = {} } = {}) {
  const lg = new GuardLogic(g);
  const reqSet = new Set(required);
  const phraseBits = phrase ? bitsOf(g, phrase).filter((i) => !reqSet.has(i)) : [];
  // first decide the property over all inputs: that answer is the one that counts
  const full = lg.mustBlock(required);
  const t0 = now();
  const decided = solve(full.n, full.clauses, opts);
  const decideMs = now() - t0;
  if (decided.sat === false) return { proved: true, cnf: full, r: decided, ms: decideMs };
  const tries = [
    { natural: true, req: [...required, ...phraseBits] },
    { natural: true, req: required },
    { natural: false, req: required },
  ];
  for (const t of tries) {
    const s = new Set(t.req);
    const optional = [...Array(g.F).keys()].filter((i) => !s.has(i));
    const res = smallest(() => lg.mustBlock(t.req, { natural: t.natural }), optional, maxM, opts);
    if (!res) continue;
    const x = lg.witness(res.r.model);
    const text = render(g, x, phrase);
    if (!text) continue;
    const check = g.verdict(text);
    return { proved: false, x, text, added: res.m, mode: tries.indexOf(t), natural: t.natural, vote: check.vote, blocked: check.blocked, ms: decideMs + res.ms, calls: res.calls + 1, decided };
  }
  // no short one: fall back to the witness of the decision itself
  const x = lg.witness(decided.model);
  const text = render(g, x, phrase);
  const check = text ? g.verdict(text) : null;
  return { proved: false, x, text, added: null, mode: 3, natural: false, vote: check && check.vote, blocked: check && check.blocked, ms: decideMs, calls: 1, decided };
}
