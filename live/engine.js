// Everything the prover page can ask, as plain functions. The page runs them
// in a Web Worker (worker.js); the film runs them directly.
import { Guard } from './guard.js';
import { GuardLogic } from './logic.js';
import { solve, satisfies } from './cdcl.js';
import { checkRUP } from './rup.js';
import { smallest, render, shortestBypass } from './explain.js';
import { minimalRepair, repaired, fixClause } from './repair.js';
import { now } from './timer.js';

// words a person would call "attack words": the audited cue list for the false alarm search
export const CUES = ['ignore', 'forget', 'previous', 'instructions', 'prompt', 'system', 'vergiss', 'anweisungen', 'act', 'now', 'pretend', 'dan', 'jailbreak', 'rules'];

const toIdx = (g, words) => words.map((w) => g.index.get(w)).filter((i) => i !== undefined);
const onBits = (x) => { const out = []; for (let i = 0; i < x.length; i++) if (x[i]) out.push(i); return out; };

// keep at most n frames, evenly spaced, always the last one
function thin(frames, n = 140) {
  if (frames.length <= n) return frames;
  const out = [];
  for (let k = 0; k < n; k++) out.push(frames[Math.round((k * (frames.length - 1)) / (n - 1))]);
  return out;
}
const pack = (frames) => frames.map((f) => ({ kind: f.kind, a: Array.from(f.a) }));

export function dimacs(cnf) {
  const lines = [`p cnf ${cnf.n} ${cnf.clauses.length}`];
  for (const c of cnf.clauses) lines.push(c.join(' ') + ' 0');
  return lines.join('\n') + '\n';
}
export const drup = (learnts) => learnts.map((c) => c.join(' ') + ' 0').join('\n') + (learnts.length ? '\n' : '') + '0\n';

export function createEngine(model, rows = []) {
  const g = new Guard(model);
  const lg = new GuardLogic(g);
  const describe = (gg, c) => ({ id: c.id, unsafe: c.unsafe, weight: c.weight || 1, fix: !!c.fix, ...gg.describe(c) });

  function verdict(text, fix = null) {
    const gg = fix ? repaired(model, fix.required, fix.weight) : g;
    const v = gg.verdict(text);
    return { vote: v.vote, blocked: v.blocked, bits: onBits(v.x), fired: v.fired.map((c) => describe(gg, c)) };
  }

  // the property "every prompt with these words is blocked", decided over all 2^256 inputs
  function attack({ required, phrase }) {
    const t0 = now();
    const res = shortestBypass(g, required, phrase, { opts: { trace: { vars: g.F, max: 3000 } } });
    const out = { ms: now() - t0, required };
    if (res.proved) {
      const t1 = now();
      const chk = checkRUP(res.cnf.n, res.cnf.clauses, res.r.learnts);
      Object.assign(out, {
        proved: true, frames: pack(thin(res.r.frames)), stats: stat(res.r, res.cnf),
        proof: { ok: chk.ok, steps: chk.steps, ms: now() - t1, lemmas: res.r.learnts.length, vars: res.cnf.n, clauses: res.cnf.clauses.length, conflicts: res.r.conflicts },
        cnf: dimacs(res.cnf), drup: drup(res.r.learnts),
      });
      return out;
    }
    const typed = phrase ? new Set(onBits(g.features(phrase))) : new Set();
    const frames = [...(res.decided.frames || [])];
    Object.assign(out, {
      proved: false, text: res.text, vote: res.vote, blocked: res.blocked, added: res.added, mode: res.mode, natural: res.natural,
      bits: onBits(res.x), extra: onBits(res.x).filter((i) => !typed.has(i) && !required.includes(i)).map((i) => g.vocab[i]),
      frames: pack(thin(frames)), stats: stat(res.decided), calls: res.calls,
      fired: res.text ? g.verdict(res.text).fired.map((c) => describe(g, c)) : [],
    });
    return out;
  }

  function stat(r, cnf) {
    return { conflicts: r.conflicts, decisions: r.decisions, props: r.props, learned: r.learned, ...(cnf ? { vars: cnf.n, clauses: cnf.clauses.length } : {}) };
  }

  // a readable bypass for a probe at this weight, if one exists
  function readable(gg, required, phrase) {
    const l = new GuardLogic(gg);
    const cnf = l.mustBlock(required, { natural: true });
    const r = solve(cnf.n, cnf.clauses);
    if (!r.sat) return null;
    const x = l.witness(r.model);
    const text = render(gg, x, phrase);
    return text ? { text, vote: gg.verdict(text).vote, words: text.split(/\s+/).length } : null;
  }

  function repair({ required, phrase }, emit = () => {}) {
    const t0 = now();
    const probes = [];
    const rep = minimalRepair(model, required, {
      opts: { trace: { vars: g.F, max: 1500 } },
      onProbe: (p, range) => {
        const item = { weight: p.weight, sat: !!p.r.sat, ms: p.ms, conflicts: p.r.conflicts, range, frames: pack(thin(p.r.frames, 40)) };
        if (p.r.sat) {
          if (!satisfies(p.r.model, p.cnf.clauses)) throw new Error('solver model failed the check');
          item.vote = p.vote;
          const rd = readable(p.g, required, phrase);
          if (rd) Object.assign(item, rd);
          else { const text = render(p.g, p.x, phrase); item.text = text; item.words = text ? text.split(/\s+/).length : null; }
        }
        probes.push(item);
        emit({ type: 'probe', probe: item });
      },
    });
    const best = rep.best;
    const t1 = now();
    const chk = checkRUP(best.cnf.n, best.cnf.clauses, best.r.learnts);
    const t2 = now();
    // re-run every labelled prompt through both guards
    const fixed = repaired(model, required, rep.weight);
    const changed = [];
    for (const r of rows) {
      const a = g.verdict(r.text).blocked, b = fixed.verdict(r.text).blocked;
      if (a !== b) changed.push({ text: r.text, attack: r.label === '1', before: a, after: b });
    }
    return {
      weight: rep.weight, safe: rep.safe, probes, searchMs: t1 - t0, required, words: required.map((i) => g.vocab[i]),
      clause: describe(fixed, fixClause(required, rep.weight)),
      proof: { ok: chk.ok, steps: chk.steps, ms: t2 - t1, lemmas: best.r.learnts.length, vars: best.cnf.n, clauses: best.cnf.clauses.length, conflicts: best.r.conflicts },
      cnf: dimacs(best.cnf), drup: drup(best.r.learnts),
      side: { rows: rows.length, changed, benign: changed.filter((c) => !c.attack).length, attacks: changed.filter((c) => c.attack).length },
    };
  }

  // which single words force a block, whatever surrounds them?
  function audit(emit = () => {}, words = null) {
    const list = words ? toIdx(g, words) : [...Array(g.F).keys()];
    const proved = [], t0 = now();
    list.forEach((i, k) => {
      const cnf = lg.mustBlock([i]);
      const r = solve(cnf.n, cnf.clauses);
      if (r.sat === false) proved.push(g.vocab[i]);
      if (k % 4 === 3 || k === list.length - 1) emit({ type: 'progress', done: k + 1, total: list.length, word: g.vocab[i], proved: proved.length });
    });
    return { total: list.length, proved, ms: now() - t0 };
  }

  // the shortest prompt with none of the cue words that the guard still blocks
  function falseAlarm({ cues = CUES } = {}) {
    const ci = toIdx(g, cues);
    const t0 = now();
    const res = smallest(() => lg.needsCue(ci, { natural: true }), [...Array(g.F).keys()], 8, { trace: { vars: g.F, max: 2000 } });
    if (!res) {
      const cnf = lg.needsCue(ci);
      const r = solve(cnf.n, cnf.clauses);
      return { proved: r.sat === false, ms: now() - t0, cues: ci.map((i) => g.vocab[i]) };
    }
    const x = lg.witness(res.r.model);
    const text = render(g, x);
    const v = g.verdict(text);
    // what each word contributes: the vote with that word taken out
    const toks = text.split(/\s+/);
    const drops = toks.map((w, k) => {
      const rest = toks.filter((_, j) => j !== k).join(' ');
      const r = g.verdict(rest);
      return { word: w, vote: r.vote, blocked: r.blocked };
    });
    // how often each word shows up in the labelled data, attacks against harmless prompts
    const counts = toks.map((w) => {
      const t = w.toLowerCase();
      let attack = 0, benign = 0;
      for (const r of rows) if (g.terms(r.text).has(t)) { if (r.label === '1') attack++; else benign++; }
      return { word: w, attack, benign };
    });
    return {
      proved: false, text, vote: v.vote, blocked: v.blocked, bits: onBits(x), ms: now() - t0, cues: ci.map((i) => g.vocab[i]),
      fired: v.fired.map((c) => describe(g, c)), frames: pack(thin(res.r.frames)), drops, counts, rows: rows.length,
    };
  }

  return {
    guard: g, vocab: g.vocab, F: g.F, clauses: g.clauses.length, safe: g.clauses.filter((c) => !c.unsafe).length,
    words(text) { return [...new Set((text.toLowerCase().match(/[a-z0-9_]+/g) || []).map((w) => g.index.get(w)).filter((i) => i !== undefined && !g.vocab[i].includes('_')))]; },
    verdict, attack, repair, audit, falseAlarm,
  };
}
