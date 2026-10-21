// A small CDCL SAT solver: two watched literals, first-UIP clause learning,
// VSIDS with a binary heap, phase saving and Luby restarts. It records every
// learned clause so that an UNSAT answer can be re-checked (see rup.js).
// trace: { vars: F, max } records the first F variables at every decision and
// conflict, so the search can be replayed on screen (it changes nothing else).
export function solve(nvars, input, { maxConflicts = 2e6, onProgress = null, every = 256, trace = null } = {}) {
  const n = nvars;
  const assign = new Int8Array(n + 1), level = new Int32Array(n + 1), reason = new Int32Array(n + 1).fill(-1);
  const phase = new Int8Array(n + 1).fill(-1), act = new Float64Array(n + 1), seen = new Uint8Array(n + 1);
  const trail = new Int32Array(n + 1);
  let tlen = 0, qhead = 0;
  const lim = [];
  const watches = Array.from({ length: 2 * n + 2 }, () => []);
  const cls = [];
  const learnts = []; // in order, as DIMACS literals
  const L = (x) => (x > 0 ? 2 * x : -2 * x + 1);
  const D = (l) => ((l & 1) ? -(l >> 1) : l >> 1);
  const val = (l) => { const a = assign[l >> 1]; return (l & 1) ? -a : a; };
  let inc = 1, conflicts = 0, decisions = 0, props = 0;
  const stats = () => ({ conflicts, decisions, props, learned: learnts.length });

  const heap = [], pos = new Int32Array(n + 1).fill(-1);
  const less = (a, b) => act[a] > act[b];
  function up(i) { const v = heap[i]; while (i > 0) { const p = (i - 1) >> 1; if (!less(v, heap[p])) break; heap[i] = heap[p]; pos[heap[i]] = i; i = p; } heap[i] = v; pos[v] = i; }
  function down(i) { const v = heap[i], m = heap.length; for (;;) { let c = 2 * i + 1; if (c >= m) break; if (c + 1 < m && less(heap[c + 1], heap[c])) c++; if (!less(heap[c], v)) break; heap[i] = heap[c]; pos[heap[i]] = i; i = c; } heap[i] = v; pos[v] = i; }
  function hpush(v) { if (pos[v] >= 0) return; heap.push(v); pos[v] = heap.length - 1; up(heap.length - 1); }
  function hpop() { const v = heap[0], last = heap.pop(); pos[v] = -1; if (heap.length) { heap[0] = last; pos[last] = 0; down(0); } return v; }
  function bump(v) { act[v] += inc; if (act[v] > 1e100) { for (let i = 1; i <= n; i++) act[i] *= 1e-100; inc *= 1e-100; } if (pos[v] >= 0) up(pos[v]); }
  function enqueue(l, r) { const v = l >> 1; assign[v] = (l & 1) ? -1 : 1; level[v] = lim.length; reason[v] = r; trail[tlen++] = l; }
  function attach(lits) { const ci = cls.length; cls.push(lits); watches[lits[0] ^ 1].push(ci); watches[lits[1] ^ 1].push(ci); return ci; }
  for (let v = 1; v <= n; v++) hpush(v);

  const frames = [];
  const snap = (kind) => { if (trace && frames.length < (trace.max || 4000)) frames.push({ kind, a: assign.slice(1, trace.vars + 1) }); };
  const done = (sat, extra = {}) => ({ sat, ...stats(), learnts, frames, ...extra });
  for (const c of input) {
    const s = [...new Set(c.map(L))];
    if (s.some((l) => s.includes(l ^ 1))) continue; // tautology
    if (s.length === 0) return done(false);
    if (s.length === 1) {
      const v = val(s[0]);
      if (v === -1) return done(false);
      if (v === 0) enqueue(s[0], -1);
      continue;
    }
    attach(Int32Array.from(s));
  }

  function propagate() {
    while (qhead < tlen) {
      const p = trail[qhead++], fl = p ^ 1;
      const ws = watches[p];
      let i = 0, j = 0;
      while (i < ws.length) {
        const ci = ws[i++], c = cls[ci];
        props++;
        if (c[0] === fl) { c[0] = c[1]; c[1] = fl; }
        if (val(c[0]) === 1) { ws[j++] = ci; continue; }
        let moved = false;
        for (let k = 2; k < c.length; k++) if (val(c[k]) !== -1) { c[1] = c[k]; c[k] = fl; watches[c[1] ^ 1].push(ci); moved = true; break; }
        if (moved) continue;
        ws[j++] = ci;
        if (val(c[0]) === -1) { while (i < ws.length) ws[j++] = ws[i++]; ws.length = j; return ci; }
        enqueue(c[0], ci);
      }
      ws.length = j;
    }
    return -1;
  }

  function analyze(confl) {
    const learnt = [0];
    let pathC = 0, p = -1, idx = tlen - 1;
    const dl = lim.length;
    do {
      const c = cls[confl];
      for (let k = p === -1 ? 0 : 1; k < c.length; k++) {
        const q = c[k], v = q >> 1;
        if (!seen[v] && level[v] > 0) { seen[v] = 1; bump(v); if (level[v] >= dl) pathC++; else learnt.push(q); }
      }
      while (!seen[trail[idx] >> 1]) idx--;
      p = trail[idx]; idx--; confl = reason[p >> 1]; seen[p >> 1] = 0; pathC--;
    } while (pathC > 0);
    learnt[0] = p ^ 1;
    let bt = 0, bi = 1;
    for (let k = 1; k < learnt.length; k++) { const lv = level[learnt[k] >> 1]; if (lv > bt) { bt = lv; bi = k; } }
    if (learnt.length > 1) { const t = learnt[1]; learnt[1] = learnt[bi]; learnt[bi] = t; }
    for (const q of learnt) seen[q >> 1] = 0;
    return { learnt, bt };
  }

  function cancelUntil(lv) {
    if (lim.length <= lv) return;
    for (let k = tlen - 1; k >= lim[lv]; k--) { const v = trail[k] >> 1; phase[v] = assign[v]; assign[v] = 0; reason[v] = -1; hpush(v); }
    tlen = lim[lv]; qhead = tlen; lim.length = lv;
  }

  const luby = (i) => { let k = 1; while ((1 << k) - 1 < i) k++; while ((1 << (k - 1)) !== i - (1 << (k - 1)) + 1 && i !== (1 << k) - 1) { i = i - (1 << (k - 1)) + 1; k = 1; while ((1 << k) - 1 < i) k++; } return 1 << (k - 1); };
  let restarts = 1, budget = 100 * luby(1), since = 0;
  if (propagate() !== -1) return done(false);
  for (;;) {
    const confl = propagate();
    if (confl !== -1) {
      conflicts++; since++;
      if (trace) snap('conflict');
      if (lim.length === 0) return done(false);
      if (conflicts > maxConflicts) return done(null);
      const { learnt, bt } = analyze(confl);
      learnts.push(learnt.map(D));
      cancelUntil(bt);
      if (learnt.length === 1) enqueue(learnt[0], -1);
      else { const ci = attach(Int32Array.from(learnt)); enqueue(learnt[0], ci); }
      inc *= 1 / 0.95;
      if (onProgress && conflicts % every === 0) onProgress(stats());
    } else {
      if (since >= budget) { cancelUntil(0); since = 0; restarts++; budget = 100 * luby(restarts); }
      let v = 0;
      while (heap.length) { const c = hpop(); if (assign[c] === 0) { v = c; break; } }
      if (!v) {
        const model = new Int8Array(n + 1);
        for (let i = 1; i <= n; i++) model[i] = assign[i];
        if (trace) snap('model');
        return done(true, { model });
      }
      decisions++;
      lim.push(tlen);
      enqueue(2 * v + (phase[v] === 1 ? 0 : 1), -1);
      if (trace) snap('decide');
    }
  }
}

// does `model` satisfy every clause? (checks a SAT answer without trusting the solver)
export function satisfies(model, clauses) {
  return clauses.every((c) => c.some((x) => model[Math.abs(x)] === (x > 0 ? 1 : -1)));
}
