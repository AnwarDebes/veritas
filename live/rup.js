// Re-checks an UNSAT answer. Every learned clause must follow from the clauses
// before it by unit propagation alone (reverse unit propagation), and the last
// one must leave the formula contradictory. If every step passes, no word set
// satisfies the query, whatever the solver did internally.
export function checkRUP(nvars, input, learnts, onProgress = null) {
  const n = nvars;
  const assign = new Int8Array(n + 1);
  const trail = [];
  const watches = Array.from({ length: 2 * n + 2 }, () => []);
  const cls = [];
  const L = (x) => (x > 0 ? 2 * x : -2 * x + 1);
  const val = (l) => { const a = assign[l >> 1]; return (l & 1) ? -a : a; };
  const set = (l) => { assign[l >> 1] = (l & 1) ? -1 : 1; trail.push(l); };

  function propagate(qhead) {
    while (qhead < trail.length) {
      const p = trail[qhead++], fl = p ^ 1;
      const ws = watches[p];
      let i = 0, j = 0;
      while (i < ws.length) {
        const ci = ws[i++], c = cls[ci];
        if (c[0] === fl) { c[0] = c[1]; c[1] = fl; }
        if (val(c[0]) === 1) { ws[j++] = ci; continue; }
        let moved = false;
        for (let k = 2; k < c.length; k++) if (val(c[k]) !== -1) { c[1] = c[k]; c[k] = fl; watches[c[1] ^ 1].push(ci); moved = true; break; }
        if (moved) continue;
        ws[j++] = ci;
        if (val(c[0]) === -1) { while (i < ws.length) ws[j++] = ws[i++]; ws.length = j; return true; }
        if (val(c[0]) === 0) set(c[0]);
      }
      ws.length = j;
    }
    return false;
  }

  // add a clause at level 0; returns true when the formula became contradictory
  function add(c) {
    const s = [...new Set(c.map(L))];
    if (s.some((l) => s.includes(l ^ 1))) return false; // tautology
    s.sort((a, b) => (val(a) === -1) - (val(b) === -1)); // non-false literals first
    const live = s.filter((l) => val(l) !== -1);
    if (live.length === 0) return true;
    if (s.length >= 2) { const ci = cls.length; cls.push(Int32Array.from(s)); watches[s[0] ^ 1].push(ci); watches[s[1] ^ 1].push(ci); }
    if (live.length === 1 && val(s[0]) === 0) { const from = trail.length; set(s[0]); return propagate(from); }
    return false;
  }

  for (const c of input) if (add(c)) return { ok: true, steps: 0 };
  if (propagate(0)) return { ok: true, steps: 0 };

  let steps = 0;
  for (const lc of learnts) {
    const base = trail.length;
    let conflict = false;
    for (const x of lc) {
      const l = L(-x), v = val(l);
      if (v === -1) { conflict = true; break; }
      if (v === 0) set(l);
    }
    if (!conflict) conflict = propagate(base);
    while (trail.length > base) assign[trail.pop() >> 1] = 0;
    if (!conflict) return { ok: false, steps, failed: lc };
    steps++;
    if (add(lc)) return { ok: true, steps };
    if (onProgress && steps % 256 === 0) onProgress(steps);
  }
  return { ok: false, steps, failed: [] };
}
