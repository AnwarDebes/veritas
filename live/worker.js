// Runs the prover off the main thread so the page stays smooth.
import { createEngine } from './engine.js';
import { parse } from './csv.js';

let engine = null;
// the labelled prompts: next to the page when published, in the repository's data/ folder when served from a clone
async function loadSplit(name) {
  for (const base of ['data/prompt_injection/', '../data/prompt_injection/']) {
    try { const r = await fetch(`${base}${name}.csv`); if (r.ok) return await r.text(); } catch (e) { /* try the next place */ }
  }
  return '';
}
const ready = (async () => {
  const [model, ...csv] = await Promise.all([
    fetch('trace-model.json').then((r) => r.json()),
    ...['train', 'test'].map(loadSplit),
  ]);
  const rows = csv.filter(Boolean).flatMap((t) => parse(t));
  engine = createEngine(model, rows);
  return { rows: rows.length };
})();

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  const emit = (data) => self.postMessage({ id, type: 'event', data });
  try {
    const info = await ready;
    let result;
    if (cmd === 'hello') result = info;
    else if (cmd === 'attack') result = engine.attack(args);
    else if (cmd === 'repair') result = engine.repair(args, emit);
    else if (cmd === 'audit') result = engine.audit(emit, args && args.words);
    else if (cmd === 'falseAlarm') result = engine.falseAlarm(args || {});
    else throw new Error(`unknown command ${cmd}`);
    self.postMessage({ id, type: 'done', data: result });
  } catch (err) {
    self.postMessage({ id, type: 'error', data: String((err && err.message) || err) });
  }
};
