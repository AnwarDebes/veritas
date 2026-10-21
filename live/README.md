# VERITAS live prover

The VERITAS method running in a web page: a SAT solver checks a claim about a
prompt-injection guard against all 2^256 inputs the guard can see, finds a bypass
when there is one, repairs the guard by construction, and hands back a proof that
an independent checker accepts.

**Open it:** https://anwardebes.github.io/veritas/

## What runs where

| File | Role |
|---|---|
| `guard.js` | The TRACE content guard (200 clauses over 256 word features), inference only |
| `logic.js` | The guard as CNF: one gate per clause, a sequential counter for the vote |
| `cdcl.js` | A conflict-driven clause-learning SAT solver (watched literals, 1UIP learning, VSIDS, restarts) |
| `rup.js` | An independent proof checker: every learned clause is re-derived by unit propagation |
| `explain.js` | Shortest bypasses, turned back into text and re-checked on the guard |
| `repair.js` | Constructive repair: one weighted clause, smallest weight found by binary search |
| `engine.js`, `worker.js` | The commands the page runs, in a Web Worker |
| `app.js`, `app.css`, `index.html` | The page |

The guard model is `trace-model.json`, the same one that runs on
[anwardebes.github.io](https://anwardebes.github.io/), trained with
[TRACE](https://github.com/AnwarDebes/TRACE) on deepset/prompt-injections.
The labelled prompts used for the side-effect check are read from `../data/prompt_injection/`.

## Run it locally

```bash
python3 -m http.server 8000      # from the repository root
# then open http://localhost:8000/live/
```

## Tests

```bash
cd live && npm test
```

The tests fuzz the solver and the proof checker against brute force, check that
the CNF agrees with the guard on concrete inputs, and run the full loop: find a
bypass, confirm it passes the guard, repair, prove, and re-check the exported proof.

## Checking a proof yourself

Every proof on the page can be downloaded as `veritas-formula.cnf` and
`veritas-proof.drup`. Any DRUP checker reads them, for example
[drat-trim](https://github.com/marijnheule/drat-trim):

```bash
drat-trim veritas-formula.cnf veritas-proof.drup   # prints: s VERIFIED
```

## Scope

A proof here is about the guard's word features, not about meaning: a paraphrase
that avoids the words is a different claim. A repair blocks every prompt that
contains all of its words, harmless ones included; the page reports how many
labelled prompts change verdict. This is a JavaScript reimplementation of the
method on the TRACE guard; the experiments in the paper use the Python package
with z3 and PySAT.
