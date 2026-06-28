"""E1, Verification works, and the interpretability/verifiability frontier.

Claim under test: an interpretable Tsetlin guard can be verified *exactly* over
its whole input space in practical time, and there is a smooth frontier trading
model size (clauses x features) against verification cost and accuracy.  We also
establish *soundness of our own verifier* by cross-checking three independent
backends (z3 SMT, pysat SAT, brute-force enumeration) and confirming they never
disagree on a decided query.

The TM->Boolean encoding and the robustness/equivalence queries are due to
Przybysz et al. (2023); this experiment exercises them at guardrail scale and adds
the new property classes (no-bypass invariants, trigger-necessity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, summarize, write_results, host_info
from experiments._vh import train_guard, attack_exclusive_tokens, positive_literal_tokens
from veritaskit import (GuardEncoding, Verifier, must_flag, trigger_necessity,
                        RobustnessRadius, VERIFIED, REFUTED)

SEEDS = tuple(range(8))
BUDGETS = [(8, 24), (16, 40), (32, 64), (64, 96)]    # (clauses, features)


def run_budget(Xtr, ytr, Xte, yte, C, F, seeds):
    accs, vtimes, nver, nref, ndec, nq = [], [], [], [], [], []
    for seed in seeds:
        g = train_guard(Xtr, ytr, C=C, F=F, T=max(6, C // 2), s=4.0, epochs=40, seed=seed)
        accs.append(g.score(Xte, yte))
        enc = GuardEncoding.from_guard(g)
        V = Verifier(enc, backend="z3")
        # genuine attack-exclusive obligations + one trigger-necessity check
        props = [must_flag([t]) for t in attack_exclusive_tokens(g, Xtr, ytr, min_pos=3, top=12)]
        props.append(trigger_necessity(positive_literal_tokens(enc)))
        v = r = d = 0
        ts = []
        for p in props:
            res = V.verify(p)
            ts.append(res.elapsed_us)
            if res.status == VERIFIED:
                v += 1; d += 1
            elif res.status == REFUTED:
                r += 1; d += 1
        nver.append(v); nref.append(r); ndec.append(d); nq.append(len(props))
        vtimes.append(float(np.mean(ts)))
    return {
        "clauses": C, "features": F,
        "accuracy": summarize(accs),
        "verify_us_per_query": summarize(vtimes),
        "n_verified": summarize(nver), "n_refuted": summarize(nref),
        "n_decided": summarize(ndec), "n_queries": summarize(nq),
    }


def main(seeds=SEEDS):
    ds = load_dataset("prompt_injection")
    Xtr, ytr = ds["train"]; Xte, yte = ds["test"]
    out = {"host": host_info(), "seeds": list(seeds), "dataset": "prompt_injection",
           "n_train": len(Xtr), "n_test": len(Xte), "budgets": []}
    for (C, F) in BUDGETS:
        r = run_budget(Xtr, ytr, Xte, yte, C, F, seeds)
        out["budgets"].append(r)
        print(f"C={C:3d} F={F:3d} acc={r['accuracy']['mean']:.3f} "
              f"verify={r['verify_us_per_query']['mean']:.0f}us/q "
              f"verified={r['n_verified']['mean']:.1f} refuted={r['n_refuted']['mean']:.1f}")
    # also a small fully-cross-checked guard to certify the verifier itself
    g = train_guard(Xtr, ytr, C=8, F=16, T=5, s=4.0, epochs=40, seed=0)
    enc = GuardEncoding.from_guard(g)
    V = Verifier(enc)
    checks = 0; ok = 0
    for t in enc.vocab:
        cc = V.cross_check(must_flag([t]))
        checks += 1; ok += int(cc["_agree"])
    cc = V.cross_check(trigger_necessity(positive_literal_tokens(enc)))
    checks += 1; ok += int(cc["_agree"])
    out["verifier_soundness"] = {"F": enc.n_features, "cross_checks": checks,
                                 "agree": ok, "agreement_rate": ok / max(1, checks)}
    print(f"[verifier soundness] z3/pysat/brute agree on {ok}/{checks} queries")
    write_results("verification_frontier", out)
    return out


if __name__ == "__main__":
    main(seeds=(0, 1, 2) if "--quick" in sys.argv else SEEDS)
