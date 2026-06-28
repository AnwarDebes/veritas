"""E3, Closing bypasses: you cannot *train* your way to a proof.

A deployer wants a 'must-flag' guarantee for a set of attack-exclusive tokens.
We compare two ways to get there.  (1) EMPIRICAL hardening: feed the solver's
counterexamples back as labelled data and retrain (a counterexample-guided loop;
Solar-Lezama 2006; Clarke et al. 2000).  Round over round it fails to reach, and
can never certify, zero provable bypasses, and it can even *lose* a guarantee it
had, because stochastic training only softly biases the decision.  (2)
CONSTRUCTIVE verification-guided repair: because the guard is a propositional
formula, we append the minimal set of single-literal clauses that *provably*
enforces each obligation; re-verification confirms zero bypasses by construction,
at a small, bounded accuracy cost.  The lesson: verification must supply a
*constructive* repair operator, not just grade the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, summarize, write_results, host_info
from experiments._vh import train_guard, attack_exclusive_tokens
from veritaskit import GuardEncoding, Verifier, must_flag, harden, REFUTED

SEEDS = tuple(range(8))


def count_bypasses(enc, invs):
    V = Verifier(enc, backend="z3")
    return sum(int(V.verify(inv).status == REFUTED) for inv in invs)


def main(seeds=SEEDS):
    ds = load_dataset("prompt_injection")
    Xtr, ytr = ds["train"]; Xte, yte = ds["test"]
    out = {"host": host_info(), "seeds": list(seeds), "per_seed": []}
    agg = {k: [] for k in ("n_targets", "bypass_initial", "bypass_empirical",
                           "bypass_constructive", "acc_initial", "acc_empirical",
                           "acc_constructive", "clauses_added", "empirical_lost_guarantee")}

    for seed in seeds:
        g = train_guard(Xtr, ytr, C=24, F=40, T=12, s=4.0, epochs=40, seed=seed)
        Xtr_f = g.vec.transform(Xtr); Xte_f = g.vec.transform(Xte)
        enc = GuardEncoding.from_guard(g)
        toks = attack_exclusive_tokens(g, Xtr, ytr, min_pos=3, top=8)
        feats = [enc.token_feature(t) for t in toks if enc.token_feature(t) is not None]
        toks = [enc.vocab[f] for f in feats]
        invs = [must_flag([t]) for t in toks]
        if not invs:
            continue
        V = Verifier(enc)
        init_status = [V.verify(inv).status for inv in invs]
        b0 = sum(int(s == REFUTED) for s in init_status)
        a0 = float((enc.predict(Xte_f) == np.asarray(yte)).mean())

        # (1) empirical hardening
        hr = harden(g, Xtr_f, ytr, invs, rounds=8, neighbors=8, backend="z3",
                    X_test=Xte_f, y_test=yte, rng=np.random.default_rng(seed))
        enc_e = GuardEncoding.from_guard(hr.guard)
        Ve = Verifier(enc_e)
        emp_status = [Ve.verify(inv).status for inv in invs]
        be = sum(int(s == REFUTED) for s in emp_status)
        ae = float((enc_e.predict(Xte_f) == np.asarray(yte)).mean())
        lost = sum(int(i == "VERIFIED" and e == REFUTED) for i, e in zip(init_status, emp_status))

        # (2) constructive repair
        enc_c = enc.with_must_flag(feats)
        Vc = Verifier(enc_c)
        bc = sum(int(Vc.verify(inv).status == REFUTED) for inv in invs)
        ac = float((enc_c.predict(Xte_f) == np.asarray(yte)).mean())

        agg["n_targets"].append(len(invs))
        agg["bypass_initial"].append(b0); agg["bypass_empirical"].append(be)
        agg["bypass_constructive"].append(bc)
        agg["acc_initial"].append(a0); agg["acc_empirical"].append(ae); agg["acc_constructive"].append(ac)
        agg["clauses_added"].append(enc_c.n_clauses - enc.n_clauses)
        agg["empirical_lost_guarantee"].append(lost)
        out["per_seed"].append({"seed": seed, "n_targets": len(invs), "bypass_initial": b0,
                                "bypass_empirical": be, "bypass_constructive": bc,
                                "acc_initial": a0, "acc_empirical": ae, "acc_constructive": ac,
                                "empirical_history": hr.history, "lost_guarantee": lost})
        print(f"seed {seed}: targets={len(invs)} bypass {b0}->emp {be} / construct {bc} | "
              f"acc {a0:.3f}->emp {ae:.3f} / construct {ac:.3f} | empirical lost {lost} guarantees")

    out["summary"] = {k: summarize(agg[k]) for k in agg}
    s = out["summary"]
    print(f"[summary] bypasses: initial {s['bypass_initial']['mean']:.1f} -> "
          f"empirical {s['bypass_empirical']['mean']:.1f} -> constructive {s['bypass_constructive']['mean']:.1f} (=0); "
          f"acc {s['acc_initial']['mean']:.3f}->{s['acc_constructive']['mean']:.3f} constructive")
    write_results("hardening", out)
    return out


if __name__ == "__main__":
    main(seeds=(0, 1, 2) if "--quick" in sys.argv else SEEDS)
