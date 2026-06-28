"""E6, Systems characterisation of verification.

How does exact verification scale, and what is verifiable in practice?  We measure
verify time as a function of guard size (clauses x features), compare the two exact
backends (z3 SMT vs pysat SAT), and break the cost down by property type
(no-bypass invariant vs exact robustness radius vs monotonicity).  The honest
message is the *frontier*: compact guards verify in milliseconds; cost grows with
size, and beyond the frontier some queries are left UNKNOWN within budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, write_results, host_info
from experiments._vh import train_guard, attack_lexicon, positive_literal_tokens
from veritaskit import (GuardEncoding, Verifier, must_flag, trigger_necessity,
                        RobustnessRadius, Monotone)


def time_invariants(enc, backend, k=12):
    V = Verifier(enc, backend=backend)
    ts, dec = [], 0
    for t in attack_lexicon(enc, top=k):
        if enc.token_feature(t) is None:
            continue
        r = V.verify(must_flag([t]))
        ts.append(r.elapsed_us); dec += int(r.status in ("VERIFIED", "REFUTED"))
    return float(np.mean(ts)) if ts else 0.0, dec, len(ts)


def main(quick=False):
    ds = load_dataset("prompt_injection")
    Xtr, ytr = ds["train"]; Xte, yte = ds["test"]
    out = {"host": host_info(), "dataset": "prompt_injection"}

    # 1. scaling in clauses (fixed F)
    Cs = [8, 16, 32, 64] if quick else [8, 16, 32, 64, 128]
    out["scale_clauses"] = []
    for C in Cs:
        g = train_guard(Xtr, ytr, C=C, F=48, T=max(6, C // 2), s=4.0, epochs=30, seed=0)
        enc = GuardEncoding.from_guard(g)
        us, dec, n = time_invariants(enc, "z3")
        out["scale_clauses"].append({"clauses": C, "features": 48, "n_clauses_nonempty": enc.n_clauses,
                                     "verify_us": us, "decided": dec, "queries": n})
        print(f"C={C:3d} F=48: {us:.0f}us/query decided={dec}/{n}")

    # 2. scaling in features (fixed C)
    Fs = [16, 32, 64, 96] if quick else [16, 32, 64, 96, 128]
    out["scale_features"] = []
    for F in Fs:
        g = train_guard(Xtr, ytr, C=32, F=F, T=16, s=4.0, epochs=30, seed=0)
        enc = GuardEncoding.from_guard(g)
        us, dec, n = time_invariants(enc, "z3")
        out["scale_features"].append({"clauses": 32, "features": F, "verify_us": us,
                                      "decided": dec, "queries": n})
        print(f"C=32 F={F:3d}: {us:.0f}us/query decided={dec}/{n}")

    # 3. backend comparison (z3 vs pysat) on a mid guard
    g = train_guard(Xtr, ytr, C=32, F=48, T=16, s=4.0, epochs=30, seed=0)
    enc = GuardEncoding.from_guard(g)
    z_us, _, n = time_invariants(enc, "z3")
    p_us, _, _ = time_invariants(enc, "pysat")
    out["backend_compare"] = {"clauses": 32, "features": 48, "queries": n,
                              "z3_us": z_us, "pysat_us": p_us}
    print(f"[backends] z3={z_us:.0f}us pysat={p_us:.0f}us per invariant")

    # 4. per-property-type cost (z3)
    V = Verifier(enc, backend="z3")
    inv_us, rad_us, mon_us = [], [], []
    anchor = None
    for x in Xte:
        fx = g.vec.transform([x])[0]
        if enc.unsafe(fx):
            anchor = fx; break
    for t in attack_lexicon(enc, top=10):
        if enc.token_feature(t) is None:
            continue
        inv_us.append(V.verify(must_flag([t])).elapsed_us)
        mon_us.append(V.verify(Monotone(f"m({t})", token=t, direction="up")).elapsed_us)
    if anchor is not None:
        for _ in range(8):
            rad_us.append(V.verify(RobustnessRadius("rr", anchor=anchor, max_radius=5)).elapsed_us)
    out["property_cost_us"] = {
        "invariant": float(np.mean(inv_us)) if inv_us else 0.0,
        "robustness_radius": float(np.mean(rad_us)) if rad_us else 0.0,
        "monotone": float(np.mean(mon_us)) if mon_us else 0.0,
    }
    print(f"[property cost] invariant={out['property_cost_us']['invariant']:.0f}us "
          f"radius={out['property_cost_us']['robustness_radius']:.0f}us "
          f"monotone={out['property_cost_us']['monotone']:.0f}us")
    write_results("systems", out)
    return out


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
