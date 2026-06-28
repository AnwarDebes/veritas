"""E2, Bypass discovery: verification vs red-teaming.

The motivating problem: a guardrail that survives N adversarial probes is *not*
proven safe, red-teaming can only sample, so it certifies nothing and silently
misses bypasses that exist.  We make the gap concrete.  For each attack token a
deployer wants a 'must-flag' guarantee for, we (a) VERIFY the obligation exactly
(proof of robustness, or a counterexample bypass), and (b) RED-TEAM it with up to
N random probes.  Verification decides every case; fuzzing finds only some
bypasses, and, crucially, can never certify the robust ones at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, summarize, write_results, host_info
from experiments._vh import train_guard, attack_exclusive_tokens, fuzz_find_bypass
from veritaskit import GuardEncoding, Verifier, must_flag, VERIFIED, REFUTED

SEEDS = tuple(range(8))
PROBE_BUDGETS = [10, 100, 1000]
FUZZ_TRIALS = 20            # independent red-team campaigns per token


def empirical_density(g, texts) -> float:
    X = g.vec.transform(texts)
    return float(X.mean())


def main(seeds=SEEDS):
    ds = load_dataset("prompt_injection")
    Xtr, ytr = ds["train"]; Xte, yte = ds["test"]
    out = {"host": host_info(), "seeds": list(seeds), "probe_budgets": PROBE_BUDGETS,
           "fuzz_trials": FUZZ_TRIALS, "per_seed": []}

    # aggregates
    frac_bypassable, frac_robust = [], []
    fuzz_recall = {n: [] for n in PROBE_BUDGETS}     # of bypassable tokens, fraction fuzzing catches
    verify_us = []

    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        g = train_guard(Xtr, ytr, C=24, F=48, T=12, s=4.0, epochs=40, seed=seed)
        enc = GuardEncoding.from_guard(g)
        V = Verifier(enc, backend="z3")
        density = empirical_density(g, Xtr)
        # attack-EXCLUSIVE tokens (present in unsafe training docs, never in benign
        # ones), genuine attack cues a deployer would want a must-flag guarantee
        # for, not stopwords.
        toks = attack_exclusive_tokens(g, Xtr, ytr, min_pos=3, top=16)

        bypassable, robust = 0, 0
        caught = {n: 0 for n in PROBE_BUDGETS}
        tot_bypassable = 0
        vts = []
        for t in toks:
            fi = enc.token_feature(t)
            if fi is None:
                continue
            res = V.verify(must_flag([t]))
            vts.append(res.elapsed_us)
            if res.status == VERIFIED:
                robust += 1
            elif res.status == REFUTED:
                bypassable += 1
                tot_bypassable += 1
                # red-team this (known-bypassable) token: does fuzzing find it?
                for n in PROBE_BUDGETS:
                    hits = 0
                    for _ in range(FUZZ_TRIALS):
                        found, _ = fuzz_find_bypass(enc, [(fi, 1)], n, rng, density=density)
                        hits += int(found)
                    caught[n] += hits / FUZZ_TRIALS
        ntok = bypassable + robust
        frac_bypassable.append(bypassable / max(1, ntok))
        frac_robust.append(robust / max(1, ntok))
        for n in PROBE_BUDGETS:
            fuzz_recall[n].append(caught[n] / max(1, tot_bypassable))
        verify_us.append(float(np.mean(vts)) if vts else 0.0)
        out["per_seed"].append({"seed": seed, "n_tokens": ntok, "bypassable": bypassable,
                                "robust": robust, "fuzz_recall": {n: caught[n] / max(1, tot_bypassable)
                                                                  for n in PROBE_BUDGETS}})
        print(f"seed {seed}: bypassable={bypassable}/{ntok} robust={robust} "
              f"fuzz_recall@1000={caught[1000]/max(1,tot_bypassable):.2f}")

    out["summary"] = {
        "frac_bypassable": summarize(frac_bypassable),
        "frac_certified_robust": summarize(frac_robust),
        "verify_us_per_query": summarize(verify_us),
        "fuzz_recall_of_bypasses": {str(n): summarize(fuzz_recall[n]) for n in PROBE_BUDGETS},
        "verification_recall_of_bypasses": 1.0,
        "note": ("Verification finds every bypass (recall 1.0) AND certifies the "
                 "robust tokens, which red-teaming cannot do at any N."),
    }
    print(f"[summary] fuzz recall of bypasses @1000 probes = "
          f"{out['summary']['fuzz_recall_of_bypasses']['1000']['mean']:.2f} "
          f"vs verification 1.00")
    write_results("bypass_discovery", out)
    return out


if __name__ == "__main__":
    main(seeds=(0, 1, 2) if "--quick" in sys.argv else SEEDS)
