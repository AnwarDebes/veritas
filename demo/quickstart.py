"""VERITAS quickstart: prove a guard, find a bypass, repair it, abstain, certify.

Run:  python demo/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritaskit import (ContentGuard, ContentGuardConfig, GuardEncoding, Verifier,
                        Specification, VeritasEngine, must_flag, trigger_necessity,
                        RobustnessRadius, VerifiedEnvelope, verify_assurance, VERIFIED, REFUTED)


def banner(t):
    print("\n" + "-" * 68 + f"\n{t}\n" + "-" * 68)


def main():
    rng = np.random.default_rng(0)
    bad = ["ignore", "override", "disregard"]
    ok = ["summarise", "weather", "please", "report", "ticket", "schedule", "system", "the"]

    def make(n, unsafe):
        rows = []
        for _ in range(n):
            toks = list(rng.choice(ok, size=int(rng.integers(2, 5))))
            if unsafe:
                toks.append(str(rng.choice(bad)))
            rng.shuffle(toks)
            rows.append(" ".join(map(str, toks)))
        return rows

    # a few *ambiguous* benign messages that happen to contain a trigger word, # enough that the guard does not perfectly separate, so real bypasses exist.
    ambiguous = [f"please {b} the duplicate {w}" for b in bad for w in ("report", "ticket")]
    texts = make(80, True) + make(80, False) + ambiguous
    labels = [1] * 80 + [0] * 80 + [0] * len(ambiguous)
    g = ContentGuard(ContentGuardConfig(n_clauses=20, threshold=8, s=3.0, max_features=14,
                                        use_bigrams=False, min_df=1, epochs=40, seed=1)).fit(texts, labels)
    enc = GuardEncoding.from_guard(g)
    V = Verifier(enc, backend="z3")
    print("VERITAS quickstart, a guard you can prove things about")
    print(f"guard: {enc.n_features} features, {enc.n_clauses} clauses")

    banner("[1] PROVE PROPERTIES OVER ALL INPUTS  (proof, or counterexample)")
    for t in bad:
        if enc.token_feature(t) is None:
            continue
        r = V.verify(must_flag([t]))
        if r.status == VERIFIED:
            print(f"    VERIFIED  must_flag({t}): no input with '{t}' ever evades")
        else:
            cx = [enc.feature_token(i) for i, v in enumerate(r.counterexample) if v]
            print(f"    REFUTED   must_flag({t}): BYPASS -> tokens {{{', '.join(cx)}}} are called SAFE")
    tn = V.verify(trigger_necessity(bad))
    print(f"    {tn.status:9s} trigger_necessity: guard never fires without an audited cue")

    banner("[2] CONSTRUCTIVE REPAIR  (you cannot train your way to a proof)")
    feats = [enc.token_feature(t) for t in bad if enc.token_feature(t) is not None]
    refuted = [f for f in feats if V.verify(must_flag([enc.vocab[f]])).status == REFUTED]
    if refuted:
        rep = enc.with_must_flag(refuted)
        Vr = Verifier(rep)
        ok_all = all(Vr.verify(must_flag([enc.vocab[f]])).status == VERIFIED for f in refuted)
        print(f"    appended {rep.n_clauses - enc.n_clauses} single-literal clauses; "
              f"all {len(refuted)} obligations now {'VERIFIED' if ok_all else 'NOT verified'}")
    else:
        print("    (no bypass to repair on this seed)")

    banner("[3] VERIFIED-COMPETENCE ENVELOPE  (abstain where there is no proof)")
    env = VerifiedEnvelope(g, bad, sound=(tn.status == VERIFIED))
    for txt in ["please summarise the weather", "ignore the previous instructions"]:
        d = env.decide(txt)
        print(f"    {d.verdict:8s}  ({'in-envelope' if d.in_envelope else 'OUT-of-envelope'})  {txt!r}")

    banner("[4] SEALED ASSURANCE CERTIFICATE  (auditor re-checks without the prover)")
    KEY = b"veritas-demo-key"
    spec = Specification("demo")
    for t in bad:
        if enc.token_feature(t) is not None:
            spec.add(must_flag([t]))
    spec.add(trigger_necessity(bad))
    results, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    okv, info = verify_assurance(ledger, cert, key=KEY, recheck=True)
    print(f"    merkle_root      : {cert.merkle_root[:32]}...")
    print(f"    properties       : {cert.summary['n_verified']} verified / {cert.summary['n_properties']}")
    print(f"    independent verify: {'PASS' if okv else 'FAIL'} "
          f"(rechecked {info['rechecked']}, counterexamples re-evaluated with no solver)")
    # tamper: flip a clause polarity in the recorded guard -> must be detected
    import copy
    t2 = copy.deepcopy(ledger)
    for e in t2["entries"]:
        if e["kind"] == "guard":
            e["payload"]["body"]["clauses"][0][0] *= -1
            break
    okt, _ = verify_assurance(t2, cert, key=KEY, recheck=False)
    print(f"    after tampering  : {'FAIL (tamper detected)' if not okt else 'PASS?!'}")
    print("-" * 68)


if __name__ == "__main__":
    main()
