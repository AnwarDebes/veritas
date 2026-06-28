"""E5, The assurance certificate: tamper-evident and prover-independent.

VERITAS seals the verification record (which properties were checked, proven, or
refuted, with counterexamples) into a hash-chained, HMAC-signed certificate.  We
check two things.  (1) Tamper-evidence: six classes of post-hoc edit plus a
signature forgery are all detected.  (2) Prover-independence: every REFUTED entry
carries a counterexample that an auditor re-evaluates on the *included* guard with
pure arithmetic, no solver, and an optional full re-run of the verifier
reproduces every recorded verdict.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, summarize, write_results, host_info
from experiments._vh import train_guard, attack_lexicon, positive_literal_tokens
from veritaskit import (GuardEncoding, VeritasEngine, Specification, must_flag,
                        trigger_necessity, Monotone, verify_assurance, encoding_from_fingerprint)
from veritaskit.encode import GuardEncoding as GE

KEY = b"veritas-assurance-key-E5"
TAMPERS = ["modify_status", "modify_counterexample", "modify_guard_clause",
           "delete_entry", "insert_entry", "reorder_entries"]


def build_spec(enc) -> Specification:
    spec = Specification("assurance-spec")
    for t in attack_lexicon(enc, top=10):
        if enc.token_feature(t) is not None:
            spec.add(must_flag([t]))
    spec.add(trigger_necessity(positive_literal_tokens(enc)))
    for t in attack_lexicon(enc, top=3):
        if enc.token_feature(t) is not None:
            spec.add(Monotone(f"monotone({t})", token=t, direction="up"))
    return spec


def apply_tamper(ledger, kind, rng):
    led = copy.deepcopy(ledger)
    E = led["entries"]
    props = [i for i, e in enumerate(E) if e["kind"] == "property"]
    guards = [i for i, e in enumerate(E) if e["kind"] == "guard"]
    if kind == "modify_status" and props:
        i = int(rng.choice(props))
        E[i]["payload"]["status"] = "VERIFIED" if E[i]["payload"]["status"] != "VERIFIED" else "REFUTED"
    elif kind == "modify_counterexample":
        cands = [i for i in props if E[i]["payload"].get("counterexample")]
        if cands:
            i = int(rng.choice(cands))
            cx = E[i]["payload"]["counterexample"]; cx[0] ^= 1
    elif kind == "modify_guard_clause" and guards:
        i = guards[0]
        cls = E[i]["payload"]["body"]["clauses"]
        if cls:
            cls[0][0] = -cls[0][0]                       # flip a clause polarity
    elif kind == "delete_entry" and len(E) > 2:
        del E[int(rng.integers(1, len(E)))]
    elif kind == "insert_entry":
        i = int(rng.integers(1, len(E)))
        E.insert(i, copy.deepcopy(E[i]))
    elif kind == "reorder_entries" and len(E) > 3:
        i, j = sorted(rng.choice(range(1, len(E)), size=2, replace=False))
        E[i], E[j] = E[j], E[i]
    return led


def main(n_sessions=24, seeds=tuple(range(6))):
    ds = load_dataset("prompt_injection")
    Xtr, ytr = ds["train"]
    out = {"host": host_info(), "n_sessions": n_sessions, "tampers": {},
           "security_property": ("tamper-evidence (SHA-256 hash chain + Merkle root) + "
                                 "symmetric authentication (HMAC-SHA-256); prover-independent "
                                 "counterexample re-evaluation")}
    rng = np.random.default_rng(7)

    # build a pool of sealed certificates
    pool = []
    clean_ok = 0; recheck_ok = 0; cx_indep = 0; cx_total = 0
    for s in range(n_sessions):
        g = train_guard(Xtr, ytr, C=14, F=28, T=7, s=4.0, epochs=30, seed=s)
        enc = GuardEncoding.from_guard(g)
        eng = VeritasEngine(g, key=KEY)
        results, ledger, cert = eng.assure(build_spec(enc))
        ok, info = verify_assurance(ledger, cert, key=KEY, recheck=True)
        clean_ok += int(ok); recheck_ok += int(info["recheck_ok"])
        # prover-independent re-check of every counterexample on the included guard
        gfp = next(e for e in ledger["entries"] if e["kind"] == "guard")
        enc2 = encoding_from_fingerprint(gfp["payload"])
        from veritaskit.ledger import _counterexample_is_real
        for e in ledger["entries"]:
            if e["kind"] == "property" and e["payload"].get("counterexample") is not None:
                cx_total += 1
                # genuinely confirm the witness violates its property on the included
                # guard, with no solver (not just that it can be evaluated).
                cx_indep += int(_counterexample_is_real(enc2, e["payload"], e["payload"]["counterexample"]))
        pool.append((ledger, cert))

    out["clean_verify_rate"] = clean_ok / n_sessions
    out["recheck_pass_rate"] = recheck_ok / n_sessions
    out["counterexamples_reevaluated"] = cx_total
    out["counterexample_independent_rate"] = cx_indep / max(1, cx_total)

    for kind in TAMPERS:
        detected = 0; trials = 0
        for (ledger, cert) in pool:
            t = apply_tamper(ledger, kind, rng)
            if t == ledger:
                continue
            trials += 1
            ok, info = verify_assurance(t, cert, key=KEY, recheck=False)
            detected += int(not ok)
        out["tampers"][kind] = {"trials": trials, "detection_rate": detected / max(1, trials)}
        print(f"{kind:22s} detected {detected}/{trials}")

    # forged signature: tamper a payload, recompute a valid hash, but cannot re-sign
    from veritaskit.ledger import canonical, sha256_hex, GENESIS
    forged_detected = 0; forged_by_sig = 0
    for (ledger, cert) in pool:
        led = copy.deepcopy(ledger)
        i = int(rng.integers(1, len(led["entries"])))
        e = led["entries"][i]
        e["payload"]["status"] = "VERIFIED"
        prev = led["entries"][i - 1]["entry_hash"] if i > 0 else GENESIS
        body = {"index": e["index"], "kind": e["kind"], "payload": e["payload"], "prev_hash": prev}
        e["entry_hash"] = sha256_hex(canonical(body))
        e["signature"] = "0" * 64
        ok, info = verify_assurance(led, cert, key=KEY, recheck=False)
        forged_detected += int(not ok); forged_by_sig += int(not ok and not info["sig_ok"])
    out["forged_signature_detection_rate"] = forged_detected / len(pool)
    out["forged_signature_caught_by_sig_rate"] = forged_by_sig / len(pool)
    print(f"forged-signature detected {forged_detected}/{len(pool)} (sig-isolated {forged_by_sig}/{len(pool)})")
    print(f"clean verify {out['clean_verify_rate']:.2f}; recheck {out['recheck_pass_rate']:.2f}; "
          f"counterexamples re-evaluable {out['counterexample_independent_rate']:.2f}")
    write_results("assurance_cert", out)
    return out


if __name__ == "__main__":
    main(n_sessions=12 if "--quick" in sys.argv else 24)
