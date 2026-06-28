"""Regression tests for issues found in the pre-release audit."""

import copy

import numpy as np

from veritaskit import (GuardEncoding, VeritasEngine, Specification, must_flag,
                        verify_assurance, Verifier, REFUTED)


def test_forged_counterexample_is_rejected(tiny_guard):
    """A fabricated REFUTED entry whose 'counterexample' does NOT actually evade
    the guard must be rejected by the prover-independent check (no solver)."""
    g, bad, ok = tiny_guard
    enc = GuardEncoding.from_guard(g)
    # pick a token the guard provably must-flags, so an all-evidence input is UNSAFE
    V = Verifier(enc)
    verified_tok = next((t for t in enc.vocab
                         if V.verify(must_flag([t])).status == "VERIFIED"), enc.vocab[0])
    spec = Specification("s").add(must_flag([verified_tok]))
    KEY = b"k"
    _, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    # forge: claim the (truly robust) obligation was REFUTED with a bogus witness
    forged = copy.deepcopy(ledger)
    fi = enc.token_feature(verified_tok)
    bogus = [0] * enc.n_features
    bogus[fi] = 1                       # token present -> guard says UNSAFE -> NOT a bypass
    for e in forged["entries"]:
        if e["kind"] == "property":
            e["payload"]["status"] = "REFUTED"
            e["payload"]["counterexample"] = bogus
            break
    # re-sign so the chain/signatures are internally consistent (attacker with key,
    # OR the prover itself); the counterexample re-evaluation must still catch it.
    from veritaskit.ledger import canonical, sha256_hex, hmac_hex, GENESIS, AssuranceLedger
    prev = GENESIS
    for e in forged["entries"]:
        body = {"index": e["index"], "kind": e["kind"], "payload": e["payload"], "prev_hash": prev}
        e["prev_hash"] = prev
        e["entry_hash"] = sha256_hex(canonical(body))
        e["signature"] = hmac_hex(KEY, e["entry_hash"].encode())
        prev = e["entry_hash"]
    # rebuild a matching certificate over the re-hashed entries
    from veritaskit.ledger import AssuranceCertificate, merkle_root
    leaves = [e["entry_hash"] for e in forged["entries"]]
    c2 = AssuranceCertificate(cert.spec_id, len(forged["entries"]), prev,
                              merkle_root(leaves), cert.guard_sha256, cert.summary).sign(KEY)
    ok_v, info = verify_assurance(forged, c2, key=KEY, recheck=False)
    assert not ok_v
    assert not info["counterexamples_ok"]   # caught by the prover-free check, not the chain


def test_real_counterexample_passes(tiny_guard):
    """A genuine REFUTED counterexample must still pass the prover-independent check."""
    g, bad, ok = tiny_guard
    enc = GuardEncoding.from_guard(g)
    V = Verifier(enc)
    refuted_tok = next((t for t in enc.vocab
                        if V.verify(must_flag([t])).status == REFUTED), None)
    if refuted_tok is None:
        return
    spec = Specification("s").add(must_flag([refuted_tok]))
    KEY = b"k"
    _, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    ok_v, info = verify_assurance(ledger, cert, key=KEY, recheck=False)
    assert ok_v and info["counterexamples_ok"]


def test_empty_clause_never_fires_consistently():
    """A clause with no literals must contribute 0 to the vote (never fires)."""
    enc = GuardEncoding(n_features=4, clauses=[(1, [])], vocab=["a", "b", "c", "d"])
    for bits in range(16):
        x = np.array([(bits >> i) & 1 for i in range(4)], dtype=np.int8)
        assert enc.vote(x) == 0          # empty positive clause never fires
        assert not enc.unsafe(x)
