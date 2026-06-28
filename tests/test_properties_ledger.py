import copy

import numpy as np

from veritaskit import (Verifier, must_flag, trigger_necessity, never_flag, Invariant,
                        VeritasEngine, Specification, verify_assurance, VERIFIED,
                        AssuranceCertificate)


def test_must_flag_and_trigger_necessity_are_invariants():
    mf = must_flag(["a", "b"])
    tn = trigger_necessity(["a", "b"])
    assert isinstance(mf, Invariant) and mf.post == "unsafe"
    assert isinstance(tn, Invariant) and tn.post == "safe"
    assert mf.pre == {"a": True, "b": True}
    assert tn.pre == {"a": False, "b": False}


def test_engine_assure_and_independent_verify(tiny_guard):
    g, bad, ok = tiny_guard
    from veritaskit import GuardEncoding
    enc = GuardEncoding.from_guard(g)
    spec = Specification("t")
    for t in enc.vocab[:4]:
        spec.add(must_flag([t]))
    spec.add(trigger_necessity([t for t in bad if enc.token_feature(t) is not None]))
    KEY = b"k"
    eng = VeritasEngine(g, key=KEY)
    results, ledger, cert = eng.assure(spec)
    assert len(results) == len(spec.properties)
    ok_v, info = verify_assurance(ledger, cert, key=KEY, recheck=True)
    assert ok_v and info["recheck_ok"] and info["counterexamples_ok"]
    assert info["rechecked"] >= 1


def test_assurance_detects_status_tamper(tiny_guard):
    g, bad, ok = tiny_guard
    from veritaskit import GuardEncoding
    enc = GuardEncoding.from_guard(g)
    spec = Specification("t")
    for t in enc.vocab[:4]:
        spec.add(must_flag([t]))
    KEY = b"k"
    results, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    tampered = copy.deepcopy(ledger)
    for e in tampered["entries"]:
        if e["kind"] == "property":
            e["payload"]["status"] = "VERIFIED"
            break
    ok_v, _ = verify_assurance(tampered, cert, key=KEY, recheck=False)
    assert not ok_v


def test_assurance_detects_guard_clause_tamper(tiny_guard):
    g, bad, ok = tiny_guard
    from veritaskit import GuardEncoding
    enc = GuardEncoding.from_guard(g)
    spec = Specification("t").add(must_flag([enc.vocab[0]]))
    KEY = b"k"
    results, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    tampered = copy.deepcopy(ledger)
    for e in tampered["entries"]:
        if e["kind"] == "guard":
            e["payload"]["body"]["clauses"][0][0] *= -1     # flip a polarity
            break
    ok_v, _ = verify_assurance(tampered, cert, key=KEY, recheck=False)
    assert not ok_v


def test_clean_certificate_verifies(tiny_guard):
    g, bad, ok = tiny_guard
    from veritaskit import GuardEncoding
    enc = GuardEncoding.from_guard(g)
    spec = Specification("t").add(must_flag([enc.vocab[0]]))
    KEY = b"k"
    results, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    ok_v, info = verify_assurance(ledger, cert, key=KEY)
    assert ok_v and info["ok"]


def test_certificate_dict_roundtrip(tiny_guard):
    g, bad, ok = tiny_guard
    from veritaskit import GuardEncoding
    enc = GuardEncoding.from_guard(g)
    spec = Specification("t").add(must_flag([enc.vocab[0]]))
    _, _, cert = VeritasEngine(g, key=b"k").assure(spec)
    again = AssuranceCertificate.from_dict(cert.to_dict())
    assert again.to_dict() == cert.to_dict()
