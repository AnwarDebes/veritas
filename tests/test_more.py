import numpy as np

from veritaskit import (GuardEncoding, Verifier, must_flag, never_flag, trigger_necessity,
                        Equivalent, Monotone, VeritasEngine, Specification, verify_assurance,
                        guard_fingerprint, encoding_from_fingerprint, VERIFIED, REFUTED)


def test_guard_fingerprint_roundtrip(tiny_enc):
    enc, bad, ok = tiny_enc
    fp = guard_fingerprint(enc)
    enc2 = encoding_from_fingerprint(fp)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = (rng.random(enc.n_features) < 0.5).astype(np.int8)
        assert enc.vote(x) == enc2.vote(x)


def test_repair_breaks_equivalence_but_satisfies_obligation(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    refuted = [enc.token_feature(t) for t in enc.vocab
               if enc.token_feature(t) is not None
               and V.verify(must_flag([t])).status == REFUTED]
    if not refuted:
        return
    rep = enc.with_must_flag(refuted[:1])
    Vr = Verifier(rep)
    # repaired guard satisfies the obligation...
    assert Vr.verify(must_flag([enc.vocab[refuted[0]]])).status == VERIFIED
    # ...and is provably NOT equivalent to the original (it changed behaviour)
    assert Vr.verify(Equivalent("eq", other=enc), backend="z3").status == REFUTED


def test_pysat_proof_plumbing(tiny_enc):
    # want_proof drives the SAT backend's DRAT proof logging; for a VERIFIED
    # (UNSAT) query the proof-length is recorded (0 for a unit-propagation proof).
    enc, bad, ok = tiny_enc
    V = Verifier(enc, backend="pysat", want_proof=True)
    for t in enc.vocab:
        r = V.verify(must_flag([t]))
        if r.status == VERIFIED:
            assert "drat_proof_lines" in r.extra
            assert r.extra["drat_proof_lines"] >= 0
            return


def test_recheck_uses_recorded_spec(tiny_guard):
    g, bad, ok = tiny_guard
    enc = GuardEncoding.from_guard(g)
    spec = Specification("s")
    spec.add(must_flag([enc.vocab[0]]))
    spec.add(Monotone("m", token=enc.vocab[0], direction="up"))
    KEY = b"k"
    results, ledger, cert = VeritasEngine(g, key=KEY).assure(spec)
    # every property carries the spec metadata needed to re-run it
    for e in ledger["entries"]:
        if e["kind"] == "property":
            assert "spec" in e["payload"]["extra"]
    ok_v, info = verify_assurance(ledger, cert, key=KEY, recheck=True)
    assert ok_v and info["rechecked"] == 2 and info["recheck_ok"]


def test_never_flag_is_safe_invariant(tiny_enc):
    enc, bad, ok = tiny_enc
    nf = never_flag(["weather", "please"])
    assert nf.post == "safe"
    V = Verifier(enc)
    cc = V.cross_check(nf)
    assert cc["_agree"]


def test_monotone_known_bad_token(tiny_enc):
    enc, bad, ok = tiny_enc
    # a token that provably must-flag is monotone-up by construction after repair
    feats = [enc.token_feature(t) for t in enc.vocab if enc.token_feature(t) is not None]
    rep = enc.with_must_flag(feats[:1])
    V = Verifier(rep)
    assert V.verify(Monotone("m", token=enc.vocab[feats[0]], direction="up")).status == VERIFIED


def test_robustness_radius_is_positive_for_clear_unsafe(tiny_enc):
    enc, bad, ok = tiny_enc
    from veritaskit import RobustnessRadius
    # an all-zero input is SAFE; an input with a strong cue may be UNSAFE
    fi = enc.token_feature(bad[0])
    if fi is None:
        return
    anchor = np.zeros(enc.n_features, dtype=np.int8)
    anchor[fi] = 1
    r = Verifier(enc).verify(RobustnessRadius("r", anchor=anchor, max_radius=4))
    assert r.extra.get("radius", 0) >= 1


def test_trigger_necessity_refuted_gives_no_cue_unsafe(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    r = V.verify(trigger_necessity([bad[0]]))   # only one cue audited -> likely refuted
    if r.status == REFUTED:
        x = np.asarray(r.counterexample, dtype=np.int8)
        assert x[enc.token_feature(bad[0])] == 0   # the one audited cue is absent
        assert enc.unsafe(x)                        # yet flagged -> a trigger outside U
