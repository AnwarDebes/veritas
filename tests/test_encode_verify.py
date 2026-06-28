import numpy as np

from veritaskit import (GuardEncoding, Verifier, must_flag, trigger_necessity,
                        RobustnessRadius, Monotone, Equivalent, VERIFIED, REFUTED)
from veritaskit.tm import to_literals


def test_encoder_is_bit_exact(tiny_enc, tiny_guard):
    enc, bad, ok = tiny_enc
    g, _, _ = tiny_guard
    rng = np.random.default_rng(3)
    for _ in range(500):
        x = (rng.random(enc.n_features) < 0.5).astype(np.int8)
        v_enc = enc.vote(x)
        v_tm, _ = g.tm.vote_sum(to_literals(x.reshape(1, -1))[0], predict=True)
        assert v_enc == int(v_tm)


def test_invariant_backends_agree(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    for t in enc.vocab:
        cc = V.cross_check(must_flag([t]))
        assert cc["_agree"], (t, {k: v.status for k, v in cc.items() if k != "_agree"})


def test_trigger_necessity_backends_agree(tiny_enc):
    enc, bad, ok = tiny_enc
    audited = [t for t in bad if enc.token_feature(t) is not None]
    V = Verifier(enc)
    cc = V.cross_check(trigger_necessity(audited))
    assert cc["_agree"]


def test_robustness_radius_z3_matches_brute(tiny_enc):
    enc, bad, ok = tiny_enc
    anchor = np.zeros(enc.n_features, dtype=np.int8)
    fi = enc.token_feature(bad[0])
    if fi is not None:
        anchor[fi] = 1
    V = Verifier(enc)
    rz = V.verify(RobustnessRadius("r", anchor=anchor, max_radius=4), backend="z3")
    rb = V.verify(RobustnessRadius("r", anchor=anchor, max_radius=4), backend="brute")
    assert rz.extra.get("radius") == rb.extra.get("radius")


def test_monotone_z3_matches_brute(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    for t in enc.vocab[:4]:
        mz = V.verify(Monotone("m", token=t, direction="up"), backend="z3")
        mb = V.verify(Monotone("m", token=t, direction="up"), backend="brute")
        assert mz.status == mb.status


def test_equivalence_self_is_verified(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    r = V.verify(Equivalent("self", other=enc), backend="z3")
    assert r.status == VERIFIED


def test_counterexample_is_self_checking(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    # find a refuted must_flag and confirm its counterexample really evades
    for t in enc.vocab:
        r = V.verify(must_flag([t]))
        if r.status == REFUTED:
            x = np.asarray(r.counterexample, dtype=np.int8)
            assert x[enc.token_feature(t)] == 1   # token present
            assert not enc.unsafe(x)               # yet SAFE -> a real bypass
            return


def test_with_must_flag_repairs_provably(tiny_enc):
    enc, bad, ok = tiny_enc
    V = Verifier(enc)
    feats = [enc.token_feature(t) for t in enc.vocab if enc.token_feature(t) is not None]
    refuted = [f for f in feats if V.verify(must_flag([enc.vocab[f]])).status == REFUTED]
    if not refuted:
        return
    rep = enc.with_must_flag(refuted)
    Vr = Verifier(rep)
    for f in refuted:
        assert Vr.verify(must_flag([enc.vocab[f]])).status == VERIFIED
