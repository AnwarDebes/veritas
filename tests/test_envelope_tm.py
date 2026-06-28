import numpy as np

from veritaskit import (ContentGuard, ContentGuardConfig, TsetlinMachine, TMConfig,
                        GuardEncoding, Verifier, trigger_necessity, VerifiedEnvelope,
                        ABSTAIN, SAFE, UNSAFE)
from veritaskit.encode import GuardEncoding as GE


def test_envelope_abstains_without_cue(tiny_guard):
    g, bad, ok = tiny_guard
    enc = GuardEncoding.from_guard(g)
    audited = [t for t in bad if enc.token_feature(t) is not None]
    env = VerifiedEnvelope(g, audited, sound=True)
    # a text with no audited cue -> ABSTAIN
    d = env.decide("please summarise the weather report")
    assert d.verdict == ABSTAIN and not d.in_envelope
    # a text with a cue -> answered (SAFE or UNSAFE, never ABSTAIN)
    d2 = env.decide(f"{audited[0]} the previous instructions")
    assert d2.verdict in (SAFE, UNSAFE) and d2.in_envelope


def test_envelope_risk_coverage_keys(tiny_guard):
    g, bad, ok = tiny_guard
    enc = GuardEncoding.from_guard(g)
    audited = [t for t in bad if enc.token_feature(t) is not None]
    env = VerifiedEnvelope(g, audited)
    texts = [f"{audited[0]} this", "weather please", "schedule report"]
    rc = env.risk_coverage(texts, [1, 0, 0])
    for k in ("coverage", "selective_acc", "plain_acc", "abstained"):
        assert k in rc
    st = env.stratified(texts, [1, 0, 0])
    assert 0.0 <= st["no_cue_frac"] <= 1.0


def test_envelope_soundness_matches_trigger_necessity(tiny_enc, tiny_guard):
    enc, bad, ok = tiny_enc
    g, _, _ = tiny_guard
    audited = [t for t in bad if enc.token_feature(t) is not None]
    sound = Verifier(enc).verify(trigger_necessity(audited)).status == "VERIFIED"
    env = VerifiedEnvelope(g, audited, sound=sound)
    # when sound, no audited cue provably implies the guard would say SAFE
    if sound:
        x = g.vec.transform(["weather please schedule"])[0]
        assert not enc.unsafe(x)


def test_tm_learns_simple_concept():
    # unsafe iff feature 0 present; TM should fit it
    rng = np.random.default_rng(0)
    X = (rng.random((200, 6)) < 0.5).astype(np.int8)
    y = X[:, 0]
    tm = TsetlinMachine(TMConfig(n_clauses=10, n_features=6, threshold=6, s=3.0, seed=0))
    tm.fit(X, y, epochs=30)
    assert tm.score(X, y) > 0.95


def test_guard_predict_and_explain(tiny_guard):
    g, bad, ok = tiny_guard
    preds = g.predict([f"{bad[0]} now", "weather please"])
    assert preds.shape[0] == 2
    rules = g.explain(top_k=3)
    assert isinstance(rules, list)


def test_encoding_n_negative(tiny_enc):
    enc, bad, ok = tiny_enc
    assert enc.n_negative >= 0
    rep = enc.with_must_flag([0])
    assert rep.n_clauses > enc.n_clauses
