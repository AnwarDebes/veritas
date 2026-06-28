"""Verified-envelope abstention: turning silent off-distribution failure into
auditable deferral.

Every learned guard has a competence boundary, and the dangerous failure is the
*silent* one: under distribution shift the guard keeps emitting confident SAFE
verdicts on inputs it has no basis to judge.  VERITAS makes that boundary
explicit and *proven*.

If the property ``trigger_necessity(U)`` is VERIFIED, the guard provably never
raises an alarm unless at least one audited cue token from ``U`` is present, then on any input with no audited cue the guard's verdict is SAFE *by absence of
evidence*, not by positive judgement.  The envelope treats exactly that region as
**out of competence**: it ABSTAINS (escalates to a human or a heavier model)
instead of forwarding an unsupported SAFE.  Inside the envelope (an audited cue is
present) the guard's verdict is backed by the verified property and is trusted.

This is selective prediction (Chow 1970; El-Yaniv & Wiener 2010) with a twist: the
accept/reject boundary is not a softmax threshold but a *formally verified
precondition*, so the abstention region is itself certifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

SAFE, UNSAFE, ABSTAIN = "SAFE", "UNSAFE", "ABSTAIN"


@dataclass
class EnvelopeDecision:
    verdict: str                    # SAFE | UNSAFE | ABSTAIN
    in_envelope: bool
    reason: str
    cues_present: List[str]
    sound: bool = False             # was trigger_necessity(U) actually verified?


class VerifiedEnvelope:
    """Wrap a guard with a verified-competence gate.

    Parameters, guard : ContentGuard
        The (verified) guard.
    audited : sequence of str
        The audited cue lexicon ``U``; should be the same set for which
        ``trigger_necessity(U)`` was verified, so the abstention region is sound.
    sound : bool
        Whether ``trigger_necessity(U)`` was actually verified for this guard
        (recorded for the certificate / honesty).
    """

    def __init__(self, guard, audited: Sequence[str], sound: bool = False):
        self.guard = guard
        self.audited = [t for t in audited if t in getattr(guard.vec, "index_", {})]
        self.sound = sound
        self._idx = [guard.vec.index_[t] for t in self.audited]

    def _cues(self, x: np.ndarray) -> List[str]:
        return [t for t, i in zip(self.audited, self._idx) if x[i]]

    def decide(self, text: str) -> EnvelopeDecision:
        x = self.guard.vec.transform([text])[0]
        cues = self._cues(x)
        v = self.guard.verdict(text)
        # The verification-backed language is only honest when trigger_necessity(U)
        # was actually verified; otherwise the abstention is a heuristic, not a proof.
        backed = "verdict backed by verified trigger-necessity" if self.sound \
            else "trigger-necessity NOT verified for this guard (heuristic only)"
        comp = "out of verified competence" if self.sound else "out of cue region"
        if cues:
            return EnvelopeDecision(UNSAFE if v.unsafe else SAFE, True,
                                    f"audited cue present; {backed}", cues, self.sound)
        # no audited cue: a verified guard would say SAFE by absence of evidence.
        return EnvelopeDecision(ABSTAIN, False,
                                f"no audited cue; {comp} -> escalate", cues, self.sound)

    def batch(self, texts: Sequence[str]) -> List[EnvelopeDecision]:
        return [self.decide(t) for t in texts]

    #
    def risk_coverage(self, texts: Sequence[str], labels: Sequence[int]) -> dict:
        """Selective-prediction report for a labelled set.

        coverage          = fraction of inputs the envelope answers (not ABSTAIN);
        selective_acc     = accuracy on the answered inputs;
        plain_acc         = the underlying guard's accuracy on ALL inputs (no gate);
        missed_recovered  = of the attacks the plain guard MISSED (called SAFE),
                            the fraction the envelope instead ABSTAINS on (i.e.
                            converts a silent miss into an escalation).
        """
        labels = np.asarray(labels)
        decs = self.batch(texts)
        answered = [(d, y) for d, y in zip(decs, labels) if d.verdict != ABSTAIN]
        cov = len(answered) / max(1, len(decs))
        if answered:
            correct = sum(int((d.verdict == UNSAFE) == bool(y)) for d, y in answered)
            sel_acc = correct / len(answered)
        else:
            sel_acc = float("nan")
        plain = self.guard.predict(texts)
        plain_acc = float((plain == labels).mean())
        missed = [(d, y) for d, y, p in zip(decs, labels, plain) if y == 1 and p == 0]
        recovered = sum(int(d.verdict == ABSTAIN) for d, _ in missed)
        return {"coverage": cov, "selective_acc": sel_acc, "plain_acc": plain_acc,
                "n": len(decs), "abstained": sum(int(d.verdict == ABSTAIN) for d in decs),
                "missed_by_guard": len(missed),
                "missed_recovered": (recovered / len(missed)) if missed else float("nan")}

    def stratified(self, texts: Sequence[str], labels: Sequence[int]) -> dict:
        """Split inputs by whether they carry an audited cue and report the *plain
        guard's* accuracy on each side, a diagnostic, not a claim of superiority.

        Under a verified ``trigger_necessity(U)`` the no-cue side is exactly where
        the guard emits a constant SAFE verdict (no positive discrimination), so it
        is where a silently-missed attack hides; the envelope abstains there.  Note
        this does NOT imply the kept (has-cue) side is more accurate: under a hard
        shift the guard can be badly miscalibrated on the cue-bearing inputs it
        keeps, so ``acc_has_cue`` may fall below ``acc_no_cue``.  The envelope's
        guarantee is the *soundness of the boundary*, not selective accuracy."""
        labels = np.asarray(labels)
        X = self.guard.vec.transform(texts)
        has_cue = np.array([any(X[i, j] for j in self._idx) for i in range(len(texts))])
        plain = self.guard.predict(texts)
        def acc(mask):
            m = np.asarray(mask)
            return float((plain[m] == labels[m]).mean()) if m.any() else float("nan")
        base_rate = float(max(labels.mean(), 1 - labels.mean()))
        return {"n": len(texts), "no_cue_frac": float((~has_cue).mean()),
                "acc_has_cue": acc(has_cue), "acc_no_cue": acc(~has_cue),
                "base_rate": base_rate}
