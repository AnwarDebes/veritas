"""Counterexample-guided hardening of a guard.

Verification does not only *grade* a guard, it *teaches* it.  When a property
is refuted, the counterexample is a concrete input the guard gets wrong; feeding
those inputs back as labelled training data and retraining is a textbook
counterexample-guided loop (CEGIS / abstraction-refinement; Solar-Lezama 2006;
Clarke et al. 2000) applied to a safety classifier.  Round after round the set of
provable bypasses shrinks; when it reaches zero the guard is not "robust on the
test set we tried", it is *provably* robust for the declared property class.

We harden at the feature level: a counterexample is a Boolean feature vector, and
its *correct* label is whatever the violated obligation demanded (an input that
should have been flagged is added as UNSAFE; a spurious alarm is added as SAFE).
The vocabulary is frozen, so every round is comparable and the proofs compose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .guard import ContentGuard
from .tm import TsetlinMachine, TMConfig
from .encode import GuardEncoding
from .properties import Invariant
from .verify import Verifier, REFUTED


@dataclass
class HardenRound:
    round: int
    n_bypassed: int
    bypassed: List[str]
    n_added: int
    test_acc: Optional[float] = None


@dataclass
class HardenResult:
    guard: ContentGuard
    rounds: List[HardenRound]
    verified: bool                  # all target invariants hold at the end
    history: List[int] = field(default_factory=list)   # bypass count per round


def _label_for(inv: Invariant) -> int:
    """The correct label a counterexample to ``inv`` should have been given."""
    return 1 if inv.post == "unsafe" else 0


def _train_tm(vec, X, y, cfg, seed) -> TsetlinMachine:
    tm = TsetlinMachine(TMConfig(
        n_clauses=cfg.n_clauses, n_features=vec.n_features,
        n_states_per_action=cfg.n_states_per_action, threshold=cfg.threshold,
        s=cfg.s, seed=seed))
    tm.fit(X, np.asarray(y, dtype=np.int8), epochs=cfg.epochs)
    return tm


def _guard_with(vec, tm, cfg) -> ContentGuard:
    g = ContentGuard(cfg)
    g.vec = vec
    g.tm = tm
    return g


def harden(guard: ContentGuard, X: np.ndarray, y: Sequence[int],
           invariants: Sequence[Invariant], rounds: int = 8,
           neighbors: int = 6, backend: str = "z3",
           X_test: Optional[np.ndarray] = None, y_test: Optional[Sequence[int]] = None,
           rng: Optional[np.random.Generator] = None, verbose: bool = False) -> HardenResult:
    """Iteratively repair ``guard`` until every invariant in ``invariants`` holds.

    Returns the hardened guard and a per-round log of how many invariants were
    still bypassable.  ``neighbors`` random same-label perturbations are added per
    counterexample to help the fix generalise beyond the single witness.
    """
    rng = rng or np.random.default_rng(0)
    vec, cfg = guard.vec, guard.cfg
    Xa = np.asarray(X, dtype=np.int8).copy()
    ya = list(y)
    cur = guard
    log: List[HardenRound] = []

    for r in range(rounds):
        enc = GuardEncoding.from_guard(cur)
        V = Verifier(enc, backend=backend)
        bypassed, added = [], []
        for inv in invariants:
            res = V.verify(inv)
            if res.status == REFUTED and res.counterexample is not None:
                bypassed.append(inv.name)
                lbl = _label_for(inv)
                cx = np.asarray(res.counterexample, dtype=np.int8)
                added.append((cx, lbl))
                for _ in range(neighbors):
                    nb = cx.copy()
                    flips = rng.integers(0, enc.n_features, size=int(rng.integers(1, 3)))
                    # keep the literals the property fixed; perturb only free bits
                    fixed = {i for (i, _) in inv.resolve(enc).lits}
                    for fi in flips:
                        if int(fi) not in fixed:
                            nb[int(fi)] ^= 1
                    added.append((nb, lbl))
        acc = None
        if X_test is not None and y_test is not None:
            acc = float((cur.tm.predict(X_test) == np.asarray(y_test)).mean())
        log.append(HardenRound(r, len(bypassed), bypassed, len(added), acc))
        if verbose:
            print(f"  [harden r{r}] bypassed={len(bypassed)} add={len(added)} acc={acc}")
        if not bypassed:
            return HardenResult(cur, log, True, [x.n_bypassed for x in log])
        # augment + retrain
        if added:
            Xa = np.vstack([Xa] + [cx.reshape(1, -1) for (cx, _) in added])
            ya = ya + [lbl for (_, lbl) in added]
        tm = _train_tm(vec, Xa, ya, cfg, seed=1000 + r)
        cur = _guard_with(vec, tm, cfg)

    # final check
    enc = GuardEncoding.from_guard(cur)
    V = Verifier(enc, backend=backend)
    remaining = [inv.name for inv in invariants if V.verify(inv).status == REFUTED]
    return HardenResult(cur, log, len(remaining) == 0, [x.n_bypassed for x in log])
