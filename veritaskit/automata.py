"""Echo-Trace Tsetlin Automata (ETTA), the learning substrate of TRACE.

TRACE re-uses the ETTA primitive introduced in THGTM (Debes, 2026): a standard
two-action Tsetlin Automaton (Granmo, 2018) with ``N`` states per action
(``2N`` total) augmented with one float of exponentially-decaying eligibility
trace.

    state in {1, ..., 2N}   ->  include literal iff state > N
    trace in [0, 1]         ->  decayed memory of recent state transitions

With ``lambda_decay == 0`` and ``trace_alpha == 0`` the bank is bit-exactly a
vanilla Tsetlin-Automaton bank given the same RNG stream.  TRACE's *content
guard* runs in exactly this vanilla regime (i.i.d. text classification); the
trace is exercised by the *behaviour guard* when bounded-history literals are
present.  Keeping a single substrate for both means every TRACE clause, and
therefore every certificate it emits, has identical, inspectable semantics.

This module is intentionally pure NumPy: the whole point is that an auditor can
read it.  Vectorised over (clause, literal) so a guard step is a handful of
array ops.
"""

from __future__ import annotations

import numpy as np


class ETTABank:
    """Vectorised bank of Echo-Trace Tsetlin Automata, one per (clause, literal).

    Parameters
    ----------
    n_clauses : int
        Number of clauses.
    n_literals : int
        Number of literals (typically ``2 * n_features`` for raw + negated).
    n_states_per_action : int
        States per action (the ``N`` parameter); total states ``2N``.
    lambda_decay : float in [0, 1)
        Eligibility-trace decay per update.  ``0`` disables the trace and
        recovers vanilla TA semantics.
    trace_alpha : float >= 0
        Multiplier controlling how much the trace amplifies update
        probabilities.  ``0`` disables the trace effect.
    rng : np.random.Generator | None
        RNG; a fresh ``default_rng`` is created if ``None``.
    """

    def __init__(
        self,
        n_clauses: int,
        n_literals: int,
        n_states_per_action: int = 100,
        lambda_decay: float = 0.0,
        trace_alpha: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        if not (0.0 <= lambda_decay < 1.0):
            raise ValueError("lambda_decay must be in [0, 1)")
        if trace_alpha < 0.0:
            raise ValueError("trace_alpha must be >= 0")

        self.n_clauses = int(n_clauses)
        self.n_literals = int(n_literals)
        self.N = int(n_states_per_action)
        self.lam = float(lambda_decay)
        self.alpha = float(trace_alpha)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Start at the exclude boundary, exactly as the canonical TM does.
        self.state = np.full(
            (self.n_clauses, self.n_literals), self.N, dtype=np.int32
        )
        self.trace = np.zeros((self.n_clauses, self.n_literals), dtype=np.float32)

    #
    # Read API
    #
    def actions(self) -> np.ndarray:
        """(n_clauses, n_literals) bool: True iff the TA includes the literal."""
        return self.state > self.N

    def clause_outputs(self, literals: np.ndarray, predict: bool = False) -> np.ndarray:
        """Evaluate every clause on a single 1-D literal vector.

        A clause outputs the AND over its included literals.  In ``predict``
        mode an all-exclude (empty) clause outputs 0; in train mode it outputs
        1 (the standard Granmo convention).
        """
        literals = np.asarray(literals, dtype=bool)
        inc = self.actions()                                 # (C, L)
        bad = inc & ~literals[np.newaxis, :]                 # included AND value=0
        any_bad = bad.any(axis=1)                            # (C,)
        out = (~any_bad).astype(np.int8)
        if predict:
            empty = ~inc.any(axis=1)
            out = np.where(empty, np.int8(0), out)
        return out

    #
    # Core feedback
    #
    def _apply_delta(self, delta: np.ndarray) -> None:
        """Apply integer state deltas in {-1, 0, +1} and update traces.

        Trace update is ``trace <- lam * trace + 1[delta != 0]`` clamped to 1,
        applied only when ``lambda_decay > 0`` (otherwise the trace stays
        identically zero, matching vanilla TA behaviour bit-for-bit).
        """
        np.clip(self.state + delta, 1, 2 * self.N, out=self.state)
        if self.lam > 0.0:
            np.multiply(self.trace, self.lam, out=self.trace)
            if (delta != 0).any():
                self.trace += (delta != 0).astype(np.float32)
                np.minimum(self.trace, 1.0, out=self.trace)

    def trace_step(self) -> None:
        """Pure trace decay with no state change (use between sequence steps)."""
        if self.lam > 0.0:
            np.multiply(self.trace, self.lam, out=self.trace)

    def _rows(self, feedback_mask) -> np.ndarray:
        if feedback_mask is None:
            return np.arange(self.n_clauses)
        return np.where(np.asarray(feedback_mask, dtype=bool))[0]

    def _apply_delta_rows(self, rows: np.ndarray, delta_rows: np.ndarray) -> None:
        """Apply (len(rows), n_literals) deltas to the given clause rows only.

        Equivalent to building a full (C, L) delta that is zero outside ``rows``
        and calling :meth:`_apply_delta`, but O(|rows| * L) instead of O(C * L).
        Most feedback events touch only a fraction of the clauses, so restricting
        to the selected rows is the dominant training speed-up.
        """
        self.state[rows] = np.clip(self.state[rows] + delta_rows, 1, 2 * self.N)
        if self.lam > 0.0:
            np.multiply(self.trace, self.lam, out=self.trace)
            nz = delta_rows != 0
            if nz.any():
                self.trace[rows] += nz.astype(np.float32)
                np.minimum(self.trace, 1.0, out=self.trace)

    def type_I_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        s: float,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        """Type I (boost-and-forget) feedback to selected clauses."""
        rows = self._rows(feedback_mask)
        if rows.size == 0:
            return
        cr = np.asarray(clause_outputs, dtype=bool)[rows]   # (R,)
        x = np.asarray(literals, dtype=bool)                # (L,)
        R, L = rows.size, self.n_literals

        boost_p = (s - 1.0) / s
        forget_p = 1.0 / s
        if self.alpha > 0.0:
            amp = 1.0 + self.alpha * self.trace[rows]       # (R, L)
            boost_p_arr = np.minimum(1.0, boost_p * amp)
            forget_p_arr = np.minimum(1.0, forget_p * amp)
        else:
            boost_p_arr = boost_p                            # scalars broadcast
            forget_p_arr = forget_p

        boost_mask = cr[:, None] & x[None, :]               # clause fires AND literal=1
        rand = self.rng.random((R, L)).astype(np.float32)
        delta = np.zeros((R, L), dtype=np.int32)
        delta[boost_mask & (rand < boost_p_arr)] = 1
        delta[(~boost_mask) & (rand < forget_p_arr)] = -1
        self._apply_delta_rows(rows, delta)

    def type_II_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        """Type II (discriminative) feedback to selected clauses.

        Deterministic: if a clause fires while an excluded literal is 0, push
        that TA toward include (+1).  Nothing else changes.
        """
        rows = self._rows(feedback_mask)
        if rows.size == 0:
            return
        cr = np.asarray(clause_outputs, dtype=bool)[rows]
        x = np.asarray(literals, dtype=bool)
        inc = self.state[rows] > self.N                     # include state of selected rows
        delta = (cr[:, None] & (~x[None, :]) & (~inc)).astype(np.int32)
        self._apply_delta_rows(rows, delta)


class EchoTraceAutomaton:
    """Single Echo-Trace Tsetlin Automaton (handy for tests and diagrams)."""

    def __init__(self, n_states_per_action: int = 100, lambda_decay: float = 0.0):
        self.N = int(n_states_per_action)
        self.lam = float(lambda_decay)
        self.state = self.N
        self.trace = 0.0

    @property
    def includes(self) -> bool:
        return self.state > self.N

    def update(self, delta: int) -> None:
        self.state = max(1, min(2 * self.N, self.state + delta))
        self.trace = min(self.lam * self.trace + (1.0 if delta != 0 else 0.0), 1.0)

    def step(self) -> None:
        self.trace = self.lam * self.trace
