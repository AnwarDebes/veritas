"""Exact verification of guard properties: proof, or counterexample.

Given a :class:`~veritaskit.encode.GuardEncoding` and a property from
:mod:`veritaskit.properties`, the :class:`Verifier` decides the property over the
*entire* input space and returns:

* ``VERIFIED``, the property provably holds for every input; or
* ``REFUTED``, with a concrete counterexample feature vector; or
* ``UNKNOWN``, the backend could not decide within its budget.

Three interchangeable backends keep the result trustworthy:

* **z3** (SMT), the default; encodes the vote as an integer term.
* **pysat** (SAT), encodes the same decision as CNF plus a cardinality
  constraint (the +/-1-weighted threshold becomes ``>= k`` once the
  negative-polarity clause literals are flipped), and can emit a proof for UNSAT.
* **brute**, exhaustive enumeration over all ``2**F`` inputs; only for small
  ``F``, but a model-free oracle the other two are checked against in the tests.

A counterexample is *self-checking*: it is a plain feature vector, and
``GuardEncoding.unsafe`` (pure NumPy, no solver) re-derives the guard's verdict on
it, so an auditor confirms a reported bypass without trusting any solver at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .encode import GuardEncoding
from .properties import (Invariant, RobustnessRadius, Monotone, Equivalent,
                         Property)

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
UNKNOWN = "UNKNOWN"


@dataclass
class VResult:
    name: str
    kind: str
    status: str
    backend: str
    elapsed_us: float
    counterexample: Optional[List[int]] = None      # feature vector witnessing a violation
    detail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "status": self.status,
                "backend": self.backend, "elapsed_us": round(self.elapsed_us, 2),
                "counterexample": self.counterexample, "detail": self.detail,
                "extra": self.extra}


class Verifier:
    def __init__(self, enc: GuardEncoding, backend: str = "z3", realizable: bool = False,
                 want_proof: bool = False):
        self.enc = enc
        self.backend = backend
        self.realizable = realizable
        self.want_proof = want_proof          # emit a DRAT proof on the pysat path

    #
    def verify(self, prop: Property, backend: Optional[str] = None) -> VResult:
        be = backend or self.backend
        if isinstance(prop, Invariant):
            if be == "brute":
                return self._invariant_brute(prop)
            if be == "pysat":
                return self._invariant_pysat(prop)
            return self._invariant_z3(prop)
        if isinstance(prop, RobustnessRadius):
            return self._robustness_brute(prop) if be == "brute" else self._robustness_z3(prop)
        if isinstance(prop, Monotone):
            return self._monotone_brute(prop) if be == "brute" else self._monotone_z3(prop)
        if isinstance(prop, Equivalent):
            return self._equivalent_brute(prop) if be == "brute" else self._equivalent_z3(prop)
        raise TypeError(f"unknown property type {type(prop).__name__}")

    def cross_check(self, prop: Property, backends=("z3", "pysat", "brute")) -> dict:
        """Run several backends and confirm they agree on the verdict (used by the
        test suite and the soundness experiment).  ``brute`` is skipped if F is
        large."""
        out = {}
        for be in backends:
            if be == "brute" and self.enc.n_features > 20:
                continue
            if be == "pysat" and not isinstance(prop, Invariant):
                continue                       # pysat path implemented for invariants
            try:
                out[be] = self.verify(prop, backend=be)
            except Exception as e:               # pragma: no cover - backend missing
                out[be] = VResult(prop.name, getattr(prop, "kind", "?"), UNKNOWN, be, 0.0,
                                  detail=f"{type(e).__name__}: {e}")
        statuses = {be: r.status for be, r in out.items() if r.status != UNKNOWN}
        out["_agree"] = len(set(statuses.values())) <= 1
        return out

    #
    # Invariant
    #
    def _invariant_z3(self, prop: Invariant) -> VResult:
        import z3
        ri = prop.resolve(self.enc)
        t0 = time.perf_counter()
        b, vote, unsafe = self.enc.z3_build()
        s = z3.Solver()
        for (i, val) in ri.lits:
            s.add(b[i] if val else z3.Not(b[i]))
        # violation = pre AND NOT(post)
        s.add(z3.Not(unsafe) if ri.post == "unsafe" else unsafe)
        if self.realizable:
            for c in self.enc.realizability_z3(b):
                s.add(c)
        r = s.check()
        el = (time.perf_counter() - t0) * 1e6
        if r == z3.unsat:
            return VResult(prop.name, "invariant", VERIFIED, "z3", el,
                           detail=self._inv_detail(ri, True))
        if r == z3.sat:
            m = s.model()
            cx = [1 if z3.is_true(m.eval(b[i], model_completion=True)) else 0
                  for i in range(self.enc.n_features)]
            return VResult(prop.name, "invariant", REFUTED, "z3", el,
                           counterexample=cx, detail=self._cx_detail(cx))
        return VResult(prop.name, "invariant", UNKNOWN, "z3", el, detail=str(r))

    def _invariant_pysat(self, prop: Invariant) -> VResult:
        from pysat.card import CardEnc, EncType
        from pysat.solvers import Solver
        ri = prop.resolve(self.enc)
        t0 = time.perf_counter()
        cnf, feat_vars, clause_vars, weights, nv = self.enc.to_cnf()
        # +/-1 threshold -> cardinality: counts-toward-unsafe literals
        P = [clause_vars[j] for j, w in enumerate(weights) if w > 0]
        N = [clause_vars[j] for j, w in enumerate(weights) if w < 0]
        lits_unsafe = P + [-c for c in N]            # vote>0  <=>  sum(lits_unsafe) >= 1+|N|
        nN = len(N)
        top = nv - 1
        if ri.post == "unsafe":      # violation: pre AND SAFE  (sum <= |N|)
            card = CardEnc.atmost(lits=lits_unsafe, bound=nN, top_id=top, encoding=EncType.seqcounter)
        else:                         # violation: pre AND UNSAFE (sum >= 1+|N|)
            card = CardEnc.atleast(lits=lits_unsafe, bound=1 + nN, top_id=top, encoding=EncType.seqcounter)
        clauses = list(cnf) + list(card.clauses)
        for (i, val) in ri.lits:
            clauses.append([feat_vars[i] if val else -feat_vars[i]])
        with Solver(name="cadical153", bootstrap_with=clauses, with_proof=self.want_proof) as sv:
            sat = sv.solve()
            model = sv.get_model() if sat else None
            proof_len = len(sv.get_proof()) if (not sat and self.want_proof) else 0
        el = (time.perf_counter() - t0) * 1e6
        if not sat:
            return VResult(prop.name, "invariant", VERIFIED, "pysat", el,
                           detail=self._inv_detail(ri, True),
                           extra={"drat_proof_lines": proof_len})
        mset = set(model)
        cx = [1 if feat_vars[i] in mset else 0 for i in range(self.enc.n_features)]
        return VResult(prop.name, "invariant", REFUTED, "pysat", el,
                       counterexample=cx, detail=self._cx_detail(cx))

    def _invariant_brute(self, prop: Invariant) -> VResult:
        ri = prop.resolve(self.enc)
        F = self.enc.n_features
        if F > 24:
            return VResult(prop.name, "invariant", UNKNOWN, "brute", 0.0,
                           detail=f"F={F} too large for enumeration")
        t0 = time.perf_counter()
        fixed = {i: val for (i, val) in ri.lits}
        free = [i for i in range(F) if i not in fixed]
        base = np.zeros(F, dtype=np.int8)
        for i, v in fixed.items():
            base[i] = 1 if v else 0
        want_unsafe = ri.post == "unsafe"
        cx = None
        for mask in range(1 << len(free)):
            x = base.copy()
            for bit, i in enumerate(free):
                if (mask >> bit) & 1:
                    x[i] = 1
            if self.enc.unsafe(x) != want_unsafe:
                cx = [int(v) for v in x]
                break
        el = (time.perf_counter() - t0) * 1e6
        if cx is None:
            return VResult(prop.name, "invariant", VERIFIED, "brute", el,
                           detail=self._inv_detail(ri, True))
        return VResult(prop.name, "invariant", REFUTED, "brute", el,
                       counterexample=cx, detail=self._cx_detail(cx))

    #
    # Robustness radius
    #
    def _robustness_z3(self, prop: RobustnessRadius) -> VResult:
        import z3
        a = np.asarray(prop.anchor).astype(int)
        a_unsafe = self.enc.unsafe(a)
        t0 = time.perf_counter()
        b, vote, unsafe = self.enc.z3_build()
        flip = z3.Not(unsafe) if a_unsafe else unsafe
        ham = z3.Sum([z3.If(b[i] != bool(a[i]), 1, 0) for i in range(self.enc.n_features)])
        radius, cx = None, None
        for d in range(1, prop.max_radius + 1):
            s = z3.Solver()
            s.add(flip)
            s.add(ham <= d)
            if s.check() == z3.sat:
                m = s.model()
                cx = [1 if z3.is_true(m.eval(b[i], model_completion=True)) else 0
                      for i in range(self.enc.n_features)]
                radius = d
                break
        el = (time.perf_counter() - t0) * 1e6
        if radius is None:
            return VResult(prop.name, "robustness_radius", VERIFIED, "z3", el,
                           detail=f"verdict robust to >{prop.max_radius} token edits",
                           extra={"radius": prop.max_radius + 1, "exceeds": True,
                                  "anchor_unsafe": bool(a_unsafe)})
        return VResult(prop.name, "robustness_radius", REFUTED, "z3", el,
                       counterexample=cx,
                       detail=f"verdict flips after {radius} token edit(s)",
                       extra={"radius": radius, "exceeds": False,
                              "anchor_unsafe": bool(a_unsafe),
                              "edited": [i for i in range(self.enc.n_features) if cx[i] != int(a[i])]})

    def _robustness_brute(self, prop: RobustnessRadius) -> VResult:
        from itertools import combinations
        a = np.asarray(prop.anchor).astype(int)
        a_unsafe = self.enc.unsafe(a)
        F = self.enc.n_features
        t0 = time.perf_counter()
        radius, cx = None, None
        for d in range(1, prop.max_radius + 1):
            for combo in combinations(range(F), d):
                x = a.copy()
                for i in combo:
                    x[i] ^= 1
                if self.enc.unsafe(x) != a_unsafe:
                    radius, cx = d, [int(v) for v in x]
                    break
            if radius is not None:
                break
        el = (time.perf_counter() - t0) * 1e6
        if radius is None:
            return VResult(prop.name, "robustness_radius", VERIFIED, "brute", el,
                           extra={"radius": prop.max_radius + 1, "exceeds": True})
        return VResult(prop.name, "robustness_radius", REFUTED, "brute", el,
                       counterexample=cx, extra={"radius": radius, "exceeds": False})

    #
    # Monotonicity (two copies sharing all but one feature)
    #
    def _monotone_z3(self, prop: Monotone) -> VResult:
        import z3
        ft = self.enc.token_feature(prop.token) if isinstance(prop.token, str) else int(prop.token)
        if ft is None:
            return VResult(prop.name, "monotone", UNKNOWN, "z3", 0.0,
                           detail=f"token '{prop.token}' not a feature")
        t0 = time.perf_counter()
        b0 = [z3.Bool(f"m0_{i}") for i in range(self.enc.n_features)]
        b1 = [z3.Bool(f"m1_{i}") for i in range(self.enc.n_features)]
        s = z3.Solver()
        for i in range(self.enc.n_features):
            if i == ft:
                s.add(z3.Not(b0[i])); s.add(b1[i])
            else:
                s.add(b0[i] == b1[i])
        v0, _ = self.enc.z3_vote(b0)
        v1, _ = self.enc.z3_vote(b1)
        s.add(v1 < v0 if prop.direction == "up" else v1 > v0)   # violation
        r = s.check()
        el = (time.perf_counter() - t0) * 1e6
        if r == z3.unsat:
            return VResult(prop.name, "monotone", VERIFIED, "z3", el,
                           detail=f"vote is monotone-{prop.direction} in '{prop.token}'")
        if r == z3.sat:
            m = s.model()
            cx = [1 if z3.is_true(m.eval(b0[i], model_completion=True)) else 0
                  for i in range(self.enc.n_features)]
            return VResult(prop.name, "monotone", REFUTED, "z3", el, counterexample=cx,
                           detail=f"adding '{prop.token}' moves the vote the wrong way")
        return VResult(prop.name, "monotone", UNKNOWN, "z3", el)

    def _monotone_brute(self, prop: Monotone) -> VResult:
        ft = self.enc.token_feature(prop.token) if isinstance(prop.token, str) else int(prop.token)
        F = self.enc.n_features
        if ft is None or F > 24:
            return VResult(prop.name, "monotone", UNKNOWN, "brute", 0.0)
        t0 = time.perf_counter()
        free = [i for i in range(F) if i != ft]
        cx = None
        for mask in range(1 << len(free)):
            x = np.zeros(F, dtype=np.int8)
            for bit, i in enumerate(free):
                if (mask >> bit) & 1:
                    x[i] = 1
            x0 = x.copy(); x0[ft] = 0
            x1 = x.copy(); x1[ft] = 1
            d = self.enc.vote(x1) - self.enc.vote(x0)
            bad = d < 0 if prop.direction == "up" else d > 0
            if bad:
                cx = [int(v) for v in x0]; break
        el = (time.perf_counter() - t0) * 1e6
        status = VERIFIED if cx is None else REFUTED
        return VResult(prop.name, "monotone", status, "brute", el, counterexample=cx)

    #
    # Equivalence between two guards
    #
    def _equivalent_z3(self, prop: Equivalent) -> VResult:
        import z3
        other: GuardEncoding = prop.other
        if other.n_features != self.enc.n_features:
            return VResult(prop.name, "equivalent", UNKNOWN, "z3", 0.0,
                           detail="feature spaces differ")
        t0 = time.perf_counter()
        b = [z3.Bool(f"e_{i}") for i in range(self.enc.n_features)]
        _, u1 = self.enc.z3_vote(b)
        _, u2 = other.z3_vote(b)
        s = z3.Solver(); s.add(u1 != u2)
        r = s.check()
        el = (time.perf_counter() - t0) * 1e6
        if r == z3.unsat:
            return VResult(prop.name, "equivalent", VERIFIED, "z3", el,
                           detail="guards agree on every input")
        if r == z3.sat:
            m = s.model()
            cx = [1 if z3.is_true(m.eval(b[i], model_completion=True)) else 0
                  for i in range(self.enc.n_features)]
            return VResult(prop.name, "equivalent", REFUTED, "z3", el, counterexample=cx,
                           detail="guards disagree on this input")
        return VResult(prop.name, "equivalent", UNKNOWN, "z3", el)

    def _equivalent_brute(self, prop: Equivalent) -> VResult:
        other: GuardEncoding = prop.other
        F = self.enc.n_features
        if other.n_features != F or F > 24:
            return VResult(prop.name, "equivalent", UNKNOWN, "brute", 0.0)
        t0 = time.perf_counter()
        cx = None
        for mask in range(1 << F):
            x = np.array([(mask >> i) & 1 for i in range(F)], dtype=np.int8)
            if self.enc.unsafe(x) != other.unsafe(x):
                cx = [int(v) for v in x]; break
        el = (time.perf_counter() - t0) * 1e6
        status = VERIFIED if cx is None else REFUTED
        return VResult(prop.name, "equivalent", status, "brute", el, counterexample=cx)

    #
    def _inv_detail(self, ri, verified: bool) -> str:
        pre = " AND ".join(f"{'' if v else 'NOT '}{self.enc.feature_token(i)}" for i, v in ri.lits) or "true"
        return f"forall x: ({pre}) => {ri.post.upper()}  [{'holds' if verified else 'violated'}]"

    def _cx_detail(self, cx: List[int]) -> str:
        present = [self.enc.feature_token(i) for i, v in enumerate(cx) if v]
        v = self.enc.vote(np.asarray(cx))
        shown = ", ".join(present[:8]) + (" ..." if len(present) > 8 else "")
        return f"counterexample: tokens={{{shown}}} vote={v} -> {'UNSAFE' if v > 0 else 'SAFE'}"
