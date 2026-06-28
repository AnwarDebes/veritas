"""Declarative global safety properties for an interpretable guard.

A *property* is a statement about the guard's behaviour on **every** input, e.g.
"no input that mentions a known attack phrase is ever waved through" or "the
guard never raises an alarm without at least one audited trigger token present".
Each property compiles to a single verification query that an exact reasoner
(:mod:`veritaskit.verify`) decides: it either returns ``VERIFIED`` (the property
holds over the whole input space) or ``REFUTED`` with a concrete counterexample.

Two everyday safety questions reduce to one primitive, :class:`Invariant`
(``forall x: PRE(x) => POST(x)``):

* **No-bypass / must-flag**, inputs matching an attack template must be UNSAFE.
* **Trigger-necessity (no spurious cue)**, if none of the audited trigger
  tokens is present, the verdict must be SAFE; equivalently, every UNSAFE verdict
  requires an audited cue.  This is the property that pins a guard to the *reasons
  a human signed off on*, and the one a distribution shift silently violates.

Two more need arithmetic over the vote and so are kept separate:
:class:`RobustnessRadius` (the exact token-edit distance to the nearest verdict
flip) and :class:`Monotone` (adding a token can only push the verdict one way).
:class:`Equivalent` compares two guards over all inputs (used to certify that
hardening did not silently change behaviour elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

Token = Union[str, int]


def _resolve(enc, key: Token) -> Optional[int]:
    """Map a token name or feature index to a feature index (or None if absent)."""
    if isinstance(key, (int, np.integer)):
        return int(key) if 0 <= int(key) < enc.n_features else None
    return enc.token_feature(key)


@dataclass
class Property:
    name: str
    kind: str = "property"

    def describe(self) -> str:
        return self.name


@dataclass
class Invariant(Property):
    """``forall x: (every literal in pre holds) => POST``.

    ``pre`` maps a token/feature to the required value (True=present, False=absent).
    ``post`` is ``"unsafe"`` (x must be flagged) or ``"safe"`` (x must be allowed).
    The verifier checks UNSAT of ``pre AND NOT post``; a model is a counterexample.
    """

    pre: Dict[Token, bool] = field(default_factory=dict)
    post: str = "unsafe"
    kind: str = "invariant"

    def resolve(self, enc) -> "ResolvedInvariant":
        lits: List[tuple] = []
        unresolved: List[str] = []
        for key, val in self.pre.items():
            fi = _resolve(enc, key)
            if fi is None:
                unresolved.append(str(key))
            else:
                lits.append((fi, bool(val)))
        return ResolvedInvariant(self.name, lits, self.post, unresolved)


@dataclass
class ResolvedInvariant:
    name: str
    lits: List[tuple]            # (feature_index, required_value)
    post: str
    unresolved: List[str]

    def holds_on(self, enc, x: np.ndarray) -> bool:
        """True if x does NOT witness a violation (used by the brute oracle)."""
        for (i, val) in self.lits:
            if bool(x[i]) != val:
                return True       # pre not satisfied -> no obligation
        want_unsafe = self.post == "unsafe"
        return enc.unsafe(x) == want_unsafe


@dataclass
class RobustnessRadius(Property):
    """Exact token-edit (Hamming) distance from ``anchor`` to the nearest input
    with the opposite verdict.  A large radius means the verdict is *robust*: an
    adversary must change many tokens to flip it.  ``anchor`` is a feature vector
    (length F); ``max_radius`` bounds the search."""

    anchor: Optional[np.ndarray] = None
    max_radius: int = 8
    kind: str = "robustness_radius"


@dataclass
class Monotone(Property):
    """Adding ``token`` never moves the vote the wrong way.

    ``direction="up"`` (default): presence of ``token`` can only *increase* the
    unsafe vote, appropriate for a known-unsafe token, so it can never be used as
    evidence of safety.  ``direction="down"``: presence can only *decrease* the
    vote, appropriate for a known-benign token."""

    token: Token = ""
    direction: str = "up"
    kind: str = "monotone"


@dataclass
class Equivalent(Property):
    """Two guards agree on every input.  Refutation returns an input on which they
    differ, the basis of a certified change-impact analysis after hardening."""

    other: object = None         # a second GuardEncoding
    kind: str = "equivalent"


#
# convenience builders
#
def must_flag(tokens: Sequence[str], name: Optional[str] = None) -> Invariant:
    """Any input containing ALL of ``tokens`` must be UNSAFE (a no-bypass invariant)."""
    nm = name or f"must_flag({'+'.join(tokens)})"
    return Invariant(nm, pre={t: True for t in tokens}, post="unsafe")


def trigger_necessity(audited: Sequence[str], name: Optional[str] = None) -> Invariant:
    """No UNSAFE verdict without an audited cue: if every audited token is absent,
    the verdict must be SAFE.  Refutation = a hidden/spurious trigger."""
    nm = name or f"trigger_necessity({len(audited)} cues)"
    return Invariant(nm, pre={t: False for t in audited}, post="safe")


def never_flag(tokens: Sequence[str], name: Optional[str] = None) -> Invariant:
    """An input whose ONLY salient tokens are these benign ones must be SAFE.
    (Encoded as: all of ``tokens`` present forces SAFE, a false-alarm guard.)"""
    nm = name or f"never_flag({'+'.join(tokens)})"
    return Invariant(nm, pre={t: True for t in tokens}, post="safe")
