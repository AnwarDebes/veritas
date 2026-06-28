"""The VERITAS engine: from a trained guard to a sealed assurance certificate.

A :class:`Specification` is a named bundle of safety properties a deployer wants
to hold.  :class:`VeritasEngine` verifies the bundle against a guard, records each
verdict (with the spec needed to re-run it) into a tamper-evident assurance
ledger, and seals a certificate an auditor re-checks independently.  Hardening and
the verified-competence envelope plug in around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .encode import GuardEncoding
from .properties import Invariant, RobustnessRadius, Monotone, Equivalent, Property
from .verify import Verifier, VResult, VERIFIED
from .ledger import AssuranceLedger, AssuranceCertificate


@dataclass
class Specification:
    spec_id: str
    properties: List[Property] = field(default_factory=list)

    def add(self, p: Property) -> "Specification":
        self.properties.append(p)
        return self


def _spec_of(prop: Property) -> Optional[dict]:
    """JSON-safe metadata that lets verify_assurance re-run this property."""
    if isinstance(prop, Invariant):
        return {"kind": "invariant",
                "pre": {str(k): bool(v) for k, v in prop.pre.items()},
                "post": prop.post}
    if isinstance(prop, RobustnessRadius):
        return {"kind": "robustness_radius",
                "anchor": [int(v) for v in np.asarray(prop.anchor).tolist()],
                "max_radius": int(prop.max_radius)}
    if isinstance(prop, Monotone):
        return {"kind": "monotone", "token": prop.token, "direction": prop.direction}
    return None


class VeritasEngine:
    def __init__(self, guard, key: bytes, backend: str = "z3", realizable: bool = False):
        self.guard = guard
        self.key = key
        self.backend = backend
        self.enc = GuardEncoding.from_guard(guard)
        self.verifier = Verifier(self.enc, backend=backend, realizable=realizable)

    def verify(self, properties: Sequence[Property], backend: Optional[str] = None) -> List[VResult]:
        out = []
        for p in properties:
            r = self.verifier.verify(p, backend=backend)
            spec = _spec_of(p)
            if spec is not None:
                r.extra = dict(r.extra)
                r.extra["spec"] = spec
            out.append(r)
        return out

    def assure(self, spec: Specification, backend: Optional[str] = None):
        """Verify a whole spec and seal a certificate.  Returns
        ``(results, ledger_dict, certificate)``."""
        results = self.verify(spec.properties, backend=backend)
        ledger = AssuranceLedger(spec.spec_id, self.key)
        ledger.record_guard(self.enc)
        for r in results:
            ledger.record_result(r)
        n_verified = sum(int(r.status == VERIFIED) for r in results)
        summary = {"n_properties": len(results), "n_verified": n_verified,
                   "n_refuted": sum(int(r.status == "REFUTED") for r in results),
                   "backend": backend or self.backend,
                   "guard": {"n_features": self.enc.n_features,
                             "n_clauses": self.enc.n_clauses}}
        cert = ledger.seal(summary)
        return results, ledger.to_dict(), cert
