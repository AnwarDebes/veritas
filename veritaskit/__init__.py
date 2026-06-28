"""VERITAS, VERified Interpretable Tsetlin ASsurance.

A formally *verified*, *competence-aware* interpretable guardrail.  VERITAS takes
a Tsetlin-Machine content guard whose decision is a native propositional formula,
compiles it exactly into Boolean / SMT form, and decides global safety properties
over the *entire* input space: it returns a proof, or a concrete counterexample.
It then (i) *hardens* the guard against its own counterexamples until a property
provably holds, and (ii) wraps it in a *verified-competence envelope* that abstains, auditably, on inputs outside the region where a guarantee holds, turning
silent off-distribution failure into escalation.  Every verdict is sealed into a
tamper-evident assurance certificate an auditor re-checks independently.

The exact TM-to-Boolean encoding and the robustness / equivalence properties are
*not* our invention: they are due to Przybysz, Bhattarai, Persia, Ozaki, Granmo &
Sharma, "Verifying Properties of Tsetlin Machines" (2023), which VERITAS builds
on.  Our contribution is the verified-guardrail *system*: new safety-property
classes (no-bypass invariants, trigger-necessity / no-spurious-cue, monotonicity
as a soundness property), counterexample-guided hardening, and verified-envelope
abstention with a machine-checkable deferral certificate.

Fourth pillar after THGTM (learning), TRACE (detect+certify), REINS (enforce).
"""

from .guard import ContentGuard, ContentGuardConfig, ContentVerdict
from .tm import TsetlinMachine, TMConfig, to_literals
from .encode import GuardEncoding
from .properties import (Property, Invariant, RobustnessRadius, Monotone, Equivalent,
                         must_flag, trigger_necessity, never_flag)
from .verify import Verifier, VResult, VERIFIED, REFUTED, UNKNOWN
from .harden import harden, HardenResult, HardenRound
from .envelope import VerifiedEnvelope, EnvelopeDecision, SAFE, UNSAFE, ABSTAIN
from .ledger import (AssuranceLedger, AssuranceCertificate, verify_assurance,
                     guard_fingerprint, encoding_from_fingerprint)
from .engine import VeritasEngine, Specification

__version__ = "1.0.0"

__all__ = [
    "ContentGuard", "ContentGuardConfig", "ContentVerdict",
    "TsetlinMachine", "TMConfig", "to_literals",
    "GuardEncoding",
    "Property", "Invariant", "RobustnessRadius", "Monotone", "Equivalent",
    "must_flag", "trigger_necessity", "never_flag",
    "Verifier", "VResult", "VERIFIED", "REFUTED", "UNKNOWN",
    "harden", "HardenResult", "HardenRound",
    "VerifiedEnvelope", "EnvelopeDecision", "SAFE", "UNSAFE", "ABSTAIN",
    "AssuranceLedger", "AssuranceCertificate", "verify_assurance",
    "guard_fingerprint", "encoding_from_fingerprint",
    "VeritasEngine", "Specification",
    "__version__",
]
