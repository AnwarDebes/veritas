# Changelog

All notable changes to VERITAS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-25

First public release accompanying the paper.

### Added
- **Exact guard encoding**, compiles a trained Tsetlin-Machine content guard into
  an exact Boolean (z3 SMT) and CNF + pseudo-Boolean (PySAT) model, with a pure-NumPy
  oracle. Builds on the TM-verification encoding of Przybysz et al. (2023), which is
  cited as prior art and **not** claimed as novel.
- **Guardrail property classes**, `must_flag` / no-bypass invariants,
  `trigger_necessity` (no-spurious-cue), `monotonicity` (soundness), plus the reused
  exact robustness radius and equivalence. Each decides over all inputs with a proof
  or a concrete counterexample.
- **Three cross-checked backends**, z3, PySAT, and brute-force enumeration; the
  test suite confirms they agree, so a verdict never rests on a single solver.
- **Constructive repair**, a sound, re-verifiable clause-append operator that
  provably enforces a `must_flag` obligation, after establishing that empirical
  counterexample-retraining cannot reach or certify zero bypasses.
- **Verified-competence envelope**, abstains on inputs outside a verified
  `trigger_necessity` region, with a certifiable abstention boundary.
- **Assurance certificate**, a tamper-evident, hash-chained, HMAC-signed,
  Merkle-rooted record of the verification outcome, re-checkable independently of the
  prover (counterexamples re-evaluated with no solver).
- Six reproducible experiments (E1-E6) on two real Apache-2.0 datasets; 31 unit
  tests; figure- and number-generation scripts; the LaTeX paper.

### Fixed (pre-release audit)
A multi-agent soundness audit before release found and corrected:
- **Prover-independent check**, `verify_assurance` now genuinely re-evaluates each
  REFUTED counterexample on the included guard (reconstructing the obligation from
  the recorded spec and failing closed if it cannot), instead of accepting it
  unconditionally; a forged refutation is now rejected with no solver.
- **Backend consistency**, an empty (never-fire) clause is now treated as output 0
  by all three backends (z3, PySAT CNF, and the NumPy oracle) and dropped on
  fingerprint deserialization, closing a latent three-way disagreement.
- **Envelope honesty**, the verified-competence reason strings are now gated on the
  recorded `trigger_necessity` soundness flag, so an unverified envelope no longer
  claims verification backing.

### Known limitations (v1.0)
- Verification is over booleanised feature vectors, not raw strings.
- Verifiability favours compact guards; large-guard queries can exceed a practical
  budget, and accuracy trades against provability (the E1 frontier).
- Only declared properties are proven; the certificate provides symmetric (HMAC)
  authentication, not non-repudiation.
