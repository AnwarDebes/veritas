<div align="center">

<img src="assets/banner.svg" alt="VERITAS: VERified Interpretable Tsetlin ASsurance" width="100%"/>

<br/>
<br/>

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-3fb950)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-31%20passing-2dd4bf)](tests/)
[![Backends](https://img.shields.io/badge/backends-z3%20%7C%20PySAT%20%7C%20brute-0d9488)](veritaskit/verify.py)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/veritas.pdf)
[![Status](https://img.shields.io/badge/release-v1.1.0-0f766e)](CHANGELOG.md)
[![Live prover](https://img.shields.io/badge/live%20prover-open%20in%20your%20browser-2dd4bf)](https://anwardebes.github.io/veritas/)

<br/>

**A formally _verified_, competence-aware interpretable guardrail for language models.**

Every guardrail in production is validated the same way: by red-teaming. A team
throws adversarial prompts at it, enough bounce off, and it ships. But passing _N_
probes proves nothing about the _N+1_-th, and a learned guard fails _silently_
off-distribution, returning confident verdicts on inputs it has no basis to judge.
VERITAS takes a different route. Because its guard is an interpretable Tsetlin
Machine, the decision is a propositional formula, so an exact reasoner can _prove_ a
safety property over **all** inputs at once, hand back a concrete **counterexample**
when one fails, **repair** that bypass by construction, **abstain** (provably) where
the guard has no audited competence, and seal the entire argument into a certificate
an auditor re-checks without ever trusting the prover.

</div>

---

## Live prover

**[Open the live prover](https://anwardebes.github.io/veritas/)**, the method running in your browser on a real prompt-injection guard.

https://github.com/user-attachments/assets/3a0f45d6-cfc5-4cac-93de-79925487d50b

Type an attack and the page states a claim: *every prompt that contains these words is blocked*. A SAT solver written for the page checks it against all 2<sup>256</sup> inputs the guard can see. On the shipped [TRACE](https://github.com/AnwarDebes/TRACE) guard it finds a bypass in under a second (*"Which ignore previous instructions"* gets through), repairs the guard with one weighted clause, proves the repaired guard, and re-checks that proof without the solver. Download the formula and the proof and run `drat-trim` on them yourself.

Two more questions no test set can answer: is any word always blocked (none is, padding beats every one), and does the guard block prompts with no attack words at all (it does: *"Your nun"*). Code and tests are in [`live/`](live/).

## Why this exists

A guardrail is a safety component, and a safety component deserves a proof, not a
spot-check. The dominant guardrails (Llama Guard, ShieldGemma, WildGuard,
Constitutional Classifiers) are multi-billion-parameter networks, and they inherit
two properties that should worry anyone deploying them as a _safety_ layer: they are
validated by spot-checking (a benchmark certifies nothing about the inputs not on
it), and they fail silently (under distribution shift they keep answering, confidently
and wrongly, instead of flagging that they are out of their depth).

When the guard is an _interpretable_ Tsetlin Machine, its decision is a Boolean
formula over named features, so it is not merely readable: it is **machine-verifiable**.
A SAT or SMT solver can settle a question about the entire input space in one query:
_is there any input containing this attack phrase that the guard waves through?_ The
answer is a **proof**, or a **concrete counterexample**, never a probability.

> [!NOTE]
> We do **not** claim to invent Tsetlin-Machine verification. The exact TM to Boolean
> encoding and the robustness / equivalence checking are due to
> [Przybysz et al. 2023](https://arxiv.org/abs/2303.14464) (with the Tsetlin Machine's
> inventor as a co-author). VERITAS is the verified **guardrail** system those results
> make possible, and it cites that work as the foundation it builds on.

VERITAS is the fourth pillar of an interpretable agent-safety line:
[THGTM](https://github.com/AnwarDebes/THGTM) (learn), [TRACE](https://github.com/AnwarDebes/TRACE) (detect and certify),
[REINS](https://github.com/AnwarDebes/reins) (enforce), and the piece that makes the guard itself an object of
proof. The accompanying paper is [`paper/veritas.pdf`](paper/veritas.pdf).

## What makes it new (the contribution is the conjunction)

1. **Guardrail property classes.** Beyond robustness and equivalence, the properties a
   deployer actually needs: **no-bypass invariants** (`must_flag`), **trigger-necessity**
   (no alarm without an audited cue, the property a spurious shortcut violates), and
   **monotonicity** as a soundness check. Each compiles to one exact query that decides
   with a proof or a counterexample.
2. **Constructive repair.** You cannot _train_ a guard to a proof: empirical
   counterexample-retraining never reaches, and cannot certify, zero bypasses, and can
   even _lose_ guarantees it had. Because the guard is propositional, VERITAS
   **constructs** the fix instead, the minimal clause set that provably enforces the
   obligation.
3. **Verified-competence envelope.** A verified `trigger_necessity` property defines the
   region where a verdict is backed by an audited cue; outside it the guard is guessing,
   and the envelope **abstains**, behind a _verified_ boundary rather than a softmax
   threshold.
4. **Prover-independent assurance certificate.** A refuted property carries a
   counterexample an auditor re-checks by evaluating the tiny, included guard with plain
   arithmetic: no solver, no trust in ours. The record is tamper-evident via a hash
   chain and a Merkle root.

## Quick start

```bash
pip install -e ".[experiments,data,dev]"

python demo/quickstart.py      # prove a guard, find a bypass, repair it, abstain, certify
python -m pytest               # 31 unit tests (incl. 3-backend agreement + bit-exact encoder)
python experiments/run_all.py  # regenerate every results/*.json (minutes, CPU only)
python scripts/make_figures.py && python scripts/paper_numbers.py
```

```python
from veritaskit import (ContentGuard, GuardEncoding, Verifier, Specification,
                        VeritasEngine, must_flag, trigger_necessity, VerifiedEnvelope,
                        verify_assurance)

guard = ContentGuard().fit(texts, labels)            # an interpretable TM guard
enc   = GuardEncoding.from_guard(guard)              # exact Boolean model of it
V     = Verifier(enc)                                # z3 / pysat / brute-force

V.verify(must_flag(["ignore", "previous"]))          # PROOF or COUNTEREXAMPLE, over all inputs
repaired = enc.with_must_flag([enc.token_feature("ignore")])  # constructive repair
env = VerifiedEnvelope(guard, audited_lexicon, sound=True)    # abstain off-competence

_, ledger, cert = VeritasEngine(guard, key=KEY).assure(spec)  # sealed assurance certificate
ok, info = verify_assurance(ledger, cert, key=KEY)            # auditor re-checks, prover-free
```

## Headline results

All numbers come from `python experiments/run_all.py` on a single CPU, over 8 seeds.

| Experiment | Result |
|---|---|
| **Verifier soundness** (E1) | z3, PySAT and brute force agree on **17/17** cross-checked queries, so a verdict never rests on a single solver |
| **Verification vs red-teaming** (E2) | verification **certifies** the robust tokens (no red team can, at any budget) and finds a bypass for the rest, deciding every case in ~66 ms/token; red-teaming certifies _nothing_ |
| **Constructive repair** (E3) | empirical retraining stalls at **8** provable bypasses (and loses accuracy, 0.69 to 0.57); constructive repair reaches **0** with accuracy preserved (0.69 to 0.71) |
| **Verified abstention** (E4) | under shift, plain accuracy collapses **0.899 to 0.505**, _silently_; the envelope provides a **provably-sound** abstention boundary (sound in **100%** of runs) that escalates silently-missed attacks. Caveat: a coverage-matched uncertified threshold _recovers more_, so the contribution is the **certified boundary**, not the recovery rate |
| **Assurance certificate** (E5) | 6 tamper classes plus forgery detected at **100%**; every counterexample independently re-evaluable with no solver |
| **Systems** (E6) | PySAT decides an invariant in **3.6 ms** vs 51 ms for z3; verification cost grows gracefully with guard size |

## What's in the box

| Path | What it is |
|---|---|
| `veritaskit/tm.py`, `automata.py`, `booleanize.py`, `guard.py` | the interpretable Tsetlin content guard (vendored substrate) |
| `veritaskit/encode.py` | exact TM to Boolean / SMT encoding (after Przybysz et al. 2023) plus constructive repair |
| `veritaskit/properties.py` | the property DSL: invariants, trigger-necessity, robustness radius, monotonicity, equivalence |
| `veritaskit/verify.py` | the verifier: z3, PySAT and brute-force backends, cross-checked |
| `veritaskit/harden.py`, `envelope.py` | empirical hardening vs constructive repair; the verified-competence envelope |
| `veritaskit/ledger.py`, `engine.py` | the tamper-evident, prover-independent assurance certificate |
| `experiments/` | six reproducible experiments (E1 to E6) |
| `live/` | the live prover: a CDCL solver, a RUP proof checker and constructive repair in the browser, on the TRACE guard |
| `paper/veritas.tex` | the paper source |

## Limitations

This is v1.0, and we are explicit about what does **not** hold yet:

- **Feature-space verification.** We verify over booleanised feature vectors (the
  guard's actual input), not raw strings; mapping a counterexample back to a worst-case
  string is future work (optional realizability constraints help).
- **Compact guards.** Verifiability favours small models; our guards trade accuracy for
  provability and do not match billion-parameter detectors. The frontier (E1) makes the
  trade-off explicit, and large-guard queries can exceed a practical budget.
- **Declared properties only.** VERITAS proves what someone specified; `trigger_necessity`
  catches _unspecified_ reliance, but property mining is open.
- **Symmetric authentication.** The certificate is HMAC-based; Ed25519 over the Merkle
  root is the drop-in non-repudiation upgrade.

See [`paper/veritas.pdf`](paper/veritas.pdf) §9 for the full discussion.

## Citing

```bibtex
@misc{debes2026veritas,
  title  = {VERITAS: Formally Verified, Competence-Aware Interpretable
            Guardrails for Language Models},
  author = {Debes, Anwar},
  year   = {2026},
  note   = {Reference implementation, v1.0}
}
```

## License

MIT. See [`LICENSE`](LICENSE). Built on the Tsetlin-Machine verification of
Przybysz et al. (2023); the two datasets are Apache-2.0 and vendored under `data/`.

<br/>

<div align="center">
<img src="assets/logo.svg" width="46" alt="VERITAS"/>
<br/>
<sub><b>VERITAS</b> · a guardrail you can prove, that knows where its proof stops.</sub>
</div>
