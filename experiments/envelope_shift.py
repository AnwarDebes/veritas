"""E4, Verified-envelope abstention under distribution shift.

A learned guard's most dangerous failure is the *silent* one: trained on one
corpus and run on another it keeps emitting confident verdicts on inputs it has no
basis to judge.  We reproduce that here, a guard trained on jailbreak text is
accurate in-domain but collapses toward the base rate on prompt-injection text, and show the verified envelope turns the silent failure into auditable deferral.

If ``trigger_necessity(U)`` is verified, then on any input with no audited cue the
guard emits a constant SAFE verdict by absence of evidence (no positive
discrimination); the envelope ABSTAINS (escalates) on exactly that region, which is
where the guard's only failure mode, a silently-missed attack, can hide.  Its
value is converting those silent misses into auditable escalations behind a
*formally verified* (and thus certifiable) abstention boundary.  We are explicit
about what this does NOT buy: an uncertified margin-threshold selective baseline
(Geifman & El-Yaniv 2017), coverage-matched on the OOD set, recovers a comparable or
larger fraction of silent misses, because that metric simply rewards abstaining
more.  The contribution is the certification of the boundary, not a higher recovery
rate; we report the coverage-matched baseline so the comparison is honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import load_dataset, summarize, write_results, host_info
from experiments._vh import train_guard, positive_literal_tokens
from veritaskit import GuardEncoding, Verifier, trigger_necessity, VerifiedEnvelope, ABSTAIN

SEEDS = tuple(range(8))


def margin_selective(guard, texts, labels, tau):
    labels = np.asarray(labels)
    vs = [guard.verdict(t) for t in texts]
    answered = [(v, y) for v, y in zip(vs, labels) if abs(v.margin) > tau]
    cov = len(answered) / max(1, len(vs))
    sel = (sum(int(v.unsafe == bool(y)) for v, y in answered) / len(answered)) if answered else float("nan")
    missed = [(v, y) for v, y in zip(vs, labels) if y == 1 and not v.unsafe]
    rec = sum(int(abs(v.margin) <= tau) for v, _ in missed)
    return {"coverage": cov, "selective_acc": sel,
            "missed_recovered": (rec / len(missed)) if missed else float("nan")}


def main(seeds=SEEDS):
    jb = load_dataset("jailbreak"); pi = load_dataset("prompt_injection")
    Xtr, ytr = jb["train"]; Xin, yin = jb["test"]
    Xout, yout = pi["test"]
    out = {"host": host_info(), "seeds": list(seeds), "in_domain": "jailbreak",
           "out_domain": "prompt_injection", "per_seed": []}
    agg = {k: [] for k in ("plain_in", "plain_out", "sound",
                           "in_acc_cue", "in_acc_nocue", "out_acc_cue", "out_acc_nocue",
                           "out_nocue_frac", "out_cov", "out_sel", "out_missed_rec",
                           "base_out_cov", "base_out_sel", "base_out_missed_rec",
                           "base_matched_cov", "base_matched_missed_rec")}

    for seed in seeds:
        g = train_guard(Xtr, ytr, C=40, F=64, T=20, s=4.0, epochs=40, seed=seed)
        enc = GuardEncoding.from_guard(g)
        U = positive_literal_tokens(enc)
        sound = Verifier(enc).verify(trigger_necessity(U)).status == "VERIFIED"
        env = VerifiedEnvelope(g, U, sound=sound)

        st_in = env.stratified(Xin, yin)
        st_out = env.stratified(Xout, yout)
        rc_in = env.risk_coverage(Xin, yin)
        rc_out = env.risk_coverage(Xout, yout)

        # margin baseline tuned to the envelope's IN-domain coverage (its natural
        # operating point), then evaluated OOD, where its coverage diverges.
        margins = sorted(abs(g.verdict(t).margin) for t in Xin)
        tau = margins[min(len(margins) - 1, int((1 - rc_in["coverage"]) * len(margins)))] if margins else 0
        base = margin_selective(g, Xout, yout, tau)
        # FAIR comparison: a margin baseline coverage-matched to the envelope ON THE
        # OOD set itself. missed_recovered is monotone in the abstention rate, so an
        # apples-to-apples comparison must match OOD coverage. This baseline tends to
        # recover MORE, the envelope's contribution is the *certified* boundary, not
        # a higher recovery rate.
        mo = sorted(abs(g.verdict(t).margin) for t in Xout)
        tau_m = mo[min(len(mo) - 1, int((1 - rc_out["coverage"]) * len(mo)))] if mo else 0
        base_m = margin_selective(g, Xout, yout, tau_m)

        agg["plain_in"].append(rc_in["plain_acc"]); agg["plain_out"].append(rc_out["plain_acc"])
        agg["sound"].append(int(sound))
        agg["in_acc_cue"].append(st_in["acc_has_cue"]); agg["in_acc_nocue"].append(st_in["acc_no_cue"])
        agg["out_acc_cue"].append(st_out["acc_has_cue"]); agg["out_acc_nocue"].append(st_out["acc_no_cue"])
        agg["out_nocue_frac"].append(st_out["no_cue_frac"])
        agg["out_cov"].append(rc_out["coverage"]); agg["out_sel"].append(rc_out["selective_acc"])
        agg["out_missed_rec"].append(rc_out["missed_recovered"])
        agg["base_out_cov"].append(base["coverage"]); agg["base_out_sel"].append(base["selective_acc"])
        agg["base_out_missed_rec"].append(base["missed_recovered"])
        agg["base_matched_cov"].append(base_m["coverage"])
        agg["base_matched_missed_rec"].append(base_m["missed_recovered"])
        out["per_seed"].append({"seed": seed, "sound": sound, "strat_in": st_in, "strat_out": st_out,
                                "rc_in": rc_in, "rc_out": rc_out, "margin_baseline_out": base,
                                "margin_baseline_matched": base_m})
        print(f"seed {seed}: plain acc in {rc_in['plain_acc']:.3f}->out {rc_out['plain_acc']:.3f} | "
              f"out no-cue acc={st_out['acc_no_cue']:.3f} (base {st_out['base_rate']:.2f}) "
              f"has-cue acc={st_out['acc_has_cue']:.3f} | env out cov={rc_out['coverage']:.2f} "
              f"missed_rec={rc_out['missed_recovered']:.2f} sound={sound}")

    def S(k):
        vals = [v for v in agg[k] if not (isinstance(v, float) and np.isnan(v))]
        return summarize(vals)
    out["summary"] = {k: S(k) for k in agg}
    s = out["summary"]
    print(f"[summary] plain acc collapses {s['plain_in']['mean']:.3f}->{s['plain_out']['mean']:.3f}; "
          f"OOD no-cue acc {s['out_acc_nocue']['mean']:.3f} vs has-cue {s['out_acc_cue']['mean']:.3f}; "
          f"envelope recovers {s['out_missed_rec']['mean']*100:.0f}% of silent misses")
    write_results("envelope_shift", out)
    return out


if __name__ == "__main__":
    main(seeds=(0, 1, 2) if "--quick" in sys.argv else SEEDS)
