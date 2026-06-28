"""Generate every VERITAS paper figure from results/*.json.

Run after experiments/run_all.py.  Each figure is guarded so a missing result is
skipped, not fatal."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
TEAL, RED, GREY, AMBER = "#0d9488", "#dc2626", "#94a3b8", "#d97706"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure {name}")


def fig_frontier():
    d = load("verification_frontier")
    if not d:
        return
    b = d["budgets"]
    sizes = [f"{x['clauses']}x{x['features']}" for x in b]
    acc = [x["accuracy"]["mean"] for x in b]
    us = [x["verify_us_per_query"]["mean"] / 1000 for x in b]
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    xs = np.arange(len(b))
    ax1.plot(xs, acc, "s--", color=RED, label="test accuracy")
    ax1.set_ylabel("test accuracy", color=RED); ax1.tick_params(axis="y", labelcolor=RED)
    ax1.set_ylim(0.5, 0.85); ax1.set_xticks(xs); ax1.set_xticklabels(sizes, rotation=15)
    ax1.set_xlabel("guard size (clauses x features)")
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(xs, us, "o-", color=TEAL, label="verify ms/query")
    ax2.set_ylabel("verify ms / query", color=TEAL); ax2.tick_params(axis="y", labelcolor=TEAL)
    ax1.set_title("Interpretability / verifiability / accuracy frontier")
    save(fig, "frontier")


def fig_scaling():
    d = load("systems")
    if not d:
        return
    sc = d["scale_clauses"]
    cs = [r["clauses"] for r in sc]
    us = [r["verify_us"] / 1000 for r in sc]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.plot(cs, us, "o-", color=TEAL)
    ax1.set_xlabel("clauses"); ax1.set_ylabel("verify ms / query")
    ax1.set_title("Verification cost vs clause budget")
    bc = d["backend_compare"]
    ax2.bar([0, 1], [bc["z3_us"] / 1000, bc["pysat_us"] / 1000],
            color=[GREY, TEAL], edgecolor="black", linewidth=0.5)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["z3 (SMT)", "PySAT (SAT)"])
    ax2.set_ylabel("ms / invariant"); ax2.set_title("Backend gap")
    for x, v in zip([0, 1], [bc["z3_us"] / 1000, bc["pysat_us"] / 1000]):
        ax2.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    save(fig, "scaling")


def fig_bypass():
    d = load("bypass_discovery")
    if not d:
        return
    s = d["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.bar([0, 1], [s["frac_certified_robust"]["mean"], s["frac_bypassable"]["mean"]],
            color=[TEAL, RED], edgecolor="black", linewidth=0.5)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["certified\nrobust", "bypass\nfound"])
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("fraction of attack tokens")
    ax1.set_title("Verification decides every obligation")
    ns = d["probe_budgets"]
    fr = [s["fuzz_recall_of_bypasses"][str(n)]["mean"] for n in ns]
    ax2.plot(ns, fr, "o-", color=AMBER, label="red-team recall of bypasses")
    ax2.axhline(1.0, color=TEAL, ls="--", lw=1.5, label="verification (certifies robust too)")
    ax2.set_xscale("log"); ax2.set_xlabel("red-team probes"); ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("recall of bypasses"); ax2.legend(fontsize=8)
    ax2.set_title("Red-teaming certifies nothing robust")
    save(fig, "bypass")


def fig_harden():
    d = load("hardening")
    if not d:
        return
    s = d["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    vals = [s["bypass_initial"]["mean"], s["bypass_empirical"]["mean"], s["bypass_constructive"]["mean"]]
    ax1.bar([0, 1, 2], vals, color=[GREY, AMBER, TEAL], edgecolor="black", linewidth=0.5)
    ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(["initial", "empirical\nretrain", "constructive\nrepair"])
    ax1.set_ylabel("provable bypasses"); ax1.set_title("Only constructive repair reaches zero")
    for x, v in zip([0, 1, 2], vals):
        ax1.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    accs = [s["acc_initial"]["mean"], s["acc_empirical"]["mean"], s["acc_constructive"]["mean"]]
    ax2.bar([0, 1, 2], accs, color=[GREY, AMBER, TEAL], edgecolor="black", linewidth=0.5)
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["initial", "empirical", "constructive"])
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("test accuracy"); ax2.set_title("...without losing accuracy")
    save(fig, "harden")


def fig_envelope():
    d = load("envelope_shift")
    if not d:
        return
    s = d["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.bar([0, 1], [s["plain_in"]["mean"], s["plain_out"]["mean"]],
            color=[TEAL, RED], edgecolor="black", linewidth=0.5)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["in-domain", "shifted (OOD)"])
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("plain guard accuracy")
    ax1.set_title("Silent collapse under shift")
    for x, v in zip([0, 1], [s["plain_in"]["mean"], s["plain_out"]["mean"]]):
        ax1.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    bm = s.get("base_matched_missed_rec", s["base_out_missed_rec"])["mean"]
    env_mr = s["out_missed_rec"]["mean"]
    ax2.bar([0, 1], [env_mr, bm], color=[TEAL, GREY], edgecolor="black", linewidth=0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["verified envelope\n(boundary certified)", "margin baseline\n(matched cov., uncertified)"], fontsize=8)
    ax2.set_ylim(0, max(0.7, bm * 1.3)); ax2.set_ylabel("silent misses recovered (OOD)")
    ax2.set_title("Recovery at matched coverage (honest)")
    for x, v in zip([0, 1], [env_mr, bm]):
        ax2.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.annotate("only this boundary\nis verifiable", xy=(0, env_mr), xytext=(0.15, env_mr + 0.18),
                 fontsize=7.5, color=TEAL, ha="left",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))
    save(fig, "envelope")


def main():
    print("building figures...")
    for fn in (fig_frontier, fig_scaling, fig_bypass, fig_harden, fig_envelope):
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    print("done ->", FIG)


if __name__ == "__main__":
    main()
