"""Run every VERITAS experiment end-to-end and regenerate results/*.json.

    python -u experiments/run_all.py            # full
    python -u experiments/run_all.py --quick    # fast smoke
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import (verification_frontier, bypass_discovery, hardening,
                         envelope_shift, assurance_cert, systems)


def main():
    quick = "--quick" in sys.argv
    fast = (0, 1, 2)
    steps = [
        ("E1 verification frontier", lambda: verification_frontier.main(
            seeds=fast if quick else verification_frontier.SEEDS)),
        ("E2 bypass discovery", lambda: bypass_discovery.main(
            seeds=fast if quick else bypass_discovery.SEEDS)),
        ("E3 counterexample-guided hardening", lambda: hardening.main(
            seeds=fast if quick else hardening.SEEDS)),
        ("E4 verified-envelope abstention", lambda: envelope_shift.main(
            seeds=fast if quick else envelope_shift.SEEDS)),
        ("E5 assurance certificate", lambda: assurance_cert.main(
            n_sessions=8 if quick else 24)),
        ("E6 systems characterisation", lambda: systems.main(quick=quick)),
    ]
    t0 = time.perf_counter()
    for name, fn in steps:
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        s = time.perf_counter()
        fn()
        print(f"[{name}] done in {time.perf_counter()-s:.1f}s")
    print(f"\nALL EXPERIMENTS done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
