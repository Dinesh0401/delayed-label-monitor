"""Compare matured shadow adaptation with a frozen rule on identical streams."""

from __future__ import annotations

import json

import numpy as np

from research.long_tail_study import run_matured
from research.research_suite import COUNT_CACHE, stream

DAYS = 160
SEEDS = range(10000, 10100)


def evaluate() -> dict:
    """Return paired error differences and seed-level bootstrap intervals."""
    rows = []
    for scenario in ("stable", "long_tail"):
        for size in (2, 8, 24):
            scores = []
            runs = []
            for seed in SEEDS:
                data = stream(seed, scenario, size, days=DAYS)
                days, errors, _, _ = data[1]
                baseline = float(errors[(days >= 0) & (days < DAYS)].mean())
                COUNT_CACHE.clear()
                try:
                    adaptive = run_matured(data, scenario, days=DAYS)
                finally:
                    COUNT_CACHE.clear()
                scores.append((baseline, adaptive["minority_error"]))
                runs.append(adaptive)

            frozen, adaptive = np.asarray(scores).T
            gain = 100 * (frozen - adaptive)
            rng = np.random.default_rng(71)
            indices = rng.integers(len(gain), size=(5000, len(gain)))
            interval = np.quantile(gain[indices].mean(axis=1), [.025, .975])
            rows.append(dict(
                scenario=scenario, minority_per_day=size, runs=len(scores),
                frozen_error=float(frozen.mean()), adaptive_error=float(adaptive.mean()),
                gain_pp=float(gain.mean()), paired_bootstrap_95_pp=interval.tolist(),
                detections=sum(r["detected"] for r in runs),
                false_alarm_runs=sum(r["false_alarm"] for r in runs),
                refits=sum(r["refits"] for r in runs),
                promotions=sum(r["promotions"] for r in runs),
            ))
    return {"days": DAYS, "seed_range": [SEEDS.start, SEEDS.stop - 1], "rows": rows}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
