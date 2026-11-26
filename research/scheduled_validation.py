"""Compare fixed waits and a pre-specified sequence of validation looks."""

import json

import numpy as np

from research.long_tail_study import run_matured
from research.research_suite import COUNT_CACHE, stream

DAYS = 160
SEEDS = range(13000, 13100)
SCHEDULES = {"fixed_9": (9,), "fixed_30": (30,), "scheduled": (9, 16, 23, 30)}


def evaluate() -> dict:
    """Score all policies on the same streams for each seed."""
    rows = []
    for scenario in ("harm", "long_tail"):
        for size in (2, 8):
            samples = {name: [] for name in SCHEDULES}
            for seed in SEEDS:
                data = stream(seed, scenario, size, days=DAYS)
                for name, waits in SCHEDULES.items():
                    trace = []
                    COUNT_CACHE.clear()
                    try:
                        result = run_matured(
                            data, scenario, days=DAYS, trace=trace, validation_schedule=waits,
                        )
                    finally:
                        COUNT_CACHE.clear()
                    promotion_day = next(
                        (event["day"] for event in trace
                         if event["action"] == "promote" and event["group"] == 1),
                        None,
                    )
                    samples[name].append((result["minority_error"], promotion_day))
            for name, runs in samples.items():
                days = [day for _, day in runs if day is not None]
                rows.append(dict(
                    scenario=scenario, minority_per_day=size, policy=name,
                    minority_error=float(np.mean([error for error, _ in runs])),
                    minority_promotions=len(days),
                    mean_promotion_day=float(np.mean(days)) if days else None,
                ))
    return dict(days=DAYS, seeds=[SEEDS.start, SEEDS.stop - 1],
                schedules=SCHEDULES, rows=rows)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
