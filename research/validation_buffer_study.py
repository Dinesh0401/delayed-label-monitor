"""Compare fixed prospective validation buffers on paired synthetic streams."""

import json

import numpy as np

from research.long_tail_study import run_matured
from research.research_suite import COUNT_CACHE, stream

DAYS = 160
SEEDS = range(12000, 12100)
BUFFERS = (9, 30, 45)


def evaluate() -> dict:
    """Measure promotion, delay and minority error under fixed buffers."""
    rows = []
    for scenario in ("harm", "long_tail"):
        for size in (2, 8):
            cohorts = {buffer: [] for buffer in BUFFERS}
            for seed in SEEDS:
                data = stream(seed, scenario, size, days=DAYS)
                for buffer in BUFFERS:
                    trace = []
                    COUNT_CACHE.clear()
                    try:
                        result = run_matured(
                            data, scenario, days=DAYS, trace=trace,
                            validation_buffer=buffer,
                        )
                    finally:
                        COUNT_CACHE.clear()
                    promotion_days = [
                        event["day"] for event in trace
                        if event["action"] == "promote" and event["group"] == 1
                    ]
                    cohorts[buffer].append((result, promotion_days))
            for buffer, runs in cohorts.items():
                days = [day for _, promotions in runs for day in promotions]
                rows.append(dict(
                    scenario=scenario, minority_per_day=size, buffer_days=buffer,
                    runs=len(runs), detections=sum(r["detected"] for r, _ in runs),
                    minority_promotions=len(days),
                    mean_promotion_day=float(np.mean(days)) if days else None,
                    minority_error=float(np.mean([r["minority_error"] for r, _ in runs])),
                ))
    return dict(days=DAYS, seed_range=[SEEDS.start, SEEDS.stop - 1], rows=rows)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
