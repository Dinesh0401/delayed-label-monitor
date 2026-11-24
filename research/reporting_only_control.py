"""Stress-test the matured detector on reporting slowdown without drift."""

import json

import numpy as np

from research.long_tail_study import run_matured
from research.research_suite import CHANGE, COUNT_CACHE, stream

DAYS = 160
SEEDS = range(11000, 11100)


def evaluate() -> dict:
    rows = []
    for size in (2, 8, 24):
        for regime in ("stable", "reporting_slow_30"):
            alarms = refits = promotions = pending = total = 0
            for seed in SEEDS:
                stable = stream(seed, "stable", size, days=DAYS)
                data = stable
                if regime != "stable":
                    data = [
                        (day, error, np.where((day >= CHANGE) & (arrival - day == 9),
                                              day + 30, arrival), count)
                        for day, error, arrival, count in
                        stream(seed, "reporting_slow", size, days=DAYS)
                    ]
                for original, replay in zip(stable, data):
                    np.testing.assert_array_equal(original[1], replay[1])
                COUNT_CACHE.clear()
                try:
                    # This null scenario counts every alarm over the horizon.
                    result = run_matured(data, regime if regime == "stable" else
                                         "reporting_slow", days=DAYS)
                finally:
                    COUNT_CACHE.clear()
                alarms += result["false_alarm"]
                refits += result["refits"]
                promotions += result["promotions"]
                day, _, arrival, _ = data[1]
                cohort = (day >= CHANGE) & (day < CHANGE + 28)
                pending += int((cohort & (arrival > CHANGE + 28)).sum())
                total += int(cohort.sum())
            rows.append(dict(minority_per_day=size, regime=regime,
                             false_alarm_runs=alarms, refits=refits,
                             promotions=promotions, pending_fraction=pending / total))
    return dict(days=DAYS, seed_range=[SEEDS.start, SEEDS.stop - 1], rows=rows)


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
