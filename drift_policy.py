"""Posterior risk for a cohort evaluated at one fixed reporting horizon."""
import math
from functools import lru_cache


@lru_cache(maxsize=8192)
def posterior(errors, correct, pending, early_error, early_correct):
    """Return posterior mean error and probability that flipping improves by >10 points."""
    if min(errors, correct, pending) < 0 or not all(0 < p < 1 for p in (early_error, early_correct)):
        raise ValueError("Counts must be nonnegative and reporting probabilities interior")
    grid = [(i + .5) / 160 for i in range(160)]
    log_weights = [
        errors * math.log(r) + correct * math.log1p(-r)
        + pending * math.log(r * (1 - early_error) + (1 - r) * (1 - early_correct))
        for r in grid
    ]
    peak = max(log_weights)
    weights = [math.exp(w - peak) for w in log_weights]
    total = sum(weights)
    return (sum(r*w for r,w in zip(grid, weights))/total,
            sum(w for r,w in zip(grid, weights) if r > .55)/total)


def reporting_rate(early, total):
    return (early + 1) / (total + 2)
