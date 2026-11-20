"""Conservative fixed-cohort bounds and non-parametric drift detection.

Provides Hoeffding-based interval estimation for cohort errors where outcomes
may still be pending, guaranteeing finite-horizon family-wise error control.
"""

from __future__ import annotations
import math
from typing import Any


def interval(errors: int, pending: int, total: int, alpha: float) -> tuple[float, float]:
    """Calculate the conservative confidence interval for cohort accuracy.

    Uses Hoeffding's inequality on fixed-cohort predictions to bound empirical error
    without requiring delay-distribution estimates or stationarity assumptions.

    Args:
        errors: Number of confirmed incorrect predictions in the cohort.
        pending: Number of predictions whose ground-truth outcome is unresolved.
        total: Total number of predictions scheduled in the fixed cohort.
        alpha: One-sided failure probability threshold (0 < alpha < 1).

    Returns:
        A tuple of (lower_bound, upper_bound) in the unit interval [0, 1].

    Raises:
        ValueError: If counts are negative, errors + pending > total, or alpha is invalid.
    """
    if not (0.0 < alpha < 1.0) or min(errors, pending, total) < 0 or errors + pending > total:
        raise ValueError("Invalid counts or alpha threshold")

    if total == 0:
        return (0.0, 1.0)

    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * total))
    lower = max(0.0, (errors / total) - radius)
    upper = min(1.0, ((errors + pending) / total) + radius)
    return (lower, upper)


def harmful(
    reference: tuple[int, int, int],
    current: tuple[int, int, int],
    alpha: float = 0.05,
    groups: int = 2,
    looks: int = 120,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Test whether current cohort error significantly exceeds the reference cohort.

    Applies Bonferroni allocation across groups, decision looks, and tail sides.

    Args:
        reference: Tuple of (errors, pending, total) for the mature reference cohort.
        current: Tuple of (errors, pending, total) for the recent evaluation cohort.
        alpha: Family-wise significance level across all comparisons (default: 0.05).
        groups: Number of monitored subgroups (default: 2).
        looks: Predetermined decision epochs across the monitoring horizon (default: 120).
        margin: Safety margin required above reference upper bound (default: 0.05).

    Returns:
        Dictionary containing alarm flag, interval bounds, and margin parameters.

    Raises:
        ValueError: If groups or looks are non-positive.
    """
    if groups < 1 or looks < 1:
        raise ValueError("Positive groups and looks counts are required")

    tail_alpha = alpha / (2.0 * groups * looks)
    ref_interval = interval(*reference, tail_alpha)
    cur_interval = interval(*current, tail_alpha)

    alarm = cur_interval[0] > (ref_interval[1] + margin)

    return {
        "alarm": alarm,
        "reference_upper": ref_interval[1],
        "current_lower": cur_interval[0],
        "current_upper": cur_interval[1],
        "margin": margin,
    }


def promotion_bound(
    observed_gain: int,
    pending: int,
    total: int,
    alpha: float,
) -> float:
    """Calculate conservative lower bound on prospective paired model improvement.

    Each paired loss difference (old_loss - candidate_loss) lies in [-1, 1].
    Pending labels are imputed with the adversarial worst-case difference of -1.

    Args:
        observed_gain: Sum of paired improvements on resolved validation events.
        pending: Number of unresolved validation events.
        total: Total scheduled predictions in the prospective validation cohort.
        alpha: One-sided failure probability allocated to the promotion test.

    Returns:
        Conservative lower bound on paired error gain.

    Raises:
        ValueError: If inputs violate paired validation constraints or alpha bounds.
    """
    if (
        total <= 0
        or pending < 0
        or pending > total
        or abs(observed_gain) > total - pending
        or not (0.0 < alpha < 1.0)
    ):
        raise ValueError("Invalid paired validation data or alpha")

    # Worst-case imputation: each pending outcome contributes -1 to gain
    worst_case_gain = observed_gain - pending
    radius = math.sqrt(2.0 * math.log(1.0 / alpha) / total)
    lower_bound = (worst_case_gain / total) - radius

    return max(-1.0, lower_bound)
