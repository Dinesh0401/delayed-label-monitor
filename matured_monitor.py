"""Multiscale fixed-cohort monitoring and matured-cohort adaptation.

Resolves the long-tail delay blindspot by evaluating both short-horizon (e.g. 9-day buffer)
and mature-horizon (e.g. 30-day buffer) cohorts simultaneously using union-bound family-wise
error correction across time horizons.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
from cautious_monitor import interval


def multiscale_harmful(
    reference_short: tuple[int, int, int],
    current_short: tuple[int, int, int],
    reference_mature: tuple[int, int, int],
    current_mature: tuple[int, int, int],
    alpha: float = 0.05,
    groups: int = 2,
    looks: int = 120,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Evaluate both short-horizon and matured-horizon cohorts with Bonferroni correction.

    Splits the significance level across the two monitoring scales to maintain the
    joint family-wise error guarantee across all looks and groups.

    Args:
        reference_short: (errors, pending, total) for the mature baseline reference cohort.
        current_short: (errors, pending, total) for the recent 9-day buffered cohort.
        reference_mature: (errors, pending, total) for the mature baseline reference cohort.
        current_mature: (errors, pending, total) for the 30-day buffered matured cohort.
        alpha: Overall family-wise error rate threshold (default: 0.05).
        groups: Number of monitored groups (default: 2).
        looks: Scheduled monitoring decision epochs (default: 120).
        margin: Required error excess above reference upper bound (default: 0.05).

    Returns:
        Dictionary detailing alarm status, triggering scale ('short', 'mature', 'both', 'none'),
        and respective interval bounds.
    """
    # Allocate alpha across 2 horizon scales: tail_alpha = alpha / (2 * groups * looks * 2)
    tail_alpha = alpha / (2.0 * groups * looks * 2.0)

    ref_s = interval(*reference_short, tail_alpha)
    cur_s = interval(*current_short, tail_alpha)
    alarm_s = cur_s[0] > (ref_s[1] + margin)

    ref_m = interval(*reference_mature, tail_alpha)
    cur_m = interval(*current_mature, tail_alpha)
    alarm_m = cur_m[0] > (ref_m[1] + margin)

    triggering_scale = (
        "both" if (alarm_s and alarm_m)
        else ("short" if alarm_s else ("mature" if alarm_m else "none"))
    )

    return {
        "alarm": bool(alarm_s or alarm_m),
        "scale": triggering_scale,
        "short_alarm": bool(alarm_s),
        "mature_alarm": bool(alarm_m),
        "short_cur_lower": cur_s[0],
        "short_ref_upper": ref_s[1],
        "mature_cur_lower": cur_m[0],
        "mature_ref_upper": ref_m[1],
    }


def find_matured_training_cohort(
    day: np.ndarray,
    arrival: np.ndarray,
    error: np.ndarray,
    now: int,
    min_days: int = 14,
    max_lookback: int = 60,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Scan backwards from the current decision epoch to identify a fully observed cohort.

    Searches for the most recent window of length `min_days` where every prediction's
    outcome has arrived at or before `now`.

    Args:
        day: 1D array of prediction event creation days.
        arrival: 1D array of label arrival days.
        error: 1D boolean array indicating whether prediction was incorrect.
        now: Current decision epoch day.
        min_days: Duration in days of the candidate training window (default: 14).
        max_lookback: Maximum days to scan backwards from `now` (default: 60).

    Returns:
        A tuple of (candidate_rule, start_day, end_day) if a fully mature cohort is found,
        or (None, None, None) if no window has all outcomes resolved.
    """
    for offset in range(9, max_lookback + 1):
        window_end = now - offset
        window_start = window_end - min_days
        mask = (day >= window_start) & (day < window_end)

        if mask.any() and np.all(arrival[mask] <= now):
            # Candidate rule: 1 if error rate > 50% (inversion), else 0 (identity)
            candidate = int(error[mask].mean() > 0.5)
            return (candidate, window_start, window_end)

    return (None, None, None)
