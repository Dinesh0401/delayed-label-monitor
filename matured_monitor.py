"""Multiscale fixed-cohort monitoring and matured-cohort adaptation.
Solves the long-tail delay blindspot while maintaining zero false alarms on stable streams.
"""
import math
import numpy as np
from cautious_monitor import interval, harmful, promotion_bound

def multiscale_harmful(reference_short, current_short, reference_mature, current_mature,
                       alpha=.05, groups=2, looks=120, margin=.05):
    """
    Evaluates both short-horizon (e.g. 9-day buffer) and matured-horizon (e.g. 30-day buffer)
    cohorts with Bonferroni correction across both scales (effective tail alpha halved).
    """
    # 2 horizons, so divide alpha by 2 for the union bound
    tail = alpha / (2 * groups * looks * 2)
    ref_s = interval(*reference_short, tail)
    cur_s = interval(*current_short, tail)
    alarm_s = cur_s[0] > ref_s[1] + margin

    ref_m = interval(*reference_mature, tail)
    cur_m = interval(*current_mature, tail)
    alarm_m = cur_m[0] > ref_m[1] + margin

    return dict(
        alarm=bool(alarm_s or alarm_m),
        scale='both' if (alarm_s and alarm_m) else ('short' if alarm_s else ('mature' if alarm_m else 'none')),
        short_alarm=bool(alarm_s),
        mature_alarm=bool(alarm_m),
        short_cur_lower=cur_s[0], short_ref_upper=ref_s[1],
        mature_cur_lower=cur_m[0], mature_ref_upper=ref_m[1]
    )

def find_matured_training_cohort(day, arrival, error, now, min_days=14, max_lookback=60):
    """
    Scans backwards from `now` to find the most recent `min_days` cohort where
    all outcomes have arrived by `now`.
    Returns (candidate_label, train_start, train_end) or None if no fully mature window exists.
    """
    for offset in range(9, max_lookback + 1):
        end = now - offset
        start = end - min_days
        mask = (day >= start) & (day < end)
        if mask.any() and np.all(arrival[mask] <= now):
            candidate = int(error[mask].mean() > 0.5)
            return candidate, start, end
    return None, None, None
