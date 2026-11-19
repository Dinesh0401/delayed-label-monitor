"""Targeted study evaluating multiscale cohort monitoring and matured shadow adaptation
on long-delay streams (seeds 7000-7099).
"""
import argparse, csv, gzip, json
from pathlib import Path
import numpy as np
from cautious_monitor import promotion_bound
from research_suite import stream, counts, COUNT_CACHE, DAYS, WARM, CHANGE, WINDOW, AGE
from matured_monitor import multiscale_harmful, find_matured_training_cohort

MATURE_AGE = 30  # Matured cohort buffer for long-tail delays

def run_matured(data, scenario, trace=None):
    """
    Runs the matured shadow policy that uses:
    1. Multiscale detection (short age=9 and mature age=30)
    2. Backward scan to find fully matured training cohorts
    3. Prospective validation with adaptive holdout
    """
    model = [0, 0]
    attempts = [0, 0]
    promotions = [0, 0]
    loss = [0, 0]
    alarms = [[], []]
    unchanged = 0
    shadow = [None, None]
    last_attempt = [-100, -100]

    for t in range(DAYS):
        for g in range(2):
            # Short cohort (AGE=9, WINDOW=28)
            ref_s = counts(data[g], -WARM, 0, t)
            cur_s = counts(data[g], t - AGE - WINDOW, t - AGE, t)

            # Mature cohort (MATURE_AGE=30, WINDOW=28)
            ref_m = counts(data[g], -WARM, 0, t)
            cur_m = counts(data[g], t - MATURE_AGE - WINDOW, t - MATURE_AGE, t)

            ms = multiscale_harmful(ref_s, cur_s, ref_m, cur_m, groups=2, looks=DAYS)
            flag = ms['alarm']

            if flag and not promotions[g]:
                alarms[g].append(t)

            day, error, arrival, n = data[g]

            # 1. Check prospective shadow model resolution
            if shadow[g] is not None:
                candidate, opened = shadow[g]
                # Validate after 14 days of prospective data + AGE buffer
                # In long tail scenario, wait up to MATURE_AGE or at least 14 + MATURE_AGE
                val_buffer = MATURE_AGE if scenario == 'long_tail' else AGE
                if t >= opened + 14 + val_buffer:
                    valid = (day >= opened) & (day < opened + 14)
                    seen = valid & (arrival <= t)
                    gain = int(((error[seen] != model[g]).astype(int) - (error[seen] != candidate).astype(int)).sum())
                    pending = int(valid.sum() - seen.sum())
                    lower = promotion_bound(gain, pending, int(valid.sum()), .05 / (2 * DAYS * 2 * 2))
                    if lower > .02:
                        model[g] = candidate
                        promotions[g] += 1
                        if trace is not None:
                            trace.append(dict(action='promote', group=g, day=t, opened=opened))
                    shadow[g] = None

            # 2. On alarm, train candidate on the most recent matured cohort
            if flag and shadow[g] is None and attempts[g] < 2 and not promotions[g] and t - last_attempt[g] >= 28:
                candidate, t_start, t_end = find_matured_training_cohort(day, arrival, error, t, min_days=14, max_lookback=45)
                if candidate is not None:
                    attempts[g] += 1
                    last_attempt[g] = t
                    if candidate != model[g]:
                        shadow[g] = (candidate, t)
                        if trace is not None:
                            trace.append(dict(action='train', group=g, day=t, window=(t_start, t_end)))
                    else:
                        unchanged += 1

            loss[g] += int(np.sum(error[(t + WARM) * n:(t + WARM + 1) * n] != model[g]))

    pre = any(t < CHANGE for a in alarms for t in a)
    hits = [t for t in alarms[1] if t >= CHANGE]
    null = scenario in ('stable', 'reporting_slow', 'reporting_fast')
    return dict(
        false_alarm=int(bool(any(alarms))) if null else int(pre),
        eligible=int(not null and not pre),
        detected=int(not null and not pre and bool(hits)),
        capped_delay=(hits[0] - CHANGE if hits else DAYS - CHANGE) if not null and not pre else None,
        majority_error=loss[0] / (DAYS * data[0][3]),
        minority_error=loss[1] / (DAYS * data[1][3]),
        refits=sum(attempts),
        promotions=sum(promotions),
        unchanged_refits=unchanged
    )

def main():
    print("Running Long-Tail & Multiscale Study on seeds 7000-7099...")
    seeds = range(7000, 7100)
    scenarios = ('stable', 'long_tail', 'both', 'harm')
    minorities = (2, 8, 24)

    results = []
    for n in minorities:
        for s in scenarios:
            matured_runs = []
            for seed in seeds:
                COUNT_CACHE.clear()
                data = stream(seed, s, n)
                res = run_matured(data, s)
                matured_runs.append(res)

            false_alarms = sum(r['false_alarm'] for r in matured_runs)
            eligible = sum(r['eligible'] for r in matured_runs)
            detections = sum(r['detected'] for r in matured_runs)
            mean_delay = float(np.mean([r['capped_delay'] for r in matured_runs if r['capped_delay'] is not None])) if eligible > 0 else None
            mean_min_err = float(np.mean([r['minority_error'] for r in matured_runs]))
            mean_refits = float(np.mean([r['refits'] for r in matured_runs]))
            mean_promotions = float(np.mean([r['promotions'] for r in matured_runs]))

            results.append(dict(
                minority_per_day=n,
                scenario=s,
                runs=len(matured_runs),
                false_alarms=false_alarms,
                eligible=eligible,
                detections=detections,
                mean_capped_delay=mean_delay,
                minority_error=mean_min_err,
                refits=mean_refits,
                promotions=mean_promotions
            ))
            print(f"n={n:2d} s={s:10s} | FA: {false_alarms:2d} | Det: {detections:3d}/{eligible:3d} | Delay: {mean_delay} | MinErr: {mean_min_err:.3f} | Prom: {mean_promotions:.2f}")

    out_file = Path('research/results/longtail-summary.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(dict(protocol='multiscale_long_tail', seeds=[7000, 7099], rows=results), indent=2) + '\n')
    print(f"Saved results to {out_file}")

if __name__ == '__main__':
    main()
