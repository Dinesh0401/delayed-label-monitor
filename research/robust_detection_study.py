"""Ablate null-regime calibration without changing either detector's scores."""
import json
import math
from pathlib import Path
from statistics import mean
from research.detection_study import scores, METHODS, CHANGE, DAYS

CALIBRATION_SEEDS = range(2000,2100)
TEST_SEEDS = range(3000,3300)
SCENARIOS = ('stable','delay','harm','both')


def empirical_threshold(maxima, alpha=.05):
    """Conservative finite-sample order statistic; alarms use strict >."""
    rank = math.ceil((len(maxima)+1)*(1-alpha))
    if rank > len(maxima):
        raise ValueError('Not enough calibration runs for this alpha')
    return sorted(maxima)[rank-1]


def summarize(histories, threshold, harmful):
    false = 0
    delays = []
    eligible = 0
    capped_delays = []
    for history in histories:
        crossings = [day for day, value in enumerate(history,12) if value > threshold]
        before = any(day < CHANGE for day in crossings)
        false += before if harmful else bool(crossings)
        if harmful and not before:
            eligible += 1
            hits = [day for day in crossings if day >= CHANGE]
            if hits:
                delays.append(hits[0]-CHANGE)
            capped_delays.append(hits[0]-CHANGE if hits else DAYS-CHANGE)
    return dict(runs=len(histories), false_alarm_runs=false,
                eligible_runs=eligible if harmful else None,
                detected_runs=len(delays) if harmful else None,
                missed_runs=eligible-len(delays) if harmful else None,
                mean_detection_delay=mean(delays) if delays else None,
                mean_capped_delay=mean(capped_delays) if capped_delays else None)


def run():
    assert set(CALIBRATION_SEEDS).isdisjoint(TEST_SEEDS)
    calibration = {s:[scores(seed,s) for seed in CALIBRATION_SEEDS]
                   for s in ('stable','delay')}
    per_regime = {s:{m:empirical_threshold([max(h[m]) for h in runs])
                     for m in METHODS} for s,runs in calibration.items()}
    thresholds = {'stable_only':per_regime['stable'],
                  'reporting_robust':{m:max(per_regime[s][m] for s in calibration)
                                      for m in METHODS}}
    rows=[]
    for scenario in SCENARIOS:
        runs=[scores(seed,scenario) for seed in TEST_SEEDS]
        for protocol, values in thresholds.items():
            for method, threshold in values.items():
                rows.append(dict(scenario=scenario,protocol=protocol,method=method,
                                 threshold=threshold,
                                 **summarize([r[method] for r in runs],threshold,
                                             scenario in ('harm','both'))))
    return dict(calibration_seeds=[2000,2099],test_seeds=[3000,3299],
                alpha=.05,calibration_rank=96,calibration_regimes=['stable','delay'],
                per_regime_thresholds=per_regime,rows=rows)


if __name__=='__main__':
    result=run()
    (Path(__file__).resolve().parents[1] / 'robust-detection-results.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for row in result['rows']:
        print(row)
