"""Independent null calibration and test evaluation of delayed-error detection."""
import random, math, csv, json
from pathlib import Path
from statistics import mean
from drift_policy import posterior

DAYS, CHANGE, WINDOW = 70, 30, 7
METHODS = ("arrived", "censored")
def scores(seed, scenario):
    rng=random.Random(seed)
    events=[]
    for day in range(DAYS):
        for _ in range(12):
            error=rng.random() < (.38 if scenario in ("harm","both") and day>=CHANGE else .12)
            p=.3 if error else .8
            if scenario in ("delay","both") and day>=CHANGE:
                p=.08 if error else .45
            events.append((day,error,1 if rng.random()<p else 9))
    history={m:[] for m in METHODS}
    for day in range(12,DAYS):
        recent=[e for e in events if day-WINDOW<=e[0]<day]
        observed=[e for e in recent if e[0]+e[2]<=day]
        errors=sum(e[1] for e in observed)
        history["arrived"].append(errors/len(observed) if observed else 0)
        # In this window age <=7, so only early arrivals can be observed.
        history["censored"].append(posterior(errors,len(observed)-errors,len(recent)-len(observed),.3,.8)[0])
    return history

def main():
    calibration=[scores(seed,"stable") for seed in range(100)]
    thresholds={m:sorted(max(s[m]) for s in calibration)[94] for m in METHODS}
    rows=[]
    for scenario in ("stable","delay","harm","both"):
        runs=[scores(seed,scenario) for seed in range(1000,1100)]
        for method in METHODS:
            false=0; delays=[]; eligible=0
            for run in runs:
                crossings=[i+12 for i,x in enumerate(run[method]) if x>thresholds[method]]
                before=any(t<CHANGE for t in crossings)
                if scenario in ("stable","delay"):
                    false+=bool(crossings)
                else:
                    false+=before
                    if not before:
                        eligible+=1
                        hits=[t for t in crossings if t>=CHANGE]
                        if hits:delays.append(hits[0]-CHANGE)
            rows.append(dict(scenario=scenario,method=method,threshold=thresholds[method],
                             false_alarm_runs=false,runs=100,eligible_runs=eligible,
                             detected_runs=len(delays),mean_detection_delay=mean(delays) if delays else None))
    target=Path(__file__).resolve().parents[1]/"research/legacy/results/detection-results.json"
    target.write_text(json.dumps(dict(calibration_seeds=[0,99],test_seeds=[1000,1099],rows=rows),indent=2))
    print(target.read_text())
if __name__=="__main__":main()
