"""Paired synthetic stream replay. Standard library, deterministic seeds."""
import csv
import random
from statistics import mean
from drift_policy import posterior, reporting_rate

SIZES = (80, 8)
DAYS = 64
BUDGET = 2
POLICIES = ("frozen", "periodic", "pooled", "arrived_group", "posterior_group")


def stream(seed, scenario):
    rng = random.Random(seed)
    days = []
    for day in range(-16, DAYS):
        cohorts = []
        for g, n in enumerate(SIZES):
            cohort = []
            for _ in range(n):
                x = rng.randrange(2)
                noise = rng.random() < .12
                shift = g == 1 and scenario in ("harm", "both") and day >= 24
                y = x ^ noise ^ shift
                slow = g == 1 and scenario in ("delay", "both") and day >= 20
                p = .88 if y == 1 else (.08 if slow else .30)
                cohort.append((x,y,1 if rng.random() < p else 5))
            cohorts.append(cohort)
        days.append(cohorts)
    return days


def run(days, policy):
    flip = [0,0]
    spend = redundant = 0
    loss = [0,0]
    last = [-20,-20]
    for d in range(16, len(days)):
        # Score predictions only after deciding with data available at day start.
        candidates = []
        for g in range(2):
            if d-last[g] < 5:
                continue
            # Learn outcome-specific reporting from fully resolved prior cohorts.
            history = [row for past in days[d-16:d-4] for row in past[g]]
            visible = [(x,y) for age in range(1,5) for x,y,delay in days[d-age][g] if delay<=age]
            n = 4*SIZES[g]
            wrong = sum((x ^ flip[g]) != y for x,y in visible)
            if policy == "posterior_group":
                # Stratification by x gives a fixed predicted class in each stratum.
                # Use a conservative change rule requiring support in both strata.
                probs = []
                for x in (0,1):
                    pred = x ^ flip[g]
                    rates = []
                    for label in (1-pred,pred):
                        delays = [lag for hx,y,lag in history if hx==x and y==label]
                        rates.append(reporting_rate(delays.count(1),len(delays)))
                    obs = [y for hx,y in visible if hx==x]
                    count = sum(hx==x for age in range(1,5) for hx,_,_ in days[d-age][g])
                    errors = sum(y!=pred for y in obs)
                    probs.append(posterior(errors,len(obs)-errors,count-len(obs),*rates)[1])
                if min(probs)>.90:
                    candidates.append((min(probs),g))
            elif policy == "arrived_group" and visible and wrong/len(visible)>.55:
                candidates.append((wrong/len(visible),g))
            elif policy == "periodic" and d-16 in (16,32):
                candidates.append((1,g))
            elif policy == "pooled":
                pooled = [(x ^ flip[h],y) for h in range(2) for age in range(1,5)
                          for x,y,lag in days[d-age][h] if lag<=age]
                if pooled and mean(p!=y for p,y in pooled)>.55:
                    candidates.append((wrong/max(1,len(visible)),g))
        if candidates and spend<BUDGET:
            _,g = max(candidates)
            # ERM over identity and inversion, using only labels that have arrived.
            training = [(x,y) for age in range(1,5) for x,y,lag in days[d-age][g] if lag<=age]
            fitted = int(sum(x!=y for x,y in training)>len(training)/2)
            redundant += fitted==flip[g]
            flip[g]=fitted
            spend+=1
            last[g]=d
        for g in range(2):
            loss[g]+=sum((x ^ flip[g])!=y for x,y,_ in days[d][g])
    return loss[0]/(DAYS*SIZES[0]),loss[1]/(DAYS*SIZES[1]),spend,redundant


if __name__ == "__main__":
    with open("benchmark-results.csv","w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["scenario","policy","majority_error","minority_error","refits","unchanged_refits"])
        for scenario in ("stable","delay","harm","both"):
            streams=[stream(seed,scenario) for seed in range(20)]
            for policy in POLICIES:
                rows=[run(s,policy) for s in streams]
                writer.writerow([scenario,policy,*[round(mean(col),5) for col in zip(*rows)]])
