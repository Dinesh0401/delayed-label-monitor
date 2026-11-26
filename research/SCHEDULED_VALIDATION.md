# Scheduled candidate validation under delayed feedback

A fixed 9-day wait can miss improvements when labels arrive late.
A fixed 30-day wait can leave a degraded model active after evidence is
available. This experiment compares both waits with a schedule fixed
before the replay: 9, 16, 23, and 30 days after the 14-day holdout closes.
The candidate and holdout stay fixed between checks. The runner divides
its existing one-look Hoeffding tail budget by four for the scheduled
policy and continues to score pending labels at their worst-case gain.

All policies use identical streams at each seed. Each cell has 100 seeds
(13000–13099) and 160 days. The harm scenario includes delays up to nine
days. Long tail combines harm and reporting changes with delays up to
30 days. Table cells show minority promotions and full-horizon error.

| Scenario | Minority predictions/day | Fixed 9 | Fixed 30 | Scheduled |
| --- | ---: | --- | --- | --- |
| Harm | 2 | 21 / 63.35% | 20 / 65.77% | 29 / 62.92% |
| Harm | 8 | 100 / 33.54% | 100 / 43.56% | 100 / 33.54% |
| Long tail | 2 | 0 / 69.08% | 6 / 68.49% | 6 / 68.49% |
| Long tail | 8 | 0 / 69.08% | 100 / 53.51% | 100 / 53.51% |

At eight predictions per day, the schedule matched the better fixed
policy in both tested regimes without choosing a wait based on the
scenario label. Mean promotion days were 85.16 and 127.16 respectively.
At two predictions per day with long delays, the evidence remained too
weak to promote in 94 of 100 runs.

This is a simulation of abrupt harm with a binary rule and a finite
horizon. It does not establish production performance, quantify false
promotions under additional null regimes, or replace a historical-order
replay. The correction is for these pre-specified looks, under the
existing bounded independent-loss assumptions.

```sh
python -m research.scheduled_validation > /tmp/scheduled-validation.json
diff -u research/results/scheduled-validation.json /tmp/scheduled-validation.json
python -m unittest discover -q
```
