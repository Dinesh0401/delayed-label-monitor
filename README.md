# Delayed Label Monitor

A personal ML reliability research project for engineers deciding when a deployed prediction service needs investigation. A small delivery region is the initial simulated population.

## Research question

At equal false-alarm rates, can an estimate that accounts for unresolved outcomes detect subgroup performance loss sooner than waiting for all labels? The long-term product is a web dashboard that shows pending outcomes, evidence behind alerts, and the cost of retraining.

## Experimental design

Replay three interventions independently: slower reporting, worse prediction quality, and both. Restrict each decision to outcomes received by its cutoff. Use eventual outcomes only for retrospective scoring. Compare arrived-label accuracy, waiting for a mature cohort, and a historical delay correction. Calibrate alert thresholds on development periods and evaluate on later periods. Compare missed incidents, time to detection, subgroup sample size and eventual prediction loss under equal retraining budgets.

The candidate correction weights each observed correct or incorrect prediction by the inverse probability that its outcome has arrived by the cutoff. Estimate those probabilities from previously resolved cohorts. Check how estimator variance changes for small groups and how errors change when delays themselves shift.

## Related work

- [Chapelle, Modeling Delayed Feedback in Display Advertising](https://doi.org/10.1145/2623330.2623634) models conversion delays.
- [FAIRCAST](https://doi.org/10.1145/3614419.3644028) studies subgroup performance under temporal shifts and delayed labeling.
- [Request-and-Reverify](https://doi.org/10.24963/ijcai.2018/421) obtains labels selectively to verify drift.

The contribution under investigation is whether delay correction improves operator decisions under changing reporting patterns. Novelty and benefit require further comparisons.

## Milestones

1. Reproduce a controlled reporting-delay failure.
2. Compare methods across held-out timelines with realistic cohort sizes.
3. Replay a public dataset with documented prediction and outcome timestamps.
4. Build a working monitor and evaluate decisions with consenting ML practitioners.

Commit each reproducible change with a message explaining what it establishes. Record unfavorable results alongside positive ones.
