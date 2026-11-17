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

## Reproduce the first experiment

Run `python3 replay.py` and compare its output with results.csv. Each row simulates 100 predictions for the monitored region. The monitor alerts below 70% accuracy. Reporting delays are assigned by correctness to isolate selection bias. The example is deterministic and demonstrates a failure mechanism. Evaluating a correction on real streams is still pending.

## Run the live monitor

Use Python 3.12. In the project directory run:

    python -m pip install -r requirements.txt
    python -m uvicorn monitor:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000 for the dashboard. Use /docs to send binary predictions and outcomes, or run the historical demo from a second terminal:

    python historical_replay.py historical-input.csv

SQLite retains incoming events between restarts. Set MONITOR_DB to choose a new database. Exact retries are idempotent. The outcome endpoint requires an existing prediction. The dashboard polls every two seconds. Use GET /metrics with timezone-aware since and at parameters to inspect a prediction cohort as of a cutoff. Timestamps are producer-supplied event times. The local demo assumes a trusted event producer. It has no authentication.

For N predictions, C confirmed correct and U pending, final cohort accuracy lies between C/N and (C+U)/N. These are exact missing-outcome bounds, not statistical confidence intervals. They can be wide for small cohorts. The dashboard shows cumulative metrics; use a time window for incident analysis.

## Historical integration experiment

Source: IBM and MSFT columns of Matplotlib sample_data/Stocks.csv (source credited there to Yahoo Finance). The included extract preserves the original dates and values. Historical dates serve as replay ticks, not verified market publication timestamps. The predictor repeats the last observed direction. Outcomes use the next observation. Correct IBM predictions receive an additional simulated 21-day reporting delay. Events are posted in chronological order through HTTP. The market data exercise tests the service mechanics; delivery monitoring remains the intended application.

See historical-results.txt for two retrospective cutoffs. On May 10 the IBM cohort had one correct result and two unresolved outcomes. Observed accuracy was 100%, while final accuracy could still be as low as one third.

## Research follow-up

Elapsed-time weighting has prior art in conversion prediction ([Yang et al., AAAI 2021](https://doi.org/10.1609/aaai.v35i5.16587)). Compare learned corrections against the observed-label and fully resolved baselines at matched false-alarm rates. For the logistics study, investigate [Olist's public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and establish precisely when each label becomes knowable, including missed deadlines and cancellation. No logistics performance result or novelty claim has been established.
