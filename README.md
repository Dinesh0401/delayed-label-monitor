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

## Budgeted subgroup adaptation experiment

Run `python benchmark.py` to regenerate benchmark-results.csv using only the Python standard library. Each of 20 paired seeds has 16 warm-up days and 64 scored days, with 80 large-group and 8 small-group observations per day. The binary feature initially predicts the label with 12% noise. Under harmful drift the small group's mapping reverses on day 24. Under a reporting shift the early arrival probability for its negative labels drops on day 20. Labels arrive after one or five days. Simulated drift is abrupt and deliberately recoverable by this small hypothesis class.

Within each group and feature stratum, r is the current model error probability. The early reporting probabilities for errors and correct predictions are a and b. Estimate each from the most recent 12 fully resolved daily cohorts with Beta(1,1) smoothing. For four recent cohorts the outcome is either visible or pending. For E visible errors, C visible correct labels and U pending cases, the likelihood is proportional to:

    r^E (1-r)^C [r(1-a) + (1-r)(1-b)]^U

An equally weighted 160-point grid approximates a uniform prior on r. A flip changes expected error from r to 1-r. Thus a gain exceeding ten percentage points requires r > 0.55. Trigger when both feature strata individually assign more than 90% posterior probability to that gain. This conjunction is a heuristic and does not give a 90% joint guarantee. Under a shared two-refit budget, choose the eligible group with the strongest weaker-stratum evidence. Refit by minimizing arrived-label errors over two rules, identity and inversion. Outcomes from the decision day enter retrospective scoring only. Pending feature counts use observable inputs.

Comparators use the same streams and refit training rule: frozen model, scheduled refits (days 16 and 32, ties favor the small group), pooled arrived-error trigger, and subgroup arrived-error trigger. The unchanged-refit metric counts identical parameters after fitting; a changed model can also be harmful. Every policy including the proposed one is heuristic. No global optimality, group fairness guarantee or real-world improvement has been demonstrated. Each unit of budget represents one refit, independent of compute time.

With simultaneous drift and slower reporting, small-group errors averaged 17.207% versus 17.246% for the simple subgroup trigger. Both far outperformed pooled monitoring in this constructed setting. With drift alone the simple subgroup trigger was better. Reporting probabilities are treated as point estimates and may be inaccurate after a delay shift. Future work should vary drift strength, group size and delay mechanisms and compare detection delay at matched false-alarm rates on independent calibration and test periods. This experiment runs offline; the hosted demo illustrates pending labels.

## Delivery ML benchmark on Brazilian e-commerce orders

Download the [original Olist archive](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and retain its ZIP structure. Research dependencies are separate from the service dependencies.

    python -m pip install -r requirements-research.txt
    OMP_NUM_THREADS=2 python delivery_study.py /path/to/brazilian-ecommerce.zip

Archive SHA-256 used: 967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784. Only aggregate outputs are tracked. Training used 41,197 delivered orders resolved before Jan 1, 2018. Calibration used 16,782 later purchases resolved before Apr 1. The held-out evaluation includes 32,150 purchases from Apr 1 through Aug 31, 2018. Orders without confirmed fulfillment were excluded. Scores therefore apply to eventually delivered orders; the composition of the deployable population may differ. Purchases still pending at either fitting cutoff cannot provide training outcomes. This can bias fitted delay rates.

At purchase, a histogram gradient boosted tree uses promised lead time, purchase calendar features and destination state. A delivery after the promised calendar day counts as late. The model is held fixed through testing. Threshold selection maximizes calibration F1. The constant forecast uses training late prevalence. Average precision quantifies ranking for the rare late class; the Brier score measures squared probability error.

For each purchase week and region, inspect outcomes available by each order's 14th day. These age-aligned checkpoints are retrospective cohort summaries. Assume confirmation at delivery; timestamped ingestion logs are unavailable. Some late labels could be inferred earlier if reliable in-transit status were available. Estimate early-report rates conditional on forecast correctness using resolved calibration orders. Pool each group's risk with the censored likelihood in drift_policy.py. Evaluate estimated error against eventual realized error. Score weekly cohorts with at least ten orders, giving each included week equal weight. Partial boundary weeks are included if eligible. Geographic grouping: São Paulo, seven northern states, remaining states.

Open delivery-results/report.html for the results. forecast.csv, weekly.csv and monitor.csv provide the underlying numbers. Better regional estimation on these dates does not establish faster drift detection at equal false-alarm rates, better refit decisions, or demographic fairness. The earlier budgeted retraining experiment remains synthetic.

## Does the regional result depend on one week?

Run `python weekly_stability.py` on the committed held-out weekly aggregates. The paired gain equals the arrived-label absolute error minus the corrected absolute error, in percentage points. Positive values favor correction. The output delivery-results/stability.csv includes week-level wins, the median, an order-count-weighted mean of cohort gains, and the range of means after omitting each week in turn. This is a sensitivity analysis on the same test period. The omission range is not a confidence interval or an independent replication.

## Independent detection calibration

Run `python detection_study.py`. Seeds 0–99 calibrate two run-maximum thresholds at the empirical 95th percentile of stable streams. Independent seeds 1000–1099 test stable, reporting-only, harmful-drift, and combined changes. A stream has 12 observations per day for 70 days. Labels arrive after 1 or 9 days; monitoring windows include the preceding 7 days. Drift begins on day 30. The model error rises from 0.12 to 0.38. Both methods use the same generated events.

The censored method assumes initial early-arrival probabilities are known (0.3 for errors and 0.8 for correct predictions). It keeps them fixed during test. In reporting-shift scenarios they become 0.08 and 0.45. Those scenarios explicitly test misspecification. Only predictions whose arrival timestamp is at or before each decision time contribute observed outcomes. Candidate error probabilities use a uniform prior and the existing numerical posterior.

Held-out stable streams produced zero alarms for both methods, which is an empirical outcome rather than a guaranteed error rate. Harmful-drift detection averaged 5.70 days with censoring versus 6.53 days with arrived labels. However reporting-only shifts caused alarms in all 100 censored runs versus 2 arrived-label runs. This failure prevents claiming an effective general drift detector. Detection rates use only streams without a pre-change alarm; delay averages include only detected streams. Next work should jointly model delay changes and performance changes, calibrate under multiple null delay regimes, and evaluate on fresh seeds.

The hosted laboratory illustrates two groups and adjustable reporting scenarios using JavaScript. Its illustrative 25% error alert is separate from this calibrated offline Python benchmark.
