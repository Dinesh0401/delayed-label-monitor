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

## Reporting-regime calibration ablation (2026-11-18)

Run `python robust_detection_study.py` and `python -m unittest test_robust_detection -v` (standard library only). This follow-up preserves both detector scores and the original four data-generating scenarios. Calibration uses fresh seeds 2000–2099; evaluation uses 300 independent seeds 3000–3299 per scenario. Scenarios share seed-level random draws for paired comparisons. Do not treat the four scenarios as 1,200 independent seeds.

For each method and calibration regime, take each run's maximum score over days 12–69. With n=100 and alpha=0.05, set the threshold to order statistic ceil((n+1)(1-alpha))=96. An alarm requires a score strictly above the threshold. The stable-only ablation uses the stable regime threshold. The reporting-robust ablation uses the maximum of stable and reporting-only regime thresholds. Both ablations use the same rank convention; the original study used rank 95, so its archived results are not overwritten. Calibration happens before evaluation and never uses harmful test outcomes.

| Censored detector outcome | Stable-only calibration | Reporting-robust calibration |
| --- | ---: | ---: |
| Stable false alarms | 13/300 | 0/300 |
| Reporting-only false alarms | 300/300 | 18/300 |
| Harm-only detections among eligible runs | 296/296 | 1/300 |
| Harm-only pre-change alarms | 4/300 | 0/300 |
| Combined-change detections among eligible runs | 296/296 | 292/300 |
| Combined-change mean capped delay | 3.32 days | 13.86 days |

The arrived-label baseline's threshold is unchanged by broader calibration. It detects 294/294 eligible harm-only streams, with six pre-change alarms and a mean delay of 6.17 days. Under combined changes it detects 255/294 eligible streams; the mean capped delay is 18.00 days. All rows and thresholds are in `robust-detection-results.json`.

This is a negative result: broader null calibration reduces reporting-shift false alarms but nearly eliminates censored-detector power for harm-only drift. A high score can reflect misspecified reporting probabilities rather than increased model error. The observed reporting-only false-alarm rate remains 6%, not a demonstrated 5% guarantee for this fitted threshold. No parameter was retuned after seeing these evaluation results.

Evaluation excludes pre-change-alarmed runs from harmful-drift detection denominators and reports those exclusions. Conditional mean delay includes detections only. Mean capped delay assigns the 40-day remaining monitoring horizon to every eligible miss, making failures visible rather than rewarding a detector that rarely fires. These are finite-horizon synthetic experiments with known initial reporting probabilities, a single slowdown mechanism, no group stratification, and no retraining action. They establish neither fairness nor production readiness. Next: compare matured-cohort and jointly estimated reporting/performance baselines on fresh seeds, then test minority groups and unseen reporting regimes.

## Working delivery application

The operational application is now separate from the illustrative stream and the original generic monitor. It supports actual model inference, durable saved orders, immutable delivery outcomes, state-level descriptive accuracy, pending-outcome bounds, and JSON export of the latest 200 records. Metrics include all records. No artificial stream is injected into the operational database.

```sh
python -m pip install -r requirements.txt
python -m uvicorn serving:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 on the computer running Python. The SQLite file defaults to `delivery.sqlite`; set `DELIVERY_DB` to change it. This local API has no authentication and is intended for trusted local use. Hosted access remains owner-private and uses a separate persistent D1 database.

1. Enter a non-identifying order ID, Brazilian state, purchase time, and exclusive delivery deadline.
2. The server constructs features and applies the saved logistic regression. A score at or above 6% is classified as late risk. It stores the inputs, prediction, model version, and creation time.
3. Record the delivery timestamp. Late means delivery at or after the deadline; reporting time is recorded separately by the server. The outcome cannot silently replace an existing different result.
4. The queue and per-state metrics update. Missing outcomes produce bounds `[correct/total, (correct+pending)/total]`; these are not confidence intervals.

All entered times are UTC. The original historical training data used its recorded naive calendar values; this demo's UTC convention should be adapted and evaluated for an operational source. No courier integration or automatic label collection is configured: the operator supplies actual order/outcome data. No automatic drift-triggered retraining is enabled because the research detector has not passed validation.

### Model and deployment parity

`train_serving.py` trains the portable logistic baseline on the same chronological split as the historical study. With the Olist ZIP available, run `python train_serving.py /path/to/orders.zip` after installing `requirements-research.txt`. Train: 41,197 orders. Held-out test: 32,150 orders. Logistic average precision 0.102634 and Brier score 0.041126. The previous historical study reports the separate gradient-boosting model, so its numbers are not the serving-model metrics. The logistic decision threshold is selected on calibration data, never the test set. The script checks exported predictions against scikit-learn on every held-out order; 30 fixtures are retained for cross-runtime checks.

Python serves `serving-model.json` through FastAPI. Sites hosts a JavaScript Worker that uses the identical coefficients, feature order, intercept and threshold; Python does not execute inside Sites. Build the Worker with `python build_hosted.py /path/to/site/dist/server`. The source manifest uses `d1: "DB"`, and `hosted/drizzle` contains the schema-only hosted migration. Apply this migration only once through Sites' migration workflow, not on requests. `hosted/dev.mjs` is a local preview adapter, not production storage.

Verification:

```sh
python -m pip install httpx
python -m unittest test_serving test_robust_detection -v
node hosted/test-worker.mjs
```

Node 24 supplies the SQLite adapter used by Worker tests. Both runtimes agree with the scikit-learn fixtures within 1e-12. Tests cover input rejection, duplicate retries, conflicting inputs and outcomes, timestamp ordering, saved state across clients, and pending-to-resolved metric bounds. Browser QA exercised prediction and outcome submission on the local Worker preview. Research and product claims remain limited to a working research MVP, not a validated production drift or fairness system.

## Fixed-cohort monitoring and prospective retraining (2026-11-19)

The new research milestone implements missing-label error bounds, subgroup monitoring, and a prospective shadow candidate gate. See `research/METHOD.md` for equations, assumptions and related research; `research/PROTOCOL.md` records the initial method and the fresh-seed follow-up after its failure.

```sh
python research_suite.py --phase initial
python research_suite.py --phase shadow
python research_report.py
python -m unittest discover -v
node hosted/test-worker.mjs
```

Both complete experiment phases were rerun and matched all four saved CSV/JSON outputs byte for byte. Per-run CSVs are committed as deterministic gzip files; the script produces both raw and compressed files. Uncompressed checksums are in `research/results/reproduction-sha256.json`. NumPy is pinned in `requirements-research.txt`. The suite covers 9 scenarios, 3 group sizes and 100 seeds per phase: 16,200 initial policy runs and 18,900 follow-up policy runs, with shared random streams for paired comparisons. These are not 35,100 independent experiments.

In the follow-up, with 8 minority predictions/day under severe drift and reporting slowdown, prospective shadow adaptation reduced minority error from 62.6% to 40.1%, with one refit per run. The paired 95% bootstrap interval for the gain is approximately 22.2–22.7 percentage points. It observed zero false alarms/refits in stable and reporting-only settings. Scheduled periodic refits achieved lower severe-drift error (~37%) at higher training cost (4 attempts). The prospective method is not uniformly best. At 2/day only 38/100 severe-drift runs promoted a candidate. At 8/day under 30-day delays it detected 0/100 harmful streams. Full results and the failed original gate are retained.

The operational API `/api/diagnostics` evaluates a fixed model per state once per UTC-day snapshot. It starts after 67 days of prediction-creation history; new or sparse data is explicitly inconclusive. A fixed 30-day reference is compared with a later 28-day cohort, buffered by 9 days. Missing labels retain their place in the denominator. The family budget covers 27 states and at most 120 daily looks per model version. It is not a perpetual guarantee, does not restart automatically and must not be interpreted as proof of safety. The live serving model is never replaced by this endpoint.

`candidate_retrain.py` trains a real scikit-learn logistic shadow candidate from an exported order JSON, evaluates a later fixed cohort, writes a candidate artifact and a per-state promotion report, and leaves the deployed model untouched. Example:

```sh
python candidate_retrain.py export.json --as-of 2026-11-19T00:00:00Z --train-before 2026-10-01T00:00:00Z --out candidate-run
```

This requires at least 30 labeled training orders, both classes, and later validation records; a small sample should normally fail the evidence gate. Pending labels count against approval. It certifies neither absent states nor repeated tuning on the same holdout. The browser export contains only the latest 200 orders, so it may be inadequate for training. No live courier credentials, automatic ingestion, or production retraining rollout is configured.

The research report is generated at `research/report.html` and served at `/research-study.html`. `research/USER_STUDY.md` provides a consented usability-study protocol only: no participants or feedback have been fabricated. Remaining publication blockers include strong external baselines, real-stream adaptation replication, temporal-dependence validation, independent novelty review and actual consented users.
