# Fixed validation buffers under delayed feedback

The previous study selected the validation buffer from the scenario name.
This ablation assigns 9, 30, or 45 days in advance and applies each choice
to the same streams. The buffer starts after the 14-day prospective
candidate holdout. Detection, candidate fitting, and promotion criteria
are otherwise unchanged.

The experiment uses 100 seeds (12000–12099) per scenario and volume.
Each run has a 160-day horizon. "Harm" has reporting lags up to 9 days.
"Long tail" also slows reporting and has lags up to 30 days. Minority
error is measured over the whole horizon, including before drift and
while the candidate awaits validation.

| Scenario | Minority predictions/day | Promotions (9 / 30 / 45 days) | Minority error (9 / 30 / 45 days) |
| --- | ---: | --- | --- |
| Harm | 2 | 21 / 14 / 10 | 64.12% / 66.97% / 67.70% |
| Harm | 8 | 100 / 100 / 100 | 33.50% / 43.43% / 50.55% |
| Long tail | 2 | 0 / 10 / 10 | 69.00% / 67.95% / 68.63% |
| Long tail | 8 | 0 / 100 / 100 | 68.93% / 53.37% / 60.55% |

Detection reached 100/100 in each cell. At eight predictions per day in
the long-tail setting, extending the 30-day buffer to 45 days postponed
mean promotion from day 127.42 to day 142.42 and raised error by 7.18
percentage points. With the shorter reporting regime, a 30-day buffer
also incurred additional error relative to 9 days. A long fixed
wait can therefore trade approval evidence for avoidable time on a
degraded model. Some late candidates can run out of follow-up at day 160.

These simulations use a binary identity-versus-inversion policy. They
do not select an optimal production buffer or validate real-order
retraining. A later experiment can compare a pre-specified sequence of
checks with statistical correction for repeated looks.

```sh
python -m research.validation_buffer_study > /tmp/validation-buffer-study.json
diff -u research/results/validation-buffer-study.json /tmp/validation-buffer-study.json
```
