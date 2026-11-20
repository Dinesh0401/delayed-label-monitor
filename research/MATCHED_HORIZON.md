# Long-delay adaptation against a matched frozen baseline

The earlier 120- and 160-day results used different horizons and therefore
different proportions of post-drift predictions. This follow-up estimates the
benefit of adaptation by scoring the frozen identity rule and matured shadow
policy on exactly the same 160 days and prediction events.

```sh
python -m research.matched_horizon > /tmp/matched-horizon.json
diff -u research/results/matched-horizon.json /tmp/matched-horizon.json
```

Each cell uses 100 seeds (10000–10099). The frozen rule's loss is the
simulator's original error bit. The adaptive policy observes outcomes only
after arrival. Evaluation uses all eventual outcomes. Intervals are
percentile bootstrap intervals from 5,000 resamples of paired seed-level
gains. Positive gain means lower minority error with adaptation.

| Minority predictions/day | Frozen error | Adaptive error | Gain (95% interval), pp | Detections | Promotions | Refit attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 69.26% | 68.02% | 1.24 [0.62, 1.96] | 100/100 | 12 | 186 |
| 8 | 69.03% | 53.42% | 15.61 [15.42, 15.81] | 100/100 | 100 | 100 |
| 24 | 69.09% | 51.23% | 17.86 [17.74, 17.98] | 100/100 | 100 | 100 |

Stable controls produced zero alarms, refits, and promotions in 300 runs.
At two predictions per day, detection was common but promotion remained
rare and the end-to-end benefit was much smaller.

These results concern a synthetic identity-versus-inversion task with an
abrupt minority error increase and delays up to 30 days. The existing
runner uses the simulated long-delay regime to choose its holdout buffer.
The comparison quantifies adaptation under that configuration. It does
not establish real-order retraining performance or identify the appropriate
buffer from production data.
