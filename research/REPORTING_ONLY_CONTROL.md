# Reporting slowdown without model drift

The matched-horizon study compared stable reporting with a scenario that
changed both prediction error and reporting delay. This experiment isolates
reporting changes to test whether a backlog alone triggers retraining.

Use 100 seeds (11000–11099) per group size, 160 days per replay, and the
existing matured detector. The paired control and stress stream have
identical prediction errors, checked for every seed and group. At day 40,
the stress stream reduces early arrivals using the existing reporting-slow
scenario and stretches its post-change nine-day delays to 30 days.
Both streams are null cases, so an alarm on any day is a false alarm.

| Minority predictions/day | Pending in control | Pending under slowdown | False-alarm runs under slowdown | Refit attempts |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 7.63% | 60.18% | 0/100 | 0 |
| 8 | 7.49% | 59.61% | 0/100 | 0 |
| 24 | 7.40% | 59.49% | 0/100 | 0 |

Pending means the fraction of minority predictions made on days 40–67
still missing an outcome at day 68. The stable controls also recorded zero
alarms and refits. These are synthetic results for a specific reporting
shift. Zero observed alarms does not establish a zero population rate.
The earlier matched-horizon study provides the separate harm-detection
comparison. Neither study establishes operational performance on live orders.

Reproduce from the repository root:

```sh
python -m research.reporting_only_control > /tmp/reporting-only-control.json
diff -u research/results/reporting-only-control.json /tmp/reporting-only-control.json
```
