# Fixed-cohort evidence and prospective promotion

## User problem and hypothesis

Operators can confuse slow outcome reporting with deteriorating predictions, while pooled accuracy can hide minority-group harm. Hypothesis: keeping missing labels inside fixed prediction cohorts prevents reporting alone from manufacturing error evidence; a separate prospective candidate gate reduces unnecessary model changes. The price is latency and reduced power. These hypotheses are tested here on controlled streams, not asserted for arbitrary production data.

## Formulation

Let a predetermined cohort contain n binary losses, e observed errors and u unresolved labels. Its complete empirical error is always inside [e/n, (e+u)/n], even when label arrival depends on the outcome. For tail probability a, use radius r(n,a)=sqrt(log(1/a)/(2n)). The interval is [max(0,e/n-r), min(1,(e+u)/n+r)]. Empty cohorts have interval [0,1].

For independent bounded losses and fixed prediction-cohort membership, Hoeffding's inequality bounds each one-sided failure probability by a. We allocate a=0.05/(2GT) for G groups and T predetermined decision times. A union bound covers the reference upper and current lower bounds for all scheduled group/time comparisons. Overlapping cohorts do not require independent comparisons; independent losses within each cohort, or an appropriate martingale replacement, are still assumptions. Alarm when current lower error > reference upper error +0.05. Under the stated no-harm null this gives a finite-horizon family error bound. It is NOT a guarantee under temporal dependence, adaptively chosen cohorts, repeated horizon resets, or unspecified model changes.

For validation paired differences d=old loss minus candidate loss in [-1,1], replace each missing d by -1. Lower improvement is (sum(observed d)-u)/n - sqrt(2 log(1/a)/n). A candidate must exceed 0.02 in the synthetic experiment. The real-logistic shadow CLI uses an overall minimum of 0.005 and a per-observed-state deterioration tolerance of 0.02. These are explicitly different experiments. Missing/small groups can block approval; absent groups are NOT certified.

## Adaptation and group scope

Synthetic prediction rules are identity or inversion, fitted separately by group. A group promotion leaves every other group's predictions unchanged by construction. This isolates subgroup adaptation; it is not a general fairness algorithm. Severe drift reverses the binary relationship (.12 to .88 error); mild .35 noise cannot be repaired by inversion. The overall benchmark loss is measured prequentially on all generated outcomes by an offline evaluator. Those hidden outcomes are never supplied to the policy before arrival.

The prospective policy freezes a candidate and validates on the next 14-day cohort, then waits nine days. Validation never overlaps training. A maximum of two attempts per group and one promotion per group controls cost, not a universal optimality claim. Repeated validation is budgeted over the finite schedule. The real-logistic CLI is a ONE-SHOT chronological evaluation and must not be repeatedly tuned on the same holdout.

## Baselines and weaknesses

Compare frozen, periodic, arrived-only, pooled fixed-cohort, group fixed-cohort, ungated group and prospective shadow policies. The arrived-only baseline uses the same nine-day buffer: with delays at most nine days it equals the fixed-cohort score. It is not a fast-arrival state-of-the-art competitor. Under long tails its selected observed subset can differ. Periodic actions must not be interpreted as detector alarms. No ADWIN/DDM, DriftSurf implementation, external real-world drift replication, patent clearance or exhaustive novelty review has been completed.

Reference/current windows overlap early in simulation; the valid union-bound argument does not assume disjoint windows, but early sample information is limited. In production the daily reference and current periods are disjoint and begin comparisons at day 67. Production uses a larger 27-state multiplicity budget, so synthetic two-group power estimates do not transfer directly. Diagnostics use actual prediction creation times and freeze label availability at UTC day start. A page refresh cannot allocate a new statistical look. No evidence yet means uncertainty, not model safety.

## Related primary research

- Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Annals of Statistics (2021), https://arxiv.org/abs/1810.08240 . Establishes the broader sequential-inference literature. This project currently uses a simpler finite-horizon allocation, not their confidence-sequence construction.
- Tahmasbi et al., *DriftSurf*, ICML (2021), https://proceedings.mlr.press/v139/tahmasbi21a.html . Studies stable/reactive adaptation to improve the detection–adaptation tradeoff. Our shadow gate is related in motivation, not a reproduction or claimed invention of staged adaptation.
- Yu et al., *Request-and-Reverify*, IJCAI (2018), https://ijcai.org/proceedings/2018/421 . Uses hierarchical testing and selective label requests. This project does not implement active label acquisition.

Originality remains an open question. A functioning implementation and favorable synthetic result are not proof of scientific novelty.
