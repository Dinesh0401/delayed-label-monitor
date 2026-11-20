# Empirical Boundaries of Subgroup Protection

## Executive Summary

Conservative fixed-cohort bounds guarantee finite-horizon family-wise error control,
but statistical power and adaptation speed depend strictly on subgroup sample size.
This stress study evaluates minority prediction volumes from 1 to 48 per day (seeds 8000–8049, 50 paired seeds per cell)
under severe drift (error rises from 0.12 to 0.88 on day 40).

## Experimental Results

| Minority / Day | 28-Day Cohort Size | Harm Detection Rate | Both (Drift+Slow) Det. | Promotion Rate | Error Reduction | Empirical Status |
|---|---|---|---|---|---|---|
| 1 | 28 | 46.0% | 46.0% | 0.0% | +0.0 pp | 🔴 Severe sample deficit: detection <50%, adaptation fails |
| 2 | 56 | 100.0% | 100.0% | 36.0% | +5.5 pp | 🟡 Reliable detection, partial adaptation (36% promoted) |
| 4 | 112 | 100.0% | 100.0% | 100.0% | +20.2 pp | 🟢 Full synthetic detection & adaptation (100% promoted) |
| 8 | 224 | 100.0% | 100.0% | 100.0% | +22.2 pp | 🟢 Full synthetic detection & adaptation (100% promoted) |
| 16 | 448 | 100.0% | 100.0% | 100.0% | +24.5 pp | 🟢 Rapid detection (< 18 days), 100% adaptation |
| 24 | 672 | 100.0% | 100.0% | 100.0% | +24.8 pp | 🟢 Rapid detection (< 18 days), 100% adaptation |
| 48 | 1344 | 100.0% | 100.0% | 100.0% | +10.2 pp | 🟢 Rapid detection (< 18 days), 100% adaptation |

## Critical Limitations & Non-Claims

1. **Extreme Small-Sample Deficit (n = 1 / day)**:
   - At n=1 (28 observations in a 28-day window), the Hoeffding radius is ~0.40.
   - The monitor detects drift in fewer than half the runs (46%) and authorizes zero promotions (0%).
   - The monitor abstains rather than making unverified model modifications.

2. **Partial Adaptation at n = 2 / day**:
   - At n=2 (56 observations per window), the monitor reliably detects the harm signal (100% detection rate),
     but the prospective validation gate only approves candidates in 36% of runs due to missing validation evidence.

3. **Full Adaptation at n ≥ 4 / day (Synthetic Setting)**:
   - In this simplified binary hypothesis class (identity vs. inversion), n ≥ 4 supplies sufficient sample
     evidence for the prospective gate to promote the inverted model, recovering ~20+ percentage points of accuracy.

4. **Synthetic Non-Claim**:
   - These results evaluate an artificial two-rule hypothesis class on independent streams.
   - They **do not establish universal fairness, demographic parity, or real-world subgroup protection** for complex multi-class or continuous models.
   - Stable streams generated **0% false alarms** across all tested group sizes (1 to 48).
