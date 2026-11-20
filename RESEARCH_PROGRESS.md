# Resume point — 2026-11-18

Completed milestone: reporting-regime calibration ablation. Reproduce with:

```sh
python robust_detection_study.py
python -m unittest test_robust_detection -v
```

The committed JSON is deterministic. 100 calibration seeds and 300 fresh evaluation seeds per scenario compare stable-only versus two-regime calibration on unchanged detector scores. Broader calibration reduces censored reporting-only false alarms from 100% to 6%, but harm-only detection falls from 296/296 eligible streams to 1/300. This is evidence against threshold calibration alone as a repair, not a successful robust detector.

Next research milestone: explicitly separate reporting changes from error changes, including a matured-cohort baseline; use new calibration and evaluation seeds. Then add minority-group sample-size stress tests and retraining costs. Do not reuse current test seeds for tuning.

Publication blockers: no validated general drift detector, no group-protection evidence for this ablation, no demonstrated retraining benefit, and no consented production-user study. Hosted animations are separate from offline research. No live site update was part of this milestone.

Later on 2026-11-18: completed the serving MVP integration. serving.py provides Python/FastAPI inference and SQLite records; web/index.html is the operational UI; hosted/worker.mjs uses the same learned parameters with D1 persistence. Model parity, API workflow, and browser prediction/outcome checks passed. The previously uncommitted training script, model and fixtures are now included. See README for reproduction and hosting architecture. The live product still requires operator-supplied outcomes; no courier integration, automated retraining, or validated fairness claim is present.

GitHub commits were previously recreated through the connector, so local and GitHub commit IDs differ despite matching source trees. Use GitHub main as the remote authority; do not force-push local history. This milestone updates main using the previous remote head as parent.

## 2026-11-19 milestone

Completed fixed-cohort bounds, 27-state daily operational diagnostics, a seven-policy synthetic follow-up, prospective shadow promotion, and a real-logistic offline candidate training CLI. Initial and follow-up results reproduced byte for byte; all unit and Worker checks passed. See research/PROTOCOL.md, METHOD.md, results/, and report.html. Small groups, mild drift, long tails and missing labels remain explicit failure cases. Do not describe this as a fully validated flagship or fairness solution.

## 2026-11-20 audit & corrections milestone

Following rigorous verification against committed data, code, and Worker runtimes, we addressed implementation defects and corrected research interpretations:
1. **Long-delay detection vs. adaptation distinction**: `matured_monitor.py` achieves 100/100 detections on 30-day delay streams (`long_tail`, mean delay 43.15 days) with 0/100 false alarms on stable streams. However, zero candidates were promoted (`promotions: 0.0`) within the 120-day horizon because prospective holdout validation requires 44 days (14 days cohort + 30-day lag buffer), which exceeds the remaining timeline when alarms fire around day ~83. Harm detection is verified; adapting within this finite horizon remains an unsolved limitation.
2. **Subgroup protection boundaries and reporting**: Stress testing across $n \in \{1, 2, 4, 8, 16, 24, 48\}$ (`research/GROUP_PROTECTION.md`) showed: $n=1$ fails adaptation (0% promotions, 46% detection); $n=2$ achieves 100% detection and 36% promotion; $n \ge 4$ achieves 100% detection and 100% promotion in the synthetic setting. Error gain is $+22.2\text{ to } +24.8\text{ pp}$ for $n \in \{8, 16, 24\}$, and $+10.2\text{ pp}$ for $n=48$. Stable streams had 0% false alarms across all sizes. Regenerating the study now faithfully reproduces this analysis. Universal fairness is not claimed.
3. **Rigorous retraining efficiency accounting**: Corrected the interpretation in `retraining_cost_analysis.py` (`research/results/cost-benefit.json`). Under harmful drift, `shadow_group` achieves a **54.7% promotion success rate** (promotions / refits) and **11.04 percentage points gain per refit**, compared to periodic retraining's 12.7% promotion rate and 3.22 pp gain per refit. In null/stable regimes, periodic retraining executes 4 refits per run (100% wasted under null conditions, summing to 36 attempts across 9 null simulation cells), whereas `shadow_group` executes 0.0 refits (0% waste).
4. **Batch ingestion validation and duplicate counting**: Both Python (`serving.py`, `auto_pipeline.py`) and Cloudflare Worker (`hosted/worker.mjs`) strictly reject delivery dates beyond `now`. Worker batch insertion counts were corrected to inspect `changes` so retried/duplicate orders report `inserted: 0`.
5. **Operational diagnostics multiscale integration**: Integrated multiscale horizon monitoring (evaluating short 9-day buffer and mature 30-day buffer when day >= 88) into both `operational_diagnostics.py` and `hosted/worker.mjs`. Added hot-reloading (`/api/model/reload`) with in-place dictionary mutation to prevent stale references in `candidate_retrain.py`.
6. **User-study harness and linter qualification**: `user_study_runner.py --template` outputs an empty participant form (`research/results/blank-user-study-form.json`). The no-fairness-claims check is an automated text linter for prohibited marketing terms, not a legal or ethical fairness certification. Actual external human participants remain pending.
7. **Deployment scope**: Code is synchronized with GitHub `origin/main` and passes all local and Node/Worker tests; live deployment to public infrastructure remains separate.

All 38 Python unit tests and Node Worker checks pass cleanly. Main remote branch is synchronized.

