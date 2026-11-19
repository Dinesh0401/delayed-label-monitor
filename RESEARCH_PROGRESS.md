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

## 2026-11-19 milestone (Phase 2)

Completed five core enhancements addressing all project requirements:
1. **Long-delay multiscale monitoring**: Implemented `matured_monitor.py` and evaluated on fresh seeds 7000–7099 (`long_tail_study.py`). Multiscale horizon evaluation and backward matured training scans resolved the 30-day delay blindspot, achieving 100/100 detections in `long_tail` (at minority=8, delay=43.15 days) while strictly preserving 0/100 false alarms on stable streams.
2. **Retraining cost-benefit analysis**: Implemented `retraining_cost_analysis.py` (`research/results/cost-benefit.json`). Proved that `shadow_group` achieves an 84.2% useful promotion rate (only 15.8% wasted refits) and 0.1656 gain per refit, compared to periodic retraining's 92.4% wasted refit rate and 36 wasted refits in stable regimes.
3. **Subgroup protection boundaries**: Implemented `group_protection_study.py` across sizes 1 to 48 (`research/GROUP_PROTECTION.md`). Established that reliable protection holds at n >= 8 (100% detection and adaptation), while documenting transparently that n <= 2 abstains due to wide Hoeffding bounds.
4. **Automated data integration & candidate staging**: Created `auto_pipeline.py` with batch ingest, stale order anomaly detection, and prospective candidate staging. Added `/api/export`, `/api/batch-orders`, `/api/batch-outcomes`, and `/api/pipeline-status` to `serving.py`. Guaranteed operator-in-the-loop protection: live models are never replaced automatically without explicit confirmation.
5. **Consented user study harness**: Implemented `user_study_runner.py` enforcing strict pre-session consent and administering the 6 standardized operator tasks against pre-registered criteria.

All 37 unit tests and Node Worker checks pass. Main remote branch is synchronized.

