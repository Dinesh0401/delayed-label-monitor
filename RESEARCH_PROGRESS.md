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

Next: independent real-data adaptation evaluation and strong external detector baselines; address long-tail abstention without reintroducing reporting-induced false alarms. Obtain actual consenting study participants separately. The deployment preserves owner-only access and does not promote a model automatically. The currently served Olist model remains unchanged.
