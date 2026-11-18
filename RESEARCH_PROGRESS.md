# Resume point — 2026-11-18

Completed milestone: reporting-regime calibration ablation. Reproduce with:

```sh
python robust_detection_study.py
python -m unittest test_robust_detection -v
```

The committed JSON is deterministic. 100 calibration seeds and 300 fresh evaluation seeds per scenario compare stable-only versus two-regime calibration on unchanged detector scores. Broader calibration reduces censored reporting-only false alarms from 100% to 6%, but harm-only detection falls from 296/296 eligible streams to 1/300. This is evidence against threshold calibration alone as a repair, not a successful robust detector.

Next research milestone: explicitly separate reporting changes from error changes, including a matured-cohort baseline; use new calibration and evaluation seeds. Then add minority-group sample-size stress tests and retraining costs. Do not reuse current test seeds for tuning.

Publication blockers: no validated general drift detector, no group-protection evidence for this ablation, no demonstrated retraining benefit, and no consented production-user study. Hosted animations are separate from offline research. No live site update was part of this milestone.

Existing uncommitted train_serving.py, serving-model.json, and inference-fixtures.json belong to the interrupted model-serving task. They were left untouched, are not validated by this milestone, and are not part of its commit. Complete their integration separately.

GitHub commits were previously recreated through the connector, so local and GitHub commit IDs differ despite matching source trees. Use GitHub main as the remote authority; do not force-push local history. This milestone updates main using the previous remote head as parent.
