"""Train a shadow model from exported orders; never overwrite the deployed model.

Example:
    python candidate_retrain.py export.json --as-of 2026-11-19T00:00:00Z \\
        --train-before 2026-10-01T00:00:00Z --out candidate-run

Run once on a held-out validation period; repeated tuning invalidates the statistical gate.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cautious_monitor import promotion_bound
import serving

MODEL = serving.MODEL


def time(value: str) -> datetime:
    """Parse an ISO 8601 string into a timezone-aware UTC datetime.

    Args:
        value: Datetime string.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: If value cannot be parsed or lacks timezone information.
    """
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps require explicit timezone information")
    return dt.astimezone(timezone.utc)


def evaluate_gate(
    old: np.ndarray,
    new: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
    cost: float = 0.005,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Evaluate whether candidate model satisfies the conservative promotion bound.

    Requires an overall paired error reduction > cost (0.005) and forbids any
    individual observed group from degrading by more than tolerance (0.02).
    Pending validation outcomes are penalized with worst-case improvement (-1).

    Args:
        old: Binary predictions from current baseline model.
        new: Binary predictions from candidate model.
        labels: Ground-truth binary labels (-1 for unresolved/pending).
        groups: Array of group/state identifiers.
        alpha: Overall significance level.
        cost: Minimum required global improvement (default: 0.005).
        tolerance: Maximum tolerable per-group degradation (default: 0.02).

    Returns:
        Dictionary detailing acceptance status, per-group records, and bounds.
    """
    grp_arr = np.asarray(groups)
    lbl_arr = np.asarray(labels)
    old_arr = np.asarray(old)
    new_arr = np.asarray(new)

    group_names = sorted(set(grp_arr))
    num_comparisons = len(group_names) + 1  # ALL + per-group
    allocated_alpha = alpha / num_comparisons
    records: list[dict[str, Any]] = []

    for name in ["ALL", *group_names]:
        mask = np.ones(len(lbl_arr), dtype=bool) if name == "ALL" else (grp_arr == name)
        known = mask & (lbl_arr >= 0)

        # Paired difference: (old_loss - new_loss)
        old_loss = (old_arr[known] != lbl_arr[known]).astype(int)
        new_loss = (new_arr[known] != lbl_arr[known]).astype(int)
        gain = int((old_loss - new_loss).sum())

        total_n = int(mask.sum())
        pending_n = total_n - int(known.sum())

        lower_bound = promotion_bound(gain, pending_n, total_n, allocated_alpha)
        required_gain = cost if name == "ALL" else -tolerance
        passes = lower_bound > required_gain

        records.append({
            "group": name,
            "total": total_n,
            "pending": pending_n,
            "lower_gain": lower_bound,
            "required_gain": required_gain,
            "passes": passes,
        })

    return {
        "accepted": all(r["passes"] for r in records),
        "groups": records,
        "scope": "One pre-specified candidate and held-out split; not a perpetual guarantee",
    }


def retrain(
    source_path: str,
    as_of: datetime,
    train_cutoff: datetime,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Train a real scikit-learn logistic shadow candidate from exported orders.

    Trains on orders purchased before `train_cutoff` and validates on a later
    disjoint holdout cohort up to `as_of`. Never modifies the deployed serving model.

    Args:
        source_path: Path to order JSON file.
        as_of: Upper evaluation cutoff time.
        train_cutoff: Boundary separating training purchases from holdout purchases.
        output_dir: Output directory to write candidate artifacts and report.

    Returns:
        Promotion report dictionary.

    Raises:
        ValueError: If cutoffs or training sample requirements are violated.
    """
    if train_cutoff >= as_of:
        raise ValueError("Training cutoff must strictly precede evaluation time")

    payload = json.loads(Path(source_path).read_text())
    orders = payload["orders"]

    if len({r["id"] for r in orders}) != len(orders):
        raise ValueError("Duplicate order IDs detected in source export")

    eligible = [
        r for r in orders
        if time(r["purchased"]) < as_of and time(r["created"]) <= as_of
    ]
    train = [
        r for r in eligible
        if time(r["purchased"]) < train_cutoff and r.get("reported") and time(r["reported"]) <= as_of
    ]
    valid = [
        r for r in eligible
        if time(r["purchased"]) >= train_cutoff
    ]

    if len(train) < 30 or not valid:
        raise ValueError("Need >= 30 labeled training orders and a non-empty later validation cohort")

    def extract_features(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([
            serving.vector(r["state"], time(r["purchased"]), time(r["due"]))
            for r in rows
        ])

    def extract_labels(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([
            int(time(r["delivered"]) >= time(r["due"]))
            if r.get("reported") and time(r["reported"]) <= as_of
            else -1
            for r in rows
        ])

    x_train = extract_features(train)
    y_train = extract_labels(train)

    if len(set(y_train)) != 2:
        raise ValueError("Training requires both on-time (0) and late (1) outcomes")

    scaler = StandardScaler().fit(x_train)
    fitted = LogisticRegression(C=1.0, max_iter=1000, random_state=17).fit(
        scaler.transform(x_train), y_train
    )

    coef = fitted.coef_[0] / scaler.scale_
    intercept = float(fitted.intercept_[0] - coef @ scaler.mean_)

    x_valid = extract_features(valid)
    logits = np.clip(x_valid @ coef + intercept, -700.0, 700.0)
    candidate_prob = 1.0 / (1.0 + np.exp(-logits))

    # Numerical consistency assertion against standard scikit-learn predict_proba
    sklearn_prob = fitted.predict_proba(scaler.transform(x_valid))[:, 1]
    assert np.max(np.abs(candidate_prob - sklearn_prob)) < 1e-12

    current_model = serving.MODEL
    baseline_pred = np.asarray([serving.probability(row) for row in x_valid]) >= current_model["threshold"]
    candidate_pred = candidate_prob >= current_model["threshold"]

    report = evaluate_gate(
        baseline_pred,
        candidate_pred,
        extract_labels(valid),
        [r["state"] for r in valid],
    )
    report.update({
        "training_orders": len(train),
        "validation_orders": len(valid),
        "as_of": as_of.isoformat(),
        "train_before": train_cutoff.isoformat(),
        "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "deployment": "No deployed artifact was changed",
        "export_scope": payload.get("scope", "unspecified"),
    })

    artifact = {
        **current_model,
        "version": f"shadow-{report['source_sha256'][:12]}",
        "coefficients": coef.tolist(),
        "intercept": intercept,
        "train_count": len(train),
        "training_cutoff": train_cutoff.isoformat(),
    }
    # Clear obsolete metrics from original base model
    for key in ("test_count", "average_precision", "brier", "test_prevalence", "test_period"):
        artifact.pop(key, None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "candidate-model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (out_path / "promotion-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """CLI entry point for training and gating shadow candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to exported orders JSON")
    parser.add_argument("--as-of", required=True, help="Evaluation cutoff (ISO)")
    parser.add_argument("--train-before", required=True, help="Training cutoff (ISO)")
    parser.add_argument("--out", required=True, help="Output directory for staged artifacts")
    args = parser.parse_args()

    result = retrain(args.input, time(args.as_of), time(args.train_before), args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
