"""Automated data integration and gated model replacement pipeline.

Supports scheduled batch order ingestion, stale-label anomaly audits,
and prospective shadow candidate training with statistical evidence gates.

Guarantees:
- Live model replacement requires explicit operator confirmation (--confirm-replace).
- Staged candidate models must pass the statistical evidence gate on held-out validation cohorts.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from candidate_retrain import retrain, time

ROOT = Path(__file__).parent
DEFAULT_DB = os.environ.get("DELIVERY_DB", str(ROOT / "delivery.sqlite"))

DATABASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    purchased TEXT NOT NULL,
    due TEXT NOT NULL,
    probability REAL NOT NULL,
    predicted INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created TEXT NOT NULL,
    delivered TEXT,
    reported TEXT
);
"""


def get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open and initialize a SQLite database connection with row factory.

    Args:
        db_path: Optional path to the database file; defaults to DELIVERY_DB.

    Returns:
        Configured sqlite3.Connection object.
    """
    path = db_path or DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(DATABASE_SCHEMA)
    return conn


def ingest_from_dict(
    payload: dict[str, list[dict[str, Any]]],
    db_path: Optional[str] = None,
) -> dict[str, int]:
    """Ingest orders and outcomes from an in-memory dictionary atomically.

    Args:
        payload: Dictionary containing 'orders' and/or 'outcomes' lists.
        db_path: Optional SQLite database file path.

    Returns:
        Dictionary with 'orders_inserted' and 'outcomes_updated' counts.
    """
    from serving import MODEL, iso, probability, vector

    conn = get_db(db_path)
    orders_inserted = 0
    outcomes_updated = 0
    now_utc = datetime.now(timezone.utc)
    now_iso = iso(now_utc)

    with conn:
        orders = payload.get("orders", [])
        for order in orders:
            p_time = datetime.fromisoformat(order["purchased"].replace("Z", "+00:00"))
            d_time = datetime.fromisoformat(order["due"].replace("Z", "+00:00"))
            if p_time > now_utc:
                continue
            prob = probability(vector(order["state"], p_time, d_time))
            cursor = conn.execute(
                "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
                (
                    order["id"],
                    order["state"],
                    iso(p_time),
                    iso(d_time),
                    prob,
                    int(prob >= MODEL["threshold"]),
                    MODEL["version"],
                    now_iso,
                ),
            )
            orders_inserted += cursor.rowcount

        outcomes = payload.get("outcomes", [])
        for oc in outcomes:
            del_time = datetime.fromisoformat(oc["delivered"].replace("Z", "+00:00"))
            # Strictly reject future delivery dates
            if del_time > now_utc:
                continue
            del_iso = iso(del_time)
            cursor = conn.execute(
                "UPDATE orders SET delivered=?, reported=COALESCE(reported,?) "
                "WHERE id=? AND delivered IS NULL AND purchased <= ? AND ? <= ?",
                (del_iso, now_iso, oc["id"], del_iso, del_iso, now_iso),
            )
            outcomes_updated += cursor.rowcount

    conn.close()
    return {"orders_inserted": orders_inserted, "outcomes_updated": outcomes_updated}


def check_stale_orders(
    db_path: Optional[str] = None,
    stale_days: int = 14,
) -> list[dict[str, Any]]:
    """Identify pending orders that have exceeded their delivery deadline by more than stale_days.

    Args:
        db_path: Optional database file path.
        stale_days: Number of days past due required to flag an order as stale.

    Returns:
        List of dictionaries representing stale unresolved orders.
    """
    conn = get_db(db_path)
    now_utc = datetime.now(timezone.utc)
    cutoff = (now_utc - timedelta(days=stale_days)).isoformat()
    with conn:
        rows = conn.execute(
            "SELECT id, state, purchased, due FROM orders WHERE delivered IS NULL AND due < ? ORDER BY due ASC",
            (cutoff,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_gated_promotion(
    input_json: str,
    as_of_str: str,
    train_before_str: str,
    stage_dir: str = "staged",
    confirm_replace: bool = False,
) -> dict[str, Any]:
    """Train a shadow logistic model on historical data and evaluate against holdout gate.

    If the candidate passes statistical evidence bounds:
    - Stages candidate model and report into stage_dir.
    - If confirm_replace is explicitly True, atomically backups serving-model.json and promotes.
    - Otherwise leaves the live model untouched.

    Args:
        input_json: Path to JSON export containing order records.
        as_of_str: Evaluation cutoff timestamp string in ISO format.
        train_before_str: Training cutoff timestamp string in ISO format.
        stage_dir: Directory where approved candidates and reports are staged.
        confirm_replace: Explicit operator flag allowing live model replacement.

    Returns:
        Dictionary containing the full promotion evaluation report.
    """
    as_of = time(as_of_str)
    train_before = time(train_before_str)
    stage_path = ROOT / stage_dir
    stage_path.mkdir(parents=True, exist_ok=True)

    report = retrain(input_json, as_of, train_before, stage_path)
    is_accepted = report.get("accepted", False)

    action_taken = "none"
    if is_accepted:
        if confirm_replace:
            live_model_path = ROOT / "serving-model.json"
            backup_path = ROOT / "serving-model.backup.json"
            candidate_path = stage_path / "candidate-model.json"
            if live_model_path.exists():
                shutil.copy2(live_model_path, backup_path)
            shutil.copy2(candidate_path, live_model_path)
            action_taken = "promoted_to_live"

            # Attempt live in-process reload if serving is in the same runtime
            try:
                from serving import load_model
                load_model()
            except Exception:
                pass
        else:
            action_taken = "staged_for_review"
    else:
        action_taken = "rejected_by_gate"

    report["action_taken"] = action_taken
    report["confirm_replace_flag"] = confirm_replace
    (stage_path / "promotion-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """CLI entry point for pipeline operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", help="Path to JSON file with orders/outcomes")
    parser.add_argument("--stale-check", action="store_true", help="Audit stale pending orders")
    parser.add_argument("--run-promotion", action="store_true", help="Run prospective shadow candidate training")
    parser.add_argument("--input", help="Exported orders JSON for promotion")
    parser.add_argument("--as-of", help="Evaluation cutoff timestamp (ISO)")
    parser.add_argument("--train-before", help="Training cutoff timestamp (ISO)")
    parser.add_argument("--confirm-replace", action="store_true", help="Explicitly promote candidate to live serving-model.json")
    args = parser.parse_args()

    if args.ingest:
        data = json.loads(Path(args.ingest).read_text())
        res = ingest_from_dict(data)
        print("Ingest result:", json.dumps(res, indent=2))

    if args.stale_check:
        stale = check_stale_orders()
        print(f"Found {len(stale)} stale pending orders:")
        for s in stale[:10]:
            print(f"  Order {s['id']} (State: {s['state']}) due {s['due']}")

    if args.run_promotion:
        if not (args.input and args.as_of and args.train_before):
            parser.error("--run-promotion requires --input, --as-of, and --train-before")
        rep = run_gated_promotion(args.input, args.as_of, args.train_before, confirm_replace=args.confirm_replace)
        print("Promotion Gate Report:")
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
