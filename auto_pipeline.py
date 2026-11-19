"""Automated data integration and gated model replacement pipeline.
Supports scheduled batch ingest, stale-label anomaly audits, and prospective candidate staging.
Guarantees:
- Live model replacement requires explicit operator confirmation flag (--confirm-replace).
- Staged candidate models must pass the statistical evidence gate on held-out validation cohorts.
"""
import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from candidate_retrain import retrain, time

ROOT = Path(__file__).parent
DEFAULT_DB = os.environ.get('DELIVERY_DB', str(ROOT / 'delivery.sqlite'))

SCHEMA = '''CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, state TEXT NOT NULL,
 purchased TEXT NOT NULL, due TEXT NOT NULL, probability REAL NOT NULL,
 predicted INTEGER NOT NULL, model_version TEXT NOT NULL, created TEXT NOT NULL,
 delivered TEXT, reported TEXT)'''

def get_db(db_path=None):
    path = db_path or DEFAULT_DB
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    return db

def ingest_from_dict(payload, db_path=None):
    """
    Ingests orders and outcomes from a dict into the SQLite database atomically.
    """
    from serving import vector, probability, iso, MODEL
    db = get_db(db_path)
    orders_inserted = 0
    outcomes_updated = 0
    now = datetime.now(timezone.utc)
    now_iso = iso(now)

    with db:
        orders = payload.get('orders', [])
        for o in orders:
            p_time = datetime.fromisoformat(o['purchased'].replace('Z', '+00:00'))
            d_time = datetime.fromisoformat(o['due'].replace('Z', '+00:00'))
            if p_time > now:
                continue
            prob = probability(vector(o['state'], p_time, d_time))
            cur = db.execute(
                'INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)',
                (o['id'], o['state'], iso(p_time), iso(d_time), prob,
                 int(prob >= MODEL['threshold']), MODEL['version'], now_iso)
            )
            orders_inserted += cur.rowcount

        outcomes = payload.get('outcomes', [])
        for oc in outcomes:
            del_time = datetime.fromisoformat(oc['delivered'].replace('Z', '+00:00'))
            del_iso = iso(del_time)
            cur = db.execute(
                'UPDATE orders SET delivered=?, reported=COALESCE(reported,?) '
                'WHERE id=? AND delivered IS NULL AND purchased <= ?',
                (del_iso, now_iso, oc['id'], del_iso)
            )
            outcomes_updated += cur.rowcount

    db.close()
    return dict(orders_inserted=orders_inserted, outcomes_updated=outcomes_updated)

def check_stale_orders(db_path=None, stale_days=14):
    """
    Identifies pending orders that have exceeded their deadline by more than `stale_days`.
    """
    db = get_db(db_path)
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=stale_days)).isoformat()
    with db:
        rows = db.execute(
            'SELECT id, state, purchased, due FROM orders WHERE delivered IS NULL AND due < ? ORDER BY due ASC',
            (cutoff,)
        ).fetchall()
    db.close()
    return [dict(r) for r in rows]

def run_gated_promotion(input_json, as_of_str, train_before_str, stage_dir='staged', confirm_replace=False):
    """
    Evaluates candidate retraining on input_json.
    If gate passes, stages candidate model in stage_dir.
    If confirm_replace is True, promotes to serving-model.json with a backup.
    """
    as_of = time(as_of_str)
    train_before = time(train_before_str)
    stage_path = ROOT / stage_dir
    stage_path.mkdir(parents=True, exist_ok=True)

    report = retrain(input_json, as_of, train_before, stage_path)
    is_accepted = report.get('accepted', False)

    action_taken = 'none'
    if is_accepted:
        if confirm_replace:
            live_model_path = ROOT / 'serving-model.json'
            backup_path = ROOT / 'serving-model.backup.json'
            candidate_path = stage_path / 'candidate-model.json'
            if live_model_path.exists():
                shutil.copy2(live_model_path, backup_path)
            shutil.copy2(candidate_path, live_model_path)
            action_taken = 'promoted_to_live'
        else:
            action_taken = 'staged_for_review'
    else:
        action_taken = 'rejected_by_gate'

    report['action_taken'] = action_taken
    report['confirm_replace_flag'] = confirm_replace
    (stage_path / 'promotion-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ingest', help='Path to JSON file with orders/outcomes')
    parser.add_argument('--stale-check', action='store_true', help='Audit stale pending orders')
    parser.add_argument('--run-promotion', action='store_true', help='Run prospective shadow candidate training')
    parser.add_argument('--input', help='Exported orders JSON for promotion')
    parser.add_argument('--as-of', help='Evaluation cutoff timestamp (ISO)')
    parser.add_argument('--train-before', help='Training cutoff timestamp (ISO)')
    parser.add_argument('--confirm-replace', action='store_true', help='Explicitly promote candidate to live serving-model.json')
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

if __name__ == '__main__':
    main()
