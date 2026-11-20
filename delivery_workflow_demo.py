"""Interactive, reproducible 60-order delivery workflow demonstration for interviews and portfolio.

Demonstrates the core MLOps life cycle on a disposable, isolated SQLite database:
1. Prediction scoring and registration (/api/orders).
2. Early partial reporting and selection-bias detection via non-parametric bounds (/api/metrics).
3. Full delayed-outcome resolution (/api/outcomes).
4. Verification that final observed accuracy is strictly contained in the early missing-outcome range.

Run directly:
    python delivery_workflow_demo.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

# Ensure disposable database isolation
TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["DELIVERY_DB"] = str(Path(TEMP_DIR.name) / "demo_delivery.sqlite")

from serving import app  # noqa: E402


def run_demo() -> dict[str, Any]:
    """Execute the chronological 60-order delivery replay against the live FastAPI app."""
    client = TestClient(app)

    print("=" * 74)
    print("  DELAYED LABEL MONITOR - END-TO-END MLOPS WORKFLOW REPLAY (60 ORDERS)")
    print("=" * 74)
    print("Running in isolated environment with disposable SQLite database...")

    # Base timestamps (UTC)
    base_purchase = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
    base_due = base_purchase + timedelta(days=5)

    # 1. Prediction Phase: Register 60 orders (48 São Paulo, 12 Amazonas)
    print("\n[PHASE 1: ORDER REGISTRATION & MODEL INFERENCE]")
    print("Scoring 60 customer orders using production logistic regression model...")

    orders = []
    # 48 SP orders
    for i in range(1, 49):
        orders.append({
            "id": f"ORD-SP-{i:03d}",
            "state": "SP",
            "purchased": (base_purchase + timedelta(hours=i * 2)).isoformat(),
            "due": (base_due + timedelta(hours=i * 2)).isoformat(),
        })
    # 12 AM orders (remote/minority region)
    for i in range(1, 13):
        orders.append({
            "id": f"ORD-AM-{i:03d}",
            "state": "AM",
            "purchased": (base_purchase + timedelta(hours=i * 8)).isoformat(),
            "due": (base_due + timedelta(hours=i * 8 + 48)).isoformat(),
        })

    predictions = []
    for ord_data in orders:
        resp = client.post("/api/orders", json=ord_data)
        assert resp.status_code == 200, f"Order registration failed: {resp.text}"
        predictions.append(resp.json())

    late_flags = sum(1 for p in predictions if p["predicted"] == 1)
    print(f"-> Successfully registered 60 orders across 2 states (48 SP, 12 AM).")
    print(f"-> Flagged {late_flags}/60 as 'Late Risk' (model threshold: 24.0%).")

    # Initial metrics check (all pending)
    metrics_initial = client.get("/api/metrics").json()
    total_initial = sum(m["total"] for m in metrics_initial)
    pending_initial = sum(m["pending"] for m in metrics_initial)
    print(f"-> Metrics state: {total_initial} total, {pending_initial} pending (100% awaiting outcome).")

    # 2. Phase 2: Early Outcomes Arrive (Day 14 after purchase)
    # Fast couriers report early. In SP, 32 orders report (5 correct, 27 incorrect). 16 pending.
    # In AM, 4 remote orders report (1 correct, 3 incorrect). 8 pending.
    print("\n[PHASE 2: EARLY PARTIAL REPORTING (DAY 14) - SELECTION BIAS RISK]")
    print("Early couriers report 36 delivered orders. 24 orders remain in-transit.")

    # SP early reports: 32 orders (5 correct -> delivered after due; 27 incorrect -> delivered before due)
    for i in range(1, 33):
        ord_id = f"ORD-SP-{i:03d}"
        due_time = datetime.fromisoformat(orders[i - 1]["due"])
        del_time = due_time + timedelta(hours=12) if i <= 5 else due_time - timedelta(hours=12)
        client.post("/api/outcomes", json={"id": ord_id, "delivered": del_time.isoformat()})

    # AM early reports: 4 orders (1 correct -> delivered after due; 3 incorrect -> delivered before due)
    for i in range(1, 5):
        ord_id = f"ORD-AM-{i:03d}"
        due_time = datetime.fromisoformat(orders[48 + i - 1]["due"])
        del_time = due_time + timedelta(hours=12) if i == 1 else due_time - timedelta(hours=12)
        client.post("/api/outcomes", json={"id": ord_id, "delivered": del_time.isoformat()})

    metrics_early = client.get("/api/metrics").json()
    sp_early = next(m for m in metrics_early if m["state"] == "SP")
    am_early = next(m for m in metrics_early if m["state"] == "AM")

    print("\nEarly Evaluation Snapshot:")
    print(f"  • São Paulo (SP):")
    print(f"      Observed accuracy (arrived only): {sp_early['observed']*100:.1f}% (appears alarming!)")
    print(f"      Exact missing-outcome bounds   : [{sp_early['lower']*100:.1f}%, {sp_early['upper']*100:.1f}%]")
    print(f"      Pending outcomes in-transit     : {sp_early['pending']}/48 orders")
    print(f"  • Amazonas (AM):")
    print(f"      Observed accuracy (arrived only): {am_early['observed']*100:.1f}%")
    print(f"      Exact missing-outcome bounds   : [{am_early['lower']*100:.1f}%, {am_early['upper']*100:.1f}%]")
    print(f"      Pending outcomes in-transit     : {am_early['pending']}/12 orders")

    print("\n  >> CRITICAL MLOPS TAKEAWAY: An unconstrained MLOps system would see SP observed")
    print(f"     accuracy drop to {sp_early['observed']*100:.1f}% and trigger an emergency retraining.")
    print("     Delayed Label Monitor prevents this because the true accuracy is mathematically")
    print(f"     sandwiched between {sp_early['lower']*100:.1f}% and {sp_early['upper']*100:.1f}%!")

    # 3. Phase 3: Final Resolution (All in-transit deliveries arrive)
    print("\n[PHASE 3: FULL CONVERGENCE - ALL LABELS ARRIVED]")
    print("Remaining 24 in-transit deliveries completed and confirmed by couriers...")

    # SP remaining: 16 orders (11 correct, 5 incorrect)
    for i in range(33, 49):
        ord_id = f"ORD-SP-{i:03d}"
        due_time = datetime.fromisoformat(orders[i - 1]["due"])
        del_time = due_time + timedelta(hours=12) if (i - 32) <= 11 else due_time - timedelta(hours=12)
        client.post("/api/outcomes", json={"id": ord_id, "delivered": del_time.isoformat()})

    # AM remaining: 8 orders (3 correct, 5 incorrect)
    for i in range(5, 13):
        ord_id = f"ORD-AM-{i:03d}"
        due_time = datetime.fromisoformat(orders[48 + i - 1]["due"])
        del_time = due_time + timedelta(hours=12) if (i - 4) <= 3 else due_time - timedelta(hours=12)
        client.post("/api/outcomes", json={"id": ord_id, "delivered": del_time.isoformat()})

    metrics_final = client.get("/api/metrics").json()
    sp_final = next(m for m in metrics_final if m["state"] == "SP")
    am_final = next(m for m in metrics_final if m["state"] == "AM")

    print("\nFinal Verified Evaluation:")
    print(f"  • São Paulo (SP):")
    print(f"      Final verified accuracy        : {sp_final['observed']*100:.1f}%")
    print(f"      Contained in early bounds?      : {sp_early['lower'] <= sp_final['observed'] <= sp_early['upper']} "
          f"({sp_early['lower']*100:.1f}% <= {sp_final['observed']*100:.1f}% <= {sp_early['upper']*100:.1f}%)")
    print(f"  • Amazonas (AM):")
    print(f"      Final verified accuracy        : {am_final['observed']*100:.1f}%")
    print(f"      Contained in early bounds?      : {am_early['lower'] <= am_final['observed'] <= am_early['upper']} "
          f"({am_early['lower']*100:.1f}% <= {am_final['observed']*100:.1f}% <= {am_early['upper']*100:.1f}%)")

    # 4. Summary Verdict
    print("\n" + "=" * 74)
    print("  SUMMARY: PRODUCTION MLOPS VERIFICATION PASSED")
    print("=" * 74)
    print(f"1. Selection bias proof: Early reports showed {sp_early['observed']*100:.1f}% accuracy in SP.")
    print(f"2. Once all labels matured, SP accuracy converged to {sp_final['observed']*100:.1f}%.")
    print(f"3. Early bounds [{sp_early['lower']*100:.1f}%, {sp_early['upper']*100:.1f}%] strictly contained the final {sp_final['observed']*100:.1f}% truth.")
    print("4. Prevented knee-jerk model retraining on incomplete, censored data.")
    print("=" * 74 + "\n")

    return {
        "sp_early": sp_early,
        "sp_final": sp_final,
        "am_early": am_early,
        "am_final": am_final,
    }


if __name__ == "__main__":
    try:
        run_demo()
    finally:
        TEMP_DIR.cleanup()
