"""Delivery inference and persistent delayed-outcome API service.

Provides FastAPI endpoints for operational delivery risk prediction,
immutable delayed outcome recording, diagnostic monitoring, and batch ingestion.
"""

from __future__ import annotations
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import AwareDatetime, BaseModel, Field

ROOT = Path(__file__).parent
MODEL: dict[str, Any] = json.loads((ROOT / "serving-model.json").read_text())

app = FastAPI(
    title="Delayed Label Monitor — Delivery Operations API",
    description="Operational prediction serving, outcome recording, and non-parametric drift diagnostics.",
)

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


@contextmanager
def database() -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing an active SQLite connection with initialized schema.

    Yields:
        sqlite3.Connection: Database connection configured with sqlite3.Row factory.
    """
    db_path = os.environ.get("DELIVERY_DB", "delivery.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute(DATABASE_SCHEMA)
            yield conn
    finally:
        conn.close()


class Order(BaseModel):
    """Input payload for scoring and registering a customer delivery order."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$", description="Unique non-identifying order identifier")
    state: str = Field(description="Two-letter destination state abbreviation (e.g. 'SP')")
    purchased: AwareDatetime = Field(description="Order purchase timestamp with timezone")
    due: AwareDatetime = Field(description="Promised delivery deadline timestamp with timezone")


class Delivery(BaseModel):
    """Input payload for confirming an actual delivery event."""

    id: str = Field(description="Order identifier corresponding to an existing order")
    delivered: AwareDatetime = Field(description="Actual delivery timestamp with timezone")


def vector(state: str, purchased: datetime, due: datetime) -> list[float]:
    """Construct the feature vector from raw order parameters matching serving model features.

    Args:
        state: Destination state abbreviation.
        purchased: Purchase datetime.
        due: Delivery deadline datetime.

    Returns:
        List of numerical feature values aligned with MODEL['features'].

    Raises:
        HTTPException: If state is unsupported or delivery promise window is outside (0, 180] days.
    """
    purchased_utc = purchased.astimezone(timezone.utc)
    promise_days = (due - purchased_utc).total_seconds() / 86400.0

    if state not in MODEL["states"] or not (0.0 < promise_days <= 180.0):
        raise HTTPException(
            status_code=422,
            detail="Choose a supported state and a deadline within 180 days after purchase",
        )

    values = {
        "promise_days": promise_days,
        "dow": purchased_utc.weekday(),
        "month": purchased_utc.month,
        "hour": purchased_utc.hour,
        f"customer_state_{state}": 1.0,
    }
    return [float(values.get(f, 0.0)) for f in MODEL["features"]]


def probability(features: list[float]) -> float:
    """Calculate late-delivery probability via standard logistic sigmoid.

    Args:
        features: Numerical feature vector.

    Returns:
        Probability of late delivery in [0, 1].
    """
    logit = MODEL["intercept"] + sum(w * x for w, x in zip(MODEL["coefficients"], features))
    return 1.0 / (1.0 + math.exp(-logit))


def iso(dt: datetime) -> str:
    """Format a timezone-aware datetime into UTC ISO 8601 millisecond string.

    Args:
        dt: Datetime object.

    Returns:
        ISO 8601 formatted string ending in 'Z'.
    """
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_model() -> dict[str, Any]:
    """Reload the active serving model parameters in place from disk.

    Mutates the existing global MODEL dictionary in place to ensure that all
    external modules importing MODEL observe the updated parameters immediately.

    Returns:
        Updated dictionary of model parameters.
    """
    new_model = json.loads((ROOT / "serving-model.json").read_text())
    MODEL.clear()
    MODEL.update(new_model)
    return MODEL


@app.get("/api/model")
def get_model_metadata() -> dict[str, Any]:
    """Return descriptive metadata about the deployed prediction model."""
    excluded = {"coefficients", "intercept", "features"}
    return {k: v for k, v in MODEL.items() if k not in excluded}


@app.post("/api/model/reload")
def reload_model_endpoint() -> dict[str, Any]:
    """Reload model parameters in place from serving-model.json."""
    updated = load_model()
    return {"reloaded": True, "version": updated["version"]}


@app.post("/api/orders")
def create_order(order: Order) -> dict[str, Any]:
    """Score a delivery order and persist the prediction record idempotently."""
    now_utc = datetime.now(timezone.utc)
    if order.purchased > now_utc:
        raise HTTPException(status_code=422, detail="Purchase cannot be in the future")

    feat_vec = vector(order.state, order.purchased, order.due)
    late_prob = probability(feat_vec)
    predicted_flag = int(late_prob >= MODEL["threshold"])

    with database() as db:
        db.execute(
            "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
            (
                order.id,
                order.state,
                iso(order.purchased),
                iso(order.due),
                late_prob,
                predicted_flag,
                MODEL["version"],
                iso(now_utc),
            ),
        )
        row = dict(db.execute("SELECT * FROM orders WHERE id=?", (order.id,)).fetchone())

        if (row["state"], row["purchased"], row["due"]) != (order.state, iso(order.purchased), iso(order.due)):
            raise HTTPException(status_code=409, detail="Order ID already exists with different input")

    return row


@app.post("/api/outcomes")
def record_outcome(event: Delivery) -> dict[str, Any]:
    """Record an immutable actual delivery timestamp for an existing order."""
    now_utc = datetime.now(timezone.utc)
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM orders WHERE id=?", (event.id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Order not found")

        delivered_iso = iso(event.delivered)
        if delivered_iso < row["purchased"] or event.delivered > now_utc:
            raise HTTPException(status_code=422, detail="Delivery must be between purchase and now")

        if row["delivered"] and row["delivered"] != delivered_iso:
            raise HTTPException(status_code=409, detail="This order already has a different outcome")

        db.execute(
            "UPDATE orders SET delivered=?, reported=COALESCE(reported,?) WHERE id=?",
            (delivered_iso, iso(now_utc), event.id),
        )
        updated = dict(db.execute("SELECT * FROM orders WHERE id=?", (event.id,)).fetchone())
        return updated


@app.post("/api/batch-orders")
def batch_orders(orders: list[Order]) -> dict[str, int]:
    """Atomically ingest an array of orders, skipping future dates and duplicate IDs."""
    now_utc = datetime.now(timezone.utc)
    now_iso = iso(now_utc)
    inserted_count = 0

    with database() as db:
        for order in orders:
            if order.purchased > now_utc:
                continue
            feat_vec = vector(order.state, order.purchased, order.due)
            prob = probability(feat_vec)
            pred = int(prob >= MODEL["threshold"])

            cursor = db.execute(
                "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
                (
                    order.id,
                    order.state,
                    iso(order.purchased),
                    iso(order.due),
                    prob,
                    pred,
                    MODEL["version"],
                    now_iso,
                ),
            )
            inserted_count += cursor.rowcount

    return {"received": len(orders), "inserted": inserted_count}


@app.post("/api/batch-outcomes")
def batch_outcomes(outcomes: list[Delivery]) -> dict[str, int]:
    """Atomically record delivery outcomes, strictly rejecting dates in the future."""
    now_utc = datetime.now(timezone.utc)
    now_iso = iso(now_utc)
    updated_count = 0

    with database() as db:
        for ev in outcomes:
            # Strictly reject future delivery dates
            if ev.delivered > now_utc:
                continue
            del_iso = iso(ev.delivered)

            # Ensure delivery is between purchase and now
            cursor = db.execute(
                "UPDATE orders SET delivered=?, reported=COALESCE(reported,?) "
                "WHERE id=? AND delivered IS NULL AND purchased <= ? AND ? <= ?",
                (del_iso, now_iso, ev.id, del_iso, del_iso, now_iso),
            )
            updated_count += cursor.rowcount

    return {"received": len(outcomes), "updated": updated_count}


@app.get("/api/orders")
def get_orders() -> list[dict[str, Any]]:
    """Retrieve the 200 most recently created orders."""
    with database() as db:
        rows = db.execute("SELECT * FROM orders ORDER BY created DESC, id LIMIT 200").fetchall()
        return [dict(r) for r in rows]


@app.get("/api/export")
def export_orders(state: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
    """Export all recorded orders for offline validation and shadow retraining."""
    query = "SELECT * FROM orders"
    params: list[Any] = []
    if state:
        query += " WHERE state = ?"
        params.append(state)
    query += " ORDER BY created ASC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with database() as db:
        rows = [dict(r) for r in db.execute(query, params).fetchall()]

    return {"orders": rows, "total": len(rows), "scope": "operational_export"}


@app.get("/api/metrics")
def get_metrics() -> list[dict[str, Any]]:
    """Calculate empirical and bounded accuracy per geographic state."""
    with database() as db:
        rows = db.execute("""
            SELECT state, COUNT(*) as n, COUNT(delivered) as resolved,
                   SUM(CASE WHEN delivered IS NOT NULL AND predicted = (delivered >= due) THEN 1 ELSE 0 END) as correct
            FROM orders GROUP BY state ORDER BY state
        """).fetchall()

    return [
        {
            "state": r["state"],
            "total": r["n"],
            "pending": r["n"] - r["resolved"],
            "observed": (r["correct"] / r["resolved"]) if r["resolved"] else None,
            "lower": r["correct"] / r["n"],
            "upper": (r["correct"] + r["n"] - r["resolved"]) / r["n"],
        }
        for r in rows
    ]


@app.get("/api/diagnostics")
def get_diagnostics() -> dict[str, Any]:
    """Run once-daily fixed-cohort operational diagnostics across all states."""
    from operational_diagnostics import diagnose

    with database() as db:
        records = [dict(r) for r in db.execute("SELECT * FROM orders").fetchall()]
        return diagnose(records)


@app.get("/api/pipeline-status")
def get_pipeline_status() -> dict[str, Any]:
    """Retrieve ingestion health, pending counts, stale orders, and staged candidate model status."""
    now_utc = datetime.now(timezone.utc)
    now_iso = iso(now_utc)

    with database() as db:
        total = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        resolved = db.execute("SELECT COUNT(*) FROM orders WHERE delivered IS NOT NULL").fetchone()[0]

        stale_threshold = datetime.fromtimestamp(now_utc.timestamp() - 14 * 86400, timezone.utc)
        stale_iso = iso(stale_threshold)
        stale_count = db.execute(
            "SELECT COUNT(*) FROM orders WHERE delivered IS NULL AND due < ?",
            (stale_iso,),
        ).fetchone()[0]

    staged_report = None
    report_file = ROOT / "staged" / "promotion-report.json"
    if report_file.exists():
        try:
            staged_report = json.loads(report_file.read_text())
        except Exception:
            pass

    return {
        "as_of": now_iso,
        "live_model_version": MODEL["version"],
        "total_orders": total,
        "resolved_orders": resolved,
        "pending_orders": total - resolved,
        "stale_orders": stale_count,
        "staged_candidate": staged_report is not None,
        "staged_report": staged_report,
        "gated_replacement_policy": "Operator-confirmed shadow promotion; no automatic hot-swapping.",
    }


# Static web UI routes
@app.get("/")
def home() -> FileResponse:
    """Serve the primary interactive operations workbench."""
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/delivery-study.html")
def delivery_study() -> FileResponse:
    """Serve the historical Olist logistics benchmark report."""
    return FileResponse(ROOT / "delivery-results" / "report.html")


@app.get("/research.json")
def research_data() -> FileResponse:
    """Serve the raw JSON dataset from the detection study."""
    return FileResponse(ROOT / "robust-detection-results.json")


@app.get("/research-study.html")
def research_study() -> FileResponse:
    """Serve the comprehensive research findings interactive study."""
    return FileResponse(ROOT / "research" / "report.html")
