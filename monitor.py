"""Local binary-prediction monitor. All API times are timezone-aware."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import AwareDatetime, BaseModel, Field, StrictBool

app = FastAPI(title="Delayed Label Monitor")


@contextmanager
def database():
    db = sqlite3.connect(os.getenv("MONITOR_DB", "monitor.sqlite"))
    try:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, segment TEXT, predicted INTEGER, issued REAL,
                outcome INTEGER, arrived REAL)""")
            yield db
    finally:
        db.close()


class Prediction(BaseModel):
    id: str = Field(min_length=1, max_length=150)
    segment: str = Field(min_length=1, max_length=100)
    predicted: StrictBool
    issued: AwareDatetime


class Outcome(BaseModel):
    id: str
    outcome: StrictBool
    arrived: AwareDatetime


@app.post("/predictions")
def predict(event: Prediction):
    values = (event.segment, event.predicted, event.issued.timestamp())
    with database() as db:
        db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,NULL,NULL)", (event.id, *values))
        saved = db.execute("SELECT segment,predicted,issued FROM events WHERE id=?", (event.id,)).fetchone()
        if saved != values:
            raise HTTPException(409, "Prediction ID already has different data")
    return {"id": event.id}


@app.post("/outcomes")
def resolve(event: Outcome):
    arrival = event.arrived.timestamp()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT issued,outcome,arrived FROM events WHERE id=?", (event.id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Submit prediction first")
        if arrival < row[0]:
            raise HTTPException(422, "Outcome precedes prediction")
        if row[1] is not None and row[1:] != (event.outcome, arrival):
            raise HTTPException(409, "Conflicting outcome")
        db.execute("UPDATE events SET outcome=?,arrived=? WHERE id=?", (event.outcome, arrival, event.id))
    return {"id": event.id}


@app.get("/metrics")
def metrics(at: AwareDatetime | None = None, since: AwareDatetime | None = None):
    cutoff = (at or datetime.now(timezone.utc)).timestamp()
    start = since.timestamp() if since else float("-inf")
    if start > cutoff:
        raise HTTPException(422, "Cohort start exceeds cutoff")
    with database() as db:
        rows = db.execute("""SELECT segment, COUNT(*),
            SUM(CASE WHEN arrived<=? THEN 1 ELSE 0 END),
            SUM(CASE WHEN arrived<=? AND predicted=outcome THEN 1 ELSE 0 END)
            FROM events WHERE issued BETWEEN ? AND ? GROUP BY segment ORDER BY segment""",
            (cutoff, cutoff, start, cutoff)).fetchall()
    return [dict(segment=g,total=n,pending=n-k,observed=c/k if k else None,
                 lower=c/n,upper=(c+n-k)/n) for g,n,k,c in rows]


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).with_name("dashboard.html"))


@app.get("/delivery-study", include_in_schema=False)
def delivery_study():
    return FileResponse(Path(__file__).parent / "delivery-results" / "report.html")
