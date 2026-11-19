"""Delivery inference and persistent delayed-outcome API. Run uvicorn serving:app."""
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, AwareDatetime

ROOT=Path(__file__).parent
MODEL=json.loads((ROOT/'serving-model.json').read_text())
app=FastAPI(title='Delayed Label Monitor — delivery operations')
SCHEMA='''CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, state TEXT NOT NULL,
 purchased TEXT NOT NULL, due TEXT NOT NULL, probability REAL NOT NULL,
 predicted INTEGER NOT NULL, model_version TEXT NOT NULL, created TEXT NOT NULL,
 delivered TEXT, reported TEXT)'''

@contextmanager
def database():
    db=sqlite3.connect(os.environ.get('DELIVERY_DB','delivery.sqlite'))
    db.row_factory=sqlite3.Row
    try:
        with db:
            db.execute(SCHEMA)
            yield db
    finally:
        db.close()

class Order(BaseModel):
    id: str=Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    state: str
    purchased: AwareDatetime
    due: AwareDatetime

class Delivery(BaseModel):
    id: str
    delivered: AwareDatetime


def vector(state,purchased,due):
    purchased=purchased.astimezone(timezone.utc)
    days=(due-purchased).total_seconds()/86400
    if state not in MODEL['states'] or not 0<days<=180:
        raise HTTPException(422,'Choose a supported state and a deadline within 180 days after purchase')
    values={'promise_days':days,'dow':purchased.weekday(),'month':purchased.month,'hour':purchased.hour,
            'customer_state_'+state:1}
    return [values.get(f,0) for f in MODEL['features']]


def probability(x):
    z=MODEL['intercept']+sum(a*b for a,b in zip(x,MODEL['coefficients']))
    return 1/(1+math.exp(-z))


def iso(t):
    return t.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')

@app.get('/api/model')
def model():
    return {k:v for k,v in MODEL.items() if k not in ('coefficients','intercept','features')}

@app.post('/api/orders')
def create(order:Order):
    if order.purchased>datetime.now(timezone.utc):
        raise HTTPException(422,'Purchase cannot be in the future')
    p=probability(vector(order.state,order.purchased,order.due))
    with database() as db:
        db.execute('INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)',
                   (order.id,order.state,iso(order.purchased),iso(order.due),p,int(p>=MODEL['threshold']),
                    MODEL['version'],iso(datetime.now(timezone.utc))))
        row=dict(db.execute('SELECT * FROM orders WHERE id=?',(order.id,)).fetchone())
        if (row['state'],row['purchased'],row['due'])!=(order.state,iso(order.purchased),iso(order.due)):
            raise HTTPException(409,'Order ID already exists with different input')
    return row

@app.post('/api/outcomes')
def outcome(event:Delivery):
    now=datetime.now(timezone.utc)
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM orders WHERE id=?',(event.id,)).fetchone()
        if row is None: raise HTTPException(404,'Order not found')
        if iso(event.delivered)<row['purchased'] or event.delivered>now:
            raise HTTPException(422,'Delivery must be between purchase and now')
        if row['delivered'] and row['delivered']!=iso(event.delivered):
            raise HTTPException(409,'This order already has a different outcome')
        db.execute('UPDATE orders SET delivered=?,reported=COALESCE(reported,?) WHERE id=?',
                   (iso(event.delivered),iso(now),event.id))
        return dict(db.execute('SELECT * FROM orders WHERE id=?',(event.id,)).fetchone())

@app.get('/api/orders')
def orders():
    with database() as db:
        return [dict(r) for r in db.execute('SELECT * FROM orders ORDER BY created DESC,id LIMIT 200')]

@app.get('/api/metrics')
def metrics():
    with database() as db:
        rows=db.execute('''SELECT state,COUNT(*) n,COUNT(delivered) resolved,
          SUM(CASE WHEN delivered IS NOT NULL AND predicted=(delivered>=due) THEN 1 ELSE 0 END) correct
          FROM orders GROUP BY state ORDER BY state''').fetchall()
    return [dict(state=r['state'],total=r['n'],pending=r['n']-r['resolved'],
                 observed=r['correct']/r['resolved'] if r['resolved'] else None,
                 lower=r['correct']/r['n'],upper=(r['correct']+r['n']-r['resolved'])/r['n']) for r in rows]

@app.get('/')
def home(): return FileResponse(ROOT/'web/index.html')

@app.get('/delivery-study.html')
def study(): return FileResponse(ROOT/'delivery-results/report.html')

@app.get('/research.json')
def research(): return FileResponse(ROOT/'robust-detection-results.json')


@app.get('/api/diagnostics')
def diagnostics():
    from operational_diagnostics import diagnose
    with database() as db:
        return diagnose([dict(r) for r in db.execute('SELECT * FROM orders')])


@app.get('/api/export')
def export_orders(state: str = None, limit: int = None):
    query = 'SELECT * FROM orders'
    params = []
    if state:
        query += ' WHERE state = ?'
        params.append(state)
    query += ' ORDER BY created ASC'
    if limit:
        query += ' LIMIT ?'
        params.append(limit)
    with database() as db:
        rows = [dict(r) for r in db.execute(query, params).fetchall()]
    return dict(orders=rows, total=len(rows), scope='operational_export')


@app.post('/api/batch-orders')
def batch_orders(orders: list[Order]):
    now = datetime.now(timezone.utc)
    now_iso = iso(now)
    created_count = 0
    with database() as db:
        for order in orders:
            if order.purchased > now:
                continue
            p = probability(vector(order.state, order.purchased, order.due))
            cursor = db.execute(
                'INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)',
                (order.id, order.state, iso(order.purchased), iso(order.due), p,
                 int(p >= MODEL['threshold']), MODEL['version'], now_iso)
            )
            created_count += cursor.rowcount
    return dict(received=len(orders), inserted=created_count)


@app.post('/api/batch-outcomes')
def batch_outcomes(outcomes: list[Delivery]):
    now = datetime.now(timezone.utc)
    now_iso = iso(now)
    updated_count = 0
    with database() as db:
        for ev in outcomes:
            del_iso = iso(ev.delivered)
            cursor = db.execute(
                'UPDATE orders SET delivered=?, reported=COALESCE(reported,?) '
                'WHERE id=? AND delivered IS NULL AND purchased <= ?',
                (del_iso, now_iso, ev.id, del_iso)
            )
            updated_count += cursor.rowcount
    return dict(received=len(outcomes), updated=updated_count)


@app.get('/api/pipeline-status')
def pipeline_status():
    now = datetime.now(timezone.utc)
    now_iso = iso(now)
    with database() as db:
        total = db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]
        resolved = db.execute('SELECT COUNT(*) FROM orders WHERE delivered IS NOT NULL').fetchone()[0]
        # Stale: due more than 14 days ago with no delivered outcome
        stale_threshold = datetime.fromtimestamp(now.timestamp() - 14 * 86400, timezone.utc)
        stale_iso = iso(stale_threshold)
        stale = db.execute('SELECT COUNT(*) FROM orders WHERE delivered IS NULL AND due < ?', (stale_iso,)).fetchone()[0]

    staged_report = None
    report_file = ROOT / 'staged/promotion-report.json'
    if report_file.exists():
        try:
            staged_report = json.loads(report_file.read_text())
        except Exception:
            pass

    return dict(
        as_of=now_iso,
        live_model_version=MODEL['version'],
        total_orders=total,
        resolved_orders=resolved,
        pending_orders=total - resolved,
        stale_orders=stale,
        staged_candidate=staged_report is not None,
        staged_report=staged_report,
        gated_replacement_policy='Operator-confirmed shadow promotion; no automatic hot-swapping.'
    )


@app.get('/research-study.html')
def research_study(): return FileResponse(ROOT/'research/report.html')
