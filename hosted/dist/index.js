/**
 * Cloudflare Worker implementation for Delayed Label Monitor.
 *
 * Provides API endpoints and server-rendered HTML dashboards with persistent
 * Cloudflare D1 SQLite storage, maintaining exact parameter and numerical parity
 * with the Python reference service.
 *
 * @module worker
 */

/**
 * Computes the late-delivery logistic probability from feature values and model parameters.
 *
 * @param {number[]} features - Array of extracted numerical feature values.
 * @param {object} model - Model configuration containing intercept, coefficients, and threshold.
 * @returns {number} Predicted probability in (0, 1).
 */
export function probability(features, model) {
  const logit = model.intercept + features.reduce((sum, val, idx) => sum + val * model.coefficients[idx], 0);
  return 1 / (1 + Math.exp(-logit));
}

/**
 * Throws a formatted HTTP error response object.
 *
 * @param {number} code - HTTP status code.
 * @param {string} detail - Error message description.
 * @throws {{code: number, detail: string}}
 */
function fail(code, detail) {
  throw { code, detail };
}

/**
 * Normalizes and validates an ISO 8601 timezone-aware timestamp string.
 *
 * @param {string} value - Timestamp input.
 * @returns {string} Standard ISO 8601 UTC string.
 * @throws {{code: number, detail: string}} If timestamp is missing or lacks timezone.
 */
function timestamp(value) {
  if (typeof value !== 'string' || !/(Z|[+-]\d\d:\d\d)$/.test(value) || !Number.isFinite(Date.parse(value))) {
    fail(422, 'Use a timezone-aware timestamp');
  }
  return new Date(value).toISOString();
}

/**
 * Validates and transforms raw order inputs into a standardized database record.
 *
 * @param {object} input - Raw order payload containing id, state, purchased, and due timestamps.
 * @param {object} model - Model configuration containing supported states and features.
 * @returns {object} Normalized order record with calculated probability and prediction.
 */
export function normalizeOrder(input, model) {
  if (typeof input.id !== 'string' || !/^[A-Za-z0-9_-]{1,80}$/.test(input.id)) {
    fail(422, 'Order ID must contain 1–80 letters, numbers, underscores or hyphens');
  }

  const purchased = timestamp(input.purchased);
  const due = timestamp(input.due);
  const purchaseDate = new Date(purchased);
  const promiseDays = (Date.parse(due) - purchaseDate.getTime()) / 86400000;

  if (!model.states.includes(input.state) || !(promiseDays > 0 && promiseDays <= 180)) {
    fail(422, 'Choose a supported state and a deadline within 180 days after purchase');
  }

  if (purchaseDate.getTime() > Date.now()) {
    fail(422, 'Purchase cannot be in the future');
  }

  // Feature vector construction
  const featureValues = {
    promise_days: promiseDays,
    dow: (purchaseDate.getUTCDay() + 6) % 7,
    month: purchaseDate.getUTCMonth() + 1,
    hour: purchaseDate.getUTCHours(),
    ['customer_state_' + input.state]: 1,
  };

  const featureVector = model.features.map(feat => featureValues[feat] || 0);
  const lateProb = probability(featureVector, model);

  return {
    id: input.id,
    state: input.state,
    purchased,
    due,
    probability: lateProb,
    predicted: Number(lateProb >= model.threshold),
    model_version: model.version,
    created: new Date().toISOString(),
  };
}

/**
 * Creates the Cloudflare Worker fetch event handler with injected model and HTML assets.
 *
 * @param {object} model - Active logistic model parameters.
 * @param {string} html - HTML source for the primary operations dashboard.
 * @param {string} study - HTML source for the historical delivery benchmark report.
 * @param {object} research - JSON results of the robust detection study.
 * @param {string} [study2='Research report unavailable'] - HTML source for the research findings report.
 * @returns {{fetch: (request: Request, env: object) => Promise<Response>}} Worker handler object.
 */
export function createHandler(model, html, study, research, study2 = 'Research report unavailable') {
  const jsonResponse = (data, status = 200) =>
    new Response(JSON.stringify(data), {
      status,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    });

  return {
    async fetch(request, env) {
      const url = new URL(request.url);
      const path = url.pathname;

      try {
        // Static content routes
        if (request.method === 'GET' && path === '/') {
          return new Response(html, { headers: { 'Content-Type': 'text/html;charset=utf-8' } });
        }
        if (request.method === 'GET' && (path === '/delivery-study.html' || path === '/delivery-study')) {
          return new Response(study, { headers: { 'Content-Type': 'text/html;charset=utf-8' } });
        }
        if (request.method === 'GET' && path === '/research-study.html') {
          return new Response(study2, { headers: { 'Content-Type': 'text/html;charset=utf-8' } });
        }
        if (request.method === 'GET' && path === '/research.json') {
          return jsonResponse(research);
        }

        // Model inspection route
        if (request.method === 'GET' && path === '/api/model') {
          const { coefficients, intercept, features, ...card } = model;
          return jsonResponse(card);
        }

        if (!path.startsWith('/api/')) {
          return jsonResponse({ detail: 'Not found' }, 404);
        }

        if (!env.DB) {
          fail(503, 'Persistent database is not configured');
        }

        // POST mutation routes
        if (request.method === 'POST') {
          const origin = request.headers.get('Origin');
          if (origin && origin !== url.origin) {
            fail(403, 'Cross-origin writes are not allowed');
          }
          if (!request.headers.get('Content-Type')?.startsWith('application/json')) {
            fail(415, 'Send JSON');
          }

          const rawText = await request.text();
          if (rawText.length > 65536) {
            fail(413, 'Request is too large');
          }

          let body;
          try {
            body = JSON.parse(rawText);
          } catch {
            fail(400, 'Invalid JSON');
          }

          // Batch orders ingestion
          if (path === '/api/batch-orders') {
            if (!Array.isArray(body)) {
              fail(422, 'Expected an array of orders');
            }
            let insertedCount = 0;
            for (const item of body) {
              try {
                const orderRecord = normalizeOrder(item, model);
                const result = await env.DB.prepare(
                  'INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)'
                ).bind(
                  orderRecord.id,
                  orderRecord.state,
                  orderRecord.purchased,
                  orderRecord.due,
                  orderRecord.probability,
                  orderRecord.predicted,
                  orderRecord.model_version,
                  orderRecord.created
                ).run();

                const actualChanges = result?.changes ?? result?.meta?.changes ?? 0;
                if (actualChanges > 0) {
                  insertedCount++;
                }
              } catch {
                // Skip invalid rows within batch
              }
            }
            return jsonResponse({ received: body.length, inserted: insertedCount });
          }

          // Batch outcomes ingestion
          if (path === '/api/batch-outcomes') {
            if (!Array.isArray(body)) {
              fail(422, 'Expected an array of outcomes');
            }
            let updatedCount = 0;
            const nowTime = Date.now();
            const nowIso = new Date().toISOString();

            for (const item of body) {
              try {
                if (typeof item.id !== 'string') continue;
                const delivered = timestamp(item.delivered);

                // Strictly reject deliveries in the future
                if (Date.parse(delivered) > nowTime) continue;

                const row = await env.DB.prepare(
                  'SELECT purchased, delivered FROM orders WHERE id = ?'
                ).bind(item.id).first();

                if (!row || row.delivered || Date.parse(delivered) < Date.parse(row.purchased)) {
                  continue;
                }

                await env.DB.prepare(
                  'UPDATE orders SET delivered = ?, reported = ? WHERE id = ? AND delivered IS NULL'
                ).bind(delivered, nowIso, item.id).run();

                updatedCount++;
              } catch {
                // Skip invalid rows within batch
              }
            }
            return jsonResponse({ received: body.length, updated: updatedCount });
          }

          if (!body || typeof body !== 'object') {
            fail(422, 'Expected an object');
          }

          // Single order creation
          if (path === '/api/orders') {
            const orderRecord = normalizeOrder(body, model);
            await env.DB.prepare(
              'INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)'
            ).bind(
              orderRecord.id,
              orderRecord.state,
              orderRecord.purchased,
              orderRecord.due,
              orderRecord.probability,
              orderRecord.predicted,
              orderRecord.model_version,
              orderRecord.created
            ).run();

            const existing = await env.DB.prepare('SELECT * FROM orders WHERE id = ?').bind(orderRecord.id).first();
            if (
              existing.state !== orderRecord.state ||
              existing.purchased !== orderRecord.purchased ||
              existing.due !== orderRecord.due
            ) {
              fail(409, 'Order ID already exists with different input');
            }
            return jsonResponse(existing);
          }

          // Single outcome recording
          if (path === '/api/outcomes') {
            if (typeof body.id !== 'string') {
              fail(422, 'Order ID is required');
            }
            const delivered = timestamp(body.delivered);
            const row = await env.DB.prepare('SELECT * FROM orders WHERE id = ?').bind(body.id).first();

            if (!row) {
              fail(404, 'Order not found');
            }
            if (delivered < row.purchased || Date.parse(delivered) > Date.now()) {
              fail(422, 'Delivery must be between purchase and now');
            }

            await env.DB.prepare(
              'UPDATE orders SET delivered = ?, reported = ? WHERE id = ? AND delivered IS NULL'
            ).bind(delivered, new Date().toISOString(), body.id).run();

            const saved = await env.DB.prepare('SELECT * FROM orders WHERE id = ?').bind(body.id).first();
            if (saved.delivered !== delivered) {
              fail(409, 'This order already has a different outcome');
            }
            return jsonResponse(saved);
          }
        }

        // GET query routes
        if (request.method === 'GET' && path === '/api/diagnostics') {
          const rows = (await env.DB.prepare('SELECT * FROM orders').all()).results;
          return jsonResponse(diagnose(rows));
        }

        if (request.method === 'GET' && path === '/api/orders') {
          const rows = (await env.DB.prepare('SELECT * FROM orders ORDER BY created DESC, id LIMIT 200').all()).results;
          return jsonResponse(rows);
        }

        if (request.method === 'GET' && path === '/api/export') {
          const rows = (await env.DB.prepare('SELECT * FROM orders ORDER BY created ASC').all()).results;
          return jsonResponse({ orders: rows, total: rows.length, scope: 'hosted_export' });
        }

        if (request.method === 'GET' && path === '/api/pipeline-status') {
          const stats = (await env.DB.prepare(
            'SELECT COUNT(*) as total, COUNT(delivered) as resolved FROM orders'
          ).all()).results[0];

          const staleThreshold = new Date(Date.now() - 14 * 86400000).toISOString();
          const staleStats = (await env.DB.prepare(
            'SELECT COUNT(*) as stale FROM orders WHERE delivered IS NULL AND due < ?'
          ).bind(staleThreshold).all()).results[0];

          return jsonResponse({
            as_of: new Date().toISOString(),
            live_model_version: model.version,
            total_orders: stats.total,
            resolved_orders: stats.resolved,
            pending_orders: stats.total - stats.resolved,
            stale_orders: staleStats.stale,
            hosted_runtime: 'Cloudflare Worker (D1)',
          });
        }

        if (request.method === 'GET' && path === '/api/metrics') {
          const { results } = await env.DB.prepare(`
            SELECT state, COUNT(*) as n, COUNT(delivered) as resolved,
                   SUM(CASE WHEN delivered IS NOT NULL AND predicted = (delivered >= due) THEN 1 ELSE 0 END) as correct
            FROM orders GROUP BY state ORDER BY state
          `).all();

          return jsonResponse(
            results.map(r => ({
              state: r.state,
              total: r.n,
              pending: r.n - r.resolved,
              observed: r.resolved ? r.correct / r.resolved : null,
              lower: r.correct / r.n,
              upper: (r.correct + r.n - r.resolved) / r.n,
            }))
          );
        }

        return jsonResponse({ detail: 'Not found' }, 404);
      } catch (err) {
        return jsonResponse({ detail: err.detail || 'Server error; please retry' }, err.code || 500);
      }
    },
  };
}

/**
 * Executes statistical drift diagnostics across cohorts using fixed-creation windows.
 *
 * @param {Array<object>} rows - Array of order objects from database.
 * @param {Date} [now=new Date()] - Evaluation epoch date.
 * @returns {object} Diagnostic summary across geographic state groups.
 */
export function diagnose(rows, now = new Date()) {
  const DAY_MS = 86400000;
  const cutoffTime = Math.floor(now.getTime() / DAY_MS) * DAY_MS;
  const cutoffIso = new Date(cutoffTime).toISOString();
  const groups = new Map();
  const output = [];

  for (const r of rows) {
    if (Date.parse(r.created) >= cutoffTime) continue;
    const key = r.state + '|' + r.model_version;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }

  for (const [key, events] of [...groups.entries()].sort()) {
    const firstEpoch = Math.floor(events.reduce((min, r) => Math.min(min, Date.parse(r.created)), Infinity) / DAY_MS) * DAY_MS;
    const activeDay = Math.floor((cutoffTime - firstEpoch) / DAY_MS);

    const baseRecord = {
      state: events[0].state,
      model_version: events[0].model_version,
      day: activeDay,
      as_of: cutoffIso,
    };

    if (activeDay < 67) {
      output.push({ ...baseRecord, status: 'collecting_reference', days_until_first_look: 67 - activeDay });
      continue;
    }
    if (activeDay >= 150) {
      output.push({ ...baseRecord, status: 'horizon_complete' });
      continue;
    }

    /**
     * Extracts errors, pending count, and total count for a given cohort window.
     *
     * @param {number} lo - Relative start day.
     * @param {number} hi - Relative end day.
     * @returns {[number, number, number]} [errors, pending, total]
     */
    function cohort(lo, hi) {
      const chosen = events.filter(r => Date.parse(r.created) >= firstEpoch + lo * DAY_MS && Date.parse(r.created) < firstEpoch + hi * DAY_MS);
      const known = chosen.filter(r => r.reported && Date.parse(r.reported) <= cutoffTime);
      const incorrect = known.filter(r => Boolean(r.predicted) !== (Date.parse(r.delivered) >= Date.parse(r.due))).length;
      return [incorrect, chosen.length - known.length, chosen.length];
    }

    const ref = cohort(0, 30);
    const curShort = cohort(activeDay - 37, activeDay - 9);

    if (!ref[2] || !curShort[2]) {
      output.push({ ...baseRecord, status: 'insufficient_data' });
      continue;
    }

    // Multiscale cohort evaluation: when activeDay >= 88, evaluate 30-day mature buffer
    const isMultiscale = activeDay >= 88;
    const curMature = isMultiscale ? cohort(activeDay - 58, activeDay - 30) : null;
    const scales = isMultiscale && curMature && curMature[2] > 0 ? 2 : 1;

    const tailAlpha = 0.05 / (2 * 27 * 120 * scales);
    const radius = n => Math.sqrt(Math.log(1 / tailAlpha) / (2 * n));

    const refUpper = Math.min(1, (ref[0] + ref[1]) / ref[2] + radius(ref[2]));
    const lowerShort = Math.max(0, curShort[0] / curShort[2] - radius(curShort[2]));
    const lowerMature = scales === 2 ? Math.max(0, curMature[0] / curMature[2] - radius(curMature[2])) : 0;

    const alarmShort = lowerShort > refUpper + 0.05;
    const alarmMature = scales === 2 && lowerMature > refUpper + 0.05;
    const alarm = alarmShort || alarmMature;

    const scaleTriggered = alarmShort && alarmMature ? 'both' : alarmShort ? 'short' : alarmMature ? 'mature' : 'none';

    output.push({
      ...baseRecord,
      alarm,
      status: alarm ? 'harm_evidence' : 'no_confirmed_harm',
      scale_triggered: scaleTriggered,
      multiscale_active: scales === 2,
      reference_upper: refUpper,
      current_lower: Math.max(lowerShort, lowerMature),
      current_upper: Math.min(1, (curShort[0] + curShort[1]) / curShort[2] + radius(curShort[2])),
      margin: 0.05,
      reference_n: ref[2],
      current_n: curShort[2],
      pending: curShort[1],
    });
  }

  return {
    as_of: cutoffIso,
    groups: output,
    policy: 'Daily multiscale fixed cohorts, 5 percentage-point harm margin; no automatic promotion',
    assumptions:
      'Independent bounded losses, fixed model, fixed 27 states; one 120-look budget per model version. Reusing versions or repeated restarts invalidates the budget.',
  };
}

const MODEL={
  "version": "olist-logistic-v1",
  "features": [
    "promise_days",
    "dow",
    "month",
    "hour",
    "customer_state_AC",
    "customer_state_AL",
    "customer_state_AM",
    "customer_state_AP",
    "customer_state_BA",
    "customer_state_CE",
    "customer_state_DF",
    "customer_state_ES",
    "customer_state_GO",
    "customer_state_MA",
    "customer_state_MG",
    "customer_state_MS",
    "customer_state_MT",
    "customer_state_PA",
    "customer_state_PB",
    "customer_state_PE",
    "customer_state_PI",
    "customer_state_PR",
    "customer_state_RJ",
    "customer_state_RN",
    "customer_state_RO",
    "customer_state_RR",
    "customer_state_RS",
    "customer_state_SC",
    "customer_state_SE",
    "customer_state_SP",
    "customer_state_TO"
  ],
  "coefficients": [
    -0.07250431927058545,
    -0.017228748265602646,
    0.01911295825200569,
    0.003268289448828683,
    1.0587245633939804,
    2.4463549565255573,
    0.13251670654152115,
    2.0180256166588535,
    1.1301415377167223,
    1.0838825371472074,
    -0.06231965224531349,
    0.6078268714153207,
    0.14384884926582467,
    1.606715321581723,
    -0.6122813109054956,
    -0.4687057089806343,
    0.6521390470979659,
    1.0249442996690106,
    1.6974373659253068,
    1.2238439754796606,
    1.322484354905075,
    -0.48661194992346185,
    0.4326707674261953,
    1.2443996723354218,
    0.4611566287535685,
    2.4183360084849252,
    -0.17258997473023988,
    0.5474265071758191,
    1.77250559027199,
    -0.5337269650296255,
    0.17621415696313172
  ],
  "intercept": -1.5165097301921209,
  "threshold": 0.060000000000000005,
  "train_count": 41197,
  "test_count": 32150,
  "average_precision": 0.102634048638878,
  "brier": 0.04112643276343646,
  "test_prevalence": 0.04419906687402799,
  "states": [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO"
  ],
  "training_cutoff": "2018-01-01",
  "test_period": "Apr\u2013Aug 2018"
};

const HTML="<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Delayed Label Monitor \u00b7 Operations</title><style>\n:root{--ink:#162c2a;--muted:#61716c;--line:#dbe0d9;--paper:#f5f5ef;--green:#20624e;--orange:#a65d2b}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}button,input,select{font:inherit}button,a,input,select{outline-offset:4px}button{cursor:pointer;border:1px solid var(--line);border-radius:5px;padding:10px 14px;background:white;color:var(--ink)}button.primary{background:var(--ink);color:white;border-color:var(--ink)}button:disabled{opacity:.5;cursor:wait}a{color:var(--green)}header{height:76px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 4vw;background:#fff}.brand{font-size:16px;font-weight:750;letter-spacing:-.5px}.mark{background:var(--ink);color:#fff;margin-right:12px;padding:7px 10px;border-radius:5px}.status{font-size:12px;color:var(--muted)}.status:before{content:'\u25cf';color:#3d8864;margin-right:8px}main{max-width:1400px;margin:auto;padding:40px 4vw}h1{font-size:36px;letter-spacing:-1.6px;line-height:1.15;margin:8px 0 14px;font-weight:600}h2{font-size:17px;margin:0;letter-spacing:-.4px}p{color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:1.6px;font-size:12px;font-weight:700;color:var(--muted)}.intro{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:28px}.intro p{max-width:630px;margin:0}.workflow{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-radius:8px;background:#eaf0e8;margin:26px 0}.step{padding:16px 20px;position:relative}.step+.step{border-left:1px solid var(--line)}.step b{display:block;font-weight:600}.step span{font-size:12px;color:var(--muted)}.step em{float:right;font-style:normal;font-size:12px;color:var(--green)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin:26px 0;gap:16px}.metric{padding:20px;background:white;border:1px solid var(--line);border-radius:8px}.metric strong{font-size:30px;display:block;letter-spacing:-1px;font-weight:550;margin:6px 0}.metric small{color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:22px;align-items:start}.panel{background:white;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:22px}.panelhead{padding:20px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px}.panelbody{padding:22px}.sub{font-size:12px;color:var(--muted);margin:4px 0 0}.tabs{display:flex;gap:5px;margin:0 0 16px}.tabs button{padding:6px 12px;background:transparent}.tabs button[aria-pressed=true]{background:var(--ink);color:#fff;border-color:var(--ink)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}th{text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);padding:12px 20px;background:#fafbf8}td{padding:16px 20px;border-top:1px solid #edf0eb}td strong{display:block;font-size:13px}td small{display:block;color:var(--muted)}.pill{font-size:12px;padding:4px 8px;border-radius:4px;background:#edf2eb;color:#3b6554}.pill.wait{color:#8b632f;background:#f8f0e1}.pill.late{color:#9c4535;background:#f9eee9}.empty{padding:55px 30px;text-align:center;white-space:normal}.empty b{font-size:17px;font-weight:550}.empty p{max-width:340px;margin:10px auto}.field{display:block;margin:0 0 14px;font-size:12px;font-weight:550}.field input,.field select{display:block;width:100%;margin-top:7px;padding:10px;border:1px solid #cbd4cc;border-radius:5px;background:#fff;color:var(--ink)}.field small{display:block;color:var(--muted);font-weight:400;margin-top:4px}.full{width:100%}.note{font-size:12px;color:var(--muted);padding:14px;background:#f3f5ef;border-radius:5px}.result{padding:18px;margin-top:18px;border-left:3px solid var(--green);background:#edf5ed}.result strong{font-size:22px;display:block}.region{padding:16px 0;border-bottom:1px solid var(--line)}.region:last-child{border:0}.regionhead{display:flex;justify-content:space-between;margin-bottom:8px}.track{position:relative;height:8px;background:#edf0e9;border-radius:4px;overflow:hidden}.range{height:100%;position:absolute;background:#9eaf90;transition:left .4s,width .4s}.legend{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:5px}.modelrow{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--line);font-size:12px}.modelrow:last-child{border:0}.message{min-height:24px;margin:12px 0;color:var(--green);font-size:13px}.error{color:#a2352a}.flash{animation:fade .4s ease-out}@keyframes fade{from{opacity:.4;transform:translateY(5px)}to{opacity:1;transform:none}}footer{border-top:1px solid var(--line);padding:22px 4vw;color:var(--muted);font-size:12px}dialog{border:1px solid var(--line);border-radius:10px;width:min(460px,90vw);padding:28px;color:var(--ink)}dialog::backdrop{background:#102e2866}.actions{display:flex;gap:10px;justify-content:end}#connection.offline:before{color:#b35936}.research{background:#edf0e7}.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}@media(max-width:1000px){.grid{grid-template-columns:1fr}.aside{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:650px){main{padding:24px 5vw}.intro{display:block}h1{font-size:30px}.intro button{margin-top:20px}.metrics{grid-template-columns:1fr 1fr;gap:10px}.metric{padding:14px}.metric strong{font-size:24px}.workflow{grid-template-columns:1fr}.step+.step{border-left:0;border-top:1px solid var(--line)}.aside{display:block}header{padding:0 5vw}.status{max-width:120px}.panelhead{padding:18px}td,th{padding:12px}.brand{font-size:13px}.mark{margin-right:6px}}@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}\nselect{appearance:none;padding-right:32px!important;background-image:linear-gradient(45deg,transparent 50%,#19342d 50%),linear-gradient(135deg,#19342d 50%,transparent 50%);background-position:calc(100% - 18px) 50%,calc(100% - 13px) 50%;background-size:5px 5px;background-repeat:no-repeat}.help{background:#fff;border:1px solid var(--line);border-radius:8px;margin:20px 0}summary{cursor:pointer;font-weight:600;padding:16px 20px}.helpbody{padding:0 20px 16px;max-width:850px}.helpbody p{font-size:14px}.field{font-size:14px}table{font-size:14px}.metrics strong{font-size:28px}.workflow{background:#edf3f6}body{background:#f3f6f8}header{border-top:3px solid #20624e}details.panel{margin-bottom:22px}details.panel summary{list-style:revert}.grid{grid-template-columns:minmax(0,1fr) 350px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}@media(prefers-reduced-motion:no-preference){button{transition:background .15s,border-color .15s}button:hover{border-color:var(--green)}} </style></head><body><header><div class=\"brand\"><span class=\"mark\">d.</span> Delayed Label Monitor</div><span class=\"status\" id=\"connection\">Connecting to server</span></header><main><div class=\"intro\"><div><div class=\"eyebrow\">Delivery operations</div><h1>Track orders. Verify predictions.</h1><p>Add an order to estimate its late-delivery risk. Record the actual delivery to see whether the prediction was right.</p></div><button id=\"export\">Export records </button></div><details class=\"help\"><summary>How does this work in practice?</summary><div class=\"helpbody\"><p><strong>Who is this for?</strong> A delivery analyst who wants to check whether a prediction model stays reliable as real outcomes arrive.</p><p><strong>What happens now?</strong> You enter an order. The server runs the trained model and saves the prediction. You later record the delivery time, and the dashboard immediately recalculates the results.</p><p><strong>What is not connected?</strong> There is no courier feed. Orders and delivery outcomes must currently be entered by you. Model health uses daily snapshots; it does not retrain or replace the live model automatically.</p><p><strong>Example:</strong> An order is flagged as late risk. If its recorded delivery happens after the deadline, that prediction is correct. One correct order is one successful check\u2014not evidence of 100% future accuracy.</p></div></details><div class=\"workflow\"><div class=\"step\"><b>Add an order</b><span>Enter the destination and promised delivery time.</span></div><div class=\"step\"><b>View the prediction</b><span>The model scores the order and saves the result.</span></div><div class=\"step\"><b>Record the delivery</b><span>Compare the prediction with what actually happened.</span></div></div><div id=\"notice\" class=\"message\" role=\"status\" aria-live=\"polite\"></div><section class=\"metrics\" aria-label=\"Current monitoring metrics\"><div class=\"metric\"><div class=\"eyebrow\">Saved orders</div><strong id=\"total\">\u2014</strong><small>Your saved order records</small></div><div class=\"metric\"><div class=\"eyebrow\">Awaiting outcome</div><strong id=\"pending\">\u2014</strong><small>Record delivery to verify these orders</small></div><div class=\"metric\"><div class=\"eyebrow\">Correct predictions</div><strong id=\"accuracy\">\u2014</strong><small id=\"resolved\">No verified outcomes yet</small></div><div class=\"metric\"><div class=\"eyebrow\">Accuracy including pending orders</div><strong id=\"bounds\">\u2014</strong><small>Range until every outcome is known</small></div></section><div class=\"grid\"><div><section class=\"panel\"><div class=\"panelhead\"><div><h2>Orders</h2><p class=\"sub\">Latest 200 orders \u00b7 all records contribute to metrics</p></div><button id=\"refresh\">Refresh</button></div><div class=\"panelbody\" style=\"padding-bottom:0\"><div class=\"tabs\"><button data-filter=\"all\" aria-pressed=\"true\">All orders</button><button data-filter=\"pending\" aria-pressed=\"false\">Awaiting outcome</button><button data-filter=\"resolved\" aria-pressed=\"false\">Verified</button></div></div><div class=\"scroll\"><table><thead><tr><th>Order / State</th><th>Model prediction</th><th>Deadline (UTC)</th><th>Outcome</th><th>Action</th></tr></thead><tbody id=\"orders\"></tbody></table></div></section><section class=\"panel\"><div class=\"panelhead\"><div><h2>Reliability by state</h2><p class=\"sub\">Missing outcomes widen the range. Small samples remain inconclusive.</p></div></div><div class=\"panelbody\" id=\"regions\"></div></section><section class=\"panel\"><div class=\"panelhead\"><div><h2>Model health</h2><p class=\"sub\">Checks whether prediction quality is getting worse</p></div></div><div class=\"panelbody\"><div id=\"diagnostics\">Loading evidence status\u2026</div><p class=\"note\">The monitor needs a 30-day reference and a later 28-day cohort, with a 9-day reporting buffer. Earliest decision: day 67. Pending labels remain in the denominator. No confirmed harm does not establish safety.</p><a href=\"/research-study.html\">Explore the detector & retraining experiments </a></div></section></div><aside class=\"aside\"><section class=\"panel\"><div class=\"panelhead\"><div><h2>Add an order</h2><p class=\"sub\">Get a prediction and save it to your order list</p></div></div><div class=\"panelbody\"><form id=\"create\"><label class=\"field\">Order ID<input name=\"id\" required maxlength=\"80\" pattern=\"[A-Za-z0-9_-]+\" placeholder=\"e.g. DELIVERY-1042\" autocomplete=\"off\"></label><label class=\"field\">Destination state (Brazil)<select name=\"state\" id=\"states\" required><option value=\"\">Select state</option></select></label><label class=\"field\">Purchase time (UTC)<input type=\"datetime-local\" name=\"purchased\" required></label><label class=\"field\">Delivery deadline (UTC)<input type=\"datetime-local\" name=\"due\" required><small>For a promised calendar day, enter midnight at the start of the following day.</small></label><button type=\"submit\" class=\"primary full\" id=\"score\">Get prediction & save</button></form><div id=\"prediction\" hidden class=\"result\" role=\"status\"></div><p class=\"sub\" style=\"margin-top:16px\">No customer names or addresses needed. Use an internal order reference.</p></div></section><details class=\"panel\"><summary class=\"panelhead\">Model details & limitations</summary><div class=\"panelbody\"><div class=\"eyebrow\">Logistic regression \u00b7 Python trained</div><div id=\"model\"></div><p class=\"note\">Historical Olist data, 2017\u20132018. A research baseline for Brazilian deliveries; performance on your current orders is unknown. Risk scores are not guaranteed probabilities.</p><p class=\"sub\">Threshold selected on a separate calibration period. Prediction features use UTC calendar values. No automatic retraining or fairness guarantee.</p></div></details></aside></div></main><footer>Delayed Label Monitor \u00b7 <a href=\"/research-study.html\">Evaluation & research</a> \u00b7 <a href=\"/delivery-study.html\">Historical delivery study</a><p>Accuracy ranges describe possible outcomes for saved orders; they are not statistical confidence intervals.</p></footer><dialog id=\"dialog\"><h2>Record a delivery</h2><p id=\"outcomeTitle\"></p><form id=\"outcome\"><input type=\"hidden\" name=\"id\"><label class=\"field\">Delivered at (UTC)<input type=\"datetime-local\" name=\"delivered\" required></label><p class=\"note\">The server compares the delivery time with the saved deadline. Existing verified outcomes cannot be overwritten.</p><div id=\"outcomeError\" role=\"alert\" class=\"message error\"></div><div class=\"actions\"><button type=\"button\" id=\"cancel\">Cancel</button><button class=\"primary\" type=\"submit\">Verify outcome</button></div></form></dialog><script>\nlet records=[],groups=[],filter='all',model;const $=id=>document.getElementById(id);const pct=x=>x==null?'\u2014':(100*x).toFixed(1)+'%';const esc=x=>String(x).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));const date=x=>x.slice(0,16).replace('T',' ');function message(text,error=false){$('notice').textContent=text;$('notice').className='message'+(error?' error':'')}async function api(path,body){const r=await fetch('/api/'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});let data;try{data=await r.json()}catch{throw Error('The server did not return data. Please refresh.')}if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Check the form fields and try again.');return data}\nfunction render(){const n=groups.reduce((s,g)=>s+g.total,0),pending=groups.reduce((s,g)=>s+g.pending,0),correct=groups.reduce((s,g)=>s+g.lower*g.total,0);$('total').textContent=n;$('pending').textContent=pending;$('accuracy').textContent=n>pending?pct(correct/(n-pending)):'\u2014';$('resolved').textContent=(n-pending)+' verified outcomes'+(n-pending<30?' \u00b7 too few to judge overall performance':'');$('bounds').textContent=n?Math.round(100*correct/n)+'\u2013'+Math.round(100*(correct+pending)/n)+'%':'\u2014';const list=records.filter(r=>filter==='all'||(filter==='pending'?!r.delivered:!!r.delivered));$('orders').innerHTML=list.length?list.map(r=>`<tr class=\"flash\"><td><strong>${esc(r.id)}</strong><small>${esc(r.state)} \u00b7 ${date(r.purchased)}</small></td><td><span class=\"pill ${r.predicted?'late':''}\">${r.predicted?'Late risk':'On time'}</span><small>${pct(r.probability)} late score</small></td><td>${date(r.due)}</td><td>${r.delivered?`<span class=\"pill ${r.delivered>=r.due?'late':''}\">${r.delivered>=r.due?'Late':'On time'}</span><small>${r.predicted===Number(r.delivered>=r.due)?'Prediction correct':'Prediction incorrect'}</small>`:'<span class=\"pill wait\">Awaiting outcome</span>'}</td><td>${r.delivered?'<span class=\"sub\">Verified</span>':`<button data-record=\"${esc(r.id)}\">Record delivery</button>`}</td></tr>`).join(''):`<tr><td colspan=\"5\"><div class=\"empty\"><b>${filter==='all'?'Add your first order.':'No orders in this view.'}</b><p>Use the Add an order form. Your saved prediction will appear here, with a Record delivery action for completing the workflow.</p></div></td></tr>`;$('regions').innerHTML=groups.length?groups.map(g=>`<div class=\"region\"><div class=\"regionhead\"><b>${esc(g.state)} <span class=\"sub\">\u00b7 ${g.total} orders \u00b7 ${g.pending} pending</span></b><span>${pct(g.observed)} observed</span></div><div class=\"track\" aria-label=\"Accuracy including pending orders ${pct(g.lower)} to ${pct(g.upper)}\"><div class=\"range\" style=\"left:${100*g.lower}%;width:${Math.max(.5,100*(g.upper-g.lower))}%\"></div></div><div class=\"legend\"><span>Possible final: ${pct(g.lower)}\u2013${pct(g.upper)}</span><span>${g.total-g.pending<30?'Limited evidence (<30 outcomes)':'Descriptive metrics only'}</span></div></div>`).join(''):'<p class=\"sub\">State-level reliability will appear after you save an order.</p>'}\nasync function refresh(){try{[records,groups]=await Promise.all([api('orders'),api('metrics')]);render();const diagnosis=await api('diagnostics');$('diagnostics').innerHTML=diagnosis.groups.length?diagnosis.groups.map(g=>`<div class=\"region\"><b>${esc(g.state)} \u00b7 ${esc(({collecting_reference:'Building comparison history',horizon_complete:'Review monitoring period',insufficient_data:'More orders needed',harm_evidence:'Performance review needed',no_confirmed_harm:'No confirmed deterioration'})[g.status]||g.status)}</b><p class=\"sub\">${g.status==='collecting_reference'?`First scheduled check in ${g.days_until_first_look} days. Enough orders are also required in both comparison periods.`:g.current_n?`${g.current_n} current predictions \u00b7 ${g.pending} pending \u00b7 error lower bound ${pct(g.current_lower)} vs reference upper bound ${pct(g.reference_upper)}.`:'Insufficient data or monitoring horizon ended.'}</p></div>`).join(''):'<p class=\"sub\">Collecting evidence. Predictions created today enter tomorrow\u2019s daily snapshot.</p>';$('connection').textContent='Server connected';$('connection').className='status'}catch(e){$('connection').textContent='Connection unavailable';$('connection').className='status offline';message(e.message,true);throw e}}\n$('create').onsubmit=async e=>{e.preventDefault();$('score').disabled=true;try{let b=Object.fromEntries(new FormData(e.target));b.purchased+=':00Z';b.due+=':00Z';const r=await api('orders',b);$('prediction').hidden=false;$('prediction').innerHTML=`<span>Saved \u00b7 ${esc(r.id)}</span><strong>${pct(r.probability)} late score</strong><span>${r.predicted?'Flagged as late risk':'Predicted on time'} \u00b7 threshold ${pct(model.threshold)}</span>`;await refresh();message('Prediction saved. Record the delivery when the outcome is available.')}catch(e){message(e.message,true)}finally{$('score').disabled=false}};\n$('orders').onclick=e=>{const b=e.target.closest('[data-record]');if(!b)return;const r=records.find(x=>x.id===b.dataset.record);$('outcome').elements.id.value=r.id;$('outcome').elements.delivered.value=new Date().toISOString().slice(0,16);$('outcomeTitle').textContent=r.id+' \u00b7 Deadline '+date(r.due)+' UTC';$('outcomeError').textContent='';$('dialog').showModal()};$('cancel').onclick=()=>$('dialog').close();$('outcome').onsubmit=async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;try{const b=Object.fromEntries(new FormData(e.target));b.delivered+=':00Z';await api('outcomes',b);$('dialog').close();await refresh();message('Delivery verified. Accuracy and uncertainty bounds have been updated.')}catch(e){$('outcomeError').textContent=e.message}finally{button.disabled=false}};document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render()});$('refresh').onclick=()=>refresh().catch(()=>{});$('export').onclick=()=>{const a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify({exported_at:new Date().toISOString(),scope:'latest 200 orders; metrics include all orders',orders:records,metrics:groups},null,2)],{type:'application/json'}));a.href=u;a.download='delivery-monitor-export.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};(async()=>{try{model=await api('model');$('states').innerHTML+= model.states.map(s=>`<option>${s}</option>`).join('');$('model').innerHTML=[['Version',model.version],['Training orders',model.train_count.toLocaleString()],['Held-out orders',model.test_count.toLocaleString()],['Average precision',model.average_precision.toFixed(3)],['Test late prevalence',pct(model.test_prevalence)],['Brier score',model.brier.toFixed(4)],['Decision threshold',pct(model.threshold)]].map(([k,v])=>`<div class=\"modelrow\"><span>${k}</span><b>${esc(v)}</b></div>`).join('');await refresh()}catch(e){message(e.message,true)}})();\n</script></body></html>\n";

const STUDY="<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Delivery monitoring field study</title><style>body{font:17px/1.6 system-ui;color:#14352b;background:#f5f8f5;max-width:960px;margin:5vh auto;padding:18px}h1{line-height:1.1;font-size:clamp(36px,6vw,60px)}section{overflow:auto}table{background:white;border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #cad7cc}a{color:#12684b}</style><a href=\"/\">Dashboard</a><h1>How reliable are early delivery labels?</h1><p>Held-out evaluation on 32,150 fulfilled Brazilian orders purchased April\u2013August 2018. A gradient boosted model predicts late arrival at purchase. Outcomes become visible at delivery.</p><h2>Estimating model error after 14 days</h2><p>Mean absolute difference from eventual error, in percentage points (pp). Lower is better. Each eligible week gets equal weight.</p><section><table class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th>Region</th>\n      <th>Weeks</th>\n      <th>Orders</th>\n      <th>Arrived only (pp)</th>\n      <th>Delay corrected (pp)</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>North</td>\n      <td>19</td>\n      <td>494</td>\n      <td>10.75</td>\n      <td>9.96</td>\n    </tr>\n    <tr>\n      <td>Other</td>\n      <td>23</td>\n      <td>16894</td>\n      <td>3.47</td>\n      <td>2.13</td>\n    </tr>\n    <tr>\n      <td>SP</td>\n      <td>23</td>\n      <td>14751</td>\n      <td>1.30</td>\n      <td>2.45</td>\n    </tr>\n  </tbody>\n</table></section><p>Correction helped in the North and Other groups. In S\u00e3o Paulo it nearly doubled the estimation error. Northern estimates cover 494 orders in eligible weeks.</p><h2>Forecast quality</h2><p>Average precision was 0.088 versus 0.044 for a constant forecast. Forecast quality remains modest. Training and threshold selection used earlier orders.</p><p>This offline replay evaluates a fixed model. The operational value of retraining remains an open question.</p><a href=\"https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\">Source: Brazilian E-Commerce Public Dataset by Olist</a></html>";

const RESEARCH={
  "calibration_seeds": [
    2000,
    2099
  ],
  "test_seeds": [
    3000,
    3299
  ],
  "alpha": 0.05,
  "calibration_rank": 96,
  "calibration_regimes": [
    "stable",
    "delay"
  ],
  "per_regime_thresholds": {
    "stable": {
      "arrived": 0.14285714285714285,
      "censored": 0.294968512121058
    },
    "delay": {
      "arrived": 0.12698412698412698,
      "censored": 0.6418604651162791
    }
  },
  "rows": [
    {
      "scenario": "stable",
      "protocol": "stable_only",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 16,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "stable",
      "protocol": "stable_only",
      "method": "censored",
      "threshold": 0.294968512121058,
      "runs": 300,
      "false_alarm_runs": 13,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "stable",
      "protocol": "reporting_robust",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 16,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "stable",
      "protocol": "reporting_robust",
      "method": "censored",
      "threshold": 0.6418604651162791,
      "runs": 300,
      "false_alarm_runs": 0,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "delay",
      "protocol": "stable_only",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 10,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "delay",
      "protocol": "stable_only",
      "method": "censored",
      "threshold": 0.294968512121058,
      "runs": 300,
      "false_alarm_runs": 300,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "delay",
      "protocol": "reporting_robust",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 10,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "delay",
      "protocol": "reporting_robust",
      "method": "censored",
      "threshold": 0.6418604651162791,
      "runs": 300,
      "false_alarm_runs": 18,
      "eligible_runs": null,
      "detected_runs": null,
      "missed_runs": null,
      "mean_detection_delay": null,
      "mean_capped_delay": null
    },
    {
      "scenario": "harm",
      "protocol": "stable_only",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 6,
      "eligible_runs": 294,
      "detected_runs": 294,
      "missed_runs": 0,
      "mean_detection_delay": 6.166666666666667,
      "mean_capped_delay": 6.166666666666667
    },
    {
      "scenario": "harm",
      "protocol": "stable_only",
      "method": "censored",
      "threshold": 0.294968512121058,
      "runs": 300,
      "false_alarm_runs": 4,
      "eligible_runs": 296,
      "detected_runs": 296,
      "missed_runs": 0,
      "mean_detection_delay": 5.125,
      "mean_capped_delay": 5.125
    },
    {
      "scenario": "harm",
      "protocol": "reporting_robust",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 6,
      "eligible_runs": 294,
      "detected_runs": 294,
      "missed_runs": 0,
      "mean_detection_delay": 6.166666666666667,
      "mean_capped_delay": 6.166666666666667
    },
    {
      "scenario": "harm",
      "protocol": "reporting_robust",
      "method": "censored",
      "threshold": 0.6418604651162791,
      "runs": 300,
      "false_alarm_runs": 0,
      "eligible_runs": 300,
      "detected_runs": 1,
      "missed_runs": 299,
      "mean_detection_delay": 12,
      "mean_capped_delay": 39.906666666666666
    },
    {
      "scenario": "both",
      "protocol": "stable_only",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 6,
      "eligible_runs": 294,
      "detected_runs": 255,
      "missed_runs": 39,
      "mean_detection_delay": 14.631372549019607,
      "mean_capped_delay": 17.996598639455783
    },
    {
      "scenario": "both",
      "protocol": "stable_only",
      "method": "censored",
      "threshold": 0.294968512121058,
      "runs": 300,
      "false_alarm_runs": 4,
      "eligible_runs": 296,
      "detected_runs": 296,
      "missed_runs": 0,
      "mean_detection_delay": 3.3175675675675675,
      "mean_capped_delay": 3.3175675675675675
    },
    {
      "scenario": "both",
      "protocol": "reporting_robust",
      "method": "arrived",
      "threshold": 0.14285714285714285,
      "runs": 300,
      "false_alarm_runs": 6,
      "eligible_runs": 294,
      "detected_runs": 255,
      "missed_runs": 39,
      "mean_detection_delay": 14.631372549019607,
      "mean_capped_delay": 17.996598639455783
    },
    {
      "scenario": "both",
      "protocol": "reporting_robust",
      "method": "censored",
      "threshold": 0.6418604651162791,
      "runs": 300,
      "false_alarm_runs": 0,
      "eligible_runs": 300,
      "detected_runs": 292,
      "missed_runs": 8,
      "mean_detection_delay": 13.14041095890411,
      "mean_capped_delay": 13.856666666666667
    }
  ]
}
;

const STUDY2="<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Delayed Label Monitor \u00b7 Research evidence</title><style>\nbody{font:16px/1.65 system-ui,sans-serif;color:#19342d;background:#f5f5ef;margin:0}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:38px;line-height:1.15;letter-spacing:-1px;max-width:700px}h2{font-size:22px;margin-top:32px}p{max-width:850px}a{color:#20624e}section{padding:24px;background:white;border:1px solid #dbe0d9;border-radius:8px;margin:20px 0}.controls{display:flex;gap:20px;flex-wrap:wrap}label{display:grid;gap:6px;font-size:13px}select,button{font:inherit;padding:10px;background:white;border:1px solid #bdcbbf;border-radius:5px;color:#19342d}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #dbe0d9}th{font-size:12px;text-transform:uppercase}tr.chosen{background:#edf5e9}.scroll{overflow:auto}.note{color:#5a6f63;font-size:13px}code{background:#eaf0e8;padding:2px 5px}.bar{height:5px;background:#53775a;max-width:100%;margin-top:4px}li{margin:8px 0}button{cursor:pointer}:focus-visible{outline:3px solid #759e6b;outline-offset:3px}select{appearance:none;padding-right:32px!important;background-image:linear-gradient(45deg,transparent 50%,#19342d 50%),linear-gradient(135deg,#19342d 50%,transparent 50%);background-position:calc(100% - 18px) 50%,calc(100% - 13px) 50%;background-size:5px 5px;background-repeat:no-repeat}</style><main><a href=\"/\"> Delivery workbench</a><p class=\"note\">RESEARCH EVIDENCE \u00b7 19 NOVEMBER 2026 \u00b7 SYNTHETIC ADAPTATION STUDY</p><h1>Detect harm. Then earn the model change.</h1><p>A fixed-cohort detector preserves missing outcomes in its denominator. A prospective shadow model must demonstrate improvement on later predictions before promotion. These experiments show both the benefits and the remaining failures.</p><section><h2 style=\"margin-top:0\">A result with boundaries</h2><p>With eight minority predictions per day and severe drift plus slower reporting, the prospective policy reduced average minority error from <b>62.6% to 40.1%</b>, with <b>one refit per run</b>. Stable and reporting-only scenarios caused no alarms or refits in these 100 runs per scenario. This is a synthetic result, not a production accuracy or fairness claim.</p><p class=\"note\">At two minority predictions per day, only 38 of 100 severe-drift runs promoted a candidate. With 30-day delayed labels, the same eight-per-day setting detected none of 100 harmful streams. Missing feedback can make the system abstain indefinitely.</p></section><h2>Compare every policy</h2><div class=\"controls\"><label>Minority predictions per day<select id=\"size\"><option>2</option><option selected>8</option><option>24</option></select></label><label>Scenario<select id=\"scenario\"></select></label><button id=\"download\">Download all aggregate results </button></div><section class=\"scroll\"><table><thead><tr><th>Policy</th><th>Minority error</th><th>Refits</th><th>Promotions</th><th>False alarms</th><th>Detections / eligible</th><th>Capped delay</th></tr></thead><tbody id=\"rows\"></tbody></table></section><p class=\"note\">Every cell summarizes 100 paired seeds. Refits/promotions are mean counts per run. Detection excludes streams with a pre-change alarm; capped delay assigns 80 days to misses. Frozen and periodic policies have no meaningful detector rates and are shown as \u2014. Periodic training is a scheduled baseline, not a detector.</p><h2>What was tested</h2><ul><li>Initial seeds 5000\u20135099: six policies, nine scenarios, three group sizes. The first gate exhausted its budget too early and failed to adapt.</li><li>Follow-up seeds 6000\u20136099: the same six policies plus a prospective shadow policy. All parameters were fixed before this follow-up; the failed initial results are retained.</li><li>Majority: 80 predictions/day. Minority: 2, 8 or 24. Monitoring lasts 120 days, with 30 reference days and drift beginning on day 40.</li><li>Severe drift changes error from 0.12 to 0.88; mild drift changes it to 0.35. Normal reporting is after 1 or 9 days. Stress cases include 30-day delays and permanently missing labels.</li><li>Candidate classifiers are identity/inversion by group. They are not the logistic model serving real delivery predictions.</li></ul><h2>The decision rules</h2><p>For n predictions, e observed errors and u pending outcomes, the complete empirical error lies between e/n and (e+u)/n. We widen this range using a Hoeffding radius, allocating a finite error budget across groups and scheduled looks. A harm alarm requires the current lower bound to exceed the reference upper bound by five percentage points.</p><p>The candidate is frozen before its prospective validation cohort. Every pending paired outcome receives the worst possible improvement, \u22121. Promotion requires the lower improvement bound to exceed two percentage points. Other groups' synthetic models remain unchanged.</p><p class=\"note\">The bound assumes independent bounded losses and fixed cohorts. It is not valid without qualification under temporal dependence, adaptive cohort selection, or repeated horizon resets. The operational dashboard uses 27 states and a later day-67 start; its power is not measured by this two-group experiment.</p><h2>Reproduce and review</h2><p><code>python research_suite.py --phase initial</code><br><code>python research_suite.py --phase shadow</code><br><code>python -m unittest discover -v</code><br><code>node hosted/test-worker.mjs</code></p><p><a href=\"https://github.com/Dinesh0401/delayed-label-monitor\">Source, per-run results, protocol and tests </a></p><h2>Related work and remaining evidence</h2><p>Sequential inference, hierarchical verification and staged adaptation have substantial prior work: <a href=\"https://arxiv.org/abs/1810.08240\">Howard et al.</a>, <a href=\"https://ijcai.org/proceedings/2018/421\">Request-and-Reverify</a>, and <a href=\"https://proceedings.mlr.press/v139/tahmasbi21a.html\">DriftSurf</a>. We do not claim to have invented these ideas or reproduced their algorithms.</p><p>Outstanding: strong external detector baselines, real-stream adaptation evaluation, dependence and unseen-delay stress tests, independent novelty/patent review, public participant access and a consented user study. No real-user feedback has been collected. Automatic deployment of retrained delivery models remains disabled.</p><script>const data={\"protocol\": \"research/PROTOCOL.md\", \"seeds\": [6000, 6099], \"rows\": [{\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"stable\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_slow\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"reporting_fast\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11725, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3716666666666668, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.2540833333333333, \"paired_gain_bootstrap_95\": [0.2491645833333333, 0.25849999999999995]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5409166666666667, \"refits\": 1.79, \"promotions\": 0.28, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.08483333333333334, \"paired_gain_bootstrap_95\": [0.059997916666666665, 0.11199999999999999]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5409166666666667, \"refits\": 1.79, \"promotions\": 0.28, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.08483333333333334, \"paired_gain_bootstrap_95\": [0.059997916666666665, 0.11199999999999999]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.33699999999999997, \"refits\": 1.18, \"promotions\": 0.95, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.28875, \"paired_gain_bootstrap_95\": [0.27407499999999996, 0.3015020833333333]}, {\"minority_per_day\": 2, \"scenario\": \"harm\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5648333333333334, \"refits\": 1.62, \"promotions\": 0.38, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.06091666666666666, \"paired_gain_bootstrap_95\": [0.04516666666666667, 0.07641666666666666]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3716666666666668, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.2540833333333333, \"paired_gain_bootstrap_95\": [0.2491645833333333, 0.25849999999999995]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5409166666666667, \"refits\": 1.79, \"promotions\": 0.28, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.08483333333333334, \"paired_gain_bootstrap_95\": [0.059997916666666665, 0.11199999999999999]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5409166666666667, \"refits\": 1.79, \"promotions\": 0.28, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.08483333333333334, \"paired_gain_bootstrap_95\": [0.059997916666666665, 0.11199999999999999]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.33699999999999997, \"refits\": 1.18, \"promotions\": 0.95, \"unchanged_refits\": 0.23, \"minority_gain_vs_frozen\": 0.28875, \"paired_gain_bootstrap_95\": [0.27407499999999996, 0.3015020833333333]}, {\"minority_per_day\": 2, \"scenario\": \"both\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.93, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5648333333333334, \"refits\": 1.62, \"promotions\": 0.38, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.06091666666666666, \"paired_gain_bootstrap_95\": [0.04516666666666667, 0.07641666666666666]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"mild\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.26958333333333334, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.55925, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.47666666666666657, \"refits\": 4.0, \"promotions\": 0.32, \"unchanged_refits\": 3.68, \"minority_gain_vs_frozen\": 0.08258333333333333, \"paired_gain_bootstrap_95\": [0.05891458333333333, 0.10567291666666663]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 43.04, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.4870833333333333, \"refits\": 1.83, \"promotions\": 0.32, \"unchanged_refits\": 0.33, \"minority_gain_vs_frozen\": 0.07216666666666667, \"paired_gain_bootstrap_95\": [0.051583333333333335, 0.09483541666666666]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.55925, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 43.04, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.4870833333333333, \"refits\": 1.83, \"promotions\": 0.32, \"unchanged_refits\": 0.33, \"minority_gain_vs_frozen\": 0.07216666666666667, \"paired_gain_bootstrap_95\": [0.051583333333333335, 0.09483541666666666]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 43.04, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3419166666666667, \"refits\": 1.26, \"promotions\": 0.94, \"unchanged_refits\": 0.32, \"minority_gain_vs_frozen\": 0.2173333333333333, \"paired_gain_bootstrap_95\": [0.20475000000000002, 0.22850208333333336]}, {\"minority_per_day\": 2, \"scenario\": \"gradual\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 43.04, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5275, \"refits\": 1.64, \"promotions\": 0.35, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.03175, \"paired_gain_bootstrap_95\": [0.0235, 0.04066874999999998]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.02, \"promotions\": 0.0, \"unchanged_refits\": 0.02, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 59, \"mean_capped_delay\": 61.64, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"long_tail\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 33.88, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6245, \"refits\": 0.08, \"promotions\": 0.01, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.00125, \"paired_gain_bootstrap_95\": [0.0, 0.00375]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 80, \"mean_capped_delay\": 51.79, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6245, \"refits\": 0.04, \"promotions\": 0.01, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.00125, \"paired_gain_bootstrap_95\": [0.0, 0.00375]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 80, \"mean_capped_delay\": 51.79, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6234166666666667, \"refits\": 0.03, \"promotions\": 0.03, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0023333333333333327, \"paired_gain_bootstrap_95\": [0.0, 0.005587499999999977]}, {\"minority_per_day\": 2, \"scenario\": \"missing\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 80, \"mean_capped_delay\": 51.79, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6257500000000001, \"refits\": 0.06, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"stable\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_slow\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"reporting_fast\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.12045833333333336, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3724791666666667, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.253375, \"paired_gain_bootstrap_95\": [0.25102083333333336, 0.25568749999999996]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"harm\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.40114583333333337, \"refits\": 1.0, \"promotions\": 1.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.2247083333333334, \"paired_gain_bootstrap_95\": [0.22206145833333332, 0.22714583333333332]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3724791666666667, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.253375, \"paired_gain_bootstrap_95\": [0.25102083333333336, 0.25568749999999996]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"both\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 21.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.40114583333333337, \"refits\": 1.0, \"promotions\": 1.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.2247083333333334, \"paired_gain_bootstrap_95\": [0.22206145833333332, 0.22714583333333332]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 8, \"mean_capped_delay\": 77.74, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.15, \"promotions\": 0.0, \"unchanged_refits\": 0.15, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 8, \"mean_capped_delay\": 77.74, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.15, \"promotions\": 0.0, \"unchanged_refits\": 0.15, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 8, \"mean_capped_delay\": 77.74, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.15, \"promotions\": 0.0, \"unchanged_refits\": 0.15, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"mild\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 8, \"mean_capped_delay\": 77.74, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.2739479166666666, \"refits\": 0.08, \"promotions\": 0.0, \"unchanged_refits\": 0.08, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5590729166666665, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.4864270833333333, \"refits\": 4.0, \"promotions\": 0.29, \"unchanged_refits\": 3.71, \"minority_gain_vs_frozen\": 0.07264583333333334, \"paired_gain_bootstrap_95\": [0.05089270833333333, 0.09566874999999998]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 32.13, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5590729166666665, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5590729166666665, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 32.13, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5590729166666665, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 32.13, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5590729166666665, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"gradual\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 32.13, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.40148958333333334, \"refits\": 1.0, \"promotions\": 1.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.15758333333333333, \"paired_gain_bootstrap_95\": [0.15508333333333335, 0.16]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 33.74, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"long_tail\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 23.34, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.95, \"promotions\": 0.0, \"unchanged_refits\": 0.95, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 25.17, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.28, \"promotions\": 0.0, \"unchanged_refits\": 0.28, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 25.17, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.28, \"promotions\": 0.0, \"unchanged_refits\": 0.28, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 8, \"scenario\": \"missing\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 25.17, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6258541666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"stable\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_slow\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 100, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"reporting_fast\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 0, \"detections\": 0, \"mean_capped_delay\": null, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.11957291666666667, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3734895833333334, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.25333333333333335, \"paired_gain_bootstrap_95\": [0.2520137152777778, 0.2546597222222222]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 29.25, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.48134374999999996, \"refits\": 3.71, \"promotions\": 0.47, \"unchanged_refits\": 3.24, \"minority_gain_vs_frozen\": 0.14547916666666666, \"paired_gain_bootstrap_95\": [0.11715902777777777, 0.17641458333333332]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"harm\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.38061458333333337, \"refits\": 1.03, \"promotions\": 1.0, \"unchanged_refits\": 0.03, \"minority_gain_vs_frozen\": 0.24620833333333336, \"paired_gain_bootstrap_95\": [0.23956822916666667, 0.25145156249999995]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3734895833333334, \"refits\": 4.0, \"promotions\": 1.0, \"unchanged_refits\": 3.0, \"minority_gain_vs_frozen\": 0.25333333333333335, \"paired_gain_bootstrap_95\": [0.2520137152777778, 0.2546597222222222]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 29.25, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.48134374999999996, \"refits\": 3.71, \"promotions\": 0.47, \"unchanged_refits\": 3.24, \"minority_gain_vs_frozen\": 0.14547916666666666, \"paired_gain_bootstrap_95\": [0.11715902777777777, 0.17641458333333332]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"both\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 17.26, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.38061458333333337, \"refits\": 1.03, \"promotions\": 1.0, \"unchanged_refits\": 0.03, \"minority_gain_vs_frozen\": 0.24620833333333336, \"paired_gain_bootstrap_95\": [0.23956822916666667, 0.25145156249999995]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 4.0, \"promotions\": 0.0, \"unchanged_refits\": 4.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 36.92, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 36.92, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 36.92, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"mild\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 36.92, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.27300694444444445, \"refits\": 1.88, \"promotions\": 0.0, \"unchanged_refits\": 1.88, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5604305555555555, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.4974722222222222, \"refits\": 4.0, \"promotions\": 0.25, \"unchanged_refits\": 3.75, \"minority_gain_vs_frozen\": 0.06295833333333334, \"paired_gain_bootstrap_95\": [0.04252048611111111, 0.08389722222222222]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 27.55, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5604305555555555, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 39.91, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.41672916666666665, \"refits\": 3.69, \"promotions\": 0.59, \"unchanged_refits\": 3.1, \"minority_gain_vs_frozen\": 0.14370138888888884, \"paired_gain_bootstrap_95\": [0.12069756944444444, 0.1676552083333333]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 27.55, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5604305555555555, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 27.55, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.5604305555555555, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"gradual\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 27.55, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.3796597222222223, \"refits\": 1.03, \"promotions\": 1.0, \"unchanged_refits\": 0.03, \"minority_gain_vs_frozen\": 0.18077083333333335, \"paired_gain_bootstrap_95\": [0.17406215277777778, 0.18563888888888885]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 31.71, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 17, \"mean_capped_delay\": 75.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 17, \"mean_capped_delay\": 75.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"long_tail\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 17, \"mean_capped_delay\": 75.63, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"frozen\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 0, \"mean_capped_delay\": 80.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"periodic\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 0.0, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"arrived_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 18.71, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"cohort_pooled\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 54, \"mean_capped_delay\": 62.38, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"cohort_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 19.6, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"ungated_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 19.6, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 2.0, \"promotions\": 0.0, \"unchanged_refits\": 2.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}, {\"minority_per_day\": 24, \"scenario\": \"missing\", \"policy\": \"shadow_group\", \"runs\": 100, \"false_alarms\": 0, \"eligible\": 100, \"detections\": 100, \"mean_capped_delay\": 19.6, \"majority_error\": 0.11954791666666667, \"minority_error\": 0.6268229166666666, \"refits\": 0.0, \"promotions\": 0.0, \"unchanged_refits\": 0.0, \"minority_gain_vs_frozen\": 0.0, \"paired_gain_bootstrap_95\": [0.0, 0.0]}]};const names={frozen:'Frozen model',periodic:'Periodic refit',arrived_group:'Arrived-only group',cohort_pooled:'Fixed-cohort pooled',cohort_group:'Fixed-cohort group',ungated_group:'Ungated group',shadow_group:'Prospective shadow'};const scenarios=[...new Set(data.rows.map(r=>r.scenario))];const s=document.getElementById('scenario'),n=document.getElementById('size');for(const x of scenarios){const o=document.createElement('option');o.value=x;o.textContent=x.replaceAll('_',' ');s.append(o)}s.value='both';function render(){document.getElementById('rows').innerHTML=data.rows.filter(r=>r.minority_per_day===Number(n.value)&&r.scenario===s.value).map(r=>{const detector=!['frozen','periodic'].includes(r.policy);return `<tr class=\"${r.policy==='shadow_group'?'chosen':''}\"><td>${names[r.policy]}</td><td>${(r.minority_error*100).toFixed(1)}%<div class=\"bar\" style=\"width:${r.minority_error*100}%\"></div></td><td>${r.refits.toFixed(2)}</td><td>${r.promotions.toFixed(2)}</td><td>${detector?r.false_alarms+'/100':'\u2014'}</td><td>${detector&&r.eligible?r.detections+'/'+r.eligible:'\u2014'}</td><td>${detector&&r.mean_capped_delay!==null?r.mean_capped_delay.toFixed(1)+' days':'\u2014'}</td></tr>`}).join('')}n.onchange=s.onchange=render;render();document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='shadow-study-summary.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></main></html>";

export default createHandler(MODEL,HTML,STUDY,RESEARCH,STUDY2);
