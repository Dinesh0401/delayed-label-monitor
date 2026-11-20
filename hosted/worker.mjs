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
