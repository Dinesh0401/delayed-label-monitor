# Demo and hosting

## Record a 60-second portfolio demo

- **0–10 seconds:** Open the hosted dashboard. Say “A delivery prediction is available immediately, but its confirmation can arrive days later.”
- **10–30 seconds:** Add a fictitious order with a past deadline. Point to the saved prediction and its pending outcome. Record one delivery time and show the verification result.
- **30–48 seconds:** Run `python delivery_workflow_demo.py`. Compare an early accuracy estimate with the final result and point out the missing-outcome range. Label the replay as synthetic.
- **48–60 seconds:** Open the historical delivery study. Explain that it measures reporting delays on Olist orders. End on the GitHub repository and hosted demo links.

Use fictitious order IDs and timestamps for recordings. For a portfolio post, share the hosted URL together with the recording and the GitHub repository.

## Build for live hosting

Use Node 22.13 or newer for the SQLite-based Worker parity tests. From the repository root:

```sh
node hosted/test-worker.mjs
python build_hosted.py dist/server
```

Deploy `dist/server/index.js` as an ES module Worker. Bind a D1 database as `DB` and apply `hosted/drizzle/0000_delivery_orders.sql` to a new database before serving traffic. For an existing deployment, preserve its database and apply any outstanding migrations through the hosting provider.

For a shared portfolio demo, use an isolated database with fictitious records and limit write/export access to authorized users. The current hosted dashboard is linked from the README. After deploying a new build, check the homepage, `/api/model`, and `/api/diagnostics`, then verify one fictitious order from prediction through outcome on the hosted URL.
