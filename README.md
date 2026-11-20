# Delayed Label Monitor

An MLOps monitoring application for delivery predictions whose outcomes arrive days later. It records model versions and predictions, matches them to confirmed outcomes, and shows how pending deliveries could change measured accuracy.

**[Live dashboard](https://delayed-label-monitor-demo.bubblybobcat3.chatgpt.site/)** · **[Historical delivery study](https://delayed-label-monitor-demo.bubblybobcat3.chatgpt.site/delivery-study.html)** · **[Research experiments](https://delayed-label-monitor-demo.bubblybobcat3.chatgpt.site/research-study.html)**

## See it in 60 seconds

1. Add a fictitious order and note its saved late-delivery risk.
2. Show the order under **Awaiting outcome**, then record its delivery.
3. Show how the dashboard updates verified accuracy. Run the replay below to demonstrate why early accuracy can differ from the final result.

The dashboard uses operator-entered outcomes. The historical study analyzes 32,150 Olist orders. The replay and retraining experiments simulate delayed outcomes and model changes.

## Run and verify

```sh
python -m pip install -r requirements.txt -r requirements-research.txt
python -m uvicorn serving:app --port 8000
```

From the repository root, run these in another terminal:

```sh
python delivery_workflow_demo.py
python -m unittest discover -v
node hosted/test-worker.mjs
```

## Repository map

| Location | Contents |
| --- | --- |
| `serving.py`, `web/` | FastAPI operations API and dashboard |
| `hosted/` | Cloudflare Worker, D1 schema, and runtime parity checks |
| `tests/` | Python tests |
| `delivery-results/`, `research/` | Historical results, protocols, experiment artifacts, and archived examples |

[Hosting and demo recording guide](docs/DEMO_AND_HOSTING.md) · [Research progress](research/PROGRESS.md)
