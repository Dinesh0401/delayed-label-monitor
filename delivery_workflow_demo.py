"""Replay synthetic delivery events through FastAPI in an isolated database."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from serving import app


def replay(client: TestClient) -> list[dict]:
    """Capture reliability after prediction, early reports and late reports."""
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    outcomes: list[dict[str, str]] = []
    for i in range(60):
        purchased = start + timedelta(days=i // 10)
        due = purchased + timedelta(days=7)
        order_id = f"DEMO-{i:03d}"
        state = "AM" if i % 5 == 0 else "SP"
        response = client.post(
            "/api/orders",
            json={
                "id": order_id,
                "state": state,
                "purchased": purchased.isoformat(),
                "due": due.isoformat(),
            },
        )
        response.raise_for_status()
        # Use a fixed scenario independent of the model's predictions.
        late = i % 3 == 0
        delivered = due + timedelta(days=3 if late else -1)
        outcomes.append({"id": order_id, "delivered": delivered.isoformat()})

    def snapshot(phase: str) -> dict:
        response = client.get("/api/metrics")
        response.raise_for_status()
        return {"phase": phase, "states": response.json()}

    snapshots = [snapshot("predictions saved")]
    ordered = sorted(outcomes, key=lambda event: event["delivered"])
    for index, outcome in enumerate(ordered, start=1):
        response = client.post("/api/outcomes", json=outcome)
        response.raise_for_status()
        if index == 40:
            snapshots.append(snapshot("40 early reports received"))
    snapshots.append(snapshot("all 60 reports received"))

    final = {row["state"]: row for row in snapshots[-1]["states"]}
    for stage in snapshots:
        for row in stage["states"]:
            accuracy = final[row["state"]]["observed"]
            if not row["lower"] <= accuracy <= row["upper"]:
                raise RuntimeError(f"Final accuracy outside bounds for {row['state']}")
    return snapshots


def main() -> None:
    """Run against a disposable SQLite database and print the evidence."""
    with TemporaryDirectory(prefix="delivery-replay-") as directory:
        database_path = Path(directory) / "demo.sqlite"
        with patch.dict(os.environ, {"DELIVERY_DB": str(database_path)}):
            with TestClient(app) as client:
                result = replay(client)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
