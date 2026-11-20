"""Send a time-ordered historical demonstration to a running local API."""
import csv
import json
import sys
from datetime import date, timedelta
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"


def send(path, data):
    req = Request(BASE + path, json.dumps(data).encode(), {"Content-Type":"application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def build_events(rows):
    events = []
    for group in ("IBM", "MSFT"):
        values = [(date.fromisoformat(r["Date"]), float(r[group])) for r in rows if r[group]]
        for (prior, old), (today, price), (future, actual) in zip(values, values[1:], values[2:]):
            prediction = price >= old
            outcome = actual >= price
            # Reporting friction is simulated separately from the observed price series.
            delay = 21 if group == "IBM" and prediction == outcome else 0
            arrival = future + timedelta(days=delay)
            key = f"{group}-{today}"
            events.append((today, "/predictions", dict(id=key, segment=group, predicted=prediction, issued=f"{today}T00:00:00Z")))
            events.append((arrival, "/outcomes", dict(id=key, outcome=outcome, arrived=f"{arrival}T00:00:00Z")))
    return sorted(events, key=lambda e:e[0])


if __name__ == "__main__":
    with open(sys.argv[1]) as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    events = build_events(rows)
    for timestamp, endpoint, payload in events:
        send(endpoint, payload)
    print(f"Replayed {len(events)} events")
    for when in ("2021-05-10", "2021-06-30"):
        with urlopen(BASE + f"/metrics?since=2021-03-01T00:00:00Z&at={when}T00:00:00Z") as r:
            print(when, json.load(r))
