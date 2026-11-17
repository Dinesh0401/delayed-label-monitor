"""Paired weekly robustness audit. Positive gain favors delay correction."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

directory = Path(__file__).resolve().parent / "delivery-results"


def audit(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["observed"] and row["posterior"]:
            truth = float(row["truth"])
            gain = 100 * (abs(float(row["observed"]) - truth) - abs(float(row["posterior"]) - truth))
            grouped[row["region"]].append((int(row["n"]), gain))
    for region, pairs in sorted(grouped.items()):
        gains = [g for _, g in pairs]
        n = len(gains)
        omissions = [(sum(gains)-g)/(n-1) for g in gains] if n>1 else gains
        yield [region,n,sum(g>0 for g in gains),mean(gains),median(gains),
               sum(size*g for size,g in pairs)/sum(size for size,_ in pairs),
               min(omissions),max(omissions)]


if __name__ == "__main__":
    with (directory/"weekly.csv").open() as source:
        results = list(audit(csv.DictReader(source)))
    with (directory/"stability.csv").open("w",newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["region","paired_weeks","weeks_improved","mean_gain_pp","median_gain_pp",
                         "order_weighted_gain_pp","leave_one_week_out_min_pp","leave_one_week_out_max_pp"])
        for row in results:
            writer.writerow(row[:3]+[round(x,5) for x in row[3:]])
