"""Deterministic single-cohort replay. Python 3, standard library only."""


def cohort(correct_count, early_correct_fraction):
    events = []
    for correct, count, fraction in [
        (True, correct_count, early_correct_fraction),
        (False, 100 - correct_count, 0.5),
    ]:
        for i in range(count):
            arrival_day = 1 if i < round(count * fraction) else 5
            events.append((arrival_day, correct))
    return events


def accuracy(events, day):
    observed = [correct for arrival, correct in events if arrival <= day]
    return sum(observed) / len(observed) if observed else None


if __name__ == "__main__":
    print("scenario,observed_day1,pending_day1,final_day5,day1_alert")
    for name, correct, fraction in [
        ("reference", 80, 0.5),
        ("reporting_change", 80, 0.2),
        ("quality_loss", 60, 0.5),
        ("both", 60, 0.2),
    ]:
        events = cohort(correct, fraction)
        early = accuracy(events, 1)
        pending = sum(day > 1 for day, _ in events)
        print(f"{name},{early:.4f},{pending},{accuracy(events, 5):.4f},{early < 0.7}")
