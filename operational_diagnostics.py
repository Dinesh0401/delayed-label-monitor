"""Daily operational diagnostics evaluation across geographic subgroups.

Computes fixed-cohort reference and current evaluation intervals using immutable
prediction creation timestamps rather than arrival-time selection.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
from cautious_monitor import harmful

SECONDS_PER_DAY = 86400


def parse_timestamp_epoch(iso_string: str) -> float:
    """Parse an ISO 8601 UTC timestamp into a Unix epoch timestamp in seconds.

    Args:
        iso_string: Time string in ISO 8601 format (e.g. '2026-01-01T00:00:00Z').

    Returns:
        Float timestamp representing seconds since Unix epoch.
    """
    cleaned = iso_string.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned).timestamp()


def diagnose(
    rows: list[dict[str, Any]],
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Perform once-daily snapshot diagnostic across all monitored states and model versions.

    Evaluates whether recent 28-day cohorts exhibit statistically significant
    accuracy degradation compared to the fixed 30-day baseline reference period.
    Incorporates multiscale matured evaluation when timeline allows (day >= 88).

    Args:
        rows: List of order dictionaries from the database.
        now: Optional datetime override representing snapshot evaluation time.

    Returns:
        Dictionary containing snapshot timestamp, state diagnostic reports,
        and statistical policy assumptions.
    """
    eval_time = now or datetime.now(timezone.utc)
    cutoff_epoch = int(eval_time.timestamp() // SECONDS_PER_DAY) * SECONDS_PER_DAY
    cutoff_iso = datetime.fromtimestamp(cutoff_epoch, timezone.utc).isoformat()

    # Group valid orders by (state, model_version)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if parse_timestamp_epoch(row["created"]) < cutoff_epoch:
            key = (row["state"], row["model_version"])
            groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []

    for (state, version), events in sorted(groups.items()):
        start_epoch = int(min(parse_timestamp_epoch(r["created"]) for r in events) // SECONDS_PER_DAY) * SECONDS_PER_DAY
        active_day = int((cutoff_epoch - start_epoch) // SECONDS_PER_DAY)

        base_info = {
            "state": state,
            "model_version": version,
            "day": active_day,
            "as_of": cutoff_iso,
        }

        # Early warm-up period: collecting baseline reference
        if active_day < 67:
            output.append({
                **base_info,
                "status": "collecting_reference",
                "days_until_first_look": 67 - active_day,
            })
            continue

        # Complete monitoring horizon
        if active_day >= 150:
            output.append({
                **base_info,
                "status": "horizon_complete",
            })
            continue

        def compute_cohort_stats(day_start_offset: int, day_end_offset: int) -> tuple[int, int, int]:
            """Compute errors, pending labels, and total orders for a relative cohort window."""
            window_start = start_epoch + day_start_offset * SECONDS_PER_DAY
            window_end = start_epoch + day_end_offset * SECONDS_PER_DAY

            cohort_orders = [
                r for r in events
                if window_start <= parse_timestamp_epoch(r["created"]) < window_end
            ]
            reported_orders = [
                r for r in cohort_orders
                if r.get("reported") and parse_timestamp_epoch(r["reported"]) <= cutoff_epoch
            ]

            wrong_predictions = sum(
                bool(r["predicted"]) != (parse_timestamp_epoch(r["delivered"]) >= parse_timestamp_epoch(r["due"]))
                for r in reported_orders
            )
            total_count = len(cohort_orders)
            pending_count = total_count - len(reported_orders)

            return (wrong_predictions, pending_count, total_count)

        # Baseline reference: first 30 days of the deployment
        ref_cohort = compute_cohort_stats(0, 30)

        # Current evaluation: 28-day window ending 9 days before snapshot
        cur_short_cohort = compute_cohort_stats(active_day - 37, active_day - 9)

        if not ref_cohort[2] or not cur_short_cohort[2]:
            output.append({
                **base_info,
                "status": "insufficient_data",
            })
            continue

        # Multiscale evaluation: when active_day >= 88, evaluate 30-day mature buffer cohort
        is_multiscale = active_day >= 88
        cur_mature_cohort = compute_cohort_stats(active_day - 58, active_day - 30) if is_multiscale else None

        if is_multiscale and cur_mature_cohort and cur_mature_cohort[2] > 0:
            from matured_monitor import multiscale_harmful
            ms_result = multiscale_harmful(
                ref_cohort, cur_short_cohort,
                ref_cohort, cur_mature_cohort,
                groups=27, looks=120, margin=0.05
            )
            alarm = ms_result["alarm"]
            scale_triggered = ms_result["scale"]
            ref_upper = min(ms_result["short_ref_upper"], ms_result["mature_ref_upper"])
            cur_lower = max(ms_result["short_cur_lower"], ms_result["mature_cur_lower"])
        else:
            std_result = harmful(ref_cohort, cur_short_cohort, groups=27, looks=120, margin=0.05)
            alarm = std_result["alarm"]
            scale_triggered = "short" if alarm else "none"
            ref_upper = std_result["reference_upper"]
            cur_lower = std_result["current_lower"]

        output.append({
            **base_info,
            "alarm": alarm,
            "status": "harm_evidence" if alarm else "no_confirmed_harm",
            "scale_triggered": scale_triggered,
            "multiscale_active": is_multiscale,
            "reference_upper": ref_upper,
            "current_lower": cur_lower,
            "reference_n": ref_cohort[2],
            "current_n": cur_short_cohort[2],
            "pending": cur_short_cohort[1],
            "margin": 0.05,
        })

    return {
        "as_of": cutoff_iso,
        "groups": output,
        "policy": "Daily multiscale fixed cohorts, 5 percentage-point harm margin; no automatic promotion",
        "assumptions": (
            "Independent bounded losses, fixed model, fixed 27 states; "
            "one 120-look budget per model version. Reusing versions or repeated restarts invalidates the budget."
        ),
    }
