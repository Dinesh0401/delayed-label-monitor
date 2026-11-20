"""Unit tests for real-world retraining and subgroup protection validation."""

import json
import unittest
from pathlib import Path

from real_world_validation import load_weekly_cohorts, simulate_real_world_policies


class RealWorldValidationTests(unittest.TestCase):
    """Test suite covering real-world Olist delivery cohort evaluation."""

    def setUp(self) -> None:
        """Set up test cohort paths."""
        self.csv_path = Path("delivery-results/weekly.csv")
        self.report_path = Path("research/REAL_WORLD_VALIDATION.md")
        self.json_path = Path("research/results/real-world-validation.json")

    def test_load_weekly_cohorts(self) -> None:
        """Verify that weekly cohorts load correctly and parse numeric fields."""
        cohorts = load_weekly_cohorts(self.csv_path)
        self.assertEqual(len(cohorts), 65)
        for c in cohorts:
            self.assertIn(c["region"], ("North", "SP", "Other"))
            self.assertGreaterEqual(c["n"], 10)
            self.assertGreaterEqual(c["pending"], 0)
            self.assertLessEqual(c["pending"], c["n"])

    def test_sandwich_property_on_all_real_cohorts(self) -> None:
        """Verify that exact [lower, upper] interval sandwiches ground-truth error on all cohorts."""
        cohorts = load_weekly_cohorts(self.csv_path)
        for c in cohorts:
            self.assertLessEqual(
                c["lower"] - 1e-12,
                c["truth"],
                f"Lower bound violated for {c['region']} {c['week']}: {c['lower']} > {c['truth']}",
            )
            self.assertGreaterEqual(
                c["upper"] + 1e-12,
                c["truth"],
                f"Upper bound violated for {c['region']} {c['week']}: {c['upper']} < {c['truth']}",
            )

    def test_shadow_group_gate_prevents_minority_harms(self) -> None:
        """Verify that cautious shadow gate produces 0 minority degradation events."""
        cohorts = load_weekly_cohorts(self.csv_path)
        results = simulate_real_world_policies(cohorts, rng_seed=42)
        comp = results["policy_comparison"]
        # Cautious shadow group gate forbids minority harms
        self.assertEqual(comp["shadow_group"]["minority_harm_incidents"], 0)
        # Unconstrained periodic allows minority regressions
        self.assertGreaterEqual(comp["periodic"]["minority_harm_incidents"], 1)

    def test_report_and_json_generation(self) -> None:
        """Verify that markdown report and JSON summary exist and have expected structure."""
        if self.report_path.exists():
            text = self.report_path.read_text(encoding="utf-8")
            self.assertIn("Real-World Retraining and Subgroup Protection Validation", text)
            self.assertIn("Zero Minority Harms Under Shadow Gate", text)
            self.assertIn("Exact Sandwich Sandwiching", text)
        if self.json_path.exists():
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            self.assertIn("bias_audit", data)
            self.assertIn("policy_comparison", data)


if __name__ == "__main__":
    unittest.main()
