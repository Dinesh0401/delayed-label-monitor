"""Unit tests for matched-horizon long-delay adaptation benchmarking."""

import json
import unittest
from pathlib import Path

import numpy as np

from research.matched_horizon import DAYS
from research.research_suite import COUNT_CACHE, stream
from research.long_tail_study import run_matured


class MatchedHorizonTests(unittest.TestCase):
    """Test suite covering matched frozen vs adaptive evaluation on identical streams."""

    def setUp(self) -> None:
        """Set up artifact file paths."""
        self.report_path = Path("research/MATCHED_HORIZON.md")
        self.json_path = Path("research/results/matched-horizon.json")

    def test_artifacts_exist_and_document_findings(self) -> None:
        """Verify that markdown report and JSON summary exist and have expected structure."""
        self.assertTrue(self.report_path.exists())
        self.assertTrue(self.json_path.exists())

        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["days"], 160)
        self.assertEqual(data["seed_range"], [10000, 10099])

        rows = data["rows"]
        self.assertEqual(len(rows), 6)  # 2 scenarios x 3 minority sizes

        # Check stable controls have zero false alarms and zero refits
        stable_rows = [r for r in rows if r["scenario"] == "stable"]
        for r in stable_rows:
            self.assertEqual(r["false_alarm_runs"], 0)
            self.assertEqual(r["refits"], 0)
            self.assertEqual(r["promotions"], 0)

        # Check long_tail adaptation produces substantial gain at n >= 8
        long_tail_rows = {r["minority_per_day"]: r for r in rows if r["scenario"] == "long_tail"}
        self.assertGreater(long_tail_rows[8]["gain_pp"], 15.0)
        self.assertEqual(long_tail_rows[8]["promotions"], 100)
        self.assertGreater(long_tail_rows[24]["gain_pp"], 17.0)
        self.assertEqual(long_tail_rows[24]["promotions"], 100)

    def test_single_seed_paired_evaluation(self) -> None:
        """Verify that paired scoring on a single stream reproduces deterministically."""
        seed = 10000
        scenario = "long_tail"
        size = 8

        data = stream(seed, scenario, size, days=DAYS)
        days, errors, _, _ = data[1]
        baseline = float(errors[(days >= 0) & (days < DAYS)].mean())

        COUNT_CACHE.clear()
        adaptive = run_matured(data, scenario, days=DAYS)
        COUNT_CACHE.clear()

        # In long_tail, adaptive minority error should be lower than frozen baseline
        self.assertEqual(adaptive["detected"], 1)
        self.assertEqual(adaptive["promotions"], 1)
        self.assertLess(adaptive["minority_error"], baseline)


if __name__ == "__main__":
    unittest.main()
