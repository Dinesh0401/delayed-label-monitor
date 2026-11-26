"""Regression check for retaining a candidate across inconclusive looks."""

import unittest

from research.long_tail_study import run_matured
from research.research_suite import COUNT_CACHE, stream


class ScheduledValidationTest(unittest.TestCase):
    def test_inconclusive_early_check_retains_candidate(self):
        data = stream(7000, "long_tail", 8, days=160)
        trace = []
        COUNT_CACHE.clear()
        self.addCleanup(COUNT_CACHE.clear)
        run_matured(data, "long_tail", days=160, trace=trace,
                    validation_schedule=(9, 16, 23, 30))
        promotions = [e for e in trace if e["action"] == "promote" and e["group"] == 1]
        self.assertEqual(len(promotions), 1)
        event = promotions[0]
        self.assertIn(event["day"] - event["opened"] - 14, (16, 23, 30))
