import unittest
import numpy as np
from matured_monitor import multiscale_harmful, find_matured_training_cohort
from research.research_suite import stream, counts, COUNT_CACHE

class MaturedMonitorTests(unittest.TestCase):
    def test_multiscale_harmful_detection(self):
        # When short horizon is blind due to pending labels, mature horizon detects harm
        ref_short = (12, 0, 100)
        cur_short = (10, 80, 100)  # 80 pending labels, only 10 observed errors -> no alarm

        ref_mature = (12, 0, 100)
        cur_mature = (85, 0, 100)  # fully matured cohort -> clear alarm

        res = multiscale_harmful(ref_short, cur_short, ref_mature, cur_mature)
        self.assertTrue(res['alarm'])
        self.assertEqual(res['scale'], 'mature')
        self.assertFalse(res['short_alarm'])
        self.assertTrue(res['mature_alarm'])

    def test_multiscale_no_false_alarm_on_stable(self):
        # Stable cohorts at both horizons
        ref_s = (12, 0, 100)
        cur_s = (12, 0, 100)
        ref_m = (12, 0, 100)
        cur_m = (12, 0, 100)
        res = multiscale_harmful(ref_s, cur_s, ref_m, cur_m)
        self.assertFalse(res['alarm'])

    def test_find_matured_training_cohort_under_long_delays(self):
        # Under 30-day delays, an old cohort has matured and can be found
        data = stream(7000, 'long_tail', 8)[1]  # minority group
        day, error, arrival, n = data
        now = 95
        candidate, start, end = find_matured_training_cohort(day, arrival, error, now, min_days=14, max_lookback=50)
        self.assertIsNotNone(candidate)
        self.assertLess(end, now)
        # Verify that all labels in [start, end) indeed arrived by `now`
        mask = (day >= start) & (day < end)
        self.assertTrue(np.all(arrival[mask] <= now))

    def test_run_matured_long_tail_detection(self):
        from research.long_tail_study import run_matured
        data = stream(7000, 'long_tail', 8)
        COUNT_CACHE.clear()
        res = run_matured(data, 'long_tail', days=120)
        # In long_tail, multiscale detector successfully triggers
        self.assertEqual(res['detected'], 1)
        self.assertEqual(res['eligible'], 1)
        self.assertEqual(res['false_alarm'], 0)
        # At 120 days, promotion is 0 because 44-day quarantine exceeds day 120
        self.assertEqual(res['promotions'], 0)

    def test_run_matured_adaptation_success_at_160_days(self):
        from research.long_tail_study import run_matured
        data = stream(7000, 'long_tail', 8, days=160)
        COUNT_CACHE.clear()
        res = run_matured(data, 'long_tail', days=160)
        self.assertEqual(res['detected'], 1)
        # At 160 days, quarantine completes and candidate model is promoted
        self.assertEqual(res['promotions'], 1)
        self.assertLess(res['minority_error'], 0.60)

if __name__ == '__main__':
    unittest.main()
