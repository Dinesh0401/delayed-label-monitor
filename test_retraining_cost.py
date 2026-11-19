import json
import unittest
from pathlib import Path
from retraining_cost_analysis import analyze_cost

class RetrainingCostTests(unittest.TestCase):
    def setUp(self):
        self.summary_path = Path('research/results/summary.json')

    def test_analysis_runs_and_has_structure(self):
        res = analyze_cost(str(self.summary_path))
        self.assertIn('cross_policy_summary', res)
        self.assertIn('detailed_records', res)
        summary = res['cross_policy_summary']
        self.assertIn('shadow_group', summary)
        self.assertIn('periodic', summary)
        self.assertIn('frozen', summary)

    def test_shadow_group_outperforms_periodic_efficiency(self):
        res = analyze_cost(str(self.summary_path))
        summary = res['cross_policy_summary']
        # Shadow group has much lower wasted refit rate than periodic
        self.assertLess(summary['shadow_group']['overall_wasted_refit_rate'],
                        summary['periodic']['overall_wasted_refit_rate'])
        # Shadow group has much higher gain per refit than periodic
        self.assertGreater(summary['shadow_group']['gain_per_refit'],
                           summary['periodic']['gain_per_refit'])
        # Periodic refits wastefully in stable regimes
        self.assertGreater(summary['periodic']['stable_regime_wasted_refits'], 0)
        self.assertEqual(summary['shadow_group']['stable_regime_wasted_refits'], 0.0)

if __name__ == '__main__':
    unittest.main()
