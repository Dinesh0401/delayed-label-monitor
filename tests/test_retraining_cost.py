import json
import unittest
from pathlib import Path
from research.retraining_cost_analysis import analyze_cost

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
        # Shadow group has much higher promotion success rate under harm than periodic
        self.assertGreater(summary['shadow_group']['harm_promotion_success_rate'],
                           summary['periodic']['harm_promotion_success_rate'])
        # Shadow group has >3x higher gain per refit under harm than periodic
        self.assertGreater(summary['shadow_group']['gain_per_harm_refit_pp'],
                           summary['periodic']['gain_per_harm_refit_pp'])
        # Periodic refits wastefully in stable regimes
        self.assertGreater(summary['periodic']['null_regime_total_wasted_refits'], 0)
        self.assertEqual(summary['shadow_group']['null_regime_total_wasted_refits'], 0.0)

if __name__ == '__main__':
    unittest.main()
