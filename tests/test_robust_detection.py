import unittest
from research.robust_detection_study import empirical_threshold, summarize, CALIBRATION_SEEDS, TEST_SEEDS
from research.detection_study import scores

class CalibrationTests(unittest.TestCase):
    def test_finite_sample_rank(self):
        self.assertEqual(empirical_threshold(list(range(100))),95)
        with self.assertRaises(ValueError):
            empirical_threshold([1,2])

    def test_strict_threshold_and_missed_runs(self):
        no_alarm=[.5]*58
        hit=[0.]*58
        hit[20]=1 # day 32, delay 2
        pre=[0.]*58
        pre[0]=1
        result=summarize([no_alarm,hit,pre],.5,True)
        self.assertEqual(result['false_alarm_runs'],1)
        self.assertEqual(result['eligible_runs'],2)
        self.assertEqual(result['detected_runs'],1)
        self.assertEqual(result['missed_runs'],1)
        self.assertEqual(result['mean_detection_delay'],2)
        self.assertEqual(result['mean_capped_delay'],21)

    def test_null_counts_entire_horizon(self):
        history=[0.]*58
        history[-1]=1
        self.assertEqual(summarize([history],.5,False)['false_alarm_runs'],1)

    def test_disjoint_seeds_and_identical_prefixes(self):
        self.assertTrue(set(CALIBRATION_SEEDS).isdisjoint(TEST_SEEDS))
        base=scores(3000,'stable')
        for scenario in ('delay','harm','both'):
            other=scores(3000,scenario)
            for method in base:
                self.assertEqual(base[method][:19],other[method][:19])

if __name__=='__main__':
    unittest.main()
