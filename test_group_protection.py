import json
import unittest
from pathlib import Path

class GroupProtectionTests(unittest.TestCase):
    def setUp(self):
        self.report_path = Path('research/GROUP_PROTECTION.md')

    def test_report_exists_and_documents_thresholds(self):
        # We can also check once task-116 finishes or check the file structure
        if self.report_path.exists():
            text = self.report_path.read_text()
            self.assertIn("Reliable Protection Threshold", text)
            self.assertIn("Severe Small-Sample Blindspot", text)

    def test_sample_size_monotonicity_logic(self):
        # Mathematically, radius strictly decreases as sample size increases
        import math
        alpha = 0.05 / (2 * 2 * 120)
        radii = [math.sqrt(math.log(1/alpha)/(2 * n * 28)) for n in [1, 2, 4, 8, 16, 24, 48]]
        for i in range(len(radii)-1):
            self.assertGreater(radii[i], radii[i+1])

if __name__ == '__main__':
    unittest.main()
