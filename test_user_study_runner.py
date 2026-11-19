import unittest
from user_study_runner import conduct_session, TASKS

class UserStudyTests(unittest.TestCase):
    def test_consent_is_strictly_enforced(self):
        with self.assertRaises(PermissionError):
            conduct_session("P-001", consent_given=False)

    def test_protocol_evaluates_tasks_and_criteria(self):
        responses = {
            1: {"completed": True, "correct_concept": True, "time_seconds": 30, "confidence_rating": 4},
            2: {"completed": True, "correct_concept": True, "time_seconds": 45, "confidence_rating": 4},
            3: {"completed": True, "correct_concept": True, "time_seconds": 40, "confidence_rating": 5},
            4: {"completed": True, "correct_concept": True, "time_seconds": 60, "confidence_rating": 5},
            5: {"completed": True, "correct_concept": True, "time_seconds": 50, "confidence_rating": 4},
            6: {"completed": True, "correct_concept": True, "time_seconds": 55, "confidence_rating": 4}
        }
        res = conduct_session("TEST-PARTICIPANT", consent_given=True, responses=responses)
        self.assertEqual(len(res["tasks"]), len(TASKS))
        self.assertTrue(res["summary"]["meets_completion_criterion"])
        self.assertTrue(res["summary"]["meets_interpretation_criterion"])
        self.assertTrue(res["summary"]["no_fairness_claims_confirmed"])

    def test_failure_on_poor_interpretation(self):
        responses = {
            1: {"completed": True, "correct_concept": True},
            2: {"completed": True, "correct_concept": True},
            3: {"completed": True, "correct_concept": True},
            4: {"completed": True, "correct_concept": False}  # Failed interpretation
        }
        res = conduct_session("TEST-FAIL", consent_given=True, responses=responses)
        self.assertFalse(res["summary"]["meets_interpretation_criterion"])

if __name__ == '__main__':
    unittest.main()
