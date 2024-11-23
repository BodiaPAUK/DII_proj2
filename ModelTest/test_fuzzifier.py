import unittest
import numpy as np
from Model.t2_fuzzifier import fuzzification

class TestFuzzification(unittest.TestCase):

    def setUp(self):
        # Example input for fuzzification
        self.crisp_values = [5, 15]
        self.input_lvs = [
            {
                'X': (0, 10, 1),  # Range: (min, max, step)
                'U': np.linspace(0, 10, 11),  # Universe
                'terms': {
                    'Low': {'lmf': np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0, 0]),
                            'umf': np.array([1, 1, 1, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])},
                    'High': {'lmf': np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]),
                             'umf': np.array([0, 0, 0, 0, 0.4, 0.6, 1, 1, 1, 1, 1])}
                }
            },
            {
                'X': (10, 20, 1),  # Range: (min, max, step)
                'U': np.linspace(10, 20, 11),  # Universe
                'terms': {
                    'Medium': {'lmf': np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1, 0.8, 0.6, 0.4, 0.2]),
                               'umf': np.array([0, 0.4, 0.6, 1, 1, 1, 1, 1, 0.6, 0.4, 0])}
                }
            }
        ]

    def test_fuzzification(self):
        # Run fuzzification
        result = fuzzification(self.crisp_values, self.input_lvs)

        # Expected results
        expected_result = {
            0: {'Low': (0.0, 0.2), 'High': (0.4, 0.6)},
            1: {}
        }

        self.assertEqual(result, expected_result)

    def test_empty_terms(self):
        # Test with no terms in input
        empty_lvs = [{'X': (0, 10, 1), 'U': np.linspace(0, 10, 11), 'terms': {}}]
        result = fuzzification([5], empty_lvs)
        self.assertEqual(result, {0: {}})  # Expect empty result for no terms

if __name__ == '__main__':
    unittest.main()
