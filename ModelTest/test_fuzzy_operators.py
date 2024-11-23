import unittest
import numpy as np
from Model.t2_fuzzy_operators import t2_fuzzy_min, t2_fuzzy_union

class TestT2FuzzyOperations(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.lmf1 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.umf1 = np.array([0.4, 0.6, 0.8, 1.0, 1.0])

        self.lmf2 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.umf2 = np.array([0.2, 0.5, 0.7, 0.9, 1.0])

        self.term = {'lmf': np.array([0.2, 0.3, 0.5, 0.7, 0.8]),
                     'umf': np.array([0.3, 0.4, 0.6, 0.8, 0.9])}

    def test_t2_fuzzy_min(self):
        # Test MIN operator
        lmf, umf = t2_fuzzy_min(self.term, self.lmf1, self.umf1)

        expected_lmf = np.minimum(self.lmf1, self.term['lmf'])
        expected_umf = np.minimum(self.umf1, self.term['lmf'])

        np.testing.assert_array_almost_equal(lmf, expected_lmf, decimal=6)
        np.testing.assert_array_almost_equal(umf, expected_umf, decimal=6)

    def test_t2_fuzzy_union(self):
        # Test Union operator
        lmf, umf = t2_fuzzy_union((self.lmf1, self.umf1), (self.lmf2, self.umf2))

        expected_lmf = np.amax(np.array([self.lmf1, self.lmf2]), axis=0)
        expected_umf = np.amax(np.array([self.umf1, self.umf2]), axis=0)

        np.testing.assert_array_almost_equal(lmf, expected_lmf, decimal=6)
        np.testing.assert_array_almost_equal(umf, expected_umf, decimal=6)

if __name__ == '__main__':
    unittest.main()
