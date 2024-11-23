import unittest
import numpy as np
from Model.t1_membership_functions import trapmf, trimf

class TestMembershipFunctions(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.x = np.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]

    def test_trapmf(self):
        # Test trapezoidal membership function
        a, b, c, d, h = 2, 4, 6, 8, 1  # Trapezoidal parameters
        result = trapmf(self.x, a, b, c, d, h)
        expected = np.array([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0])  # Precomputed values
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_trimf(self):
        # Test triangular membership function
        a, b, c, h = 2, 5, 8, 1  # Triangular parameters
        result = trimf(self.x, a, b, c, h)
        expected = np.array([0, 0, 0, 0.333333, 0.666667, 1, 0.666667, 0.333333, 0, 0, 0])  # Precomputed values
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_trapmf_with_h(self):
        # Test trapezoidal membership function with h < 1
        a, b, c, d, h = 2, 4, 6, 8, 0.8
        result = trapmf(self.x, a, b, c, d, h)
        expected = np.array([0, 0, 0, 0.5, 0.8, 0.8, 0.8, 0.5, 0, 0, 0])  # Precomputed values with h scaling
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_trimf_with_h(self):
        # Test triangular membership function with h < 1
        a, b, c, h = 2, 5, 8, 0.7
        result = trimf(self.x, a, b, c, h)
        expected = np.array([0, 0, 0, 0.333333, 0.666667, 0.7, 0.666667, 0.333333, 0, 0, 0])  # Precomputed values
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

if __name__ == '__main__':
    unittest.main()
