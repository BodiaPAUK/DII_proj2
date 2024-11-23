import unittest
import numpy as np
from Model.t1_defuzzifier import defuzzification

class TestDefuzzification(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.x = np.array([0, 1, 2, 3, 4, 5])
        self.mf = np.array([0, 0.5, 1, 1, 0.5, 0])

    def test_cog(self):
        # Test Center of Gravity method
        result = defuzzification(self.x, self.mf, defuzz_type='cog')
        expected = np.sum(self.x * self.mf) / np.sum(self.mf)
        self.assertAlmostEqual(result, expected, places=6)

    def test_fom(self):
        # Test First of Maximum method
        result = defuzzification(self.x, self.mf, defuzz_type='fom')
        expected = self.x[np.argmax(self.mf)]
        self.assertEqual(result, expected)

    def test_invalid_defuzz_type(self):
        # Test invalid defuzzification type
        result = defuzzification(self.x, self.mf, defuzz_type='invalid')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()