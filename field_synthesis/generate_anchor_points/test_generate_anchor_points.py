import unittest
import numpy as np
from generate_anchor_points import generate_anchor_points

import unittest
import numpy as np
from generate_anchor_points import generate_anchor_points

class TestGenerateAnchorPoints(unittest.TestCase):

    def test_standardni_vstup(self):
        pocet = 20
        dist = 10
        area = 100
        body = generate_anchor_points(pocet, dist, area)
        
        self.assertIsInstance(body, np.ndarray)
        self.assertEqual(body.shape[1], 2)
        self.assertLessEqual(len(body), pocet)

    def test_zaporny_pocet_bodu(self):
        body = generate_anchor_points(-5, 10, 100)
        self.assertEqual(len(body), 0)
        self.assertIsInstance(body, np.ndarray)

    def test_nulova_plocha(self):
        body = generate_anchor_points(10, 5, 0)
        self.assertEqual(len(body), 0)

    def test_extremni_vzdalenost(self):
        body = generate_anchor_points(10, 500, 100)
        self.assertIsInstance(body, np.ndarray)
        self.assertLessEqual(len(body), 10)

    def test_zaporna_vzdalenost(self):
        body = generate_anchor_points(10, -10, 100)
        self.assertEqual(len(body), 10)

    def test_koordinaty_v_mezich(self):
        area = 50
        body = generate_anchor_points(100, 1, area)
        if len(body) > 0:
            self.assertTrue(np.all(body >= 0))
            self.assertTrue(np.all(body <= area))

    def test_minimalni_vzdalenost(self):
        pocet = 10
        dist = 20
        area = 100
        body = generate_anchor_points(pocet, dist, area)
        
        if len(body) >= 2:
            from scipy.spatial.distance import pdist
            vzdalenosti = pdist(body)
            self.assertGreaterEqual(np.min(vzdalenosti), dist * 0.99)

if __name__ == '__main__':
    unittest.main()