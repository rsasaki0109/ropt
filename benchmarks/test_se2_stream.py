import math
import unittest
from se2_stream import compose, energy_gradient, exp, fixture


class OracleTests(unittest.TestCase):
    def test_original_body_gradient_matches_central_differences_with_delayed_loops(self):
        root, frames = fixture(25)
        poses = [root]+[seed for seed, _ in frames]
        edges = [e for _, added in frames for e in added]
        cost, norm, gradient = energy_gradient(poses, edges)
        self.assertTrue(math.isfinite(cost) and norm > 0.)
        self.assertTrue(any(a > 0 and b < 25 and b-a > 1 for a,b,*_ in edges))
        for k in (1, 8, 20, 25):
            for c in range(3):
                d = [0.,0.,0.]; d[c] = 1e-6
                plus, minus = list(poses), list(poses)
                plus[k] = compose(poses[k], exp(d))
                minus[k] = compose(poses[k], exp([-v for v in d]))
                fd = (energy_gradient(plus, edges)[0]-energy_gradient(minus, edges)[0])/2e-6
                self.assertAlmostEqual(fd, gradient[k][c], delta=2e-6)
