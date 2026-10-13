"""Independent regression checks for benchmark format conversion and the oracle."""
import math
from pathlib import Path
import tempfile
import unittest

from convert_toro import information_to_g2o
from pose_graph import gradient_inf, objective, verify_export
from scalable_pose_graph import gradient_inf_incident
from paired_large import quality
from isam_stream import oracle as stream_oracle, energy_gradient as stream_energy


class Validation(unittest.TestCase):
    def test_stream_block_oracle_is_stationary_with_late_factors(self):
        for frames in (1, 24, 25, 50, 100):
            x = stream_oracle(frames)
            cost, gradient = stream_energy(x)
            self.assertGreaterEqual(cost, 0.)
            self.assertEqual(x[0], (0., 0.))
            self.assertLess(max(abs(v) for p in gradient[1:] for v in p), 1e-12)

    def test_stream_scalar_objective_gradient_agrees_with_central_differences(self):
        x = stream_oracle(25)
        # Perturb an existing state affected by the delayed root constraint.
        x[12] = (x[12][0]+.03, x[12][1]-.02)
        _, gradient = stream_energy(x)
        for coordinate in (0, 1):
            plus, minus = list(x), list(x)
            p, m = list(x[12]), list(x[12])
            p[coordinate] += 1e-6
            m[coordinate] -= 1e-6
            plus[12], minus[12] = tuple(p), tuple(m)
            numerical = (stream_energy(plus)[0]-stream_energy(minus)[0])/(2e-6)
            self.assertAlmostEqual(gradient[12][coordinate], numerical, places=8)

    def test_large_comparison_rejects_pose_disagreement_even_with_small_gradient(self):
        reference = {0: (0., 0., 0.), 1: (1., 0., 0.)}
        # Weak information makes both cost and gradient pass despite a real
        # pose discrepancy. The existing 1e-6 pose gate must still reject it.
        edges = [(0, 1, 1., 0., 0., 1e-8, 0., 0., 1e-8, 0., 1e-8)]
        candidate = {**reference, 1: (1.001, 0., 0.)}
        check = quality(candidate, edges, reference, reference)
        self.assertTrue(check["checks"]["cost"])
        self.assertTrue(check["checks"]["gradient"])
        self.assertFalse(check["checks"]["pose_agreement"])
        self.assertFalse(check["eligible"])
        self.assertTrue(quality(reference, edges, reference, reference)["eligible"])
        self.assertFalse(quality({**reference, 1: (math.nan, 0., 0.)},
                                 edges, reference, reference)["eligible"])

    def test_incident_gradient_matches_full_objective_with_loops_and_correlations(self):
        poses = {0: (.1, -.2, .1), 1: (.5, -.4, .3), 2: (1., .2, -.2)}
        info = (6., .7, -.4, 5., .2, 4.)
        edges = [(0, 1, .3, -.1, .2, *info), (1, 2, .4, .2, -.3, *info),
                 (2, 0, -.8, -.1, .3, *info), (1, 1, .1, .2, .1, *info)]
        for anchors in ({0}, {1, 2}, set(), set(poses)):
            if anchors == set(poses):
                self.assertEqual(gradient_inf_incident(poses, edges, anchors), 0.)
            else:
                self.assertAlmostEqual(gradient_inf_incident(poses, edges, anchors),
                                       gradient_inf(poses, edges, anchors), places=8)

    def test_incident_gradient_ignores_unrelated_large_constant_cost(self):
        poses = {0: (0., 0., 0.), 1: (.2, -.4, .3), 2: (1e10, 0., 0.)}
        info = (6., .7, -.4, 5., .2, 4.)
        edges = [(0, 1, 0., 0., 0., *info), (0, 2, 0., 0., 0., *info)]
        self.assertAlmostEqual(gradient_inf_incident(poses, edges, {0, 2}), 1.8, places=8)

    def test_correlated_toro_energy_and_gradient(self):
        # Independently specified SPD matrix [[6,.7,-.4],[.7,5,.2],[-.4,.2,4]].
        info = information_to_g2o([6, .7, 5, 4, -.4, .2])
        poses = {0: (0, 0, 0), 1: (.2, -.4, .3)}
        edges = [(0, 1, 0, 0, 0, *info)]
        self.assertAlmostEqual(objective(poses, edges), .596, places=14)
        # Omega*r = [.8,-1.8,1.04]; anchor excluded.
        self.assertAlmostEqual(gradient_inf(poses, edges), 1.8, places=8)

    def test_export_accepts_physically_identical_wrapped_measurement(self):
        original = {0: (0, 0, 0), 1: (1, 0, 0)}
        edge = (0, 1, .2, -.4, 3.2, 6, .7, -.4, 5, .2, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.g2o"
            fields = ["EDGE_SE2", *edge]
            fields[5] = 3.2 - 2*math.pi
            prefix = "VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 1 0 0\n"
            path.write_text(prefix + " ".join(map(str, fields)) + "\n")
            self.assertEqual(verify_export(path, original, [edge], set()), original)
            # Each changes a real constraint, rather than just its angle representation.
            for index in (3, 4, 5, 6, 7, 8, 9, 10, 11):
                changed = fields.copy()
                changed[index] += .01
                path.write_text(prefix + " ".join(map(str, changed)) + "\n")
                with self.assertRaisesRegex(ValueError, "edge data"):
                    verify_export(path, original, [edge], set())

    def test_wrong_information_size_is_rejected(self):
        with self.assertRaises(ValueError):
            information_to_g2o([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
