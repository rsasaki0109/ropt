"""Conversion checks on hand-built directed constraints, not optimizer copies."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import real_se2_stream as real
from se2_stream import compose, inverse, exp, energy_gradient


class RealStreamTests(unittest.TestCase):
    def test_normalized_original_gradient_matches_numeric_right_tangent_columns(self):
        poses = [(2., -1., .4), (3., .5, .9), (4., -.3, 1.2)]
        edges = [(0, 1, 1., .2, .3, 4., .5, .2, 3., .1, 5.),
                 (1, 2, .8, -.1, .2, 6., .3, -.2, 4., .1, 8.),
                 (0, 2, 1.8, .1, .5, 3., -.1, .2, 2., .3, 7.)]
        _, _, gradient = energy_gradient(poses, edges)
        reference = 0.
        h = 1e-6
        for key in (1, 2):
            for coordinate in range(3):
                delta = [0., 0., 0.]; delta[coordinate] = h
                plus, minus = list(poses), list(poses)
                plus[key] = compose(poses[key], exp(delta))
                minus[key] = compose(poses[key], exp([-v for v in delta]))
                square = 0.
                for a, b, *data in edges:
                    def residual(values):
                        return compose(inverse(tuple(data[:3])), compose(inverse(values[a]), values[b]))
                    column = [(p - m)/(2*h) for p, m in zip(residual(plus), residual(minus))]
                    x, y, z = column
                    i11, i12, i13, i22, i23, i33 = data[3:]
                    square += i11*x*x + i22*y*y + i33*z*z + 2*(i12*x*y + i13*x*z + i23*y*z)
                fd = (energy_gradient(plus, edges)[0] - energy_gradient(minus, edges)[0])/(2*h)
                reference = max(reference, abs(fd)/math.sqrt(square))
        self.assertAlmostEqual(real.column_normalized_gradient(poses, edges, gradient), reference, delta=1e-8)

    def fixture(self, text, directory):
        source = Path(directory) / 'input.g2o'; source.write_text(text)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        record = {'name': 'test', 'poses': 3, 'edges': 3, 'commit': 'test', 'url': 'test'}
        return source, digest, record

    def test_arrival_preserves_file_order_measurements_information_and_provided_seeds(self):
        text = ('VERTEX_SE2 0 2 -1 0.4\nVERTEX_SE2 1 3 4 0.8\nVERTEX_SE2 2 8 9 1.2\n'
                'EDGE_SE2 0 2 7 8 0.7 4 0.5 0.2 3 0.1 5\n'
                'EDGE_SE2 0 1 2 3 0.3 4 0.5 0.2 3 0.1 5\n'
                'EDGE_SE2 1 2 5 6 0.6 4 0.5 0.2 3 0.1 5\n')
        with tempfile.TemporaryDirectory() as d:
            source, digest, record = self.fixture(text, d)
            with patch.dict(real.DATASETS, {digest: record}):
                root, frames = real.load(source)
                self.assertEqual(root, (2., -1., real.wrap(.4)))
                self.assertEqual([len(e) for _, e in frames], [1, 2])
                self.assertEqual(frames[1][0], (8., 9., real.wrap(1.2)))
                self.assertEqual(frames[1][1][0], (0, 2, 7., 8., .7, 4., .5, .2, 3., .1, 5.))
                self.assertEqual(frames[1][1][1][:2], (1, 2))
                output = Path(d) / 'input.stream'; metadata = real.write(source, output)
                self.assertTrue(metadata['complete_dataset'])
                self.assertEqual(metadata['used_edges'], 3)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), metadata['stream_sha256'])
                metadata = real.write(source, output, 1)
                self.assertFalse(metadata['complete_dataset'])
                self.assertEqual(metadata['used_edges'], 1)

    def test_mutated_unpinned_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input.g2o'; source.write_text('VERTEX_SE2 0 0 0 0\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                real.load(source)

    def test_edge_only_seeds_integrate_original_odometry_without_loop_measurements(self):
        text = ('EDGE_SE2 0 2 99 98 0 4 0 0 3 0 5\n'
                'EDGE_SE2 1 2 2 0 0 4 0 0 3 0 5\n'
                f'EDGE_SE2 0 1 1 0 {math.pi/2:.17g} 4 0 0 3 0 5\n')
        with tempfile.TemporaryDirectory() as d:
            source, digest, record = self.fixture(text, d)
            record.update(seed_policy='odometry-chain', recording='test recording')
            with patch.dict(real.DATASETS, {digest: record}):
                root, frames = real.load(source)
                self.assertEqual(root, (0., 0., 0.))
                self.assertAlmostEqual(frames[1][0][0], 1.)
                self.assertAlmostEqual(frames[1][0][1], 2.)
                self.assertAlmostEqual(frames[1][0][2], math.pi/2)
                self.assertEqual(frames[1][1][0][2:5], (99., 98., 0.))
                self.assertEqual(frames[1][1][1][0:2], (1, 2))
                metadata = real.write(source, Path(d)/'stream')
                self.assertEqual(metadata['dataset'], 'test recording')
                self.assertIn('odometry', metadata['initialization'])
                self.assertTrue(metadata['complete_dataset'])

    def test_edge_only_requires_unique_complete_chain_and_no_supplied_vertices(self):
        edge = 'EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1\n'
        loop = 'EDGE_SE2 0 2 2 0 0 1 0 0 1 0 1\n'
        for text in (edge + loop + loop, edge + edge + loop,
                     'VERTEX_SE2 0 0 0 0\n' + edge + loop + loop):
            with tempfile.TemporaryDirectory() as d:
                source, digest, record = self.fixture(text, d)
                record['seed_policy'] = 'odometry-chain'
                with patch.dict(real.DATASETS, {digest: record}), self.assertRaises(ValueError):
                    real.load(source)

    def test_non_spd_information_and_disconnected_arrival_are_rejected(self):
        header = 'VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 1 0 0\nVERTEX_SE2 2 2 0 0\n'
        for body in [
            'EDGE_SE2 0 1 1 0 0 -1 0 0 1 0 1\nEDGE_SE2 0 2 2 0 0 1 0 0 1 0 1\nEDGE_SE2 1 2 1 0 0 1 0 0 1 0 1\n',
            'EDGE_SE2 0 2 2 0 0 1 0 0 1 0 1\nEDGE_SE2 1 2 1 0 0 1 0 0 1 0 1\nEDGE_SE2 0 2 2 0 0 1 0 0 1 0 1\n',
        ]:
            with tempfile.TemporaryDirectory() as d:
                source, digest, record = self.fixture(header + body, d)
                with patch.dict(real.DATASETS, {digest: record}), self.assertRaises(ValueError):
                    real.load(source)


if __name__ == '__main__':
    unittest.main()
