"""Pinned recorded Intel pose graph as an inferred pose-index arrival stream.

Measurements, information matrices and provided initial values are preserved.
The g2o file has no timestamps: factors arrive when both endpoints are present,
keeping their original file order within each arrival. No false-loop injection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pose_graph import require
from se2_stream import wrap


def column_normalized_gradient(poses, edges, gradient):
    """Original-information right Jacobian column norms, independent of whitening."""
    diagonal = [[0., 0., 0.] for _ in poses]
    for a, b, zx, zy, zt, i11, i12, i13, i22, i23, i33 in edges:
        c, s = math.cos(poses[a][2]), math.sin(poses[a][2])
        dx, dy = poses[b][0] - poses[a][0], poses[b][1] - poses[a][1]
        qx, qy = c * dx + s * dy, -s * dx + c * dy
        cz, sz = math.cos(zt), math.sin(zt)
        t = poses[b][2] - poses[a][2] - zt
        ct, st = math.cos(t), math.sin(t)
        columns = [(a, ((-cz, sz, 0.), (-sz, -cz, 0.),
                       (cz * qy - sz * qx, -sz * qy - cz * qx, -1.))),
                   (b, ((ct, st, 0.), (-st, ct, 0.), (0., 0., 1.)))]
        for key, cols in columns:
            if key == 0:
                continue
            for i, (x, y, z) in enumerate(cols):
                value = i11*x*x + i22*y*y + i33*z*z + 2*(i12*x*y + i13*x*z + i23*y*z)
                require(math.isfinite(value) and value > 0., 'invalid original column energy')
                diagonal[key][i] += value
    return max(abs(g) / math.sqrt(s) for grads, cols in zip(gradient[1:], diagonal[1:])
               for g, s in zip(grads, cols))

INTEL_SHA256 = '4d87aaf96e1e04e47c723c371386b15358c71e98c05dad16b786d585f9fd70ff'
INTEL_COMMIT = '5587024b17fd812c66d91740716fbf0bf5824fbc'
INTEL_URL = ('https://raw.githubusercontent.com/uoip/g2opy/' + INTEL_COMMIT +
             '/contrib/vertigo/datasets/intel/originalDataset/intel.g2o')
DATASETS = {
    INTEL_SHA256: {'name': 'intel-vertigo943', 'poses': 943, 'edges': 1837,
                   'commit': INTEL_COMMIT, 'url': INTEL_URL},
    'e648e42b1f24ab01cce76f56c8d8dad0b606f712afe2b92356bf26f195c602be': {
        'name': 'intel-mit1228', 'poses': 1228, 'edges': 1483,
        'commit': '68a0d2419028d598f823c2b35f53b38eacfd0ca1',
        'url': 'https://raw.githubusercontent.com/JeffLIrion/python-graphslam/68a0d2419028d598f823c2b35f53b38eacfd0ca1/data/input_INTEL.g2o'},
}


def identity(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(digest in DATASETS, 'recorded Intel artifact checksum differs')
    return {**DATASETS[digest], 'sha256': digest}


def load(path, frames=None):
    path = Path(path)
    dataset = identity(path)
    vertices, edges, fixes = {}, [], []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        f = text.split('#', 1)[0].split()
        if not f:
            continue
        if f[0] == 'VERTEX_SE2':
            require(len(f) == 5, f'invalid vertex at {number}')
            key = int(f[1]); value = tuple(map(float, f[2:]))
            require(key >= 0 and key not in vertices and all(map(math.isfinite, value)), 'bad vertex')
            vertices[key] = value
        elif f[0] == 'EDGE_SE2':
            require(len(f) == 12, f'invalid edge at {number}')
            a, b = map(int, f[1:3]); values = tuple(map(float, f[3:]))
            require(a >= 0 and a < b and all(map(math.isfinite, values)), 'invalid directed edge')
            i11, i12, i13, i22, i23, i33 = values[3:]
            require(i11 > 0 and i11 * i22 - i12 * i12 > 0 and
                    i11 * (i22 * i33 - i23 * i23) - i12 * (i12 * i33 - i13 * i23) +
                    i13 * (i12 * i23 - i13 * i22) > 0, 'non-SPD information')
            edges.append((a, b, *values))
        elif f[0] == 'FIX':
            require(len(f) == 2, 'invalid fixed record'); fixes.append(int(f[1]))
        else:
            raise ValueError('unsupported dataset record: ' + f[0])
    require(sorted(vertices) == list(range(dataset['poses'])) and len(edges) == dataset['edges'] and not fixes,
            'unexpected Intel structure')
    require(all(a in vertices and b in vertices for a, b, *_ in edges), 'unknown edge endpoint')
    count = len(vertices) - 1 if frames is None else frames
    require(1 <= count < len(vertices), 'invalid real-data prefix length')
    arrivals = [[] for _ in range(count)]
    for edge in edges:
        if edge[1] <= count:
            arrivals[edge[1] - 1].append(edge)
    require(all(arrivals), 'a pose arrives without an earlier connecting constraint')
    return (vertices[0][0], vertices[0][1], wrap(vertices[0][2])), [
        ((vertices[k][0], vertices[k][1], wrap(vertices[k][2])), arrivals[k - 1])
        for k in range(1, count + 1)]


def write(source, output, frames=None):
    root, arrivals = load(source, frames)
    lines = ['ROOT ' + ' '.join(format(v, '.17g') for v in root)]
    for k, (seed, added) in enumerate(arrivals, 1):
        lines.append(f'FRAME {k} ' + ' '.join(format(v, '.17g') for v in seed) + f' {len(added)}')
        lines.extend('EDGE ' + ' '.join(str(v) if isinstance(v, int) else format(v, '.17g')
                                       for v in edge) for edge in added)
    Path(output).write_text('\n'.join(lines) + '\n')
    dataset = identity(source)
    return {'dataset': 'Intel Research Lab recorded SE2 pose graph', 'variant': dataset['name'], 'source_url': dataset['url'],
            'source_commit': dataset['commit'], 'source_sha256': dataset['sha256'],
            'total_source_poses': dataset['poses'], 'total_source_edges': dataset['edges'], 'used_arrivals': len(arrivals),
            'used_edges': sum(len(e) for _, e in arrivals), 'complete_dataset': len(arrivals) == dataset['poses'] - 1,
            'arrival_policy': 'ascending original pose IDs; each directed edge at its later endpoint; stable original file order within arrival; inferred without timestamps',
            'initialization': 'provided original vertex values; angles normalized; no optimized/truth seeds',
            'measurement_information_policy': 'all original selected constraints unchanged; no generated false loops or pruning',
            'stream_sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int)
    a = p.parse_args(); print(json.dumps(write(a.input, a.output, a.frames), indent=2))
