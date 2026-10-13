"""Independent scalar check for the documented w100_30 validation run (stdlib only).

Usage: python3 check_pose_graph.py input.g2o dense.g2o sparse.g2o
This checks the unrobust g2o coordinate objective, not a Lie Log objective.
"""
import math
import sys


def read(path):
    poses, edges, fixes = {}, [], set()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] in ("VERTEX_SE2", "VERTEX2"):
                poses[int(fields[1])] = tuple(map(float, fields[2:]))
            elif fields[0] in ("EDGE_SE2", "EDGE2"):
                edges.append((int(fields[1]), int(fields[2]), *map(float, fields[3:])))
            elif fields[0] == "FIX":
                fixes.add(int(fields[1]))
            else:
                raise ValueError(f"unsupported tag {fields[0]}")
    return poses, edges, fixes


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def objective(poses, edges):
    terms = []
    for i, j, zx, zy, za, w00, w01, w02, w11, w12, w22 in edges:
        ax, ay, aa = poses[i]
        bx, by, ba = poses[j]
        dx, dy = bx - ax, by - ay
        px = math.cos(aa) * dx + math.sin(aa) * dy - zx
        py = -math.sin(aa) * dx + math.cos(aa) * dy - zy
        ex = math.cos(za) * px + math.sin(za) * py
        ey = -math.sin(za) * px + math.cos(za) * py
        ea = angle(ba - aa - za)
        terms.append(0.5 * (w00 * ex**2 + w11 * ey**2 + w22 * ea**2
                            + 2 * (w01 * ex * ey + w02 * ex * ea + w12 * ey * ea)))
    return math.fsum(terms)


def check(input_path, dense_path, sparse_path):
    original, edges, fixes = read(input_path)
    dense, dense_edges, dense_fixes = read(dense_path)
    sparse, sparse_edges, sparse_fixes = read(sparse_path)
    # This sample has one connected component and no FIX record; gauge anchor is 0.
    assert len(original) == 30 and len(edges) == 59 and not fixes
    assert original.keys() == dense.keys() == sparse.keys()
    assert dense_fixes == sparse_fixes == fixes
    for output_edges in (dense_edges, sparse_edges):
        assert len(output_edges) == len(edges)
        for before, after in zip(edges, output_edges):
            assert before[:2] == after[:2]
            assert max(abs(a - b) for a, b in zip(before[2:], after[2:])) < 1e-12
    assert original[0] == dense[0] == sparse[0]
    initial_cost = objective(original, edges)
    dense_cost, sparse_cost = objective(dense, edges), objective(sparse, edges)
    assert abs(initial_cost - 10.72025504017) < 1e-10
    assert abs(dense_cost - 0.08290751089989) < 1e-12
    assert abs(dense_cost - sparse_cost) < 1e-12
    max_difference = max(math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2
                                  + angle(a[2] - b[2])**2)
                         for key in dense for a, b in [(dense[key], sparse[key])])
    assert max_difference < 1e-7
    # Gradient in absolute stored coordinates, computed by central differences.
    gradient = []
    h = 1e-6
    for key, value in sparse.items():
        if key == 0:
            continue
        for k in range(3):
            plus, minus = dict(sparse), dict(sparse)
            vp, vm = list(value), list(value)
            vp[k] += h
            vm[k] -= h
            plus[key], minus[key] = vp, vm
            gradient.append((objective(plus, edges) - objective(minus, edges)) / (2 * h))
    gradient_inf = max(map(abs, gradient))
    assert gradient_inf < 1e-5
    print(f"independent cost: {initial_cost:.12e} -> {sparse_cost:.12e}")
    print(f"max dense/sparse pose difference: {max_difference:.3e}")
    print(f"finite-difference stored-coordinate gradient infinity norm: {gradient_inf:.3e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_pose_graph.py input.g2o dense.g2o sparse.g2o")
    check(*sys.argv[1:])
