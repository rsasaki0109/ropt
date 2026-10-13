"""Independent scalar validation of the synthetic correlated linear stream."""
import csv
import math
from pathlib import Path

COVARIANCES = ((1., .2, .2, 2.), (.5, -.1, -.1, .8), (.3, .04, .04, .4))


def inverse(a):
    det = a[0]*a[3]-a[1]*a[2]
    if not (det > 0 and math.isfinite(det)):
        raise ValueError("oracle block is not SPD")
    return (a[3]/det, -a[1]/det, -a[2]/det, a[0]/det)


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(a, scalar):
    return tuple(x*scalar for x in a)


def matvec(a, x):
    return (a[0]*x[0]+a[1]*x[1], a[2]*x[0]+a[3]*x[1])


def matmul(a, b):
    return (a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3])


def observed(k):
    return (.1*k+.012*math.sin(.37*k), -.03*k+.015*math.cos(.21*k))


def odometry(k):
    return (.1+.002*math.sin(.51*k), -.03+.003*math.cos(.19*k))


def closure(k):
    key = k//2
    return key, (.1*key+.004*math.cos(.17*k), -.03*key+.003*math.sin(.31*k))


def oracle(n):
    observation, between, loop = map(inverse, COVARIANCES)
    diagonal = [observation]*n
    rhs = [matvec(observation, observed(k)) for k in range(1, n+1)]
    for k in range(1, n+1):
        diagonal[k-1] = add(diagonal[k-1], between)
        rhs[k-1] = add(rhs[k-1], matvec(between, odometry(k)))
        if k > 1:
            diagonal[k-2] = add(diagonal[k-2], between)
            rhs[k-2] = add(rhs[k-2], scale(matvec(between, odometry(k)), -1))
        if k % 25 == 0:
            old, target = closure(k)
            diagonal[old-1] = add(diagonal[old-1], loop)
            rhs[old-1] = add(rhs[old-1], matvec(loop, target))
    inverses = []
    for k in range(n):
        if k:
            multiplier = matmul(scale(between, -1), inverses[k-1])
            diagonal[k] = add(diagonal[k], scale(matmul(multiplier, scale(between, -1)), -1))
            rhs[k] = add(rhs[k], scale(matvec(multiplier, rhs[k-1]), -1))
        inverses.append(inverse(diagonal[k]))
    x = [(0., 0.)]*(n+1)
    for k in reversed(range(n)):
        tail = matvec(between, x[k+2]) if k+1 < n else (0., 0.)
        x[k+1] = matvec(inverses[k], add(rhs[k], tail))
    return x


def energy_gradient(x):
    observation, between, loop = map(inverse, COVARIANCES)
    terms, gradient = [], [(0., 0.)]*len(x)

    def factor(a, b, target, precision):
        residual = add(x[b], scale(target, -1))
        if a is not None:
            residual = add(residual, scale(x[a], -1))
        weighted = matvec(precision, residual)
        terms.append(.5*sum(p*q for p, q in zip(residual, weighted)))
        gradient[b] = add(gradient[b], weighted)
        if a is not None:
            gradient[a] = add(gradient[a], scale(weighted, -1))
    for k in range(1, len(x)):
        factor(None, k, observed(k), observation)
        factor(k-1, k, odometry(k), between)
        if k % 25 == 0:
            old, target = closure(k)
            factor(0, old, target, loop)
    return math.fsum(terms), gradient


def validate(directory, frames, incremental):
    with (directory/"estimates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    poses = {int(r["key"]): (float(r["x"]), float(r["y"])) for r in rows}
    if len(rows) != frames+1 or poses.keys() != set(range(frames+1)):
        raise ValueError("returned keys differ")
    x = [poses[k] for k in range(frames+1)]
    if not all(math.isfinite(v) for p in x for v in p):
        raise ValueError("nonfinite estimates")
    reference = oracle(frames)
    cost, gradient = energy_gradient(x)
    delta = max(math.hypot(a[0]-b[0], a[1]-b[1]) for a, b in zip(x, reference))
    norm = max(abs(v) for p in gradient[1:] for v in p)
    checks = {"anchor": x[0] == (0., 0.), "pose": delta <= 1e-8,
              "gradient": norm <= 1e-8,
              "cost": abs(cost-energy_gradient(reference)[0]) <= 1e-9*max(1, abs(cost))}
    with (directory/"updates.csv").open() as stream:
        updates = list(csv.DictReader(stream))
    checks["all_prefixes"] = (len(updates) == frames and
            {int(r["frame"]) for r in updates} == set(range(1, frames+1)) and
            all(float(r["max_pose_delta"]) <= 1e-8 and float(r["gradient_inf"]) <= 1e-8
                and abs(float(r["cost"])-float(r["reference_cost"])) <= 1e-9*max(1, abs(float(r["reference_cost"])))
                for r in updates))
    checks["callbacks"] = (not incremental or all(int(r["callbacks"]) == 2+int(int(r["frame"]) % 25 == 0) for r in updates))
    return {"eligible": all(checks.values()), "checks": checks,
            "independent_cost": cost, "gradient_inf": norm, "pose_delta": delta}
