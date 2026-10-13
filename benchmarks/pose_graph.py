"""Pinned public-sample benchmark with independent quality gates.

Run the prebuilt release example, then optionally GTSAM's native Pose2 LM.
Raw timings are retained even when a candidate fails the common quality gate.
No cross-library speed ratio is computed. See benchmarks/README.md.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/validation"))
from check_pose_graph import angle, objective, read  # noqa: E402

SHA256 = "69d9959b6e6c530ad268b4d9999383acbd262b54319fd830e1d3075142c73343"
SOURCE = "https://raw.githubusercontent.com/borglab/gtsam/a74146dbea788ea25828041065d82674d6430ada/examples/Data/w100_30.g2o"
EXPECTED_COST = 0.08290751089989


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose_difference(a, b):
    require(a.keys() == b.keys(), "pose IDs differ")
    return max(math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1],
                          angle(a[k][2] - b[k][2])) for k in a)


def gradient_inf(poses, edges, anchors=(0,)):
    derivatives = []
    h = 1e-6
    for key, value in poses.items():
        if key in anchors:
            continue
        for coordinate in range(3):
            plus, minus = dict(poses), dict(poses)
            a, b = list(value), list(value)
            a[coordinate] += h
            b[coordinate] -= h
            plus[key], minus[key] = a, b
            derivatives.append((objective(plus, edges) - objective(minus, edges)) / (2 * h))
    return max(map(abs, derivatives))


def quality(poses, edges, initial, dense):
    cost = objective(poses, edges)
    gradient = gradient_inf(poses, edges)
    delta = pose_difference(poses, dense)
    # These are smoke-test accuracy gates, not global optimality certificates.
    checks = {
        "finite": all(math.isfinite(v) for p in poses.values() for v in p),
        "anchor": poses[0] == initial[0],
        "cost": math.isfinite(cost) and abs(cost - EXPECTED_COST) <= 1e-9,
        "gradient": math.isfinite(gradient) and gradient <= 1e-5,
        "pose_agreement": math.isfinite(delta) and delta <= 1e-6,
    }
    return {"eligible": all(checks.values()), "checks": checks,
            "independent_cost": cost, "stored_gradient_inf": gradient,
            "max_pose_difference_to_dense": delta}


def timing_summary(samples):
    ms = [int(row["elapsed_ns"]) / 1e6 for row in samples]
    require(ms and all(math.isfinite(v) and v > 0 for v in ms), "invalid timing samples")
    return {"samples": len(ms), "median_ms": statistics.median(ms),
            "min_ms": min(ms), "max_ms": max(ms),
            "mean_ms": statistics.mean(ms), "stdev_ms": statistics.pstdev(ms)}


def command_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def repository_state():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).split(b"\0")
    fingerprint = hashlib.sha256()
    for raw in sorted(set(paths) - {b""}):
        path = ROOT / os.fsdecode(raw)
        if path.is_file() and (path.suffix == ".rs" or path.name in ("Cargo.toml", "Cargo.lock")):
            fingerprint.update(raw + b"\0" + path.read_bytes() + b"\0")
    return {"head": command_output("git", "rev-parse", "HEAD"),
            "dirty": bool(command_output("git", "status", "--porcelain")),
            "solver_source_sha256": fingerprint.hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def environment():
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {"platform": platform.platform(), "cpu": model,
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
            "python": platform.python_version(), "rustc": command_output("rustc", "-Vv"),
            "cargo": command_output("cargo", "-V"),
            "thread_variables": {k: os.environ.get(k) for k in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def verify_export(path, original, edges, fixes):
    poses, output_edges, output_fixes = read(path)
    require(poses.keys() == original.keys(), "exported pose IDs differ")
    require(output_fixes == fixes and len(output_edges) == len(edges), "exported constraints differ")
    for a, b in zip(edges, output_edges):
        require(a[:2] == b[:2]
                and max(abs(x - y) for x, y in zip(a[2:4], b[2:4])) < 1e-12
                and abs(angle(a[4] - b[4])) < 1e-12
                and max(abs(x - y) for x, y in zip(a[5:], b[5:])) < 1e-12,
                "export changed edge data")
    return poses


def gtsam_run(original, edges, out, repeats, warmup, dense):
    import gtsam
    import numpy as np

    require(importlib.metadata.version("gtsam") == "4.2.1", "use gtsam==4.2.1")
    require(np.__version__ == "1.26.4", "use numpy==1.26.4")
    values = gtsam.Values()
    for key, p in original.items():
        values.insert(key, gtsam.Pose2(*p))
    graph = gtsam.NonlinearFactorGraph()
    for a, b, zx, zy, za, w00, w01, w02, w11, w12, w22 in edges:
        info = np.array([[w00, w01, w02], [w01, w11, w12], [w02, w12, w22]])
        graph.add(gtsam.BetweenFactorPose2(a, b, gtsam.Pose2(zx, zy, za),
                                         gtsam.noiseModel.Gaussian.Information(info)))
    require(abs(graph.error(values) - objective(original, edges)) < 1e-10,
            "GTSAM initial objective differs")
    # Test off-solution, rotated measurement coordinates, not just the sample's initial energy.
    jacobian_difference = 0.0
    for za in (0.0, math.pi / 2, -2.1):
        probe = gtsam.Values()
        probe.insert(0, gtsam.Pose2())
        probe.insert(1, gtsam.Pose2(3.0, 1.0, za + 0.4))
        factor = gtsam.BetweenFactorPose2(0, 1, gtsam.Pose2(1.0, 0.0, za),
                                        gtsam.noiseModel.Unit.Create(3))
        r = factor.unwhitenedError(probe)
        expected = (2 * math.cos(za) + math.sin(za),
                    -2 * math.sin(za) + math.cos(za), 0.4)
        require(np.max(np.abs(r - expected)) < 1e-12, "GTSAM residual convention differs")
        # Diagnose this exact installed native factor, without assuming how its wheel was built.
        numerical = np.empty((3, 6))
        h = 1e-6
        for column in range(6):
            key = column // 3
            delta = np.zeros(3)
            delta[column % 3] = h
            plus, minus = gtsam.Values(probe), gtsam.Values(probe)
            plus.update(key, probe.atPose2(key).retract(delta))
            minus.update(key, probe.atPose2(key).retract(-delta))
            numerical[:, column] = (factor.unwhitenedError(plus)
                                     - factor.unwhitenedError(minus)) / (2 * h)
        native = factor.linearize(probe).jacobian()[0]
        jacobian_difference = max(jacobian_difference, float(np.max(np.abs(native - numerical))))
    # Exact equality, not a small-sigma approximate prior. QR handles this constraint.
    graph.add(gtsam.NonlinearEqualityPose2(0, values.atPose2(0)))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setlambdaInitial(1e-3)
    params.setAbsoluteErrorTol(1e-12)
    params.setRelativeErrorTol(1e-12)
    params.setErrorTol(0.0)
    params.setLinearSolverType("MULTIFRONTAL_QR")
    (out / "gtsam-options.txt").write_text(str(params), encoding="utf-8")
    samples, solution = [], None
    for iteration in range(warmup + repeats):
        initial = gtsam.Values(values)
        start = time.perf_counter_ns()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        elapsed = time.perf_counter_ns() - start
        poses = {k: (result.atPose2(k).x(), result.atPose2(k).y(), result.atPose2(k).theta())
                 for k in original}
        cost = objective(poses, edges)
        require(poses[0] == original[0], "GTSAM anchor moved")
        require(abs(cost - graph.error(result)) < 1e-10, "GTSAM final objective differs")
        require(math.isfinite(cost) and cost <= objective(original, edges), "GTSAM invalid cost")
        if solution is not None:
            require(pose_difference(poses, solution) <= 1e-10, "GTSAM repeated solution differs")
        solution = poses
        if iteration >= warmup:
            samples.append({"sample": iteration - warmup, "elapsed_ns": elapsed,
                            "initial_cost": graph.error(values), "final_cost": cost,
                            "optimizer_iterations": optimizer.iterations(),
                            "final_lambda": optimizer.lambda_()})
    with (out / "gtsam-samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (out / "gtsam-poses.json").write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
    native_module = Path(gtsam.gtsam.__file__)
    return {"timing": timing_summary(samples), "quality": quality(solution, edges, original, dense),
            "versions": {p: importlib.metadata.version(p) for p in ("gtsam", "numpy", "pyparsing")},
            "native_module_sha256": hashlib.sha256(native_module.read_bytes()).hexdigest(),
            "diagnostics": {"probe_native_vs_central_jacobian_max_difference": jacobian_difference},
            "optimizer_iterations": sorted({row["optimizer_iterations"] for row in samples}),
            "iteration_semantics": "GTSAM optimizer iterations, not ropt LM trial counts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ropt", type=Path, required=True, help="prebuilt release benchmark_pose_graph")
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the checkout")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--without-gtsam", action="store_true")
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmup >= 0, "invalid repetition counts")
    require(hashlib.sha256(args.input.read_bytes()).hexdigest() == SHA256, "dataset checksum differs")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "store generated results outside the checkout")
    original, edges, fixes = read(args.input)
    require(len(original) == 30 and len(edges) == 59 and not fixes and 0 in original,
            "unexpected pinned sample structure")
    metadata = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": {"source": SOURCE, "sha256": SHA256, "poses": 30, "edges": 59,
                            "active_tangent": 87, "residuals": 177, "anchor": 0},
                "repository": repository_state(), "environment": environment(),
                "release_executable_sha256": hashlib.sha256(args.ropt.read_bytes()).hexdigest(),
                "warmup": args.warmup, "repeats": args.repeats,
                "quality_thresholds": {"cost_absolute": 1e-9, "gradient_inf": 1e-5, "pose_delta": 1e-6},
                "memory_measured": False,
                "scope": "small-sample configured solver runs; no SOTA or general speed ranking"}
    out.mkdir()
    # Persist identity even if a subprocess or later validation fails.
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    subprocess.run([str(args.ropt.resolve()), "--input", str(args.input.resolve()),
                    "--output", str(out / "ropt"), "--repeats", str(args.repeats),
                    "--warmup", str(args.warmup)], check=True, cwd=ROOT)
    solutions = {name: verify_export(out / "ropt" / f"{name}.g2o", original, edges, fixes)
                 for name in ("dense", "qr", "scalar_cg", "block_cg")}
    with (out / "ropt/samples.csv").open(encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    cases = {}
    for name, poses in solutions.items():
        rows = [row for row in raw if row["backend"] == name]
        require(len(rows) == args.repeats, "missing timing samples")
        for row in rows:
            require(row["termination"] in ("Cost", "Step", "Gradient"), "nonconverged ropt sample")
            require(abs(float(row["final_cost"]) - objective(poses, edges)) < 1e-10,
                    "ropt reported cost differs from independent objective")
        cases[f"ropt_{name}"] = {"timing": timing_summary(rows),
                                 "quality": quality(poses, edges, original, solutions["dense"]),
                                 "nonlinear_trials": sorted({int(r["nonlinear_trials"]) for r in rows}),
                                 "linear_iterations": sorted({int(r["linear_iterations"]) for r in rows})}
    if not args.without_gtsam:
        cases["gtsam_native"] = gtsam_run(original, edges, out, args.repeats, args.warmup,
                                          solutions["dense"])
    passed = all(case["quality"]["eligible"] for case in cases.values())
    report = {"metadata": metadata, "cases": cases, "all_quality_gates_passed": passed,
              "cross_library_quality_eligible": "gtsam_native" in cases and passed}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, case in cases.items():
        q, t = case["quality"], case["timing"]
        print(f"{name}: median={t['median_ms']:.4f} ms, cost={q['independent_cost']:.12e}, "
              f"gradient_inf={q['stored_gradient_inf']:.3e}, eligible={q['eligible']}")
    print(f"Raw samples, options, solutions and environment saved to {out}")
    if not passed:
        print("A common quality gate failed; cross-library speed comparison is ineligible.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
