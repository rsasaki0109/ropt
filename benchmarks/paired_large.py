"""Balanced large-graph comparisons with a native QR consistency reference.

No dense reference is allocated. The scalar cost, numerical gradient, exact
anchor and pose-agreement gates are unchanged. Failed cases retain their raw
samples and checks, and cannot support a speed claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import math

from exact_pose_graph import dataset_identity, load_samples, native_poses
from paired_pcg import source_fingerprint
from pose_graph import (ROOT, environment, objective, pose_difference, read,
                        require, timing_summary, verify_export)
from scalable_pose_graph import gradient_inf_incident


def quality(poses, edges, original, reference):
    require(poses.keys() == original.keys(), "pose IDs differ")
    finite = all(math.isfinite(v) for pose in poses.values() for v in pose)
    if not finite:
        return {"eligible": False, "checks": {"finite": False}}
    cost = objective(poses, edges)
    gradient = gradient_inf_incident(poses, edges)
    delta = pose_difference(poses, reference)
    reference_cost = objective(reference, edges)
    checks = {"finite": all(map(math.isfinite, (cost, gradient, delta))),
              "anchor": poses[0] == original[0],
              "cost": abs(cost-reference_cost) <= 1e-9*max(1, abs(reference_cost)),
              "gradient": gradient <= 1e-5, "pose_agreement": delta <= 1e-6}
    return {"eligible": all(checks.values()), "checks": checks,
            "independent_cost": cost, "stored_gradient_inf": gradient,
            "max_pose_difference_to_native_qr": delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ropt", "gtsam", "build-manifest", "input", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--cost-tolerance", type=float, default=1e-15)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--native-solvers", choices=("qr,cholesky", "qr", "cholesky"), default="qr,cholesky")
    args = parser.parse_args()
    variants = ("ropt_block_analytic",)+tuple("gtsam_"+name for name in args.native_solvers.split(','))
    require(args.rounds > 0 and args.rounds % (2*len(variants)) == 0,
            "rounds must be a positive multiple of twice the candidate count")
    require(args.max_iterations > 0 and math.isfinite(args.cost_tolerance) and args.cost_tolerance >= 0,
            "invalid stopping configuration")
    identity = dataset_identity(args.input)
    original, edges, fixes = read(args.input)
    require(not fixes and min(original) == 0, "comparison uses automatic anchor 0")
    adjacent = {k: set() for k in original}
    for a, b, *_ in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    visited, stack = set(), [0]
    while stack:
        key = stack.pop()
        if key not in visited:
            visited.add(key)
            stack.extend(adjacent[key]-visited)
    require(visited == set(original), "comparison requires a connected graph")
    build = json.loads(args.build_manifest.read_text())
    require(build["exact_between"] and build["archive_sha256"] ==
            "96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2",
            "unexpected native build provenance")
    require(build["executable_sha256"] == hashlib.sha256(args.gtsam.read_bytes()).hexdigest(),
            "native executable differs from manifest")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "results must be outside checkout")
    out.mkdir()
    metadata = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": {**identity, "poses": len(original), "edges": len(edges)},
                "environment": environment(), "rounds": args.rounds,
                "variants": variants,
                "source_fingerprint": source_fingerprint(ROOT),
                "executables": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (("ropt", args.ropt), ("gtsam", args.gtsam))},
                "gtsam_build": build, "max_iterations": args.max_iterations,
                "ropt_cost_tolerance": args.cost_tolerance,
                "gtsam_cost_tolerances": {"absolute": 1e-12, "relative": 1e-12},
                "consistency_reference": "separate native exact-Jacobian multifrontal QR solve; excluded from samples; no dense/global-optimality reference",
                "warmup": "one untimed consistency solve per ropt process, one discarded solve per native process",
                "memory_measured": False, "allocation_instrumentation": False,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in ("benchmarks/scalable_pose_graph.py", "benchmarks/pose_graph.py", "docs/validation/check_pose_graph.py")},
                "quality_thresholds": {"relative_cost": 1e-9, "gradient_inf": 1e-5, "pose_delta": 1e-6}}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

    def native(directory, solver, warmup):
        subprocess.run([str(args.gtsam.resolve()), str(args.input.resolve()), str(directory),
                        "1", str(warmup), solver, str(args.max_iterations)],
                       check=True, stdout=subprocess.DEVNULL)
        diagnostic = json.loads((directory/"diagnostics.json").read_text())
        require(diagnostic["exact_between"] and diagnostic["jacobian_max_difference"] < 1e-7
                and diagnostic["linear_solver"] == solver, "native derivative/backend diagnostic differs")
        return load_samples(directory/"samples.csv"), native_poses(directory/"poses.csv")

    reference_rows, reference = native(out/"qr-reference", "qr", 0)
    reference_quality = quality(reference, edges, original, reference)
    reference_quality["iteration_limit_not_reached"] = int(reference_rows[0]["optimizer_iterations"]) < args.max_iterations
    initial_cost = objective(original, edges)
    samples, checks = [], {name: [] for name in variants}
    for round_index in range(args.rounds):
        offset = round_index % len(variants)
        rotated = variants[offset:]+variants[:offset]
        order = rotated if (round_index//len(variants)) % 2 == 0 else tuple(reversed(rotated))
        for position, variant in enumerate(order):
            directory = out/f"{round_index}-{variant}"
            if variant == "ropt_block_analytic":
                subprocess.run([str(args.ropt.resolve()), "--input", str(args.input.resolve()),
                                "--output", str(directory), "--backends", "sparse_ldl_block",
                                "--pose-jacobians", "analytic", "--repeats", "1", "--warmup", "0",
                                "--cost-tolerance", str(args.cost_tolerance),
                                "--max-iterations", str(args.max_iterations)],
                               check=True, stdout=subprocess.DEVNULL)
                rows = load_samples(directory/"samples.csv")
                poses = verify_export(directory/"sparse_ldl_block.g2o", original, edges, fixes)
            else:
                rows, poses = native(directory, "qr" if variant == "gtsam_qr" else "cholesky", 1)
            require(len(rows) == 1, "missing sample")
            row = rows[0]
            check = quality(poses, edges, original, reference)
            check["checks"]["termination"] = (row.get("termination") not in ("NoProgress", "MaxIterations")
                    if variant.startswith("ropt") else int(row["optimizer_iterations"]) < args.max_iterations)
            check["checks"]["reported_objectives"] = (abs(float(row["initial_cost"])-initial_cost) <= 1e-9
                    and abs(float(row["final_cost"])-check.get("independent_cost", math.inf)) <= 1e-9)
            check["eligible"] = all(check["checks"].values())
            checks[variant].append(check)
            samples.append({"variant": variant, "round": round_index, "position": position,
                            **{key: row.get(key, "") for key in ("elapsed_ns", "initial_cost", "final_cost",
                               "termination", "nonlinear_trials", "optimizer_iterations")}})
            print(f"round {round_index} {variant}: eligible={check['eligible']}", flush=True)
    cases = {name: {"timing": timing_summary([r for r in samples if r["variant"] == name]),
                    "quality": checks[name]} for name in variants}
    passed = (reference_quality["eligible"] and reference_quality["iteration_limit_not_reached"]
              and all(c["eligible"] for records in checks.values() for c in records))
    with (out/"samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report = {"metadata": metadata, "reference_quality": reference_quality, "cases": cases,
              "all_quality_gates_passed": passed,
              "scope": "configured complete solve calls; native QR consistency instead of dense; no general ranking; failed cases cannot support speed claims"}
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for name, case in cases.items():
        print(f"{name}: median {case['timing']['median_ms']:.4f} ms")
    print(f"all quality gates passed: {passed}")


if __name__ == "__main__":
    main()
