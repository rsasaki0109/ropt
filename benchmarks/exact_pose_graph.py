"""Native exact-Jacobian reference comparison with independent quality checks.

Accepts the pinned public sample or a checksum-validated convert_toro prefix.
Timing covers optimization calls in both executables, without Python callbacks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

from pose_graph import (ROOT, SHA256, SOURCE, environment, gradient_inf,
                        objective, pose_difference, read, repository_state,
                        require, timing_summary, verify_export)
from convert_toro import SHA256 as TORO_SHA256


def dataset_identity(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == SHA256:
        return {"source": SOURCE, "sha256": digest, "derived": False}
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    provenance = json.loads(sidecar.read_text())
    require(provenance["source_sha256"] == TORO_SHA256 and provenance["output_sha256"] == digest,
            "derived dataset provenance/checksum differs")
    return {"sha256": digest, "derived": True, "provenance": provenance}


def evaluate_quality(poses, edges, original, anchors, reference):
    cost = objective(poses, edges)
    gradient = gradient_inf(poses, edges, anchors)
    delta = pose_difference(poses, reference)
    reference_cost = objective(reference, edges)
    checks = {"anchor": all(poses[k] == original[k] for k in anchors),
              "cost": abs(cost - reference_cost) <= 1e-9 * max(1, abs(reference_cost)),
              "gradient": gradient <= 1e-5, "pose_agreement": delta <= 1e-6}
    return {"eligible": all(checks.values()), "checks": checks,
            "independent_cost": cost, "stored_gradient_inf": gradient,
            "max_pose_difference_to_reference": delta}


def native_poses(path):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    poses = {int(r["id"]): (float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows}
    require(len(poses) == len(rows), "duplicate native output ID")
    return poses


def load_samples(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "ropt", "gtsam", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backends", default="dense,scalar_cg,block_cg")
    parser.add_argument("--pose-jacobians", choices=("autodiff", "analytic"), default="autodiff")
    parser.add_argument("--cg-relative-tolerance", type=float, default=1e-10)
    parser.add_argument("--cg-absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    require(args.repeats > 0 and args.warmup >= 0, "invalid repetition count")
    names = args.backends.split(',')
    require(len(set(names)) == len(names) and all(n in ("dense", "qr", "scalar_cg", "block_cg", "sparse_ldl", "sparse_ldl_natural", "sparse_ldl_block") for n in names),
            "invalid backend selection")
    identity = dataset_identity(args.input)
    original, edges, fixes = read(args.input)
    # The selected public graphs are connected. Check this rather than guessing anchors.
    adjacent = {k: set() for k in original}
    for a, b, *_ in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    visited, stack = set(), [min(original)]
    while stack:
        key = stack.pop()
        if key not in visited:
            visited.add(key)
            stack.extend(adjacent[key] - visited)
    require(visited == set(original), "comparison requires a connected graph")
    anchors = fixes or {min(original)}
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "results must be outside checkout")
    build = json.loads(args.build_manifest.read_text())
    require(build["exact_between"] and build["archive_sha256"] ==
            "96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2",
            "unexpected native build provenance")
    require(build["executable_sha256"] == hashlib.sha256(args.gtsam.read_bytes()).hexdigest(),
            "native executable does not match build manifest")
    metadata = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": {**identity, "poses": len(original), "edges": len(edges),
                            "anchors": sorted(anchors), "active_tangent": 3*(len(original)-len(anchors))},
                "repository": repository_state(), "environment": environment(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_sha256": hashlib.sha256((ROOT / "docs/validation/check_pose_graph.py").read_bytes()).hexdigest(),
                "converter_sha256": hashlib.sha256((ROOT / "benchmarks/convert_toro.py").read_bytes()).hexdigest(),
                "release_executable_sha256": hashlib.sha256(args.ropt.read_bytes()).hexdigest(),
                "gtsam_build": build, "repeats": args.repeats, "warmup": args.warmup,
                "backends": names, "pose_jacobians": args.pose_jacobians, "memory_measured": False,
                "cg_relative_tolerance": args.cg_relative_tolerance,
                "cg_absolute_tolerance": args.cg_absolute_tolerance,
                "quality_thresholds": {"relative_cost": 1e-9, "gradient_inf": 1e-5, "pose_delta": 1e-6}}
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run([str(args.ropt.resolve()), "--input", str(args.input.resolve()),
                    "--output", str(out / "ropt"), "--repeats", str(args.repeats),
                    "--warmup", str(args.warmup), "--backends", args.backends,
                    "--pose-jacobians", args.pose_jacobians,
                    "--cg-relative-tolerance", str(args.cg_relative_tolerance),
                    "--cg-absolute-tolerance", str(args.cg_absolute_tolerance)], check=True)
    subprocess.run([str(args.gtsam.resolve()), str(args.input.resolve()), str(out / "gtsam"),
                    str(args.repeats), str(args.warmup)], check=True)
    solutions = {"ropt_"+n: verify_export(out / "ropt" / f"{n}.g2o", original, edges, fixes) for n in names}
    solutions["gtsam_exact"] = native_poses(out / "gtsam/poses.csv")
    require(solutions["gtsam_exact"].keys() == original.keys(), "native pose IDs differ")
    reference = solutions["ropt_dense"] if "dense" in names else solutions["ropt_"+names[0]]
    raw, cases = load_samples(out / "ropt/samples.csv"), {}
    for name, poses in solutions.items():
        rows = load_samples(out / "gtsam/samples.csv") if name == "gtsam_exact" else [
            r for r in raw if "ropt_"+r["backend"] == name]
        require(len(rows) == args.repeats and {int(r["sample"]) for r in rows} == set(range(args.repeats)),
                "missing timing samples")
        for row in rows:
            require(abs(float(row["initial_cost"]) - objective(original, edges)) < 1e-9,
                    "initial objective differs")
            require(abs(float(row["final_cost"]) - objective(poses, edges)) < 1e-9,
                    "final objective differs")
        cases[name] = {"timing": timing_summary(rows),
                       "quality": evaluate_quality(poses, edges, original, anchors, reference)}
        for count in ("nonlinear_trials", "linear_iterations", "optimizer_iterations"):
            if count in rows[0]:
                cases[name][count] = sorted({int(r[count]) for r in rows})
    diagnostic = json.loads((out / "gtsam/diagnostics.json").read_text())
    require(diagnostic["exact_between"] and diagnostic["jacobian_max_difference"] < 1e-7,
            "native Jacobian diagnostic failed")
    passed = all(c["quality"]["eligible"] for c in cases.values())
    report = {"metadata": metadata, "cases": cases, "gtsam_diagnostic": diagnostic,
              "all_quality_gates_passed": passed,
              "scope": "configured complete solve calls; different derivative, retraction, damping and linear methods; no general speed ranking"}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, case in cases.items():
        print(f"{name}: median={case['timing']['median_ms']:.4f} ms, "
              f"cost={case['quality']['independent_cost']:.12e}, "
              f"gradient={case['quality']['stored_gradient_inf']:.3e}, "
              f"eligible={case['quality']['eligible']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
