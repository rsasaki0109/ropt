"""Alternate the original LDL/AD, reused LDL/AD, reused LDL/analytic and native GTSAM.

All candidates use the same input/anchor/objective. One discarded solve warms each
process; only complete optimization calls are timed. Every output passes the
unchanged independent gates against a separately generated dense AD reference.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

from exact_pose_graph import dataset_identity, evaluate_quality, native_poses, load_samples
from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, objective, read, require, verify_export, timing_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "before-source", "after", "gtsam", "build-manifest", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--include-cholesky", action="store_true")
    args = parser.parse_args()
    require(args.rounds > 0, "round count must be positive")
    original, edges, fixes = read(args.input)
    require(not fixes and min(original) == 0, "this comparison uses automatic anchor 0")
    adjacent = {k: set() for k in original}
    for a, b, *_ in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    visited, stack = set(), [0]
    while stack:
        key = stack.pop()
        if key not in visited:
            visited.add(key)
            stack.extend(adjacent[key] - visited)
    require(visited == set(original), "comparison requires a connected graph")
    build = json.loads(args.build_manifest.read_text())
    require(build["exact_between"] and build["archive_sha256"] ==
            "96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2",
            "unexpected native build provenance")
    require(build["executable_sha256"] == hashlib.sha256(args.gtsam.read_bytes()).hexdigest(),
            "native executable does not match manifest")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "results must be outside checkout")
    out.mkdir()
    metadata = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": {**dataset_identity(args.input), "poses": len(original), "edges": len(edges)},
                "environment": environment(), "rounds": args.rounds,
                "source_fingerprints": {"before": source_fingerprint(args.before_source),
                                        "after": source_fingerprint(ROOT)},
                "executables": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                (("before", args.before), ("after", args.after), ("gtsam", args.gtsam))},
                "gtsam_build": build,
                "rollback_patch_sha256": hashlib.sha256((ROOT / "benchmarks/ldl_reuse.patch").read_bytes()).hexdigest(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_sha256": hashlib.sha256((ROOT / "docs/validation/check_pose_graph.py").read_bytes()).hexdigest(),
                "warmup": "one untimed consistency solve per ropt process; one discarded solve per native process",
                "dense_reference": "separate before/dense AD process; excluded from comparison samples",
                "memory_measured": False, "allocation_instrumentation": False,
                "quality_thresholds": {"relative_cost": 1e-9, "gradient_inf": 1e-5, "pose_delta": 1e-6}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reference_dir = out / "dense-reference"
    subprocess.run([str(args.before.resolve()), "--input", str(args.input.resolve()),
                    "--output", str(reference_dir), "--backends", "dense", "--repeats", "1",
                    "--warmup", "0"], check=True, stdout=subprocess.DEVNULL)
    reference = verify_export(reference_dir / "dense.g2o", original, edges, fixes)
    require(evaluate_quality(reference, edges, original, {0}, reference)["eligible"],
            "dense reference fails independent gates")
    variants = ("rebuild_ad", "reuse_ad", "reuse_analytic", "reuse_block_analytic", "gtsam_exact")
    if args.include_cholesky:
        variants += ("gtsam_cholesky",)
    samples, checks = [], {name: [] for name in variants}
    for round_index in range(args.rounds):
        position = round_index % len(variants)
        rotated = variants[position:] + variants[:position]
        order = rotated if (round_index // len(variants)) % 2 == 0 else tuple(reversed(rotated))
        for variant in order:
            directory = out / f"{round_index}-{variant}"
            if variant.startswith("gtsam_"):
                command = [str(args.gtsam.resolve()), str(args.input.resolve()), str(directory), "1", "1"]
                if variant == "gtsam_cholesky":
                    command += ["cholesky"]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                rows = load_samples(directory / "samples.csv")
                poses = native_poses(directory / "poses.csv")
                diagnostic = json.loads((directory / "diagnostics.json").read_text())
                require(diagnostic["exact_between"] and diagnostic["jacobian_max_difference"] < 1e-7,
                        "native derivative diagnostic failed")
                if variant == "gtsam_cholesky":
                    require(diagnostic["linear_solver"] == "cholesky", "native solver differs")
            else:
                executable = args.before if variant == "rebuild_ad" else args.after
                command = [str(executable.resolve()), "--input", str(args.input.resolve()),
                           "--output", str(directory), "--backends",
                           "sparse_ldl_block" if variant == "reuse_block_analytic" else "sparse_ldl",
                           "--repeats", "1", "--warmup", "0"]
                if variant in ("reuse_analytic", "reuse_block_analytic"):
                    command += ["--pose-jacobians", "analytic"]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                rows = load_samples(directory / "samples.csv")
                backend = "sparse_ldl_block" if variant == "reuse_block_analytic" else "sparse_ldl"
                poses = verify_export(directory / (backend + ".g2o"), original, edges, fixes)
            require(poses.keys() == original.keys() and len(rows) == 1, "missing poses/samples")
            quality = evaluate_quality(poses, edges, original, {0}, reference)
            checks[variant].append(quality)
            require(quality["eligible"], f"{variant} round {round_index} failed common gates")
            row = rows[0]
            require(abs(float(row["final_cost"]) - quality["independent_cost"]) <= 1e-9,
                    "reported objective differs")
            require(abs(float(row["initial_cost"]) - objective(original, edges)) <= 1e-9,
                    "initial objective differs")
            samples.append({"variant": variant, "round": round_index,
                            **{key: row.get(key, "") for key in ("elapsed_ns", "initial_cost", "final_cost",
                               "termination", "nonlinear_trials", "linear_iterations", "optimizer_iterations")}})
    cases = {}
    for variant in variants:
        rows = [row for row in samples if row["variant"] == variant]
        cases[variant] = {"timing": timing_summary(rows), "quality": checks[variant],
                          "nonlinear_trials": sorted({int(r["nonlinear_trials"]) for r in rows if r["nonlinear_trials"]}),
                          "optimizer_iterations": sorted({int(r["optimizer_iterations"]) for r in rows if r["optimizer_iterations"]})}
    with (out / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    report = {"metadata": metadata, "cases": cases, "all_quality_gates_passed": True,
              "scope": "configured complete solve calls on these inputs, not a general performance ranking"}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, case in cases.items():
        print(f"{name}: median={case['timing']['median_ms']:.4f} ms; all {args.rounds} independent gates passed")


if __name__ == "__main__":
    main()
