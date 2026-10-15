"""Alternate isolated allocating/buffered PCG executables on the same input.

The rollback patch restores the old hot paths while leaving unused new APIs in
the baseline. Only optimization calls are timed; each process has an untimed
consistency reference. Profiling allocator instrumentation is not used here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

from exact_pose_graph import dataset_identity, evaluate_quality
from pose_graph import ROOT, environment, pose_difference, read, require, timing_summary, verify_export


def source_fingerprint(root):
    fingerprint = hashlib.sha256()
    paths = [root / "Cargo.toml", root / "Cargo.lock", root / "crates/ropt/Cargo.toml"]
    paths += sorted((root / "crates/ropt/src").rglob("*.rs"))
    paths += [root / "crates/ropt/examples/benchmark_pose_graph.rs"]
    for path in paths:
        fingerprint.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return fingerprint.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "before-source", "input", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    require(args.rounds > 0, "round count must be positive")
    identity = dataset_identity(args.input)
    original, edges, fixes = read(args.input)
    require(not fixes and min(original) == 0, "this paired workload uses automatic anchor 0")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "results must be outside checkout")
    variants = {"allocating": args.before.resolve(), "buffered": args.after.resolve()}
    metadata = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": {**identity, "poses": len(original), "edges": len(edges)},
                "environment": environment(), "rounds": args.rounds,
                "source_fingerprints": {"allocating": source_fingerprint(args.before_source),
                                        "buffered": source_fingerprint(ROOT)},
                "executables": {v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in variants.items()},
                "rollback_patch_sha256": hashlib.sha256((ROOT / "benchmarks/pcg_buffer_reuse.patch").read_bytes()).hexdigest(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cg_relative_tolerance": 1e-10, "cg_absolute_tolerance": 1e-12,
                "warmup": "one untimed consistency-reference solve in each process; no extra warmup rounds",
                "allocation_instrumentation": False}
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples, solutions = [], {}
    for round_index in range(args.rounds):
        order = ("allocating", "buffered") if round_index % 2 == 0 else ("buffered", "allocating")
        backends = "scalar_cg,block_cg" if round_index % 2 == 0 else "block_cg,scalar_cg"
        for variant in order:
            directory = out / f"{variant}-{round_index}"
            subprocess.run([str(variants[variant]), "--input", str(args.input.resolve()),
                            "--output", str(directory), "--repeats", "1", "--warmup", "0",
                            "--backends", backends, "--cg-relative-tolerance", "1e-10",
                            "--cg-absolute-tolerance", "1e-12"], check=True, stdout=subprocess.DEVNULL)
            with (directory / "samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == 2, "paired samples missing")
            for row in rows:
                samples.append({"variant": variant, "round": round_index, **row})
                poses = verify_export(directory / (row["backend"] + ".g2o"), original, edges, fixes)
                key = (variant, row["backend"])
                if key in solutions:
                    require(pose_difference(poses, solutions[key]) < 1e-10, "repeated paired solution changed")
                else:
                    solutions[key] = poses
    reference = solutions[("allocating", "scalar_cg")]
    cases = {}
    for (variant, backend), poses in solutions.items():
        rows = [r for r in samples if r["variant"] == variant and r["backend"] == backend]
        cases[variant+"_"+backend] = {"timing": timing_summary(rows),
                                   "quality": evaluate_quality(poses, edges, original, {0}, reference),
                                   "nonlinear_trials": sorted({int(r["nonlinear_trials"]) for r in rows}),
                                   "linear_iterations": sorted({int(r["linear_iterations"]) for r in rows})}
    with (out / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    passed = all(c["quality"]["eligible"] for c in cases.values())
    report = {"metadata": metadata, "cases": cases, "all_quality_gates_passed": passed}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, case in cases.items():
        print(f"{name}: median={case['timing']['median_ms']:.4f} ms, eligible={case['quality']['eligible']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
