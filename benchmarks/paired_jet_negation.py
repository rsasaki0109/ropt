"""Balanced negation-only latency and separate allocator diagnostics."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "before-binary", "before-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4194304)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    require(args.rounds > 0 and args.rounds % 2 == 0 and args.iterations > 0 and args.iterations % 2 == 0, "invalid counts")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output outside checkout required")
    out.mkdir()
    binaries = {"before": args.before_binary.resolve(), "after": args.binary.resolve()}
    files = ("crates/ropt/examples/profile_dynamic_negation.rs", "crates/ropt/examples/support/allocation_counter.rs", "benchmarks/paired_jet_negation.py")
    metadata = {"source_sha256": source_fingerprint(ROOT), "before_source_sha256": source_fingerprint(args.before_source),
                "executable_sha256": {v: digest(p) for v, p in binaries.items()}, "files_sha256": {p: digest(ROOT / p) for p in files},
                "environment": environment(), "cpu_affinity": [args.cpu], "rounds": args.rounds, "iterations": args.iterations,
                "warmup_operations": 4, "scope": "isolated owned/borrowed negation; seeding, validation and serialization outside timing; no whole-AD/stream/native/peak-memory claim"}
    require(metadata["before_source_sha256"] == "5e03888fc601316ba2c877e98b0544c6e5daab767dad1b9540d93e29d4760b8c", "baseline differs")
    require(metadata["executable_sha256"]["before"] != metadata["executable_sha256"]["after"], "identical executables")
    for name in files:
        if name.startswith("crates/"):
            require(digest(args.before_source / name) == metadata["files_sha256"][name], "driver differs")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples, cases = [], []
    for dimension in (1, 2, 6, 16, 32, 65):
        for ownership in ("owned", "borrowed"):
            times = {v: [] for v in binaries}
            allocations = {v: [] for v in binaries}
            for mode, rounds in (("timing", args.rounds), ("allocations", 2)):
                reference = None
                iterations = args.iterations if mode == "timing" else 1
                for index in range(rounds):
                    order = ("before", "after") if index % 2 == 0 else ("after", "before")
                    for position, variant in enumerate(order):
                        process = subprocess.run(["taskset", "-c", str(args.cpu), str(binaries[variant]), str(dimension), str(iterations), mode, ownership],
                                                 env=env, check=True, capture_output=True, text=True)
                        (out / f"width{dimension}-{ownership}-{mode}-{index}-{variant}.txt").write_text(process.stdout)
                        lines = process.stdout.splitlines()
                        require(len(lines) == 4, "incomplete diagnostic")
                        row = list(csv.DictReader(io.StringIO("\n".join(lines[:2]))))[0]
                        if reference is None:
                            reference = lines[2:]
                        require(lines[2:] == reference, "before/after bits differ")
                        require(int(row["dimension"]) == dimension and int(row["iterations"]) == iterations and row["ownership"] == ownership,
                                "work changed")
                        require(float(row["seconds"]) > 0 and int(row["jet_bytes"]) == 40, "invalid timing/layout")
                        if mode == "timing":
                            require(int(row["allocations"]) == int(row["requested_bytes"]) == 0, "instrumented timing")
                            times[variant].append(float(row["seconds"]))
                        else:
                            allocations[variant].append((int(row["allocations"]), int(row["requested_bytes"])))
                        samples.append({"mode": mode, "round": index, "position": position, "variant": variant, **row})
            expected = (0, 0) if ownership == "owned" else (1, dimension * 8)
            require(all(values == [expected, expected] for values in allocations.values()), "allocation work differs")
            before, after = (statistics.median(times[v]) for v in binaries)
            faster = sum(a < b for b, a in zip(times["before"], times["after"]))
            cases.append({"dimension": dimension, "ownership": ownership, "before_seconds": before, "after_seconds": after,
                          "median_speedup": before / after, "time_reduction_percent": 100 * (1 - after / before), "faster_pairs": faster,
                          "total_pairs": args.rounds, "before_after_bits_identical": True, "allocations_per_operation": {v: allocations[v][0] for v in binaries}})
            print(f"width {dimension} {ownership}: {before / after:.3f}x; faster {faster}/{args.rounds}", flush=True)
    require(source_fingerprint(ROOT) == metadata["source_sha256"] and source_fingerprint(args.before_source) == metadata["before_source_sha256"], "source changed")
    require({v: digest(p) for v, p in binaries.items()} == metadata["executable_sha256"], "binary changed")
    require({name: digest(ROOT / name) for name in files} == metadata["files_sha256"], "driver changed")
    with (out / "samples.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (out / "summary.json").write_text(json.dumps({"eligible": True, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
