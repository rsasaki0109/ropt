"""Runtime-AD width diagnostic: separate balanced latency and allocation records."""
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
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    require(args.rounds > 0 and args.rounds % 2 == 0 and args.iterations > 0, "invalid counts")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output must be outside checkout")
    out.mkdir()
    binaries = {"before": args.before_binary.resolve(), "after": args.binary.resolve()}
    metadata = {"source_sha256": source_fingerprint(ROOT), "before_source_sha256": source_fingerprint(args.before_source),
                "executable_sha256": {v: digest(p) for v, p in binaries.items()}, "environment": environment(),
                "cpu_affinity": [args.cpu], "rounds": args.rounds, "iterations": args.iterations,
                "scope": "Euclidean AD adapter width diagnostic; no native/stream/peak memory claim",
                "files_sha256": {p: digest(ROOT/p) for p in ("crates/ropt/examples/profile_dynamic_jet.rs",
                    "crates/ropt/examples/support/allocation_counter.rs", "benchmarks/paired_jet_loop_widths.py")}}
    require(metadata["before_source_sha256"] == "0c171c75e64947bb8995b1b77be8d16fb7b9fb1eca3938c5c19f67906dc0c469", "baseline differs")
    require(metadata["executable_sha256"]["before"] != metadata["executable_sha256"]["after"], "identical executables")
    for p in metadata["files_sha256"]:
        if p.startswith("crates/"):
            require(digest(args.before_source/p) == metadata["files_sha256"][p], "driver differs")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    samples, summaries = [], []
    for dimension in (1, 2, 6, 7, 12, 13, 32, 65):
        reference = None
        times = {v: [] for v in binaries}
        allocations = {v: [] for v in binaries}
        # Instrumentation is never enabled in the latency campaign.
        for mode, rounds in (("timing", args.rounds), ("allocations", 2)):
            iterations = args.iterations if mode == "timing" else 1
            for index in range(rounds):
                order = ("before", "after") if index%2 == 0 else ("after", "before")
                for position, variant in enumerate(order):
                    process = subprocess.run(["taskset", "-c", str(args.cpu), str(binaries[variant]), str(dimension),
                                              str(iterations), mode], env=env, check=True, capture_output=True, text=True)
                    (out/f"width{dimension}-{mode}-{index}-{variant}.txt").write_text(process.stdout)
                    lines = process.stdout.splitlines()
                    require(len(lines) == 4, "incomplete diagnostic")
                    row = list(csv.DictReader(io.StringIO("\n".join(lines[:2]))))[0]
                    if reference is None:
                        reference = lines[2:]
                    require(lines[2:] == reference, "residual/Jacobian bits differ")
                    require(int(row["dimension"]) == dimension and int(row["iterations"]) == iterations, "work changed")
                    require(float(row["seconds"]) > 0, "invalid timing")
                    if mode == "timing":
                        require(int(row["allocations"]) == int(row["requested_bytes"]) == 0, "instrumented timing")
                        times[variant].append(float(row["seconds"]))
                    else:
                        allocations[variant].append((int(row["allocations"]), int(row["requested_bytes"])))
                    samples.append({"mode": mode, "round": index, "position": position, "variant": variant, **row})
        require(all(len(set(values)) == 1 for values in allocations.values()), "allocation counts changed")
        before, after = (statistics.median(times[v]) for v in binaries)
        faster = sum(a < b for b, a in zip(times["before"], times["after"]))
        summaries.append({"dimension": dimension, "before_seconds": before, "after_seconds": after,
                          "median_speedup": before/after, "time_reduction_percent": 100*(1-after/before),
                          "faster_pairs": faster, "total_pairs": args.rounds,
                          "allocations_per_call": {v: allocations[v][0] for v in binaries},
                          "all_result_bits_identical": True})
        print(f"width {dimension}: {before/after:.3f}x; faster {faster}/{args.rounds}; "
              f"allocations/bytes {allocations['before'][0]} -> {allocations['after'][0]}", flush=True)
    require(source_fingerprint(ROOT) == metadata["source_sha256"] and
            source_fingerprint(args.before_source) == metadata["before_source_sha256"], "source changed")
    require({v: digest(p) for v, p in binaries.items()} == metadata["executable_sha256"], "binary changed")
    require({p: digest(ROOT/p) for p in metadata["files_sha256"]} == metadata["files_sha256"], "driver changed")
    with (out/"samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    (out/"summary.json").write_text(json.dumps({"eligible": True, "dimensions": summaries}, indent=2)+"\n")


if __name__ == "__main__":
    main()
