"""Balanced before/after measurements of synthetic current-estimate AD probes.

This measures query latency only, not optimization or native/GTSAM performance.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require

DRIVER = "crates/ropt/examples/benchmark_autodiff_probe.rs"
COUNTER = "crates/ropt/examples/support/allocation_counter.rs"
BEFORE = "a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(text, blocks, percent, chunk, queries, allocations):
    rows = list(csv.DictReader(io.StringIO(text)))
    require([r["norm"] for r in rows] == ["Absolute", "ColumnNormalized"], "missing norm")
    ad = blocks * percent // 100
    for row in rows:
        require([int(row[k]) for k in ("blocks", "ad_factors", "chunk", "queries")] ==
                [blocks, ad, chunk, queries], "workload changed")
        require(math.isfinite(float(row["seconds"])) and float(row["seconds"]) > 0, "bad timer")
        require(int(row["callback_calls"]) == queries * (ad * math.ceil(3 / chunk) + blocks - ad),
                "callback/chunk count changed")
        for key in ("cost_bits", "gradient_bits", "variable_bits", "estimate_bits"):
            require(len(row[key]) == 16 and 0 <= int(row[key], 16) < 2**64, "invalid result bits")
        if allocations:
            require(int(row["allocation_calls"]) > 0 and int(row["allocation_bytes"]) > 0,
                    "allocation diagnostic disabled")
        else:
            require(int(row["allocation_calls"]) == int(row["allocation_bytes"]) == 0,
                    "instrumented timing is ineligible")
    return rows


def result_fields(row):
    return {k: v for k, v in row.items() if k not in ("seconds", "allocation_calls", "allocation_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "before-binary", "before-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--allocations", action="store_true")
    args = parser.parse_args()
    require(args.allocations and args.queries == 1 and args.rounds == 2, "allocation diagnostic requires two pairs, one query")
    require(args.rounds > 0 and args.rounds % 2 == 0, "need balanced even rounds")
    require(args.queries > 0, "invalid query count")
    require(args.cpu in os.sched_getaffinity(0), "CPU unavailable")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output must be outside checkout")
    out.mkdir()
    binaries = {"before": args.before_binary.resolve(), "after": args.binary.resolve()}
    require(all(os.access(p, os.X_OK) for p in binaries.values()), "executable missing")
    require(source_fingerprint(args.before_source) == BEFORE, "unexpected baseline source")
    files = (DRIVER, COUNTER, "benchmarks/paired_inline_ad_inputs_allocations.py")
    hashes = {name: digest(ROOT / name) for name in files}
    for name in (DRIVER, COUNTER):
        require(digest(args.before_source / name) == hashes[name], "baseline driver differs")
    executable_hashes = {name: digest(p) for name, p in binaries.items()}
    require(executable_hashes["before"] != executable_hashes["after"], "identical executables")
    fingerprint = source_fingerprint(ROOT)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               RAYON_NUM_THREADS="1")
    metadata = {"source_sha256": fingerprint, "before_source_sha256": BEFORE,
                "executable_sha256": executable_hashes, "files_sha256": hashes,
                "environment": environment(), "cpu_affinity": [args.cpu],
                "thread_environment": {k: env[k] for k in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")},
                "rounds": args.rounds, "queries_per_norm": args.queries, "warmup_per_norm": 4,
                "allocation_diagnostic": args.allocations, "native_timing_measured": False,
                "scope": "synthetic query only; initialization and independent validation excluded"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = list(itertools.product((32, 512), (0, 50, 100), (1, 8)))
    samples, summaries = [], []
    mode = "allocations" if args.allocations else "timing"
    for blocks, percent, chunk in cases:
        case = f"blocks{blocks}-ad{percent}-chunk{chunk}"
        reference = None
        values = {norm: {v: [] for v in binaries} for norm in ("Absolute", "ColumnNormalized")}
        allocation_values = {norm: {v: [] for v in binaries} for norm in values}
        for round_index in range(args.rounds):
            order = ("before", "after") if round_index % 2 == 0 else ("after", "before")
            for position, variant in enumerate(order):
                command = ["taskset", "-c", str(args.cpu), str(binaries[variant]),
                           str(blocks), str(percent), str(chunk), str(args.queries), mode]
                process = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
                (out / f"{case}-round{round_index}-{variant}.csv").write_text(process.stdout)
                (out / f"{case}-round{round_index}-{variant}.log").write_text(process.stderr)
                rows = validate(process.stdout, blocks, percent, chunk, args.queries, args.allocations)
                bits = [result_fields(row) for row in rows]
                if reference is None:
                    reference = bits
                require(bits == reference, f"before/after results or repeated work differ: {case}")
                for row in rows:
                    norm = row["norm"]
                    values[norm][variant].append(float(row["seconds"]))
                    allocation_values[norm][variant].append(
                        (int(row["allocation_calls"]), int(row["allocation_bytes"])))
                    samples.append({"case": case, "variant": variant, "round": round_index,
                                    "position": position, **row})
        for norm, times in values.items():
            ratios = [b / a for b, a in zip(times["before"], times["after"])]
            faster = sum(r > 1 for r in ratios)
            before, after = (statistics.median(times[v]) for v in ("before", "after"))
            entry = {"case": case, "norm": norm, "before_seconds": before, "after_seconds": after,
                     "median_speedup": before / after, "time_reduction_percent": 100 * (1-after/before),
                     "paired_speedups": ratios, "faster_pairs": faster, "total_pairs": args.rounds,
                     "meets_case_speed_criterion": not args.allocations and before > after and
                     faster >= math.ceil(0.75 * args.rounds)}
            if args.allocations:
                for variant in binaries:
                    require(len(set(allocation_values[norm][variant])) == 1, "allocation counts vary")
                entry["allocation_totals"] = {v: allocation_values[norm][v][0] for v in binaries}
                ad = blocks * percent // 100
                before_counts, after_counts = (allocation_values[norm][v][0] for v in ("before", "after"))
                require(before_counts[0] - after_counts[0] == 2 * ad * math.ceil(3 / chunk) and
                        before_counts[1] - after_counts[1] == 128 * ad * math.ceil(3 / chunk),
                        "AD input-wrapper allocation reduction differs")
            summaries.append(entry)
        print(f"Validated {case}: exact cost, gradients, estimates and callback counts", flush=True)
    require(source_fingerprint(ROOT) == fingerprint, "source changed during campaign")
    require(source_fingerprint(args.before_source) == BEFORE, "baseline changed during campaign")
    require({v: digest(p) for v, p in binaries.items()} == executable_hashes, "binary changed")
    require({name: digest(ROOT / name) for name in files} == hashes, "driver changed")
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (out / "summary.json").write_text(json.dumps({"eligible": True,
        "exact_results_and_work": True, "synthetic_only": True, "cases": summaries}, indent=2) + "\n")
    print(f"Saved {len(samples)} validated norm samples under {out}")


if __name__ == "__main__":
    main()
