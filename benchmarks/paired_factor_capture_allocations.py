"""Count forced-refresh allocation traffic separately from comparative timing."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require

BEFORE = "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413"
AFTER = "8c04f9ad7802899dee48b1114765b34f3fb4a475202d5d9a5b6ebf1b2926caad"
DRIVER = "crates/ropt/examples/benchmark_factor_cache.rs"
COUNTER = "crates/ropt/examples/support/allocation_counter.rs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_fields(row):
    return {k: v for k, v in row.items() if k not in
            ("elapsed_ns", "allocation_calls", "requested_bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "before-binary", "before-source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output outside checkout required")
    out.mkdir()
    require(source_fingerprint(args.before_source) == BEFORE and source_fingerprint(ROOT) == AFTER,
            "unexpected kernels")
    binaries = {"before": args.before_binary.resolve(), "after": args.binary.resolve()}
    hashes = {name: digest(ROOT/name) for name in (DRIVER, COUNTER,
              "benchmarks/paired_factor_capture_allocations.py")}
    require(all(digest(args.before_source/name) == hashes[name] for name in (DRIVER, COUNTER)),
            "diagnostic driver differs")
    metadata = {
        "before_source_sha256": BEFORE, "source_sha256": AFTER,
        "executable_sha256": {name: digest(path) for name, path in binaries.items()},
        "files_sha256": hashes, "environment": environment(), "cpu_affinity": [0],
        "rounds": 2, "refreshes_per_process": 1, "warmup_refreshes": 4,
        "allocation_counting": True, "native_timing_measured": False,
        "scope": "forced refresh via public API; counts successful allocation/reallocation requests and requested bytes, not live or peak memory; instrumented timers are noncomparative"}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    samples, cases = [], []
    for blocks in (32, 512):
        for rows in (1, 3, 16):
            case = f"blocks{blocks}-rows{rows}"
            reference = None
            totals = {variant: [] for variant in binaries}
            for index in range(2):
                order = ("before", "after") if index == 0 else ("after", "before")
                for position, variant in enumerate(order):
                    result = subprocess.run(["taskset", "-c", "0", str(binaries[variant]),
                                             str(blocks), str(rows), "1"], env=env,
                                            text=True, capture_output=True, check=True)
                    (out/f"{case}-{index}-{variant}.csv").write_text(result.stdout)
                    require(not result.stderr, "unexpected diagnostic stderr")
                    records = list(csv.DictReader(io.StringIO(result.stdout)))
                    require(len(records) == 1, "missing diagnostic result")
                    row = records[0]
                    require([int(row[k]) for k in ("blocks", "rows", "refreshes", "callback_calls")]
                            == [blocks, rows, 1, blocks], "fixture work differs")
                    require(int(row["allocation_calls"]) > 0 and int(row["requested_bytes"]) > 0,
                            "allocation counting disabled")
                    bits = result_fields(row)
                    if reference is None:
                        reference = bits
                    require(bits == reference, "diagnostic result/work bits differ")
                    totals[variant].append((int(row["allocation_calls"]), int(row["requested_bytes"])))
                    samples.append({"case": case, "round": index, "variant": variant,
                                    "position": position, **row})
            require(all(len(set(values)) == 1 for values in totals.values()), "allocation counts vary")
            before, after = (totals[v][0] for v in ("before", "after"))
            cases.append({"case": case, "allocation_totals": {v: list(totals[v][0]) for v in totals},
                          "saved_calls": before[0]-after[0], "saved_requested_bytes": before[1]-after[1]})
            print(case, cases[-1], flush=True)
    require(source_fingerprint(ROOT) == AFTER and source_fingerprint(args.before_source) == BEFORE,
            "kernel changed during diagnostic")
    require(all(digest(ROOT/name) == value for name, value in hashes.items()), "driver changed")
    require(all(digest(path) == metadata["executable_sha256"][name] for name, path in binaries.items()),
            "binary changed")
    with (out/"samples.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (out/"summary.json").write_text(json.dumps({"eligible": True, "exact_results_and_work": True,
        "timers_noncomparative": True, "cases": cases}, indent=2)+"\n")


if __name__ == "__main__":
    main()
