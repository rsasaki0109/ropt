"""Audit complete-stream timings for factor-cache append reservation."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from paired_pcg import source_fingerprint
from pose_graph import ROOT, require, timing_summary
from paired_isam2 import agreement
from paired_stationarity_se2 import reference
from real_se2_stream import load, column_normalized_gradient
from se2_stream import energy_gradient


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/cache_reserve_timing")
    parser.add_argument("--current-source", action="store_true")
    parser.add_argument("--restored-production", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    require(set(manifest) == {str(p.relative_to(root)) for p in root.rglob("*")
                             if p.is_file() and p.name != "artifacts_sha256.json"}, "file layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact changed: " + name)
    require(source_fingerprint(root/"candidate_source") ==
            "939279f7aa51cd59af04323230a1433620751c75df1625fbf3c62d26243bc1de",
            "archived candidate source differs")
    if args.restored_production:
        require(source_fingerprint(ROOT) ==
                "10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3",
                "restored production source differs")
    timed_updates = 0
    for campaign in ("primary", "confirmation"):
        directory = root / campaign
        metadata = json.loads((directory / "metadata.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        require(report["eligible"] and report["before_after_means_byte_identical"] and
                report["before_after_non_timing_fields_identical"], "ineligible accuracy")
        require(metadata["rounds"] == 6 and metadata["variants"] == ["before", "after"] and
                metadata["ad_chunk_size"] == 6 and metadata["ad_edge_stride"] == 2 and
                metadata["factors"] == 1172 and metadata["ad_factors"] == 586, "workload differs")
        require(metadata["before_source_sha256"] == "10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3"
                and metadata["source_sha256"] == "939279f7aa51cd59af04323230a1433620751c75df1625fbf3c62d26243bc1de",
                "kernel fingerprints differ")
        require(metadata["selected_cap"] == 0, "requested policy differs")
        require(metadata["warmup_streams_per_process"] == metadata["timed_streams_per_process"] == 1
                and metadata["cpu_affinity"] == [0] and metadata["dataset"]["complete_dataset"],
                "complete-stream protocol differs")
        require(all(metadata["before_driver_sha256"][name] == metadata["files_sha256"][name]
                    for name in metadata["before_driver_sha256"]), "comparison drivers differ")
        require(metadata["before_executable_sha256"] != metadata["executable_sha256"], "identical binaries")
        require(not metadata["native_timing_measured"] and not metadata["allocator_counting"], "scope differs")
        require(hashlib.sha256((directory / "input.stream").read_bytes()).hexdigest() ==
                metadata["dataset"]["stream_sha256"], "input changed")
        if args.current_source:
            require(source_fingerprint(ROOT) == metadata["source_sha256"], "current kernel differs")
            for name, value in metadata["files_sha256"].items():
                require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "driver changed: " + name)
            for name, value in metadata["native_reference_sha256"].items():
                require(hashlib.sha256((ROOT / metadata["native_reference_directory"] / name).read_bytes()).hexdigest()
                        == value, "native reference changed")
        samples, updates, quality = (rows(directory / name) for name in ("samples.csv", "updates.csv", "quality.csv"))
        require(len(samples) == 12 and len(updates) == len(quality) == 12528, "incomplete stream records")
        require(report["timed_updates"] == len(updates) and not report["native_timing_measured"],
                "report scope differs")
        timed_updates += len(updates)
        for index in range(6):
            work = {}
            for variant in metadata["variants"]:
                sample = [r for r in samples if r["variant"] == variant and int(r["round"]) == index]
                raw = [r for r in updates if r["variant"] == variant and int(r["round"]) == index]
                q = [r for r in quality if r["variant"] == variant and int(r["round"]) == index]
                require(len(sample) == 1 and len(raw) == len(q) == 1044, "missing paired round")
                require([int(r["frame"]) for r in raw] == [int(r["frame"]) for r in q] == list(range(1, 1045)),
                        "arrival layout differs")
                require(int(sample[0]["position"]) == ((variant == "after") ^ (index % 2)), "order unbalanced")
                require(int(sample[0]["elapsed_ns"]) == sum(int(r["elapsed_ns"]) for r in raw), "timers differ")
                for r, checked in zip(raw, q):
                    require(int(r["elapsed_ns"]) > 0 and 1 <= int(r["iterations"]) <= 200 and
                            int(r["requested_calls"]) == 0, "work bounds differ")
                    require(int(r["allocations"]) == int(r["requested_bytes"]) == 0 and
                            r["initial_damping"] == r["final_damping"] == "", "measurement mode differs")
                    cost, actual, gradient = float(checked["cost"]), float(r["cost"]), float(checked["gradient_inf"])
                    internal = float(r["gradient_norm"])
                    require(all(map(math.isfinite, (cost, actual, gradient, internal))) and
                            0 <= gradient <= 1e-7 and 0 <= internal <= 8e-8 and
                            abs(cost-actual) <= 1e-10*max(1., cost),
                            "original-information quality failed")
                work[variant] = [{k: v for k, v in r.items() if k not in ("variant", "round", "elapsed_ns")}
                                 for r in raw]
            require(work["before"] == work["after"], "non-timing work differs")
        for variant in metadata["variants"]:
            case = report["cases"][variant]
            raw = [r for r in updates if r["variant"] == variant]
            require(case["timing"] == timing_summary([r for r in samples if r["variant"] == variant]), "timing summary differs")
            require(case["max_normalized_gradient"] == max(float(r["gradient_inf"]) for r in quality
                    if r["variant"] == variant), "gradient summary differs")
            for key, field in (("forced_recoveries", "forced_recovery"), ("requested_calls", "requested_calls"),
                               ("queries", "queries"), ("gaussian_reeliminated", "gaussian_reeliminated")):
                require(case[key] == sum(int(r[field]) for r in raw), "work summary differs")
        before, after = (report["cases"][v]["timing"]["median_ms"] for v in ("before", "after"))
        pairs = {v: {int(r["round"]): int(r["elapsed_ns"]) for r in samples if r["variant"] == v}
                 for v in metadata["variants"]}
        faster = sum(pairs["after"][i] < pairs["before"][i] for i in range(6))
        case = report["cases"]["after"]
        require(case["speedup_over_before"] == before/after and case["faster_pairs_over_before"] == faster and
                case["beats_before_in_this_case"] == (before > after and faster*4 >= 6*3), "speed criterion differs")
        require(case["time_reduction_percent"] == 100*(1-1/(before/after)), "time reduction differs")
        require(len(report["comparisons"]) == 12 and all(c["max_world_pose_norm"] <= 2e-6 and
                c["max_cost_difference"] <= 1e-9 for c in report["comparisons"]) and
                {(c["round"], c["variant"], c["reference"]) for c in report["comparisons"]} ==
                {(i, v, "archived native") for i in range(6) for v in ("before", "after")},
                "native agreement coverage differs")
        require((directory / "before-options.txt").read_bytes() ==
                (directory / "after-options.txt").read_bytes(), "benchmark options differ")
        require(gzip.decompress((directory / "before-poses.csv.gz").read_bytes()) ==
                gzip.decompress((directory / "after-poses.csv.gz").read_bytes()), "complete pose bytes differ")
        frame, key, count = 1, 0, 0
        with gzip.open(directory / "after-poses.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                require((int(row["frame"]), int(row["key"])) == (frame, key), "pose layout differs")
                value = tuple(float(row[k]) for k in ("x", "y", "angle"))
                require(all(map(math.isfinite, value)) and (key != 0 or value == (0., 0., 0.)), "pose/root invalid")
                count += 1
                key += 1
                if key > frame:
                    frame, key = frame+1, 0
        require((frame, key, count) == (1045, 0, 546534), "incomplete pose archive")
        distributions = json.loads((directory / "arrival_latency.json").read_text())
        for variant in metadata["variants"]:
            values = sorted(int(row["elapsed_ns"])/1e6 for row in updates if row["variant"] == variant)
            require(distributions[variant] == {
                "arrival_samples": len(values), "median_ms": statistics.median(values),
                "p95_nearest_rank_ms": values[(len(values)*95+99)//100-1],
                "p99_nearest_rank_ms": values[(len(values)*99+99)//100-1],
                "max_ms": max(values)}, "arrival latency summary differs")
        # Recompute quality from retained poses rather than only trusting CSV summaries.
        source = ROOT/"benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o"
        require(hashlib.sha256(source.read_bytes()).hexdigest() == metadata["dataset"]["source_sha256"],
                "original CSAIL source changed")
        problem = load(source)
        root_pose, arrivals = problem
        poses = [[] for _ in arrivals]
        with gzip.open(directory/"after-poses.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                poses[int(row["frame"])-1].append(tuple(float(row[k]) for k in ("x", "y", "angle")))
        edges, checked = [], []
        saved = [r for r in quality if r["variant"] == "after" and r["round"] == "0"]
        for index, ((_, added), values, q) in enumerate(zip(arrivals, poses, saved), 1):
            require(len(values) == index+1 and values[0] == root_pose, "retained means/root differ")
            edges.extend(added)
            cost, raw_gradient, gradient = energy_gradient(values, edges)
            normalized = column_normalized_gradient(values, edges, gradient)
            require((cost, raw_gradient, normalized) == tuple(float(q[k]) for k in
                    ("cost", "raw_gradient_inf", "gradient_inf")), "recomputed pose quality differs")
            checked.append({"cost": cost, "gradient_inf": normalized})
        native, native_quality, _ = reference(ROOT/metadata["native_reference_directory"],
                                               metadata["dataset"], 1044)
        gap = agreement(poses, native, checked, native_quality)
        archived = next(c for c in report["comparisons"] if c["round"] == 0 and c["variant"] == "after")
        require(all(gap[k] == archived[k] for k in gap), "recomputed native agreement differs")
        print(f"{campaign}: all 1044 retained pose prefixes independently rechecked", flush=True)
    require(timed_updates == 25056, "timed arrival total differs")
    print(f"Verified {len(manifest)} artifact hashes, {timed_updates} timed arrivals, "
          "exact poses/work and reported case-specific speed criteria")


if __name__ == "__main__":
    main()
