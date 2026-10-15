"""Audit complete MIT policy timings, including original-information pose rechecks."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
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


def recheck_poses(path, problem, raw, recorded):
    root, arrivals = problem
    poses = [[] for _ in arrivals]
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            frame, key = int(row["frame"]), int(row["key"])
            require(1 <= frame <= len(arrivals) and key == len(poses[frame-1]), "pose layout differs")
            value = tuple(float(row[k]) for k in ("x", "y", "angle"))
            require(all(map(math.isfinite, value)), "nonfinite pose")
            poses[frame-1].append(value)
    require(sum(map(len, poses)) == 754605, "incomplete pose archive")
    edges, checked = [], []
    for index, ((_, added), values, update, saved) in enumerate(zip(arrivals, poses, raw, recorded), 1):
        require(len(values) == index+1 and values[0] == root, "incomplete means or moved root")
        edges.extend(added)
        cost, raw_gradient, gradient = energy_gradient(values, edges)
        normalized = column_normalized_gradient(values, edges, gradient)
        require(all(map(math.isfinite, (cost, raw_gradient, normalized))) and normalized <= 1e-7 and
                abs(cost-float(update["cost"])) <= 1e-10*max(1., cost), "original pose quality failed")
        require((cost, raw_gradient, normalized) == tuple(float(saved[k]) for k in
                ("cost", "raw_gradient_inf", "gradient_inf")), "recorded quality differs from poses")
        checked.append({"cost": cost, "gradient_inf": normalized})
    return poses, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/mit_recovery_caps")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    require(set(manifest) == {str(p.relative_to(root)) for p in root.rglob("*")
                             if p.is_file() and p.name != "artifacts_sha256.json"}, "file layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact changed: " + name)
    timed_updates = 0
    executable = None
    for campaign in ("primary", "confirmation"):
        directory = root / campaign
        metadata = json.loads((directory / "metadata.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        require(report["eligible"] and report["policies_agree_at_common_gate"] and
                report["same_executable"], "ineligible accuracy")
        require(metadata["rounds"] == 6 and metadata["variants"] == ["before", "after"] and
                metadata["ad_chunk_size"] == 6 and metadata["ad_edge_stride"] == 2 and
                metadata["factors"] == 1483 and metadata["ad_factors"] == 742, "workload differs")
        require(metadata["cpu_affinity"] == [0] and metadata["warmup_streams_per_process"] == 1 and
                metadata["timed_streams_per_process"] == 1 and metadata["dataset"]["complete_dataset"],
                "complete-stream protocol differs")
        require(metadata["gates"] == {
            "original_column_normalized_gradient_inf": 1e-7, "world_pose_norm_difference": 2e-6,
            "relative_cost_difference": 1e-9, "relative_reported_cost_difference": 1e-10,
            "fixed_root": "exact"}, "quality gates differ")
        require(metadata["before_source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413"
                and metadata["source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413",
                "kernel fingerprints differ")
        require(metadata["before_executable_sha256"] == metadata["executable_sha256"], "policy study executable differs")
        if executable is None:
            executable = metadata["executable_sha256"]
        require(metadata["executable_sha256"] == executable, "campaign executables differ")
        require(all(metadata["before_driver_sha256"][name] == metadata["files_sha256"][name]
                    for name in metadata["before_driver_sha256"]), "comparison drivers differ")
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
        require(len(samples) == 12 and len(updates) == len(quality) == 14724, "incomplete stream records")
        require(report["timed_updates"] == len(updates) and not report["native_timing_measured"],
                "report scope differs")
        timed_updates += len(updates)
        for index in range(6):
            for variant in metadata["variants"]:
                sample = [r for r in samples if r["variant"] == variant and int(r["round"]) == index]
                raw = [r for r in updates if r["variant"] == variant and int(r["round"]) == index]
                q = [r for r in quality if r["variant"] == variant and int(r["round"]) == index]
                require(len(sample) == 1 and len(raw) == len(q) == 1227, "missing paired round")
                require([int(r["frame"]) for r in raw] == [int(r["frame"]) for r in q] == list(range(1, 1228)),
                        "arrival layout differs")
                require(int(sample[0]["position"]) == ((variant == "after") ^ (index % 2)), "order unbalanced")
                require(int(sample[0]["elapsed_ns"]) == sum(int(r["elapsed_ns"]) for r in raw), "timers differ")
                for r, checked in zip(raw, q):
                    require(int(r["elapsed_ns"]) > 0 and 1 <= int(r["iterations"]) <= 200 and
                            int(r["requested_calls"]) <= (1 if variant == "before" else 0), "work bounds differ")
                    require(int(r["allocations"]) == int(r["requested_bytes"]) == 0 and
                            r["initial_damping"] == r["final_damping"] == "", "measurement mode differs")
                    cost, actual, gradient = float(checked["cost"]), float(r["cost"]), float(checked["gradient_inf"])
                    internal = float(r["gradient_norm"])
                    require(all(map(math.isfinite, (cost, actual, gradient, internal))) and
                            0 <= gradient <= 1e-7 and 0 <= internal <= 8e-8 and
                            abs(cost-actual) <= 1e-10*max(1., cost),
                            "original-information quality failed")
            require(metadata["caps"] == {"before": 1, "after": 0}, "policy caps differ")
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
        before_options = (directory / "before-options.txt").read_text()
        after_options = (directory / "after-options.txt").read_text()
        require("requested_cap=1;" in before_options and "requested_cap=0;" in after_options and
                before_options.replace("requested_cap=1;", "requested_cap=0;") == after_options,
                "options differ beyond requested cap")
        require(len(report["comparisons"]) == 18 and all(c["max_world_pose_norm"] <= 2e-6
                and c["max_cost_difference"] <= 1e-9 for c in report["comparisons"]), "policy/native agreement differs")
        require({(c["round"], c["variant"], c["reference"]) for c in report["comparisons"]} ==
                {(i, variant, "archived native") for i in range(6) for variant in ("before", "after")} |
                {(i, "after versus before", "same quality gate") for i in range(6)},
                "comparison round coverage differs")
        problem = load(ROOT / "benchmarks/results/real_isam2/intel-mit1228-original.g2o")
        require(metadata["dataset"]["source_sha256"] == hashlib.sha256(
            (ROOT / "benchmarks/results/real_isam2/intel-mit1228-original.g2o").read_bytes()).hexdigest(),
            "original MIT dataset changed")
        native_poses, native_quality, _ = reference(
            ROOT / metadata["native_reference_directory"], metadata["dataset"], 1227)
        checked_policies = {}
        for variant in metadata["variants"]:
            raw = [r for r in updates if r["variant"] == variant and int(r["round"]) == 0]
            recorded = [r for r in quality if r["variant"] == variant and int(r["round"]) == 0]
            poses, checked = recheck_poses(directory / f"{variant}-poses.csv.gz", problem, raw, recorded)
            compared = agreement(poses, native_poses, checked, native_quality)
            archived = next(c for c in report["comparisons"]
                            if c["round"] == 0 and c["variant"] == variant)
            require(all(compared[k] == archived[k] for k in compared), "native pose recheck differs")
            checked_policies[variant] = poses, checked
            print(f"{campaign} {variant}: all 1227 retained pose prefixes rechecked", flush=True)
        compared = agreement(checked_policies["after"][0], checked_policies["before"][0],
                             checked_policies["after"][1], checked_policies["before"][1])
        archived = next(c for c in report["comparisons"]
                        if c["round"] == 0 and c["variant"] == "after versus before")
        require(all(compared[k] == archived[k] for k in compared), "policy pose recheck differs")
        distributions = json.loads((directory / "arrival_latency.json").read_text())
        for variant in metadata["variants"]:
            values = sorted(int(row["elapsed_ns"]) / 1e6 for row in updates if row["variant"] == variant)
            require(distributions[variant] == {
                "arrival_samples": len(values), "median_ms": statistics.median(values),
                "p95_nearest_rank_ms": values[(len(values) * 95 + 99) // 100 - 1],
                "p99_nearest_rank_ms": values[(len(values) * 99 + 99) // 100 - 1],
                "max_ms": max(values)}, "arrival latency distribution differs")
    require(timed_updates == 29448, "timed arrival total differs")
    print(f"Verified {len(manifest)} hashes and {timed_updates} policy-study arrivals with unchanged quality gates")


if __name__ == "__main__":
    main()
