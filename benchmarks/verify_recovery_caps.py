"""Audit same-binary recovery-cap timings and independent original-information quality."""
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


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/recovery_caps")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    require(set(manifest) == {str(p.relative_to(root)) for p in root.rglob("*")
                             if p.is_file() and p.name != "artifacts_sha256.json"}, "file layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact changed: " + name)
    timed_updates = 0
    for campaign in ("primary", "confirmation", "isolated"):
        directory = root / campaign
        metadata = json.loads((directory / "metadata.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        require(report["eligible"] and report["policies_agree_at_common_gate"] and
                report["same_executable"], "ineligible accuracy")
        require(metadata["rounds"] == 6 and metadata["variants"] == ["before", "after"] and
                metadata["ad_chunk_size"] == 6 and metadata["ad_edge_stride"] == 2 and
                metadata["factors"] == 1172 and metadata["ad_factors"] == 586, "workload differs")
        require(metadata["before_source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413"
                and metadata["source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413",
                "kernel fingerprints differ")
        require(metadata["before_executable_sha256"] == metadata["executable_sha256"], "policy study executable differs")
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
                            int(r["requested_calls"]) <= (1 if variant == "before" else 0), "work bounds differ")
                    require(int(r["allocations"]) == int(r["requested_bytes"]) == 0 and
                            r["initial_damping"] == r["final_damping"] == "", "measurement mode differs")
                    cost, actual, gradient = float(checked["cost"]), float(r["cost"]), float(checked["gradient_inf"])
                    require(all(map(math.isfinite, (cost, actual, gradient))) and gradient <= 1e-7 and
                            float(r["gradient_norm"]) <= 8e-8 and abs(cost-actual) <= 1e-10*max(1., cost),
                            "original-information quality failed")
                work[variant] = [{k: v for k, v in r.items() if k not in ("variant", "round", "elapsed_ns")}
                                 for r in raw]
            require(metadata["caps"] == {"before": 1, "after": 0}, "policy caps differ")
        for variant in metadata["variants"]:
            case = report["cases"][variant]
            raw = [r for r in updates if r["variant"] == variant]
            require(case["timing"] == timing_summary([r for r in samples if r["variant"] == variant]), "timing summary differs")
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
        require(len(report["comparisons"]) == 18 and all(c["max_world_pose_norm"] <= 2e-6
                and c["max_cost_difference"] <= 1e-9 for c in report["comparisons"]), "policy/native agreement differs")
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
            values = sorted(int(row["elapsed_ns"]) / 1e6 for row in updates if row["variant"] == variant)
            require(distributions[variant] == {
                "arrival_samples": len(values), "median_ms": statistics.median(values),
                "p95_nearest_rank_ms": values[(len(values) * 95 + 99) // 100 - 1],
                "p99_nearest_rank_ms": values[(len(values) * 99 + 99) // 100 - 1],
                "max_ms": max(values)}, "arrival latency distribution differs")
    require(timed_updates == 37584, "timed arrival total differs")
    functional = root / "functional"
    m = json.loads((functional / "metadata.json").read_text())
    require(m["source_sha256"] == metadata["source_sha256"] and m["requested_cap"] == 0 and
            m["executable_sha256"] == metadata["executable_sha256"], "functional policy provenance differs")
    for name, n, recoveries, queries in (("csail", 1044, 129, 1173), ("mit", 1227, 111, 1338)):
        path = functional / name
        r = json.loads((path / "report.json").read_text())
        require(r["exit_code"] == 0 and r["quality"]["eligible_quality"] and r["native_pass"] and
                r["quality"]["arrivals"] == n and r["dataset"]["complete_dataset"], "functional policy quality failed")
        records = rows(path / "replay/updates.csv")
        require(len(records) == n and
                all(float(row["gradient_norm"]) <= 8e-8 and int(row["requested_calls"]) == 0 for row in records),
                "functional policy updates differ")
        work = {key: sum(int(row[key]) for row in records) for key in r["work"]}
        require(work == r["work"] and work["queries"] == queries and
                work["forced_recovery"] == recoveries, "functional policy work differs")
    require(timed_updates == 37584, "policy timed arrival count differs")
    print(f"Verified {len(manifest)} hashes and {timed_updates} policy-study arrivals with unchanged quality gates")


if __name__ == "__main__":
    main()
