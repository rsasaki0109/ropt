"""Audit contiguous Gaussian factor capture, complete-stream timings and allocation traffic."""
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
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/factor_capture")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    require(set(manifest) == {str(p.relative_to(root)) for p in root.rglob("*")
                             if p.is_file() and p.name != "artifacts_sha256.json"}, "file layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact changed: " + name)
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
        require(metadata["before_source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413"
                and metadata["source_sha256"] == "8c04f9ad7802899dee48b1114765b34f3fb4a475202d5d9a5b6ebf1b2926caad",
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
    require(timed_updates == 25056, "timed arrival total differs")
    functional = root / "mit_functional"
    r = json.loads((functional / "report.json").read_text())
    m = json.loads((functional / "metadata.json").read_text())
    require(r["candidate"]["eligible_quality"] and r["reference"]["eligible_quality"] and
            r["candidate"]["arrivals"] == 1227 and r["agreement"]["max_world_pose_norm"] == 0 and
            r["agreement"]["max_original_cost_difference"] == 0 and m["ill_conditioned_edge_160_161_uses_ad"],
            "MIT regression failed")
    require(gzip.decompress((functional / "before/poses.csv.gz").read_bytes()) ==
            gzip.decompress((functional / "after/poses.csv.gz").read_bytes()), "MIT pose bytes differ")
    require(m["source_sha256"] == metadata["source_sha256"] and
            m["before_source_sha256"] == metadata["before_source_sha256"] and
            m["executable_sha256"] == metadata["executable_sha256"] and
            m["factors"] == 1483 and m["ad_factors"] == 742 and
            m["ad_chunk"] == 6 and m["ad_stride"] == 2 and
            m["dataset"]["complete_dataset"] and
            hashlib.sha256((functional / "input.stream").read_bytes()).hexdigest() ==
            m["dataset"]["stream_sha256"], "MIT provenance differs")
    before, after = (rows(functional / v / "updates.csv") for v in ("before", "after"))
    require(len(before) == len(after) == 1227 and
            [{k: v for k, v in row.items() if k != "elapsed_ns"} for row in before] ==
            [{k: v for k, v in row.items() if k != "elapsed_ns"} for row in after],
            "MIT non-timing work differs")
    require((functional / "before/differentiation.txt").read_bytes() ==
            (functional / "after/differentiation.txt").read_bytes(), "MIT AD workload differs")
    q = rows(functional / "quality.csv")
    require(len(q) == 2454 and all(row[field] == "True" for row in q
            for field in ("root_exact", "gradient_pass", "reported_cost_pass")), "MIT quality rows fail")
    native = m["archived_native_agreement"]
    require(native["max_world_pose_norm"] <= 2e-6 and
            native["max_cost_difference"] <= 1e-9, "MIT native agreement fails")
    if args.current_source:
        for name, expected in m["before_archive_sha256"].items():
            require(hashlib.sha256((ROOT / m["before_archive"] / name).read_bytes()).hexdigest() ==
                    expected, "MIT frozen archive changed")
        for name, expected in m["native_reference_sha256"].items():
            require(hashlib.sha256((ROOT / "benchmarks/results/precise_real_isam2/primary" / name)
                    .read_bytes()).hexdigest() == expected, "MIT native archive changed")
    counts = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", (root / "rust_tests.log").read_text())
    require(sum(map(int, counts)) == 333 and "test result: ok. 2 passed; 0 failed;" in
            (root / "example_tests.log").read_text() and "Ran 23 tests" in
            (root / "python_tests.log").read_text(), "test counts differ")
    directory = root / "allocations"
    m = json.loads((directory / "metadata.json").read_text())
    r = json.loads((directory / "summary.json").read_text())
    require(m["source_sha256"] == metadata["source_sha256"] and
            m["before_source_sha256"] == metadata["before_source_sha256"], "allocation kernels differ")
    require(m["rounds"] == 2 and m["refreshes_per_process"] == 1 and m["warmup_refreshes"] == 4 and
            m["allocation_counting"] and not m["native_timing_measured"], "allocation protocol differs")
    require(r["eligible"] and r["exact_results_and_work"] and r["timers_noncomparative"] and
            len(r["cases"]) == 6 and len(rows(directory / "samples.csv")) == 24,
            "allocation coverage incomplete")
    from paired_factor_capture_allocations import result_fields
    samples = rows(directory / "samples.csv")
    for blocks in (32, 512):
        for residual_rows, per_factor_calls, per_factor_bytes in ((1, 1, 72), (3, 5, 312), (16, 8, 656)):
            case = f"blocks{blocks}-rows{residual_rows}"
            reference = None
            totals = {v: [] for v in ("before", "after")}
            for index in range(2):
                for variant in ("before", "after"):
                    raw = rows(directory / f"{case}-{index}-{variant}.csv")
                    require(len(raw) == 1, "missing allocation row")
                    row = raw[0]
                    require([int(row[k]) for k in ("blocks", "rows", "refreshes", "callback_calls")]
                            == [blocks, residual_rows, 1, blocks] and int(row["elapsed_ns"]) > 0,
                            "allocation fixture differs")
                    require(all(re.fullmatch(r"[0-9a-f]{16}", row[key]) for key in
                            ("cost_bits", "gradient_bits", "estimate_bits", "cached_cost_bits", "anchor_delta_bits")),
                            "invalid diagnostic result bits")
                    bits = result_fields(row)
                    if reference is None:
                        reference = bits
                    require(bits == reference, "allocation diagnostic result/work bits differ")
                    totals[variant].append((int(row["allocation_calls"]), int(row["requested_bytes"])))
                    saved = [s for s in samples if s["case"] == case and s["round"] == str(index)
                             and s["variant"] == variant]
                    require(len(saved) == 1 and all(saved[0][key] == value for key, value in row.items()) and
                            int(saved[0]["position"]) == ((variant == "after") ^ index), "allocation sample differs")
            require(all(len(set(values)) == 1 for values in totals.values()), "allocation counts vary")
            before, after = (totals[v][0] for v in ("before", "after"))
            require(before[0]-after[0] == blocks*per_factor_calls and
                    before[1]-after[1] == blocks*per_factor_bytes, "capture allocation reduction differs")
            entry = next(c for c in r["cases"] if c["case"] == case)
            require(entry["allocation_totals"] == {v: list(totals[v][0]) for v in totals} and
                    entry["saved_calls"] == before[0]-after[0] and
                    entry["saved_requested_bytes"] == before[1]-after[1], "allocation summary differs")
    if args.current_source:
        for name, value in m["files_sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "allocation driver changed")
    rollback=json.loads((root/"rollback_stages.json").read_text())
    require(len(rollback["stages"])==46 and rollback["source_only"] and rollback["older_builds_skipped"] and all(s["matches_archived_kernel"] for s in rollback["stages"]), "rollback records differ")
    print(f"Verified {len(manifest)} artifact hashes, {timed_updates} timed arrivals, "
          "exact poses/work and reported case-specific speed criteria")


if __name__ == "__main__":
    main()
