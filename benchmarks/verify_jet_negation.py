"""Audit retained mixed-AD stream timings, quality, work and artifact hashes."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from paired_pcg import source_fingerprint
from pose_graph import ROOT, require, timing_summary


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/jet_negation")
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
        require(metadata["before_source_sha256"] == "5e03888fc601316ba2c877e98b0544c6e5daab767dad1b9540d93e29d4760b8c"
                and metadata["source_sha256"] == "81722fe58b43addfedeae12d13a94c3062c5731627ff1b62285ffbeee305415d",
                "kernel fingerprints differ")
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
                            int(r["requested_calls"]) <= 1, "work bounds differ")
                    require(int(r["allocations"]) == int(r["requested_bytes"]) == 0 and
                            r["initial_damping"] == r["final_damping"] == "", "measurement mode differs")
                    cost, actual, gradient = float(checked["cost"]), float(r["cost"]), float(checked["gradient_inf"])
                    require(all(map(math.isfinite, (cost, actual, gradient))) and gradient <= 1e-7 and
                            float(r["gradient_norm"]) <= 8e-8 and abs(cost-actual) <= 1e-10*max(1., cost),
                            "original-information quality failed")
                work[variant] = [{k: v for k, v in r.items() if k not in ("variant", "round", "elapsed_ns")}
                                 for r in raw]
            require(work["before"] == work["after"], "non-timing work differs")
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
    counts = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", (root / "rust_tests.log").read_text())
    require(sum(map(int, counts)) == 323 and "test result: ok. 2 passed; 0 failed;" in
            (root / "example_tests.log").read_text() and "Ran 23 tests" in
            (root / "python_tests.log").read_text(), "test counts differ")
    for campaign in ("negation", "negation_confirmation"):
        directory = root / campaign
        m = json.loads((directory / "metadata.json").read_text())
        r = json.loads((directory / "summary.json").read_text())
        require(m["source_sha256"] == metadata["source_sha256"] and m["before_source_sha256"] == metadata["before_source_sha256"], "negation kernels differ")
        require(m["rounds"] == 6 and m["iterations"] == 4194304 and m["warmup_operations"] == 4 and r["eligible"], "negation policy differs")
        require([(c["dimension"], c["ownership"]) for c in r["cases"]] == [(d, o) for d in (1, 2, 6, 16, 32, 65) for o in ("owned", "borrowed")], "negation coverage differs")
        require(len(rows(directory / "samples.csv")) == 192, "negation samples incomplete")
        if args.current_source:
            for name, value in m["files_sha256"].items():
                require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "negation driver changed")
        for case in r["cases"]:
            dimension, ownership = case["dimension"], case["ownership"]
            times = {v: [] for v in ("before", "after")}
            allocations = {v: [] for v in times}
            for mode, rounds in (("timing", 6), ("allocations", 2)):
                reference = None
                for index in range(rounds):
                    for variant in times:
                        output = (directory / f"width{dimension}-{ownership}-{mode}-{index}-{variant}.txt").read_text().splitlines()
                        require(len(output) == 4, "negation record incomplete")
                        if reference is None:
                            reference = output[2:]
                        require(output[2:] == reference, "negation bits changed")
                        import io
                        record = list(csv.DictReader(io.StringIO("\n".join(output[:2]))))[0]
                        expected = m["iterations"] if mode == "timing" else 1
                        require(int(record["dimension"]) == dimension and record["ownership"] == ownership and int(record["iterations"]) == expected and int(record["jet_bytes"]) == 40, "negation work/layout changed")
                        if mode == "timing":
                            require(int(record["allocations"]) == int(record["requested_bytes"]) == 0, "instrumented negation latency")
                            times[variant].append(float(record["seconds"]))
                        else:
                            allocations[variant].append((int(record["allocations"]), int(record["requested_bytes"])))
            import statistics
            before, after = (statistics.median(times[v]) for v in times)
            expected = (0, 0) if ownership == "owned" else (1, dimension * 8)
            require(all(counts == [expected, expected] for counts in allocations.values()), "negation allocation work changed")
            require(case["before_seconds"] == before and case["after_seconds"] == after and case["median_speedup"] == before / after and case["time_reduction_percent"] == 100 * (1 - after / before), "negation timing differs")
            require(case["faster_pairs"] == sum(a < b for b, a in zip(times["before"], times["after"])), "negation paired wins differ")
            require(case["allocations_per_operation"] == {v: list(expected) for v in times}, "negation allocation summary differs")
    for experiment in ("rejected_fusion", "unbounded_seeds", "guarded_seeds"):
        directory = root / experiment
        source = source_fingerprint(directory / "source")
        for campaign in ("widths", "widths_confirmation"):
            m = json.loads((directory / campaign / "metadata.json").read_text())
            r = json.loads((directory / campaign / "summary.json").read_text())
            require(m["source_sha256"] == source and r["eligible"], "experiment source/eligibility differs")
            require(hashlib.sha256((directory / "profile_dynamic_jet").read_bytes()).hexdigest() == m["executable_sha256"]["after"], "experiment binary differs")
            for case in r["dimensions"]:
                times = {v: [] for v in ("before", "after")}
                for mode, count in (("timing", 6), ("allocations", 2)):
                    reference = None
                    for index in range(count):
                        for variant in times:
                            lines = (directory / campaign / f"width{case['dimension']}-{mode}-{index}-{variant}.txt").read_text().splitlines()
                            if reference is None:
                                reference = lines[2:]
                            require(lines[2:] == reference, "experiment bits differ")
                            record = list(csv.DictReader(io.StringIO("\n".join(lines[:2]))))[0]
                            if mode == "timing": times[variant].append(float(record["seconds"]))
                before, after = (statistics.median(times[v]) for v in times)
                require(case["median_speedup"] == before / after and case["faster_pairs"] == sum(a < b for b, a in zip(times["before"], times["after"])), "experiment timing differs")
    functional = root / "ad_functional"
    r = json.loads((functional / "report.json").read_text())
    require(r["source_sha256"] == metadata["source_sha256"] and [d["width"] for d in r["dimensions"]] == [1,2,6,7,12,13,16,17,32,65], "AD functionality coverage differs")
    for entry in r["dimensions"]:
        records = {}
        for variant in ("before", "after"):
            lines = (functional / f"{entry['width']}-{variant}.txt").read_text().splitlines()
            record = list(csv.DictReader(io.StringIO("\n".join(lines[:2]))))[0]
            records[variant] = lines[2:], int(record["allocations"]), int(record["requested_bytes"])
        require(records["before"] == records["after"] and records["after"][1:] == (entry["allocation_calls"], entry["requested_bytes"]), "AD functionality/allocations differ")
    for campaign in ("primary", "confirmation"):
        directory = root / "guarded_seeds" / campaign
        r = json.loads((directory / "report.json").read_text())
        samples, updates, quality = (rows(directory / name) for name in ("samples.csv", "updates.csv", "quality.csv"))
        require(r["eligible"] and not r["cases"]["after"]["beats_before_in_this_case"] and len(updates) == len(quality) == 12528 and len(samples) == 12, "guarded stream status differs")
        for variant in ("before", "after"):
            require(r["cases"][variant]["timing"] == timing_summary([row for row in samples if row["variant"] == variant]), "guarded stream timing differs")
        before, after = (r["cases"][v]["timing"]["median_ms"] for v in ("before", "after"))
        require(r["cases"]["after"]["speedup_over_before"] == before / after and before < after, "guarded regression differs")
        for index in range(6):
            work = {v: [{k: value for k, value in row.items() if k not in ("variant", "round", "elapsed_ns")}
                        for row in updates if row["variant"] == v and int(row["round"]) == index]
                    for v in ("before", "after")}
            require(work["before"] == work["after"], "guarded work differs")
        require(all(float(row["gradient_inf"]) <= 1e-7 and math.isfinite(float(row["cost"])) for row in quality), "guarded quality failed")
        require(gzip.decompress((directory / "before-poses.csv.gz").read_bytes()) == gzip.decompress((directory / "after-poses.csv.gz").read_bytes()), "guarded poses differ")
    rollback=json.loads((root/"rollback_stages.json").read_text())
    require(len(rollback["stages"])==41 and rollback["source_only"] and rollback["older_builds_skipped"] and all(s["matches_archived_kernel"] for s in rollback["stages"]), "rollback records differ")
    print(f"Verified {len(manifest)} artifact hashes, {timed_updates} timed arrivals, "
          "exact poses/work and reported case-specific speed criteria")


if __name__ == "__main__":
    main()
