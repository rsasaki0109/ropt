"""Audit stack AD input wrappers, stream/probe timings, rejected trials and allocation counts."""
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


def audit_probes(directory, source=None, current=False):
    from paired_ad_layout_probe import validate, result_fields
    metadata = json.loads((directory / "metadata.json").read_text())
    report = json.loads((directory / "summary.json").read_text())
    require(metadata["before_source_sha256"] ==
            "a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080" and
            metadata["rounds"] == 6 and metadata["queries_per_norm"] == 128 and
            metadata["warmup_per_norm"] == 4 and not metadata["allocation_diagnostic"] and
            not metadata["native_timing_measured"], "probe policy differs")
    require(report["eligible"] and report["exact_results_and_work"] and
            len(report["cases"]) == 16, "probe coverage/quality differs")
    if source is not None:
        require(source_fingerprint(source) == metadata["source_sha256"], "probe source differs")
        for name, expected in metadata["files_sha256"].items():
            if name.startswith("crates/") or current:
                path = (source if name.startswith("crates/") else ROOT) / name
                require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "probe driver differs")
    samples = []
    for blocks in (512, 2048):
        for percent in (50, 100):
            for chunk in (1, 8):
                case = f"blocks{blocks}-ad{percent}-chunk{chunk}"
                reference = None
                times = {norm: {v: [] for v in ("before", "after")}
                         for norm in ("Absolute", "ColumnNormalized")}
                for index in range(6):
                    order = ("before", "after") if index % 2 == 0 else ("after", "before")
                    for position, variant in enumerate(order):
                        records = validate((directory / f"{case}-round{index}-{variant}.csv").read_text(),
                                           blocks, percent, chunk, 128, False)
                        bits = [result_fields(row) for row in records]
                        if reference is None:
                            reference = bits
                        require(bits == reference, "probe result/work bits differ")
                        for row in records:
                            times[row["norm"]][variant].append(float(row["seconds"]))
                            samples.append({"case": case, "variant": variant, "round": str(index),
                                            "position": str(position), **row})
                for norm, values in times.items():
                    entry = next(c for c in report["cases"] if c["case"] == case and c["norm"] == norm)
                    before, after = (statistics.median(values[v]) for v in ("before", "after"))
                    ratios = [b / a for b, a in zip(values["before"], values["after"])]
                    faster = sum(r > 1 for r in ratios)
                    require(entry["before_seconds"] == before and entry["after_seconds"] == after and
                            entry["median_speedup"] == before / after and
                            entry["time_reduction_percent"] == 100 * (1 - after / before) and
                            entry["paired_speedups"] == ratios and entry["faster_pairs"] == faster and
                            entry["total_pairs"] == 6 and entry["meets_case_speed_criterion"] ==
                            (before > after and faster >= 5), "probe summary/criterion differs")
    require(rows(directory / "samples.csv") == samples and len(samples) == 192,
            "probe raw samples differ")
    return metadata, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/inline_ad_inputs")
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
        require(metadata["before_source_sha256"] == "a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080"
                and metadata["source_sha256"] == "408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413",
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
    require(sum(map(int, counts)) == 331 and "test result: ok. 2 passed; 0 failed;" in
            (root / "example_tests.log").read_text() and "Ran 23 tests" in
            (root / "python_tests.log").read_text(), "test counts differ")
    directory = root / "allocations"
    m = json.loads((directory / "metadata.json").read_text())
    r = json.loads((directory / "summary.json").read_text())
    require(m["source_sha256"] == metadata["source_sha256"] and m["before_source_sha256"] == metadata["before_source_sha256"], "allocation kernels differ")
    require(m["rounds"] == 2 and m["queries_per_norm"] == 1 and m["allocation_diagnostic"] and not m["native_timing_measured"], "allocation policy differs")
    require(r["eligible"] and r["exact_results_and_work"] and len(r["cases"]) == 24 and len(rows(directory / "samples.csv")) == 96, "allocation coverage incomplete")
    from paired_ad_layout_probe import validate, result_fields
    for blocks in (32, 512):
        for percent in (0, 50, 100):
            for chunk in (1, 8):
                case = f"blocks{blocks}-ad{percent}-chunk{chunk}"
                reference = None
                totals = {norm: {v: [] for v in ("before", "after")} for norm in ("Absolute", "ColumnNormalized")}
                for index in range(2):
                    for variant in ("before", "after"):
                        records = validate((directory / f"{case}-round{index}-{variant}.csv").read_text(), blocks, percent, chunk, 1, True)
                        bits = [result_fields(row) for row in records]
                        if reference is None: reference = bits
                        require(bits == reference, "allocation diagnostic result/work bits differ")
                        for row in records:
                            totals[row["norm"]][variant].append((int(row["allocation_calls"]), int(row["allocation_bytes"])))
                for norm, values in totals.items():
                    require(all(len(set(counts)) == 1 for counts in values.values()), "allocation counts vary")
                    before, after = (values[v][0] for v in ("before", "after"))
                    ad = blocks * percent // 100
                    require(before[0] - after[0] == 2 * ad * math.ceil(3 / chunk) and
                            before[1] - after[1] == 128 * ad * math.ceil(3 / chunk),
                            "AD input-wrapper allocation reduction differs")
                    entry = next(c for c in r["cases"] if c["case"] == case and c["norm"] == norm)
                    require(entry["allocation_totals"] == {v: list(values[v][0]) for v in values} and not entry["meets_case_speed_criterion"], "allocation summary differs")
    if args.current_source:
        for name, value in m["files_sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "allocation driver changed")
    rollback=json.loads((root/"rollback_stages.json").read_text())
    require(len(rollback["stages"])==45 and rollback["source_only"] and rollback["older_builds_skipped"] and all(s["matches_archived_kernel"] for s in rollback["stages"]), "rollback records differ")
    reports = []
    for campaign in ("primary", "confirmation"):
        pm, pr = audit_probes(root / ("probe_" + campaign), ROOT if args.current_source else None,
                              args.current_source)
        require(pm["source_sha256"] == metadata["source_sha256"], "formal probe kernel differs")
        reports.append(pr)
    jointly_passing = sum(a["meets_case_speed_criterion"] and b["meets_case_speed_criterion"]
                          for a, b in zip(*(r["cases"] for r in reports)))
    require(jointly_passing == 12, "joint probe classification differs")
    trials = root / "trials"
    for name, passed in (("strided_write", 2), ("inline_ad_inputs", 14)):
        source = trials / name / "source"
        _, trial = audit_probes(trials / name / "primary", source, args.current_source)
        require(sum(c["meets_case_speed_criterion"] for c in trial["cases"]) == passed,
                "trial selection evidence differs")
        require("test result: ok. 49 passed; 0 failed;" in
                (trials / name / "tests.log").read_text(), "trial tests differ")
    if args.current_source:
        file = "crates/ropt/src/factor/autodiff.rs"
        require((ROOT / file).read_text().split("#[cfg(test)]")[0].strip() ==
                (trials / "inline_ad_inputs/source" / file).read_text().strip(),
                "selected trial production code differs")
    print(f"Verified {len(manifest)} artifact hashes, {timed_updates} timed arrivals, "
          f"exact poses/work; {jointly_passing}/16 probe cases meet the criterion in both campaigns")


if __name__ == "__main__":
    main()
