"""Audit dense probe timings, allocation diagnostics, profiles and regressions."""
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
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/dense_probe")
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
        require(metadata["before_source_sha256"] == "41c27190a3d4eed0dcb9a00c48886cdd60222df5f20e33c0759dc119911d64f4"
                and metadata["source_sha256"] == "af6a93f7252853127e972b3128d208ce0cb5798a59c46603565a346b6f96f1ad",
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
    require(sum(map(int, counts)) == 327 and "test result: ok. 2 passed; 0 failed;" in
            (root / "example_tests.log").read_text() and "Ran 23 tests" in
            (root / "python_tests.log").read_text(), "test counts differ")
    directory = root / "allocations"
    m = json.loads((directory / "metadata.json").read_text())
    r = json.loads((directory / "summary.json").read_text())
    require(m["source_sha256"] == metadata["source_sha256"] and m["before_source_sha256"] == metadata["before_source_sha256"], "allocation kernels differ")
    require(m["rounds"] == 2 and m["queries_per_norm"] == 1 and m["allocation_diagnostic"] and not m["native_timing_measured"], "allocation policy differs")
    require(r["eligible"] and r["exact_results_and_work"] and len(r["cases"]) == 24 and len(rows(directory / "samples.csv")) == 96, "allocation coverage incomplete")
    from paired_autodiff_probe import validate, result_fields
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
                    require(after[0] < before[0] and after[1] < before[1], "dense layout allocation reduction missing")
                    entry = next(c for c in r["cases"] if c["case"] == case and c["norm"] == norm)
                    require(entry["allocation_totals"] == {v: list(values[v][0]) for v in values} and not entry["meets_case_speed_criterion"], "allocation summary differs")
    if args.current_source:
        for name, value in m["files_sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "allocation driver changed")
    rollback=json.loads((root/"rollback_stages.json").read_text())
    require(len(rollback["stages"])==43 and rollback["source_only"] and rollback["older_builds_skipped"] and all(s["matches_archived_kernel"] for s in rollback["stages"]), "rollback records differ")
    profile = root / "profiling"
    pm = json.loads((profile / "source/profile_metadata.json").read_text())
    ps = json.loads((profile / "summary.json").read_text())
    require(pm == ps["metadata"] and pm["before_source_sha256"] == metadata["before_source_sha256"]
            and pm["sample_every"] == 17 and source_fingerprint(profile / "source") ==
            pm["profile_source_sha256"], "profile source/provenance differs")
    raw = []
    for line in (profile / "samples.log").read_text().splitlines():
        require(line.startswith("probe_profile "), "unexpected profile record")
        fields = dict(field.split("=", 1) for field in line.split()[1:])
        require(fields["norm"] == "ColumnNormalized", "profile norm differs")
        raw.append({k: int(v) for k, v in fields.items() if k != "norm"})
    large = [row for row in raw if row["factors"] >= 512]
    require(len(raw) == ps["records"] == 1277 and len(large) == ps["large_records"] == 730,
            "profile coverage differs")
    totals = {key: 0. for key in ps["weighted_estimated_phase_percent"]}
    for row in large:
        require(row["sampled"] == (row["factors"] + 16) // 17, "profile sampling differs")
        for key in totals:
            totals[key] += row[key] * (1. if key in ("layout", "finish") else
                                      row["factors"] / row["sampled"])
    total = sum(totals.values())
    require(all(math.isclose(100 * value / total, ps["weighted_estimated_phase_percent"][key],
                            rel_tol=1e-12) for key, value in totals.items()), "profile aggregate differs")
    pa = json.loads((profile / "agreement.json").read_text())
    require(pa["exact_complete_pose_bytes"] and pa["exact_non_timing_update_fields"],
            "instrumented profile changed results")
    print(f"Verified {len(manifest)} artifact hashes, {timed_updates} timed arrivals, "
          "exact poses/work and reported case-specific speed criteria")


if __name__ == "__main__":
    main()
