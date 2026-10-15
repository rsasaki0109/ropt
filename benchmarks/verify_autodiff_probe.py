"""Verify retained probe samples, comparisons, work counts and artifact hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

from paired_autodiff_probe import BEFORE, result_fields, validate
from paired_pcg import source_fingerprint
from pose_graph import ROOT, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/mixed_probe")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*")
                    if p.is_file() and p.name != "artifacts_sha256.json"}
    require(set(manifest) == actual_files, "manifest file layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f"artifact changed: {name}")
    total_queries = 0
    for campaign in ("primary", "confirmation", "allocations"):
        directory = root / campaign
        metadata = json.loads((directory / "metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        allocations = campaign == "allocations"
        require(metadata["before_source_sha256"] == BEFORE, "baseline changed")
        require(metadata["executable_sha256"]["before"] != metadata["executable_sha256"]["after"],
                "identical executable fingerprints")
        require(metadata["allocation_diagnostic"] == allocations, "mode changed")
        require(summary["eligible"] and summary["exact_results_and_work"] and summary["synthetic_only"],
                "ineligible campaign")
        require(len(summary["cases"]) == 16, "missing workload conditions")
        rounds, queries = metadata["rounds"], metadata["queries_per_norm"]
        require((rounds, queries) == ((2, 1) if allocations else (6, 128)), "workload count changed")
        if args.current_source:
            require(source_fingerprint(ROOT) == metadata["source_sha256"], "current kernel differs")
            for name, expected in metadata["files_sha256"].items():
                require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                        f"current driver differs: {name}")
        with (directory / "samples.csv").open() as handle:
            samples = list(csv.DictReader(handle))
        require(len(samples) == 16 * rounds * 2, "sample count differs")
        if not allocations:
            total_queries += len(samples) * queries
        for entry in summary["cases"]:
            case, norm = entry["case"], entry["norm"]
            match = re.fullmatch(r"blocks(512|2048)-ad(50|100)-chunk(1|8)", case)
            require(match is not None, "unexpected workload")
            blocks, percent, chunk = map(int, match.groups())
            times = {variant: [] for variant in ("before", "after")}
            work = {variant: [] for variant in times}
            reference = None
            for index in range(rounds):
                for variant in times:
                    rows = validate((directory / f"{case}-round{index}-{variant}.csv").read_text(),
                                    blocks, percent, chunk, queries, allocations)
                    row = next(r for r in rows if r["norm"] == norm)
                    fields = result_fields(row)
                    if reference is None:
                        reference = fields
                    require(fields == reference, "result or callback work differs")
                    aggregate = [r for r in samples if r["case"] == case and r["norm"] == norm
                                 and r["variant"] == variant and int(r["round"]) == index]
                    require(len(aggregate) == 1 and all(aggregate[0][k] == v for k, v in row.items()),
                            "raw/aggregate samples differ")
                    expected_position = (0 if variant == "before" else 1) ^ (index % 2)
                    require(int(aggregate[0]["position"]) == expected_position, "unbalanced order")
                    times[variant].append(float(row["seconds"]))
                    work[variant].append((int(row["allocation_calls"]), int(row["allocation_bytes"])))
            before, after = (statistics.median(times[v]) for v in times)
            ratios = [b / a for b, a in zip(times["before"], times["after"])]
            faster = sum(r > 1 for r in ratios)
            require(entry["before_seconds"] == before and entry["after_seconds"] == after and
                    entry["median_speedup"] == before / after and entry["paired_speedups"] == ratios and
                    entry["faster_pairs"] == faster and entry["total_pairs"] == rounds and
                    entry["time_reduction_percent"] == 100 * (1-after/before), "summary differs from timers")
            require(entry["meets_case_speed_criterion"] ==
                    (not allocations and before > after and faster >= (3 * rounds + 3) // 4),
                    "speed criterion differs")
            if allocations:
                for variant in work:
                    require(len(set(work[variant])) == 1 and
                            entry["allocation_totals"][variant] == list(work[variant][0]),
                            "allocation counts differ")
    require(total_queries == 49152, "timed query total differs")
    tests = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", (root / "rust_tests.log").read_text())
    require(sum(map(int, tests[:-1])) == 302 and int(tests[-1]) == 18, "Rust test count differs")
    require("Ran 23 tests" in (root / "python_tests.log").read_text() and
            "\nOK\n" in (root / "python_tests.log").read_text(), "Python tests failed")
    stages = json.loads((root / "rollback_stages.json").read_text())
    require(stages["source_only"] and stages["older_builds_skipped"] and len(stages["stages"]) == 38
            and all(s["matches_archived_kernel"] for s in stages["stages"]), "rollback records differ")
    arrivals = 0
    for dataset, expected in (("csail", 1044), ("mit", 1227)):
        directory = root / f"{dataset}_functional"
        report = json.loads((directory / "report.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        require(report["candidate"]["eligible_quality"] and report["candidate"]["arrivals"] == expected
                and report["agreement"]["all_prefixes_pass"] and
                report["agreement"]["max_world_pose_norm"] == 0.0 and
                report["agreement"]["max_original_cost_difference"] == 0.0, "functional regression")
        require(metadata["exact_complete_pose_bytes"] and metadata["exact_non_timing_update_fields"],
                "functional equality check missing")
        arrivals += expected
    require(arrivals == 2271, "functional arrival total differs")
    print(f"Verified {len(manifest)} artifact hashes, {total_queries} timed queries, "
          f"{arrivals} functional arrivals and 38 source-only rollback records")


if __name__ == "__main__":
    main()
