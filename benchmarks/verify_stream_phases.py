"""Read-only audit of instrumented phase budgets and complete recorded replays."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from paired_pcg import source_fingerprint
from pose_graph import ROOT, require
from profile_stream_phases import API, DETAIL


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/stream_phases")
    p.add_argument("--current-source", action="store_true")
    args = p.parse_args(); root = args.results.resolve()
    manifest = json.loads((root / "artifacts_sha256.json").read_text())
    require(set(manifest) == {str(f.relative_to(root)) for f in root.rglob("*")
                              if f.is_file() and f.name != "artifacts_sha256.json"}, "artifact layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, "artifact differs: " + name)
    metadata = json.loads((root / "metadata.json").read_text())
    require(source_fingerprint(root / "source") == metadata["instrumented_source_sha256"] and
            hashlib.sha256((root / "source/crates/ropt/examples/support/real_selective_se2.rs").read_bytes())
            .hexdigest() == metadata["driver_sha256"] and
            hashlib.sha256((root / "recipe.py").read_bytes()).hexdigest() == metadata["script_sha256"],
            "instrumentation source differs")
    if args.current_source:
        require(source_fingerprint(ROOT) == metadata["before_source_sha256"] and
                hashlib.sha256((ROOT / "benchmarks/profile_stream_phases.py").read_bytes()).hexdigest() ==
                metadata["script_sha256"], "current production/recipe differs")
    counts = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", (root / "build.log").read_text())
    require(sum(map(int, counts)) == 331, "instrumented snapshot tests differ")
    reported, raw_records = 0, 0
    for name, n, warmup in (("csail_cold", 1044, 0), ("csail_warm", 1044, 1), ("mit_cold", 1227, 0)):
        path = root / name
        summary = json.loads((path / "summary.json").read_text())
        dataset = json.loads((path / "dataset.json").read_text())
        require(dataset["complete_dataset"] and dataset["used_arrivals"] == n and
                hashlib.sha256((path / "input.stream").read_bytes()).hexdigest() == dataset["stream_sha256"],
                "dataset differs")
        phases = [{key: int(value) for key, value in row.items()} for row in rows(path / "phases.csv")]
        details = [{key: value if key == "phase" else int(value) for key, value in row.items()}
                   for row in rows(path / "details.csv")]
        parsed = {"stream_profile": [], "stream_detail": []}
        for line in (path / "stderr.log").read_text().splitlines():
            prefix = line.split(" ", 1)[0]
            if prefix in parsed:
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                parsed[prefix].append({key: value if key == "phase" else int(value)
                                       for key, value in fields.items()})
        require(parsed["stream_profile"] == phases and parsed["stream_detail"] == details,
                "raw profile logs differ")
        require(len(phases) == n * (warmup + 1) and summary["warmup_streams"] == warmup and
                summary["arrivals"] == n and summary["exact_complete_pose_bytes"] and
                summary["exact_non_timing_work"], "profile coverage/quality differs")
        raw_records += len(phases); reported += n
        for index in range(warmup + 1):
            require([r["frame"] for r in phases if r["round"] == index] == list(range(1, n + 1)),
                    "profile arrival layout differs")
        selected = [r for r in phases if r["round"] == warmup]
        updates, quality = rows(path / "updates.csv"), rows(path / "quality.csv")
        require(len(updates) == len(quality) == n and
                all(float(r["gradient_inf"]) <= 1e-7 for r in quality), "quality records differ")
        for record, update in zip(selected, updates):
            require(record["total"] == int(update["elapsed_ns"]) and
                    sum(record[phase] for phase in API) <= record["total"], "outer/phase timer differs")
            require(record["initial_calls"] == 1 and record["probe_calls"] == int(update["queries"]) and
                    record["requested_calls"] - record["requested_errors"] == int(update["requested_calls"]) and
                    record["recovery_calls"] == int(update["forced_recovery"]), "API work differs")
        total = sum(r["total"] for r in selected)
        require(total == summary["instrumented_total_ns"] and
                total == sum(int(r["elapsed_ns"]) for r in updates), "total differs")
        remainder = total
        for phase in API:
            ns = sum(r[phase] for r in selected); remainder -= ns
            require(summary["api_phases"][phase] == {"ns": ns, "percent_of_outer": 100 * ns / total,
                    "calls": sum(r[phase + "_calls"] for r in selected),
                    "errors": sum(r[phase + "_errors"] for r in selected)}, "API summary differs")
        require(summary["other"] == {"ns": remainder, "percent_of_outer": 100 * remainder / total},
                "remainder differs")
        for api, phase in enumerate(API):
            entries = {}
            for detail in DETAIL:
                records = [r for r in details if r["round"] == warmup and r["api"] == api and r["phase"] == detail]
                if records:
                    ns = sum(r["ns"] for r in records)
                    require(ns <= summary["api_phases"][phase]["ns"] and
                            all(0 <= r["errors"] <= r["calls"] for r in records), "inclusive detail invalid")
                    entries[detail] = {"ns": ns, "percent_of_outer": 100 * ns / total,
                                      "calls": sum(r["calls"] for r in records),
                                      "errors": sum(r["errors"] for r in records)}
            require(entries == summary["inclusive_details"][phase], "detail summary differs")
        frame, key, count = 1, 0, 0
        with gzip.open(path / "poses.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                require((int(row["frame"]), int(row["key"])) == (frame, key), "pose layout differs")
                count += 1; key += 1
                if key > frame: frame, key = frame + 1, 0
        require((frame, key, count) == (n + 1, 0, n * (n + 3) // 2), "pose archive incomplete")
        if args.current_source:
            baseline = ROOT / "benchmarks/results/inline_ad_inputs"
            frozen = baseline / ("primary/after-poses.csv.gz" if n == 1044 else "mit_functional/after/poses.csv.gz")
            require(gzip.decompress(frozen.read_bytes()) == gzip.decompress((path / "poses.csv.gz").read_bytes()),
                    "frozen pose bytes differ")
            native = ROOT / ("benchmarks/results/csail_real_isam2/primary" if n == 1044
                             else "benchmarks/results/precise_real_isam2/primary")
            for file, expected in summary["native_reference_sha256"].items():
                require(hashlib.sha256((native / file).read_bytes()).hexdigest() == expected, "native reference differs")
    rollback = json.loads((root / "rollback.json").read_text())
    require(rollback["source_only"] and rollback["matches_frozen_kernel"] and
            rollback["before_source_sha256"] == metadata["before_source_sha256"] and
            hashlib.sha256((root / "instrumentation.patch").read_bytes()).hexdigest() ==
            rollback["instrumentation_patch_sha256"], "source rollback differs")
    require((reported, raw_records) == (3315, 4359), "diagnostic coverage differs")
    print(f"Verified {len(manifest)} artifact hashes, {reported} reported arrivals, {raw_records} phase records; inclusive budgets consistent")


if __name__ == "__main__":
    main()
