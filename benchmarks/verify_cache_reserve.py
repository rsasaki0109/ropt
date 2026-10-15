"""Audit factor-cache append reservation and original-information MIT regression."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from paired_cache_reserve_allocations import BEFORE, AFTER, result_fields
from paired_pcg import source_fingerprint
from pose_graph import ROOT, require


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT/"benchmarks/results/cache_reserve")
    p.add_argument("--current-source", action="store_true")
    args = p.parse_args()
    root = args.results
    manifest = json.loads((root/"artifacts_sha256.json").read_text())
    require(set(manifest) == {str(f.relative_to(root)) for f in root.rglob("*")
                             if f.is_file() and f.name != "artifacts_sha256.json"}, "archive layout differs")
    for name, expected in manifest.items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, "artifact changed: "+name)
    directory = root/"allocations"
    m = json.loads((directory/"metadata.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    require(m["before_source_sha256"] == BEFORE and m["source_sha256"] == AFTER and
            m["rounds"] == 2 and m["appends_per_process"] == 1 and m["warmup_empty_updates"] == 4,
            "allocation provenance differs")
    require(m["allocation_counting"] and not m["native_timing_measured"] and
            summary["eligible"] and summary["exact_results_and_work"] and summary["timers_noncomparative"],
            "diagnostic scope differs")
    samples = rows(directory/"samples.csv")
    require(len(samples) == 24 and len(summary["cases"]) == 6, "incomplete diagnostic")
    for blocks in (32, 512):
        for residuals in (1, 3, 16):
            case = f"blocks{blocks}-rows{residuals}"
            reference = None
            totals = {v: [] for v in ("before", "after")}
            for index in range(2):
                for variant in totals:
                    raw = rows(directory/f"{case}-{index}-{variant}.csv")
                    require(len(raw) == 1, "missing allocation row")
                    row = raw[0]
                    require([int(row[k]) for k in ("blocks", "rows", "appends", "callback_calls")]
                            == [blocks, residuals, 1, 1], "fixture changed")
                    bits = result_fields(row)
                    if reference is None:
                        reference = bits
                    require(bits == reference, "numeric results/work differ")
                    totals[variant].append((int(row["allocation_calls"]), int(row["requested_bytes"])))
                    sample = [s for s in samples if s["case"] == case and s["round"] == str(index)
                              and s["variant"] == variant]
                    require(len(sample) == 1 and all(sample[0][k] == v for k, v in row.items()) and
                            int(sample[0]["position"]) == ((variant == "after") ^ index), "sample differs")
            require(all(len(set(v)) == 1 for v in totals.values()), "allocation counts vary")
            before, after = (totals[v][0] for v in ("before", "after"))
            require(before[0]-after[0] == 1 and before[1]-after[1] == 16*blocks-8, "clone savings differ")
            item = next(c for c in summary["cases"] if c["case"] == case)
            require(item["allocation_totals"] == {v: list(totals[v][0]) for v in totals} and
                    item["saved_calls"] == 1 and item["saved_requested_bytes"] == 16*blocks-8, "summary differs")
    if args.current_source:
        require(source_fingerprint(ROOT) == AFTER, "current source differs")
        for name, value in m["files_sha256"].items():
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value, "driver changed")
    functional = root/"mit_functional"
    m = json.loads((functional/"metadata.json").read_text())
    report = json.loads((functional/"report.json").read_text())
    require(m["source_sha256"] == AFTER and m["before_source_sha256"] == BEFORE and
            m["dataset"]["complete_dataset"] and m["ill_conditioned_edge_160_161_uses_ad"],
            "MIT provenance differs")
    require(all(report[v]["eligible_quality"] and report[v]["arrivals"] == 1227
                for v in ("candidate", "reference")), "MIT quality failed")
    require(gzip.decompress((functional/"before/poses.csv.gz").read_bytes()) ==
            gzip.decompress((functional/"after/poses.csv.gz").read_bytes()), "MIT poses differ")
    raw = [rows(functional/v/"updates.csv") for v in ("before", "after")]
    require(all(len(v) == 1227 for v in raw) and
            [{k:v for k,v in r.items() if k != "elapsed_ns"} for r in raw[0]] ==
            [{k:v for k,v in r.items() if k != "elapsed_ns"} for r in raw[1]], "MIT work differs")
    q = rows(functional/"quality.csv")
    require(len(q) == 2454 and all(r[k] == "True" for r in q
            for k in ("root_exact", "gradient_pass", "reported_cost_pass")), "MIT independent quality failed")
    gap = m["archived_native_agreement"]
    require(gap["max_world_pose_norm"] <= 2e-6 and gap["max_cost_difference"] <= 1e-9,
            "native agreement failed")
    counts = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", (root/"rust_tests.log").read_text())
    require(sum(map(int, counts)) == 335 and "test result: ok. 2 passed; 0 failed;" in
            (root/"example_tests.log").read_text() and "Ran 23 tests" in
            (root/"python_tests.log").read_text(), "test results differ")
    stages = json.loads((root/"rollback_stages.json").read_text())
    require(len(stages["stages"]) == 48 and all(s["matches_archived_kernel"] for s in stages["stages"]),
            "rollback verification incomplete")
    print(f"Verified {len(manifest)} hashes, 24 allocation diagnostics and all 1227 MIT arrivals")


if __name__ == "__main__":
    main()
