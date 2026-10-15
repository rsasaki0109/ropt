"""Diagnose full mixed-AD streams in an isolated instrumented source snapshot.

Durations are inclusive instrumented diagnostics, never comparative timings.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require
from real_se2_stream import write, load, column_normalized_gradient
from paired_stationarity_se2 import reference
from paired_isam2 import validate, agreement

API = ("initial", "requested", "recovery", "probe")
DETAIL = ("graph_build", "sparse_evaluation", "cache_solve", "batch_solver",
          "sparse_qr_workspace", "qr_rebuild", "factor_capture", "tree_update",
          "tree_build", "coordinates", "selected_relinearize")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closing(text, start):
    """Match a Rust body while skipping ordinary strings and comments."""
    depth, index = 1, start + 1
    while depth:
        require(index < len(text), "unterminated instrumentation anchor")
        if text.startswith("//", index):
            index = text.index("\n", index)
        elif text.startswith("/*", index):
            index = text.index("*/", index + 2) + 2
        elif text[index] == '"':
            index += 1
            while text[index] != '"':
                index += 2 if text[index] == "\\" else 1
            index += 1
        else:
            depth += (text[index] == "{") - (text[index] == "}")
            index += 1
    return index - 1


def instrument(source):
    module = '''use std::{cell::{Cell, RefCell}, time::Instant};
#[derive(Clone, Copy)]
pub struct Record { pub ns: u128, pub calls: u64, pub errors: u64 }
const ZERO: Record = Record { ns: 0, calls: 0, errors: 0 };
thread_local! {
    static DATA: RefCell<[[Record; 11]; 4]> = const { RefCell::new([[ZERO; 11]; 4]) };
    static API: Cell<usize> = const { Cell::new(0) };
}
pub fn reset() { DATA.with(|d| *d.borrow_mut() = [[ZERO; 11]; 4]); }
pub fn enter(api: usize) { API.with(|a| a.set(api)); }
pub fn snapshot() -> [[Record; 11]; 4] { DATA.with(|d| *d.borrow()) }
pub fn measure<T, E>(slot: usize, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let api = API.with(Cell::get);
    let start = Instant::now();
    let result = f();
    let ns = start.elapsed().as_nanos();
    DATA.with(|d| {
        let mut d = d.borrow_mut();
        let r = &mut d[api][slot];
        r.ns += ns;
        r.calls += 1;
        r.errors += u64::from(result.is_err());
    });
    result
}
'''
    (source / "crates/ropt/src/stream_profile.rs").write_text(module)
    lib = source / "crates/ropt/src/lib.rs"
    lib.write_text(lib.read_text() + "\npub mod stream_profile;\n")
    targets = (
        ("fixed_lag.rs", "fn build_graph(", 0),
        ("factor.rs", "fn evaluate_sparse(", 1),
        ("isam.rs", "fn solve(", 2),
        ("solver.rs", "fn solve_local_gaussian_model(", 3),
        ("sparse_qr.rs", "fn solve(", 4),
        ("incremental_qr.rs", "fn rebuild_with_row_order(", 5),
        ("isam/selective.rs", "fn capture(", 6),
        ("bayes_tree/editable.rs", "fn update_shared(", 7),
        ("isam/selective.rs", "fn build_tree(", 8),
        ("isam.rs", "fn anchor_coordinates(", 9),
        ("isam/selective.rs", "fn relinearize(", 10),
    )
    for file, anchor, slot in targets:
        path = source / "crates/ropt/src" / file
        text = path.read_text()
        require(text.count(anchor) == 1, "ambiguous function anchor: " + file + ":" + anchor)
        start = text.index("{", text.index(anchor))
        end = closing(text, start)
        text = (text[:start + 1] + f"\ncrate::stream_profile::measure({slot}, || {{" +
                text[start + 1:end] + "\n})\n" + text[end:])
        path.write_text(text)
    driver = source / "crates/ropt/examples/support/real_selective_se2.rs"
    text = driver.read_text()
    anchor = "            let start = Instant::now();"
    require(text.count(anchor) == 1, "driver timer anchor differs")
    text = text.replace(anchor, '''            ropt::stream_profile::reset();
            let mut profile_ns = [0_u128; 4];
            let mut profile_calls = [0_usize; 4];
            let mut profile_errors = [0_usize; 4];
            macro_rules! profile_call {
                ($slot:expr, $call:expr) => {{
                    ropt::stream_profile::enter($slot);
                    let t = Instant::now();
                    let result = $call;
                    profile_ns[$slot] += t.elapsed().as_nanos();
                    profile_calls[$slot] += 1;
                    profile_errors[$slot] += usize::from(result.is_err());
                    result
                }};
            }
            let start = Instant::now();''')
    replacements = (
        ('smoother.update(update(), policy == "full")', 0, 1),
        ('smoother.update(update(), true)', 2, 1),
        ('smoother.update(Default::default(), true)', 2, 1),
        ('smoother.relinearize_selected(&keys)', 1, 1),
        ('smoother.nonlinear_summary(GradientNorm::ColumnNormalized)', 3, 3),
    )
    for call, slot, count in replacements:
        require(text.count(call) == count, "driver API anchor differs: " + call)
        text = text.replace(call, f"profile_call!({slot}, {call})")
    anchor = "            total += elapsed;"
    require(text.count(anchor) == 1, "driver output anchor differs")
    names = ", ".join('"' + name + '"' for name in DETAIL)
    text = text.replace(anchor, '''            total += elapsed;
            eprintln!("stream_profile round={} frame={} total={} initial={} requested={} recovery={} probe={} initial_calls={} requested_calls={} recovery_calls={} probe_calls={} initial_errors={} requested_errors={} recovery_errors={} probe_errors={}",
                round, index + 1, elapsed, profile_ns[0], profile_ns[1], profile_ns[2], profile_ns[3],
                profile_calls[0], profile_calls[1], profile_calls[2], profile_calls[3],
                profile_errors[0], profile_errors[1], profile_errors[2], profile_errors[3]);
            let names = [''' + names + '''];
            for (api, records) in ropt::stream_profile::snapshot().iter().enumerate() {
                for (name, r) in names.iter().zip(records) {
                    if r.calls != 0 {
                        eprintln!("stream_detail round={} frame={} api={} phase={} ns={} calls={} errors={}", round, index + 1, api, name, r.ns, r.calls, r.errors);
                    }
                }
            }''')
    driver.write_text(text)


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def check_and_summarize(directory, baseline, dataset, g2o, native, warmup):
    poses, quality, updates, _ = validate(directory, dataset["used_arrivals"], load(g2o),
                                          column_normalized_gradient, 1e-7)
    require(gzip.decompress((baseline / "poses.csv.gz").read_bytes()) ==
            (directory / "poses.csv").read_bytes(), "profile changed complete pose bytes")
    require([{k: v for k, v in row.items() if k != "elapsed_ns"} for row in updates] ==
            [{k: v for k, v in row.items() if k != "elapsed_ns"} for row in csv_rows(baseline / "updates.csv")],
            "profile changed non-timing work")
    native_poses, native_quality, native_hashes = reference(native, dataset, len(updates))
    native_gap = agreement(poses, native_poses, quality, native_quality)
    phases, details = [], []
    for line in (directory / "stderr.log").read_text().splitlines():
        if line.startswith(("stream_profile ", "stream_detail ")):
            fields = dict(field.split("=", 1) for field in line.split()[1:])
            fields = {key: int(value) if key != "phase" else value for key, value in fields.items()}
            (phases if line.startswith("stream_profile ") else details).append(fields)
    require(len(phases) == len(updates) * (warmup + 1), "profile arrivals incomplete")
    for round_index in range(warmup + 1):
        records = [row for row in phases if row["round"] == round_index]
        require([r["frame"] for r in records] == list(range(1, len(updates) + 1)), "profile layout differs")
    records = [row for row in phases if row["round"] == warmup]
    require(all(row["total"] == int(update["elapsed_ns"]) for row, update in zip(records, updates)),
            "profile outer timers differ")
    for row, update in zip(records, updates):
        require(sum(row[name] for name in API) <= row["total"], "phase intervals overlap")
        require(row["initial_calls"] == 1 and row["probe_calls"] == int(update["queries"]) and
                row["requested_calls"] - row["requested_errors"] == int(update["requested_calls"]) and
                row["recovery_calls"] == int(update["forced_recovery"]) and
                all(row[name + "_errors"] == 0 for name in ("recovery", "probe")), "profile call counts differ")
    total = sum(r["total"] for r in records)
    summary = {"arrivals": len(records), "warmup_streams": warmup,
               "scope": "inclusive instrumented phase diagnosis; clock/bookkeeping/output affect execution; no comparative latency or speed claim",
               "exact_complete_pose_bytes": True, "exact_non_timing_work": True,
               "max_original_normalized_gradient": max(r["gradient_inf"] for r in quality),
               "native_agreement": native_gap, "native_reference_sha256": native_hashes,
               "instrumented_total_ns": total, "api_phases": {}}
    other = total
    for name in API:
        ns = sum(r[name] for r in records)
        other -= ns
        summary["api_phases"][name] = {"ns": ns, "percent_of_outer": 100 * ns / total,
            "calls": sum(r[name + "_calls"] for r in records),
            "errors": sum(r[name + "_errors"] for r in records)}
    summary["other"] = {"ns": other, "percent_of_outer": 100 * other / total}
    summary["inclusive_details"] = {}
    for api, name in enumerate(API):
        entry = {}
        for phase in DETAIL:
            selected = [r for r in details if r["round"] == warmup and r["api"] == api and r["phase"] == phase]
            if not selected:
                continue
            ns = sum(r["ns"] for r in selected)
            require(ns <= summary["api_phases"][name]["ns"], "detail exceeds API duration")
            entry[phase] = {"ns": ns, "percent_of_outer": 100 * ns / total,
                           "calls": sum(r["calls"] for r in selected),
                           "errors": sum(r["errors"] for r in selected)}
        summary["inclusive_details"][name] = entry
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("phases.csv", phases), ("details.csv", details), ("quality.csv", quality)):
        with (directory / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (directory / "poses.csv.gz").open("wb") as dest:
        with gzip.GzipFile(fileobj=dest, mode="wb", filename="", mtime=0) as handle:
            handle.write((directory / "poses.csv").read_bytes())
    (directory / "poses.csv").unlink()
    print(json.dumps({"case": directory.name, **summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output outside checkout required")
    baseline = ROOT / "benchmarks/results/inline_ad_inputs"
    m = json.loads((baseline / "primary/metadata.json").read_text())
    before = source_fingerprint(args.source)
    require(before == m["source_sha256"], "source differs from frozen reference")
    manifest = json.loads((baseline / "artifacts_sha256.json").read_text())
    for name, expected in manifest.items():
        require(digest(baseline / name) == expected, "frozen reference changed: " + name)
    out.mkdir()
    source = out / "source"; source.mkdir()
    for name in ("Cargo.toml", "Cargo.lock", "README.md"):
        shutil.copyfile(args.source / name, source / name)
    shutil.copytree(args.source / "crates", source / "crates")
    require(source_fingerprint(source) == before, "snapshot differs")
    instrument(source)
    with (out / "build.log").open("w") as log:
        for command in (["cargo", "fmt", "--all"], ["cargo", "test", "--workspace", "--locked", "--offline"],
                        ["cargo", "build", "--release", "--locked", "--offline", "--example", "benchmark_ad_selective_se2_isam"]):
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    fingerprint = source_fingerprint(source)
    binary = source / "target/release/examples/benchmark_ad_selective_se2_isam"
    metadata = {"before_source_sha256": before, "instrumented_source_sha256": fingerprint,
                "executable_sha256": digest(binary), "environment": environment(),
                "api_phases": API, "detail_phases": DETAIL,
                "driver_sha256": digest(source / "crates/ropt/examples/support/real_selective_se2.rs"),
                "script_sha256": digest(Path(__file__)), "cpu_affinity": [0],
                "scope": "instrumented diagnosis only; details overlap by containment; no native/comparative timing"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = {**os.environ, "ROPT_BENCH_AD_CHUNK": "6", "ROPT_BENCH_AD_STRIDE": "2",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    for case, dataset_name, warmup in (("csail_cold", "csail", 0), ("csail_warm", "csail", 1), ("mit_cold", "mit", 0)):
        g2o = ROOT / ("benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o" if dataset_name == "csail"
                      else "benchmarks/results/real_isam2/intel-mit1228-original.g2o")
        path = out / case; path.mkdir()
        dataset = write(g2o, path / "input.stream")
        replay = path / "replay"
        with (path / "stderr.log").open("w") as err, (path / "stdout.log").open("w") as stdout:
            subprocess.run(["taskset", "-c", "0", str(binary), str(path / "input.stream"), str(replay), "1",
                            str(warmup), "editable", "1"], env=env, stdout=stdout, stderr=err, check=True)
        for file in replay.iterdir():
            shutil.move(str(file), path / file.name)
        replay.rmdir()
        old = baseline / "mit_functional/after"
        if dataset_name == "csail":
            # Adapt the frozen paired archive into the single-replay schema.
            old = path / "reference"; old.mkdir()
            shutil.copyfile(baseline / "primary/after-poses.csv.gz", old / "poses.csv.gz")
            records = [{k: v for k, v in r.items() if k not in ("variant", "round")}
                       for r in csv_rows(baseline / "primary/updates.csv") if r["variant"] == "after" and r["round"] == "0"]
            with (old / "updates.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        native = ROOT / ("benchmarks/results/csail_real_isam2/primary" if dataset_name == "csail"
                         else "benchmarks/results/precise_real_isam2/primary")
        check_and_summarize(path, old, dataset, g2o, native, warmup)
        (path / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
        if dataset_name == "csail":
            shutil.rmtree(old)
    require(source_fingerprint(args.source) == before and source_fingerprint(source) == fingerprint and
            digest(binary) == metadata["executable_sha256"], "source/binary changed during diagnosis")


if __name__ == "__main__":
    main()
