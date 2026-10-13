"""Balanced before/after incremental and persistent all-history batch comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require, timing_summary
from isam_stream import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "before-source", "after", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    require(1 <= args.frames <= 1000 and args.rounds > 0 and args.rounds % 6 == 0,
            "invalid frames or unbalanced round count")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output must be outside checkout")
    sources = ("crates/ropt/examples/benchmark_isam.rs", "crates/ropt/examples/support/isam_stream.rs")
    for name in sources:
        require((ROOT/name).read_bytes() == (args.before_source/name).read_bytes(), "stream template differs between binaries")
    out.mkdir()
    metadata = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "frames": args.frames, "rounds": args.rounds, "environment": environment(),
                "dataset": "synthetic Euclidean2 correlated Gaussian observation/odometry stream; every 25 updates adds fixed-root to existing half-history key factor; root fixed exactly",
                "derivatives": "explicit analytic constant Jacobians",
                "source_fingerprints": {"before": source_fingerprint(args.before_source), "after": source_fingerprint(ROOT)},
                "stream_source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                "executables": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("before", args.before), ("after", args.after))},
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python_oracle_sha256": hashlib.sha256((ROOT/"benchmarks/isam_stream.py").read_bytes()).hexdigest(),
                "rollback_patch_sha256": hashlib.sha256((ROOT/"benchmarks/isam_graph_reuse.patch").read_bytes()).hexdigest(),
                "scheduled_refresh_patch_sha256": hashlib.sha256((ROOT/"benchmarks/isam_scheduled_refresh.patch").read_bytes()).hexdigest(),
                "warmup": "one discarded complete stream per process; initialization outside timers",
                "timed": "sum of complete smoother.update calls including owned reports; recipe construction, oracle/validation, output and report destruction outside timers",
                "callback_counting_enabled": True, "allocation_counting_enabled": False,
                "memory_measured": False, "quality_thresholds": {"pose_delta": 1e-8, "gradient_inf": 1e-8, "relative_cost": 1e-9}}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    variants = ("before_incremental", "after_incremental", "batch")
    samples, updates, checks = [], [], {name: [] for name in variants}
    for round_index in range(args.rounds):
        offset = round_index % 3
        rotated = variants[offset:]+variants[:offset]
        order = rotated if (round_index//3) % 2 == 0 else tuple(reversed(rotated))
        for position, variant in enumerate(order):
            directory = out/f"{round_index}-{variant}"
            mode = "batch" if variant == "batch" else "incremental"
            binary = args.before if variant == "before_incremental" else args.after
            subprocess.run([str(binary.resolve()), mode, str(args.frames), str(directory), "1", "1"], check=True, stdout=subprocess.DEVNULL)
            quality = validate(directory, args.frames, mode == "incremental")
            require(quality["eligible"], f"{variant} failed stream oracle")
            checks[variant].append(quality)
            with (directory/"samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == 1 and int(rows[0]["updates"]) == args.frames, "missing measured stream")
            row = rows[0]
            require(abs(float(row["final_cost"])-quality["independent_cost"]) <= 1e-9, "reported objective differs")
            samples.append({"variant": variant, "round": round_index, "position": position, **row})
            with (directory/"updates.csv").open() as stream:
                updates.extend({"variant": variant, "round": round_index, **row} for row in csv.DictReader(stream))
            print(f"round {round_index} {variant}: eligible=True", flush=True)
    cases = {name: {"timing": timing_summary([s for s in samples if s["variant"] == name]),
                    "quality": checks[name]} for name in variants}
    for name, rows in (("samples.csv", samples), ("updates.csv", updates)):
        with (out/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report = {"metadata": metadata, "cases": cases, "all_quality_gates_passed": True,
              "scope": "configured synthetic linear all-history updates; batch is FixedLagSmoother with persistent variables/no expiry; no GTSAM/iSAM2/general ranking"}
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for name, case in cases.items():
        print(f"{name}: median total update time {case['timing']['median_ms']:.4f} ms")


if __name__ == "__main__":
    main()
