"""Retain offline nonlinear SE2 accuracy evidence; no timing comparison."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    require(25 <= args.frames <= 300 and args.frames % 25 == 0, "invalid frame count")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "output must be outside checkout")
    binary = args.binary.resolve()
    sources = ("crates/ropt/examples/validate_isam_se2.rs", "crates/ropt/examples/support/isam_se2.rs",
               "benchmarks/nonlinear_isam.py")
    metadata = {"purpose": "accuracy only; no timings or external library comparison",
                "frames": args.frames, "environment": environment(),
                "source_sha256": source_fingerprint(ROOT),
                "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "templates_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                "options": {"threshold": 0.1, "interval": 10, "force_on_loops": True,
                            "refresh": "analytic SE2 sparse LDL block minimum degree 3",
                            "reference": "AD SE2 dense QR from independent odometry seeds",
                            "gradient_tolerance": 5e-7, "cost_tolerance": 1e-15,
                            "step_tolerance": 1e-10, "initial_damping": 1e-3, "max_iterations": 200},
                "gates": {"pose_local_norm": 2e-6, "refreshed_gradient_inf": 1e-6,
                          "central_difference_steps": [1e-6, 5e-7], "gradient_step_difference": 2e-7,
                          "relative_original_cost_difference": 1e-9,
                          "relative_reported_cost_difference": 1e-10,
                          "root": "exactly fixed at every prefix"}}
    out.mkdir()
    result = subprocess.run([str(binary), str(args.frames)], capture_output=True, text=True)
    (out/"updates.csv").write_text(result.stdout)
    (out/"summary.txt").write_text(result.stderr)
    metadata["exit_code"] = result.returncode
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    require(result.returncode == 0, f"validation failed; evidence retained at {out}")
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    require(len(rows) == args.frames, "missing update rows")
    require([int(r["frame"]) for r in rows] == list(range(1, args.frames+1)), "invalid frame sequence")
    numeric = ("nonlinear_cost", "cached_cost", "gradient_inf", "qr_entries")
    require(all(math.isfinite(float(r[k])) for r in rows for k in numeric), "non-finite metric")
    refreshed = [r for r in rows if r["reason"] != "None"]
    incremental = [r for r in rows if r["reason"] == "None"]
    checkpoints = [r for r in rows if r["batch_pose_difference"] != ""]
    require([int(r["frame"]) for r in checkpoints] == list(range(25, args.frames+1, 25)), "missing reference checkpoint")
    require(incremental and refreshed, "missing update paths")
    require(all(r["reason"] == "Some(Forced)" for r in checkpoints), "loop refresh not forced")
    require(all(float(r["gradient_inf"]) <= 1e-6 for r in refreshed), "nonstationary refresh")
    require(all(abs(float(r["nonlinear_cost"])-float(r["cached_cost"])) <= 1e-10*max(1., float(r["nonlinear_cost"])) for r in refreshed), "reported refresh cost differs")
    require(all(math.isfinite(float(r["batch_pose_difference"])) and float(r["batch_pose_difference"]) <= 2e-6 for r in checkpoints), "reference poses differ")
    require(all(math.isfinite(float(r["batch_cost_difference"])) and float(r["batch_cost_difference"]) <= 1e-9*max(1., float(r["nonlinear_cost"])) for r in checkpoints), "reference cost differs")
    report = {"eligible": True, "updates": len(rows), "reference_checkpoints": len(checkpoints),
              "update_reasons": {kind: sum(r["reason"] == kind for r in rows) for kind in sorted({r["reason"] for r in rows})},
              "max_pose_difference": max(float(r["batch_pose_difference"]) for r in checkpoints),
              "max_cost_difference": max(float(r["batch_cost_difference"]) for r in checkpoints),
              "max_refreshed_gradient": max(float(r["gradient_inf"]) for r in refreshed),
              "max_incremental_gradient": max(float(r["gradient_inf"]) for r in incremental),
              "max_cached_cost_gap": max(abs(float(r["nonlinear_cost"])-float(r["cached_cost"])) for r in rows),
              "final_qr_entries": int(rows[-1]["qr_entries"])}
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
