"""Offline integration check of the release timing executable and scalar oracle."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

from pose_graph import gradient_inf, objective, pose_difference, require, verify_export, read


def main():
    executable = str(Path(sys.argv[1]).resolve())
    fixture = Path(__file__).resolve().parents[1] / "crates/ropt/tests/data/loop_se2.g2o"
    original, edges, fixes = read(fixture)
    with tempfile.TemporaryDirectory(prefix="ropt-benchmark-smoke-") as directory:
        output = Path(directory) / "results"
        args = [executable, "--input", str(fixture), "--output", str(output),
                "--repeats", "2", "--warmup", "1"]
        subprocess.run(args, check=True)
        with (output / "samples.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 8, "expected eight measured solves, excluding warmup")
        dense = None
        for name in ("dense", "qr", "scalar_cg", "block_cg"):
            poses = verify_export(output / f"{name}.g2o", original, edges, fixes)
            dense = poses if dense is None else dense
            require(pose_difference(poses, dense) < 1e-6, "solutions disagree")
            require(all(poses[k] == original[k] for k in fixes), "fixed variable moved")
            require(gradient_inf(poses, edges, fixes) < 1e-5,
                    "solution fails independent stationarity check")
            samples = [row for row in rows if row["backend"] == name]
            require({int(row["sample"]) for row in samples} == {0, 1}, "sample IDs differ")
            for row in samples:
                require(int(row["elapsed_ns"]) > 0, "missing time")
                require(abs(float(row["final_cost"]) - objective(poses, edges)) < 1e-12,
                        "reported cost differs from scalar objective")
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        failed = subprocess.run(args, capture_output=True, text=True)
        require(failed.returncode != 0, "existing output directory was accepted")
        require(before == {p.name: p.read_bytes() for p in output.iterdir()}, "existing results changed")
        invalid = Path(directory) / "invalid"
        failed = subprocess.run([executable, "--input", str(fixture), "--output", str(invalid),
                                 "--repeats", "0"], capture_output=True, text=True)
        require(failed.returncode != 0 and not invalid.exists(), "invalid repetitions accepted")
        sparse = Path(directory) / "sparse-only"
        subprocess.run([executable, "--input", str(fixture), "--output", str(sparse),
                        "--repeats", "1", "--warmup", "0", "--backends", "scalar_cg,block_cg"], check=True)
        with (sparse / "samples.csv").open() as stream:
            sparse_rows = list(csv.DictReader(stream))
        require(len(sparse_rows) == 2 and {r["backend"] for r in sparse_rows} == {"scalar_cg", "block_cg"},
                "selected backend set differs")
        require("consistency_reference=SparseCg" in (sparse / "options.txt").read_text(),
                "sparse-only selection used a different reference")
        for selection in ("unknown", "dense,dense"):
            invalid_backend = Path(directory) / (selection.replace(',', '-'))
            failed = subprocess.run([executable, "--input", str(fixture), "--output", str(invalid_backend),
                                     "--backends", selection], capture_output=True, text=True)
            require(failed.returncode != 0 and not invalid_backend.exists(), "invalid backend list accepted")
        direct = Path(directory) / "direct"
        subprocess.run([executable, "--input", str(fixture), "--output", str(direct),
                        "--repeats", "1", "--warmup", "0", "--backends",
                        "sparse_ldl,sparse_ldl_natural,sparse_ldl_block"], check=True)
        with (direct / "samples.csv").open() as stream:
            direct_rows = list(csv.DictReader(stream))
        require(len(direct_rows) == 3 and {r["backend"] for r in direct_rows}
                == {"sparse_ldl", "sparse_ldl_natural", "sparse_ldl_block"}, "direct backend set differs")
        require("consistency_reference=SparseLdl" in (direct / "options.txt").read_text(),
                "direct-only selection used a different reference")
        for name in ("sparse_ldl", "sparse_ldl_natural", "sparse_ldl_block"):
            poses = verify_export(direct / (name + ".g2o"), original, edges, fixes)
            require(pose_difference(poses, dense) < 1e-6 and gradient_inf(poses, edges, fixes) < 1e-5,
                    "sparse direct backend failed common accuracy checks")
        analytic = Path(directory) / "analytic"
        subprocess.run([executable, "--input", str(fixture), "--output", str(analytic),
                        "--repeats", "1", "--warmup", "0", "--backends", "sparse_ldl",
                        "--pose-jacobians", "analytic", "--cost-tolerance", "1e-15",
                        "--max-iterations", "200"], check=True)
        poses = verify_export(analytic / "sparse_ldl.g2o", original, edges, fixes)
        require("pose_jacobians=Analytic" in (analytic / "options.txt").read_text(),
                "analytic derivative selection was not recorded")
        recorded = (analytic / "options.txt").read_text()
        require("cost_tolerance: 1e-15" in recorded and "max_iterations: 200" in recorded,
                "explicit stopping configuration was not recorded")
        require(pose_difference(poses, dense) < 1e-6 and gradient_inf(poses, edges, fixes) < 1e-5,
                "analytic backend failed common accuracy checks")
        bad_mode = Path(directory) / "bad-mode"
        failed = subprocess.run([executable, "--input", str(fixture), "--output", str(bad_mode),
                                 "--pose-jacobians", "unknown"], capture_output=True, text=True)
        require(failed.returncode != 0 and not bad_mode.exists(), "invalid derivative mode accepted")
        for index, (flag, value) in enumerate((("--cost-tolerance", "-1"),
                ("--cost-tolerance", "NaN"), ("--cost-tolerance", "inf"), ("--max-iterations", "0"))):
            invalid = Path(directory) / f"invalid-stopping-{index}"
            failed = subprocess.run([executable, "--input", str(fixture), "--output", str(invalid),
                                     flag, value], capture_output=True, text=True)
            require(failed.returncode != 0 and not invalid.exists(), "invalid stopping configuration accepted")
    print("Offline benchmark smoke passed: four defaults, all LDL orders, analytic mode, stationarity and output protection")


if __name__ == "__main__":
    main()
