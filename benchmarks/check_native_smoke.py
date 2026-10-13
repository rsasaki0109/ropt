"""Optional offline native-GTSAM check; does not add a C++ dependency to ropt CI."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from exact_pose_graph import native_poses
from pose_graph import gradient_inf, objective, read, require


def main():
    executable = str(Path(sys.argv[1]).resolve())
    fixture = Path(__file__).resolve().parents[1] / "crates/ropt/tests/data/loop_se2.g2o"
    original, edges, anchors = read(fixture)
    require(sys.argv[2:] in ([], ["--with-cholesky"]), "unknown smoke option")
    modes = ["qr", "cholesky"] if sys.argv[2:] else [None]
    for mode in modes:
        with tempfile.TemporaryDirectory(prefix="ropt-native-smoke-") as directory:
            out = Path(directory) / "result"
            args = [executable, str(fixture), str(out), "2", "1"]
            if mode is not None:
                args.extend([mode, "50"])
            subprocess.run(args, check=True)
            poses = native_poses(out / "poses.csv")
            require(poses.keys() == original.keys(), "native IDs differ")
            require(all(poses[k] == original[k] for k in anchors), "native FIX moved")
            require(objective(poses, edges) < 1e-12 and gradient_inf(poses, edges, anchors) < 1e-5,
                    "native solution failed independent checks")
            with (out / "samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == 2 and all(int(r["elapsed_ns"]) > 0 for r in rows), "native samples missing")
            diagnostic = json.loads((out / "diagnostics.json").read_text())
            require(diagnostic["exact_between"] and diagnostic["jacobian_max_difference"] < 1e-7,
                    "native derivative check failed")
            if mode is not None:
                require(diagnostic["linear_solver"] == mode, "native backend selection differs")
                invalid_args = args.copy()
                invalid_args[2] = str(Path(directory)/"invalid-limit")
                invalid_args[-1] = "0"
                failure = subprocess.run(invalid_args, capture_output=True, text=True)
                require(failure.returncode != 0 and not Path(invalid_args[2]).exists(),
                        "invalid native iteration limit accepted")
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            failure = subprocess.run(args, capture_output=True, text=True)
            require(failure.returncode != 0 and before == {p.name: p.read_bytes() for p in out.iterdir()},
                    "native overwrite protection failed")
    print("Native GTSAM smoke passed: exact derivatives, fixed noncontiguous IDs, scalar objective and output protection")


if __name__ == "__main__":
    main()
