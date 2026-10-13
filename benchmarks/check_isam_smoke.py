"""Offline functional check; no timing thresholds or external dependencies."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

from isam_stream import validate
from pose_graph import require


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="ropt-isam-smoke-") as temporary:
        root = Path(temporary)
        for mode in ("incremental", "batch"):
            out = root/mode
            command = [executable, mode, "50", str(out), "1", "0"]
            subprocess.run(command, check=True)
            require(validate(out, 50, mode == "incremental")["eligible"], "stream oracle failed")
            with (out/"samples.csv").open() as stream:
                samples = list(csv.DictReader(stream))
            require(len(samples) == 1 and int(samples[0]["updates"]) == 50, "measured stream differs")
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            failed = subprocess.run(command, capture_output=True, text=True)
            require(failed.returncode != 0 and before == {p.name: p.read_bytes() for p in out.iterdir()},
                    "existing output protection failed")
        for index, (mode, frames, repeats) in enumerate((("wrong", "50", "1"),
                ("incremental", "0", "1"), ("incremental", "1001", "1"), ("batch", "50", "0"))):
            out = root/f"invalid-{index}"
            failed = subprocess.run([executable, mode, frames, str(out), repeats, "0"],
                                    capture_output=True, text=True)
            require(failed.returncode != 0 and not out.exists(), "invalid arguments accepted")
    print("Streaming smoke passed: both backends, independent oracle, delayed factors and output protection")


if __name__ == "__main__":
    main()
