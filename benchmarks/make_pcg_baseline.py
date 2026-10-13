"""Build an isolated benchmark-only snapshot with the PCG optimization reversed.

Never modifies the working checkout or creates a Git worktree. The saved patch
restores the allocating numerical paths, retaining unused buffered APIs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside checkout")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("baseline must be outside checkout")
    out.mkdir()
    for name in ("Cargo.toml", "Cargo.lock", "README.md"):
        shutil.copyfile(ROOT / name, out / name)
    crate = out / "crates/ropt"
    crate.mkdir(parents=True)
    shutil.copyfile(ROOT / "crates/ropt/Cargo.toml", crate / "Cargo.toml")
    shutil.copytree(ROOT / "crates/ropt/src", crate / "src")
    (crate / "examples/support").mkdir(parents=True)
    for name in ("benchmark_pose_graph.rs", "profile_pose_graph.rs", "support/allocation_counter.rs"):
        shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    patch = ROOT / "benchmarks/pcg_buffer_reuse.patch"
    subprocess.run(["patch", "--batch", "-p1", "-R", "-d", str(out), "-i", str(patch)], check=True)
    metadata = {"patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "purpose": "restored allocating hot paths; unused buffered APIs retained; benchmark-only snapshot"}
    (out / "baseline.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(["cargo", "build", "--manifest-path", str(out / "Cargo.toml"), "-p", "ropt",
                    "--example", "benchmark_pose_graph", "--example", "profile_pose_graph",
                    "--release", "--locked", "--offline"], check=True)
    print(f"Baseline executables saved under {out}/target/release/examples")


if __name__ == "__main__":
    main()
