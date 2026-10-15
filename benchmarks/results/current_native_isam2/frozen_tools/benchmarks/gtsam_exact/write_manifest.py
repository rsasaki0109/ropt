"""Capture a checked source archive, CMake build options and actual native artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ARCHIVE_SHA256 = "96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "cache", "library", "executable", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, default=Path(__file__).resolve().parent,
                        help="driver source directory (defaults to batch reference)")
    parser.add_argument("--reference-cache", type=Path, help="optional separate driver CMake cache")
    parser.add_argument("--verify-linked-library", action="store_true",
                        help="verify the Linux runtime resolves the supplied GTSAM library")
    args = parser.parse_args()
    if digest(args.archive) != ARCHIVE_SHA256:
        raise ValueError("GTSAM source archive checksum differs")
    options = {}
    for line in args.cache.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            options[key.split(":", 1)[0]] = value
    if options.get("GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR") != "ON" or options.get("CMAKE_BUILD_TYPE") != "Release":
        raise ValueError("native build must use Release and exact BetweenFactor Jacobians")
    relevant = ["CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                "GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR", "GTSAM_BUILD_WITH_MARCH_NATIVE", "GTSAM_WITH_TBB",
                "GTSAM_WITH_EIGEN_MKL", "GTSAM_USE_SYSTEM_EIGEN", "GTSAM_DEFAULT_ALLOCATOR",
                "Boost_INCLUDE_DIR", "GTSAM_BUILD_TESTS", "GTSAM_BUILD_UNSTABLE"]
    root = args.reference_directory.resolve()
    manifest = {"archive_source": "https://codeload.github.com/borglab/gtsam/tar.gz/refs/tags/4.2.1",
                "archive_sha256": ARCHIVE_SHA256, "exact_between": True,
                "library_sha256": digest(args.library), "executable_sha256": digest(args.executable),
                "cmake_cache_sha256": digest(args.cache), "options": {k: options.get(k) for k in relevant},
                "reference_sources_sha256": {n: digest(root / n) for n in ("benchmark.cpp", "CMakeLists.txt")},
                "compiler": subprocess.check_output([options["CMAKE_CXX_COMPILER"], "--version"], text=True).splitlines()[0]}
    if args.reference_cache:
        manifest["reference_cache_sha256"] = digest(args.reference_cache)
    if args.verify_linked_library:
        lines = subprocess.check_output(["ldd", str(args.executable.resolve())], text=True).splitlines()
        linked = [line.split("=>", 1)[1].strip().split()[0] for line in lines
                  if line.strip().startswith("libgtsam.so") and "=>" in line]
        if len(linked) != 1 or digest(Path(linked[0])) != digest(args.library):
            raise ValueError("runtime GTSAM library differs from supplied artifact")
        manifest["runtime_library_sha256"] = digest(Path(linked[0]))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
