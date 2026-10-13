# Exact-Jacobian native GTSAM reference

This page preserves the original scale sweep. The later
[PCG workspace comparison](../BUFFERED_PCG.md) records updated ropt timings and
an alternating comparison with the allocating implementation.

The earlier [wheel baseline](../README.md) matched the g2o residual coordinates
but failed the common accuracy gate. GTSAM 4.2.1 provides the build option
`GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON`: its native BetweenFactor multiplies
the between-operation Jacobian by the local-coordinate Jacobian. The source
[implementation](https://github.com/borglab/gtsam/blob/4.2.1/gtsam/slam/BetweenFactor.h)
and [option](https://github.com/borglab/gtsam/blob/4.2.1/cmake/HandleGeneralOptions.cmake)
describe this choice. This benchmark builds that configuration and uses the
native factor, Gaussian information, exact pose equality and LM/QR optimizer.
It has no Python factor callback or numerical derivative in the optimization.

This is an external reference executable. ropt still has its Rust-only Cargo
dependency graph; its CI does not build or link GTSAM. Native Jacobians are
checked against central differences before any timing run. The largest observed
entry difference is 4.48e-10, below the required 1e-7. Independent scalar objective
and stationarity checks then validate the actual optimized graph.

## Build

Prerequisites: CMake, Ninja, a C++17 compiler, Python 3, and Boost development
headers/libraries for serialization, filesystem, program_options, timer and
chrono. The recorded build used CMake 3.31.10, Ninja 1.13.0, GCC 14.2.0,
Boost 1.83.0 and GTSAM's bundled Eigen 3.3.7. Boost packages were verified by
APT's signed Debian repository metadata and extracted to a local `/tmp` prefix;
no system installation or ropt dependency change was needed. Build parallelism
was three jobs under the cloud environment's four-CPU quota.

The following workflow uses an existing Boost installation. Store all build
products outside the checkout. The archive checksum pins the actual source
bytes in addition to the 4.2.1 tag.

```sh
ropt_reference_root=/tmp/ropt-gtsam-reference
mkdir "$ropt_reference_root"
curl -fL https://codeload.github.com/borglab/gtsam/tar.gz/refs/tags/4.2.1 \
  -o "$ropt_reference_root/source.tar.gz"
printf '%s  %s\n' \
  96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2 \
  "$ropt_reference_root/source.tar.gz" | sha256sum -c -
tar -xzf "$ropt_reference_root/source.tar.gz" -C "$ropt_reference_root"
cmake -S "$ropt_reference_root/gtsam-4.2.1" -B "$ropt_reference_root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ropt_reference_root/prefix" \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN' -DBoost_NO_BOOST_CMAKE=ON \
  -DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_WITH_EIGEN_MKL=OFF
cmake --build "$ropt_reference_root/build" --target gtsam CppUnitLite -j 3
cmake --install "$ropt_reference_root/build"
cmake -S benchmarks/gtsam_exact -B "$ropt_reference_root/reference" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ropt_reference_root/prefix" \
  -DBoost_NO_BOOST_CMAKE=ON
cmake --build "$ropt_reference_root/reference" -j 2
python3 benchmarks/gtsam_exact/write_manifest.py \
  --archive "$ropt_reference_root/source.tar.gz" \
  --cache "$ropt_reference_root/build/CMakeCache.txt" \
  --library "$ropt_reference_root/prefix/lib/libgtsam.so.4.2.1" \
  --executable "$ropt_reference_root/reference/gtsam_exact" \
  --output "$ropt_reference_root/manifest.json"
python3 benchmarks/check_native_smoke.py "$ropt_reference_root/reference/gtsam_exact"
```

For a local Boost prefix, both configure commands additionally need
`-DBOOST_ROOT=/path/to/boost/prefix -DBoost_NO_SYSTEM_PATHS=ON` and the appropriate
`-DBOOST_LIBRARYDIR=/path/to/boost/libraries`. The GTSAM install RPATH should
include that library directory alongside `$ORIGIN`. All required runtime
libraries must be present; a development package's dangling shared-library
symlink can cause CMake to select a non-PIC static archive and fail to link.
The recorded cloud prefix was `/tmp/ropt-benchmark-sysroot/usr`, with libraries
under `lib/x86_64-linux-gnu`.

The reference refuses debug builds, a GTSAM configuration without the exact
Jacobian option, and existing output directories. The optional offline native
smoke check validates correlated measurements, noncontiguous vertex IDs, the
explicit fixed pose, independent cost/gradient, derivatives and output protection.
GTSAM's own full test suite was not run; this validates this reference workflow.

The executable also accepts `input.g2o NEW_DIRECTORY repeats warmup
[qr|cholesky [max-iterations]]`. The default iteration limit remains 100.
The optional limit is recorded in `options.txt`; the
[large-graph workflow](../LARGE_POSE_GRAPH.md) uses it to investigate a case that
reaches 100 iterations. Historical reports retain their original executables
and settings.

## Input and reproduction

The 30-pose problem is the original pinned public `w100_30.g2o` sample from the
[initial validation](../../docs/validation/pose_graph_2d.md). The 100- and
300-pose problems are **derived between-factor subgraphs** of GTSAM 4.2.1's
`w10000.graph`. They are related inputs from the w100 family, not three
independent datasets or the complete original 10,000-pose SLAM task.

The source contains TORO-order information and `EQUIV` records. The converter
requires explicit `--between-only` selection, retains vertex IDs `0..N-1` and
edges whose endpoints both lie in the prefix, and omits equivalence records.
There are 5,875 source equivalence records; 40 within the 100-prefix and 94
within the 300-prefix are omitted. The output does not model these constraints.
The provenance sidecar records those counts, the exact information permutation
and both checksums. No pose coordinate or retained relative measurement is
changed. The scalar oracle also accepts principal-angle normalization in exports
while still rejecting actual changes to translation, rotation or information.

```sh
python3 benchmarks/convert_toro.py \
  --input "$ropt_reference_root/gtsam-4.2.1/examples/Data/w10000.graph" \
  --output /tmp/w100-prefix300.g2o --vertices 300 --between-only
cargo build -p ropt --example benchmark_pose_graph --release --locked
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/exact_pose_graph.py \
  --input /tmp/w100-prefix300.g2o --ropt target/release/examples/benchmark_pose_graph \
  --gtsam "$ropt_reference_root/reference/gtsam_exact" \
  --build-manifest "$ropt_reference_root/manifest.json" \
  --output /tmp/ropt-exact-300-results --repeats 10 --warmup 3
```

Use fresh output paths and change the prefix size to 100 for that case. For the
original 30-pose case, pass the pinned sample directly; no conversion is needed.
Generated datasets, pose solutions, native binaries and external library sources
remain outside the checkout. The reports/raw timings contain no vendored data.

The exact comparison uses ropt's unchanged default nonlinear LM options, with
explicitly tighter CG tolerances: relative 1e-10 and absolute 1e-12. These are
the comparison runner defaults and are captured in options/metadata; the Rust
timing executable and library defaults retain their earlier tolerances. At 300
poses, the earlier CG settings (1e-8/1e-10) gave pose differences of 3.18e-6 and
7.38e-6 from dense, despite matching cost and small gradients. Tightening CG
passes the original 1e-6 pose gate; the gate was not relaxed.

The generic timing example supports `--backends scalar_cg,block_cg` for larger
sparse-only measurements, without constructing an untimed dense reference. Its
consistency reference then uses the first selected backend. Backend selection
and CG tolerances are explicit; unknown/duplicate backends and invalid tolerances
are rejected. QR remains available with `--backends`, but was not measured in
this scale sweep.

## Recorded comparison

All problems use squared loss and one exact anchor at vertex 0. GTSAM uses
COLAMD multifrontal QR and the LM settings of the earlier baseline. ropt uses
forward AD and the selected dense or PCG backend. GTSAM optimizer construction
and `optimize()` are inside its C++ timer. Input parsing, graph construction,
initial cloning, final object destruction, output and validation are excluded
for both programs. Intermediate allocations and cleanup remain inside their
solve calls. The ropt backend order rotates each round; GTSAM runs afterward.
These are complete configured solves with different damping/retraction/derivative
and linear methods, not a controlled comparison of just the linear algorithms.

The [30-pose](../results/exact/poses30/report.json),
[100-pose](../results/exact/poses100/report.json) and
[300-pose](../results/exact/poses300/report.json) reports include the build
manifest, machine/thread settings, source/driver/oracle fingerprints, diagnostic,
accuracy gates and raw timing summaries. Each problem has 10 samples per solver
after three warmups, totaling 120 measured solves. Every sample is retained.
CPU affinity was `{0}`, the thread variables were 1, and CPU frequency/cloud
co-tenancy remain uncontrolled. Memory was not measured.

| Poses / edges | ropt dense ms | ropt scalar CG ms | ropt block CG ms | GTSAM exact ms |
| --- | ---: | ---: | ---: | ---: |
| 30 / 59 | 3.6579 | 4.3330 | 5.4231 | 1.5122 |
| 100 / 300, derived | 58.1138 | 43.2870 | 48.7329 | 6.7144 |
| 300 / 721, derived | 2033.5098 | 708.7621 | 795.2902 | 98.5955 |

These are median complete-solve times. **GTSAM is faster in every recorded
configuration**, including the 300-pose case where ropt's fastest measured
backend takes about 7.2 times as long. This does not establish a general speed
ranking or predict other datasets, but it is a concrete performance gap for
further profiling. Block CG reduces linear work here and still costs more time
than scalar CG. No ropt superiority or SOTA claim follows from this comparison.

All four candidates pass the same independent gates at each size: exact anchor,
cost within `1e-9*max(1,abs(reference_cost))`, stored-coordinate gradient infinity
norm at most 1e-5, and per-pose translation/wrapped-angle difference at most
1e-6. The dense regression reference is not a global-optimality oracle. Final
costs are 0.08290751089989, 0.56891259146098 and 1.05375815353753 respectively.
Largest independent gradient infinity norm is 2.85e-8; largest pose difference
from dense is 5.99e-8. GTSAM iterations and ropt LM trials are recorded separately;
at 300 poses they are 19 GTSAM iterations versus 16 dense/block and 17 scalar-CG
ropt trials. Comparing those counts as identical work would be misleading.

Further work needs profiling and allocation counts, sparse direct factorization,
larger complete datasets, BA/robust cases, peak memory, and Ceres/g2o references.

The later [PCG workspace](../BUFFERED_PCG.md) and [sparse LDLᵀ](../SPARSE_LDL.md)
workflows implement two of these next steps and preserve this historical sweep.

The reference executable now accepts an optional final `qr` or `cholesky`
argument after `repeats warmup`; omission retains `MULTIFRONTAL_QR`. Cholesky
selects `MULTIFRONTAL_CHOLESKY` with the same exact native derivatives, anchors
and LM settings. Diagnostics record this choice. The later
[reusable LDL comparison](../REUSED_LDL.md) rebuilds this reference, captures a
new manifest and alternates both configurations with ropt. Earlier binaries,
manifests and reports remain historical records.
