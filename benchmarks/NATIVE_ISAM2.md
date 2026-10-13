# Native iSAM2 comparison and sparse QR merge improvement

This compares real GTSAM 4.2.1 **iSAM2**, rather than repeated batch LM, with
ropt before/after a sparse QR change. The workload requires an accurate nonlinear
solution and complete owned estimates after **every** arrival. It is synthetic
accuracy-conditioned inference, not either library's default incremental policy,
a bounded-window smoother, or a general SLAM speed claim.

These are the historical sorted-QR checkpoint measurements. The later
[model/input reuse](MODEL_REUSE.md) removes redundant final evaluation and
input allocations, recording a separate before/after/native sweep.

## Input, algorithms and gates

Both executables consume identical text-stream bytes: fixed nonzero root, seeds,
arrival order, measurements and correlated SPD information. There are 100/300
added SE2 poses and two non-root loops every 25 arrivals, including an old-to-old
edge. Seeds use known synthetic truth plus small deterministic perturbations,
precomputed identically for both solvers. They do not use each solver's previous
mean. This near-solution workload does not validate difficult initialization.

Residuals are translation and principal angle of `Z^-1*(A^-1*B)`; objective is
`0.5*rᵀΩr`. ropt fixes root exactly. Native eliminates root: its root-to-pose
factor is an equivalent Pose2 prior with target `root*Z` and identical information,
not a soft anchor. Both complete reports include the original root. Before timing,
native residual/Jacobian probes check these coordinates and exact derivatives
against central differences. GTSAM has `GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON`,
STL allocator and no TBB/MKL/march-native.

ropt forces full refresh on every arrival, using analytic SE2 factors, sparse
LDL with block minimum degree size 3, LM gradient tolerance 5e-7, cost tolerance
1e-15, step tolerance 1e-10, damping 1e-3 and maximum 200 trials, followed by an
undamped Givens QR rebuild. Native iSAM2 uses QR, Gauss-Newton, relinearization
threshold 0, skip 1, wildfire 0, full relinearization checks and cached factors.
`calculateBestEstimate()` obtains full back-substitution. Extra empty iSAM2
updates run until original gradient <=5e-7, with maximum 200 calls per arrival.
These calls, gradient probes, full cost and owned report extraction are **timed**.
Algorithms, retractions and orderings differ; input, objective and output accuracy
are shared. The native driver uses standard iSAM2, with no batch replacement.

An independent pure-Python oracle checks original energy and right-body gradient,
without library residuals/whitening. Its derivative components are tested against
central differences with perturbed poses and delayed loops. Every measured
prefix must have finite means/energy, exactly preserve the root, have gradient
infinity norm <=1e-6, and agree with reported cost within `1e-10*max(1,cost)`.
Before/after and after/native must agree at every prefix in maximum world
translation/angle Euclidean norm <=2e-6 and objective within `1e-9*max(1,cost)`.
The norm mixes translation and radians, not pure metres. Stationarity and
agreement do not certify a global optimum.

## Change and safety contracts

Old QR rows used `BTreeMap`; each Givens rotation allocated a `BTreeSet` column
union, then looked up/updated individual tree nodes. Allocation diagnostics and
code inspection identified this repeated sparse-entry work. R now uses sorted
nonzero vectors merged in ascending order into reusable upper/lower buffers.
Populated rows start with their diagonal, used directly in back-substitution.

Column order, Givens coefficients, arithmetic traversal order, exact-zero removal,
rank cutoff and constant residual cost are retained. Fill accounting preserves
ascending off-diagonal updates and diagonal-last checks, including transient fill
before later cancellations. No pruning, damping, tolerance change or fallback
is introduced. All measured before/after means and objective values are identical.

New tests compare scaled, disjoint-column and reversed-batch insertion with
independent SVD and original least-squares cost; check cancellation; and inject
overflowing constant cost after a successful rotation to verify append rollback.
Existing rank/fill/NaN/Inf and all-history SE2/SE3/rollback tests pass. Entry
limits count logical R entries, not capacity, scratch buffers or peak bytes.
Public append still clones transactionally. Full history and full nonlinear
refreshes remain; this is not an iSAM2/Bayes-tree implementation.

## Results

Six rotated/reversed rounds put each candidate in each position twice. Each
process discards one full warm-up stream and measures one stream: **36 measured
streams / 7,200 measured arrivals**, all prefix and pairwise gates passing.
No outliers are removed. CPU affinity is 0 with single-thread settings; no
compiler/test/other solver overlaps primary timing. Frequency/cloud co-tenancy
are uncontrolled and peak memory is not measured. Initialization, input/factor
creation, external validation, output and report destruction are outside timers.

Median sum of complete update/report-call times, milliseconds:

| Added poses | ropt before | ropt after | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 169.063 | **112.208** | **43.941** |
| 300 | 3,642.425 | **1,384.850** | **420.061** |

ropt time drops **33.6% / 62.0%**, or **1.51x / 2.63x** speedups. iSAM2 remains
**2.55x / 3.30x faster** than improved ropt. This does not establish parity or
superiority. Full-history graph/model/QR rebuilds versus Bayes-tree updates
remain an architectural gap; no exact per-operation attribution is measured.

Across six 300-arrival runs, improved ropt's per-update median is 3.90 ms, P95
10.94 ms and maximum 15.96 ms. These combine growing sizes and loop events,
not fixed-size or real-time guarantees. Native uses 329 calls per 300 arrivals
(110 per 100), all timed. CSV `iterations` counts top-level smoother/iSAM2 calls,
not ropt's internal LM trials.

Maximum original gradients at 300 arrivals are 4.88e-7 for both ropt versions
and 4.33e-7 for native. Maximum cross-library pose norm is 1.954e-6 and objective
difference 1.21e-12. Records retain raw stream/update times, every-prefix quality,
comparisons, input bytes, first-round full means losslessly compressed, options
and source/template/executable/library/configuration hashes:
[100](results/native_isam2/frames100/report.json),
[300](results/native_isam2/frames300/report.json).

A separate allocator-instrumented 300-arrival diagnostic reduces requests from
**22,216,359 to 8,937,704** (59.8%) and requested bytes from **4,449,602,278 to
2,212,836,126** (50.3%), with identical means. This is traffic, not live/peak/RSS
memory or a native memory comparison. Instrumented timings are excluded. Primary
counting is disabled; ropt's wrapper still forwards to System and checks its
disabled flag. [Allocation evidence](results/native_isam2/allocations/report.json).

## Reproduce

GTSAM is an optional external benchmark; Rust has no C++ runtime/FFI dependency.
Use the recorded native manifest's archive/version/library/settings, then build
the new driver against that exact GTSAM installation:

```sh
cmake -S benchmarks/gtsam_isam2 -B /tmp/native-isam2-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=YOUR_GTSAM_PREFIX
cmake --build /tmp/native-isam2-build -j 2
python3 benchmarks/gtsam_exact/write_manifest.py \
  --archive YOUR_PINNED_GTSAM_ARCHIVE.tar.gz --cache YOUR_GTSAM_BUILD/CMakeCache.txt \
  --library YOUR_GTSAM_PREFIX/lib/libgtsam.so.4.2.1 \
  --executable /tmp/native-isam2-build/gtsam_isam2 \
  --reference-directory benchmarks/gtsam_isam2 \
  --reference-cache /tmp/native-isam2-build/CMakeCache.txt --verify-linked-library \
  --output /tmp/native-isam2-manifest.json
cargo build -p ropt --example benchmark_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-merge \
  --output /tmp/ropt-before-qr-merge
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_isam2.py \
  --before /tmp/ropt-before-qr-merge/target/release/examples/benchmark_isam_se2 \
  --before-source /tmp/ropt-before-qr-merge \
  --after target/release/examples/benchmark_isam_se2 \
  --native /tmp/native-isam2-build/gtsam_isam2 \
  --native-manifest /tmp/native-isam2-manifest.json \
  --frames 300 --rounds 6 --output /tmp/isam2-comparison
```

Use a permitted CPU if 0 is unavailable. Output paths must be new and outside
the checkout. Finish builds before timing. The manifest must describe the
actually linked library/settings, driver source and executable hashes; a newly
built executable must not reuse the recorded executable hash. Python needs
only its standard library. `check_isam2_smoke.py ROPT_BINARY [NATIVE_BINARY]`
checks small streams and output/argument protection. CI runs Rust without GTSAM
or timing thresholds. The helper reverses `isam_qr_merge.patch` first, then older
stage patches. Historical records are unchanged and are not pooled with this
different fixture/policy.
