# Complete MIT1228 comparison with original-information precision

ropt is **1.39x faster** than the precision-corrected native GTSAM iSAM2 reference
in the primary complete replay, and **1.33x faster** in independent confirmation.
It wins **12/12 pairs** with every original-information quality gate passing.

| Campaign | ropt median | Native median | Native/ropt | Faster pairs |
|---|---:|---:|---:|---:|
| Primary | 7.947 s | 11.032 s | 1.388x | 6/6 |
| Confirmation | 8.488 s | 11.284 s | 1.329x | 6/6 |

This verifies the real-data goal on this artifact and explicit common-accuracy
policy. It is not a general or default-policy iSAM2 superiority claim.

The pinned Intel MIT1228 graph has 1,228 poses and 1,483 constraints. The earlier
LDL/LM replay stalled at arrival 166; sparse QR/GN then completed it, but ordinary
information whitening failed reported-cost verification. Compensated whitening
fixed ropt's output. This comparison additionally fixes the native reference's
measurement arithmetic, retaining the strict shared quality gates.

This is the separately selected second **artifact of the Intel Research Lab
recording**, not a second independent recording. Its source, seeds and all
information matrices remain unchanged. Arrivals are inferred from pose IDs;
there are no sensor timestamps. See [recorded data provenance](REAL_ISAM2.md).

## Native reference

[gtsam_precise_isam2](gtsam_precise_isam2/) links the same verified GTSAM **4.2.1**
library used for the previous exact-Jacobian native comparisons. Its iSAM2,
Bayes tree, ordering, QR and Gauss–Newton implementations are unchanged. This is
an external benchmark adapter; ropt has no new C++ runtime or FFI dependency.

Both libraries now retain two binary64 components in the Cholesky coefficients
and whitening products when a squared relative pivot is below `2^-26`. There is
**one** such edge in MIT1228. The native adapter supplies its compensated
residual/Jacobians with unit noise; ordinary edges use ordinary GTSAM Gaussian
noise. Its stationarity probe also computes small-pivot original-information
products and column energies with compensation. No matrix is regularized or
altered, and tolerances are not relaxed. This is finite compensated arithmetic,
not arbitrary precision in the timed solver.

The fixed root is conditioned using the exact original BetweenFactor residual
and chart Jacobian. This replaces the old adapter's PriorFactor substitution,
whose identity Jacobian is an approximation away from its prior. The original
uncompensated adapter and its historical failures remain archived separately.
The comparison is explicitly against this **precision-corrected native
reference**, not stock GTSAM defaults or the earlier failed output.

CTest checks exactly representable weak eigen-directions across power-of-two
scales, an independent **200-digit** original quadratic form, analytic weighted
right-chart Jacobians, exact root conditioning, clone preservation, nonpositive
pivots and overflow. Multiprecision is used only in the test executable.

## Cartesian SE2 chart and ropt preparation

The main change is explicit `VariableKind::SE2Cartesian`: a right/body chart
whose origin retraction is `(dx,dy,dtheta) -> (translation(dx,dy),SO2(dtheta))`.
Its local coordinates are relative body translation and principal rotation.
The zero-step differential is the same as the Lie-exponential chart, so existing
right-tangent relative-pose Jacobians remain valid. Translation and rotation
increments avoid the SE2 exponential's extra coupling away from zero.
An isolated compiled probe verifies that the linked GTSAM Pose2 origin retraction
uses this same Cartesian convention.

`VariableKind::SE2` and group `SE2::exp/log/retract/local` keep their original
Lie-exponential behavior. Cartesian retraction/local coordinates are available
explicitly, including forward AD, graph retraction, relative-pose factors and
cached Gaussian inference. Tests check inverse chart identities and their
identity AD chain, independent central differences, an exact one-trial GN solve
of a fixed-root relative pose, invalid inputs and the full cached affine energy.

`RelativePose2::linearize_ambient` takes `[x,y,angle]` directly, avoiding group
construction/angle extraction for principal input angles. Nonprincipal and
extreme finite angles use the group chart's range reduction; no finite-angle
subtraction can overflow. The constructor caches fixed measurement trigonometry.
Principal-angle branch cuts and binary64 conditioning remain limitations.

The explicit nonmonotonic sparse QR/GN algorithm is retained. Three preparations
are cheaper while retaining the exact numerical row sequence and permutation:

- Stable counting groups each row by its leading pivot, computing the key once
  instead of recomputing it during comparator sorting. Constant rows stay last.
- Block minimum-degree ordering builds its coarse graph directly from row block
  supports when a conservative bound proves the original scalar assembly budget.
  A loose or overflowed bound falls back to exact scalar counting; it does not
  reject otherwise eligible input. Degree ties and fill limits are unchanged.
- Sparse QR workspaces clear and reuse numeric row capacity between GN/LM
  trials. Every residual, coefficient, right-hand side and constant is rebuilt;
  failed arithmetic cannot contaminate a later solve.

Ordered CSR rows additionally cache their exact pattern, sorted internal columns
and original numeric-value positions, with explicit ordering validation. Changed
patterns rebuild the mapping. An empty QR pivot with an exactly unit Givens sine
can absorb the incoming row directly, preserving norm/RHS arithmetic and budgets.

Tests compare counted factors bitwise with the old stable-sort sequence, coarse
orders and budget failures with scalar projection, and reused workspaces with
fresh factors after overflow and dimension changes. All MIT1228 estimates and
non-timing report fields match the validated predecessor exactly on the existing
Lie-chart path. Ambient/Cartesian evaluations can differ by ordinary rounding;
their original-information gates are checked separately. Capacity and index
reuse are not claims of reduced peak memory or separately isolated speedups.

The initial structural tuning alone did **not** establish a stable native win:
its first campaign won 4/6 pairs, and confirmation only 1/6. Those complete
negative results remain in [tuning](results/precise_real_isam2/tuning/).
An instrumented isolated snapshot then attributed approximately 58% of update
time to QR rebuilding and 16% to graph evaluation. These nested profiling timers
are diagnostics, not primary speed samples.

## Accuracy and timing contract

Both solvers force full nonlinear refresh at every arrival and target an original
weighted-column-normalized gradient of `8e-8`. ropt keeps block minimum degree
three, the explicit Cartesian chart and ambient analytic evaluation, at most
200 GN trials and four bounded frozen-QR refinements. Native
iSAM2 uses QR/GN and up to 200 calls, including empty updates to stationarity.
This is a common-accuracy policy, not either library's default cached policy.

Independent checks retain the original gates at **every one of 1,227 arrivals**:

- exact fixed root and finite complete estimates;
- original column-normalized gradient infinity norm `<=1e-7`;
- reported cost agreement `<=1e-10 * max(1,cost)`;
- original inter-library cost agreement `<=1e-9 * max(1,cost)`;
- maximum mixed world-pose norm gap `<=2e-6` (metres and radians combined).

The independent Python checker uses original information, with 70-digit Decimal
arithmetic for small pivots. It never reads either library's square-root factor.
All original vertex seeds and observations are kept; committed estimates carry
between arrivals. No optimized seeds, edge deletion or shortened prefix is used.

Each campaign has six alternating AB/BA pairs on CPU 0, one thread, one full
untimed warm-up per process followed by one timed stream. Builds and tests finish
before measurement. Complete update, owned estimates, cost/report and native
stationarity probes are timed. Parsing, factor construction, independent
validation, serialization and report destruction are excluded. Allocation
instrumentation is disabled; peak memory is not measured.

The [primary](results/precise_real_isam2/primary/report.json) and independent
[confirmation](results/precise_real_isam2/confirmation/report.json) retain raw
samples, all update/quality records, full first-pair compressed means,
dataset/stream hashes, source/executable fingerprints and the native manifest.
Across both campaigns there are **29,448 timed arrivals** plus 24 discarded
warm-up streams. Maximum normalized gradients are `4.391e-8` for ropt and
`7.678e-8` for native; maximum inter-library pose gap is `2.044e-7`, original
cost gap `7.831e-12`. An earlier Cartesian exploratory campaign with an auxiliary
compile overlap and its aborted follow-up are explicitly excluded and retained
under `tuning`; official runs started after all checks/compilation finished.
Conclusions apply to this recorded SE2 artifact and explicit accuracy policy,
not general 3D/VIO/BA superiority, default iSAM2 latency or field deployment.

## Reproduction

Build GTSAM 4.2.1 with the exact-Jacobian option and existing benchmark prerequisites
as described in [the recorded reference setup](REAL_ISAM2.md). Then:

```sh
cmake -S benchmarks/gtsam_precise_isam2 -B /tmp/precise-native \
  -DCMAKE_BUILD_TYPE=Release -DGTSAM_DIR=/path/to/gtsam/lib/cmake/GTSAM
cmake --build /tmp/precise-native -j2
ctest --test-dir /tmp/precise-native --output-on-failure
python3 benchmarks/gtsam_precise_isam2/write_manifest.py \
  --library-manifest /tmp/verified-gtsam-library-manifest.json \
  --build /tmp/precise-native --output /tmp/precise-native-manifest.json
cargo build -p ropt --example benchmark_cartesian_se2_isam --release --locked --offline
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_real_isam2.py \
    --ropt target/release/examples/benchmark_cartesian_se2_isam \
    --ropt-policy sparse-qr-gn-cartesian --native /tmp/precise-native/gtsam_isam2 \
    --native-manifest /tmp/precise-native-manifest.json \
    --input benchmarks/results/real_isam2/intel-mit1228-original.g2o \
    --output /tmp/mit1228-primary --rounds 6 --cpu 0
```

The native manifest must describe the actual compiled executable, linked library,
reference directory and every reference source hash; a manifest from the old
adapter cannot be substituted. The saved campaign metadata contains the measured
manifest and build configuration.

[qr_structure.patch](qr_structure.patch) captures the isolated Rust production
change. `make_ldl_baseline.py --before-qr-structure` restores its predecessor.
The subsequent [cartesian_chart.patch](cartesian_chart.patch) records the
Cartesian chart, ambient evaluation and numeric row reuse changes;
`--before-cartesian-chart` restores the intermediate structural predecessor.
All **30** zero-fuzz source rollback stages match archived fingerprints; older
builds were skipped in these source-only checks.
[Rollback records](results/precise_real_isam2/rollback_stages.json),
[verification](results/precise_real_isam2/validation_checks.json).

Verification passes **271 Rust unit/integration tests + 18 doctests**, 18 Python
tests, native CTest, formatting, Clippy and rustdoc with warnings denied. The
unchanged Lie-chart Intel943 and MIT1228 paths retain complete predecessor means
and all non-timing reports exactly; this functional check adds no timing claim.

Subjective maturity stays **75%**. Passing real-data gates does not finish
selective nonlinear iSAM2, broader independent datasets or production validation.
