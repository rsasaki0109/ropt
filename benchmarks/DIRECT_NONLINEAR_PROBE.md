# Direct current-estimate probes for analytic iSAM factors

`IsamSmoother::nonlinear_summary` now evaluates all-analytic problems directly
from committed estimates without registering a temporary factor graph or
assembling a global CSR Jacobian. It still reevaluates every installed factor
and returns the same true Gaussian cost, gradient measure and per-variable
gradient measures. Queries remain read-only and do not advance arrival counters
or damping schedules.

On complete CSAIL, editable-stream time falls **8.6%** in primary measurement
and **8.9%** in confirmation (**1.094x / 1.098x**, faster in **11/12 pairs**).
Both campaigns meet the existing case-specific speed criterion: median speedup
> 1 and at least 75% faster pairs. All original-information accuracy gates pass;
complete before/after poses, costs, gradient measures and non-timing work records
are identical. This is a complete-stream improvement, not an isolated probe
microbenchmark or a native/GTSAM speed claim.

| Campaign | Current full | Editable before | Editable after | Before / after | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 3.879 s | 4.482 s | 4.096 s | 1.094x | 6/6 |
| Confirmation | 3.857 s | 4.533 s | 4.128 s | 1.098x | 5/6 |

Values are medians of six complete warmed streams. **The improved editable
policy remains slower than current full refresh**, which also uses the new
direct probe: after/full speedups are 0.947x / 0.934x, with 1/6 / 0/6 faster
pairs. That negative comparison is preserved alongside the before/after gain.
All samples, including the slower before/after pair, remain in the records.

The [primary](results/direct_probe/primary/) and
[confirmation](results/direct_probe/confirmation/) campaigns independently
check **37,584 timed arrivals**. Each editable kernel makes **780 requested
calls**, **618 full recoveries** and **7,662 probes** per campaign; current full
makes **6,264 probes**. Gaussian work remains **2,425,338 re-eliminated scalar
cliques** in each editable case's committed reports. The maximum independently
checked normalized gradient is unchanged at `5.636754354038671e-8`.
Elapsed time also includes failed attempts. [Final verification](results/direct_probe/validation_checks.json)
records kernel/driver/binary/reference hashes, exact results and checks.

## Implementation and numerical contract

One temporary keyed layout resolves current variable values and active tangent
ranges. Variables are validated again, including fixed and disconnected blocks.
Reusable input buffers preserve the caller's factor-slot order. Analytic
callbacks still run once per factor, with all returned residuals/Jacobian blocks
checked, including fixed-variable derivatives. The graph and direct paths share
the same output validation and whitening implementation.

Only reduction visits active blocks in tangent-column order. It accumulates
`J^T r` and column norms row-by-row, skipping the same exact zeros as canonical
CSR. The order is identical to the previous assembled CSR reductions. Stable
`hypot` scaling, zero-column fallback, finite-value checks and the compensated
Gaussian energy reduction are unchanged. Callbacks continue running before a
running reduction's finite check, preserving the previous post-evaluation error
behavior. Gaussian cost is not substituted with cached linearized cost.

If any recipe uses runtime AD, the entire query retains the existing graph/CSR
path, including its callback/chunk semantics. No numerical differentiation or
AD conversion is introduced implicitly. Connectivity/noise options were already
validated by the committed update; query-time value/output/domain checks remain.
This is a Gaussian iSAM probe, not a new general robust-objective evaluator or
an observability certificate.

## Validation

Three additional tests compare direct and previous CSR cost/gradient bits for
permuted variable/callback order, SO3 ambient/tangent dimension differences, SO2,
Euclidean blocks, fixed/disconnected variables, correlated information,
diagonal whitening and derivatives at `1e-150`/`1e150`. Independent Mahalanobis
energy and polynomial gradients check the physical objective. Error comparisons
include fixed-block shape/NaN failures, residual dimensions, invalid disconnected
quaternions and overflow. A mixed AD/analytic test checks unchanged callback
counts, CSR results and current-estimate energy.

Existing probe failure-state/scheduling tests, graph derivative/central-difference,
AD, noise, manifold, solver and fixed-lag tests pass. The full suite passes
**300 Rust unit/integration tests plus 18 doctests** (318 total), **23 Python
tests**, formatting and Clippy/rustdoc with warnings denied. All **37 source-only
historical fingerprints** match without patch fuzz; older binaries are not
rebuilt in that check.

The complete MIT1228 functional replay passes all 1,227 original-information
gradient, true-cost, fixed-root and archived-native agreement gates. Entire
before/after pose exports and non-timing update fields are identical. Its
[raw functional records](results/direct_probe/mit_functional/) are retained;
the uncontrolled cold timing is excluded from speed claims.

Current source fingerprint:
`daa5994fbc2bfb6b539fd90b19170048b237df3b093a9d67192d4f50ce9a0fe8`.
Preceding measured source fingerprint:
`c35917f74c8c8d32a434f5ee4f74b54eb9bca0ff2a461bfb37bd5d8afccce41b`.

## Recorded comparison

The before/after editable policies and **current full refresh** run the same
complete CSAIL1045 stream: 1,044 arrivals, 1,172 unchanged original constraints,
original-odometry-only seeds and exact fixed root. All recipes in these recorded
streams are analytic. Current full refresh also uses the new direct probe; its
timing is not a frozen old full-refresh baseline. Both editable policies keep
the same one-request cap and explicit full recovery.

Each campaign balances all six permutations, CPU 0, one thread, one discarded
complete warm stream and one timed stream per process. Timing includes actual
nonlinear probes, failed attempts, requested relinearization and full recovery;
checking and serialization are outside timers. Sources, binaries, drivers and
archived references are checked before/after campaigns. The preceding binary is
frozen from its actual build, not rebuilt through a shared target cache.
Allocator counting is disabled; neither peak/live memory nor allocation traffic
is measured by this comparison.

Every timed arrival must pass unchanged independent original-information gates:
normalized gradient <= 1e-7, exact fixed root, reported-cost error
<= 1e-10 * max(1, cost), original-cost disagreement <= 1e-9 * max(1, cost),
and mixed world-pose disagreement <= 2e-6. Entire before/after pose exports and
all non-timing update fields must match exactly in each round. All samples are
retained without trimming.

No new native/GTSAM timing is collected. Hashed archived precision-corrected
native poses/costs only check correctness agreement. Published native timings
and figures retain their originally measured kernels.

## Reproduce

```sh
python3 benchmarks/make_ldl_baseline.py \
  --output /tmp/ropt-before-direct-probe --before-direct-probe
cargo build -p ropt --example benchmark_selective_se2_isam --release --locked
python3 benchmarks/paired_direct_probe.py \
  --binary target/release/examples/benchmark_selective_se2_isam \
  --before-binary /tmp/ropt-before-direct-probe/target/release/examples/benchmark_selective_se2_isam \
  --before-source /tmp/ropt-before-direct-probe \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --reference benchmarks/results/csail_real_isam2/primary \
  --output /tmp/direct-probe-primary --rounds 6 --cpu 0 --selected-cap 1
```

Repeat into a new directory for confirmation. Partial streams, failed accuracy
gates, changed means/work or changed source/binary cannot produce eligible
reports. A query remains proportional to all installed nonlinear factors; this
does not implement wildfire, dynamic ordering or incremental fixed-lag removal.
The informal maturity estimate remains **75%**.
