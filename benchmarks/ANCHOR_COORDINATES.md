# Reusing iSAM principal local coordinates

The cached update previously evaluated `Local(anchor, estimate)` three times
for each active variable: once for the refresh threshold, once for the report's
maximum norm and once for the QR model's cost coordinates. It now computes the
owned local vector once, takes the same norm and copies it into the graph's
tangent columns. Successful cached updates reuse both the vector and maximum.

Scheduled refreshes compute coordinates once after resetting the anchors.
Threshold refreshes discard the provisional coordinates and compute new ones
after batch optimization. The raw QR step's block norms remain a separate
threshold check: a full turn can have a large step but nearly zero principal
local coordinates. `Local(x,x)` is still evaluated after refresh; floating Lie
operations can produce a small nonzero value even at identical poses.

Key order retains the previous maximum reduction. Tangent column ranges follow
graph insertion order, which may differ from key order. Fixed variables remain
excluded. No residual, Jacobian, damping policy, rank/fill checks, tolerance,
factor history, public solver API or dependency changes. Failed updates keep the
committed state. A threshold update can allocate a provisional coordinate vector
that is later discarded; coordinates can temporarily coexist with the raw step.
Allocation traffic does not establish peak-memory improvement.

One regression test covers all five variable kinds together, nonmonotonic keys,
unequal tangent dimensions, fixed blocks, a rotation across the principal-chart
boundary, exact QR model cost and post-refresh norms. A second test uses a raw
SO2 step of a full turn to verify that the raw-step threshold is retained.

## Comparison policy and accuracy

This is a **cached-policy** comparison on the existing nonlinear SE2 fixture:
threshold `0.1`, interval ten, forced delayed loops every 25 arrivals, sparse LDL
with block minimum-degree size three, restart LM damping `1e-3`, gradient
tolerance `5e-7`, cost tolerance `1e-15` and 200 trials. The initial root update
increments the interval counter, so interval refreshes occur at frames 9, 19,
and so on. Seeds, correlated information and delayed non-root loops are unchanged.

All original quality gates execute outside the timers. Every prefix evaluates
independent original energy and central-difference gradients and verifies the
exact fixed root. Every refresh requires original-gradient infinity norm at most
`1e-6` and cached/original cost gap at most `1e-10 * max(cost,1)`. Each 25-arrival
checkpoint additionally solves an independent AD/dense-QR batch reference,
checks pose gap `2e-6`, cost gap `1e-9 * max(cost,1)`, reference reported cost gap
`1e-10 * max(cost,1)` and gradient-step sensitivity `2e-7`.

Cached updates remain approximate: their largest original gradient is about
`0.01003`, exceeding the forced-refresh target. Their times cannot be compared
with the historical native GTSAM campaign as evidence of superior performance.
The separate forced-refresh regression uses its existing all-prefix native
accuracy gates and makes no timing claim.

Each process discards one complete warm-up stream and measures one stream.
Before/after execution order alternates, each variant occupying each position
equally often. Timers include complete update and owned means/report construction;
they exclude root initialization, factor recipes, external accuracy checks,
serialization and report destruction. Checks between updates can affect caches,
equally for both kernels. The same counter allocator is compiled into both
binaries, with allocation counting disabled during primary timings.
CPU affinity is zero and thread settings are one; builds, tests and other solvers
do not overlap primary samples. Cloud frequency and co-tenancy are uncontrolled.

## Measured results

Twelve balanced pairs at each size give **48 measured streams / 9,600 timed
arrivals**; warm-up streams are excluded. Group medians are computed separately
and need not add to the complete-stream median.

| Added poses | Before complete median ms | After complete median ms | Before cached median ms | After cached median ms | Faster complete / cached pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 16.922 | 16.034 | 4.849 | 3.872 | 7 / 10 of 12 |
| 300 | 171.097 | 157.977 | 46.588 | 35.579 | 11 / 12 of 12 |

At 300, cached time is **23.6% lower**, with all twelve pairs faster; complete
update time is **7.7% lower**, with eleven pairs faster. Complete-stream standard
deviations are 8.04 / 4.85 ms, and cached deviations 2.82 / 1.57 ms. Refresh medians
are 123.624 / 121.374 ms (1.8% lower); a reliable refresh-only improvement is not
established. At 100, cached time is **20.1% lower** with ten pairs faster. Complete
medians are 5.2% lower with seven pairs faster, but deviations 1.79 / 2.98 ms exceed
the 0.89 ms median gap. A dependable small-stream total improvement is not established.
These results describe this synthetic, warmed fixture and do not establish
superiority on general SLAM workloads.

Every measured prefix has exactly identical full means, report fields, original
and cached costs, gradient trace and quality summary across kernels and repeats.
At 300 there are 258 cached, 30 interval and twelve forced updates, zero threshold
updates and 316 batch trials. Refreshed original-gradient maximum is `4.905e-7`,
checkpoint pose gap `8.267e-7`, cost gap `1.561e-13`, and final QR storage 29,511
entries. At 100 there are 86 cached, ten interval and four forced updates, with
4,551 final QR entries. Cached original gradients are explicitly approximate.

- [100 report](results/anchor_coordinates/frames100/report.json),
  [metadata](results/anchor_coordinates/frames100/metadata.json),
  [samples](results/anchor_coordinates/frames100/samples.csv).
- [300 report](results/anchor_coordinates/frames300/report.json),
  [metadata](results/anchor_coordinates/frames300/metadata.json),
  [updates](results/anchor_coordinates/frames300/updates.csv),
  [quality](results/anchor_coordinates/frames300/quality.csv).

Options, summaries and compressed full means are retained beside these records.
The source fingerprint covers manifests, library Rust sources and the pose-graph
runner; the cached runner and fixture have separate recorded hashes. Test files
and documentation are not included in that kernel fingerprint.

## Separate diagnostics and verification

A serial allocation run saves **83,730 successful allocation/reallocation
requests** (979,186 to 895,456; 8.6%) and **2,009,520 requested bytes**
(224,930,394 to 222,920,874; 0.9%). These match the independent fixture prediction:
remove two local vectors per active variable on each cached update and one on
each refresh. Instrumented full means, every non-timing report field, independent
quality trace and summary match the accepted primary results exactly. Counting
runs do not contribute to timing medians. This is allocation traffic, not live,
peak or RSS memory; native memory is unmeasured.
[Record](results/anchor_coordinates/diagnostics/allocations/report.json).

The separate 300-arrival forced-refresh check has exact before/after full means,
original/reported costs, trials, damping and stored entries. Every prefix passes
the original `1e-6` gradient, exact fixed root, native pose-gap `2e-6`, original
cost relative `1e-9` and reported cost relative `1e-10` gates. Maximum ropt gradient
is `2.964e-7`, native gradient `4.328e-7`, and mixed metre/radian world-pose gap
`3.442e-8`. The actual resolved native library and executable match the verified
manifest. This single correctness run establishes no native timing advantage.
[Regression](results/anchor_coordinates/diagnostics/refresh300/report.json),
[provenance](results/anchor_coordinates/diagnostics/refresh300/metadata.json).

All **23** reconstructed source stages match archived kernel fingerprints using
zero-fuzz patch application. The immediate predecessor is rebuilt and exercised;
older-stage Cargo builds are skipped in the source-only checks.
[Rollback proof](results/anchor_coordinates/diagnostics/rollback_stages.json).

Validation passes **230 unit/integration tests + 18 doctests**, nine Python tests,
Clippy with warnings denied, formatting and documentation with warnings denied.
The strengthened mixed-chart input is rechecked after the full suite. The new
25-arrival cached runner smoke passes the same quality gates; CI now runs it.
[Final checks](results/anchor_coordinates/diagnostics/validation_checks.json).

## Reproduction

Build both kernels before timing, choose an allowed CPU and fresh output paths.
The helper now reverses the later `qr_rebuild.patch` before
[anchor_coordinates.patch](anchor_coordinates.patch). Restore this archived after
kernel with `--before-qr-rebuild`; verify both fingerprints against the saved
metadata before attributing performance to the coordinate change.
Older stages are checked against archived source fingerprints; their compilation
is skipped in this task. The immediate predecessor is independently rebuilt.

```sh
python3 benchmarks/make_ldl_baseline.py --before-qr-rebuild \
  --output /tmp/ropt-anchor-coordinates-archived-after
python3 benchmarks/make_ldl_baseline.py --before-anchor-coordinates \
  --output /tmp/ropt-before-anchor-coordinates
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_anchor_coordinates.py \
  --before /tmp/ropt-before-anchor-coordinates/target/release/examples/benchmark_cached_isam_se2 \
  --before-source /tmp/ropt-before-anchor-coordinates \
  --after /tmp/ropt-anchor-coordinates-archived-after/target/release/examples/benchmark_cached_isam_se2 \
  --after-source /tmp/ropt-anchor-coordinates-archived-after \
  --frames 300 --rounds 12 --output /tmp/anchor-coordinates-comparison
```

Use `benchmark_cached_isam_se2 300 NEW_DIRECTORY --allocations` separately for
allocation requests/bytes. Instrumented times are excluded from speed claims. To verify the counting
outputs against an accepted primary comparison, use:

```sh
python3 benchmarks/check_anchor_allocations.py \
  --before /tmp/ropt-before-anchor-coordinates/target/release/examples/benchmark_cached_isam_se2 \
  --after /tmp/ropt-anchor-coordinates-archived-after/target/release/examples/benchmark_cached_isam_se2 \
  --after-source /tmp/ropt-anchor-coordinates-archived-after \
  --primary /tmp/anchor-coordinates-comparison --output /tmp/anchor-coordinates-allocations
```
For the native correctness regression, build the verified executable and manifest
following [NATIVE_ISAM2.md](NATIVE_ISAM2.md), then run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/check_anchor_refresh.py \
  --before /tmp/ropt-before-anchor-coordinates/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-anchor-coordinates \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --output /tmp/anchor-coordinates-refresh-check
```

Bayes-tree selective updates, reordering and field validation remain missing.
Subjective maturity remains 75%; this optimization does not close those gaps.
