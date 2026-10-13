# Explicit adaptive LM damping reuse

This follows [accepted-model/input reuse](MODEL_REUSE.md). Repeated full refreshes
were restarting LM at the configured damping even when the preceding problem
had converged with much smaller damping. The new opt-in policy retains that
adaptive state. This changes the optimization trajectory; it is a policy
comparison, not a same-arithmetic kernel optimization.

The subsequent [compact structural storage](SYMBOLIC_PATTERN.md) has a separate
same-policy comparison. The measurements here retain the earlier kernels.

## API and state contract

```rust
use ropt::isam::{IsamOptions, IsamSmoother};
let smoother = IsamSmoother::new(IsamOptions::default())?
    .with_batch_damping_reuse()?;
assert_eq!(smoother.next_batch_damping(), Some(1e-3));
# Ok::<(), ropt::Error>(())
```

The default continues to restart from `batch_solver.initial_damping`. Enabling
reuse requires Levenberg-Marquardt and finite, positive initial damping. The
first refresh starts with that configured value. Each successful full refresh
proposes its final adapted damping for the next refresh; ordinary cached
updates leave it unchanged. Enabling reuse again preserves the existing value.

`IsamReport::batch_trials` counts batch LM/GN trials, including rejected trials,
and is zero for cached updates or convergence at the initial state.
`batch_initial_damping` and `batch_final_damping` are `Some` on LM refreshes and
`None` on cached updates or Gauss-Newton. The final value is the solver's
**adapted next-trial proposal**, rather than necessarily the damping used in the
last linear factorization. `next_batch_damping()` returns `None` for the default
restart policy, or the committed next-refresh value when reuse is enabled.

Gain-ratio acceptance, rejection growth, the existing damping floor and all
termination tolerances remain unchanged. There is no automatic reset or backend
fallback. Damping commits only after the batch solve, final **undamped** QR
rank/fill/solve checks and estimate/cost checks succeed. Failures preserve the
previous damping, estimates, QR, factor history and update count. External
callback side effects are outside this transaction.

Four regressions check reduced trial counts on repeated linear observations,
cached-update preservation, repeated enabling, rollback after a successful
batch solve exceeds final QR fill, invalid/Gauss-Newton options, and recovery
from a sudden nonlinear change that rejects an overshooting trial. Reuse does
not promise faster or successful convergence on every problem.

## Matched measurements

Four candidates run against identical input bytes: the previous kernel with
restart damping, the current kernel with default restart damping, the current
kernel with explicit reuse, and the same verified native GTSAM 4.2.1 iSAM2
executable used in the preceding sweep. Rust candidates share one benchmark
runner, input parsing, factor recipes, solver options and output scope. Telemetry
extraction/formatting is outside clocks; the complete owned update report is
inside clocks. Native stationarity feedback, any extra updates and complete
estimates are also timed. Constructors, recipe creation, validation, output and
report destruction are excluded.

The [native comparison](NATIVE_ISAM2.md) describes the synthetic near-truth SE2
streams and reference configuration. All ropt candidates explicitly force a
full nonlinear refresh on **every arrival**, using sparse LDL/block ordering,
initial damping `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`, step
tolerance `1e-10` and at most 200 trials. Only the reuse candidate carries
adaptive damping between refreshes. This is an accuracy-conditioned workload,
not either library's default incremental policy.

Eight rotated/reversed rounds put each candidate in each execution position
twice. Each process performs one untimed warm-up stream and one timed stream.
Across both sizes there are **64 measured streams / 12,800 timed arrivals**.
Every prefix passes the unchanged original-energy gradient bound `1e-6`, exact
fixed-root check, mixed translation/angle world-pose agreement bound `2e-6`,
relative original-cost agreement `1e-9` and reported-cost agreement `1e-10`.
No failed sample or outlier was discarded.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Previous restart | Current restart | Explicit reuse | Native iSAM2 |
| --- | ---: | ---: | ---: | ---: |
| 100 | 98.169 | 96.218 | **62.347** | 42.990 |
| 300 | 1,251.914 | 1,277.554 | **890.504** | 432.524 |

Reuse reduces time relative to the current restart policy by **35.2% / 30.3%**.
Total LM trials per stream fall from **690 to 213 / 2,402 to 631**. These are
ropt batch trial counts, not native iSAM2 update counts. Default restart means
and original costs are exactly identical to the previous kernel; its roughly
2% timing differences do not establish an improvement.

Different damping policies produce different termination trajectories. The
largest restart/reuse pose difference is **1.954e-6** at 300 arrivals, within
the existing `2e-6` gate; means are not bitwise identical. Reuse/native maximum
pose differences are **1.422e-9 / 3.442e-8**, and their maximum original-cost
difference at 300 arrivals is **4.242e-14**. Reuse's largest original gradient
is **2.780e-7 / 2.964e-7**. These tolerances mix metres and radians in the pose
norm and describe this fixture, not general SLAM accuracy or covariance coverage.

Native iSAM2 remains **1.45x / 2.06x faster**. Full batch optimization and QR
rebuilds, all-history storage and lack of Bayes-tree/selective updates remain
architectural gaps. No real-time, default-policy or general OSS superiority
claim follows from this result. Allocation counting is disabled and memory was
not measured. Runs use CPU 0 and single-thread settings, with no concurrent
compiler/test/solver; cloud frequency and co-tenancy are uncontrolled.

Raw samples, every-prefix update times/telemetry/quality, exact input, first-round
losslessly compressed full means, options and source/runner/executable/library
fingerprints are retained separately from all historical sweeps:

- [100-arrival report](results/damping_reuse/frames100/report.json),
  [metadata](results/damping_reuse/frames100/metadata.json),
  [samples](results/damping_reuse/frames100/samples.csv).
- [300-arrival report](results/damping_reuse/frames300/report.json),
  [metadata](results/damping_reuse/frames300/metadata.json),
  [updates](results/damping_reuse/frames300/updates.csv),
  [quality](results/damping_reuse/frames300/quality.csv).

## Reproduce

Build the optional native executable and verified manifest using
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). The Rust library has no C++ runtime dependency.

```sh
cargo build -p ropt --release --locked --offline \
  --example benchmark_isam_se2 --example benchmark_restart_isam_se2 \
  --example benchmark_warm_isam_se2
python3 benchmarks/make_ldl_baseline.py --before-damping-reuse \
  --output /tmp/ropt-before-damping-reuse
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_damping.py \
  --before /tmp/ropt-before-damping-reuse/target/release/examples/benchmark_isam_se2 \
  --before-source /tmp/ropt-before-damping-reuse \
  --restart target/release/examples/benchmark_restart_isam_se2 \
  --reuse target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 8 --output /tmp/damping-comparison
```

Finish builds before timing, select a permitted CPU, and use new output paths.
The helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`, then `canonical_csr.patch`, then
`qr_merge_buffers.patch`, then
`symbolic_pattern.patch`, then
`isam_damping_reuse.patch`, retaining model/input reuse and
sorted QR. Its rebuilt baseline matches the frozen predecessor's source
fingerprint and passes the native smoke comparison. Older rollback stages
reverse this patch first; the full historical rollback chain still recovers
the original map-LDL kernel fingerprint. CI exercises both policies against
original-energy correctness gates without timing thresholds.
