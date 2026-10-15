# Direct nonlinear probes for mixed and automatic-differentiation factors

`IsamSmoother::nonlinear_summary` now evaluates analytic, runtime AD and mixed
factor recipes directly from committed estimates. It avoids temporary graph
registration and global CSR Jacobian assembly for every supported recipe.
Factor graphs and probes share one internal tangent-space AD evaluator; the
public API and differentiation policies are unchanged.

In the synthetic query workload below, **all 16 conditions** improve in both
six-pair campaigns. Median query time falls **11.1–25.8%** in primary and
**9.6–26.3%** in confirmation. Each condition wins at least **5/6 pairs** in
both campaigns. Every slower sample is retained. This measures **query latency
only**, excluding solver initialization and independent validation. It does
not establish complete AD stream speed, native/GTSAM performance or SOTA.

| Blocks / AD percent / chunk | Absolute reduction, primary / confirmation | Column-normalized reduction, primary / confirmation |
| --- | ---: | ---: |
| 512 / 50 / 1 | 19.0% / 18.0% | 20.0% / 21.4% |
| 512 / 50 / 8 | 25.1% / 25.1% | 25.3% / 26.3% |
| 512 / 100 / 1 | 11.9% / 9.6% | 12.4% / 10.9% |
| 512 / 100 / 8 | 19.5% / 18.6% | 20.5% / 21.0% |
| 2048 / 50 / 1 | 16.1% / 17.5% | 18.1% / 17.2% |
| 2048 / 50 / 8 | 22.8% / 20.9% | 25.8% / 20.9% |
| 2048 / 100 / 1 | 11.1% / 13.0% | 12.9% / 13.0% |
| 2048 / 100 / 8 | 11.7% / 17.7% | 14.5% / 18.0% |

Complete [primary](results/mixed_probe/primary/) and
[confirmation](results/mixed_probe/confirmation/) records contain **49,152 timed
queries** across both campaigns. Each process first initializes a fresh graph,
then makes four discarded warmup queries per norm followed by 128 timed queries.
Before/after process order alternates across six rounds on CPU 0 with one thread.
The two norms share a process and are correlated measurements. No compilation,
other benchmark, profiling or quality-check process runs during timed campaigns.
The source/driver/executable hashes are pinned and checked again at completion;
the two kernels are built in separate Cargo target directories.

## Workload and correctness

Each free two-dimensional Euclidean block has one three-row nonlinear factor
connected to a fixed scalar root. Its residual is
`[x - 0.3, y + 0.2, x*x + sin(y) - root]`, with the root fixed exactly at 1.
The 512/2,048-block workloads use 50% or 100% AD recipes and chunk widths 1 or 8.
Callback slots alternate between `[free, fixed]` and `[fixed, free]`. The remaining
50% use explicit analytic derivatives of the same residual. All factors are
Gaussian with identity whitening. One Gaussian initialization leaves a useful
nonzero nonlinear gradient; **these are probe workloads, not convergence tests**.

Every query is checked outside its timer against independently written residual,
energy, `J^T r` and column-norm formulas at the actual committed estimates.
Before/after cost and overall-gradient bits, per-variable gradient digests,
estimate digests and exact callback/chunk counts match in every round.
The deterministic digests are 64-bit FNV over little-endian words. Unit tests
separately compare each returned gradient component bit-for-bit against the
original CSR implementation.

A separate [allocation diagnostic](results/mixed_probe/allocations/) repeats
both kernels twice with counters enabled, one query per norm. Query allocation
calls fall **8.2–17.5%** and requested bytes **24.5–46.8%**, with identical
repeated counts. Counters remain disabled during latency measurements.
These count successful allocation/reallocation traffic, not live or peak memory;
instrumented diagnostic timers are not used for speed conclusions.

## Preserved numerical and callback behavior

The shared AD layout preserves right manifold retractions, ambient storage
coordinates, tangent derivative dimensions and every derivative chunk.
**Fixed blocks still participate in AD chunks**, even though their Jacobian
columns are excluded from the gradient. Graph registration caches this layout;
a standalone probe constructs it only for each AD recipe it evaluates.

The direct reducer preserves callback slot order, residual/Jacobian validation
including fixed blocks, whitening, canonical active-column reduction order,
zero-entry skipping, stable `hypot` scaling, zero-column fallback and compensated
Gaussian energy. Queries remain read-only and do not advance update counters,
anchors, factor caches or damping schedules. No central differences or implicit
AD conversion are introduced. Global residual/gradient buffers and per-factor
Jet/Jacobian allocations remain; this is not an allocation-free implementation.

The new coverage compares direct and CSR results for Euclidean, SO2, SO3, SE2,
Cartesian SE2 and SE3, chunk widths 1/2/32, permuted slots/order, disconnected
variables, correlated information and both free and all-fixed factor layouts.
It checks wrong output dimensions, NaNs, mismatched derivative dimensions,
invalid fixed-block derivatives and callback errors, including callback counts.
Existing central-difference derivative, noise, manifold, failed-probe ownership,
solver, fixed-lag and incremental-update tests also pass.

**Rust: 302 unit/integration tests + 18 doctests = 320. Python: 23.**
Clippy on all targets, formatting and Rustdoc with warnings denied pass.
[Validation records](results/mixed_probe/validation_checks.json) retain hashes,
checks and logs. All **38 source-only historical rollback fingerprints** match
archived kernels with zero-fuzz patches; older kernels are not rebuilt this turn.

Complete analytic [CSAIL](results/mixed_probe/csail_functional/) and
[MIT](results/mixed_probe/mit_functional/) functional replays retain the original
information matrices and existing gates: normalized gradient <= `1e-7`, reported
cost tolerance `1e-10 * max(1,cost)` and an exact fixed root. All **2,271 arrivals**
pass. Complete pose bytes and all non-timing update fields match the preceding
kernel exactly. Independent maximum normalized gradients remain
`5.636754354038671e-8` / `7.793438692606408e-8`; independent pose/cost differences
are zero at every prefix. These cold functional replay times are noncomparative.
The previous [analytic complete-stream speed result](DIRECT_NONLINEAR_PROBE.md)
and native figures retain their measured kernels and are not relabeled here.

## Reproduction

Use the prepared Rust toolchain and cached dependencies described in the
repository setup instructions. Restore the preceding source into a new directory:

```sh
python3 benchmarks/make_ldl_baseline.py --before-mixed-probe --source-only --output /tmp/ropt-before-mixed
cargo build --manifest-path /tmp/ropt-before-mixed/Cargo.toml --release --locked --offline -p ropt --example benchmark_autodiff_probe
cargo build --release --locked --offline -p ropt --example benchmark_autodiff_probe
python3 benchmarks/paired_autodiff_probe.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-mixed/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-mixed --output /tmp/ropt-mixed-primary --rounds 6 --queries 128 --cpu 0
```

Repeat the last command with a fresh output directory for confirmation. Use
`--allocations --rounds 2 --queries 1` in a separate diagnostic run. Keep the
kernels in separate target directories and finish compilation/tests before any
measurement. The driver fails on formula/count inconsistencies; the runner fails
on result/work/hash differences. All validated samples, including slow pairs,
remain in each campaign's raw CSVs.

Before source fingerprint:
`daa5994fbc2bfb6b539fd90b19170048b237df3b093a9d67192d4f50ce9a0fe8`.
After source fingerprint:
`5a2f130b59b3b3c8a45f97a825d3339c68f0611fa03c00b63c667246c1eef63e`.
Driver hashes are recorded separately; the fingerprint excludes documentation and
this additional example. Both measured builds used the same archived
[README](results/mixed_probe/build-readme.md) for embedded crate documentation.

Verify the retained raw samples and their artifact manifest with
`python3 benchmarks/verify_autodiff_probe.py --current-source`. Omit
`--current-source` when inspecting archived results after subsequent development.
