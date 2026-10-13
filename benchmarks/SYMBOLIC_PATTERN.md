# Compact structural columns for sparse LDL preparation

This follows the [explicit LM damping reuse comparison](DAMPING_REUSE.md).
Here both ropt candidates use that same opt-in policy. Only symbolic structural
storage changes; the optimization trajectory and numeric arithmetic are preserved.

The later [QR processing improvement](QR_MERGE_BUFFERS.md) retains this change and
has a separate matched sweep. The measurements here keep their historical kernels.

## Change and contracts

Sparse LDL previously assembled normal/coarse structural columns in tree maps
whose numeric values were always zero. Permuted fill columns used tree sets,
then allocated another vector to extract each completed column. These structures
are recreated when the exact stored CSR pattern changes, including each new
arrival in the growing-graph benchmark.

Short structural columns now use sorted index vectors, initially reserving eight
entries. After 64 entries a column promotes to a tree set; insertion shifts are
therefore bounded rather than growing linearly with a high-degree column.
Both representations iterate in ascending order. Completed short fill columns
move their vectors directly into the fill pass without another allocation.
Normal assembly reads borrowed CSR row indices instead of copying unused values.

Normal/coarse pattern deduplication, diagonals for uncovered columns, deterministic
minimum-degree tie breaking, block ordering, fill accounting and optional operation
plan budgets are unchanged. Numeric assembly, LDL updates, original-operator
residual checks/refinement, LM damping and tolerances are unchanged. No public API,
automatic fallback solver, extra damping or total-memory guarantee is introduced.
Large columns still use logarithmic tree insertion. The private eight-entry
reservation and 64-entry promotion are storage choices, not convergence options.

Two independent regressions use 96-column reverse-order star and dense systems.
They check the known star elimination order, exact normal/factor entry counts,
boundary budgets and failure below them, empty rows, cancelled duplicate triplets,
repeated dense rows, all three ordering modes and dense Cholesky solution/residual
agreement. Existing pattern-change, rejected-trial, rank/fill rollback and prepared
workspace tests still pass. There are **188 unit/integration tests + 18 doctests**;
Clippy, formatting and the nine Python validation tests pass.

## Matched streaming measurements

The [native comparison](NATIVE_ISAM2.md) describes input generation and verified
GTSAM 4.2.1 configuration. Identical near-truth synthetic SE2 streams are supplied
to the previous and current kernels through the **same shared runner**, using
sparse LDL/block ordering and explicit adaptive damping reuse. Every arrival
forces a full nonlinear refresh. Initial damping is `1e-3`, gradient tolerance
`5e-7`, cost tolerance `1e-15`, step tolerance `1e-10` and the trial limit 200.
The unchanged native iSAM2 executable also targets every-prefix stationarity.
This is not either library's default incremental policy.

Each of six rotated/reversed rounds puts each of the three candidates in each
position twice. One complete warm-up stream is discarded before one measured
stream per process. Both sizes total **36 measured streams / 7,200 timed arrivals**.
All-prefix original gradients must be at most `1e-6`; the fixed root must match
exactly. Native pose agreement remains bounded by `2e-6` in mixed metre/radian
world-pose norm, original costs by relative `1e-9`, and reported costs by `1e-10`.
No accuracy gate is relaxed and no failed sample or timing outlier is removed.

The new driver additionally requires **identical before/after means, original
costs, batch trial counts, damping telemetry, reported costs and QR entry counts**
after every arrival. Both configurations take 213 / 631 LM trials at 100 / 300
arrivals. The first-round full mean CSVs also match byte for byte. The largest
original gradients remain `2.780e-7 / 2.964e-7` and native pose differences
`1.422e-9 / 3.442e-8`.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Previous structural storage | Compact columns | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 61.714 | **57.550** | 43.688 |
| 300 | 846.584 | **817.370** | 428.374 |

The observed median reductions are **6.7% / 3.5%**. Native iSAM2 remains
**1.32x / 1.91x faster**. At 300 arrivals, before/after standard deviations over the retained samples
are 28.8 / 30.8 ms, and one paired round is 1.0% slower after the change; a small
aggregate improvement is not a per-update or general-workload speed guarantee.
Reported standard deviations use denominator N over the retained samples.
The old damping sweep is retained separately, not pooled into this result.

Whole update and complete owned report construction are timed. Native extra
updates, stationarity feedback and full estimate extraction are included.
Initialization, input parsing, recipe creation, quality checks, telemetry
formatting, serialization and report destruction are outside clocks. Allocation
counting is disabled. CPU affinity is 0 with single-thread settings and no
concurrent compiler/test/solver. Frequency and cloud co-tenancy are uncontrolled.

Raw samples, every-prefix times/reports/quality, exact input, options, compressed
first-round means and source/runner/executable/library fingerprints are retained:

- [100-arrival report](results/symbolic_pattern/frames100/report.json),
  [metadata](results/symbolic_pattern/frames100/metadata.json),
  [samples](results/symbolic_pattern/frames100/samples.csv).
- [300-arrival report](results/symbolic_pattern/frames300/report.json),
  [metadata](results/symbolic_pattern/frames300/metadata.json),
  [updates](results/symbolic_pattern/frames300/updates.csv),
  [quality](results/symbolic_pattern/frames300/quality.csv).

## Allocation diagnostics

A separate warmed 300-arrival stream instruments allocations only within the
same complete update/report calls. Counts fall from **3,757,409 to 3,594,324**
(4.3%), and requested bytes from **1,026,512,409 to 977,028,313** (4.8%). Full means
and reports match their primary-run counterparts exactly. Instrumented times
are excluded from timing comparisons. These are allocation traffic, not live,
peak or RSS memory, and there is no native memory comparison.
[Allocation record](results/symbolic_pattern/diagnostics/allocations.json).

Fresh symbolic-only allocation diagnostics repeat analysis ten times on the
existing 300/1,000/3,000-pose derived `w10000.graph` between-factor subgraphs and
a synthetic 6,000-column reverse-order star. These are separate from the SE2
stream timing workload. Permutations, normal/factor fill and operation-plan
counts match the predecessor exactly.

| Fixture | Allocation requests, before → after | Requested bytes, before → after |
| --- | ---: | ---: |
| 300-pose subgraph | 53,530 → 40,040 | 20,833,230 → 17,321,390 |
| 1,000-pose subgraph | 236,080 → 168,200 | 141,544,470 → 128,708,070 |
| 3,000-pose subgraph | 997,890 → 783,560 | 312,984,470 → 282,036,230 |
| Reverse-order star | 320,090 → 260,100 | 39,766,960 → 28,820,160 |

The star exercises promotion at high degree rather than relying on short columns
only. Derived graph identities, transformation/source/input hashes, complete
permutations and allocation counts are retained in the
[symbolic diagnostic](results/symbolic_pattern/diagnostics/report.json).
Diagnostic elapsed times include allocator overhead and are not primary benchmarks.

## Reproduce

Build the optional native reference/verified manifest as described in
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust remains C++-independent.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example profile_symbolic --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-symbolic-pattern \
  --output /tmp/ropt-before-symbolic-pattern
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_symbolic.py \
  --before /tmp/ropt-before-symbolic-pattern/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-symbolic-pattern \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 6 --output /tmp/symbolic-comparison
```

Finish all builds before timing, select a permitted CPU, and use fresh output
paths. The helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`, then `canonical_csr.patch`, then
`qr_merge_buffers.patch`, then
`symbolic_pattern.patch`, retaining damping reuse.
Its rebuilt source fingerprint matches the frozen predecessor. Older stages
reverse this patch first, then their previous rollback chain.

For diagnostics only, generate a pinned between-only prefix with
`convert_toro.py` as described in [large-graph validation](LARGE_POSE_GRAPH.md),
then run each candidate's `profile_symbolic /tmp/prefix.g2o 10`; the high-degree
case is `profile_symbolic reverse-star:6000 10`. Whole-stream allocation counting
uses the runner's final `alloc` argument. Do not use those instrumented runs as
primary timing candidates. CI checks functionality, not performance thresholds.

The remaining gap includes complete batch refresh/undamped QR rebuilding,
all-history storage and missing Bayes-tree/selective updates. This change does
not claim real-time behavior, iSAM2 equivalence or general OSS superiority.
