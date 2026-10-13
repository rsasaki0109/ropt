# Compact adjacency for greedy minimum-degree ordering

This follows [whitening buffer reuse](WHITENING_BUFFERS.md). Sparse LDL's greedy
minimum-degree ordering previously used tree sets for every live neighborhood
and allocated a neighbor vector for every eliminated vertex. It now reuses the
existing hybrid structural storage: sorted vectors hold up to 64 distinct
neighbors, and larger sets promote to a tree. One local neighbor vector is
cleared/refilled across eliminations. Tree-backed sets do not demote after removal.
Small cleared vectors retain capacity until analysis ends.

Ascending neighbor traversal, symmetric insertion/removal, degree/key tie-breaking,
heap stale-entry checks and cumulative fill accounting are unchanged. Removing a
vertex does not refund that cumulative budget. Each new fill edge checks the same
limit before insertion. Scalar and block greedy orderings share this path; natural
ordering, normal-pattern construction, numeric arithmetic, plan budgets and public
APIs remain unchanged. This is still exact greedy minimum degree, not AMD/COLAMD.
No unsafe code or dependency is introduced. Entry limits do not bound peak memory.

A container regression checks reverse insertion, duplicates, the 64/65 promotion
boundary, missing/present removal, clear and reinsertion. A second regression uses
an independent dense boolean graph and full degree/key scan, sharing neither heap
nor hybrid sets, to compare complete orders and budget errors on isolated vertices,
disconnected components, cycles, a high-degree wheel, dense graphs and a reverse
star. Existing star/dense Cholesky references, block ordering, optional-plan fallback,
exact-pattern changes, rank and rollback regressions pass. There are **205
unit/integration tests + 18 doctests**, plus nine Python quality-oracle tests;
Clippy, formatting and documentation checks pass.

## Matched complete-update comparison

The near-truth synthetic SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both candidates retain whitening reuse,
in-place QR, direct CSR, graph retraction, compact symbolic columns and explicit
adaptive LM damping reuse. They use identical input bytes, shared runners,
sparse LDL/block ordering and full nonlinear refresh after every arrival.
Initial damping is `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`,
step tolerance `1e-10` and trial limit 200. This is accuracy-conditioned timing,
not either library's default incremental policy.

Twelve rotated/reversed rounds place each candidate in each position four times.
Each process discards one warm-up stream and measures one full stream: **72
measured streams / 14,400 timed arrivals** over 100 and 300 added poses.
Every prefix passes the unchanged original-gradient bound `1e-6`, exact fixed
root, native mixed metre/radian pose difference `2e-6`, relative original cost
bound `1e-9` and reported cost bound `1e-10`. Before/after estimates, costs,
LM trials, damping telemetry and QR entries match exactly. Both use **213 / 631
trials**, and first-round full mean CSVs are byte-identical. Original gradient
maxima stay `2.780e-7 / 2.964e-7`; native pose gaps `1.422e-9 / 3.442e-8`.
No failed sample or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Tree-only adjacency | Hybrid/reused neighbors | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 50.441 | 49.584 | 44.127 |
| 300 | 689.723 | 676.304 | 428.188 |

Observed median reductions are **1.7% / 1.9%**, with seven of twelve pairs faster
at each size. Standard deviations (denominator N) are 3.7 / 1.4 ms before/after
at 100 and 31.9 / 38.4 ms at 300. The effects are small relative to timing
variation and do not demonstrate statistical significance or a dependable
whole-update speedup. No real-time, per-update or general SLAM guarantee is
claimed. Native remains **1.12x / 1.58x faster**. Historical sweeps stay separate;
they are not pooled or used to infer a cross-turn native-gap improvement.

Timing includes complete updates and owned reports/full means. Native additional
updates and stationarity feedback are included. Parsing, initialization, recipe
creation, telemetry formatting, external validation, serialization and report
destruction are outside clocks. Allocation counting is disabled. CPU affinity is
0 and thread settings one; no compiler/test/other solver overlaps primary timing.
Cloud frequency/co-tenancy remain uncontrolled. Native executable and actual
resolved `libgtsam` hashes are verified by the driver.

Raw samples, every-prefix time/quality/telemetry, input, options, compressed means
and source/runner/executable/linked-library fingerprints:

- [100 report](results/ordering_adjacency/frames100/report.json),
  [metadata](results/ordering_adjacency/frames100/metadata.json),
  [samples](results/ordering_adjacency/frames100/samples.csv).
- [300 report](results/ordering_adjacency/frames300/report.json),
  [metadata](results/ordering_adjacency/frames300/metadata.json),
  [updates](results/ordering_adjacency/frames300/updates.csv),
  [quality](results/ordering_adjacency/frames300/quality.csv).

## Separate allocation and cached correctness diagnostics

A warmed 300-arrival allocation run counts complete update/report calls. Requests
fall from **2,651,399 to 2,583,592 (2.6%)**, and requested bytes from **627,669,433
to 622,117,529 (0.9%)**. Means and non-timing reports match primary runs exactly.
These are cumulative allocation traffic, not live/peak/RSS or native memory
comparisons; instrumented times are excluded.
[Allocation record](results/ordering_adjacency/diagnostics/allocations.json).

A separate `profile_symbolic reverse-star:256 100` diagnostic performs 100 fresh
symbolic analyses. Requests decrease from **112,100 to 86,900 (22.5%)**, and
requested bytes from **12,384,800 to 10,652,000 (14.0%)**. Both produce the same
permutation, 511 normal/factor entries, 766 assembly destinations and 255 update
destinations. This is a high-degree structural workload, not a whole-update or
native performance comparison; its instrumented times are excluded from primary
samples. [Record](results/ordering_adjacency/diagnostics/symbolic.json),
[before](results/ordering_adjacency/diagnostics/before-symbolic.csv),
[after](results/ordering_adjacency/diagnostics/after-symbolic.csv).

The default-policy 300-arrival cached validator includes 258 cached updates,
30 interval refreshes, 12 forced loop refreshes and twelve AD/dense-QR reference
checkpoints. Complete before/after traces and summaries are byte-identical.
Original-energy checks pass; cached estimates remain approximate between
refreshes, with maximum original gradient about `0.0100` (unchanged). These
are correctness checks, not cached performance measurements.
[Record](results/ordering_adjacency/diagnostics/cached_validation.json),
[trace](results/ordering_adjacency/diagnostics/cached_trace.csv),
[summary](results/ordering_adjacency/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Rust still has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --example profile_symbolic --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-ordering-adjacency \
  --output /tmp/ropt-before-ordering-adjacency
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_adjacency.py \
  --before /tmp/ropt-before-ordering-adjacency/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-ordering-adjacency \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/ordering-adjacency-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then
`ordering_adjacency.patch` for this predecessor;
older stages reverse it first, then their existing patches. Its rebuilt source
matches the frozen predecessor and passes six-round 25-arrival native smoke.
All fifteen stages apply and match archived kernels in source-only checks;
older stages skip compilation while the latest predecessor is built separately.
[Rollback record](results/ordering_adjacency/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` separately.
Allocation diagnostics append `alloc` to the stream runner. For fresh symbolic
traffic run each candidate's `profile_symbolic reverse-star:256 100` outside
primary timing. Exclude those instrumented times. CI checks correctness without
timing thresholds. Selective Bayes-tree updates, retained history and field
validation remain gaps.

The subsequent [normal-graph rebinding](NORMAL_PATTERN_REBIND.md) is measured separately.
