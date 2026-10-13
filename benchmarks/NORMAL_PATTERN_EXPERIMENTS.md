# Rejected normal-graph assembly experiments

Two candidates follow the retained [Schur update cursor](UPDATE_CURSOR.md).
Neither is adopted: the matched 300-arrival complete-update comparisons do not
demonstrate a benefit. Production normal-graph assembly was restored exactly; the twenty-stage
rollback helper was unchanged at measurement time. The subsequent
[QR reductions](QR_REDUCTIONS.md) are measured separately.

The first candidate inserts each ascending CSR support suffix with the existing
bulk structural cursor. The second retains individual insertion and skips the
diagonals already present in every column. Numeric arithmetic, ordering, fill
limits, optional-plan budgets and APIs are unchanged. Both forward patches
include two private dense structural-oracle tests used during the experiments:
[bulk patch](normal_cursor.patch), [diagonal patch](normal_diagonal.patch).

## Complete-update measurements

Each candidate receives a separate campaign: twelve rotated/reversed rounds at
each of 100 and 300 arrivals, with one discarded warm-up stream and one measured
stream per process. Each variant occupies every position four times. This gives
**144 measured streams / 28,800 timed arrivals** across both campaigns. Successful
six-round 25-arrival smoke checks are separate from primary results.

Both ropt kernels explicitly reuse LM damping and refresh nonlinearly at every
arrival. They use block-minimum-degree sparse LDL, initial damping `1e-3`, gradient
tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10`, and 200 trials.
This is accuracy-conditioned timing, not default cached-update timing. Native
GTSAM 4.2.1 uses the verified configuration in [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Its executable and actual resolved library hashes are checked against the manifest.

| Candidate | Arrivals | Before median ms | Candidate median ms | Native median ms | Candidate faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bulk cursor | 100 | 40.452 | 39.753 | 44.752 | 10 / 12 |
| Bulk cursor | 300 | 533.388 | 548.003 | 436.724 | 5 / 12 |
| Skip diagonals | 100 | 42.336 | 41.271 | 44.273 | 6 / 12 |
| Skip diagonals | 300 | 545.167 | 551.226 | 443.592 | 4 / 12 |

The 100-arrival median reductions are 1.7% and 2.5%, while the 300-arrival medians
increase 2.7% and 1.1%. At 300, sample standard deviations are 20–31 ms, larger
than either difference. These observations establish neither a dependable
speedup nor a general regression. Native is approximately 1.25x and 1.24x faster
than the candidates at 300 in their respective campaigns. Comparing medians
between campaigns cannot establish a causal candidate comparison or general
SLAM superiority.

CPU affinity is 0 with one-thread environment settings. No compiler, test or
other solver overlaps primary samples. Cloud frequency and co-tenancy remain
uncontrolled. Timers cover the complete accuracy-target update and owned means,
including native extra empty updates, stationarity feedback and full cost.
Parsing, initialization, recipe construction, external validation, serialization
and report destruction are excluded; allocation counting is disabled.

Every prefix passes unchanged original-gradient `1e-6`, exact fixed-root,
native pose-gap `2e-6`, original-cost relative `1e-9` and reported-cost relative
`1e-10` gates. Candidate/before means, original and QR costs, entries, LM trials
and damping sequences match exactly in every round. Maximum ropt gradients are
`2.780e-7` and `2.964e-7` at 100 and 300; maximum native pose gaps are `1.422e-9`
and `3.442e-8`. Total ropt LM trials remain 213 and 631.

- Bulk: [decision](results/normal_cursor/decision.json),
  [100 report](results/normal_cursor/frames100/report.json),
  [300 report](results/normal_cursor/frames300/report.json),
  [metadata](results/normal_cursor/frames300/metadata.json),
  [samples](results/normal_cursor/frames300/samples.csv).
- Diagonal: [decision](results/normal_diagonal/decision.json),
  [100 report](results/normal_diagonal/frames100/report.json),
  [300 report](results/normal_diagonal/frames300/report.json),
  [metadata](results/normal_diagonal/frames300/metadata.json),
  [samples](results/normal_diagonal/frames300/samples.csv).

Each result directory also retains all-prefix quality, update reports, options,
input and compressed full means. Original measurement drivers remain preserved;
their after-source was the active candidate checkout at measurement time.

## Separate diagnostics and regression coverage

Warmed inclusive normal-pattern profiles observe 483 calls per kernel:
**14.226 / 9.334 ms** for the bulk candidate and **14.533 / 14.306 ms** for the
diagonal candidate. A faster isolated scope does not establish a complete-update
benefit. All fifteen counters are verified; estimates/non-timing reports match
each kernel's primary run. Scopes overlap and are not additive or causal
percentages. The same [profiling patch](symbolic_merge_stage_profile.patch)
applies with zero fuzz to isolated sources using fresh Cargo targets.
[Bulk profiles](results/normal_cursor/diagnostics/stages_after.json),
[diagonal profiles](results/normal_diagonal/diagnostics/stages_after.json).

The diagonal candidate's allocation diagnostic remains **2,403,726 requests /
568,194,510 requested bytes**, before and after, with exact primary means/reports.
This is cumulative traffic, not live/peak/RSS or native memory. Its 300-arrival
cached-policy trace and summary are byte-identical: 258 cached updates, 30 interval
refreshes, twelve forced loops and twelve AD/dense-QR checkpoints. Maximum refreshed
gradient is `4.905e-7`, checkpoint pose gap `8.267e-7`; the approximately `0.0100`
cached original gradient still describes approximate estimates between refreshes.
These are correctness checks, not cached-performance timings.
[Allocations](results/normal_diagonal/diagnostics/allocations.json),
[cached validation](results/normal_diagonal/diagnostics/cached_validation.json).

Production adds two public-API tests in
[normal_structure.rs](../crates/ropt/tests/normal_structure.rs). An independent
dense support graph checks normal-entry counts, every entry-budget boundary,
gaps, empty rows, reordered/non-adjacent repeated supports and 64-entry tree
promotion. Gapped/hub systems check all three orderings and three damping levels
against independent dense Cholesky, including disconnected diagonals and
zero-dimensional errors. Production and both candidates each pass **223
unit/integration tests + 18 doctests**. Nine Python oracle tests, Clippy, formatting
and documentation checks pass for the final checkout.

[Final validation record](results/normal_patterns_validation.json).

## Reproduce rejected candidates

At the end of these experiments the production kernel fingerprint matched the
pre-experiment kernel. To repeat them after later optimizations, restore that
historical predecessor first. The candidate helper accepts `--source` and applies
either forward patch in a fresh external snapshot without changing production
or making a Git worktree. The comparison driver hashes both supplied source trees and uses
the same original-energy and native-agreement validators.

```sh
python3 benchmarks/make_ldl_baseline.py --before-qr-reductions \
  --output /tmp/ropt-normal-predecessor
python3 benchmarks/make_normal_candidate.py --source /tmp/ropt-normal-predecessor \
  --candidate bulk \
  --output /tmp/ropt-normal-bulk
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_normal_patterns.py --candidate bulk \
  --before /tmp/ropt-normal-predecessor/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-normal-predecessor \
  --after /tmp/ropt-normal-bulk/target/release/examples/benchmark_warm_isam_se2 \
  --after-source /tmp/ropt-normal-bulk \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/normal-bulk-comparison
```

For the other candidate replace `bulk` with `diagonal` and use fresh directories.
Finish builds first and choose an allowed CPU. Both helper outputs reproduce
archived candidate kernel fingerprints and pass six-round native smoke:
[original reproduction record](results/normal_patterns_reproduction.json),
[explicit predecessor reproduction](results/qr_reductions/diagnostics/historical_normal_reproduction.json).

Inclusive diagnostics still show much more time in full nonlinear solving and
QR rebuilding than normal-pattern construction. Selective nonlinear updates
and Bayes-tree work remain architectural gaps. Field validation is still absent;
subjective maturity remains 75%.
