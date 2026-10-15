# Complete-stream timing after refinement termination change

This follow-up measures the [unused-step solve removal](REFINEMENT_STEP.md) on
the complete CSAIL SE2 recording. It compares the preceding and changed ropt
kernels with the identical mixed-AD editable-tree driver. Archived native poses
and costs are an accuracy reference; no native timing is collected here.

## Result: adoption withdrawn

| Variant | Stream median | Arrival median | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 11.029 s | 4.617 ms | 59.620 ms | 119.662 ms | 748.545 ms |
| Candidate | 11.930 s | 4.797 ms | 60.918 ms | 116.836 ms | 1252.352 ms |

The candidate has **8.2% higher observed stream median** and only **1/6 faster
pairs**, missing the predeclared criterion. No confirmation campaign is run.
The pooled P99 is lower but the observed maximum is higher; neither supports a
repeated tail improvement. Stream ranges are 9.088–16.938 s before and
9.653–13.694 s after. This noisy cloud campaign does not establish a causal
slowdown or statistical significance; it does not support a speed benefit.

**Production is restored to the preceding kernel.** The two-request / 2 KiB
allocation saving at 128 variables remains a rejected candidate result. The
expanded cap/backend anchor and rollback regressions remain in the test suite.
The live historical builder returns to its preceding 53-stage chain.
Restoration validation passes **353 workspace tests (including 19 doctests),
2 benchmark tests, 27 Python checks**, all-target Clippy with warnings denied,
formatting and all 53 source-only rollback fingerprints.

All **12,528 timed arrivals** pass the unchanged original-information and
archived native agreement gates. Complete poses and all non-timing work fields
are exactly equal before/after and within variants. Maximum original normalized
gradient remains `2.516765743356827e-8`. Saved-data audit independently recomputes
both retained first-stream pose sequences (2,088 arrivals), checks all repeated
work/quality records and reconstructs paired timing summaries.

[Raw samples and source snapshots](results/refinement_step_timing/manifest.json)
retain the plan, conditional-stop decision, failed preflight, binary/tool
identities and build logs. The [audit](results/refinement_step_timing/audit.json)
is separate from the runner's checks of all timed arrivals. No Rust functionality
is removed by the restoration; candidate source and its patch stay archived.

## Predeclared protocol

Run one campaign of six alternating before/after pairs. If its after median
is lower and at least five of six after samples are faster, run one additional
six-pair confirmation. Otherwise stop with speed improvement unconfirmed.
At most two campaigns are allowed; do not add campaigns to obtain a favorable
result. The plan is recorded before the first timed process.

Each fresh process performs one full discarded warmup and one full timed replay,
pinned to CPU 0 and one thread, with allocation counting disabled. No builds,
tests, profilers or separate audits overlap comparative timing. The runner's
own quality checks and first-pose compression run between timed processes.

Every stream includes all 1,044 arrivals, 1,045 poses and 1,172 original edges.
The policy remains editable selective AD chunk six / edge stride two, zero
explicit requested passes, automatic threshold passes retained and explicit full
recovery as needed. The internal normalized gradient target is `8e-8`.
Independent checks require original-information normalized gradient `<=1e-7`,
exact fixed root, reported cost tolerance `1e-10 * max(1,cost)`, native pose gap
`<=2e-6` and cost gap `<=1e-9 * max(1,cost)`. Before/after full poses and all
non-timing update fields must match exactly in addition to these gates.

Timers include complete updates, probes, recovery, failed attempts and owned
means/reports. Parsing, factor recipes, serialization and independent checking
are excluded. Pooled arrival P95/P99 use nearest rank over 6,264 updates per
variant per campaign. Maxima are observations. No memory or statistical
significance claim is made.

## Reproduction

Restore the exact **before** kernel with:

```sh
python3 benchmarks/make_ldl_baseline.py --before-covariance-preflight \
  --source-only --output /tmp/ropt-before-covariance-preflight
```

This removes the later covariance input-validation change as well. To reconstruct
the candidate independently of later source changes, copy the frozen snapshot and
measured helpers into a new directory outside the checkout:

```sh
cp -R benchmarks/results/refinement_step_timing/after_source /tmp/ropt-refinement-step-candidate
cp -R benchmarks/results/refinement_step_timing/frozen_tools/. /tmp/ropt-refinement-step-candidate/
mkdir -p /tmp/ropt-refinement-step-candidate/benchmarks/results/csail_real_isam2
cp benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o /tmp/ropt-refinement-step-candidate/benchmarks/results/csail_real_isam2/
cp -R benchmarks/results/csail_real_isam2/primary /tmp/ropt-refinement-step-candidate/benchmarks/results/csail_real_isam2/
```

Build `benchmark_ad_selective_se2_isam` with `cargo build --release --locked
--offline` in each source tree, using separate target directories. Activate the
Rust toolchain on PATH; environment metadata collection invokes `rustc`.
Run the candidate's frozen runner from that tree, with `--binary` pointing to
its own release example, `--before-binary` to the restored tree's release
example and `--before-source` to the restored tree. The required source hashes
are `--expected-before-source
12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288`
and `--expected-source
ffdd6fd2033b67900137ccbca263df7749c4f6ecc681d98601821e8a8fc2fd74`.
Use the copied g2o via `--input`, copied `primary` directory via `--reference`,
and a fresh directory outside both checkouts via `--output`.
`paired_refinement_step.py --help` lists every argument. The six-round, CPU 0,
chunk-six/stride-two, zero-requested-cap settings are the study defaults.
Follow the recorded conditional confirmation rule; do not add favorable runs.

The before binary matches the preceding fresh-native study's executable SHA.
Both variants use the same compiler, release configuration and driver.
One preflight attempt lacked `rustc` on PATH during environment collection; it
ran no benchmark process. Its log is retained, and activation corrected the
environment before the planned campaign began.

Recheck retained data without rebuilding:

```sh
python3 benchmarks/verify_refinement_step_timing.py
```

`--current-source` checks the candidate fingerprint and intentionally rejects
the production tree. The restoration record binds the preceding source and
release driver at the time of the restoration. Historical native timings are not mixed
with these new ropt timings.
