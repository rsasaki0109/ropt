# Complete real SE2 replay with mixed automatic-differentiation factors

The direct nonlinear-probe kernel is now measured inside complete **mixed-AD
CSAIL updates**, rather than only synthetic query workloads. Every second
original edge uses runtime forward-mode AD: **586 AD and 586 analytic factors**,
with derivative chunk size 6, Cartesian right SE2 tangents and the exact fixed
root. The input, odometry seeds, measurements, information matrices, whitening,
linear solver and convergence policy are unchanged.

Update time has lower medians in both campaigns, but **a repeatable complete-stream
speedup is not established**. Primary improves 3.0% and wins 6/6 pairs;
confirmation improves 1.6% but wins only 4/6. The existing case-specific criterion
requires median speedup > 1 and at least 75% faster pairs **in each campaign**.
Confirmation fails that criterion. All slow samples remain in the records;
there is no extra retry, sample exclusion or accuracy relaxation.

| Campaign | Graph/CSR probes before | Direct probes after | Median time reduction | Faster pairs | Speed criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary | 6.213 s | 6.024 s | 3.0% | 6/6 | Pass |
| Confirmation | 6.376 s | 6.275 s | 1.6% | 4/6 | Fail |

The [primary](results/real_ad_probe/primary/) and
[confirmation](results/real_ad_probe/confirmation/) archives contain **25,056 timed
arrivals**: two kernels × six complete streams × 1,044 arrivals per campaign.
One complete warmup stream is discarded in each fresh process. Before/after
order alternates across six rounds, pinned to CPU 0 with one thread. Builds,
tests, profiling and other replay jobs finish before timed campaigns. Each
complete stream is independently checked after timing; external checking is
excluded from update timers.

**Complete poses are byte-identical before/after in every paired round.**
All non-timing update fields also match: true cost, current nonlinear gradient,
steps, successful relinearizations, recoveries, probe count and Gaussian tree work.
Per kernel/campaign there are **780 requested calls, 618 full recoveries,
7,662 probes and 2,425,338 re-eliminated scalar cliques** in committed reports.
Timings include failed attempts; the successful-work counters do not account for
all arithmetic performed inside rejected calls.

## Accuracy and scope

All timed arrivals pass the unchanged original-information checks:

- Fixed root: exact identity, with no approximate prior.
- Independent column-normalized gradient: <= `1e-7`; maximum `5.8075741199751337e-8`.
- Reported actual cost: error <= `1e-10 * max(1, original cost)`.
- Archived native/GTSAM agreement at every prefix: world pose norm <= `2e-6`
  and original cost difference <= `1e-9 * max(1, original cost)`.

Maximum archived-native pose difference is `3.772474963380071e-7`, and maximum
original-cost difference is `4.3165471197426086e-13`. The hashed native archive is
an accuracy reference only. **No fresh native/GTSAM timings, full-refresh policy
comparison, live/peak memory or allocation diagnostic are collected here.**
The previous [query-only measurements](MIXED_AUTODIFF_PROBE.md) retain their
49,152 queries and case-specific speed conclusions. Their 10–26% reductions are
not substituted for complete-stream gains.

Complete [MIT functional regression](results/real_ad_probe/mit_functional/)
adds **1,227 before/after arrivals** with **742 AD / 741 analytic factors**.
All original-information gates pass, with exact before/after complete pose bytes
and non-timing records. Maximum independently checked normalized gradient is
`7.793438497199398e-8`. The ill-conditioned original edge **160→161**, whose
information includes values near `1e12`, is ordinal 160 and explicitly uses AD.
Its information is not changed. Native pose/cost agreement remains within the
original tolerances: maximum pose gap `1.945092214966316e-7` and cost gap
`6.423306331271306e-12`. MIT runs are cold functional diagnostics and overlap;
those timers are **not comparative speed evidence**.

## Driver and validation

The stream parser has one explicit differentiation selector. Existing analytic
entry points select `None`; the new
`benchmark_ad_selective_se2_isam` entry point selects runtime AD through
`ROPT_BENCH_AD_CHUNK` and `ROPT_BENCH_AD_STRIDE`. Selection starts at original
stream edge zero, continues across arrivals, and never drops/reweights an edge.
The library's existing `RelativePose2::residual_autodiff` is evaluated on the
same right retractions. Fixed blocks still participate in derivative chunks.
Factor parsing/construction happens before timing; each process records its exact
recipe counts in `differentiation.txt`. No numerical differentiation is introduced.

The AD and analytic recipes share the same replay loop, selection threshold,
requested-key policy and full recovery logic. Every update's nonlinear summary
is evaluated at committed estimates. The selected policy requests at most one
pass at keys exceeding half the `8e-8` target and forces full recovery when
needed. GN uses ordered sparse QR, column-normalized convergence, unchanged
`1e-12` step / `1e-15` cost tolerances, at most 200 trials, topology reuse and four
final refinement steps. Failed attempts and recovery stay inside timing.

Two additional benchmark tests compare analytic, mixed/full AD and chunked
selection on a nonlinear inconsistent loop, including independently computed
original Mahalanobis energy, exact seeds/root and solved pose agreement. Invalid
chunk/stride values fail before file loading. **Rust 320 library/integration/
doctests + 2 benchmark tests; Python 23; Clippy all targets, formatting and Rustdoc
with warnings denied pass.** The core remains exactly the preceding measured
kernel; this turn adds benchmark/validation coverage and makes no new solver
algorithm change. The pre-refactor shared driver sources are preserved in
[previous_driver_sources](results/real_ad_probe/previous_driver_sources/) so old
measurements keep their original driver hashes.

## Reproduction

Build both kernels with the prepared Rust toolchain and cached dependencies.
Each snapshot uses its own Cargo target directory; never share targets between
different sources. The new entry point is included in the preceding-kernel setup:

```sh
python3 benchmarks/make_ldl_baseline.py --before-mixed-probe --source-only --output /tmp/ropt-before-real-ad
cargo build --manifest-path /tmp/ropt-before-real-ad/Cargo.toml --release --locked --offline -p ropt --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_real_ad_probe.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-real-ad/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-real-ad --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-real-ad-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
```

Repeat the last command with a fresh output directory for confirmation. Finish
builds/tests before timing. The runner checks complete original-information
accuracy, native reference hashes, exact before/after results/work, recipe counts
and unchanged kernel/driver/executable hashes. Retained slower pairs remain in
CSV and summary output. The speed criterion is reported separately from accuracy
eligibility, so a correct run may still fail to establish faster performance.

Current core fingerprint:
`5a2f130b59b3b3c8a45f97a825d3339c68f0611fa03c00b63c667246c1eef63e`.
Before direct mixed-AD probes:
`daa5994fbc2bfb6b539fd90b19170048b237df3b093a9d67192d4f50ce9a0fe8`.
Drivers are hashed separately. Both measured kernels used the same archived
[build README](results/real_ad_probe/build-readme.md). CSAIL source SHA256:
`66d99ac857a9849d814d214a9ebd0d4876d5d40f0a37be9330c1ff6e6e9daaa6`.
Stream SHA256:
`fe3ffed0fdae1d69d0e374a915f3364860602a7fe1ad34d19d6de7571e98718a`.

Audit the retained samples, manifest and failed confirmation speed criterion with
`python3 benchmarks/verify_real_ad_probe.py --current-source`. Omit
`--current-source` when inspecting this archive after subsequent development.
