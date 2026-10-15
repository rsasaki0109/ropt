# Dense variable indexing in nonlinear iSAM probes

Current-estimate probes use an offset-indexed vector when the registered keys
form a contiguous interval and the caller orders all registered variables.
Building the temporary index takes one allocation; factor lookups use checked
offsets. Empty, partial and gapped key spaces retain the preceding BTreeMap path.
The vector is bounded by the number of variables, including fixed blocks, rather
than the largest key. Arbitrary starting keys and intervals ending at `Key::MAX`
are supported. This is a private per-query index, with no public API or dependency
change and no persistent cache.

**Allocation traffic improves; repeatable whole-update speedup is not established.**
In separate successful-probe diagnostics, 32-factor cases save **5 allocation
requests / 1,776 requested bytes**, and 512-factor cases save **83 requests /
22,944 bytes per query**. All 96 norm samples pass bitwise result/work checks and
repeat allocation totals exactly. These are allocation requests, not live/peak
memory measurements. Complete CSAIL median update totals decrease in both
campaigns, but neither passes the existing 75%-of-pairs requirement:

| Campaign | Tree index before | Dense index after | Time reduction | Faster pairs | Criterion met |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary | 6.005 s | 5.912 s | 1.5% | 4/6 | No |
| Confirmation | 6.148 s | 5.821 s | 5.3% | 4/6 | No |

The criterion is median ratio > 1 AND at least 75% faster pairs in each campaign.
All rounds, including slower rounds, are retained. No fresh GTSAM/native timings
or general speed claim are made.

## Measurement and correctness

The complete CSAIL workload has 1,045 poses, 1,172 original constraints and 1,044
arrivals; 586 AD / 586 analytic factors, AD chunk 6, stride 2. Original observations,
information and initialization are unchanged. Each fresh process discards one
complete warmup stream and measures one stream. Six paired rounds alternate
before/after order on CPU 0 with one thread. Builds, tests and allocation
diagnostics finish before comparative timing. Parsing and factor recipes,
independent checks and serialization are outside timing; updates, probes,
requested passes, failed attempts, full recoveries and owned reports are included.

All **25,056 timed arrivals** pass the original-information quality gates and
archived-native pose/cost agreement. Complete before/after pose bytes and every
non-timing update field match in every paired round. Independent maximum
column-normalized gradient remains `5.8075741199751337e-8` (gate `1e-7`, internal
target `8e-8`), with an exact fixed root and reported-cost tolerance
`1e-10 * max(1,cost)`. Archived-native pose/cost gates remain `2e-6` /
`1e-9 * max(1,cost)`. Per kernel and campaign, work remains 780 requested passes,
618 full recoveries, 7,662 queries and 2,425,338 committed re-eliminated scalar
cliques. Failed work is timed but not fully represented by these counters; they
cannot establish whole-stream allocation traffic.

The unchanged allocation driver separately measures one successful query per
norm after four warmups. Two balanced pairs cover 32/512 factors, 0/50/100% AD,
chunk 1/8 and Absolute/ColumnNormalized norms. Setup and independent checks are
outside measurement. Fixed slots and alternating callback slot order are retained.
Allocation instrumentation is absent from comparative stream timing; diagnostic
times are noncomparative. Sparse-layout allocation traffic is not measured here.

Examples for ColumnNormalized, chunk 8:

| Factors / AD | Allocation requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| 32 / 0% | 270 → 265 | 12,312 → 10,536 |
| 32 / 50% | 590 → 585 | 30,232 → 28,456 |
| 32 / 100% | 910 → 905 | 48,152 → 46,376 |
| 512 / 0% | 4,267 → 4,184 | 187,848 → 164,904 |
| 512 / 50% | 9,387 → 9,304 | 474,568 → 451,624 |
| 512 / 100% | 14,507 → 14,424 | 761,288 → 738,344 |

Caller variable order still sets tangent-column ranges; callback slot order is
unchanged. Gradient reduction remains canonical column order, and output maps
remain sorted by key. Input validation, whitening, finite checks, compensated
cost arithmetic and error precedence are unchanged. Duplicate order entries
overwrite the same binding as before; unfilled dense slots remain absent and
unknown factor keys fail. Two regression tests compare offset, high-end and sparse
layouts with the independent legacy FactorGraph/CSR path using nonchronological
order, fixed slots, correlated whitening and both gradient norms, and exercise
partial/duplicate/absent-key binding semantics.

Rust **327 library/integration/doctests + 2 benchmark tests**, Python **23**, all-
target Clippy, formatting and Rustdoc with warnings denied pass. All **43** source-
only historical rollback fingerprints match using zero-fuzz patches; older
historical kernels are not rebuilt in this turn.

The complete [MIT functional regression](results/dense_probe/mit_functional/)
retains exact pose bytes and non-timing work versus the frozen preceding probe-cost
kernel. All 1,227 arrivals pass original-information gates and archived-native
agreement, with 742 AD / 741 analytic constraints and AD on ill-conditioned edge
160→161 near `1e12`. The candidate replay is cold and functional only; its timers
are noncomparative. Archived-native maximum pose/cost gaps remain
`1.945092214966316e-7` / `6.423306331271306e-12`.

## Diagnostic profile

A separate instrumented snapshot of the preceding kernel sampled every 17th
factor while timing complete index construction and final normalization/maps.
It recorded 1,277 successful probes, including 730 with at least 512 factors.
Weighted estimates attribute 6.39% to layout construction and 7.25% to input
lookup/copy in these larger probes. Callback evaluation accounts for 62.05%.
Clock overhead and sampling bias apply: these are phase estimates within probes,
not whole-update fractions, allocation measurements or comparative speed results.
The instrumented replay retains exact complete pose bytes and non-timing work.
The [source, raw samples and aggregate](results/dense_probe/profiling/) are retained
separately from the uninstrumented measured candidate.

## Reproduction

Use the prepared toolchain/cache and independent Cargo targets per source snapshot.
This run used the actual preceding measured executables, checked against their
archived hashes; before/after benchmark driver hashes match. A fresh rebuild of
the frozen source is available with the same toolchain and lockfile:

```sh
python3 benchmarks/make_ldl_baseline.py --before-dense-probe --source-only --output /tmp/ropt-before-dense-probe
cargo build --manifest-path /tmp/ropt-before-dense-probe/Cargo.toml --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_dense_probe_allocations.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-dense-probe/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-dense-probe --output /tmp/ropt-dense-allocations --rounds 2 --queries 1 --allocations --cpu 0
python3 benchmarks/paired_dense_probe.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-dense-probe/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-dense-probe --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-dense-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_dense_probe.py --current-source
```

Repeat the stream command in a fresh directory for confirmation. Omit
`--current-source` when auditing records after further development. Before kernel:
`41c27190a3d4eed0dcb9a00c48886cdd60222df5f20e33c0759dc119911d64f4`.
After kernel: `af6a93f7252853127e972b3128d208ce0cb5798a59c46603565a346b6f96f1ad`.
Instrumented diagnostic snapshot:
`6f0c20f3f67debef045b861ec0a8b2c7ede0635961fe6651c2c585a4aedcb261`.
Drivers are hashed separately. Informal maturity remains 75/100.
