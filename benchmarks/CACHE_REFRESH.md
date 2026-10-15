# Avoid discarded factor-cache copies on scheduled refresh

Forced and interval iSAM refreshes rebuild all factor-owned Gaussian rows. The
preceding code first cloned the old cache's `Vec<Arc<GaussianFactor>>`, then
discarded that clone during reset. A scheduled refresh now starts with an empty
candidate carrying the same configured selective pass cap. The committed cache
remains intact until success. Ordinary and threshold updates still clone the
existing rows they actually use. QR ordering, algorithms, defaults and numerical
operations are unchanged.

The separate forced-refresh allocation diagnostic uses the same public API,
fixture, driver, four warmups and two alternating before/after pairs as the
[preceding factor-capture study](FACTOR_CAPTURE.md). All six workloads preserve
exact result bits and work, with stable counts across both pairs:

| Factors | Additional saved allocation requests per forced refresh | Saved requested bytes |
| --- | ---: | ---: |
| 32 | 1 | 256 |
| 512 | 1 | 4,096 |

Savings are identical for 1/3/16 residual rows per factor on this 64-bit build.
The 512-factor, 3-row case changes **52,063 → 52,062** requests and
**4,166,160 → 4,162,064** requested bytes. The omitted vector also avoids cloning
and releasing its factor references. These are allocation-traffic measurements,
not live/peak memory or comparative runtime measurements. No whole-stream speed
claim is made.

Subsequent [complete CSAIL timing](CACHE_REFRESH_TIMING.md) finds **0.9% less /
0.5% more median stream time**, with **2/6 faster pairs in each campaign**.
Neither campaign qualifies; arrival-latency changes are also mixed. All 25,056
timed arrivals retain exact poses/work and unchanged gates. The allocation
reduction is verified, while a whole-stream runtime improvement remains
unestablished.

A new regression checks both global QR/selective and editable-tree backends:
failed forced refresh leaves the old cache usable without callbacks, failed
interval refresh preserves the update schedule, and a successful retry repopulates
the selective cache. The full suite passes **334 library/integration/doc + 2
benchmark tests**, Python **23**, all-target Clippy, formatting and warning-free
Rustdoc. All **47** historical source-only rollback stages restore recorded
fingerprints with zero fuzz.

Complete MIT's **1,227 arrivals** retain exact pose bytes and every non-timing
update field against the frozen preceding kernel with requested cap 0. Original
constraints/information remain intact, including the near-`1e12` information edge
160→161 under AD. Original-information normalized gradient remains
`7.793438497199398e-8`, below the unchanged `1e-7` gate; internal target is `8e-8`,
reported-cost tolerance `1e-10 * max(1,cost)` and fixed root exact. Archived-native
pose/cost gaps remain `2.0375466499172747e-7` / `7.574385563202668e-12`, within
`2e-6` / `1e-9 * max(1,cost)`. Cold replay timers are noncomparative and no fresh
native timing is measured.

Use the recorded source and prepared toolchain/cache to reproduce:

```sh
python3 benchmarks/make_ldl_baseline.py --before-cache-refresh --source-only --output /tmp/ropt-refresh-before
cargo build --manifest-path /tmp/ropt-refresh-before/Cargo.toml --release --locked --offline --example benchmark_factor_cache
cargo build --release --locked --offline --example benchmark_factor_cache --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_cache_refresh_allocations.py --binary target/release/examples/benchmark_factor_cache --before-binary /tmp/ropt-refresh-before/target/release/examples/benchmark_factor_cache --before-source /tmp/ropt-refresh-before --output /tmp/ropt-refresh-allocations
python3 benchmarks/verify_cache_refresh.py --current-source
```

[Records](results/cache_refresh/) include all raw allocation diagnostics, complete
MIT regression, replay recipe, tests, rollback logs, patch and a hash manifest.
Source before is `8c04f9ad7802899dee48b1114765b34f3fb4a475202d5d9a5b6ebf1b2926caad`,
after `10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`.
Omit `--current-source` after further development. Informal maturity stays 75/100;
historical native timings remain unchanged.
