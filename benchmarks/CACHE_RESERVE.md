# Reserve factor-cache append capacity before cloning

**Rejected candidate, no longer enabled in production.** Subsequent
[complete-stream timings](CACHE_RESERVE_TIMING.md) show about 2% higher observed
medians in both campaigns (2/6 / 3/6 faster pairs). The prior implementation has
been restored; the candidate source and all allocation/timing records remain
available. The savings below describe the candidate, not the current default.

Ordinary iSAM updates previously cloned the cache reference vector at its exact
length, then grew it again when appending new factors. The candidate now reserves
`old_count + incoming_count` before cloning references. This avoids a second
allocation/copy without changing factor order, Gaussian rows, numeric operations,
entry limits or the configured selective pass cap. Forced/interval refreshes
retain the empty-candidate optimization from [scheduled refresh](CACHE_REFRESH.md).
The committed cache remains untouched on error. Capacity addition is checked.

A new public-API diagnostic initializes an observable Gaussian graph, performs
four empty warmup updates, then adds one neutral linear factor to an existing
variable. Fixture/factor construction and independent checks are outside the
allocation window. The counted window includes candidate copying, new-factor
evaluation, QR/tree update, full back-substitution and owned report creation.
Callbacks execute once in both kernels. Original costs, zero gradients, estimate
bits, cached costs, anchor coordinates and work match exactly.

Two alternating before/after pairs cover 32/512 initial factors with 1/3/16
residual rows per original factor. Counts are identical across pairs:

| Existing factors | Saved allocation/reallocation requests per append | Saved requested bytes |
| --- | ---: | ---: |
| 32 | 1 | 504 |
| 512 | 1 | 8,184 |

For 512 original 3-row factors, complete-update traffic changes **3,687 → 3,686**
requests and **1,080,256 → 1,072,072** requested bytes. On the recorded 64-bit
build, savings are `16 * old_count - 8` bytes for this one-factor append, reflecting
the old clone plus doubling growth versus one exact-sized candidate allocation.
These are requested-traffic measurements, not live/peak memory. Instrumented
timers are noncomparative; no whole-stream speed improvement is claimed.

The new regression covers empty/exact/overestimated append hints, overflow,
preserved selective pass cap, failed append budgets, shared immutable old rows
and unchanged committed residual/Jacobian data after successful candidate append.
All **335 library/integration/doc + 2 replay-example tests**, Python **23**,
all-target Clippy, formatting and warning-free Rustdoc pass. All **48** historical
source-only rollback stages restore their recorded fingerprints with zero fuzz.

Complete MIT's **1,227 arrivals** preserve exact pose bytes and every non-timing
update field against the preceding kernel at the same requested cap 0. All
original constraints/information, including the near-`1e12` information on edge
160→161 selected for AD, remain intact. Maximum independent normalized gradient
is `7.793438497199398e-8` under the unchanged `1e-7` gate; internal target `8e-8`,
reported-cost tolerance `1e-10 * max(1,cost)` and exact fixed root are unchanged.
Archived-native pose/cost gaps remain `2.0375466499172747e-7` /
`7.574385563202668e-12`, within `2e-6` / `1e-9 * max(1,cost)`. Cold replay timers
are noncomparative and native timing is not measured.

## Reproduction

Use the prepared toolchain/cache and the archived candidate checkout described
in the [timing reproduction instructions](CACHE_RESERVE_TIMING.md). The commands
below run there; the live builder no longer exposes `--before-cache-reserve`:

```sh
python3 benchmarks/make_ldl_baseline.py --before-cache-reserve --source-only --output /tmp/ropt-reserve-before
cargo build --manifest-path /tmp/ropt-reserve-before/Cargo.toml --release --locked --offline --example benchmark_cache_append
cargo build --release --locked --offline --example benchmark_cache_append --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_cache_reserve_allocations.py --binary target/release/examples/benchmark_cache_append --before-binary /tmp/ropt-reserve-before/target/release/examples/benchmark_cache_append --before-source /tmp/ropt-reserve-before --output /tmp/ropt-reserve-allocations
python3 benchmarks/verify_cache_reserve.py --current-source
```

[Records](results/cache_reserve/) retain all raw diagnostics, complete MIT
regression, recipe, logs, patch and artifact hash manifest. Source before:
`10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`;
after `939279f7aa51cd59af04323230a1433620751c75df1625fbf3c62d26243bc1de`.
Omit `--current-source` after further development. Library defaults, historical
native figures and informal maturity remain unchanged at 75/100.
