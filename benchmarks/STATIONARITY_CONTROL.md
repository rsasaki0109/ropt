# Recorded-data stationarity control and selected updates

Explicit `IsamSmoother::nonlinear_summary(norm)` now evaluates the actual
nonlinear objective and current tangent gradient, rather than cached Gaussian R.
`relinearize_selected(keys)` moves requested anchors and reevaluates their
incident factors without adding observations or advancing the arrival counter.
These APIs allow applications to check a nonlinear accuracy target after bounded
selected updates. They leave the default update policy unchanged.

On complete CSAIL, the **global-QR selected policy** is **1.12x faster** than
current forced-full ropt in primary measurement, and **1.11x faster** in
confirmation, winning **12/12 pairs**. Complete-update time is approximately
**11.0% / 10.1% lower**, with all original-information accuracy gates passing.

| Campaign | Current full | Selected/global QR | Selected/editable tree |
|---|---:|---:|---:|
| Primary | 3.861 s | **3.436 s** | 4.964 s |
| Confirmation | 3.883 s | **3.491 s** | 4.833 s |

The editable policy is slower on this stream, winning **0/12 pairs** over full.
Local clique work alone does not establish lower complete latency. Both positive
and negative results are retained. This comparison measures **current ropt
policies**, with archived native poses/costs used for correctness only.

## Explicit convergence checks

The summary reevaluates all installed factors at committed estimates. It reports
whitened squared cost, maximum gradient measure, and per-nonfixed-variable
gradient maxima. Column normalization uses stable hypot accumulation over current
whitened Jacobian columns; nonfinite scales are errors. Cost uses the existing
stable Gaussian reduction arithmetic. A small gradient remains a stationarity
measure, not a rank/global-optimality certificate. The probe is explicit full
factor evaluation and must be included in any complete-update timing.

Requested relinearization requires a selected-factor mode and unique, known,
nonfixed keys. It preserves the observation history, arrival counter and damping
schedule. Its report reason is `Requested`; configured threshold passes can
additionally select other variables. Callback, finite-value, rank and fill errors
leave owned state unchanged. Probe callbacks can also fail; callback side effects
remain outside the smoother's transactional guarantee.

## Recorded-stream policy

[benchmark_selective_se2_isam](../crates/ropt/examples/benchmark_selective_se2_isam.rs)
provides three explicit current-ropt policies on the same original data and seeds:

- `full`: the existing forced Cartesian sparse QR/nonmonotonic-GN batch update,
  ordering reuse and at most four frozen-QR refinements, followed by a true probe;
- `global`: new-factor cached update and selected factor evaluation, rebuilding
  Gaussian QR globally when selected;
- `editable`: the same selected policy with editable Gaussian Bayes-tree messages.

The two selected policies use the existing anchor threshold `0.1`, one pass per
call, no periodic refresh, and at most **one additional requested call** per
arrival. A probe above the internal normalized-gradient target `8e-8` selects keys
above half that target. Failure or remaining gradient triggers an explicitly
recorded full recovery. Every attempted call, probe, recovery and owned report is
timed; failed work is timed even though committed-work counters cannot describe
it. Factor construction, external verification and serialization are excluded.
Reported cost is the actual nonlinear probe cost, not the cached Gaussian cost.
The cached cost is retained in a separate column to expose the distinction.

The external gates are unchanged at **every arrival**: exact fixed root, complete
finite poses, original-information column-normalized gradient `<=1e-7`, reported
cost agreement `<=1e-10 * max(1,cost)`, inter-policy/reference original cost
agreement `<=1e-9 * max(1,cost)`, and maximum mixed world-pose norm gap `<=2e-6`.
The independent checker uses original information, including 70-digit arithmetic
for MIT1228's small-pivot information matrix. No seed optimization, edge pruning,
prefix shortening or tolerance relaxation is used.

## Symbolic work improvement and tuning

The editable backend now reuses exact symbolic separators/ownership for
same-whole-block-support replacements. Added supports propagate only new filled
separator tails in fixed elimination order. These monotonic additions cannot
remove previous symbolic fill; same-support replacements only change numerical
rows. Removal or changed support retains the existing complete pruning analysis.
Original Gaussian factors, messages and unchanged symbolic separators remain
shared. Metadata cloning/bookkeeping and back-substitution still visit global
state; this is not constant-time updating or dynamic reordering.

A unit test compares added-tail propagation to the independent complete symbolic
analysis, including an earlier new parent, permutation, repeated additions,
removal, changed support and actual Arc identity on same-support replacement.
Complete CSAIL/MIT1228 selected replays retain **byte-identical complete poses and
identical non-timing fields** before and after this symbolic improvement.

All cold functional/tuning replays are archived separately under
[tuning](results/stationarity_control/tuning/). They are excluded from official
speed claims. The exploratory eight-request cap completed all CSAIL arrivals at
the common accuracy but made **817 requested calls** and **94 full recoveries**.
Reducing the cap to one uses **130 requested calls / 103 recoveries** on CSAIL;
MIT1228 uses **111 / 111** with the editable policy. Thus additional local calls
often repeat work before recovery. The selected cap and kernel are fixed before
the official campaigns; the negative eight-call result remains archived.

The archived GTSAM 4.2.1 precision-corrected native poses/costs are hashed and
checked for agreement at every new measured arrival. **No fresh native timing is
collected in this comparison.** Existing native speed results retain their old
measured kernels; this new experiment compares current ropt policies only.

## Validation and archived evidence

Each [primary](results/stationarity_control/primary/report.json) and
[confirmation](results/stationarity_control/confirmation/report.json) has six
balanced three-policy rounds. Across both campaigns, **37,584 timed arrivals**
pass all gates, plus 36 discarded complete warm-up streams with internal target
checks. Maximum independently evaluated original normalized gradients are
`2.443e-8` for full and `5.637e-8` for both selected policies; the maximum
inter-policy/archived-native pose gap is `3.782e-7` and original cost gap
`5.827e-13`. Each selected-policy campaign records **780 requested calls** and
**618 explicit full recoveries**, equivalent to 130/103 per complete replay.
No timing claims are made from cold tuning or archived native timers.

Complete cold checks on MIT1228 also pass all 1,227 original-information arrivals
and archived native agreement for all three policies. Those checks are separate
functional evidence; MIT1228 is not included in the official speed claim here.
The complete source suite passes **294 Rust unit/integration tests plus 18
doctests** (312 total) and **23 Python tests**, with formatting, Clippy and
rustdoc warnings denied. Probe tests check independent polynomial energy and
gradient, fixed-variable conditioning, overflow/failure state preservation,
requested precision improvement and unchanged interval scheduling.

All **33 source-only historical fingerprints** match without patch fuzz; older
binaries are not rebuilt. Restore the preceding editable-tree kernel with
`--before-stationarity-probes` in `make_ldl_baseline.py`; its source fingerprint
is `5ed015a439306d7e2527f5f226220a6324a418469d79798389a9d4298fdc4b38`.
Raw records, input/stream/source/binary/reference hashes and
[verification](results/stationarity_control/validation_checks.json) are saved.
The subjective maturity estimate remains **75%**; editable-tree latency,
dynamic reordering, wildfire and incremental fixed-lag removal remain open.

## Reproduce

```sh
cargo build -p ropt --example benchmark_selective_se2_isam --release --locked
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 benchmarks/paired_stationarity_se2.py \
  --binary target/release/examples/benchmark_selective_se2_isam \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --reference benchmarks/results/csail_real_isam2/primary \
  --output /tmp/stationarity-primary --rounds 6 --cpu 0 --selected-cap 1
```

Repeat with a fresh output directory for confirmation. Activate Rust on PATH for
environment metadata. Six permutations of the three policies balance each
position; every process discards one complete warm-up before its timed stream.
Builds, numerical tests and source rollback checks finish before official timing.
Raw samples, every update/quality record, full first-round poses, options,
dataset/stream hashes, executable/source fingerprints and reference hashes are
retained. Explicit recoveries are never hidden as cached updates.
