# Grouped sparse QR rebuilds and native iSAM2 comparison

The full-refresh path rebuilds an undamped square-root model from the accepted
whitened Jacobian and residual. Previously it inserted rows in factor-arrival
order. A delayed loop could then rotate through many already populated downstream
rows. The new private rebuild groups rows by their first nonzero column, ascending,
with stable original order within each group. Empty rows are placed last and
retain their irreducible residual cost. Column order is unchanged.

The numeric elimination remains orthogonal Givens QR; no normal matrix is formed
for the cache and no rank, finite, exact-zero or fill tolerance is relaxed. The
new path requires a fresh factor. Public incremental `append` retains input row
order and its failure transaction. Forced, interval and threshold refreshes use
the grouped rebuild; cached appends retain the existing path. Smoother state is
committed only after all checks pass.

Row permutation preserves the Gaussian least-squares model but changes floating
rounding and potentially intermediate fill. QR coefficients and model costs need
not match bitwise. Stored-entry limits still apply to every numeric insertion;
neither the old nor new limit bounds peak memory. Sorting uses temporary row-index
storage. A general speedup or unchanged peak allocation is not claimed.

## Accuracy and comparison contract

Both ropt kernels force a full nonlinear refresh at every arrival and explicitly
reuse adaptive LM damping. They retain sparse LDL with block minimum-degree size
three, initial damping `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`,
step tolerance `1e-10` and 200 trials. This is a comparison of complete updates
at a common accuracy target, rather than default cached policies.

Native GTSAM 4.2.1 is the verified correct-Jacobian build described in
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Its executable and actual resolved library
hashes match the saved manifest. Release configuration disables TBB, MKL and
`march-native`. This is the pinned native configuration, not a claim about all
GTSAM builds or its default approximate stopping behavior.

Every prefix passes the existing independent original-gradient infinity-norm
`1e-6`, exact fixed-root, mixed metre/radian world-pose gap `2e-6`, original-cost
relative `1e-9` and reported-cost relative `1e-10` gates. **None of these gates
changes.** Before/after complete means, original costs, LM trial counts and damping
sequences match exactly in the forced-refresh runs. Reordered QR model cost is
checked against the original energy at the same existing tolerance.

Every process discards one warm-up stream and measures one stream. Three-way
order rotates and reverses across twelve rounds, placing each variant in each
position four times. Timers include complete updates and owned means/report
construction, plus native empty updates, stationarity feedback and full cost.
Parsing, factor recipes, initialization, external validation, serialization and
report destruction are excluded. Allocation counting is disabled. CPU affinity
is zero and thread settings are one; builds, tests and other solvers do not overlap
primary measurements. Cloud frequency and co-tenancy remain uncontrolled.

## Results and scope of the performance goal

| Arrivals / campaign | Before median ms | After median ms | Native median ms | Native / after | After faster than native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 39.180 | 37.367 | 44.257 | 1.184x | 12 / 12 |
| 300 | 535.104 | 405.413 | 435.659 | 1.075x | 12 / 12 |
| 300, independent confirmation | 552.540 | 417.012 | 461.996 | 1.108x | 11 / 12 |

The change lowers 300-arrival complete-update medians by **24.2%** in the first
campaign and **24.5%** in the confirmation, versus its matched predecessor.
At 100 the median is **4.6% lower**. After beats the predecessor in 11/12 pairs
at 100 and 12/12 in both 300 campaigns. The initial 25-arrival smoke is **8.5%
slower than the predecessor**; sorting is not uniformly beneficial at small sizes.
All six smoke pairs still beat native at the same accuracy target.

The first 300 campaign has standard deviations 45.58 / 18.75 / 25.13 ms for
before / after / native; confirmation has 46.36 / 28.50 / 27.25 ms. Variation is
material. All raw samples, including slower runs, are retained. The native win
holds in **23/24** 300-arrival pairs across independent campaigns. Do not attribute
differences between campaign medians to code changes.

The measured goal is met **on these synthetic SE2 streams at the common accuracy
target**. It does not establish dominance on public real-world SLAM/BA datasets,
3D/VIO, different noise, robust losses, different hardware, peak memory, real-time
deadlines, or native default cached policies. The smoother still has no Bayes
tree, selective clique relinearization, reordering or factor deletion.

There are **108 measured primary/confirmation streams / 25,200 timed arrivals**,
plus the separate 18-stream / 450-arrival smoke. The confirmation repeats the
same input, source and executable hashes. Raw input, compressed complete means,
options, every timing and independent quality record are retained:

- [100 report](results/qr_rebuild/frames100/report.json),
  [metadata](results/qr_rebuild/frames100/metadata.json),
  [samples](results/qr_rebuild/frames100/samples.csv).
- [300 report](results/qr_rebuild/frames300/report.json),
  [metadata](results/qr_rebuild/frames300/metadata.json),
  [updates](results/qr_rebuild/frames300/updates.csv),
  [quality](results/qr_rebuild/frames300/quality.csv).
- [Confirmation report](results/qr_rebuild/frames300-confirmation/report.json),
  [metadata](results/qr_rebuild/frames300-confirmation/metadata.json),
  [samples](results/qr_rebuild/frames300-confirmation/samples.csv).

## Separate rotation diagnostic

Actual Givens rotations fall from **2,998,044 to 965,443 (67.8% fewer)** in the
300-arrival stream. Each diagnostic observes 301 QR appends including the empty
root; warm/measured count sequences match exactly. Instrumented full means,
independent quality and all non-timing report fields match each variant's own
primary run. Counting runs and their timings are excluded from the comparison.
[Counts and provenance](results/qr_rebuild/diagnostics/rotations/rotations.json).

## Functional verification

Three new private tests compare grouped rebuilds and subsequent incremental
extension with an independent dense reference and the original quadratic energy.
They cover sizes 4/7/16, disconnected supports, reversed diagonal rows, empty
residual-only rows, scales `1e-100 / 1 / 1e100`, exact `2^-800 / 2^800` models,
invalid dimensions, nonfinite residuals, rank loss, reuse rejection and fill limits.

On the disconnected test supports, nalgebra SVD can leave an original gradient
around `1e-10`, even with tighter deflation. The test oracle starts with bounded
SVD, then applies up to three independent dense SPD corrections to its original
gradient. These normalized test matrices have a strong diagonal and are well
conditioned. Both oracle and candidate must have original gradient below `1e-12`;
solution and energy tolerances stay `1e-11`. This is test-only oracle refinement,
not a production fallback or relaxed benchmark gate.

The 300-arrival cached validator independently checks original energy/gradients
at every prefix and twelve AD/dense-QR checkpoints. It passes with 258 cached,
30 interval, twelve forced and zero threshold updates. After refreshed gradient
maximum is `4.914e-7`, checkpoint pose gap `8.267e-7`, final storage 29,511 entries.
Cached traces are **not bit-identical** after row permutation; their maximum
before/after original-cost gap is `5.906e-14`. Cached gradient can reach `0.01003`,
so this is functional validation, not a native speed comparison.
[Cached record](results/qr_rebuild/diagnostics/cached_validation.json).

All 24 reconstructed source stages match archived fingerprints with zero-fuzz
patch application. The immediate predecessor is rebuilt and exercised; older
stage builds are skipped in source-only checks.
[Rollback proof](results/qr_rebuild/diagnostics/rollback_stages.json).

Validation passes **233 unit/integration tests + 18 doctests**, nine Python tests,
Clippy with warnings denied, formatting and documentation with warnings denied.
[Final checks](results/qr_rebuild/diagnostics/validation_checks.json).

## Reproduction

Finish builds before timing, select an allowed CPU and use new output directories.
The helper reverses only [qr_rebuild.patch](qr_rebuild.patch) for the immediate
predecessor and reverses it first when reconstructing earlier stages.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-rebuild \
  --output /tmp/ropt-before-qr-rebuild
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_rebuild.py \
  --before /tmp/ropt-before-qr-rebuild/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-rebuild \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-rebuild-comparison
```

Repeat with 100 arrivals and a fresh independent 300 output. Run both
`validate_isam_se2 300` executables for cached accuracy. To rebuild the separate
rotation-counting snapshots with fresh targets and verify them against an
accepted primary comparison:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/profile_qr_rebuild.py \
  --before-source /tmp/ropt-before-qr-rebuild \
  --primary /tmp/qr-rebuild-comparison --output /tmp/qr-rebuild-rotations
```

CI has no timing threshold.
Subjective maturity remains 75%; the measured performance win does not fill the
Bayes-tree and field-validation gaps.
