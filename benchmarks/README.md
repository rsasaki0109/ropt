# Reproducible small pose-graph benchmark

[Gaussian Bayes-tree updates](BAYES_TREE.md) add optional cached conditional-path
re-elimination. Measurements compare the same build's existing incremental QR,
with complete original-model checks and alternating paired Gaussian streams.
Nonlinear iSAM refreshes remain full batch solves.

This measures complete optimization calls on the pinned public GTSAM
`w100_30.g2o` sample: 30 poses, 59 constraints, 177 residuals, 87 active tangent
coordinates, vertex 0 fixed exactly at its input pose, squared loss. The residual
is the translation and principal angle of `Z^-1*(Xi^-1*Xj)`, with objective
`0.5*sum(r^T*Omega*r)`. See the [dataset validation](../docs/validation/pose_graph_2d.md)
for provenance and information-matrix ordering. The dataset remains external.

A later [exact-Jacobian C++ GTSAM comparison](gtsam_exact/README.md) now passes
the common quality gate at 30, 100 and 300 poses. This page preserves the earlier
wheel baseline and its failed gate; the two configurations must not be conflated.
The subsequent [PCG workspace comparison](BUFFERED_PCG.md) measures a checked
implementation improvement and preserves both the earlier and updated runs.
The subsequent [sparse LDLᵀ comparison](SPARSE_LDL.md) adds a direct backend
and repeats the common quality gates without replacing these historical runs.
The latest [reusable LDL and analytic SE2 comparison](REUSED_LDL.md) includes
alternating measurements against native GTSAM QR and Cholesky configurations.
The [large-graph validation](LARGE_POSE_GRAPH.md) extends eligible measurements
to 1,000 poses and retains the failed 3,000-pose accuracy case.
The subsequent [stable-objective validation](STABLE_OBJECTIVE.md) resolves that
accuracy failure while preserving the earlier diagnostics.

The sample is too small to establish scalability or SOTA. Memory is not measured.
This adds a way to measure solver configurations and reject inaccurate results;
it does not establish a performance advantage over GTSAM, Ceres or g2o.

## Reproduce

Run from the repository root. Rust 1.99.0 and Python 3.12 were used. No new Rust
dependency or C++ runtime is required by ropt. GTSAM is an optional external
benchmark dependency, installed in a separate virtual environment.

```sh
curl -fL https://raw.githubusercontent.com/borglab/gtsam/a74146dbea788ea25828041065d82674d6430ada/examples/Data/w100_30.g2o -o /tmp/w100_30.g2o
cargo build -p ropt --example benchmark_pose_graph --release --locked
python3 -m venv /tmp/ropt-benchmark-venv
/tmp/ropt-benchmark-venv/bin/python -m pip install -r benchmarks/requirements.txt
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  /tmp/ropt-benchmark-venv/bin/python benchmarks/pose_graph.py \
  --input /tmp/w100_30.g2o --ropt target/release/examples/benchmark_pose_graph \
  --output /tmp/ropt-benchmark-results --repeats 30 --warmup 5
```

The runner verifies SHA256
`69d9959b6e6c530ad268b4d9999383acbd262b54319fd830e1d3075142c73343`
before evaluating input. `taskset` is optional on systems without Linux affinity;
the actual affinity is recorded. Output must be a new directory outside the
checkout; use a different path on subsequent runs. The Rust timing executable
refuses debug builds and existing output directories.

With the pinned GTSAM wheel and settings below, the full runner exits **1** after
saving its report: the native reference fails the common quality gate. This is
an observed reference-quality limitation, not a missing dependency or discarded
measurement. `report.json` marks it ineligible. Do not interpret its raw timing
as an equal-accuracy speed comparison.

To measure only ropt, add `--without-gtsam`; then the Python runner needs only
the standard library and can be called with `python3`. For an offline functional
check using the ropt-authored four-pose loop, run:

```sh
python3 benchmarks/check_smoke.py target/release/examples/benchmark_pose_graph
```

CI builds this release example and runs the offline check. It checks the four
default backends plus both explicit sparse-LDL orderings, comparing
solutions against scalar objective/central-difference calculations, sample counts,
fixed poses and protection against overwriting existing results. It does not
download public data, install GTSAM, or assert timing thresholds.

## Measurement and accuracy rules

- Each ropt backend uses the same graph and initial state. Five complete warmup
  solves per backend are discarded, followed by 30 measured solves. The first
  backend rotates each round. An extra untimed dense solve supplies a consistency
  reference. All graphs are constructed before timing.
- `Instant` measures `solve_local`, including initial evaluation, forward AD,
  whitening, matrix assembly, linear solves, damping trials, preconditioner
  construction and report construction. Initial-state cloning, input parsing,
  graph construction, output serialization and independent validation are
  excluded. Final report destruction is excluded; intermediate solver cleanup
  is part of the call. This is a solve-call benchmark, not just a linear solve.
- ropt uses default LM: 100 maximum trials, initial damping 1e-3, gradient/step
  tolerances 1e-10 and cost tolerance 1e-12. Both CG modes use 10,000 maximum
  iterations, relative tolerance 1e-8 and absolute tolerance 1e-10. True linear
  residuals are checked. `ropt/options.txt` records every backend's full options.
- GTSAM uses native `BetweenFactorPose2`, full Gaussian information matrices,
  exact `NonlinearEqualityPose2` at vertex 0, COLAMD multifrontal QR and LM with
  initial damping 1e-3, 100 maximum iterations, absolute/relative error-change
  tolerances 1e-12 and total-error threshold 0. Its other defaults are recorded
  in `gtsam-options.txt`. No Python custom factor or numerical Jacobian is used
  in its optimization. Optimizer construction and `optimize()` are timed together;
  copying initial `Values`, graph construction, validation and serialization are
  excluded. Python/native binding overhead in these calls is included.
- GTSAM and ropt have different retractions, damping policies, stopping rules,
  derivatives and linear-solver implementations. GTSAM optimizer iterations are
  reported separately from ropt LM trial counts. The libraries run sequentially,
  rather than in an interleaved paired experiment. Even passing all quality
  gates would establish only these particular configured runs.
- Scalar Python arithmetic independently evaluates the original input edges.
  Each first solution must preserve the anchor exactly, have finite coordinates,
  lie within 1e-9 of the previously validated cost 0.08290751089989, have absolute
  stored-coordinate central-difference gradient infinity norm at most 1e-5
  (step 1e-6, anchor excluded), and differ from the dense solution by at most
  1e-6 per pose in translation/wrapped-angle norm. The cost reference and dense
  pose agreement are regression checks, not an independent global-optimality
  oracle. The independent gradient adds a stationarity check.
- Every measured ropt solve is checked for convergence, cost consistency,
  anchor preservation and pose agreement outside the timer. GTSAM repeated
  solutions are also checked outside the timer. Raw timing samples are saved
  with no outlier deletion. Statistics include median, mean, standard deviation
  and minimum/maximum; no speed ratios are generated.

`metadata.json` and `report.json` record CPU, affinity/quota, OS, thread-variable
settings, Rust/Python/package versions, dataset checksum, release-binary hash,
GTSAM native-module hash, Git HEAD/dirty state, a fingerprint of Rust sources and
Cargo manifests/lockfile, and the Python driver hash. Shared-cloud scheduling and
CPU frequency are uncontrolled. Record a fresh result when changing code or
conditions; do not reuse these timings as stable machine-independent numbers.

## Recorded run

The [report](results/w100_30/report.json),
[ropt raw samples](results/w100_30/ropt-samples.csv),
[GTSAM raw samples](results/w100_30/gtsam-samples.csv) and both option files
record the 2026-10-12 run on an AMD EPYC 9V74, Debian 13 / Linux 6.18.44,
Rust 1.99.0, default Cargo release profile. CPU affinity was `{0}` and the
three recorded thread variables were set to 1. The checkout was dirty; the
report identifies the measured source fingerprint rather than pretending the
old HEAD alone reproduces the implementation. Optimized poses remain external
under the generated output directory; no third-party dataset is vendored here.

| ropt backend | Median solve ms | LM trials | Total CG iterations | Quality gate |
| --- | ---: | ---: | ---: | --- |
| Dense Cholesky | 3.4422 | 7 | 0 | Pass |
| Dense QR | 11.5485 | 7 | 0 | Pass |
| Scalar Jacobi CG | 4.3528 | 7 | 443 | Pass |
| Block Jacobi CG | 5.0173 | 7 | 406 | Pass |

All four costs were 0.08290751089989, independent gradient infinity norms were
about 1.5e-9, and maximum pose difference from dense was at most 2.12e-11.
In this small sample, block Jacobi reduced CG work but increased median solve
time. This demonstrates why iteration counts alone do not establish speed.

The GTSAM 4.2.1 wheel's native run preserved the anchor and matched the same
scalar objective both initially and finally, but ended at cost
0.08291675444042 with independent gradient infinity norm 3.5769e-3 and maximum
pose difference from dense 1.0793e-3. It fails the cost, gradient and pose gates.
Its five optimizer iterations cannot be equated with seven ropt LM trials.

An untimed diagnostic compares this wheel's native factor linearization with
central differences of that factor's own `unwhitenedError` under GTSAM Pose2
retractions. Rotated, off-solution probes show maximum Jacobian entry difference
1.247194. Residual-coordinate probes agree to 1e-12. This identifies a derivative
discrepancy in this exact native baseline; it does not establish behavior for
other GTSAM builds, factor types or problems, or isolate a sole cause of its
stopping behavior. No cross-library speed conclusion is drawn from this run.

The exact-Jacobian native configuration now has validated derivatives and a
passing common quality gate; see its comparison above. Remaining work includes
larger complete public graphs/BA datasets, representative
robust/noisy cases, peak-memory measurement and comparable Ceres/g2o runs.

## Rejected refinement termination candidate

[Unused next-step solves](REFINEMENT_STEP.md) were skipped after final refinement
reaches its cap or gradient stop. Eighteen paired allocation configurations
(180 refreshes) preserve callbacks, refinement counts and result fingerprints.
At 128 variables the editable-tree refresh saves **2 allocation requests / 2 KiB**.
[Complete-stream timing](REFINEMENT_STEP_TIMING.md) finds **8.2% higher observed
median, 1/6 faster pairs**. Adoption is withdrawn and production is restored.
All 12,528 timed arrivals preserve exact poses/non-timing work and pass unchanged
accuracy/native agreement checks. The candidate's 54-stage recipe stays frozen;
the live builder returns to the preceding 53 stages.

## Fresh native iSAM2 comparison before refinement termination change

[Mixed-AD selective ropt versus freshly built GTSAM 4.2.1](CURRENT_NATIVE_ISAM2.md)
uses two planned six-pair complete CSAIL campaigns, with no overlapping builds
or tests. Stream medians are **13.739 / 16.865 s** for ropt and
**25.609 / 26.878 s** for the accuracy-targeted native reference:
**1.86x / 1.59x**, with **6/6 / 5/6** faster pairs. All **25,056** timed arrivals
pass the unchanged original-information and inter-library agreement gates.
Timing varies substantially; the final confirmation pair favors native GTSAM.
The policies differ, and native fully relinearizes with timed empty updates to
the shared target; this is not a default-iSAM2 or general speed claim.
Arrival tails are mixed: confirmation P99 is **209.910 versus 179.719 ms**,
and both observed maxima favor native.
Native precision tests and 27 Python checks pass. The guide retains the build
provenance, raw samples, source snapshots and independently recomputed audit.

## Historical kernel reproduction

The [cache append-reservation candidate](CACHE_RESERVE.md) saves **one
reallocation / 8,184 requested bytes** on a 512-factor graph, but [complete CSAIL
timing](CACHE_RESERVE_TIMING.md) has **2.0% / 2.0% higher observed stream medians
(2/6 / 3/6 faster pairs)**. Adoption is withdrawn and production is restored to
the preceding implementation. All 25,056 timed arrivals preserve exact poses/work
and unchanged gates. Candidate source and its original 48-stage reconstruction
recipe are archived; the live builder returns to its preceding 47-stage chain.
See the timing guide for candidate reproduction and restored-source audit.

[Editable-tree recovery](TREE_REFINEMENT.md) removes the extra global QR rebuild
by using the committed editable tree's frozen factor for refinement. Two
complete CSAIL six-pair campaigns have **4.0% / 4.2% lower stream medians**, with
**5/6 / 6/6** faster pairs; both meet the predefined criterion. All 25,056 timed
arrivals pass unchanged accuracy/native-agreement gates. Complete MIT passes
functional gates. Arrival latency is mixed; the confirmation maximum rises
**73.607 → 81.491 ms**. Rounding changes means slightly; fresh native timing and
peak memory are unmeasured. Restore the preceding kernel with
`--before-tree-refinement`, or the pre-gradient-API kernel with
`--before-tree-gradient`.

The preceding [scheduled cache refresh](CACHE_REFRESH.md) skips the old factor-row
reference vector on forced/interval rebuilds, saving one allocation and **4,096
requested bytes** in a 512-factor forced refresh. Failed-refresh cache/schedule
regressions pass for both selective backends; complete MIT preserves exact
poses/work and unchanged gates. Rust 334 + 2 benchmark tests, Python 23 and 47
source-only rollback stages pass. The subsequent [complete-stream timing
follow-up](CACHE_REFRESH_TIMING.md) finds **0.9% less / 0.5% more median stream
time (2/6 faster pairs each)**; neither campaign qualifies. All 25,056 timed
arrivals preserve exact poses/work and unchanged gates. Arrival-latency changes
are mixed, including primary maximum **76.144 → 100.977 ms**. Restore the old
copies with `--before-cache-refresh`; native timing figures stay frozen.

The preceding [Gaussian factor capture](FACTOR_CAPTURE.md) copies consecutive CSR
rows without incremental vector growth. A 512-factor, 3-row steady Gaussian
refresh saves **2,560 allocation requests / 159,744 requested bytes**. Complete
CSAIL with identical cap-0 policies has **2.5% / 4.6% lower stream medians (4/6 /
5/6 faster pairs)**; only confirmation meets the criterion, so a repeatable
whole-update speedup remains unestablished. Arrival medians rise slightly, P95/P99
decrease, and observed maximum latency is mixed. All 25,056 timed arrivals and
complete MIT regression preserve exact poses/work and original gates. Rust
333 + 2 benchmark tests, Python 23 and 46 source-only rollback stages pass.
Restore row-iterator capture with `--before-factor-capture`. Native timing figures
and library defaults are unchanged.

The [full-stream phase diagnosis](STREAM_PHASES.md) retains cold/warm CSAIL and
cold MIT budgets with exact pose/work agreement. Full recovery consumes **38–46%**
of recorded time, direct probes **20–22%**. The subsequent [recovery-cap policy
comparison](RECOVERY_CAPS.md) uses one uninstrumented binary for caps 1 and 0.
The additional isolated CSAIL campaign takes **7.2% less total update time (5/6
faster pairs)**; all 37,584 timed arrivals across three campaigns pass unchanged
original-information and archived-native gates. Initial campaigns and their
overlapping archival work remain disclosed. The subsequent [complete MIT
comparison](MIT_RECOVERY_CAPS.md) takes **12.4% / 10.7% less total update time
(6/6 faster pairs each)**; all 29,448 timed arrivals pass the same gates. Pooled
median arrival latency rises approximately 1%, P95/P99 decrease, and maximum
latency is mixed (confirmation **123.898 → 131.090 ms**). Both MIT campaigns run
without overlapping separate archival/audit work. Library defaults and native
timing figures are unchanged.

The latest [stack AD input wrappers](INLINE_AD_INPUTS.md) save **1,024 allocation
requests / 65,536 requested bytes** per successful 512-factor all-AD/chunk-8 probe.
**12/16 synthetic probe cases** meet the speed criterion in both campaigns;
ColumnNormalized 2,048-factor all-AD/chunk-1 queries take **16.8% / 16.3% less time
(6/6 faster pairs each)**. Complete CSAIL medians are 1.1% faster / 0.04% slower,
with 2/6 faster pairs each; repeatable full-update speedup remains unestablished.
All 25,056 timed arrivals and complete MIT regression retain exact poses/work and
unchanged gates. Rust 331 + 2 benchmark tests/Python 23 and 45 source-only rollback
stages pass. The rejected strided-write proposal and all measurements are retained.
Restore heap wrappers with `--before-inline-ad-inputs`; native figures stay frozen.

The preceding [direct AD block Jacobians](BLOCK_JACOBIAN.md) save **1,024 allocation
requests / 53,248 requested bytes** per successful 512-factor all-AD query by
removing the flat tangent Jacobian and zero input vector. Analytic-only allocation
totals are unchanged. Complete CSAIL primary is **2.3% slower (3/6 faster pairs)**,
confirmation **1.8% faster (4/6)**; repeatable full-update speedup remains
unestablished. All 25,056 timed arrivals and complete MIT regression retain exact
poses/work and unchanged gates. Rust 329 + 2 benchmark tests/Python 23 and 44
source-only rollback stages pass. Restore the flat adapter with
`--before-block-jacobian`; native timings stay frozen.

The preceding [dense probe indexing](DENSE_PROBE.md) saves **83 allocation requests /
22,944 requested bytes** per successful 512-factor query by indexing contiguous
variable keys in a bounded vector; sparse and partial key spaces retain trees.
Exact result bits/work pass for analytic/mixed/AD diagnostics. Complete CSAIL
medians decrease **1.5% / 5.3%**, with only **4/6 faster pairs** in either campaign;
repeatable whole-update speedup remains unestablished. All 25,056 timed arrivals
and complete MIT regression pass unchanged gates with exact poses/work. Rust 327
+ 2 benchmark tests/Python 23 and 43 source-only rollback stages pass. Restore
tree-only indexing with `--before-dense-probe`; native figures stay frozen.

The preceding [streaming compensated probe cost](PROBE_COST.md) removes full residual/
zero vectors, saving two allocation requests and **16 requested bytes per residual**
for successful nonempty probes. Allocation diagnostics preserve exact cost/gradient/
estimate bits and callback counts for analytic/mixed/AD configurations and both
norms. Complete CSAIL update medians are **1.4% / 0.6% slower (2/6 and 4/6)**;
this is a workspace/allocation improvement with a recorded latency tradeoff.
All 25,056 timed arrivals and complete MIT functional regression pass unchanged
gates with exact poses/work. Rust 325 + 2 benchmark tests/Python 23 and 42 source-
only rollback stages pass. Restore full vectors with `--before-probe-cost`.
Native figures stay frozen.

The preceding [validity-preserving Jet negation](JET_NEGATION.md) improves isolated
width-6 owned negation **1.51x / 1.59x**, borrowed negation **1.19x / 1.17x**, with
6/6 faster pairs in both campaigns for either ownership form. Bits and allocation
work match. Complete CSAIL primary improves **5.1% (6/6)**, while confirmation is
**2.0% slower (3/6)**; repeatable full-stream improvement remains unestablished.
All 25,056 timed arrivals and complete MIT functional regression retain exact
poses/work and pass unchanged gates. Rust 323 + 2 benchmark tests/Python 23 and
41 source-only rollback stages pass. Three rejected broader validation proposals
remain archived. Restore generic negation with `--before-jet-negation`; native
timings are frozen. The new performance claim is negation-specific.

The preceding [width-aware owned Jet loops](JET_LOOP_SPLIT.md) reduce width-65
AD evaluation time **24.5% / 24.6%** and width-32 time **16.3% / 18.6%**, with
6/6 faster pairs in both width campaigns and unchanged allocations/result bits.
Small widths show mixed results. Complete mixed-AD CSAIL primary improves 1.7%
(4/6 pairs), but confirmation is **2.9% slower (1/6)**; repeatable complete-stream
speedup is not established. All 25,056 timed arrivals and complete MIT functional
regression pass unchanged gates, with exact before/after poses/work. Rust 322 +
2 benchmark tests/Python 23 and 40 source-only rollback fingerprints pass.
The unconditional split-loop trial is preserved with its negative complete-stream
results. Restore indexed loops with `--before-jet-loops`; native timings remain
frozen. This remains a width-specific diagnostic gain.

The [owned Jet-buffer reuse](JET_BUFFER_REUSE.md) avoids allocating when an
arithmetic operand's derivative vector can be consumed. Complete mixed-AD CSAIL
time falls **1.2% / 4.6%**, with 5/6 faster pairs in both campaigns and exact
before/after complete poses/work. All 25,056 timed arrivals pass unchanged gates.
Separate width diagnostics lower allocations/bytes while retaining a **5.5% slower
65-wide latency case**. Rejected fixed-storage, width-guard and inlining prototypes
remain archived. Complete mixed-AD MIT regression, Rust 322 + 2 benchmark tests,
Python 23 and 39 source-only rollback fingerprints pass. Restore the preceding
kernel with `--before-jet-buffers`. Native figures keep their measured kernels.

The [complete mixed-AD CSAIL replay](REAL_AD_PROBE.md) uses 586 runtime AD /
586 analytic factors with unchanged observations and information. Median update
stream time falls **3.0% / 1.6%** in paired campaigns, but confirmation wins only
4/6 pairs and misses the speed criterion; a repeatable stream gain is not
established. All 25,056 timed arrivals pass unchanged accuracy/native agreement
with exact before/after poses and non-timing work. Complete mixed-AD MIT regression
passes, including AD on the ill-conditioned edge 160→161. This measures the
existing direct-probe feature in a new workload; core fingerprints are unchanged.
Rust 320 + 2 benchmark tests/Python 23 and all-target Clippy pass. Native figures
retain their measured kernels and drivers.

The [mixed/AD probe extension](MIXED_AUTODIFF_PROBE.md) shares tangent-space AD
with factor graphs and eliminates temporary graph/CSR assembly for all supported
iSAM probe recipes. All 16 synthetic query conditions improve in both six-pair
campaigns (**11.1–25.8% / 9.6–26.3%** time reduction), with unchanged result bits,
digests and callback counts. Separate allocation diagnostics reduce calls
8.2–17.5% and requested bytes 24.5–46.8%. These are query measurements; complete
AD streams/native timings are unmeasured. Complete analytic CSAIL/MIT functional
replays pass unchanged gates with identical poses/non-timing work. Rust 320,
Python 23 and all 38 source-only rollback checks pass. Restore the preceding
kernel with `--before-mixed-probe`.

The [direct current-estimate probe comparison](DIRECT_NONLINEAR_PROBE.md)
removes temporary graph registration and global CSR assembly from all-analytic
iSAM probes, preserving exact reduction order, validation and whitening. At that
measured stage, mixed AD recipes retain the original path. Complete CSAIL editable time falls
**8.6% / 8.9%**, winning 11/12 before/after pairs; all 37,584 timed arrivals pass
unchanged gates with identical complete costs, gradients, means and work. Current
full refresh also uses direct probes and remains faster. The complete MIT
functional replay matches exactly. Rust 318/Python 23 checks and 37 source-only
rollback fingerprints pass. Restore the preceding kernel with
`--before-direct-probe`. Native timings/figures retain their measured kernels.

The [cached-incidence comparison](INCIDENCE_CACHE.md) reuses child/factor-owner
lists and updates changed memberships with sorted linear merges. Complete CSAIL
editable time falls **1.9% / 4.5%**, winning 10/12 before/after pairs; all 37,584
timed arrivals pass unchanged gates with identical complete means/work.
The improved policy remains slower than full refresh; that result is retained.
The complete MIT functional replay also matches exactly. Rust 315/Python 23
checks and 36 source-only historical fingerprints pass. Restore the preceding
metadata-sharing kernel with `--before-incidence-cache`.

The [editable metadata-sharing comparison](EDITABLE_METADATA_SHARING.md) reduces
clone-only allocation calls **99.1%** and requested-byte traffic **56.4%** in
an independently compiled 2,048-variable/64-clone diagnostic. Complete CSAIL
latency is mixed (1.021x / 0.991x, 4/6 / 2/6 faster pairs), so no repeatable
stream speedup is claimed. All 37,584 timed arrivals pass unchanged gates and
exact means/work comparisons; the complete MIT functional replay also matches.
Rust 314/Python 23 checks and 35 source-only rollback fingerprints pass.
Restore the preceding canonical-CSR kernel with `--before-editable-cow`.

The [canonical local-clique CSR comparison](LOCAL_CLIQUE_CSR.md) removes global
triplet sorting and per-clique column tree maps from editable Gaussian assembly.
Complete CSAIL stream time falls **16.0% / 11.2%** in primary/confirmation,
winning 12/12 before/after pairs, with identical complete means and non-timing
work records. Both campaigns pass unchanged accuracy gates. The editable policy
remains slower than current full refresh; its negative result is retained.
MIT1228's complete functional replay also preserves means/work exactly. Rust
313/Python 23 checks and 34 source-only historical fingerprints pass. Use
`--before-local-clique-csr` to restore the preceding measured kernel.

The [stationarity-controlled recorded comparison](STATIONARITY_CONTROL.md)
adds explicit nonlinear probes/requested keys and reuses symbolic topology for
same-support replacements and additions. On complete CSAIL, selected global QR
beats current full ropt by **1.12x / 1.11x** in primary/confirmation, winning
12/12 pairs with unchanged gates and explicit timed recovery. Editable-tree
latency remains worse and is archived alongside it. Native poses/costs provide
hashed correctness agreement only; there is no new native speed comparison.
Rust 312/Python 23 checks and 33 source-only historical kernels pass.

The additional [editable Gaussian Bayes-tree mode](EDITABLE_BAYES_TREE.md)
connects selected nonlinear factors to affected clique replacement through
cached separator messages. The synthetic nonlinear example re-eliminates 2 of
1,025 scalar cliques; global metadata/solve work remains. Independent original
quadratic/SVD and mixed-chart checks pass. Source-only rollback checks 32 kernels;
the prior recorded-data speed results remain frozen. New-policy latency is not
yet measured.

The new [threshold-selected nonlinear factor mode](SELECTIVE_RELINEARIZATION.md)
retains unrelated whitened rows and exposes bounded-pass reports. Its Gaussian
QR rebuild is global. Eight independent regression tests and the complete suite
pass; forced-refresh CSAIL/MIT1228 poses and non-timing reports are unchanged.
No native speed claim is made for this new policy; preceding measurements stay
frozen. Source-only rollback now checks 31 historical kernels.

The new [optional sparse QR/GN path](SPARSE_QR.md) completes the MIT1228 replay
that stalled under LDL/LM, with unchanged gradient and native pose/cost agreement
gates. The subsequent [information precision fix](INFORMATION_PRECISION.md)
passes every ropt quality gate; the old native reference remains ineligible.
The subsequent [Cartesian/precision comparison](PRECISE_REAL_ISAM2.md) corrects
native measurement arithmetic and passes every common gate. It records 1.39x
primary and 1.33x confirmation speedups on complete MIT1228, with 12/12 faster
pairs. All preceding failed or unstable comparisons remain archived.

The additional [CSAIL recorded replay](CSAIL_REAL_ISAM2.md) extends this policy
to a different recording: 1,045 poses and 1,172 original constraints, initialized
identically from original odometry. All 1,044 arrivals pass the same gates;
primary/confirmation median speedups are **2.29x / 2.28x**, with **12/12 faster
pairs**. Archived full poses, samples and a standalone comparison figure are
included. This is common-accuracy full refresh, not default iSAM2 latency.

The [recorded Intel replay](REAL_ISAM2.md) uses complete pinned real-data graphs,
an explicitly shared column-normalized gradient criterion and all-prefix native
pose/cost checks. It has a separate adapter and runner; its stopping measure is
different from the historical synthetic absolute-gradient contract below.

The archived [grouped QR rebuild](QR_REBUILD.md) beats the verified native iSAM2
configuration at the unchanged common-accuracy target on the 100/300-arrival
synthetic streams, with a second independent 300 campaign. This is scoped evidence,
not a general SLAM or native default-policy claim.
The preceding [iSAM local-coordinate reuse](ANCHOR_COORDINATES.md) compares the
cached policy, with unchanged independent quality checks and exact full results.
Its approximate cached prefixes do not support a native speed comparison.
The [contiguous QR column checks](QR_COLUMNS.md) retain an earlier archived
before/after/native timing campaign. The preceding [aligned QR reductions](QR_REDUCTIONS.md)
remain an archived comparison. Timing claims in older documents refer to archived
kernels, not arbitrary comparisons against the latest checkout. Restore both
adjacent source stages and verify their fingerprints against the saved metadata
before attributing a timing difference to one historical change. The baseline
helper reconstructs earlier source kernels by reversing later patches first.
Older-stage source checks do not imply that every old kernel was rebuilt in the
latest task; each campaign states which binaries were built and tested.

For example, the archived Schur update-cursor after kernel is restored by
`--before-qr-reductions`, and its before kernel by `--before-update-cursor`.
The rejected normal-pattern helper accepts an explicit predecessor with
`--source`; its documented comparison driver hashes both supplied source trees.
Recorded data and timing gates are preserved rather than replaced by new runs.
