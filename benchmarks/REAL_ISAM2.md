# Recorded Intel replay: ropt and native GTSAM iSAM2

This comparison uses recorded Intel Research Lab SE2 constraints, not generated
trajectories. The complete `intel-vertigo943` graph has 943 poses and 1,837 edges.
An additional `intel-mit1228` artifact has 1,228 poses and 1,483 edges; it is a
different version, not a larger prefix of the same graph. Both were selected
before measuring. Source URLs, Git commits, SHA-256 checksums and structure are
pinned in [real_se2_stream.py](real_se2_stream.py).

Neither file provides acquisition timestamps. Replay follows ascending original
pose IDs; every edge arrives at its later endpoint, retaining file order within
that arrival. This is an **inferred arrival order**, not raw sensor replay. All
original measurements, full information matrices and vertex seeds are preserved;
angles are normalized. There are no optimized seeds, injected false loops,
constraint pruning or truncated graphs in the primary comparison. Pose zero is
fixed exactly to its original value.

## Shared accuracy contract

Information magnitudes differ substantially across these artifacts. On the MIT
version a diagonal information entry reaches about `2.69e12`; native GTSAM also
fails an absolute `5e-7` gradient target early in replay. Absolute gradient depends
on parameter units and observation scale. This experiment therefore declares a
different, shared stopping measure before the primary campaigns:

```
max_i |(Jᵀr)_i| / ||J[:,i]||₂ <= 1e-7
```

`J` and `r` are the original-information weighted right-tangent Jacobian and
residual. The independent Python checker uses `J_iᵀ Ω J_i` directly from original
information matrices, rather than reproducing either solver's whitening.
The raw gradient remains in every quality record. Both solvers internally target
`8e-8`. The objective and observation information are unchanged. This experiment
is **not directly comparable** to previous synthetic absolute-gradient campaigns.

At **every arrival**, both outputs must be finite, preserve the exact fixed root,
agree in original cost to relative `1e-9`, and differ in world-pose norm by at most
`2e-6` (mixed metres/radians). Reported cost must agree with independent original
energy to relative `1e-10`. All checks use `max(1,cost)` as the relative scale.
No failed accuracy run is eligible for a speed ratio. A stationary point is not
a global optimum certificate.

## Implementation and policies

`Options::gradient_norm` explicitly selects absolute (the unchanged default) or
column-normalized convergence. Stable column norms use bounded squared sums,
switching individual extreme columns to `hypot`; no Gram matrix is needed for
this measure. `Iteration::gradient_norm` records the selected measure.

ropt's full-refresh square-root cache can now reuse the sparse batch solver's
elimination permutation with `with_batch_ordering_reuse()`. Its internal mapping
preserves external tangent coordinates, fixed blocks and later column extension.
A refresh that converges without a linear solve extends the preceding permutation.
Public incremental append retains its transaction. Natural-order rebuilding
exceeded the existing million-entry budget at arrival 705 on this recorded graph;
the budget and rank tolerances are not relaxed.

`with_final_refinement(4)` optionally refines the accepted batch solution using
the frozen QR factor. Each correction solves `Rᵀ y = -g`, then `R δ = y`, in the
internal order, without materializing normal equations. This gradient correction
still inherits the conditioning of `RᵀR`; it is not a rank-revealing fallback.
Each accepted step has positive predicted and actual improvement and a smaller
selected gradient. At most eight backtracking trials are allowed per step.
Refinement stops on a small step or at one hundredth of the configured gradient
tolerance. `IsamReport::refinement_steps` counts accepted steps, not all trials.
The cache remains at the batch anchor, which is retained for later cached solves
and `linearized_cost`. The latter is a frozen linear-model cost, not the nonlinear
energy at the refined estimate. An evaluation or rank/fill failure preserves
committed smoother state. Default ordering and refinement policy are unchanged.

The measured ropt policy uses sparse LDL, block minimum degree of size three,
restarts LM damping at `1e-5`, allows 200 trials, and forces full relinearization
at every arrival. Step and cost tolerances are `1e-12` and `1e-15`. Optional order
reuse and four frozen-factor refinements are enabled; damping reuse is disabled.

Native GTSAM **4.2.1** uses QR, Gauss–Newton, zero wildfire/relinearization
threshold, skip one and timed empty updates until the shared gradient target.
Its official correct BetweenFactor Jacobian build is pinned and verified against
the actually linked library. See [NATIVE_ISAM2.md](NATIVE_ISAM2.md) for that build's
derivative validation. The real-data native adapter is separate from the archived
synthetic adapter. This is a complete update at a common accuracy target, not a
comparison of either library's default cached policy.

## Measurement

Each process discards one complete warm-up replay and measures one complete
replay. Six paired rounds alternate ropt/native and native/ropt; a second
independent six-round campaign repeats the same frozen input, source and binaries.
All processes are serialized on CPU zero with thread variables one. Builds,
tests and other solver measurements do not overlap these campaigns.

Timers include update, full owned means and cost/report construction. Native
stationarity probes and additional empty updates are included. Input parsing,
factor recipes, initialization, external validation, serialization and report
destruction are excluded. Allocation counters are disabled; peak memory is not
measured. Cloud frequency and co-tenancy are uncontrolled. All samples, including
slower runs, are retained.

## Results

| Complete 943-pose replay | ropt median seconds | Native median seconds | Native / ropt | ropt faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary, six balanced pairs | 15.009 | 18.260 | 1.217x | 6 / 6 |
| Independent confirmation, six balanced pairs | 15.122 | 16.553 | 1.095x | 6 / 6 |

The primary median is **17.8% lower** for ropt. Standard deviations are
1.114 / 1.356 seconds; ropt ranges 13.800–17.380 seconds and native
16.485–20.270 seconds. Cloud variation is material. Maximum normalized gradients
are `2.6264e-8` (ropt) and `7.9935e-8` (native). Raw gradient maxima are
`1.4693e-6` and `1.2976e-5`, respectively; this is why the stopping metric must be
reported explicitly. Maximum mixed metre/radian pose gap is `4.1073e-7` and
original-cost difference is `1.024e-12`. All 942 prefixes pass every shared gate.

Confirmation has an **8.6% lower** ropt median, with standard deviations
0.420 / 0.383 seconds and ranges 14.085–15.309 / 16.054–17.227 seconds. Do not
attribute differences between campaign medians to code changes. The two campaigns
use identical source, binary and input hashes, and their first complete means
match exactly within each variant. There are **24 measured complete streams /
22,608 timed arrivals**, plus 24 discarded warm-up streams. ropt wins all twelve
pairs across both campaigns under this common accuracy policy.

The separately selected **MIT1228 artifact is not a successful comparison**.
The final ropt configuration completes arrivals 1–165, then reaches 200 batch
trials at arrival 166 with normalized gradient about `0.0421`. Native completes
all 1,227 arrivals and passes the normalized-gradient gate, but its reported
cost disagrees with the independent energy beyond the existing `1e-10` gate on
47 prefixes, starting at 166 (maximum absolute difference about `9.91e-10`).
Neither output is eligible under the complete shared contract. This cold
diagnostic is not a balanced timing campaign and no speed ratio is reported.
It exposes an unresolved ropt convergence weakness and a native cost-verification
precision issue under heterogeneous information magnitudes. The gates are not
relaxed. See
[failure record](results/real_isam2/diagnostics/mit1228/report.json) and retained
logs/options/partial updates in that directory. The 943-pose win does not imply
that both real-data variants are solved successfully.

- [Primary report](results/real_isam2/intel943-primary/report.json),
  [metadata](results/real_isam2/intel943-primary/metadata.json),
  [samples](results/real_isam2/intel943-primary/samples.csv),
  [all update records](results/real_isam2/intel943-primary/updates.csv),
  [independent quality](results/real_isam2/intel943-primary/quality.csv).
- [Independent confirmation](results/real_isam2/intel943-confirmation/report.json).

## Functional verification

Checks pass **242 unique unit/integration tests + 18 doctests**, 13 Python tests,
formatting, Clippy and rustdoc with warnings denied. The full suite ran 241 tests;
the mixed-chart test file then ran three tests, including one newly added case.
New coverage includes unit rescaling of the stopping measure, numeric right-chart
validation of independent column norms, extreme/zero Jacobian coefficients,
nontrivial QR permutations with appended columns, fixed blocks and heterogeneous
Lie charts, a refresh converged without a new ordering, retained pre-refinement
anchors on later cached solves, invalid options and failure rollback.

All **25** reconstructed source stages match their archived fingerprints with
zero-fuzz patch reversal. The immediate predecessor is built with a fresh target
and passes its 25-arrival cached validator; earlier-stage builds are skipped.
The first complete means, source, binary and fixture hashes match between the
two primary campaigns. Records:

- [Validation checks](results/real_isam2/diagnostics/validation_checks.json).
- [Confirmation checks](results/real_isam2/diagnostics/confirmation_checks.json).
- [Rollback stages](results/real_isam2/diagnostics/rollback_stages.json).
- [Exploratory probes](results/real_isam2/diagnostics/exploration/probes.json),
  including failed, slow and accuracy-failing prototypes. Their cold timings
  and separately unfrozen sources are excluded from primary performance claims.

## Reproduction

Download the exact source artifact from its pinned catalog URL, then check its
SHA-256 before replay. Complete builds before timing and use a new output directory.
The native adapter links the verified exact-Jacobian GTSAM installation:

```sh
cargo build -p ropt --example benchmark_real_isam_se2 --release --locked --offline
cmake -S benchmarks/gtsam_real_isam2 -B /tmp/gtsam-real-adapter \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=YOUR_EXACT_GTSAM_PREFIX
cmake --build /tmp/gtsam-real-adapter
python3 benchmarks/gtsam_exact/write_manifest.py --help
# Generate and verify a new manifest for this native adapter and linked library.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_real_isam2.py \
  --ropt target/release/examples/benchmark_real_isam_se2 \
  --native /tmp/gtsam-real-adapter/gtsam_isam2 \
  --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --input YOUR_PINNED_INTEL_G2O --rounds 6 --cpu 0 --output /tmp/real-intel-paired
```

Repeat with a fresh output directory for confirmation. The runner verifies the
native executable, reference sources and resolved GTSAM library against the
manifest, and rejects solver or runner changes during a campaign.
[real_isam.patch](real_isam.patch) records the production change. Reconstruct the
immediate predecessor with `make_ldl_baseline.py --before-real-isam`; earlier
stages reverse this patch first. The predecessor lacks the new normalized-gradient
API, so its synthetic benchmark templates are used for reconstruction.

CI has no timing threshold. Performance conclusions apply to the measured
artifact and explicit accuracy policy. They do not establish superiority on 3D,
VIO, robust outliers, sensor timestamps, other GTSAM configurations, peak memory
or real-time deadlines. ropt still has no Bayes tree or selective clique updates.
Subjective maturity remains 75%.

The subsequent [optional sparse QR/GN path](SPARSE_QR.md) completes MIT1228 and
matches the native poses at every arrival. The strict reported-cost gate remains
unmet for both outputs in that frozen replay. The subsequent
[information precision fix](INFORMATION_PRECISION.md) passes every ropt gate;
the unchanged native reference still fails reported-cost verification.
The historical LDL/LM failure and timing campaigns above
are preserved; they are not replaced by the new functional replay.

The new [Cartesian/precision comparison](PRECISE_REAL_ISAM2.md) additionally
corrects the native whitening and conditioned-root chart. Both complete MIT1228
outputs now pass every original gate. ropt takes 7.947 s versus 11.032 s (1.39x),
confirmed at 8.488 / 11.284 s (1.33x), with 12/12 faster pairs. This separate
campaign uses explicit Cartesian SE2 retraction and the disclosed corrected
native reference; it does not revise any frozen records above.
