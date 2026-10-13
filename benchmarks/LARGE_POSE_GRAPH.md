# Large pose-graph validation and LDL refinement

Increasing the previous 300-pose workload exposed two numerical limits:
an original-system LDL residual failure at 3,000 poses, and pose disagreement
despite small final cost and gradient. The solver now attempts bounded iterative
refinement. Accuracy gates stay unchanged, and failed comparisons are retained.

This page preserves that earlier implementation and its ineligible 3,000-pose
case. The subsequent [stable-objective validation](STABLE_OBJECTIVE.md) resolves
the stopping issue with per-loss reductions and retains both stages.

## Implementation

`SparseLdlWorkspace::solve_into` first solves the computed factors and evaluates
the original CSR operator `r = rhs-(JᵀJ+lambda*D²)*x`. When its norm exceeds
`max(absolute_tolerance, relative_tolerance*norm(rhs))`, it solves those same
factors for `r`, adds the correction and recomputes the original residual.
There are at most three corrections. Every successful checked return meets the
requested tolerance; unresolved residuals still return `LinearResidualTooLarge`.
There is no extra damping, factorization, tolerance relaxation, dense fallback,
inverse or change to the nonlinear objective. Prepared solution/residual buffers
are reused, and `refinement_steps` is exposed in both checked result types.
Unchecked factor solves retain their existing behavior.

A weakly anchored chain regression verifies that a raw factor solution violates
the requested bound and a corrected solution passes the independently evaluated
original operator. Zero-tolerance failure is also retained, and ordinary solves
that already pass return the initial factor solution with zero corrections.
Prepared unchecked refactor/solve loops still request zero allocations.

## Inputs and measurement

Both inputs are explicit **between-factor subgraphs** of the pinned GTSAM 4.2.1
`w10000.graph`; they are not the complete original SLAM problem. The existing
converter retains prefix IDs and within-prefix EDGE2 records, permutes TORO
information entries and explicitly omits EQUIV constraints:

| Poses | Edges | Omitted within-prefix EQUIV records | Converted SHA256 |
| --- | ---: | ---: | --- |
| 1,000 | 3,747 | 410 | `64211be8374b942fedcaad47ac16d455f03e4c011a2a5027ed40c451c8efe7f8` |
| 3,000 | 16,834 | 1,684 | `b9e62bab6b28d75b3fc845033cdd84496c5c81fdb7faca670896bcfc0da347c4` |

`paired_large.py` uses a separate native exact-Jacobian multifrontal QR solve
as the pose-consistency reference. Unlike the previous small sweep, it does not
allocate a dense reference. This is a consistency/stationarity check, not a
global-optimality certificate. Exact anchor preservation, cost agreement within
`1e-9*max(1,abs(reference_cost))`, stored-coordinate central-difference gradient
infinity norm <=1e-5 and pose agreement <=1e-6 are all required. Iteration-limit
termination is ineligible; reported objectives are independently checked.

The numerical gradient uses the same scalar objective and step 1e-6, evaluating
only edges incident to each perturbed pose. Unrelated terms are constant in that
partial derivative. This avoids quadratic work and subtraction of unrelated
large costs; it is neither analytic differentiation nor a looser threshold.
Tests compare full/incident objective differences on correlated edges, loops,
self-edges and several anchor sets, and reject pose disagreement even when
both cost and gradient pass.

The 1,000-pose run uses ropt analytic SE2 factors, three-coordinate block minimum
degree and reused sparse LDL. Its explicit cost tolerance is **1e-15**, versus
the library default 1e-12; gradient/step defaults remain 1e-10. The default cost
tolerance leaves pose disagreement about 1.17e-5, so it cannot support this
comparison. Both native configurations retain absolute/relative error tolerances
1e-12, exact BetweenFactor Jacobians and COLAMD. All three iteration limits are
100. They perform 21 ropt nonlinear trials versus 37 native optimizer iterations.
Stopping/damping/retractions differ; this measures these complete configurations.

Each of twelve rounds runs all three candidates. Rotated/reversed order puts
each candidate in every position four times. Each process discards one solve
before measuring one complete solve. Parsing, graph construction, initial cloning
and validation are outside timers; evaluation, factorization, nonlinear trials
and report construction are inside. Native optimizer construction is timed.
All **36 samples** are retained without outlier deletion. An exploratory run
overlapping another solver process was discarded in its entirety before this
isolated comparison; it is not mixed with the retained samples.

The recorded machine is AMD EPYC 9V74, affinity CPU 0, four-CPU cgroup quota,
single-thread environment settings, Rust release defaults and the existing
GTSAM 4.2.1 exact-Jacobian build without TBB/MKL/native-architecture flags.
Frequency and cloud co-tenancy are uncontrolled. Build/source/executable hashes,
settings, dataset provenance, individual checks and timing dispersion are retained.
Peak/live/RSS memory is not measured. External graphs/optimized poses are not vendored.

## Eligible 1,000-pose result

| Configuration | Median complete solve | Min–max |
| --- | ---: | ---: |
| ropt block LDL + analytic, cost tolerance 1e-15 | **178.08 ms** | 166.34–195.15 ms |
| Native GTSAM exact QR | 1,241.95 ms | 1,126.71–1,399.31 ms |
| Native GTSAM exact Cholesky | 613.76 ms | 556.05–640.98 ms |

Every sample passes every gate. The independent final objective is
7.096253719379179; ropt's maximum stored-gradient norm is 3.30e-11 and pose
difference from the QR reference is 2.13e-8. The configured ropt median is
about 3.45x faster than the faster native configuration on this derived graph.
This is not a general GTSAM/SOTA claim or a benchmark of fixed-lag/iSAM.
See the [report](results/large_pose_graph/poses1000/report.json),
[samples](results/large_pose_graph/poses1000/samples.csv) and recorded options.

## Ineligible 3,000-pose case

Before refinement, the original-system residual was 2.118235684869705e-12
against the unchanged 1e-12 absolute bound and the solve returned an error.
The corrected implementation completes the nonlinear solve with the default
cost tolerance. Its independent cost is 36.75010834401441 and numerical gradient
norm 6.20e-9. GTSAM reaches its original 100-iteration limit; increasing its
limit to 300 yields convergence after 148 iterations for both QR and Cholesky.
Those native solutions agree within 3.22e-9.

However, ropt differs from the converged native QR poses by **9.10e-6**, exceeding
the unchanged 1e-6 gate. Cost and gradient agreement are insufficient here.
A ropt cost tolerance of 1e-14 leaves the same accepted state; 1e-15 produces
`NoProgress` rather than successful convergence. These results are preserved as
[failed diagnostics](results/large_pose_graph/poses3000/report.json), with raw
samples/options and the pre-refinement error. No equal-accuracy speed comparison
is claimed for this case. Resolving objective-roundoff/stopping behavior and
pose accuracy at this scale remains necessary.

## Reproduce

Build the optional native reference and manifest using its
[build instructions](gtsam_exact/README.md), then run from the checkout:

```sh
cargo build -p ropt --example benchmark_pose_graph --release --locked
python3 benchmarks/convert_toro.py --input /tmp/w10000.graph \
  --output /tmp/ropt-prefix1000.g2o --vertices 1000 --between-only
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/paired_large.py --input /tmp/ropt-prefix1000.g2o \
  --ropt target/release/examples/benchmark_pose_graph \
  --gtsam /tmp/ropt-gtsam-reference/reference/gtsam_exact \
  --build-manifest /tmp/ropt-gtsam-reference/manifest.json \
  --output /tmp/ropt-large-results --rounds 12 --cost-tolerance 1e-15
```

Run comparisons separately, without overlapping compilers or other solver runs.
Output paths must be new and outside the checkout. For the 3,000-pose diagnostic,
convert with `--vertices 3000`, run ropt with default cost tolerance, and run the
native executable with final arguments `qr 300` or `cholesky 300`.

`make_ldl_baseline.py --before-refinement --output /tmp/ropt-before-refinement`
builds an isolated benchmark-only source snapshot that reverses
`ldl_refinement.patch`. Its source fingerprint matches the recorded predecessor.
Without that flag, the helper additionally reverses the historical LDL-reuse
patch, preserving reproducibility of the earlier map-based baseline.
