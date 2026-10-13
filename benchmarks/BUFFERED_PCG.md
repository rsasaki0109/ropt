# PCG workspace reuse

The exact-reference sweep exposed a performance gap at 300 poses. A separate
diagnostic found that graph residual/Jacobian evaluation took only about 6–7%
of solve time; most time was outside that evaluation, in linear-system work.
The block-CG solve made 6,857,251 successful allocation/reallocation requests.
This motivated optimizing the PCG/operator/preconditioner hot paths before
changing differentiation or solver tolerances.

## Implementation and contracts

`LinearOperator::apply_into` and `Preconditioner::apply_into` fill caller-owned
vectors. Their default methods delegate to the existing allocating `apply`, so
existing implementations continue to work. The full sparse normal operator,
scalar Jacobi and block Jacobi override them without per-application allocation.
PCG prepares its work vectors once per linear solve and reuses them for operator
results, preconditioning, search directions, residual checks and restarts.

The normal operator computes each CSR row's dot product and immediately
accumulates its transpose contribution. This evaluates `J^T*(J*x)` without an
intermediate residual-sized vector or a stored normal matrix. Block Jacobi
copies the residual into the destination once and uses Cholesky `solve_mut`
on each block view, rather than allocating an owned RHS and solution per block.
No inverse, extra damping, approximate derivative, unsafe library code or relaxed
convergence criterion is introduced.

Implementations must fill every output entry. PCG still checks output dimensions
and finiteness after overridden methods, propagates errors, rejects nonpositive
curvature/preconditioned inner products, recomputes the true linear residual
before accepting convergence and restarts when a recursive residual is too
optimistic. Buffer contents after an error are unspecified. Generic legacy
operators/preconditioners can still allocate through the compatibility defaults;
the block-Schur reduced operator currently uses that default. Zero-allocation
claims apply only to the prepared built-in full normal/Jacobi/block-Jacobi
applications, not graph evaluation, an entire solve or every custom backend.

New integration tests compare reused outputs with independent dense products
and Cholesky, test repeated overwrites of poisoned buffers, verify PCG uses
buffered overrides, check malformed/nonfinite overrides and error propagation,
and preserve legacy implementations. Existing robust BA, Schur, covariance,
fixed-variable, rejection and failure tests remain enabled.

## Allocation diagnostic

```sh
cargo build -p ropt --example profile_pose_graph --release --locked
taskset -c 0 target/release/examples/profile_pose_graph /tmp/w100-prefix300.g2o
```

The profiling executable uses two separate solves: one with graph evaluation
and retraction timers, another with allocation counting. Instrumented allocation
timing is not reported as a benchmark. Its System allocator wrapper is isolated
to this example and forwards layouts/pointers unchanged; the library does not
install a global allocator. Counters record successful allocation, zeroed
allocation and reallocation requests, including the full requested size of a
reallocation. They do not measure freed bytes, live/peak memory, resident memory
or allocator fragmentation. The disabled-counter check remains in the diagnostic
binary's timing pass; use the standard timing executable for performance results.

After preparing a Jacobian, damping and preconditioners, the diagnostic separately
checks 100 normal-operator, 100 scalar-Jacobi and 100 block-Jacobi buffered calls
make exactly zero allocation requests. CI runs this check on the offline loop.

The [before/after profiles](results/buffered_pcg/profile300) on the 300-pose
derived workload use the same profiler and single-threaded solve options:

| Backend | Allocation requests before | After | Requested bytes before | After |
| --- | ---: | ---: | ---: | ---: |
| Scalar CG | 1,469,648 | 1,436,466 | 470,373,632 | 119,950,208 |
| Block CG | 6,857,251 | 1,361,691 | 558,331,320 | 115,322,808 |

Block CG reduces request count by about 80%. Scalar request count drops only
slightly because graph/AD evaluation still makes many small allocations, while
the removed iterative vector allocations account for much of its requested-byte
traffic. Neither result establishes an 80% reduction in peak memory.

## Reproduce the alternating comparison

The saved [optimization patch](pcg_buffer_reuse.patch) restores the original
allocating PCG and built-in allocating `apply` paths when reversed. The baseline
retains the new buffered APIs but does not use them in its PCG numerical path.
The builder copies a benchmark-only source snapshot outside the checkout and
reverses that patch there. It does not modify user files, create a Git worktree,
or alter the main test suite. Build both binaries with the same locked dependencies
and default release profile. The offline build requires an already populated
Cargo cache.

```sh
cargo build -p ropt --example benchmark_pose_graph --example profile_pose_graph --release --locked
python3 benchmarks/make_pcg_baseline.py --output /tmp/ropt-pcg-before
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_pcg.py \
  --before /tmp/ropt-pcg-before/target/release/examples/benchmark_pose_graph \
  --before-source /tmp/ropt-pcg-before \
  --after target/release/examples/benchmark_pose_graph \
  --input /tmp/w100-prefix300.g2o --output /tmp/ropt-pcg-paired --rounds 8
```

The dataset and explicit TORO/equivalence handling are documented in the
[exact-reference workflow](gtsam_exact/README.md). Outputs must be new directories
outside the checkout. The builder was exercised end to end and its source
fingerprint matched the measured baseline. The reversed patch freezes this
specific comparison; later source changes may require a new reviewed baseline.

Eight rounds alternate which variant runs first and which backend runs first,
yielding 32 timed solve samples. Each process makes one extra untimed consistency
reference solve and no additional warmup rounds. Input parsing, graph construction,
initial cloning, output, validation and process startup are outside the binary's
timer. Both variants use the same input/state, default nonlinear LM, CG relative
tolerance 1e-10 and absolute tolerance 1e-12. Allocator profiling is disabled in
these timing binaries. Every sample is retained. Environment, source/executable
fingerprints, patch hash and raw times are in the
[paired report](results/buffered_pcg/paired300/report.json) and
[CSV](results/buffered_pcg/paired300/samples.csv).

| 300-pose backend | Allocating median ms | Buffered median ms | Reduction |
| --- | ---: | ---: | ---: |
| Scalar CG | 731.7904 | 413.9940 | 43.4% |
| Block CG | 822.2741 | 414.9083 | 49.5% |

All four candidates pass the unchanged independent cost, gradient, anchor and
pose-agreement gates. Their costs remain 1.05375815353753, and both variants
retain the same nonlinear/linear iteration counts: scalar 17/11,055 and block
16/9,144. The measured reduction therefore comes from implementation work,
with no changed stopping criteria or reduced iteration count. Shared-cloud
frequency/co-tenancy remain uncontrolled; these results describe this workload,
machine and build, not a general speed or memory guarantee.

## Updated exact-reference sweep

The [30](results/buffered_pcg/poses30/report.json),
[100](results/buffered_pcg/poses100/report.json) and
[300](results/buffered_pcg/poses300/report.json) reports repeat the native-GTSAM
sweep with 10 samples per candidate after three warmups. All common accuracy
gates pass. These 120 measurements are separate from the paired comparison,
and the earlier reports are preserved as historical runs.

| Poses / edges | ropt dense ms | ropt scalar CG ms | ropt block CG ms | GTSAM exact ms |
| --- | ---: | ---: | ---: | ---: |
| 30 / 59 | 3.2646 | 3.0749 | 3.2140 | 1.5100 |
| 100 / 300, derived | 70.2849 | 32.1135 | 32.4644 | 8.2476 |
| 300 / 721, derived | 2140.8906 | 412.5303 | 404.7591 | 93.5247 |

The largest problem is a derived between-factor subgraph, not the full source
SLAM task. Native GTSAM remains faster in these configured runs; at 300 poses,
the fastest measured ropt backend takes about 4.3 times as long. The relative
gap from different historical runs is not itself a paired improvement estimate;
use the alternating ropt comparison above to assess this change. Allocation
and phase results suggest further profiling of matrix-free iterations and graph/
AD allocation, alongside sparse direct solvers and broader datasets. No ropt
SOTA or general superiority claim follows from these measurements.

A subsequent [sparse LDLᵀ comparison](SPARSE_LDL.md) implements the direct
backend and records a separate sweep; this page retains the PCG measurements.
