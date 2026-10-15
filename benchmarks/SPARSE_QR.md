# Sparse QR and explicit nonmonotonic Gauss–Newton

The original recorded MIT1228 replay stopped at arrival 166: 200 LDL/LM trials
ended at cost about 2.993 with normalized gradient about 0.0421. Replaying that
single update with Dense QR/LM produces essentially the same result. It is not
simply failure of the normal-equation linear backend: 189 of 200 LDL/LM trials
are accepted but progress is slow under highly anisotropic information.

The new **optional Sparse QR / nonmonotonic GN** path solves that update in four
trials at cost about 0.208366. It also processes **all 1,227 arrivals / 1,228 poses**
of the pinned MIT artifact, without removing observations, changing information,
increasing the 200-trial limit or using optimized seeds. This is a convergence
improvement, not an eligible cross-library performance claim.

## API and numerical behavior

`LinearSolver::SparseQr(SparseQrOptions)` supports ordinary GN and LM. Ordered
Givens QR solves the original weighted sparse Jacobian; LM augments it with
`sqrt(lambda) * max(column_norm, 1)` diagonal rows. No numeric normal matrix,
Cholesky backend fallback or implicit numeric differentiation is used.
The existing symbolic normal-pattern machinery supplies structural elimination
ordering. It stores a graph/pattern, not numeric Gram entries. Its pattern and
fill budget still apply. Ordering is reused while the CSR pattern is identical;
new numeric factors are built for each step.

This QR is **ordered and unpivoted**. Its relative R-diagonal cutoff is not an SVD
rank certificate or a rank-revealing sparse method. Sparse covariance, pivoted
sparse methods, supernodal factorization and a Bayes tree remain missing.
`max_entries` bounds symbolic ordering storage and numeric R entries; it does not
bound peak memory or augmented-Jacobian input storage.

`Options::nonmonotonic_gauss_newton` defaults to false and is invalid for LM.
Opting in permits an uphill GN trial only when the linear model predicts positive
improvement. Every evaluation and solve retains finite, rank and dimension checks.
An uphill trial is recorded with `accepted=true` and negative `actual_reduction`
and gain ratio. It is not a cost-convergence outcome; an unresolved small uphill
step is `NoProgress`. Iteration limits remain unsuccessful termination.
The smoother rejects a capped, unconverged full refresh and preserves committed
state. The ordinary GN rejection rule and all LM adoption/damping rules remain
unchanged.

```rust
use ropt::{Algorithm, LinearSolver, Options, SparseQrOptions};

let options = Options {
    algorithm: Algorithm::GaussNewton,
    linear_solver: LinearSolver::SparseQr(SparseQrOptions::default()),
    nonmonotonic_gauss_newton: true,
    ..Options::default()
};
```

Sparse QR supplies its ordering to `IsamSmoother::with_batch_ordering_reuse()`.
Cached appends preserve external tangent coordinates; a refresh initially
converged without a new factorization extends the previous ordering. The real
replay also enables the existing four final frozen-QR refinements.

## Independent recorded-data checks

The unchanged gates use original information: normalized gradient `1e-7`, exact
root, world-pose norm gap `2e-6`, relative original-cost agreement `1e-9` and
reported-cost agreement `1e-10`, with `max(1,cost)` scales. The first norm mixes
metres and radians. Every arrival's result is retained, including failed gates.

On the completed MIT1228 replay:

- All root and original-gradient gates pass; maximum normalized gradient is
  about `4.3242e-8`.
- Every prefix agrees with native GTSAM poses and original cost; maximum mixed
  pose norm gap is about `2.0473e-7`, cost gap about `7.24e-10`.
- **Reported cost fails the stricter gate at 48 prefixes** for ropt, starting at
  166; maximum difference is about `7.83e-10`. The reference fails at 47 prefixes,
  with maximum difference about `9.91e-10`. Both remain **ineligible** under the
  entire comparison contract. No speed ratio is computed.

The recorded replay is a single cold diagnostic, with no paired timing campaign
or warm-up; its timing records are kept as diagnostics. The existing native
reference configuration and real-data arrival/measurement policy are unchanged.
See [complete quality and agreement](results/sparse_qr/mit1228/report.json),
[metadata](results/sparse_qr/mit1228/metadata.json), and the retained all-prefix CSVs.

The existing default LDL/LM Intel943 replay still passes every original gate.
Complete means and every non-timing report field match the archived primary
exactly, including cost, entries, batch trials, damping and refinement counts.
[Functional regression](results/sparse_qr/default943_validation.json).
This is not a new performance campaign; the earlier 1.22x/1.09x measured claims
still refer to their frozen input, source and executable snapshots.

## Verification and reproduction

Five new Rust tests compare ordered sparse GN and damped LM with independent
dense QR across scales `1e-80 / 1 / 1e80` and all ordering options. They cover rank
loss, damping, budgets, invalid dimensions/options, nonfinite residuals, a stiff
curved constraint, rejected default GN versus explicit uphill trials, successful
convergence, capped unsuccessful termination, cached column extension and
transactional rank-failure rollback.
Two Python tests ensure all failed quality prefixes are retained and truncated
means cannot be called complete/eligible.

Verification passes **247 unit/integration tests + 18 doctests**, 15 Python tests,
formatting, Clippy and rustdoc with warnings denied. All **26** source stages
match archived fingerprints under zero-fuzz rollback; older-stage builds are
skipped in these source-only checks. The final replay retains exactly the full
means and non-timing reports of its independently validated pre-cleanup replay.
[Final checks](results/sparse_qr/validation_checks.json) and
[rollback checks](results/sparse_qr/rollback_stages.json).

```sh
cargo test -p ropt --test sparse_qr --locked --offline
cargo build -p ropt --example diagnose_real_refresh \
  --example benchmark_sparse_qr_isam_se2 --release --locked --offline
# Frozen previous committed estimates at arrival 165 + original seed/edges at 166.
cargo run -p ropt --example diagnose_real_refresh --release --locked --offline -- \
  benchmarks/results/sparse_qr/prefix166/input.g2o /tmp/qr-prefix166 sparse-gn 1e-5
# Full source measurements and provided original vertex seeds.
python3 benchmarks/real_se2_stream.py \
  --input benchmarks/results/real_isam2/intel-mit1228-original.g2o \
  --output /tmp/mit1228.stream
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  target/release/examples/benchmark_sparse_qr_isam_se2 \
  /tmp/mit1228.stream /tmp/qr-mit1228 1 0
python3 benchmarks/check_real_replay.py \
  --input benchmarks/results/real_isam2/intel-mit1228-original.g2o \
  --candidate /tmp/qr-mit1228 --output /tmp/qr-mit1228-quality
```

For a cross-library quality assessment add `--reference` pointing to the saved
native complete outputs (the checker also reads `poses.csv.gz`). Its successful
exit means assessment finished; consult `eligible_quality`, not the process exit
code, for comparison eligibility. This is a diagnostic tool, not a CI speed gate.

[sparse_qr.patch](sparse_qr.patch) records the production change. The baseline
builder's `--before-sparse-qr` reconstructs the immediate predecessor and earlier
stages reverse it first. All source stages are checked with zero-fuzz reversal.

Subjective maturity stays 75%. Better convergence does not eliminate the remaining
precision-verification and production-validation gaps.

The subsequent [information precision fix](INFORMATION_PRECISION.md) resolves
ropt's reported-cost failure with unchanged gates. Its new, separate replay uses
a more accurate original-information oracle; the unchanged native reference still
fails cost verification. The records and measurements in this document remain
the frozen preceding stage.
