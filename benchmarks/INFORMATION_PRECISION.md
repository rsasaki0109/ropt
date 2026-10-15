# Preserving weak directions in information whitening

The optional sparse QR/GN replay now completes all **1,227 updates** of the
pinned Intel MIT1228 graph and passes every original-information gradient,
fixed-root and reported-cost gate. No observations, information entries, seeds,
iteration limits or tolerances were changed. This fixes the cost-verification
failure recorded in [the preceding sparse QR replay](SPARSE_QR.md).

## Cause and implementation

The information matrix on edge 160–161 has a second Schur pivot of approximately
`11.14896380301371` against a diagonal of `9.168262482076782e9`: a relative pivot
of `1.216e-9`. Ordinary Cholesky subtraction loses precision in the weak direction.
Keeping only rounded square-root coefficients also loses information when their
products nearly cancel.

`NoiseModel::from_information` retains ordinary arithmetic for ordinary pivots.
If ordinary Cholesky fails, or a squared relative diagonal pivot is below `2^-26`,
it computes a compensated Cholesky factor using two binary64 components per
coefficient. Residual and Jacobian products retain both components until their
final binary64 outputs. The original information is preserved; no jitter is
added. Nonpositive pivots are rejected. Explicit `sqrt_information` and covariance
construction keep their existing behavior.

This uses finite compensated arithmetic, **not arbitrary precision** or a
general condition-number certificate. Extreme underflow/overflow and errors in
user residuals remain limitations. Final residuals, Jacobians and solvers still
use binary64; forming a new matrix from the rounded Jacobian does not recover
the internally retained low coefficient components.

The independent checker had a separate cancellation issue in `rᵀ Omega r`.
[information_precision.py](information_precision.py) computes this original
quadratic form, its weighted residual and normalized-gradient column energies
with 70-digit Decimal arithmetic for small-pivot matrices. Decimal starts from
the exact binary64 input entries and never uses either library's whitening.
Ordinary matrices keep the historical float operations. The mathematical
criteria and thresholds are unchanged.

For the frozen residual on edge 160–161 at arrival 166:

| Calculation | Half weighted squared residual |
|---|---:|
| Ordinary whitening | 0.0012039338920490875 |
| Original quadratic form, ordinary float | 0.0012039340286823853 |
| Original quadratic form, 70-digit Decimal | 0.0012039340794135457 |
| Compensated whitening | 0.0012039340794135457 |

Both optimizer whitening and independent checking needed improved arithmetic.
The exact-binary eigenpair test additionally checks energy and gradient across
power-of-two scales from `2^-400` to `2^400`; the answer does not depend on a
second floating Cholesky implementation. Tests cover invalid matrices and
successful scratch reuse after residual and Jacobian overflow.

## Complete recorded-data validation

[Report](results/information_precision/mit1228/report.json),
[metadata](results/information_precision/mit1228/metadata.json) and complete
compressed estimates retain every prefix. This is a single cold functional
replay on CPU 0 with one thread, **not a paired timing campaign**.

- ropt passes all quality gates; maximum normalized gradient is `7.263e-8`
  against `1e-7`, maximum reported-cost difference `8.072e-12`.
- Every prefix agrees with the retained native GTSAM 4.2.1 reference: maximum
  mixed world-pose norm gap `2.037e-7` against `2e-6`, original cost gap
  `1.102e-11` against `1e-9 * max(1,cost)`.
- The native reference still fails reported-cost verification at 65 prefixes
  (166–230) under the more accurate oracle. Maximum difference is `5.933e-10`.
  Its configuration and outputs are unchanged. **No comparative speed ratio
  is reported** while the reference remains ineligible.

Reported-cost tolerance remains `1e-10 * max(1,cost)`, the root is exact,
and the normalized gradient uses original weighted columns. Pose norm mixes
metres and radians. Raw gradients are also retained. These numbers describe
this recorded SE2 workload and explicit sparse QR/GN full-refresh policy.

The ordinary Intel943 LDL/LM replay passes all quality gates and reproduces
every estimate and non-timing report field from the archived primary exactly.
[Functional regression](results/information_precision/default943_validation.json).
Historical speed campaigns, failure records and hero media remain frozen.
Their earlier checker results are historical measurements, not replaced by
these new precision checks.

## Reproduction

```sh
cargo test -p ropt --test information_precision --locked --offline
cargo build -p ropt --example benchmark_sparse_qr_isam_se2 \
  --example diagnose_information --release --locked --offline
python3 benchmarks/real_se2_stream.py \
  --input benchmarks/results/real_isam2/intel-mit1228-original.g2o \
  --output /tmp/precise-mit1228.stream
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  target/release/examples/benchmark_sparse_qr_isam_se2 \
  /tmp/precise-mit1228.stream /tmp/precise-mit1228 1 0
python3 benchmarks/check_real_replay.py \
  --input benchmarks/results/real_isam2/intel-mit1228-original.g2o \
  --candidate /tmp/precise-mit1228 \
  --reference benchmarks/results/information_precision/mit1228/native \
  --output /tmp/precise-mit1228-quality
```

Consult `eligible_quality` in the JSON: the checker's successful exit only means
assessment completed. It deliberately retains failed prefixes.

[information_precision.patch](information_precision.patch) records the isolated
production change. `make_ldl_baseline.py --before-information-precision`
restores its immediate predecessor. All **27** zero-fuzz rollback source stages
match archived fingerprints; older builds were skipped for these source-only
checks. [Rollback records](results/information_precision/rollback_stages.json)
and [verification](results/information_precision/validation_checks.json).

Verification passes **251 Rust unit/integration tests + 18 doctests**, 18 Python
tests, formatting, Clippy and rustdoc with warnings denied.

Subjective maturity stays **75%**. Bayes-tree selective updates, broader 3D/VIO/BA
comparisons and production validation remain unfinished.
