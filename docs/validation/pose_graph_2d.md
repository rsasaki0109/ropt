# Public 2D pose-graph sample

This page records the initial file-interoperability and optimization validation.
Later release timings and native GTSAM accuracy diagnostics are documented in
the [benchmark report](../../benchmarks/README.md). Neither establishes
large-scale performance or real-robot evidence.
The input is the public `w100_30.g2o` sample distributed with GTSAM. Its 30 poses
are too small to establish large-scale performance or general convergence.
The dataset remains external; it is not vendored or relicensed by ropt.

## Source and reproduction

Source: [GTSAM w100_30.g2o](https://github.com/borglab/gtsam/blob/a74146dbea788ea25828041065d82674d6430ada/examples/Data/w100_30.g2o).
The revision and checksum pin the exact bytes used here:

```text
GTSAM revision: a74146dbea788ea25828041065d82674d6430ada
SHA256: 69d9959b6e6c530ad268b4d9999383acbd262b54319fd830e1d3075142c73343
```

Run from the repository root with the pinned Rust 1.99.0 toolchain and locked
nalgebra 0.34.2 dependency. The Python checker uses only the standard library.
Output files must be new files; choose different output names when repeating.

```sh
curl -fL https://raw.githubusercontent.com/borglab/gtsam/a74146dbea788ea25828041065d82674d6430ada/examples/Data/w100_30.g2o -o /tmp/w100_30.g2o
sha256sum /tmp/w100_30.g2o
cargo run -p ropt --example pose_graph_g2o --locked -- --input /tmp/w100_30.g2o --dense --output /tmp/w100-dense.g2o
cargo run -p ropt --example pose_graph_g2o --locked -- --input /tmp/w100_30.g2o --output /tmp/w100-sparse.g2o
python3 docs/validation/check_pose_graph.py /tmp/w100_30.g2o /tmp/w100-dense.g2o /tmp/w100-sparse.g2o
```

## Problem and objective

The sample contains `VERTEX2` / `EDGE2` records. These aliases use g2o's
upper-triangle matrix ordering in this file, as also used by GTSAM's
`readG2o(..., false)` path. They must not be confused with TORO's different
matrix ordering. ropt accepts these aliases with g2o ordering only.

There are 30 vertices, 59 relative constraints, 177 residual rows and 87 active
tangent coordinates. No input vertex is marked `FIX`. Vertex 0 is anchored at its
initial pose to remove the connected graph's three gauge freedoms. The optimized
output preserves IDs, constraints, information matrices and original `FIX`
records; automatic solver anchors are not inserted into the exported dataset.

For a constraint from i to j with measurement Z, the raw residual is the
translation and principal angle of `Z^-1 * (Xi^-1 * Xj)`, matching g2o's
[EdgeSE2 convention](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/types/slam2d/edge_se2.h).
It is **not** a Lie Log residual. The information matrix is in those coordinates.
`Omega=L*L^T` is whitened with `W=L^T`, so `||W*r||^2=r^T*Omega*r`.
The reported objective is `0.5 * sum(r^T*Omega*r)` with squared loss; g2o's
unhalved chi-square would be twice this number. Comparing costs from solvers
using a different residual convention or normalization would be misleading.

Both runs use AD local Jacobians, right SE2 retractions and default LM nonlinear
options (100 trials, initial damping 1e-3, gradient/step tolerance 1e-10 and cost
change tolerance 1e-12). Sparse CG uses at most 10,000 iterations per solve,
relative residual tolerance 1e-8 and absolute tolerance 1e-10, with Jacobi
preconditioning and a checked true linear residual. Dense uses Cholesky.

## Observed result

| Measurement | Dense LM | Sparse LM |
| --- | ---: | ---: |
| Initial cost | 10.72025504017 | 10.72025504017 |
| Final cost | 0.08290751089989 | 0.08290751089989 |
| Nonlinear trials | 7 | 7 |
| Total CG iterations | 0 | 443 |
| Termination | Cost | Cost |

`check_pose_graph.py` independently evaluates the objective from original edges
and exported poses using scalar Python arithmetic. It verifies preserved input
constraints, unchanged anchor, cost decrease and agreement between backends.
Observed maximum per-pose difference (translation and wrapped angle norm) was
2.113e-11. A separate central-difference check in absolute stored coordinates,
with step 1e-6 and vertex 0 excluded, found a gradient infinity norm of 1.464e-9.
Small floating-point differences are expected across toolchains/machines; the
checker uses tolerances rather than demanding byte-identical optimized poses.

CI uses the ropt-authored four-pose loop in `crates/ropt/tests/data/loop_se2.g2o`
and requires no dataset downloads. Parser tests cover malformed/unsupported
records, duplicate or missing IDs, forward references, SPD information,
comments and noncontiguous IDs. Relative factors are checked against central
differences, and correlated whitening against raw Mahalanobis energy and its
numerical gradient. The synthetic loop recovers a known solution with both
backends and preserves its fixed anchor through serialization and reload.

Remaining limits include principal-angle branch cuts, normal-equation conditioning,
no sparse covariance method and limited dataset coverage. A later
[sparse LDLᵀ backend](../../benchmarks/SPARSE_LDL.md) now provides checked direct
LM solves; it has no indefinite pivoting or cached symbolic analysis.
The earlier GTSAM wheel baseline fails its independent quality gate; the later
[exact-Jacobian C++ reference](../../benchmarks/gtsam_exact/README.md) passes it
at 30, 100 and 300 poses. See both reports for the settings and recorded results.
`Cost` termination alone is not a proof of a globally optimal solution.
