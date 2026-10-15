# Frozen-factor gradient solves

`GaussianBayesTree::solve_gradient(&gradient)` and
`EditableBayesTree::solve_gradient(&gradient)` solve
`Rᵀ R step = -gradient` with the currently stored triangular factor. Both the
input and output use original variable coordinates, even with a permuted
elimination order. Forward substitution through `Rᵀ` and backward substitution
through `R` avoid explicitly constructing the normal matrix.

This supports repeated corrections with a frozen linear model. The caller must
supply the gradient for the intended variable layout; the cached residual RHS
is not used. Factor replacement in an editable tree changes the factor used by
the next solve. Neither method mutates the tree or applies damping.

The existing relative diagonal rank criterion applies. Dimension mismatches,
nonfinite gradients, rank deficiency and nonfinite arithmetic results return
errors. Two triangular solves remain sensitive to conditioning; avoiding an
explicit normal matrix does not remove that sensitivity.

Editable iSAM nonlinear refresh now reuses the editable tree factor for
observability and refinement, avoiding a separate global QR rebuild. Other
backends retain their existing global QR path. This changes arithmetic order;
complete real-stream accuracy and timing evidence is recorded in
[the refinement study](../benchmarks/TREE_REFINEMENT.md). The API alone does not
establish performance parity with GTSAM.

Tests manufacture independent solutions using `-Jᵀ(J step)`, cover three
permutations and scales `1e-80`, `1`, `1e80`, and verify factor replacement,
empty models, rank loss, malformed inputs and arithmetic overflow.

The integration regression in
[`bayes_tree_gradient_updates.rs`](../crates/ropt/tests/bayes_tree_gradient_updates.rs)
compares an incrementally edited, permuted tree with an independently assembled
dense reference and a freshly built Gaussian tree. It covers factor addition,
removal and replacement, rejected removal, variable extension with temporary
rank deficiency, and restoration by a new constraint. Gradient solves and failed
queries must preserve the cached solution and cost. These are correctness
checks; they add no runtime performance claim.

[`isam_refinement_rollback.rs`](../crates/ropt/tests/isam_refinement_rollback.rs)
checks failed nonlinear refinement in both selective backends. It dynamically
identifies the batch-only evaluation prefix, injects a callback error at every
subsequent refinement evaluation, and compares each retry with a clean run.
Committed estimates, factor history, new-variable registration and the interval
schedule must survive each failure unchanged. This transaction guarantee covers
the smoother's owned state; callback-owned external effects remain the caller's
responsibility.
