# CSR row-storage dimension validation

`CsrMatrix::from_triplets` now returns `Error::InvalidDimensions` when
`rows + 1` overflows or the corresponding `usize` row-offset array exceeds
Rust's allocation layout limits. Validation runs before sorting the triplets
or allocating constructor-owned storage. It does not impose a memory budget
or guarantee that every representable allocation can succeed.

The regression reproduces an addition-overflow panic before the change and
checks both count overflow and allocation-layout overflow after the change.
Separate coverage preserves empty shapes and sparse matrices whose column
count is large but requires no column-sized allocation at construction.
Existing sparse tests check duplicate reduction and matrix products.

The [evidence manifest](results/csr_dimensions/manifest.json) records the
before/after source fingerprints, test logs and file hashes. The previous
source can be reconstructed with:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-csr-dimensions --output /tmp/ropt-before-csr-dimensions
```

All 58 historical source reconstruction stages retain their expected hashes.
These are source checks, not compilation checks for every historical stage.
This change makes no timing or comparative performance claim.

## Arithmetic regression follow-up

The separate [arithmetic evidence](results/csr_arithmetic/manifest.json) records
three regression tests for existing numerical error handling. They cover
overflow in individual products, finite products whose accumulation overflows,
overflow in a column's sum of squares, NaN/Inf entries in structurally unused
vector positions, and overflowing duplicate reduction. Finite cancellation
and reuse of a matrix after a failed product are also checked.

These cases already return the expected errors; the follow-up changes tests
and documentation, without changing the numerical implementation or its source
fingerprint. Both debug and release runs exercise the new cases.
