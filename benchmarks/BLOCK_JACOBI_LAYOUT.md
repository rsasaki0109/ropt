# Block Jacobi allocation layout preflight

`BlockJacobi::from_normal` now validates the storage layouts of the column
mapping and every dense diagonal block before allocating constructor-owned
storage. An unrepresentable layout returns `Error::InvalidDimensions`.

The constructor first validates damping and the contiguous block partition.
It then checks the `n`-entry mapping of `(usize, usize)` pairs, each block's
square element count with checked multiplication, and its `f64` allocation
layout. This prevents both integer overflow and allocations whose byte size
exceeds Rust's layout limits. The check precedes mapping allocation, so an
invalid dense block cannot first trigger a very large mapping allocation.

The mapping-overflow regression was run before the fix and reproduced a
`capacity overflow` panic. After the fix, three regression tests cover:

- Unrepresentable column mapping sizes, using empty CSR matrices whose column
  counts require no column-sized storage in the matrix itself.
- Dense block element-count overflow and byte-layout overflow, even though the
  corresponding column mapping layout is representable.
- Valid damped blocks with no residual rows, compared with the independent
  inverse of `0.5*I` for both one-block and mixed-block partitions.

The integration test binary has a test-only allocator that rejects individual
requests larger than 16 MiB. If layout preflight regresses and a large request
reaches allocation, the binary fails without attempting that allocation. The
dense-block cases were not run against the old implementation; only the
mapping-overflow case was used for the before-fix reproduction. Debug and
release runs of the fixed implementation execute all three tests.

This is a representability check, not an available-memory budget. A valid layout
can still require too much memory for a particular machine. Normal construction,
Cholesky factorization and preconditioner application use the existing numerical
paths. No timing or comparative performance claim is made.

The [evidence manifest](results/block_jacobi_layout/manifest.json) records the
source fingerprints, test logs and hashes. To reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-block-layout --output /tmp/ropt-before-block-layout
```

All 62 source-only historical reconstruction stages preserve their expected
fingerprints. This does not claim compilation of every historical stage.
