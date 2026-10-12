# Contributing

Use the pinned Rust toolchain and commands in README. Include numerical regression
cases for solver changes and compare analytic/AD Jacobians with central differences.
Keep sparse and manifold solver work separate from the dense baseline. Document
conventions, conditioning and failure behavior. Do not claim a speed advantage
without reproducible comparative benchmarks. Never weaken assertions to pass CI.
