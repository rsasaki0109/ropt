# Rejected candidate: avoid unused solves at refinement termination

**Adoption withdrawn.** The [complete-stream follow-up](REFINEMENT_STEP_TIMING.md)
observes **8.2% higher stream median**, with only **1/6 faster pairs**. Production
is restored to the preceding implementation. The allocation reduction below is
a retained candidate result, not a current production or speed improvement.

After a successful final iSAM refinement, the preceding implementation always
solved the frozen normal model for another step. That step was discarded when
the refinement cap was reached or the existing gradient stop test succeeded.
The tested candidate only prepares a next step when another pass can use
it. The initial undamped solve still checks observability; accepted trials,
gain-ratio checks, tolerances, anchors and transactional commit are unchanged.

This is a small allocation/work reduction. It does **not** establish lower
complete-stream time, better P99 or improved real-time behavior.

## Motivation from recorded work

`analyze_native_latency.py` checks hashes of the retained inputs and summarizes
both fresh-native campaigns without collecting new timings:

```sh
python3 benchmarks/analyze_native_latency.py
```

In the confirmation campaign, recovery updates are **774 / 6,264 (12.36%)** of
ropt arrivals but **62.12%** of its recorded wall time. This identifies the full
refresh/refinement path as a useful investigation area; it does not attribute
its entire latency, cloud variation or native differences to the removed solve.
The [native comparison](CURRENT_NATIVE_ISAM2.md) retains the differing policies,
unchanged gates and mixed tail-latency observations.

## Complete-update allocation diagnostic

`benchmark_refinement` counts successful allocation/reallocation requests during
a complete forced nonlinear refresh through the public API. It excludes recipe
construction and setup; the current-estimate probe and subsequent cached update
run after counting. These later operations check gradient accuracy and preserved
anchors. The independent scalar nonlinear factors use analytic derivatives;
this is not a recorded-graph latency measurement or peak-memory measurement.

Eighteen paired configurations cover QR, Gaussian Bayes tree and editable Bayes
tree, 1/16/128 variables, and refinement caps 1/4. Five repeats per variant give
**180 complete refreshes**. The same driver is compiled against both kernels.
Callbacks, accepted refinement counts and result fingerprints (including the
following cached update, costs and gradient) match exactly in every pair. These
fixtures accept one refinement, exercising both cap exhaustion and the existing
gradient stop. Requested traffic per 128-variable refresh is:

| Backend | Allocations before → after | Requested bytes before → after |
| --- | ---: | ---: |
| QR | 4,535 → 4,534 | 422,384 → 421,360 |
| Gaussian Bayes tree | 4,795 → 4,794 | 437,744 → 436,720 |
| Editable Bayes tree | 7,885 → 7,883 | 609,760 → 607,712 |

```sh
cargo run --release --locked --offline -p ropt --example benchmark_refinement -- editable 128 4 5
```

The current checkout retains the preceding refinement algorithm. Restore its
exact measured fingerprint with `make_ldl_baseline.py --before-covariance-preflight`
to remove the later covariance input-validation change too. Build the retained
candidate in a new directory outside the checkout:

```sh
cp -R benchmarks/results/refinement_step_timing/after_source /tmp/ropt-refinement-step-candidate
cargo build --release --locked --offline \
  --manifest-path /tmp/ropt-refinement-step-candidate/Cargo.toml \
  --example benchmark_refinement
```

The restored production fingerprint is
`12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288`;
the changed fingerprint is
`ffdd6fd2033b67900137ccbca263df7749c4f6ecc681d98601821e8a8fc2fd74`.
The original candidate's **54 source-only rollback stages**, including the
default oldest baseline, retain their expected fingerprints in the archive.
The live builder returns to its preceding **53-stage** chain; the candidate's
added restore flag is available only in its frozen recipe. Historical measured
tools remain frozen.

## Functional replay and evidence

The complete mixed-AD CSAIL replay uses the unchanged chunk-six/stride-two,
editable, zero-requested-pass policy and independent original-information gates.
All **1,044 arrivals** pass; full pose bytes and non-timing work records match
the preceding kernel exactly. Maximum normalized gradient is
`2.516765743356827e-8`; native pose/cost gaps remain
`3.772476536257072e-7` / `3.9968028886505635e-13`.
This single functional replay has no warmup and supports no new timing ratio.
Validation passes **353 workspace tests (including 19 doctests), 2 benchmark
tests, 27 Python checks**, Clippy for all targets with warnings denied, and
formatting. The anchor/rollback regression covers caps 1/4 across all three
Gaussian backends; injected refinement callback failures still retry cleanly.

[Evidence](results/refinement_step/manifest.json) retains the diagnostic CSV,
source/binary identities, latency analysis, functional replay check, tests and
rollback records. The changed source is represented by its frozen file and
forward patch; the preceding full source and identical compressed poses are
already retained in `results/current_native_isam2`.
