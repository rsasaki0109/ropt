# Reusing the editable tree during batch recovery

An editable iSAM batch recovery previously built a full global incremental QR
factor for observability and frozen-factor refinement, then independently built
an editable Bayes tree from the same anchored factor rows. The candidate builds
the editable tree first and uses its triangular factor for both purposes. Other
Gaussian cache backends retain their existing rebuild path.

The cache stays anchored at the batch solution even when subsequent corrections
are accepted. Predicted/actual reduction, backtracking, gradient improvement,
rank checks, finite-value checks and transactional commit remain required. A
regression exercises accepted corrections across repeated forced refreshes and
compares the editable and global QR results.

Eliminating by cliques changes arithmetic order. Complete trajectories need not
be byte-identical between variants. Both must meet the existing original
information gradient threshold `1e-7`, internal target `8e-8`, exact fixed root,
reported cost tolerance `1e-10 * max(1,cost)`, and archived native pose/cost
agreement tolerances `2e-6` and `1e-9 * max(1,cost)`. No pruning, relaxed
information, altered automatic threshold passes or explicit requested passes
are introduced. The measured policy remains explicit requested cap zero.

## Protocol

The planned comparison is two campaigns of six alternating before/after pairs
on the complete CSAIL stream: 1,044 arrivals, 1,172 original constraints, 586 AD
factors with chunk six and stride two. Each fresh process discards one full
warmup, then times one stream on CPU 0 with one thread and allocator counting
disabled. No builds, tests, profilers, or independent audits run concurrently
with comparative timing. The runner performs its own checking and compression
between timed processes.

The timer includes updates, probes, failed attempts, recovery and owned reports;
it excludes parsing, factor recipes, serialization and independent validation.
Saved GTSAM poses and costs are accuracy references, not fresh native timings.
A repeated speed improvement requires a lower median and at least five of six
faster pairs in each campaign. No confidence interval or significance claim is
made. Pooled arrival P95/P99 use nearest rank; maxima are observations.

An initial launch failed while collecting environment metadata because `rustc`
was absent from that shell's PATH. It executed no benchmark process. The launch
was corrected by activating the existing workspace toolchain; both planned
campaigns are retained without selecting among completed campaigns.

## Reproduction

Activate the workspace Rust toolchain and build with `--release --locked
--offline`. The retained source snapshots and reverse patch identify the exact
before and after kernels. `paired_tree_refinement.py --help` lists runner
arguments; `verify_tree_refinement.py --current-source` checks the archive and
independently recomputes original-information quality and native agreement.

## Results and decision

The candidate is adopted. Both planned campaigns meet the predefined criterion;
no extra comparative campaign was run.

| CSAIL campaign | Before stream median | After stream median | Reduction | Faster pairs |
|---|---:|---:|---:|---:|
| Primary | 5,015.465 ms | 4,815.008 ms | 4.00% | 5/6 |
| Confirmation | 4,910.447 ms | 4,702.668 ms | 4.23% | 6/6 |

All 25,056 timed arrivals meet the original gates and archived native agreement.
Within each variant, repeated trajectories and non-timing update records are
identical. Between variants, trajectories and numerical diagnostics change,
but recovery/query/re-elimination counts remain identical: 129 / 1,173 / 298,053
per stream. These counts do not include the removed redundant QR factorization.
No peak-memory or fresh native timing was measured.

Arrival distributions pool 6,264 arrivals per variant per campaign:

| Campaign / statistic (ms) | Before | After |
|---|---:|---:|
| Primary median | 2.100802 | 2.130997 |
| Primary P95 | 30.171769 | 28.201480 |
| Primary P99 | 51.783753 | 50.811757 |
| Primary maximum | 70.339628 | 69.087402 |
| Confirmation median | 2.1184885 | 2.0747595 |
| Confirmation P95 | 30.892572 | 28.891428 |
| Confirmation P99 | 51.507187 | 48.959524 |
| Confirmation maximum | 73.606674 | 81.490739 |

Thus the stream median improves repeatedly, while per-arrival latency is mixed.
In particular, the primary arrival median and confirmation maximum rise. These
observations do not establish a worst-case latency bound.

The separate complete CSAIL and MIT functional replays also pass. Maximum
original normalized gradients are `2.516765743356827e-8` and
`7.793438497199398e-8`, respectively. The original MIT edge with information near
`1e12` remains included in the AD workload. The initial functional replay source
predates comment-only edits; its exact source can be reconstructed from the
retained `functional/isam.rs` and is checked by the auditor.

Validation: **338 library/integration/doc tests + 2 benchmark tests**, **23 Python
tests**, all-target Clippy with warnings denied, formatting and Rustdoc with
warnings denied. All **49** source-only rollback stages apply with zero fuzz;
these are source reconstruction checks, not 49 compiled test suites.

Artifacts: [results/tree_refinement](results/tree_refinement), with full source,
reverse patches, raw updates/quality/samples, first trajectories for each
variant/campaign, functional replay trajectories, reference hashes, build/test
logs, the aborted pre-timing launch and the independent audit output.

The measured source is
`b46f8c21927edcadd7f032ebd15bd2b2c7a8047909386a1ee6e897cf75c87bfd`;
the preceding source (including the gradient API) is
`64a5d3724d5877ecb4a1bfea3a2db6ebe0adb31db6709d330c69ddad2f849f9a`.
`make_ldl_baseline.py --before-tree-refinement` reconstructs the preceding
kernel; `--before-tree-gradient` additionally removes the gradient API and
reconstructs source `10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`.


The original runner/auditor/baseline-builder versions are retained under
`results/tree_refinement/frozen_tools` and checked against the original tool
hashes. The default auditor validates this historical experiment even after
later source changes; `--current-source` additionally requires the exact measured
kernel and will intentionally reject a newer production source.
