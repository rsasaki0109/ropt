# Complete mixed-AD iSAM phase diagnosis

The diagnostic isolates initial updates, explicit requested passes, full recovery
and direct nonlinear probes. The initial update includes automatic threshold
relinearization; it is not just new-factor insertion. Calls and returned errors
are counted, including unsuccessful attempts. Valid rejected trials remain inside
their parent solver/evaluation durations.

| Share of recorded update time | CSAIL cold | CSAIL after one warmup | MIT cold |
| --- | ---: | ---: | ---: |
| Initial update, including threshold passes | 24.8% | 25.4% | 28.6% |
| Explicit requested passes | 9.2% | 9.2% | 11.6% |
| Full recovery | 45.6% | 44.7% | 38.0% |
| Direct nonlinear probes | 20.3% | 20.6% | 21.7% |
| Remaining control/report work and instrumentation overhead | 0.1% | 0.1% | 0.1% |

These are **instrumented phase estimates**, not comparative or native timings.
Clock reads, bookkeeping, changed inlining and logging can affect execution.
The timer excludes profile output, but output can affect subsequent execution.
Fine-grained QR timers run once per clique; overhead matters particularly for
these short calls. No absolute speed claim follows from the table.

Within recovery, sparse model evaluation accounts for **18.6% / 17.9%** of all
CSAIL time and **14.0%** of MIT time. This includes residual/Jacobian callbacks,
validation, whitening and CSR assembly. Recovery tree construction accounts for
**10.5% / 10.3%** and **7.9%**, respectively. Across all API phases, editable tree
updates account for about **21%** of each run's total. These figures are inclusive:
batch solving contains evaluations/QR work, and tree construction contains tree
updates/QR work. **Do not add overlapping detail percentages.**

The direct probe occupies about one fifth of the budget. The next work should
therefore include recovery model evaluation and tree construction, along with
whether an extra explicit selected pass is useful before recovery. With cap 1,
CSAIL makes 130 requested passes and 103 recoveries; MIT makes 111 requested passes
and 111 recoveries. Both retain automatic threshold-selected passes inside initial
updates. The [recovery-cap experiment](RECOVERY_CAPS.md) follows this observation.

## Numerical verification

Complete CSAIL cold and warm runs each report 1,044 arrivals; MIT reports 1,227.
All **3,315 reported arrivals** retain exact complete pose bytes and every
non-timing update field versus the frozen preceding kernel. All original-
information quality gates and hashed archived-native pose/cost agreement pass.
An additional 1,044 warmup phase records are retained: **4,359 raw phase records**.
Warmup trajectories are not serialized for the independent check, although the
driver enforces the same internal gradient target during warmup.

CSAIL independent maximum normalized gradient is `5.8075741199751337e-8`; MIT is
`7.793438497199398e-8`. Original-information gate `1e-7`, internal target `8e-8`,
exact fixed root and reported-cost tolerance `1e-10 * max(1,cost)` are unchanged.
Archived-native agreement uses world-pose norm `2e-6` and cost tolerance
`1e-9 * max(1,cost)`. Original constraints, information, initialization, chunk 6,
stride 2 and the editable-QR/GN policy with requested cap 1 are unchanged.
The observed API calls return no errors in these datasets; the diagnostic's
returned-error counters must not be confused with line-search rejection counts.

Instrumentation is generated in an isolated snapshot. Production source remains
`408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413`.
Instrumented source:
`73268c464f08bddbe2d7e14f360b5f21e3bf8feb7f3e8f3bbe64d8587469a458`.
The modified benchmark helper is hashed separately. All **331** snapshot library,
integration and documentation tests pass. Reversing the saved instrumentation
patch with zero fuzz restores the exact production source fingerprint.

## Reproduction and audit

Use the prepared toolchain/cache; the source must match the frozen reference.
The tool copies source to a new output directory outside the checkout, creates
instrumentation, formats/tests/builds there with an independent Cargo target,
then runs cold CSAIL, warmed CSAIL and cold MIT sequentially on CPU 0 with one
thread. Original dataset loading and output serialization are outside timers.
The generated thread-local collector records nested helper durations without
printing inside the timed API calls. API timers are disjoint; details are inclusive.

```sh
python3 benchmarks/profile_stream_phases.py --output /tmp/ropt-stream-phases
python3 benchmarks/verify_stream_phases.py --current-source
```

The [source snapshot, recipe, build log, patch, raw phases, helper timings and
complete poses](results/stream_phases/) are retained. Omit `--current-source`
when auditing after further development. The snapshot has no compiled target
directory in the archive. Native binaries are not run or benchmarked here.
