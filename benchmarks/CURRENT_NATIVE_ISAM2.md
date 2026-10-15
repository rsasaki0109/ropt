# Fresh native iSAM2 comparison

This study compares complete mixed-AD ropt updates with a freshly built GTSAM
4.2.1 reference on the complete CSAIL SE2 stream. It does not combine new ropt
timings with historical GTSAM timings.

The ropt policy uses the editable selective backend, AD chunk six/edge stride
two (586 AD and 586 analytic factors), zero explicit requested passes, retained
automatic threshold passes, and explicit recovery when the internal target is
not met. Periodic full refresh is disabled. Recovery uses sparse QR/nonmonotonic
GN, block minimum-degree ordering, ordering reuse and up to four frozen-factor
refinements.

The native policy uses analytic derivatives, QR/GN, relinearization threshold
zero/skip one, disabled partial relinearization checks, and timed empty updates
until its internal normalized stationarity target is met (maximum 200 calls).
It uses exact BetweenFactor Jacobians and the existing precision-corrected
original-information adapter. This is not GTSAM's default iSAM2 policy.

Both use internal column-normalized gradient target `8e-8`; independent checks
require every arrival's original-information gradient at most `1e-7`, an exact
fixed root, reported cost tolerance `1e-10 * max(1,cost)`, inter-library pose gap
at most `2e-6` and cost gap at most `1e-9 * max(1,cost)`. Original measurements,
information matrices, seeds and constraint order are unchanged.

## Results

| Campaign | ropt stream median | Native stream median | Native / ropt | Faster ropt pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 13.739 s | 25.609 s | 1.864x | 6/6 |
| Confirmation | 16.865 s | 26.878 s | 1.594x | 5/6 |

Both satisfy the predefined repeated-win criterion for this case. The final
confirmation pair is **36.816 s ropt versus 32.551 s native**. Variation is
substantial: ropt stream ranges are 11.434–18.966 s and 11.042–36.816 s; native
ranges are 18.494–35.536 s and 19.760–32.551 s. These cloud observations are not
a statistical significance claim or a comparison with default native settings.

Arrival timings pool 6,264 updates per variant per campaign (milliseconds):

| Campaign / variant | Median | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Primary ropt | 5.265 | 68.562 | 131.752 | 2614.807 |
| Primary native | 11.420 | 102.292 | 264.449 | 2220.549 |
| Confirmation ropt | 6.298 | 84.291 | 209.910 | 1519.522 |
| Confirmation native | 13.854 | 105.143 | 179.719 | 1269.113 |

Tail latency is mixed: confirmation P99 and both observed maxima favor native.
The lower stream medians do not establish a real-time latency guarantee.

All **25,056 timed arrivals** pass the original-information accuracy gates.
Maximum normalized gradient is `2.516765743356827e-8` for ropt and
`7.287964624451133e-8` for native. Every paired stream agrees within
`3.772476536257072e-7` world-pose norm and `3.9968028886505635e-13` absolute cost.
Within each variant, complete poses and non-timing work records are identical
across all repeats.

The [artifact manifest](results/current_native_isam2/manifest.json) binds the
raw samples, update work, quality records, compressed first-stream poses,
source/driver snapshots and native build provenance. The independent
[audit](results/current_native_isam2/audit.json) rechecks artifact/source hashes,
recomputes 4,176 retained pose-prefix checks, checks all repeated work/quality
records and reconstructs every paired summary. The runner independently checked
all 25,056 arrivals during collection. Native precision CTest passes 1/1;
Python checks pass 27/27. Production Rust sources are unchanged in this study;
the Rust workspace suite was not rerun for these benchmark-only changes.

## Protocol

Two planned campaigns each use six alternating pairs. Every fresh process
performs one complete discarded warmup and one timed complete stream, pinned to
CPU 0 and one thread. No builds, tests, profilers or separate audits run during
comparative timing. The runner's own validation/compression occurs between timed
processes. There are 25,056 timed arrivals across both campaigns; warmups are not
serialized for independent checking.

Timers include complete updates, nonlinear probes, recovery/empty updates and
owned means/reports. Parsing, factor construction, serialization and independent
validation are excluded. Allocator counting is disabled, and peak memory is not
measured. A repeated ropt win requires a lower stream median and at least five
of six faster pairs in each campaign. No confidence interval or significance
claim is made. Pooled per-arrival P95/P99 use nearest rank; maxima are observations.

## Native build

`build_native_reference.py` takes an already downloaded archive, verifies its
pinned SHA-256, extracts a new source tree outside the checkout and verifies all
source contents before and after building. Resume is allowed only with identical
recorded build inputs and unmodified source; added files or retargeted links are
rejected. It builds the library, installs into its output directory, builds the
precise adapter, runs its precision test, and records the actually linked GTSAM
and runtime-library hashes. It does not add a C++ dependency to the Rust library.

```sh
curl --fail --location --proto '=https' --tlsv1.2 \
  https://codeload.github.com/borglab/gtsam/tar.gz/refs/tags/4.2.1 \
  --output /tmp/gtsam-4.2.1.tar.gz
python3 benchmarks/build_native_reference.py \
  --archive /tmp/gtsam-4.2.1.tar.gz --output /tmp/ropt-native-reference --jobs 2
```

CMake, a C++ compiler and the Boost development components serialization,
filesystem, program_options, timer and chrono must be available. The builder
verifies archive SHA-256
`96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2`
before executing CMake. Release, exact BetweenFactor Jacobians, bundled Eigen,
STL allocation, disabled TBB/MKL/march-native/tests/unstable/examples/Python are
explicit. Adapter precision tests are enabled and executed separately.

For a relocated Debian sysroot, pass `--cmake`, `--boost-root`,
`--boost-library-dir`, `--runtime-library-path` and optionally
`--dependency-manifest`. This environment downloads signed Debian 13 repository
metadata and packages into a workspace cache, verifies package SHA-256 against
that metadata and extracts them with `dpkg-deb -x`; it does not install system
packages or execute their maintainer scripts. Compiler, CMake, Boost package
versions, signed metadata/keyring hashes and build commands are recorded.

One initial preflight configure lacked Boost's relocated multiarch library
path. It built no library or benchmark executable and is excluded from timing.
The corrected configuration explicitly supplies that path. The first install
also required the upstream CppUnitLite auxiliary archive; the builder now builds
that target alongside GTSAM before installing. Both corrections occurred before
any comparative timing. No source patch, Jacobian relaxation or numerical gate
change was used to make it build.

## Reproduction

Build the current Rust driver with `--release --locked --offline`. Run
`check_current_native_isam2.py` first for the common 25-arrival functional check
and output/argument protection. Then use `paired_current_native_isam2.py` for
six paired rounds, with a new directory outside the checkout for each campaign.
A relocated native runtime needs its prefix and dependency library paths via
repeated `--native-library-path` arguments. Read the runner's `--help` for exact arguments.

```sh
cargo build --release --locked --offline --jobs 1 \
  --example benchmark_ad_selective_se2_isam
python3 benchmarks/check_current_native_isam2.py \
  --ropt target/release/examples/benchmark_ad_selective_se2_isam \
  --native /tmp/ropt-native-reference/adapter-build/gtsam_isam2 \
  --native-library-path /tmp/ropt-native-reference/prefix/lib \
  --output /tmp/ropt-native-smoke
python3 benchmarks/paired_current_native_isam2.py \
  --ropt target/release/examples/benchmark_ad_selective_se2_isam \
  --native /tmp/ropt-native-reference/adapter-build/gtsam_isam2 \
  --native-manifest /tmp/ropt-native-reference/native-manifest.json \
  --native-library-path /tmp/ropt-native-reference/prefix/lib \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --output /tmp/ropt-native-primary
```

Run the last command again with a new `--output /tmp/ropt-native-confirmation`
after the primary campaign finishes. Do not run competing builds, tests or
audits during either campaign. Add a second `--native-library-path` if the
native dependencies also live in a relocated sysroot. Existing output directories
are rejected, rather than overwritten.

The results, complete source snapshots and independent audit are retained under
`results/current_native_isam2`. Recheck with:

```sh
python3 benchmarks/verify_current_native_isam2.py
```

Use `--current-source` only when checking the exact measured production source.
The later [refinement termination candidate](REFINEMENT_STEP_TIMING.md) failed
its before/after timing criterion and its kernel was restored. Subsequent API
validation changes can change the complete source fingerprint; the default
audit uses the retained measured source. No fresh native timing is implied.
Conclusions are restricted to this dataset and these explicit accuracy policies,
not general 3D/VIO/BA superiority, default-policy latency or field deployment.
