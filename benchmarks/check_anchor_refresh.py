"""All-prefix forced-refresh regression against the verified native iSAM2 oracle.

This is a correctness check, not a timing campaign for the cached optimization.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess

from paired_anchor_coordinates import sha
from paired_isam2 import agreement, validate
from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require
from se2_stream import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'before-source', 'after', 'native', 'native-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    require(25 <= args.frames <= 300 and args.frames % 25 == 0, 'invalid frames')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'output must be outside checkout')
    out.mkdir()
    write(out / 'input.stream', args.frames)
    manifest = json.loads(args.native_manifest.read_text())
    require(sha(args.native) == manifest['executable_sha256'], 'native executable changed')
    linked = subprocess.check_output(['ldd', str(args.native.resolve())], text=True)
    libraries = [Path(line.split('=>', 1)[1].strip().split()[0])
                 for line in linked.splitlines() if 'libgtsam.so' in line and '=>' in line]
    require(len(libraries) == 1 and sha(libraries[0]) == manifest['library_sha256'],
            'native runtime library changed')
    templates = ('crates/ropt/examples/benchmark_warm_isam_se2.rs',
                 'crates/ropt/examples/support/se2_benchmark.rs',
                 'crates/ropt/examples/support/allocation_counter.rs')
    for name in templates:
        require((ROOT / name).read_bytes() == (args.before_source / name).read_bytes(),
                'different forced-refresh workload')
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve(),
                'native': args.native.resolve()}
    results, qualities, reports = {}, [], []
    for variant, binary in binaries.items():
        directory = out / variant
        subprocess.run([str(binary), str(out / 'input.stream'), str(directory), '1', '1'], check=True)
        results[variant] = validate(directory, args.frames)
        with gzip.open(out / f'{variant}-poses.csv.gz', 'wb') as stream:
            stream.write((directory / 'poses.csv').read_bytes())
        (out / f'{variant}-options.txt').write_bytes((directory / 'options.txt').read_bytes())
        qualities.extend({'variant': variant, **row} for row in results[variant][1])
        reports.extend({'variant': variant, **row} for row in results[variant][2])
    require(results['before'][0] == results['after'][0], 'full means differ')
    for field in ('batch_trials', 'initial_damping', 'final_damping', 'iterations', 'cost', 'entries'):
        require([r[field] for r in results['before'][2]] ==
                [r[field] for r in results['after'][2]], 'reports differ: ' + field)
    require([r['cost'] for r in results['before'][1]] ==
            [r['cost'] for r in results['after'][1]], 'original costs differ')
    report = {'eligible': True, 'timing_claim': False, 'frames': args.frames,
              'before_after_full_means_costs_trials_damping_entries_exact': True,
              'native_agreement': agreement(results['after'][0], results['native'][0],
                                            results['after'][1], results['native'][1]),
              'max_original_gradient': {v: max(r['gradient_inf'] for r in results[v][1])
                                        for v in binaries}}
    metadata = {'purpose': __doc__, 'environment': environment(),
                'source_sha256': {'before': source_fingerprint(args.before_source),
                                  'after': source_fingerprint(ROOT)},
                'executable_sha256': {v: sha(p) for v, p in binaries.items()},
                'native_manifest': manifest, 'actual_runtime_library_sha256': sha(libraries[0]),
                'files_sha256': {n: sha(ROOT / n) for n in
                                 (*templates, 'benchmarks/check_anchor_refresh.py',
                                  'benchmarks/paired_isam2.py', 'benchmarks/se2_stream.py',
                                  'benchmarks/anchor_coordinates.patch')},
                'input_sha256': sha(out / 'input.stream')}
    for name, rows in [('quality.csv', qualities), ('updates.csv', reports)]:
        with (out / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, value in [('report.json', report), ('metadata.json', metadata)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
