"""Balanced cached-policy timings with unchanged independent nonlinear checks.

No native comparison: cached updates do not meet the forced-refresh accuracy
target at every prefix. Allocation mode is a separate diagnostic, not timing.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require, timing_summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'before-source', 'after', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--after-source', type=Path, default=ROOT)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--allocations', action='store_true')
    args = parser.parse_args()
    require(25 <= args.frames <= 300 and args.frames % 25 == 0,
            'frames must be a multiple of 25 in 25..300')
    require(args.rounds > 0 and args.rounds % 2 == 0, 'balanced even rounds required')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'output must be outside checkout')
    out.mkdir()
    templates = ('crates/ropt/examples/benchmark_cached_isam_se2.rs',
                 'crates/ropt/examples/support/isam_se2.rs',
                 'crates/ropt/examples/support/allocation_counter.rs')
    for name in templates:
        require((args.before_source / name).read_bytes() == (ROOT / name).read_bytes(),
                'different workload or runner: ' + name)
        require((args.after_source / name).read_bytes() == (ROOT / name).read_bytes(),
                'different archived after runner: ' + name)
    metadata = {
        'purpose': 'reuse principal local coordinates within each iSAM update',
        'frames': args.frames, 'rounds': args.rounds,
        'environment': environment(), 'warmup_streams_per_process': 1,
        'measured_streams_per_process': 1,
        'policy': 'threshold=0.1; interval=10; forced delayed loops every 25; '
                  'SparseLdl block ordering(3); restart LM damping; '
                  'original fixture seeds and non-diagonal information',
        'timed': 'complete update including owned means/report, excluding root initialization, '
                 'factor recipes, original energy/gradient and AD/dense-QR oracle checks, '
                 'serialization, observer accounting and report destruction',
        'gates': {'fixed_root': 'exact at all prefixes',
                  'refreshed_original_gradient_inf': 1e-6,
                  'refreshed_cost_gap_relative': 1e-10,
                  'checkpoint_batch_pose_difference': 2e-6,
                  'checkpoint_batch_cost_difference_relative': 1e-9,
                  'checkpoint_gradient_step_difference': 2e-7,
                  'cached_prefix_stationarity': 'not required; approximation explicitly reported',
                  'before_after': 'exact all-prefix full estimates, reports and quality traces'},
        'allocation_counting': args.allocations,
        'memory_measured': False, 'native_timing': False,
        'source_sha256': {'before': source_fingerprint(args.before_source),
                          'after': source_fingerprint(args.after_source)},
        'executable_sha256': {'before': sha(args.before), 'after': sha(args.after)},
        'files_sha256': {n: sha(ROOT / n) for n in
                         (*templates, 'benchmarks/paired_anchor_coordinates.py',
                          'benchmarks/anchor_coordinates.patch')},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    samples, updates, qualities = [], [], []
    reference = None
    for round_index in range(args.rounds):
        order = ('before', 'after') if round_index % 2 == 0 else ('after', 'before')
        observed = {}
        for position, variant in enumerate(order):
            directory = out / f'{round_index}-{variant}'
            command = [str(binaries[variant]), str(args.frames), str(directory)]
            if args.allocations:
                command.append('--allocations')
            # Each process also executes the original quality gates, outside timers.
            subprocess.run(command, check=True)
            with (directory / 'updates.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            require([int(r['frame']) for r in rows] == list(range(1, args.frames + 1)),
                    'missing, duplicate or unordered updates')
            require(all(int(r['elapsed_ns']) > 0 for r in rows), 'invalid timer')
            if not args.allocations:
                require(all(r['allocation_calls'] == r['allocation_bytes'] == '0' for r in rows),
                        'allocation instrumentation enabled during timing')
            with (directory / 'quality.csv').open() as stream:
                quality = list(csv.DictReader(stream))
            require(len(quality) == args.frames, 'missing quality records')
            require(sum(bool(r['batch_pose_difference']) for r in quality) == args.frames // 25,
                    'missing independent AD/dense-QR checkpoints')
            comparable = [{k: v for k, v in r.items() if k not in
                           ('elapsed_ns', 'allocation_calls', 'allocation_bytes')} for r in rows]
            observed[variant] = ((directory / 'means.csv').read_bytes(), comparable,
                                 (directory / 'quality.csv').read_bytes(),
                                 (directory / 'summary.txt').read_bytes(),
                                 (directory / 'options.txt').read_bytes())
            require(reference is None or observed[variant] == reference,
                    'full estimates, reports, quality, options or summary changed')
            reference = observed[variant]
            samples.append({'variant': variant, 'round': round_index, 'position': position,
                            'elapsed_ns': sum(int(r['elapsed_ns']) for r in rows),
                            'cached_ns': sum(int(r['elapsed_ns']) for r in rows if r['reason'] == 'None'),
                            'refresh_ns': sum(int(r['elapsed_ns']) for r in rows if r['reason'] != 'None'),
                            'allocation_calls': sum(int(r['allocation_calls']) for r in rows),
                            'allocation_bytes': sum(int(r['allocation_bytes']) for r in rows)})
            updates.extend({'variant': variant, 'round': round_index, **r} for r in rows)
            qualities.extend({'variant': variant, 'round': round_index, **r} for r in quality)
            if round_index == 0:
                with gzip.open(out / f'{variant}-means.csv.gz', 'wb') as stream:
                    stream.write(observed[variant][0])
                for name in ('summary.txt', 'options.txt'):
                    (out / f'{variant}-{name}').write_bytes((directory / name).read_bytes())
            print(f'round {round_index} {variant}: {args.frames} prefixes and '
                  f'{args.frames // 25} AD/dense-QR checkpoints passed', flush=True)
        require(observed['before'] == observed['after'], 'before/after differ')
    for name, rows in (('samples.csv', samples), ('updates.csv', updates), ('quality.csv', qualities)):
        with (out / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cases = {}
    for variant in binaries:
        rows = [r for r in samples if r['variant'] == variant]
        if args.allocations:
            cases[variant] = {'allocation_calls': sorted({r['allocation_calls'] for r in rows}),
                              'allocation_bytes': sorted({r['allocation_bytes'] for r in rows})}
        else:
            cases[variant] = {group: timing_summary(
                [{'elapsed_ns': r[field]} for r in rows]) for group, field in
                [('complete', 'elapsed_ns'), ('cached', 'cached_ns'), ('refresh', 'refresh_ns')]}
    report = {'eligible': True, 'cases': cases,
              'all_prefix_means_reports_and_quality_exact': True,
              'measured_updates': args.rounds * 2 * args.frames,
              'paired_faster': None if args.allocations else {
                  field: sum(samples[2 * k + (k % 2)]['elapsed_ns' if field == 'complete' else field + '_ns'] >
                             samples[2 * k + (1 - k % 2)]['elapsed_ns' if field == 'complete' else field + '_ns']
                             for k in range(args.rounds)) for field in ('complete', 'cached', 'refresh')}}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
