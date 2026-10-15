"""Balanced full recorded SE2 replay with independent all-prefix quality gates."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from paired_pcg import source_fingerprint
from paired_isam2 import agreement, validate
from pose_graph import ROOT, environment, require, timing_summary
from real_se2_stream import column_normalized_gradient, load, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ropt', 'native', 'native-manifest', 'input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ropt-policy', choices=('ldl-lm', 'sparse-qr-gn', 'sparse-qr-gn-ambient', 'sparse-qr-gn-cartesian'), default='ldl-lm')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    require(args.rounds >= 2 and args.rounds % 2 == 0, 'need balanced even rounds')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'write campaign outside checkout')
    out.mkdir()
    dataset = write(args.input, out / 'input.stream')
    require(dataset['complete_dataset'], 'partial recorded dataset is ineligible')
    problem = load(args.input)
    frames = len(problem[1])
    manifest = json.loads(args.native_manifest.read_text())
    require(digest(args.native) == manifest['executable_sha256'], 'native executable changed')
    linked = subprocess.check_output(['ldd', str(args.native.resolve())], text=True)
    libraries = [Path(line.split('=>', 1)[1].strip().split()[0])
                 for line in linked.splitlines() if 'libgtsam.so' in line and '=>' in line]
    require(len(libraries) == 1 and digest(libraries[0]) == manifest['library_sha256'],
            'native runtime library differs')
    reference_dir = ROOT / manifest.get('reference_source_directory', 'benchmarks/gtsam_real_isam2')
    require(reference_dir.resolve().is_relative_to(ROOT/'benchmarks'), 'reference source outside benchmarks')
    for name, expected in manifest['reference_sources_sha256'].items():
        require(digest(reference_dir / name) == expected,
                'native reference source changed')
    ropt_driver = ('crates/ropt/examples/benchmark_real_isam_se2.rs' if args.ropt_policy == 'ldl-lm'
                   else 'crates/ropt/examples/benchmark_cartesian_se2_isam.rs' if args.ropt_policy == 'sparse-qr-gn-cartesian'
                   else 'crates/ropt/examples/benchmark_ambient_se2_isam.rs' if args.ropt_policy == 'sparse-qr-gn-ambient'
                   else 'crates/ropt/examples/benchmark_sparse_qr_isam_se2.rs')
    files = (ropt_driver,
             'crates/ropt/examples/support/real_se2_benchmark.rs',
             'crates/ropt/examples/support/allocation_counter.rs',
             'benchmarks/real_se2_stream.py', 'benchmarks/se2_stream.py', 'benchmarks/information_precision.py',
             'benchmarks/paired_isam2.py', 'benchmarks/paired_real_isam2.py')
    ropt_policy = ('ropt LDL/LM restart damping 1e-5' if args.ropt_policy == 'ldl-lm'
                   else 'ropt explicit sparse QR/nonmonotonic GN with Cartesian SE2 right chart and ambient evaluation' if args.ropt_policy == 'sparse-qr-gn-cartesian'
                   else 'ropt explicit sparse QR/nonmonotonic GN with ambient SE2 analytic evaluation' if args.ropt_policy == 'sparse-qr-gn-ambient'
                   else 'ropt explicit sparse QR/nonmonotonic GN')
    metadata = {
        'dataset': dataset, 'rounds': args.rounds, 'cpu_affinity': [args.cpu],
        'environment': environment(), 'native_manifest': manifest,
        'source_sha256': source_fingerprint(ROOT),
        'executable_sha256': {v: digest(p) for v, p in [('ropt', args.ropt), ('native', args.native)]},
        'files_sha256': {name: digest(ROOT / name) for name in files},
        'gates': {'original_column_normalized_gradient_inf': 1e-7,
                  'world_pose_norm_difference': 2e-6, 'relative_cost_difference': 1e-9,
                  'relative_reported_cost_difference': 1e-10, 'fixed_root': 'exact'},
        'warmup_streams_per_process': 1, 'timed_streams_per_process': 1,
        'policy': f'both solvers force full relinearization every arrival; normalized internal gradient target 8e-8; {ropt_policy}, batch ordering reuse, at most four frozen-QR refinements; native full-relin iSAM2 QR/GN with timed empty updates to stationarity',
        'ropt_policy': args.ropt_policy,
        'timed': 'complete update including full owned means and cost/report; native stationarity probes included; parsing, factor construction, external validation, serialization and report destruction excluded',
        'allocator_counting': False, 'memory_measured': False,
        'scope': 'recorded measurements; pose-ID arrival inferred without sensor timestamps; this accuracy-target policy is not either library default',
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    binaries = {'ropt': args.ropt, 'native': args.native}
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    samples, updates, quality, comparisons = [], [], [], []
    # Retain failures and measured prefixes; an exception never creates eligible=true.
    for round_index in range(args.rounds):
        order = ('ropt', 'native') if round_index % 2 == 0 else ('native', 'ropt')
        measured = {}
        for position, variant in enumerate(order):
            directory = out / f'{round_index}-{variant}'
            with (out / f'{round_index}-{variant}.log').open('w') as log:
                subprocess.run(['taskset', '-c', str(args.cpu), str(binaries[variant].resolve()),
                                str(out / 'input.stream'), str(directory), '1', '1'],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            poses, qualities, raw, sample = validate(directory, frames, problem,
                                                   column_normalized_gradient, 1e-7)
            if variant == 'ropt':
                if args.ropt_policy == 'ldl-lm':
                    require(all(float(r['initial_damping']) == 1e-5 for r in raw), 'restart policy differs')
                else:
                    require(all(r['initial_damping'] == r['final_damping'] == '' for r in raw), 'GN reports damping')
            measured[variant] = poses, qualities
            samples.append({'variant': variant, 'round': round_index, 'position': position, **sample})
            updates.extend({'variant': variant, 'round': round_index, **row} for row in raw)
            quality.extend({'variant': variant, 'round': round_index, **row} for row in qualities)
            if round_index == 0:
                with gzip.open(out / f'{variant}-poses.csv.gz', 'wb') as f:
                    f.write((directory / 'poses.csv').read_bytes())
                (out / f'{variant}-options.txt').write_bytes((directory / 'options.txt').read_bytes())
            print(f'round {round_index} {variant}: {int(sample["elapsed_ns"])/1e9:.6f}s; all {frames} prefixes pass', flush=True)
        comparisons.append({'round': round_index, **agreement(
            measured['ropt'][0], measured['native'][0], measured['ropt'][1], measured['native'][1])})
        # Full means for the first pair are archived; all subsequent means were validated.
        for variant in order:
            shutil.rmtree(out / f'{round_index}-{variant}')
    require(source_fingerprint(ROOT) == metadata['source_sha256'], 'solver source changed during campaign')
    require(all(digest(ROOT / name) == h for name, h in metadata['files_sha256'].items()),
            'runner changed during campaign')
    report = {'eligible': True, 'timed_updates': args.rounds * 2 * frames,
              'cases': {v: {'timing': timing_summary([s for s in samples if s['variant'] == v]),
                            'max_normalized_gradient': max(q['gradient_inf'] for q in quality if q['variant'] == v),
                            'max_raw_gradient': max(q['raw_gradient_inf'] for q in quality if q['variant'] == v)}
                        for v in binaries}, 'comparisons': comparisons}
    paired = {v: {int(r['round']): int(r['elapsed_ns']) for r in samples if r['variant'] == v}
              for v in binaries}
    report['ropt_faster_pairs'] = sum(paired['ropt'][r] < paired['native'][r] for r in range(args.rounds))
    report['median_speedup'] = report['cases']['native']['timing']['median_ms'] / report['cases']['ropt']['timing']['median_ms']
    report['beats_native_at_common_accuracy_in_this_case'] = (
        report['median_speedup'] > 1 and report['ropt_faster_pairs'] * 4 >= args.rounds * 3)
    for name, rows in [('samples.csv', samples), ('updates.csv', updates), ('quality.csv', quality)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
