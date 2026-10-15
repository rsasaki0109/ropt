"""Isolated clone-only allocation traffic, excluding model creation and solves.

Allocation/reallocation requests are not live or peak memory. No latency claim.
Run after the non-instrumented timing campaigns, with no concurrent build/run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'diagnostic output outside checkout required')
    out.mkdir()
    files = ('benchmarks/profile_editable_clone.rs', 'benchmarks/profile_editable_metadata.py',
             'crates/ropt/examples/support/allocation_counter.rs', 'README.md')
    metadata = {'scope': 'clone-only successful allocation/reallocation traffic; no live/peak memory or latency measurement',
                'columns': 2048, 'copies': 64, 'allocator_counting': True,
                'before_source_sha256': source_fingerprint(args.before_source),
                'after_source_sha256': source_fingerprint(ROOT),
                'files_sha256': {f: digest(ROOT / f) for f in files},
                'variants': {}}
    require(metadata['before_source_sha256'] == 'cae5cfa485723513f5799fb7722043b2d39c0299639535c1bd78775d424c9323',
            'wrong preceding kernel')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for variant, source in (('before', args.before_source), ('after', ROOT)):
        tree = out / variant
        for f in ('Cargo.toml', 'Cargo.lock', 'crates/ropt/Cargo.toml'):
            dest = tree / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / f, dest)
        shutil.copy2(ROOT / 'README.md', tree / 'README.md')
        shutil.copytree(source / 'crates/ropt/src', tree / 'crates/ropt/src')
        examples = tree / 'crates/ropt/examples'
        (examples / 'support').mkdir(parents=True)
        shutil.copy2(source / 'crates/ropt/examples/benchmark_pose_graph.rs', examples / 'benchmark_pose_graph.rs')
        shutil.copy2(ROOT / files[0], examples / 'profile_editable_clone.rs')
        shutil.copy2(ROOT / files[2], examples / 'support/allocation_counter.rs')
        fingerprint = source_fingerprint(tree)
        require(fingerprint == metadata[variant + '_source_sha256'], 'snapshot kernel differs')
        env = {**os.environ, 'CARGO_TARGET_DIR': str(tree / 'target'), 'CARGO_BUILD_JOBS': '2'}
        with (out / (variant + '-build.log')).open('w') as log:
            subprocess.run(['cargo', 'build', '--manifest-path', str(tree / 'Cargo.toml'), '-p', 'ropt',
                            '--example', 'profile_editable_clone', '--release', '--locked', '--offline'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = out / (variant + '-profile_editable_clone')
        shutil.copy2(tree / 'target/release/examples/profile_editable_clone', binary)
        results = []
        for index in range(2):
            result = subprocess.check_output(['taskset', '-c', '0', str(binary), '2048', '64'], text=True)
            (out / f'{variant}-{index}.json').write_text(result)
            row = json.loads(result)
            require(row['independent_means_pass'], 'independent model validation failed')
            require(row['columns'] == 2048 and row['copies'] == 64, 'wrong diagnostic workload')
            results.append(row)
        require(results[0] == results[1], 'allocation counts are not repeatable')
        metadata['variants'][variant] = {'source_sha256': fingerprint, 'binary_sha256': digest(binary)}
    require(source_fingerprint(args.before_source) == metadata['before_source_sha256'], 'before kernel changed')
    require(source_fingerprint(ROOT) == metadata['after_source_sha256'], 'after kernel changed')
    require(all(digest(ROOT / f) == value for f, value in metadata['files_sha256'].items()), 'diagnostic source changed')
    require(metadata['variants']['before']['binary_sha256'] != metadata['variants']['after']['binary_sha256'],
            'identical executables for different clone implementations')
    before = json.loads((out / 'before-0.json').read_text())
    after = json.loads((out / 'after-0.json').read_text())
    report = {'eligible': True, 'scope': metadata['scope'], 'before': before, 'after': after,
              'counts_repeat_exactly': True,
              'allocation_call_reduction_percent': 100 * (1 - after['allocation_calls'] / before['allocation_calls']),
              'requested_byte_reduction_percent': 100 * (1 - after['requested_bytes'] / before['requested_bytes'])}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
