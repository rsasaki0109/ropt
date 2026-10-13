"""Reproduce a rejected normal-graph candidate in an isolated benchmark snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('bulk', 'diagonal'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT,
                        help='predecessor source; use the restored historical kernel for archived experiments')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('candidate must be outside checkout')
    output.mkdir()
    source = args.source.resolve()
    for name in ('Cargo.toml', 'Cargo.lock', 'README.md', 'crates/ropt/Cargo.toml'):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    for name in ('src', 'examples'):
        shutil.copytree(source / 'crates/ropt' / name, output / 'crates/ropt' / name)
    patch = ROOT / 'benchmarks' / {
        'bulk': 'normal_cursor.patch', 'diagonal': 'normal_diagonal.patch'
    }[args.candidate]
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', str(output),
                    '-i', str(patch)], check=True)
    metadata = {
        'purpose': 'rejected candidate reproduction; production checkout unchanged',
        'candidate': args.candidate,
        'before_source_sha256': source_fingerprint(source),
        'candidate_source_sha256': source_fingerprint(output),
        'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        'zero_fuzz_apply': True,
    }
    (output / 'candidate.json').write_text(json.dumps(metadata, indent=2) + '\n')
    subprocess.run(['cargo', 'build', '--manifest-path', str(output / 'Cargo.toml'),
                    '-p', 'ropt', '--example', 'benchmark_warm_isam_se2',
                    '--example', 'validate_isam_se2', '--release', '--locked',
                    '--offline'], check=True)
    print(f'Candidate executables: {output}/target/release/examples')


if __name__ == '__main__':
    main()
