"""Count actual Givens rotations separately; instrumented times are excluded."""
import argparse
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess

from paired_anchor_coordinates import sha
from paired_isam2 import validate
from paired_pcg import source_fingerprint
from pose_graph import ROOT, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-source', 'primary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--after-source', type=Path, default=ROOT)
    args = parser.parse_args()
    primary = args.primary.resolve()
    metadata = json.loads((primary / 'metadata.json').read_text())
    require(json.loads((primary / 'report.json').read_text())['eligible'], 'accepted primary required')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'output must be outside checkout')
    out.mkdir()
    cases = {}
    for variant, source in [('before', args.before_source), ('after', args.after_source)]:
        source = source.resolve()
        require(source_fingerprint(source) == metadata['source_sha256'][variant], 'wrong source stage')
        for name in ('crates/ropt/examples/benchmark_warm_isam_se2.rs',
                     'crates/ropt/examples/support/se2_benchmark.rs',
                     'crates/ropt/examples/support/allocation_counter.rs'):
            require(sha(source / name) == metadata['files_sha256'][name], 'runner changed')
        snapshot = out / (variant + '-source')
        snapshot.mkdir()
        for name in ('Cargo.toml', 'Cargo.lock', 'README.md'):
            shutil.copyfile(source / name, snapshot / name)
        crate = snapshot / 'crates/ropt'
        crate.mkdir(parents=True)
        shutil.copyfile(source / 'crates/ropt/Cargo.toml', crate / 'Cargo.toml')
        shutil.copytree(source / 'crates/ropt/src', crate / 'src')
        shutil.copytree(source / 'crates/ropt/examples', crate / 'examples')
        target = crate / 'src/incremental_qr.rs'
        original = target.read_text()
        instrumented = original.replace('        let max_entries = self.options.max_entries;',
                                        '        let max_entries = self.options.max_entries;\n        let mut rotations = 0_u64;', 1)
        instrumented = instrumented.replace('                let old = &mut self.rows[k];',
                                            '                rotations += 1;\n                let old = &mut self.rows[k];', 1)
        needle = '        Ok(())\n    }\n    /// Solve the full-column-rank cached model.'
        require(needle in instrumented, 'QR diagnostic insertion point changed')
        instrumented = instrumented.replace(needle,
            '        eprintln!("qr_rotations={rotations}");\n' + needle, 1)
        target.write_text(instrumented)
        patch = out / (variant + '-rotations.patch')
        patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),
             instrumented.splitlines(True), fromfile='a/crates/ropt/src/incremental_qr.rs',
             tofile='b/crates/ropt/src/incremental_qr.rs')))
        with (out / (variant + '-build.log')).open('w') as log:
            subprocess.run(['cargo', 'build', '--manifest-path', str(snapshot / 'Cargo.toml'),
                            '-p', 'ropt', '--example', 'benchmark_warm_isam_se2', '--release',
                            '--locked', '--offline'], stdout=log, stderr=subprocess.STDOUT, check=True)
        exe = snapshot / 'target/release/examples/benchmark_warm_isam_se2'
        directory = out / variant
        with (out / (variant + '-rotations.log')).open('w') as log:
            subprocess.run([str(exe), str(primary / 'input.stream'), str(directory), '1', '1'],
                           stderr=log, check=True)
        counts = list(map(int, re.findall(r'^qr_rotations=(\d+)$',
                         (out / (variant + '-rotations.log')).read_text(), re.M)))
        calls = metadata['frames'] + 1
        require(len(counts) == 2 * calls and counts[:calls] == counts[calls:],
                'warm/measured rotation sequences differ or calls missing')
        poses, quality, reports, _ = validate(directory, metadata['frames'])
        expected = validate(primary / ('0-' + variant), metadata['frames'])
        require(poses == expected[0] and quality == expected[1], 'instrumented means/quality changed')
        for field in ('iterations', 'cost', 'entries', 'batch_trials', 'initial_damping', 'final_damping'):
            require([r[field] for r in reports] == [r[field] for r in expected[2]], 'report changed: ' + field)
        cases[variant] = {'rotations': sum(counts[calls:]), 'qr_appends': calls,
                          'primary_source_sha256': source_fingerprint(source),
                          'instrumented_source_sha256': source_fingerprint(snapshot),
                          'executable_sha256': sha(exe), 'forward_patch_sha256': sha(patch),
                          'full_means_quality_and_nontiming_reports_match_primary': True}
    report = {'purpose': __doc__, 'instrumented_times_excluded': True, 'cases': cases,
              'input_sha256': sha(primary / 'input.stream'), 'driver_sha256': sha(Path(__file__)),
              'primary_metadata_sha256': sha(primary / 'metadata.json')}
    (out / 'rotations.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
