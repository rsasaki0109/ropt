"""Separate allocation traffic with exact agreement to accepted cached timings."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess

from paired_anchor_coordinates import sha
from pose_graph import ROOT, require


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def non_timing(row):
    return {k: v for k, v in row.items() if k not in
            ('variant', 'round', 'elapsed_ns', 'allocation_calls', 'allocation_bytes')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'primary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary.resolve()
    metadata = json.loads((primary / 'metadata.json').read_text())
    proof = json.loads((primary / 'report.json').read_text())
    require(proof['eligible'] and proof['all_prefix_means_reports_and_quality_exact'] and
            not metadata['allocation_counting'], 'accepted uninstrumented comparison required')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'output must be outside checkout')
    out.mkdir()
    reference_updates = rows(primary / 'updates.csv')
    reference_quality = rows(primary / 'quality.csv')
    cases = {}
    for variant, binary in [('before', args.before), ('after', args.after)]:
        require(sha(binary) == metadata['executable_sha256'][variant], 'primary executable changed')
        directory = out / variant
        subprocess.run([str(binary.resolve()), str(metadata['frames']), str(directory), '--allocations'],
                       check=True)
        updates = rows(directory / 'updates.csv')
        expected = [non_timing(r) for r in reference_updates
                    if r['variant'] == variant and r['round'] == '0']
        require([non_timing(r) for r in updates] == expected, 'non-timing report fields changed')
        expected_quality = [{k: v for k, v in r.items() if k not in ('variant', 'round')}
                            for r in reference_quality if r['variant'] == variant and r['round'] == '0']
        require(rows(directory / 'quality.csv') == expected_quality, 'quality records changed')
        with gzip.open(primary / f'{variant}-means.csv.gz', 'rb') as stream:
            require((directory / 'means.csv').read_bytes() == stream.read(), 'full means changed')
        require((directory / 'summary.txt').read_bytes() ==
                (primary / f'{variant}-summary.txt').read_bytes(), 'accuracy summary changed')
        require((directory / 'options.txt').read_text().replace('allocation_counting=true',
                'allocation_counting=false') == (primary / f'{variant}-options.txt').read_text(),
                'diagnostic options changed')
        cases[variant] = {'allocation_calls': sum(int(r['allocation_calls']) for r in updates),
                          'allocation_bytes': sum(int(r['allocation_bytes']) for r in updates)}
        for name in ('updates.csv', 'quality.csv', 'summary.txt', 'options.txt'):
            (out / f'{variant}-{name}').write_bytes((directory / name).read_bytes())
        with gzip.open(out / f'{variant}-means.csv.gz', 'wb') as stream:
            stream.write((directory / 'means.csv').read_bytes())
    predicted_calls = sum(int(r['frame']) * (2 if r['reason'] == 'None' else 1)
                          for r in reference_updates if r['variant'] == 'after' and r['round'] == '0')
    report = {'eligible': True, 'purpose': __doc__, 'frames': metadata['frames'],
              'instrumented_times_excluded': True, 'peak_memory_or_rss_measured': False,
              'native_memory_measured': False, 'full_means_reports_and_quality_equal_primary': True,
              'cases': cases,
              'saved_allocation_calls': cases['before']['allocation_calls'] - cases['after']['allocation_calls'],
              'saved_allocation_bytes': cases['before']['allocation_bytes'] - cases['after']['allocation_bytes'],
              'static_fixture_prediction': {'removed_local_vectors': predicted_calls,
                                            'removed_local_bytes': predicted_calls * 3 * 8},
              'primary_metadata_sha256': sha(primary / 'metadata.json'),
              'primary_report_sha256': sha(primary / 'report.json'),
              'source_sha256': metadata['source_sha256'],
              'executable_sha256': metadata['executable_sha256'],
              'driver_sha256': sha(Path(__file__)),
              'runner_files_sha256': metadata['files_sha256']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
