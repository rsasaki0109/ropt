"""Audit retained before/after unused-refinement-step solve timing."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from paired_pcg import source_fingerprint
from pose_graph import ROOT, require
from verify_tree_refinement import audit_campaign


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'benchmarks/results/refinement_step_timing')
    parser.add_argument('--current-source', action='store_true')
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        require(digest(root / name) == expected, f'artifact changed: {name}')
    source = source_fingerprint(root / 'after_source')
    require(source == manifest['source_sha256'], 'changed source snapshot differs')
    if args.current_source:
        require(source_fingerprint(ROOT) == source, 'production source differs')
    with tempfile.TemporaryDirectory(prefix='ropt-refinement-before-') as temporary:
        before = Path(temporary) / 'source'
        shutil.copytree(root / 'after_source', before)
        shutil.copyfile(root / 'isam.rs.before', before / 'crates/ropt/src/isam.rs')
        require(source_fingerprint(before) == manifest['before_source_sha256'], 'preceding source cannot be reconstructed')
    plan = json.loads((root / 'plan.json').read_text())
    decision = json.loads((root / 'decision.json').read_text())
    require(plan['primary_pairs'] == plan['confirmation_pairs'] == 6 and plan['maximum_campaigns'] == 2, 'study plan changed')
    require(plan['before_source_sha256'] == manifest['before_source_sha256'] and plan['source_sha256'] == source, 'plan source differs')
    names = ['primary'] + (['confirmation'] if decision['confirmation_started'] else [])
    require((root / 'confirmation').exists() == decision['confirmation_started'], 'confirmation presence differs from decision')
    result = {}
    for name in names:
        directory = root / name
        metadata = json.loads((directory / 'metadata.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        require(metadata['source_sha256'] == source and metadata['before_source_sha256'] == manifest['before_source_sha256'], 'campaign kernel differs')
        require(metadata['cpu_affinity'] == [0] and metadata['warmup_streams_per_process'] == metadata['timed_streams_per_process'] == 1, 'process policy differs')
        require(metadata['ad_chunk_size'] == 6 and metadata['ad_edge_stride'] == 2 and metadata['selected_cap'] == 0, 'mixed AD policy differs')
        for path, expected in metadata['files_sha256'].items():
            require(digest(root / 'frozen_tools' / path) == expected, 'measured runner/driver differs')
        require(report['before_after_means_byte_identical'] and report['before_after_non_timing_fields_identical'], 'reported exact equality failed')
        require(gzip.decompress((directory / 'before-poses.csv.gz').read_bytes()) == gzip.decompress((directory / 'after-poses.csv.gz').read_bytes()), 'before/after full poses differ')
        with (directory / 'updates.csv').open() as stream:
            updates = list(csv.DictReader(stream))
        for index in range(6):
            work = {v: [{k: value for k, value in row.items() if k not in ('variant', 'round', 'elapsed_ns')}
                        for row in updates if row['variant'] == v and int(row['round']) == index]
                    for v in ('before', 'after')}
            require(work['before'] == work['after'], 'before/after work differs')
        result[name] = audit_campaign(directory)
    require(result['primary']['qualifies'] == decision['confirmation_started'], 'conditional confirmation rule violated')
    require(result['primary']['faster_pairs'] == decision['primary_faster_pairs'], 'decision pair count differs')
    result['scope'] = 'before/after ropt wall times; archived native accuracy only; no fresh GTSAM timing'
    result['repeated_speed_improvement_confirmed'] = len(names) == 2 and all(result[name]['qualifies'] for name in names)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
