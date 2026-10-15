"""Audit retained single-factorization refresh experiments and original accuracy gates."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile

from check_real_replay import assess
from paired_pcg import source_fingerprint
from paired_isam2 import agreement, validate
from paired_stationarity_se2 import reference
from pose_graph import ROOT, require
from real_se2_stream import column_normalized_gradient, load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit_campaign(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    require(metadata['rounds'] == 6 and report['eligible'], 'incomplete campaign')
    require(not metadata['native_timing_measured'] and not metadata['allocator_counting'], 'timing scope changed')
    source = ROOT / 'benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o'
    problem = load(source)
    frames = len(problem[1])
    require(digest(source) == metadata['dataset']['source_sha256'], 'dataset changed')
    require(digest(directory / 'input.stream') == metadata['dataset']['stream_sha256'], 'stream changed')
    native, native_quality, hashes = reference(ROOT / metadata['native_reference_directory'], metadata['dataset'], frames)
    require(hashes == metadata['native_reference_sha256'], 'native archive changed')
    samples, updates, quality = [rows(directory / name) for name in ('samples.csv', 'updates.csv', 'quality.csv')]
    require(len(samples) == 12 and len(updates) == len(quality) == frames * 12, 'missing timed records')
    means, checks, output = {}, {}, {}
    for variant in ('before', 'after'):
        canonical_work = None
        canonical_quality = None
        elapsed = []
        for index in range(6):
            sample = [r for r in samples if r['variant'] == variant and int(r['round']) == index]
            raw = [r for r in updates if r['variant'] == variant and int(r['round']) == index]
            q = [r for r in quality if r['variant'] == variant and int(r['round']) == index]
            require(len(sample) == 1 and len(raw) == len(q) == frames, 'missing round')
            require(int(sample[0]['position']) == (index % 2 if variant == 'before' else 1-index % 2), 'unbalanced order')
            require([int(r['frame']) for r in raw] == list(range(1, frames + 1)), 'update order changed')
            require(int(sample[0]['elapsed_ns']) == sum(int(r['elapsed_ns']) for r in raw), 'timer sum changed')
            require(all(int(r['elapsed_ns']) > 0 and int(r['allocations']) == int(r['requested_bytes']) == 0 for r in raw), 'invalid timers/allocator mode')
            require(all(float(r['gradient_norm']) <= 8e-8 and int(r['requested_calls']) == 0 for r in raw), 'internal policy changed')
            work = [{k: v for k, v in r.items() if k not in ('variant', 'round', 'elapsed_ns')} for r in raw]
            q_values = [{k: v for k, v in r.items() if k not in ('variant', 'round')} for r in q]
            if index == 0:
                canonical_work, canonical_quality = work, q_values
                with tempfile.TemporaryDirectory(prefix='ropt-tree-audit-') as temporary:
                    path = Path(temporary)
                    (path / 'poses.csv').write_bytes(gzip.decompress((directory / f'{variant}-poses.csv.gz').read_bytes()))
                    write_csv(path / 'updates.csv', raw)
                    write_csv(path / 'samples.csv', sample)
                    poses, checked, _, _ = validate(path, frames, problem, column_normalized_gradient, 1e-7)
                for actual, stored in zip(checked, q):
                    require(all(float(stored[k]) == value for k, value in actual.items()), 'retained quality differs from recomputation')
                means[variant], checks[variant] = poses, checked
                agreement(poses, native, checked, native_quality)
            else:
                require(work == canonical_work and q_values == canonical_quality, 'within-variant recorded results changed')
            elapsed.append(int(sample[0]['elapsed_ns']))
        require(math.isclose(statistics.median(elapsed) / 1e6, report['cases'][variant]['timing']['median_ms'], rel_tol=1e-15), 'reported median changed')
        arrival_times = sorted(int(r['elapsed_ns']) / 1e6 for r in updates if r['variant'] == variant)
        output[variant] = {'stream_median_ms': statistics.median(elapsed) / 1e6,
                           'arrival_median_ms': statistics.median(arrival_times),
                           'arrival_p95_ms': arrival_times[math.ceil(.95 * len(arrival_times))-1],
                           'arrival_p99_ms': arrival_times[math.ceil(.99 * len(arrival_times))-1],
                           'arrival_max_ms': max(arrival_times)}
    agreement(means['after'], means['before'], checks['after'], checks['before'])
    ratio = output['before']['stream_median_ms'] / output['after']['stream_median_ms']
    require(math.isclose(ratio, report['cases']['after']['speedup_over_before'], rel_tol=1e-15), 'reported ratio changed')
    by_round = {v: {int(r['round']): int(r['elapsed_ns']) for r in samples if r['variant'] == v} for v in ('before', 'after')}
    faster = sum(by_round['after'][i] < by_round['before'][i] for i in range(6))
    require(faster == report['cases']['after']['faster_pairs_over_before'], 'pair count changed')
    require(report['cases']['after']['beats_before_in_this_case'] == (ratio > 1 and faster >= 5), 'qualification changed')
    output.update(time_reduction_percent=100*(1-1/ratio), faster_pairs=faster,
                  qualifies=ratio > 1 and faster >= 5, timed_arrivals=len(updates))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'benchmarks/results/tree_refinement')
    parser.add_argument('--current-source', action='store_true')
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        require(digest(root / name) == expected, f'artifact changed: {name}')
    for name, expected in manifest['tools_sha256'].items():
        require(digest(ROOT / name) == expected, f'audit/runner changed: {name}')
    candidate = source_fingerprint(root / 'candidate_source')
    require(candidate == manifest['source_sha256'], 'candidate archive changed')
    if args.current_source:
        require(source_fingerprint(ROOT) == candidate, 'production source changed')
    with tempfile.TemporaryDirectory(prefix='ropt-tree-before-audit-') as temporary:
        before = Path(temporary) / 'source'
        shutil.copytree(root / 'candidate_source', before)
        shutil.copy2(root / 'isam.rs.before', before / 'crates/ropt/src/isam.rs')
        require(source_fingerprint(before) == manifest['before_source_sha256'], 'before archive cannot be reconstructed')
    functional_report = json.loads((root / 'functional/report.json').read_text())
    with tempfile.TemporaryDirectory(prefix='ropt-tree-functional-source-') as temporary:
        snapshot = Path(temporary) / 'source'
        shutil.copytree(root / 'candidate_source', snapshot)
        shutil.copy2(root / 'functional/isam.rs', snapshot / 'crates/ropt/src/isam.rs')
        require(source_fingerprint(snapshot) == functional_report['source_sha256'], 'functional source cannot be reconstructed')
    for name, source, native_dir in [
        ('csail', 'csail_real_isam2/csail-se-sync1045-original.g2o', 'csail_real_isam2/primary'),
        ('mit', 'real_isam2/intel-mit1228-original.g2o', 'precise_real_isam2/primary'),
    ]:
        recorded = functional_report['datasets'][name]
        problem = load(ROOT / 'benchmarks/results' / source)
        poses, quality, summary = assess(root / 'functional' / name / 'replay', problem)
        require(summary == recorded['quality'] and summary['eligible_quality'], 'functional quality changed')
        native, native_quality, hashes = reference(ROOT / 'benchmarks/results' / native_dir, recorded['dataset'], len(problem[1]))
        require(hashes == recorded['native_reference_sha256'], 'functional native reference changed')
        require(agreement(poses, native, quality, native_quality) == recorded['native_agreement'], 'functional native agreement changed')
    for name in ('primary', 'confirmation'):
        metadata = json.loads((root / name / 'metadata.json').read_text())
        require(metadata['source_sha256'] == candidate and metadata['before_source_sha256'] == manifest['before_source_sha256'], 'campaign source differs')
    result = {name: audit_campaign(root / name) for name in ('primary', 'confirmation')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
