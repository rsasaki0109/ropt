"""Recompute retained fresh-native comparison quality and paired timing summaries."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

from paired_isam2 import agreement, validate
from paired_pcg import source_fingerprint
from pose_graph import ROOT, require
from real_se2_stream import column_normalized_gradient, load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def campaign(directory, source):
    metadata = json.loads((directory / 'metadata.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    require(report['eligible'] and metadata['rounds'] == 6 and metadata['fresh_native_timing'], 'incomplete campaign')
    require(metadata['dataset']['source_sha256'] == '66d99ac857a9849d814d214a9ebd0d4876d5d40f0a37be9330c1ff6e6e9daaa6', 'unexpected CSAIL source')
    require(metadata['dataset']['stream_sha256'] == 'fe3ffed0fdae1d69d0e374a915f3364860602a7fe1ad34d19d6de7571e98718a', 'unexpected CSAIL stream')
    require(digest(source) == metadata['dataset']['source_sha256'], 'dataset changed')
    require(digest(directory / 'input.stream') == metadata['dataset']['stream_sha256'], 'stream changed')
    require(not metadata['allocator_counting'] and not metadata['memory_measured'], 'measurement scope changed')
    require(metadata['cpu_affinity'] == [0] and metadata['warmup_streams_per_process'] == metadata['timed_streams_per_process'] == 1, 'process policy changed')
    require(metadata['gates'] == {'original_column_normalized_gradient_inf': 1e-7,
                                 'world_pose_norm_difference': 2e-6, 'relative_cost_difference': 1e-9,
                                 'relative_reported_cost_difference': 1e-10, 'fixed_root': 'exact'}, 'quality gates changed')
    native = metadata['native_manifest']
    require(native['archive_sha256'] == '96ea77d620fc8c2e9ed79af1b4abfdc3e73d96ee7a581a2bc326963c43dd92c2', 'native source changed')
    require(native['exact_between'] and native['options']['GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR'] == 'ON', 'native Jacobian policy changed')
    require(native['options']['CMAKE_BUILD_TYPE'] == 'Release', 'native build type changed')
    for name, expected in native['reference_sources_sha256'].items():
        require(digest(directory.parent / 'native_reference' / name) == expected, 'native reference snapshot changed')
    problem = load(source)
    frames = len(problem[1])
    require(frames == 1044 and metadata['factors'] == 1172 and metadata['ropt_ad_factors'] == 586, 'workload changed')
    require(report['timed_updates'] == 12 * frames and report['within_variant_results_identical'], 'reported completeness changed')
    samples, updates, quality = [records(directory / name) for name in ('samples.csv', 'updates.csv', 'quality.csv')]
    require(len(samples) == 12 and len(updates) == len(quality) == 12 * frames, 'missing timed records')
    poses, checked, times, result = {}, {}, {}, {}
    for variant in ('ropt', 'native'):
        times[variant] = {}
        first_work, first_quality = None, None
        for index in range(6):
            sample = [r for r in samples if r['variant'] == variant and int(r['round']) == index]
            raw = [r for r in updates if r['variant'] == variant and int(r['round']) == index]
            q = [r for r in quality if r['variant'] == variant and int(r['round']) == index]
            require(len(sample) == 1 and len(raw) == len(q) == frames, 'missing round')
            require(int(sample[0]['position']) == (index % 2 if variant == 'ropt' else 1-index % 2), 'unbalanced order')
            require([int(r['frame']) for r in raw] == list(range(1, frames+1)), 'arrival order changed')
            require(int(sample[0]['elapsed_ns']) == sum(int(r['elapsed_ns']) for r in raw), 'timing sum changed')
            require(all(int(r['elapsed_ns']) > 0 and int(r['allocations']) == int(r['requested_bytes']) == 0 for r in raw), 'invalid timing/allocator mode')
            if variant == 'ropt':
                require(all(float(r['gradient_norm']) <= 8e-8 and int(r['requested_calls']) == 0 for r in raw), 'internal ropt policy changed')
                require(all(r['initial_damping'] == r['final_damping'] == '' for r in raw), 'unexpected damping')
            work = [{k: v for k, v in r.items() if k not in ('variant', 'round', 'elapsed_ns')} for r in raw]
            q_values = [{k: v for k, v in r.items() if k not in ('variant', 'round')} for r in q]
            if index == 0:
                first_work, first_quality = work, q_values
                with tempfile.TemporaryDirectory(prefix='ropt-native-audit-') as temporary:
                    root = Path(temporary)
                    (root / 'poses.csv').write_bytes(gzip.decompress((directory / f'{variant}-poses.csv.gz').read_bytes()))
                    write_csv(root / 'updates.csv', raw)
                    write_csv(root / 'samples.csv', sample)
                    p, checks, _, _ = validate(root, frames, problem, column_normalized_gradient, 1e-7)
                require(all(all(float(stored[k]) == v for k, v in actual.items()) for stored, actual in zip(q, checks)), 'stored quality differs from recomputation')
                poses[variant], checked[variant] = p, checks
            else:
                require(work == first_work and q_values == first_quality, 'within-variant records changed')
            times[variant][index] = int(sample[0]['elapsed_ns'])
        median = statistics.median(list(times[variant].values())) / 1e6
        require(math.isclose(median, report['cases'][variant]['timing']['median_ms'], rel_tol=1e-15), 'reported median changed')
        require(max(float(r['gradient_inf']) for r in quality if r['variant'] == variant) == report['cases'][variant]['max_normalized_gradient'], 'reported quality maximum changed')
        arrival = sorted(int(r['elapsed_ns']) / 1e6 for r in updates if r['variant'] == variant)
        result[variant] = {'stream_median_ms': median, 'arrival_median_ms': statistics.median(arrival),
                           'arrival_p95_ms': arrival[math.ceil(.95*len(arrival))-1],
                           'arrival_p99_ms': arrival[math.ceil(.99*len(arrival))-1], 'arrival_max_ms': max(arrival)}
    gap = agreement(poses['ropt'], poses['native'], checked['ropt'], checked['native'])
    require(all(all(c[k] == v for k, v in gap.items()) for c in report['comparisons']), 'reported agreement changed')
    faster = sum(times['ropt'][i] < times['native'][i] for i in range(6))
    ratio = result['native']['stream_median_ms'] / result['ropt']['stream_median_ms']
    require(faster == report['ropt_faster_pairs'] and math.isclose(ratio, report['native_median_over_ropt_median'], rel_tol=1e-15), 'paired summary changed')
    require(report['ropt_qualifies_in_this_case'] == (ratio > 1 and faster >= 5), 'qualification changed')
    result.update(native_median_over_ropt_median=ratio, ropt_faster_pairs=faster, timed_arrivals=len(updates), agreement=gap)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'benchmarks/results/current_native_isam2')
    parser.add_argument('--current-source', action='store_true')
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        require(digest(root / name) == expected, f'artifact changed: {name}')
    require(source_fingerprint(root / 'ropt_source') == manifest['source_sha256'], 'ropt source snapshot changed')
    if args.current_source:
        require(source_fingerprint(ROOT) == manifest['source_sha256'], 'production source changed')
    source = ROOT / 'benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o'
    for name in ('primary', 'confirmation'):
        metadata = json.loads((root / name / 'metadata.json').read_text())
        require(digest(root / 'native_build/native-manifest.json') == metadata['native_manifest_sha256'], 'native manifest archive differs')
        require(json.loads((root / 'native_build/native-manifest.json').read_text()) == metadata['native_manifest'], 'native build description differs')
        require(metadata['source_sha256'] == manifest['source_sha256'], 'campaign kernel differs')
        for path, expected in metadata['files_sha256'].items():
            require(digest(root / 'frozen_tools' / path) == expected, 'measured runner/driver snapshot changed')
    print(json.dumps({name: campaign(root / name, source) for name in ('primary', 'confirmation')}, indent=2))


if __name__ == '__main__':
    main()
