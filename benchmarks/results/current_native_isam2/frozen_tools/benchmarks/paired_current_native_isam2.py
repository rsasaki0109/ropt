"""Fresh accuracy-matched CSAIL replay: selected mixed-AD ropt vs native reference.

Native uses full relinearization each arrival and timed empty updates to reach
its stationarity target. This is not a comparison against default iSAM2 policy.
"""
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
    parser.add_argument('--native-library-path', type=Path, action='append', default=[])
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    require(args.rounds == 6, 'planned campaign requires six alternating pairs')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'RAYON_NUM_THREADS': '1', 'ROPT_BENCH_AD_CHUNK': '6', 'ROPT_BENCH_AD_STRIDE': '2'}
    if args.native_library_path:
        env['LD_LIBRARY_PATH'] = ':'.join(str(p.resolve()) for p in args.native_library_path) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    manifest = json.loads(args.native_manifest.read_text())
    require(manifest['exact_between'] and manifest['options']['CMAKE_BUILD_TYPE'] == 'Release', 'ineligible native build')
    require(digest(args.native) == manifest['executable_sha256'], 'native executable differs from manifest')
    linked = subprocess.check_output(['ldd', str(args.native.resolve())], env=env, text=True)
    require('not found' not in linked, 'native runtime dependency missing')
    libraries = {line.split('=>', 1)[1].strip().split()[0]: None for line in linked.splitlines() if '=>' in line}
    libraries = {p: digest(p) for p in libraries if p.startswith('/')}
    gtsam = [p for p in libraries if Path(p).name.startswith('libgtsam.so')]
    require(len(gtsam) == 1 and libraries[gtsam[0]] == manifest['library_sha256'], 'native library differs from manifest')
    reference = ROOT / manifest['reference_source_directory']
    require(reference.resolve().is_relative_to(ROOT / 'benchmarks'), 'reference outside checkout')
    for name, expected in manifest['reference_sources_sha256'].items():
        require(digest(reference / name) == expected, 'native reference sources changed')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'output outside checkout required')
    out.mkdir()
    dataset = write(args.input, out / 'input.stream')
    require(dataset['complete_dataset'], 'partial dataset ineligible')
    problem = load(args.input)
    frames = len(problem[1])
    files = ('crates/ropt/examples/benchmark_ad_selective_se2_isam.rs',
             'crates/ropt/examples/support/real_selective_se2.rs', 'crates/ropt/examples/support/real_se2_benchmark.rs',
             'crates/ropt/examples/support/allocation_counter.rs', 'benchmarks/paired_current_native_isam2.py',
             'benchmarks/paired_isam2.py', 'benchmarks/real_se2_stream.py', 'benchmarks/se2_stream.py', 'benchmarks/information_precision.py')
    metadata = {'dataset': dataset, 'source_sha256': source_fingerprint(ROOT),
                'executable_sha256': {'ropt': digest(args.ropt), 'native': digest(args.native)},
                'native_manifest': manifest, 'native_manifest_sha256': digest(args.native_manifest),
                'linked_libraries_sha256': libraries, 'files_sha256': {name: digest(ROOT / name) for name in files},
                'environment': environment(), 'rounds': 6, 'cpu_affinity': [args.cpu],
                'warmup_streams_per_process': 1, 'timed_streams_per_process': 1,
                'ropt_policy': 'editable selective mixed AD chunk6 stride2; explicit requested passes zero, implicit automatic threshold passes retained; explicit recovery if target unmet; periodic refresh disabled; frozen editable-tree refinement four steps; internal column-normalized gradient target 8e-8',
                'native_policy': 'GTSAM 4.2.1 exact BetweenFactor Jacobian; compensated information adapter; QR/GN, full relinearization every arrival and timed empty updates to internal normalized target 8e-8, max 200 calls; not default iSAM2',
                'gates': {'original_column_normalized_gradient_inf': 1e-7, 'world_pose_norm_difference': 2e-6,
                          'relative_cost_difference': 1e-9, 'relative_reported_cost_difference': 1e-10, 'fixed_root': 'exact'},
                'ropt_ad_factors': (sum(len(e) for _, e in problem[1]) + 1) // 2,
                'factors': sum(len(e) for _, e in problem[1]),
                'native_runtime_search_paths': [str(p.resolve()) for p in args.native_library_path],
                'native_jacobians': 'analytic', 'fresh_native_timing': True,
                'timed': 'complete updates, nonlinear probes, recovery/empty updates, owned estimates and report; parser, factor construction, external validation and serialization excluded',
                'allocator_counting': False, 'memory_measured': False,
                'scope': 'complete recorded SE2 stream at common accuracy; explicit selected policies, no general/default-policy superiority claim'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    samples, updates, quality, comparisons = [], [], [], []
    first_poses, first_work = {}, {}
    for index in range(6):
        measured = {}
        order = ('ropt', 'native') if index % 2 == 0 else ('native', 'ropt')
        for position, variant in enumerate(order):
            directory = out / f'{index}-{variant}'
            binary = args.ropt if variant == 'ropt' else args.native
            command = ['taskset', '-c', str(args.cpu), str(binary.resolve()), str(out / 'input.stream'), str(directory), '1', '1']
            if variant == 'ropt':
                command += ['editable', '0']
            with (out / f'{index}-{variant}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            poses, checks, raw, sample = validate(directory, frames, problem, column_normalized_gradient, 1e-7)
            if variant == 'ropt':
                require(all(float(r['gradient_norm']) <= 8e-8 and int(r['requested_calls']) == 0 for r in raw), 'ropt policy changed')
                require(all(r['initial_damping'] == r['final_damping'] == '' for r in raw), 'unexpected damping')
                differentiation = (directory / 'differentiation.txt').read_text()
                require('chunk_size=6; edge_stride=2;' in differentiation and f'ad_factors={metadata["ropt_ad_factors"]};' in differentiation, 'AD workload changed')
            pose_bytes = (directory / 'poses.csv').read_bytes()
            work = [{k: v for k, v in r.items() if k != 'elapsed_ns'} for r in raw]
            if index == 0:
                first_poses[variant], first_work[variant] = pose_bytes, work
                with (out / f'{variant}-poses.csv.gz').open('wb') as destination:
                    with gzip.GzipFile(fileobj=destination, mode='wb', filename='', mtime=0) as compressed:
                        compressed.write(pose_bytes)
                for filename in ('options.txt', 'differentiation.txt'):
                    if (directory / filename).exists():
                        shutil.copyfile(directory / filename, out / f'{variant}-{filename}')
            else:
                require(pose_bytes == first_poses[variant] and work == first_work[variant], 'within-variant results changed')
            measured[variant] = poses, checks
            samples.append({'variant': variant, 'round': index, 'position': position, **sample})
            updates.extend({'variant': variant, 'round': index, **r} for r in raw)
            quality.extend({'variant': variant, 'round': index, **r} for r in checks)
            print(f'round {index} {variant}: {int(sample["elapsed_ns"])/1e9:.6f}s; all {frames} original-information prefixes pass', flush=True)
        comparisons.append({'round': index, **agreement(measured['ropt'][0], measured['native'][0], measured['ropt'][1], measured['native'][1])})
        for variant in order:
            shutil.rmtree(out / f'{index}-{variant}')
    require(source_fingerprint(ROOT) == metadata['source_sha256'], 'solver source changed')
    require(all(digest(ROOT / name) == expected for name, expected in metadata['files_sha256'].items()), 'runner/driver changed')
    require(digest(args.native_manifest) == metadata['native_manifest_sha256'], 'native manifest changed')
    require(all(digest(p) == expected for p, expected in libraries.items()), 'linked runtime changed')
    require(all(digest(p) == metadata['executable_sha256'][v] for v, p in [('ropt', args.ropt), ('native', args.native)]), 'executable changed')
    report = {'eligible': True, 'fresh_native_timing': True, 'within_variant_results_identical': True,
              'timed_updates': 12 * frames, 'comparisons': comparisons,
              'cases': {v: {'timing': timing_summary([s for s in samples if s['variant'] == v]),
                            'max_normalized_gradient': max(r['gradient_inf'] for r in quality if r['variant'] == v)} for v in ('ropt', 'native')}}
    times = {v: {int(r['round']): int(r['elapsed_ns']) for r in samples if r['variant'] == v} for v in ('ropt', 'native')}
    report['ropt_faster_pairs'] = sum(times['ropt'][i] < times['native'][i] for i in range(6))
    report['native_median_over_ropt_median'] = report['cases']['native']['timing']['median_ms'] / report['cases']['ropt']['timing']['median_ms']
    report['ropt_qualifies_in_this_case'] = report['native_median_over_ropt_median'] > 1 and report['ropt_faster_pairs'] >= 5
    for name, records in [('samples.csv', samples), ('updates.csv', updates), ('quality.csv', quality)]:
        with (out / name).open('w') as stream:
            fields = list(dict.fromkeys(k for r in records for k in r))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
