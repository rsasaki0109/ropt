"""Alternate current-build QR and Gaussian tree with matched original-model gates."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--branches', type=int, default=4)
    parser.add_argument('--leaves', type=int, default=64)
    parser.add_argument('--updates', type=int, default=120)
    args = parser.parse_args()
    require(args.rounds > 1 and args.branches > 0 and args.leaves > 0 and args.updates > 0, 'invalid workload')
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), 'measurement output must be outside checkout')
    out.mkdir()
    executable = args.executable.resolve()
    fingerprint = source_fingerprint(ROOT)
    metadata = {'purpose': 'current-build Gaussian QR versus selective tree; no native/nonlinear speed claim',
                'source_sha256': fingerprint, 'executable_sha256': digest(executable),
                'files_sha256': {name: digest(ROOT/name) for name in (
                    'benchmarks/paired_bayes_tree.py', 'crates/ropt/examples/benchmark_bayes_tree.rs')},
                'environment': environment(), 'branches': args.branches, 'leaves': args.leaves,
                'scalar_columns': args.branches*(args.leaves+1), 'updates_per_stream': args.updates,
                'rounds': args.rounds, 'order': 'AB/BA alternating, qr first in even rounds',
                'warmup_streams_per_process': 1, 'timed_streams_per_process': 1,
                'allocation_instrumentation': False,
                'timing': 'public atomic append + full solve + cached cost; original checks/output excluded',
                'gates': {'original_gradient_inf': 1e-8, 'reported_cost_relative': 1e-10,
                          'full_estimate_difference_inf': 2e-9, 'original_cost_relative': 1e-10}}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    samples, agreements = [], []
    tree_work = []
    for round_index in range(args.rounds):
        current = {}
        order = ('qr', 'tree') if round_index % 2 == 0 else ('tree', 'qr')
        for variant in order:
            directory = out/f'{round_index}-{variant}'
            subprocess.run([str(executable), variant, str(args.branches), str(args.leaves),
                            str(args.updates), str(directory)], check=True)
            updates = read(directory/'updates.csv')
            means = read(directory/'means.csv')
            sample = read(directory/'samples.csv')
            require(len(updates) == args.updates and len(sample) == 1, 'incomplete stream')
            require(len(means) == args.updates*metadata['scalar_columns'], 'incomplete estimates')
            require([int(r['frame']) for r in updates] == list(range(1,args.updates+1)), 'update layout differs')
            require([(int(r['frame']),int(r['column'])) for r in means] ==
                    [(f,c) for f in range(1,args.updates+1) for c in range(metadata['scalar_columns'])], 'estimate layout differs')
            require(int(sample[0]['elapsed_ns']) == sum(int(r['elapsed_ns']) for r in updates), 'timers differ')
            require(int(sample[0]['updates']) == args.updates, 'sample count differs')
            require(all(math.isfinite(float(r['value'])) for r in means), 'nonfinite estimate')
            for row in updates:
                require(int(row['elapsed_ns']) > 0, 'nonpositive elapsed time')
                require(all(math.isfinite(float(row[k])) for k in ('cost','original_cost','gradient_inf')), 'nonfinite quality')
                require(float(row['gradient_inf']) <= 1e-8, 'gradient fails')
                require(abs(float(row['cost'])-float(row['original_cost'])) <=
                        1e-10*max(1.,float(row['original_cost'])), 'reported cost fails')
                if variant == 'tree':
                    require(int(row['reeliminated_cliques'])+int(row['reused_cliques']) == metadata['scalar_columns'], 'work counts differ')
                    tree_work.append(int(row['reeliminated_cliques']))
            current[variant] = (updates, means)
            samples.append({'round': round_index, 'variant': variant, 'elapsed_ns': int(sample[0]['elapsed_ns'])})
            with (directory/'means.csv').open('rb') as source, (directory/'means.csv.gz').open('wb') as sink, gzip.GzipFile(fileobj=sink,mode='wb',mtime=0) as archive:
                archive.write(source.read())
            (directory/'means.csv').unlink()
        a, b = current['qr'], current['tree']
        for k in range(args.updates):
            start = k*metadata['scalar_columns']; end = start+metadata['scalar_columns']
            gap = max(abs(float(x['value'])-float(y['value'])) for x,y in zip(a[1][start:end],b[1][start:end]))
            cost = abs(float(a[0][k]['original_cost'])-float(b[0][k]['original_cost']))
            require(gap <= 2e-9 and cost <= 1e-10*max(1.,float(a[0][k]['original_cost'])), 'paired estimates/cost differ')
            agreements.append({'round':round_index,'frame':k+1,'estimate_difference_inf':gap,'original_cost_difference':cost})
    require(source_fingerprint(ROOT) == fingerprint, 'source changed during campaign')
    require(digest(executable) == metadata['executable_sha256'], 'executable changed during campaign')
    require(all(digest(ROOT/name) == value for name,value in metadata['files_sha256'].items()), 'driver changed during campaign')
    for name, rows in [('samples.csv',samples), ('agreement.csv',agreements)]:
        with (out/name).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    times = {v: [r['elapsed_ns']/1e6 for r in samples if r['variant']==v] for v in ('qr','tree')}
    report = {'all_original_quality_and_pair_agreement_gates_pass': True,
              'median_ms': {v:statistics.median(t) for v,t in times.items()},
              'median_paired_qr_over_tree_ratio': statistics.median(a/b for a,b in zip(times['qr'],times['tree'])),
              'tree_faster_pairs': sum(b<a for a,b in zip(times['qr'],times['tree'])),
              'pairs':args.rounds, 'max_estimate_difference_inf':max(r['estimate_difference_inf'] for r in agreements),
              'max_original_cost_difference':max(r['original_cost_difference'] for r in agreements),
              'tree_reeliminated_cliques': {'min':min(tree_work),'median':statistics.median(tree_work),'max':max(tree_work)},
              'total_cliques': metadata['scalar_columns'],
              'scope':'synthetic scalar Gaussian forest; current-build QR baseline; no iSAM2 or nonlinear superiority claim'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
