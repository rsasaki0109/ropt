"""Balanced native iSAM2 and before/after ropt accuracy-matched comparisons."""
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
from pose_graph import ROOT, environment, require, timing_summary
from se2_stream import energy_gradient, fixture, write, wrap


def validate(directory, n, problem=None, gradient_metric=None, gradient_tolerance=1e-6):
    root, frames = fixture(n) if problem is None else problem
    require(len(frames) == n, 'input stream length differs')
    poses = [[] for _ in range(n)]
    with (directory/'poses.csv').open() as f:
        for r in csv.DictReader(f):
            k, key = int(r['frame']), int(r['key'])
            require(1 <= k <= n and key == len(poses[k-1]), 'invalid pose ordering')
            p = tuple(float(r[name]) for name in ('x','y','angle'))
            require(all(math.isfinite(v) for v in p), 'nonfinite pose')
            poses[k-1].append(p)
    with (directory/'updates.csv').open() as f:
        updates = list(csv.DictReader(f))
    require(len(updates) == n and [int(r['frame']) for r in updates] == list(range(1,n+1)), 'missing measured update')
    edges = []
    quality = []
    for k, ((_, added), values, update) in enumerate(zip(frames, poses, updates), 1):
        require(len(values) == k+1 and values[0] == root, 'missing means or moved fixed root')
        edges.extend(added)
        cost, raw_gradient, g = energy_gradient(values, edges)
        gradient = raw_gradient if gradient_metric is None else gradient_metric(values, edges, g)
        reported = float(update['cost'])
        require(math.isfinite(cost) and math.isfinite(gradient) and math.isfinite(reported), 'nonfinite quality')
        require(gradient <= gradient_tolerance, f'nonstationary prefix {k}: {gradient}')
        require(abs(cost-reported) <= 1e-10*max(1.,cost), f'cost mismatch {k}')
        require(int(update['elapsed_ns']) > 0 and 1 <= int(update['iterations']) <= 200, 'invalid update timing/calls')
        quality.append({'frame':k,'cost':cost,'gradient_inf':gradient})
        if gradient_metric is not None:
            quality[-1]['raw_gradient_inf'] = raw_gradient
    with (directory/'samples.csv').open() as f:
        samples = list(csv.DictReader(f))
    require(len(samples)==1 and int(samples[0]['updates']) == n, 'missing measured stream')
    require(int(samples[0]['elapsed_ns']) == sum(int(r['elapsed_ns']) for r in updates), 'aggregate timer mismatch')
    require(all(int(r['allocations'])==0 and int(r['requested_bytes'])==0 for r in updates), 'allocator counting enabled during primary timing')
    return poses, quality, updates, samples[0]


def agreement(a, b, qa, qb):
    largest = 0.
    cost_difference = 0.
    for k, (ap,bp,ar,br) in enumerate(zip(a,b,qa,qb),1):
        delta = max(math.hypot(x[0]-y[0],x[1]-y[1],wrap(x[2]-y[2])) for x,y in zip(ap,bp))
        cost_delta = abs(ar['cost']-br['cost'])
        require(delta <= 2e-6, f'prefix means differ at {k}: {delta}')
        require(cost_delta <= 1e-9*max(1.,ar['cost']), f'prefix objectives differ at {k}')
        largest = max(largest,delta); cost_difference = max(cost_difference,cost_delta)
    return {'max_world_pose_norm':largest,'max_cost_difference':cost_difference}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','before-source','after','native','native-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--rounds',type=int,default=6)
    args=p.parse_args()
    require(25<=args.frames<=300 and args.frames%25==0 and args.rounds>0 and args.rounds%6==0,'invalid/unbalanced sweep')
    out=args.output.resolve(); require(not out.is_relative_to(ROOT),'output outside checkout required'); out.mkdir()
    write(out/'input.stream',args.frames)
    templates=('crates/ropt/examples/benchmark_isam_se2.rs','crates/ropt/examples/support/allocation_counter.rs')
    for name in templates:
        require((args.before_source/name).read_bytes()==(ROOT/name).read_bytes(),'before/after templates differ')
    native_manifest=json.loads(args.native_manifest.read_text())
    require(native_manifest['executable_sha256']==hashlib.sha256(args.native.read_bytes()).hexdigest(),'native executable differs from manifest')
    metadata={'purpose':'same-input all-history nonlinear stationarity after every arrival; not default-policy timing',
              'frames':args.frames,'rounds':args.rounds,'warmup_streams_per_process':1,'timed_streams_per_process':1,
              'environment':environment(),'native_manifest':native_manifest,
              'gates':{'original_gradient_inf':1e-6,'world_pose_norm_difference':2e-6,'relative_cost_difference':1e-9,'relative_reported_cost_difference':1e-10,'fixed_root':'exact'},
              'source_sha256':{'before':source_fingerprint(args.before_source),'after':source_fingerprint(ROOT)},
              'executable_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('before',args.before),('after',args.after),('native',args.native)]},
              'files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (*templates,'benchmarks/se2_stream.py','benchmarks/paired_isam2.py')},
              'input_sha256':hashlib.sha256((out/'input.stream').read_bytes()).hexdigest(),
              'timed':'complete accuracy-target update including full owned means/report; native additional empty updates, stationarity probes and full cost included; input parsing, factor recipes/graphs, initialization, external validation, serialization and report destruction excluded',
              'allocator_counting':False,'memory_measured':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    variants=('before','after','native'); samples=[]; updates=[]; comparisons=[]; qualities=[]
    binaries={'before':args.before,'after':args.after,'native':args.native}
    for round_index in range(args.rounds):
        offset=round_index%3
        order=variants[offset:]+variants[:offset]
        if (round_index//3)%2: order=tuple(reversed(order))
        validated={}
        for position,variant in enumerate(order):
            directory=out/f'{round_index}-{variant}'
            subprocess.run([str(binaries[variant].resolve()),str(out/'input.stream'),str(directory),'1','1'],check=True)
            poses, quality, raw, sample=validate(directory,args.frames)
            validated[variant]=(poses,quality)
            samples.append({'variant':variant,'round':round_index,'position':position,**sample})
            updates.extend({'variant':variant,'round':round_index,**row} for row in raw)
            qualities.extend({'variant':variant,'round':round_index,**row} for row in quality)
            if round_index==0:
                with gzip.open(out/f'{variant}-poses.csv.gz','wb') as f: f.write((directory/'poses.csv').read_bytes())
                (out/f'{variant}-options.txt').write_bytes((directory/'options.txt').read_bytes())
            print(f'round {round_index} {variant}: all {args.frames} prefixes stationary',flush=True)
        for a,b in [('before','after'),('after','native')]:
            comparisons.append({'round':round_index,'a':a,'b':b,**agreement(validated[a][0],validated[b][0],validated[a][1],validated[b][1])})
    report={'eligible':True,'cases':{v:{'timing':timing_summary([s for s in samples if s['variant']==v]),
                 'max_gradient':max(q['gradient_inf'] for q in qualities if q['variant']==v)} for v in variants},
            'comparisons':comparisons,'timed_updates':args.rounds*3*args.frames}
    for name,rows in [('samples.csv',samples),('updates.csv',updates),('quality.csv',qualities)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['cases'],indent=2))


if __name__=='__main__': main()
