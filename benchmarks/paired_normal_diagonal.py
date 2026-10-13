"""Same-policy redundant normal-graph diagonal lookup elimination comparison with all-prefix accuracy gates."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require, timing_summary
from paired_isam2 import validate, agreement
from se2_stream import write


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
    templates=('crates/ropt/examples/benchmark_warm_isam_se2.rs','crates/ropt/examples/support/se2_benchmark.rs','crates/ropt/examples/support/allocation_counter.rs')
    for name in templates:
        require((args.before_source/name).read_bytes()==(ROOT/name).read_bytes(),'before/after templates differ')
    native_manifest=json.loads(args.native_manifest.read_text())
    require(native_manifest['executable_sha256']==hashlib.sha256(args.native.read_bytes()).hexdigest(),'native executable differs from manifest')
    linked = subprocess.check_output(['ldd', str(args.native.resolve())], text=True)
    libraries = [Path(line.split('=>', 1)[1].strip().split()[0])
                 for line in linked.splitlines() if 'libgtsam.so' in line and '=>' in line]
    require(len(libraries)==1 and libraries[0].is_file(), 'native GTSAM library unresolved')
    runtime_hash = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
    require(runtime_hash==native_manifest['library_sha256'], 'native runtime library differs from manifest')
    native_manifest = {**native_manifest, 'runtime_library_sha256':runtime_hash}
    metadata={'purpose':'same-input/same-policy redundant normal-graph diagonal lookup elimination optimization; both ropt kernels explicitly reuse LM damping and force every refresh; not default-policy timing',
              'frames':args.frames,'rounds':args.rounds,'warmup_streams_per_process':1,'timed_streams_per_process':1,
              'environment':environment(),'native_manifest':native_manifest,
              'gates':{'original_gradient_inf':1e-6,'world_pose_norm_difference':2e-6,'relative_cost_difference':1e-9,'relative_reported_cost_difference':1e-10,'fixed_root':'exact'},
              'source_sha256':{'before':source_fingerprint(args.before_source),'after':source_fingerprint(ROOT)},
              'executable_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('before',args.before),('after',args.after),('native',args.native)]},
              'files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (*templates,'benchmarks/se2_stream.py','benchmarks/paired_isam2.py','benchmarks/paired_normal_diagonal.py')},
              'input_sha256':hashlib.sha256((out/'input.stream').read_bytes()).hexdigest(),
              'timed':'complete accuracy-target update including full owned means/report; native additional empty updates, stationarity probes and full cost included; input parsing, factor recipes/graphs, initialization, external validation, serialization and report destruction excluded',
              'allocator_counting':False,'memory_measured':False,
              'rollback_patch_sha256':hashlib.sha256((ROOT/'benchmarks/normal_diagonal.patch').read_bytes()).hexdigest()}
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
            if variant in ('before','after'):
                previous=1e-3
                for row in raw:
                    initial=float(row['initial_damping']);final=float(row['final_damping'])
                    require(initial==previous and math.isfinite(final) and final>0,'damping reuse policy differs')
                    previous=final
            validated[variant]=(poses,quality,raw)
            samples.append({'variant':variant,'round':round_index,'position':position,**sample})
            updates.extend({'variant':variant,'round':round_index,**row} for row in raw)
            qualities.extend({'variant':variant,'round':round_index,**row} for row in quality)
            if round_index==0:
                with gzip.open(out/f'{variant}-poses.csv.gz','wb') as f: f.write((directory/'poses.csv').read_bytes())
                (out/f'{variant}-options.txt').write_bytes((directory/'options.txt').read_bytes())
            print(f'round {round_index} {variant}: all {args.frames} prefixes stationary',flush=True)
        require(validated['before'][0]==validated['after'][0], 'before/after means are not identical')
        require([q['cost'] for q in validated['before'][1]]==[q['cost'] for q in validated['after'][1]],'before/after original costs are not identical')
        for field in ('batch_trials','initial_damping','final_damping','iterations','cost','entries'):
            require([r[field] for r in validated['before'][2]]==[r[field] for r in validated['after'][2]],'before/after reports differ: '+field)
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
