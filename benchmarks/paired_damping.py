"""Balanced restart/reused-damping/native comparisons with unchanged accuracy gates."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from paired_isam2 import agreement, validate
from paired_pcg import source_fingerprint
from pose_graph import ROOT, environment, require, timing_summary
from se2_stream import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','before-source','restart','reuse','native','native-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--rounds',type=int,default=8)
    args=p.parse_args()
    require(25<=args.frames<=300 and args.frames%25==0 and args.rounds>0 and args.rounds%8==0,'invalid/unbalanced sweep')
    out=args.output.resolve();require(not out.is_relative_to(ROOT),'output outside checkout required');out.mkdir()
    write(out/'input.stream',args.frames)
    common=('crates/ropt/examples/support/se2_benchmark.rs','crates/ropt/examples/support/allocation_counter.rs')
    for name in common:
        require((ROOT/name).read_bytes()==(args.before_source/name).read_bytes(),'before/current shared templates differ')
    binaries={'before':args.before,'restart':args.restart,'reuse':args.reuse,'native':args.native}
    native=json.loads(args.native_manifest.read_text())
    require(native['executable_sha256']==hashlib.sha256(args.native.read_bytes()).hexdigest(),'native manifest mismatch')
    templates=(*common,'crates/ropt/examples/benchmark_isam_se2.rs','crates/ropt/examples/benchmark_restart_isam_se2.rs','crates/ropt/examples/benchmark_warm_isam_se2.rs','benchmarks/paired_damping.py','benchmarks/paired_isam2.py','benchmarks/se2_stream.py')
    metadata={'purpose':'explicit damping policy change, not a same-algorithm speedup; same input/options/accuracy gates',
              'frames':args.frames,'rounds':args.rounds,'warmup_streams':1,'timed_streams_per_process':1,
              'environment':environment(),'native_manifest':native,
              'source_sha256':{'before':source_fingerprint(args.before_source),'current':source_fingerprint(ROOT)},
              'files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in templates},
              'executable_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
              'rollback_patch_sha256':hashlib.sha256((ROOT/'benchmarks/isam_damping_reuse.patch').read_bytes()).hexdigest(),
              'input_sha256':hashlib.sha256((out/'input.stream').read_bytes()).hexdigest(),
              'gates':{'original_gradient_inf':1e-6,'world_pose_norm_difference':2e-6,'relative_objective_difference':1e-9,'relative_reported_cost_difference':1e-10,'root':'exact'},
              'policies':{'before':'previous kernel, restart','restart':'current kernel, default restart','reuse':'current kernel, explicit adaptive LM damping reuse','native':'unchanged native iSAM2'},
              'timed':'complete update/full owned report; native extra updates, stationarity feedback and full means; constructor, recipe creation, telemetry extraction, quality checks, serialization and report destruction excluded',
              'allocator_counting':False,'memory_measured':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    variants=tuple(binaries);samples=[];updates=[];quality=[];comparisons=[]
    for round_index in range(args.rounds):
        offset=round_index%4;order=variants[offset:]+variants[:offset]
        if (round_index//4)%2:order=tuple(reversed(order))
        checked={}
        for position,name in enumerate(order):
            directory=out/f'{round_index}-{name}'
            subprocess.run([str(binaries[name].resolve()),str(out/'input.stream'),str(directory),'1','1'],check=True)
            poses,q,u,sample=validate(directory,args.frames)
            if name in ('restart','reuse'):
                previous=1e-3
                for r in u:
                    require(r['batch_trials']!='' and 0<=int(r['batch_trials'])<=200,'missing batch telemetry')
                    initial=float(r['initial_damping']);final=float(r['final_damping'])
                    require(math.isfinite(initial) and math.isfinite(final) and initial>0 and final>0,'invalid damping')
                    require(initial==(previous if name=='reuse' else 1e-3),'damping policy not followed')
                    previous=final
            checked[name]=(poses,q)
            samples.append({'variant':name,'round':round_index,'position':position,**sample})
            updates.extend({'variant':name,'round':round_index,**{k:r.get(k,'') for k in ('sample','frame','elapsed_ns','iterations','cost','entries','allocations','requested_bytes','batch_trials','initial_damping','final_damping')}} for r in u)
            quality.extend({'variant':name,'round':round_index,**r} for r in q)
            if round_index==0:
                with gzip.open(out/f'{name}-poses.csv.gz','wb') as f:f.write((directory/'poses.csv').read_bytes())
                (out/f'{name}-options.txt').write_bytes((directory/'options.txt').read_bytes())
            print(f'round {round_index} {name}: all {args.frames} prefixes eligible',flush=True)
        for a,b in [('before','restart'),('restart','reuse'),('reuse','native')]:
            comparisons.append({'round':round_index,'a':a,'b':b,**agreement(checked[a][0],checked[b][0],checked[a][1],checked[b][1])})
    report={'eligible':True,'timed_updates':args.rounds*4*args.frames,
            'cases':{name:{'timing':timing_summary([r for r in samples if r['variant']==name]),
                     'max_gradient':max(r['gradient_inf'] for r in quality if r['variant']==name),
                     'batch_trials_per_stream':sum(int(r['batch_trials']) for r in updates if r['variant']==name and r['round']==0) if name in ('restart','reuse') else None} for name in variants},
            'comparisons':comparisons}
    for name,rows in [('samples.csv',samples),('updates.csv',updates),('quality.csv',quality)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['cases'],indent=2))


if __name__=='__main__':main()
