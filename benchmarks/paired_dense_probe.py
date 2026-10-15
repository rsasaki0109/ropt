"""Compare complete mixed-AD real SE2 streams before/after contiguous-key probe indexing.

Native poses/costs are a hashed archived reference, not fresh native timings.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

from paired_pcg import source_fingerprint
from paired_isam2 import agreement, validate
from paired_stationarity_se2 import reference
from pose_graph import ROOT, environment, require, timing_summary
from real_se2_stream import column_normalized_gradient, load, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','before-binary','before-source','input','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--ad-chunk',type=int,default=6)
    p.add_argument('--ad-stride',type=int,default=2)
    p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--selected-cap',type=int,default=1)
    args=p.parse_args()
    require(args.rounds>0 and args.rounds%2==0,'need even balanced paired rounds')
    require(args.ad_chunk>0 and args.ad_stride>0,'invalid AD selectors')
    require(0<=args.selected_cap<=198,'invalid selected cap')
    require(all(os.access(path,os.X_OK) for path in (args.binary,args.before_binary)),'comparison executable missing or not executable')
    out=args.output.resolve();require(not out.is_relative_to(ROOT),'output outside checkout required');out.mkdir()
    dataset=write(args.input,out/'input.stream');require(dataset['complete_dataset'],'partial stream is ineligible')
    problem=load(args.input);frames=len(problem[1])
    native_poses,native_quality,reference_hashes=reference(args.reference,dataset,frames)
    files=('crates/ropt/examples/benchmark_ad_selective_se2_isam.rs','crates/ropt/examples/support/real_selective_se2.rs',
           'crates/ropt/examples/support/real_se2_benchmark.rs','crates/ropt/examples/support/allocation_counter.rs',
           'benchmarks/paired_dense_probe.py','benchmarks/paired_stationarity_se2.py','benchmarks/paired_isam2.py','benchmarks/real_se2_stream.py',
           'benchmarks/se2_stream.py','benchmarks/information_precision.py')
    metadata={'dataset':dataset,'source_sha256':source_fingerprint(ROOT),'executable_sha256':digest(args.binary),
              'files_sha256':{name:digest(ROOT/name) for name in files},'environment':environment(),
              'variants':['before','after'],'rounds':args.rounds,'cpu_affinity':[args.cpu],
              'selected_cap':args.selected_cap,'native_timing_measured':False,
              'ad_chunk_size':args.ad_chunk,'ad_edge_stride':args.ad_stride,
              'factors':sum(len(added) for _,added in problem[1]),
              'ad_factors':(sum(len(added) for _,added in problem[1])+args.ad_stride-1)//args.ad_stride,
              'before_source_sha256':source_fingerprint(args.before_source),
              'before_executable_sha256':digest(args.before_binary),
              'before_driver_sha256':{name:digest(args.before_source/name) for name in files if name.startswith('crates/')},
              'native_reference_directory':str(args.reference.resolve().relative_to(ROOT)),
              'native_reference_sha256':reference_hashes,
              'gates':{'original_column_normalized_gradient_inf':1e-7,'world_pose_norm_difference':2e-6,
                       'relative_cost_difference':1e-9,'relative_reported_cost_difference':1e-10,'fixed_root':'exact'},
              'warmup_streams_per_process':1,'timed_streams_per_process':1,
              'policy':'before/after: tree-only versus contiguous-key dense probe indexing in the same editable backend; new-factor update then at most selected_cap requested passes, explicit full recovery if target unmet; all use current nonlinear normalized gradient target 8e-8, keys above target/2; periodic refresh disabled',
              'timed':'updates, explicit nonlinear probes, requested relinearization, failed attempts, full recovery and owned reports; parsing/factor recipes, external checking and serialization excluded',
              'allocator_counting':False,'memory_measured':False,
              'scope':'complete mixed-AD real streams, tree-only versus contiguous-key dense probe indexing; archived precision-corrected GTSAM poses/costs check agreement only; no fresh native speed claim'}
    require(metadata['before_source_sha256']=='41c27190a3d4eed0dcb9a00c48886cdd60222df5f20e33c0759dc119911d64f4','unexpected before kernel')
    require(all(metadata['before_driver_sha256'][name]==metadata['files_sha256'][name] for name in metadata['before_driver_sha256']),'drivers differ')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    require(metadata['before_executable_sha256']!=metadata['executable_sha256'],'identical executables')
    orders=list(itertools.permutations(metadata['variants']))
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1',
         'ROPT_BENCH_AD_CHUNK':str(args.ad_chunk),'ROPT_BENCH_AD_STRIDE':str(args.ad_stride)}
    samples=[];updates=[];quality=[];comparisons=[]
    for index in range(args.rounds):
        measured={}; recorded={}
        for position,variant in enumerate(orders[index%2]):
            directory=out/f'{index}-{variant}'
            with (out/f'{index}-{variant}.log').open('w') as log:
                subprocess.run(['taskset','-c',str(args.cpu),str((args.before_binary if variant=='before' else args.binary).resolve()),str(out/'input.stream'),
                                str(directory),'1','1','editable',str(args.selected_cap)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            poses,checks,raw,sample=validate(directory,frames,problem,column_normalized_gradient,1e-7)
            require(all(row['initial_damping']==row['final_damping']=='' for row in raw),'unexpected damping')
            require(all(float(row['gradient_norm'])<=8e-8 for row in raw),'internal target missed')
            require(all(int(row['requested_calls'])<=args.selected_cap for row in raw),'selected cap changed')
            expected=f"runtime_ad=true; chunk_size={args.ad_chunk}; edge_stride={args.ad_stride}; factors={metadata['factors']}; ad_factors={metadata['ad_factors']}; analytic_factors={metadata['factors']-metadata['ad_factors']}; original_edges_and_information_unchanged=true\n"
            require((directory/'differentiation.txt').read_text()==expected,'AD workload changed')
            compared=agreement(poses,native_poses,checks,native_quality)
            comparisons.append({'round':index,'variant':variant,'reference':'archived native',**compared})
            measured[variant]=poses,checks
            recorded[variant]=raw,(directory/'poses.csv').read_bytes()
            samples.append({'variant':variant,'round':index,'position':position,**sample})
            updates.extend({'variant':variant,'round':index,**row} for row in raw)
            quality.extend({'variant':variant,'round':index,**row} for row in checks)
            if index==0:
                with (out/f'{variant}-poses.csv.gz').open('wb') as dest:
                    with gzip.GzipFile(fileobj=dest,mode='wb',filename='',mtime=0) as f:f.write((directory/'poses.csv').read_bytes())
                (out/f'{variant}-options.txt').write_bytes((directory/'options.txt').read_bytes())
                (out/f'{variant}-differentiation.txt').write_bytes((directory/'differentiation.txt').read_bytes())
            print(f'round {index} {variant}: {int(sample["elapsed_ns"])/1e9:.6f}s; all {frames} original-information prefixes and archived native agreement pass',flush=True)
        require(recorded['before'][1]==recorded['after'][1],'means changed bitwise')
        require([{k:v for k,v in row.items() if k!='elapsed_ns'} for row in recorded['before'][0]]==
                [{k:v for k,v in row.items() if k!='elapsed_ns'} for row in recorded['after'][0]],'non-timing work or energy changed')
        for variant in metadata['variants']:shutil.rmtree(out/f'{index}-{variant}')
    require(source_fingerprint(ROOT)==metadata['source_sha256'],'solver changed during campaign')
    require(digest(args.binary)==metadata['executable_sha256'],'binary changed during campaign')
    require(all(digest(ROOT/name)==value for name,value in metadata['files_sha256'].items()),'runner changed during campaign')
    require(source_fingerprint(args.before_source)==metadata['before_source_sha256'],'before solver changed during campaign')
    require(digest(args.before_binary)==metadata['before_executable_sha256'],'before binary changed during campaign')
    require(all(digest(args.before_source/name)==value for name,value in metadata['before_driver_sha256'].items()),'before driver changed')
    report={'eligible':True,'before_after_means_byte_identical':True,'before_after_non_timing_fields_identical':True,'timed_updates':args.rounds*2*frames,'comparisons':comparisons,'native_timing_measured':False,'cases':{}}
    for variant in metadata['variants']:
        raw=[row for row in updates if row['variant']==variant]
        report['cases'][variant]={'timing':timing_summary([row for row in samples if row['variant']==variant]),
            'max_normalized_gradient':max(row['gradient_inf'] for row in quality if row['variant']==variant),
            'forced_recoveries':sum(int(row['forced_recovery']) for row in raw),'requested_calls':sum(int(row['requested_calls']) for row in raw),
            'queries':sum(int(row['queries']) for row in raw),'gaussian_reeliminated':sum(int(row['gaussian_reeliminated']) for row in raw)}
    by_round={variant:{int(row['round']):int(row['elapsed_ns']) for row in samples if row['variant']==variant} for variant in metadata['variants']}
    case=report['cases']['after'];case['speedup_over_before']=report['cases']['before']['timing']['median_ms']/case['timing']['median_ms']
    case['faster_pairs_over_before']=sum(by_round['after'][index]<by_round['before'][index] for index in range(args.rounds))
    case['time_reduction_percent']=100*(1-1/case['speedup_over_before'])
    case['beats_before_in_this_case']=case['speedup_over_before']>1 and case['faster_pairs_over_before']*4>=args.rounds*3
    for name,rows in [('samples.csv',samples),('updates.csv',updates),('quality.csv',quality)]:
        with (out/name).open('w') as f:
            fields=list(dict.fromkeys(key for row in rows for key in row));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2),flush=True)


if __name__=='__main__':main()
