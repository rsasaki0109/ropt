import csv, gzip, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path('/workspace/ropt')
sys.path.insert(0,str(ROOT/'benchmarks'))
from paired_pcg import source_fingerprint
from paired_stationarity_se2 import reference
from paired_isam2 import agreement
from real_se2_stream import load, write
from check_real_replay import assess
from pose_graph import require
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
out=ROOT/'benchmarks/results/inline_ad_inputs'
out.mkdir()
for name in ('primary','confirmation','allocations'):
    shutil.copytree(Path('/tmp')/('ropt-inline-ad-inputs-'+('stream-' if name != 'allocations' else '')+name),out/name)
for source,name in [('tests','rust_tests'),('example_tests','example_tests'),('python_tests','python_tests'),('clippy','clippy'),('fmt','fmt'),('rustdoc','rustdoc'),('build','build'),('historical-rollback','historical_rollback'),('rollback','rollback')]:
    shutil.copyfile('/tmp/ropt-inline-ad-inputs-'+source+'.log',out/(name+'.log'))
shutil.copyfile('/tmp/ropt-inline-ad-inputs-rollback-stages.json',out/'rollback_stages.json')
with (out/'preceding_archive_audit.log').open('w') as f:
    subprocess.run([sys.executable,'benchmarks/verify_block_jacobian.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ('primary','confirmation'):
    shutil.copytree(Path('/tmp')/('ropt-inline-ad-inputs-probe-'+name),out/('probe_'+name))
trials=out/'trials'; trials.mkdir()
for name in ('strided-write','inline-ad-inputs'):
    trial=trials/name.replace('-','_'); trial.mkdir()
    snapshot=trial/'source'; snapshot.mkdir()
    old=Path('/tmp/ropt-trial-'+name)
    for file in ('Cargo.toml','Cargo.lock','README.md'): shutil.copyfile(old/file,snapshot/file)
    shutil.copytree(old/'crates',snapshot/'crates')
    shutil.copytree(Path('/tmp')/('ropt-'+name+'-primary'),trial/'primary')
    for suffix in ('tests','build'): shutil.copyfile('/tmp/ropt-trial-'+name+'-'+suffix+'.log',trial/(suffix+'.log'))
selection={'strided_write':'rejected: only 2/16 norm cases met the micro criterion; no full-stream measurement', 'inline_ad_inputs':'selected: 14/16 initial norm cases met the criterion; final candidate adds regression tests before formal probe/stream comparisons'}
(trials/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
for name in ('tests','example_tests','clippy'):
    shutil.copyfile('/tmp/ropt-inline-ad-inputs-initial-'+name+'.log',out/('initial_'+name+'.log'))
functional=out/'mit_functional'; functional.mkdir()
old=ROOT/'benchmarks/results/block_jacobian/mit_functional/after'
shutil.copytree(old,functional/'before')
source=ROOT/'benchmarks/results/real_isam2/intel-mit1228-original.g2o'
dataset=write(source,functional/'input.stream')
env={**os.environ,'ROPT_BENCH_AD_CHUNK':'6','ROPT_BENCH_AD_STRIDE':'2','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'}
binary=ROOT/'target/release/examples/benchmark_ad_selective_se2_isam'
with (functional/'replay.log').open('w') as f:
    subprocess.run(['taskset','-c','0',str(binary),str(functional/'input.stream'),str(functional/'after'),'1','0','editable','1'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
with (functional/'check.log').open('w') as f:
    subprocess.run([sys.executable,'benchmarks/check_real_replay.py','--input',str(source),'--candidate',str(functional/'after'),'--reference',str(functional/'before'),'--output',str(functional/'checks')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ('report.json','quality.csv','agreement.csv'): shutil.copyfile(functional/'checks'/name,functional/name)
shutil.rmtree(functional/'checks')
r=json.loads((functional/'report.json').read_text())
require(r['candidate']['eligible_quality'] and r['reference']['eligible_quality'] and r['agreement']['all_prefixes_pass'],'MIT original gates failed')
require(gzip.decompress((old/'poses.csv.gz').read_bytes())==(functional/'after/poses.csv').read_bytes(),'MIT poses changed')
require([{k:v for k,v in r.items() if k!='elapsed_ns'} for r in rows(old/'updates.csv')]==[{k:v for k,v in r.items() if k!='elapsed_ns'} for r in rows(functional/'after/updates.csv')],'MIT work changed')
require((old/'differentiation.txt').read_bytes()==(functional/'after/differentiation.txt').read_bytes(),'MIT AD changed')
problem=load(source)
poses,quality,_=assess(functional/'after',problem)
native,nq,hashes=reference(ROOT/'benchmarks/results/precise_real_isam2/primary',dataset,1227)
native_gap=agreement(poses,native,quality,nq)
with (functional/'after/poses.csv.gz').open('wb') as dest:
    with gzip.GzipFile(fileobj=dest,mode='wb',filename='',mtime=0) as f: f.write((functional/'after/poses.csv').read_bytes())
(functional/'after/poses.csv').unlink()
m={'source_sha256':source_fingerprint(ROOT),'before_source_sha256':'a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080','dataset':dataset,'exact_complete_pose_bytes':True,'exact_non_timing_update_fields':True,'scope':'functional only; cold candidate replay versus frozen preceding-kernel archive; timers not comparative','before_archive':str(old.relative_to(ROOT)),'before_archive_sha256':{p.name:digest(p) for p in old.iterdir() if p.is_file()},'ad_chunk':6,'ad_stride':2,'factors':1483,'ad_factors':742,'ill_conditioned_edge_160_161_uses_ad':True,'archived_native_agreement':native_gap,'native_reference_sha256':hashes,'executable_sha256':digest(binary)}
# Original stable per-arrival ordering, ordinal 160 is selected by stride 2.
edges=[e for _,added in problem[1] for e in added]
require(edges[160][:2]==(160,161),'ill-conditioned original edge ordinal changed')
(functional/'metadata.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'MIT_original_gate':r['candidate'],'native_agreement':native_gap},indent=2),flush=True)
