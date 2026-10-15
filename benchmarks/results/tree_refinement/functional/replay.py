import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/workspace/ropt')
sys.path.insert(0,str(ROOT/'benchmarks'))
from paired_pcg import source_fingerprint
from real_se2_stream import load,write
from check_real_replay import assess
from paired_stationarity_se2 import reference
from paired_isam2 import agreement
out=Path(__file__).resolve().parent
binary=ROOT/'target/release/examples/benchmark_ad_selective_se2_isam'
env={**os.environ,'ROPT_BENCH_AD_CHUNK':'6','ROPT_BENCH_AD_STRIDE':'2','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'}
results={'source_sha256':source_fingerprint(ROOT),'executable_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'scope':'functional diagnostic only; timings not comparative','datasets':{}}
for name,source,reference_dir in [
 ('csail','csail_real_isam2/csail-se-sync1045-original.g2o','csail_real_isam2/primary'),
 ('mit','real_isam2/intel-mit1228-original.g2o','precise_real_isam2/primary')]:
 d=out/name;d.mkdir();source=ROOT/'benchmarks/results'/source
 metadata=write(source,d/'input.stream')
 command=['taskset','-c','0',str(binary),str(d/'input.stream'),str(d/'replay'),'1','0','editable','0']
 with (d/'replay.log').open('w') as log:
  p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 record={'dataset':metadata,'command':command,'returncode':p.returncode}
 if p.returncode==0:
  try:
   problem=load(source);poses,quality,summary=assess(d/'replay',problem)
   native,nq,hashes=reference(ROOT/'benchmarks/results'/reference_dir,metadata,len(problem[1]))
   gap=agreement(poses,native,quality,nq)
   record.update(quality=summary,native_agreement=gap,native_reference_sha256=hashes)
  except Exception as e:record['assessment_error']=repr(e)
 results['datasets'][name]=record
 (out/'report.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,json.dumps(record),flush=True)
