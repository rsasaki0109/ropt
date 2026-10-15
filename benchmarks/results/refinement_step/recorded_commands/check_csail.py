from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys
sys.path.insert(0, '/workspace/ropt/benchmarks')
from paired_isam2 import validate, agreement
from real_se2_stream import column_normalized_gradient, load
root = Path('/workspace/ropt')
reference = root / 'benchmarks/results/current_native_isam2'
current = Path('/workspace/.ropt-refinement/csail-current')
manifest = json.loads((reference / 'manifest.json').read_text())
for name in ('primary/ropt-poses.csv.gz', 'primary/native-poses.csv.gz', 'primary/updates.csv', 'primary/quality.csv'):
    assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == manifest['files_sha256'][name]
problem = load(root / 'benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o')
poses, quality, updates, sample = validate(current, 1044, problem, column_normalized_gradient, 1e-7)
assert (current / 'poses.csv').read_bytes() == gzip.decompress((reference / 'primary/ropt-poses.csv.gz').read_bytes())
with (reference / 'primary/updates.csv').open() as stream:
    before = [r for r in csv.DictReader(stream) if r['variant']=='ropt' and r['round']=='0']
for a,b in zip(updates, before):
    assert {k:v for k,v in a.items() if k != 'elapsed_ns'} == {k:v for k,v in b.items() if k not in ('elapsed_ns','variant','round')}
native_poses = [[] for _ in range(1044)]
for row in csv.DictReader(gzip.decompress((reference / 'primary/native-poses.csv.gz').read_bytes()).decode().splitlines()):
    native_poses[int(row['frame'])-1].append(tuple(float(row[k]) for k in ('x','y','angle')))
with (reference / 'primary/quality.csv').open() as stream:
    native_quality = [{k: float(v) for k,v in r.items() if k not in ('variant','round')} for r in csv.DictReader(stream) if r['variant']=='native' and r['round']=='0']
report = {'scope':'functional replay only; no new comparative timing', 'arrivals':1044,
          'poses_and_non_timing_work_exactly_equal':True,
          'max_original_normalized_gradient':max(r['gradient_inf'] for r in quality),
          'native_agreement': agreement(poses, native_poses, quality, native_quality),
          'poses_sha256': hashlib.sha256((current / 'poses.csv').read_bytes()).hexdigest()}
(current.parent / 'csail-quality.csv').write_text('frame,cost,gradient_inf,raw_gradient_inf\n' + ''.join(','.join(str(r[k]) for k in ('frame','cost','gradient_inf','raw_gradient_inf'))+'\n' for r in quality))
(current.parent / 'csail-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
