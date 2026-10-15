from pathlib import Path
import json,re,subprocess,sys
root=Path('/workspace/ropt');out=Path('/workspace/.ropt-prior-columns/rollback');out.mkdir()
sys.path.insert(0,str(root/'benchmarks'))
from paired_pcg import source_fingerprint
expected={r['flag']:r['source_sha256'] for r in json.loads(Path('/workspace/.ropt-root-weights/rollback/report.json').read_text())}
expected['--before-prior-column-scaling']='fd585f3cee3d97b89fe7a5eda49e7d9a9a0c203ad379762a64b049543d285459'
flags=re.findall(r'stage.add_argument\("(--before-[^"]+)"',(root/'benchmarks/make_ldl_baseline.py').read_text())+['default']
report=[]
with (out/'checks.log').open('w') as log:
 for index,flag in enumerate(flags):
  directory=out/str(index)
  command=[sys.executable,str(root/'benchmarks/make_ldl_baseline.py'),'--source-only','--output',str(directory)]
  if flag!='default':command.append(flag)
  subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
  digest=source_fingerprint(directory);assert digest==expected[flag],(flag,digest,expected[flag])
  report.append({'flag':flag,'source_sha256':digest})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(len(report),'source-only rollback stages preserve expected fingerprints')
print('current',source_fingerprint(root))
