from pathlib import Path
import json,re,subprocess,sys
root=Path('/workspace/ropt');out=Path('/workspace/.ropt-empty-window/rollback');out.mkdir()
sys.path.insert(0,str(root/'benchmarks'))
from paired_pcg import source_fingerprint
expected={r['flag']:r['source_sha256'] for r in json.loads(Path('/workspace/.ropt-retired-times/rollback/report.json').read_text())}
expected['--before-empty-window']='6531c33f26a77ba2dea2fe180a44973fd832a8fef4123b58414ea90606d055fb'
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
