from pathlib import Path
import hashlib
import json
import subprocess
import sys
root=Path('/workspace/ropt')
out=Path('/workspace/.ropt-refinement/timing')
base=[sys.executable,str(root/'benchmarks/paired_refinement_step.py'),
      '--binary',str(out/'after'),'--before-binary',str(out/'before'),
      '--before-source','/workspace/.ropt-refinement/before-source',
      '--expected-before-source','12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288',
      '--expected-source','ffdd6fd2033b67900137ccbca263df7749c4f6ecc681d98601821e8a8fc2fd74',
      '--input',str(root/'benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o'),
      '--reference',str(root/'benchmarks/results/csail_real_isam2/primary')]
commands=[]
for campaign in ('primary','confirmation'):
    if campaign=='confirmation':
        report=json.loads((out/'primary/report.json').read_text())
        qualified=report['cases']['after']['beats_before_in_this_case']
        decision={'confirmation_started':qualified,'basis':'predeclared lower primary median and at least 5/6 faster pairs','primary_faster_pairs':report['cases']['after']['faster_pairs_over_before'],'primary_speedup':report['cases']['after']['speedup_over_before']}
        (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        if not qualified:
            break
    command=base+['--output',str(out/campaign)]
    commands.append(command)
    with (out/f'{campaign}.log').open('w') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print((out/'decision.json').read_text())
