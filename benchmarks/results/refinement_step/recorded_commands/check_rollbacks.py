from pathlib import Path
import json
import re
import subprocess
import sys
sys.path.insert(0, '/workspace/ropt/benchmarks')
from paired_pcg import source_fingerprint
root = Path('/workspace/ropt')
out = Path('/workspace/.ropt-refinement/rollback')
out.mkdir()
flags = re.findall(r'stage.add_argument\("(--before-[^"]+)"', (root / 'benchmarks/make_ldl_baseline.py').read_text())
previous = {r['flag']: r['source_sha256'] for r in json.loads(Path('/tmp/ropt-dynamic-lag-rollbacks/report.json').read_text())}
previous['--before-refinement-step'] = '12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288'
report = []
with (out / 'checks.log').open('w') as log:
    for index, flag in enumerate(flags):
        directory = out / str(index)
        subprocess.run([sys.executable, str(root / 'benchmarks/make_ldl_baseline.py'), flag, '--source-only', '--output', str(directory)], check=True, stdout=log, stderr=subprocess.STDOUT)
        digest = source_fingerprint(directory)
        assert digest == previous[flag], (flag, digest, previous[flag])
        report.append({'flag': flag, 'source_sha256': digest})
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'{len(report)} source-only rollback stages preserve previous fingerprints')
