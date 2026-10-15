from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
root = Path('/workspace/ropt')
out = Path('/workspace/.ropt-refinement')
binaries = {'before': out / 'before', 'after': root / 'target/release/examples/benchmark_refinement'}
results = []
for backend in ('qr', 'tree', 'editable'):
    for blocks in (1, 16, 128):
        for cap in (1, 4):
            pair = {}
            for variant, binary in binaries.items():
                command = [str(binary), backend, str(blocks), str(cap), '5']
                process = subprocess.run(command, check=True, capture_output=True, text=True)
                row = list(csv.DictReader(io.StringIO(process.stdout)))[0]
                pair[variant] = row
                results.append({'variant': variant, **row})
            a, b = pair['before'], pair['after']
            for key in ('backend', 'blocks', 'cap', 'repeats', 'callbacks_per_refresh', 'refinement_steps', 'fingerprint'):
                assert a[key] == b[key], (key, a, b)
            assert int(a['allocations']) > int(b['allocations']), pair
            assert int(a['requested_bytes']) > int(b['requested_bytes']), pair
with (out / 'allocations.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
meta = {'scope': 'requested allocation traffic for complete forced refresh only; no comparative timing or peak memory',
        'executable_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
        'driver_sha256': hashlib.sha256((root / 'crates/ropt/examples/benchmark_refinement.rs').read_bytes()).hexdigest(),
        'before_source_sha256': '12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288',
        'after_source_sha256': 'ffdd6fd2033b67900137ccbca263df7749c4f6ecc681d98601821e8a8fc2fd74',
        'cases': 18, 'total_refreshes': 180,
        'identical_callbacks_steps_and_result_fingerprints': True}
(out / 'diagnostic.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2))
print('128-block allocations and requested bytes per refresh, before -> after')
for backend in ('qr','tree','editable'):
    pair = [r for r in results if r['backend']==backend and r['blocks']=='128' and r['cap']=='4']
    print(backend, [(int(r['allocations'])//5, int(r['requested_bytes'])//5) for r in pair])
