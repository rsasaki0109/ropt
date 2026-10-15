"""Summarize archived native-comparison latency and ropt recovery work, without timing."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from pose_graph import ROOT, require


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    result = {'scope': 'derived archived wall times; different native/ropt policies; no causal attribution or new timing'}
    for name in ('primary', 'confirmation'):
        for file in ('updates.csv', 'samples.csv', 'metadata.json'):
            path = root / name / file
            require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files_sha256'][f'{name}/{file}'], 'input hash changed')
        with (root / name / 'updates.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 12528, 'incomplete campaign')
        ropt = [r for r in rows if r['variant'] == 'ropt']
        require(len(ropt) == 6264, 'incomplete ropt updates')
        groups = {}
        total = sum(int(r['elapsed_ns']) for r in ropt)
        for recovery in (False, True):
            values = [int(r['elapsed_ns']) for r in ropt if bool(int(r['forced_recovery'])) == recovery]
            groups['recovery' if recovery else 'no_recovery'] = {
                'updates': len(values), 'update_share': len(values) / len(ropt),
                'wall_time_share': sum(values) / total,
                'median_ms': statistics.median(values) / 1e6,
            }
        frames = defaultdict(list)
        for row in rows:
            frames[(row['variant'], int(row['frame']))].append(int(row['elapsed_ns']) / 1e6)
        require(all(len(v) == 6 for v in frames.values()) and len(frames) == 2088, 'missing frame repeats')
        top = sorted(((statistics.median(v), key[1]) for key, v in frames.items() if key[0] == 'ropt'), reverse=True)[:10]
        result[name] = {'groups': groups, 'largest_ropt_frame_medians': [
            {'frame': frame, 'ropt_median_ms': median,
             'native_median_ms': statistics.median(frames[('native', frame)])}
            for median, frame in top]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'benchmarks/results/current_native_isam2')
    args = parser.parse_args()
    print(json.dumps(analyze(args.results), indent=2))


if __name__ == '__main__':
    main()
