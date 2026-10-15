"""Functional smoke for mixed-AD ropt and the precise native iSAM2 adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from paired_isam2 import agreement, validate
from pose_graph import require
from real_se2_stream import column_normalized_gradient
from se2_stream import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ropt', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--native-library-path', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'summary already exists')
    env = {**os.environ, 'ROPT_BENCH_AD_CHUNK': '6', 'ROPT_BENCH_AD_STRIDE': '2',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'}
    if args.native_library_path:
        env['LD_LIBRARY_PATH'] = ':'.join(str(p.resolve()) for p in args.native_library_path) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    summary = {'scope': 'functional only; no comparative timing', 'arrivals_per_variant': 25, 'variants': {}}
    checked = {}
    with tempfile.TemporaryDirectory(prefix='ropt-current-native-smoke-') as temporary:
        root = Path(temporary)
        data = root / 'input.stream'
        write(data, 25)
        for variant, binary in [('ropt', args.ropt), ('native', args.native)]:
            directory = root / variant
            extra = ['editable', '0'] if variant == 'ropt' else []
            command = [str(binary.resolve()), str(data), str(directory), '1', '0', *extra]
            subprocess.run(command, env=env, check=True)
            result = validate(directory, 25, gradient_metric=column_normalized_gradient, gradient_tolerance=1e-7)
            checked[variant] = result
            original = {p.name: p.read_bytes() for p in directory.iterdir()}
            process = subprocess.run(command, env=env, capture_output=True)
            require(process.returncode != 0 and original == {p.name: p.read_bytes() for p in directory.iterdir()}, 'existing output changed')
            for repeat in ('0', '-1'):
                invalid = root / f'{variant}-{repeat}'
                process = subprocess.run([str(binary.resolve()), str(data), str(invalid), repeat, '0', *extra], env=env, capture_output=True)
                require(process.returncode != 0 and not invalid.exists(), 'invalid repeats created output')
            summary['variants'][variant] = {'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                                           'max_normalized_gradient': max(r['gradient_inf'] for r in result[1]),
                                           'existing_output_protection': True, 'invalid_repeat_protection': True}
        summary['agreement'] = agreement(checked['ropt'][0], checked['native'][0], checked['ropt'][1], checked['native'][1])
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
