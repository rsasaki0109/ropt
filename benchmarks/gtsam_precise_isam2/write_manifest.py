"""Record the compiled compensated native adapter against a verified GTSAM library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-manifest', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.library_manifest.read_text())
    if not metadata['exact_between']:
        raise ValueError('exact-Jacobian library required')
    executable = args.build.resolve()/'gtsam_isam2'
    linked = subprocess.check_output(['ldd', str(executable)], text=True)
    libraries = [Path(line.split('=>', 1)[1].strip().split()[0])
                 for line in linked.splitlines() if 'libgtsam.so' in line and '=>' in line]
    if len(libraries) != 1 or digest(libraries[0]) != metadata['library_sha256']:
        raise ValueError('linked GTSAM differs from verified library manifest')
    cache = args.build/'CMakeCache.txt'
    values = {}
    for line in cache.read_text().splitlines():
        if ':' in line and '=' in line and not line.startswith('//'):
            key, value = line.split('=',1)
            values[key.split(':',1)[0]] = value
    if values.get('CMAKE_BUILD_TYPE') != 'Release':
        raise ValueError('Release reference required')
    source = Path(__file__).resolve().parent
    names = ('benchmark.cpp','CMakeLists.txt','precision.hpp','factors.hpp','test_precision.cpp')
    metadata.update({
        'executable_sha256': digest(executable),
        'reference_source_directory': 'benchmarks/gtsam_precise_isam2',
        'reference_sources_sha256': {name:digest(source/name) for name in names},
        'reference_generator_sha256': digest(__file__),
        'reference_cache_sha256': digest(cache),
        'reference_compiler': values.get('CMAKE_CXX_COMPILER'),
        'reference_flags': {key:values.get(key) for key in ('CMAKE_CXX_FLAGS','CMAKE_CXX_FLAGS_RELEASE')},
        'information_whitening': 'same two-component arithmetic as ropt for squared relative pivots below 2^-26; original information unchanged',
        'root_conditioning': 'exact conditioned BetweenFactor residual and chart Jacobian',
        'stationarity_arithmetic': 'direct original-information compensated products and column energies for small pivots',
    })
    args.output.write_text(json.dumps(metadata,indent=2)+'\n')
    print('Verified native executable/library/source manifest saved.')


if __name__ == '__main__':
    main()
