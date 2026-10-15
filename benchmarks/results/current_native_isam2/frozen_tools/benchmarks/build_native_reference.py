"""Build a checked GTSAM 4.2.1 precise iSAM2 reference outside the checkout.

Requires CMake, a C++ compiler and Boost development libraries. A relocated
Debian sysroot can be supplied with --boost-root and --runtime-library-path.
Never downloads code, changes the GTSAM sources or installs into system paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from gtsam_exact.write_manifest import ARCHIVE_SHA256

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'benchmarks/gtsam_precise_isam2'
REFERENCE_FILES = ('benchmark.cpp', 'CMakeLists.txt', 'precision.hpp', 'factors.hpp', 'test_precision.cpp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(archive, directory):
    expected = set()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() or member.islnk():
                expected.add(member.name)
                path = directory / member.name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'source file missing or replaced: {member.name}')
                if hashlib.sha256(stream.extractfile(member).read()).hexdigest() != digest(path):
                    raise ValueError(f'source file modified: {member.name}')
            elif member.issym():
                expected.add(member.name)
                path = directory / member.name
                if not path.is_symlink() or os.readlink(path) != member.linkname:
                    raise ValueError(f'source link modified: {member.name}')
            elif not member.isdir():
                raise ValueError(f'unsupported archive entry: {member.name}')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError('source tree contains unexpected files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--boost-root', type=Path)
    parser.add_argument('--runtime-library-path', type=Path)
    parser.add_argument('--boost-library-dir', type=Path)
    parser.add_argument('--dependency-manifest', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError('positive build parallelism required')
    archive, out = args.archive.resolve(), args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('output must be outside the checkout')
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('GTSAM source archive checksum differs')
    cmake = shutil.which(args.cmake)
    cxx = shutil.which(args.cxx)
    if not cmake or not cxx:
        raise ValueError('CMake and a C++ compiler are required')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(out / 'prefix/lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if args.runtime_library_path:
        env['LD_LIBRARY_PATH'] = str(args.runtime_library_path.resolve()) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    version = lambda executable: subprocess.check_output([executable, '--version'], env=env, text=True).splitlines()[0]
    boost_flags = []
    if args.boost_root:
        boost_flags = [f'-DBOOST_ROOT={args.boost_root.resolve()}', '-DBoost_NO_SYSTEM_PATHS=ON', '-DBoost_NO_BOOST_CMAKE=ON', '-DCMAKE_POLICY_DEFAULT_CMP0167=OLD']
    if args.boost_library_dir:
        boost_flags.append(f'-DBOOST_LIBRARYDIR={args.boost_library_dir.resolve()}')
    flags = ['-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CXX_COMPILER={cxx}',
             '-DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON', '-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF',
             '-DGTSAM_WITH_TBB=OFF', '-DGTSAM_WITH_EIGEN_MKL=OFF', '-DGTSAM_USE_SYSTEM_EIGEN=OFF',
             '-DGTSAM_DEFAULT_ALLOCATOR=STL', '-DGTSAM_BUILD_TESTS=OFF', '-DGTSAM_BUILD_UNSTABLE=OFF',
             '-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF', '-DGTSAM_BUILD_PYTHON=OFF']
    inputs = {'archive_sha256': ARCHIVE_SHA256, 'cmake': version(cmake), 'compiler': version(cxx),
              'flags': flags + boost_flags,
              'reference_sources_sha256': {name: digest(REFERENCE / name) for name in REFERENCE_FILES},
              'dependency_manifest_sha256': digest(args.dependency_manifest) if args.dependency_manifest else None}
    source = out / 'source'
    marker = out / 'build-inputs.json'
    if out.exists():
        if not marker.exists() or json.loads(marker.read_text()) != inputs:
            raise ValueError('existing output is unrelated or build inputs changed; choose a new directory')
    else:
        out.mkdir(parents=True)
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter='data')
        marker.write_text(json.dumps(inputs, indent=2) + '\n')
    verify_source(archive, source)
    if args.dependency_manifest:
        shutil.copyfile(args.dependency_manifest, out / 'dependency-manifest.json')
    commands = []

    def run(command, name):
        command = [str(v) for v in command]
        commands.append({'log': name, 'command': command})
        print(name, flush=True)
        with (out / name).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    build, prefix, adapter = out / 'gtsam-build', out / 'prefix', out / 'adapter-build'
    run([cmake, '-S', source / 'gtsam-4.2.1', '-B', build, f'-DCMAKE_INSTALL_PREFIX={prefix}', *flags, *boost_flags], 'configure.log')
    run([cmake, '--build', build, '--target', 'gtsam', 'CppUnitLite', '--parallel', args.jobs], 'library-build.log')
    run([cmake, '--install', build], 'install.log')
    run([cmake, '-S', REFERENCE, '-B', adapter, '-DCMAKE_BUILD_TYPE=Release',
         f'-DCMAKE_CXX_COMPILER={cxx}', f'-DGTSAM_DIR={prefix / "lib/cmake/GTSAM"}', *boost_flags], 'adapter-configure.log')
    run([cmake, '--build', adapter, '--parallel', args.jobs], 'adapter-build.log')
    ctest = Path(cmake).with_name('ctest')
    run([ctest, '--test-dir', adapter, '--output-on-failure'], 'precision-tests.log')
    library = prefix / 'lib/libgtsam.so.4.2.1'
    # The existing writer intentionally creates manifests exclusively. Preserve
    # previous successful manifests on a resumed build rather than overwriting.
    library_manifest = out / 'library-manifest.json'
    generated = out / 'library-manifest.generated.json'
    if generated.exists():
        raise ValueError('unfinished manifest generation exists; inspect it before resuming')
    run([sys.executable, ROOT / 'benchmarks/gtsam_exact/write_manifest.py', '--archive', archive,
         '--cache', build / 'CMakeCache.txt', '--library', library,
         '--executable', adapter / 'gtsam_isam2', '--reference-directory', REFERENCE,
         '--reference-cache', adapter / 'CMakeCache.txt', '--verify-linked-library', '--output', generated], 'library-manifest.log')
    if library_manifest.exists() and library_manifest.read_bytes() != generated.read_bytes():
        raise ValueError('resumed build changed the recorded native artifacts; choose a new directory')
    generated.replace(library_manifest)
    run([sys.executable, REFERENCE / 'write_manifest.py', '--library-manifest', library_manifest,
         '--build', adapter, '--output', out / 'native-manifest.json'], 'native-manifest.log')
    verify_source(archive, source)
    linked = subprocess.check_output(['ldd', str(adapter / 'gtsam_isam2')], env=env, text=True)
    linked_hashes = {}
    for line in linked.splitlines():
        if '=>' in line:
            target = line.split('=>', 1)[1].strip().split()[0]
            if target.startswith('/'):
                linked_hashes[target] = digest(target)
    provenance = {'linked_libraries_sha256': linked_hashes, 'inputs': inputs, 'commands': commands, 'builder_sha256': digest(__file__),
                  'native_manifest_sha256': digest(out / 'native-manifest.json'),
                  'runtime_library_path': str(args.runtime_library_path.resolve()) if args.runtime_library_path else None,
                  'scope': 'reproducible build and native precision test; no comparative timing'}
    (out / 'setup.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'binary': str(adapter / 'gtsam_isam2'), 'manifest': str(out / 'native-manifest.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
