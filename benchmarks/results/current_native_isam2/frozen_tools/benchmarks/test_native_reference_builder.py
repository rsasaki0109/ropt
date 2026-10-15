"""Integrity and failure-isolation checks for the optional native build helper."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from build_native_reference import verify_source


class NativeReferenceBuilderTests(unittest.TestCase):
    def fixture(self, root):
        archive = root / 'source.tar'
        content = b'project(checked_source)\n'
        with tarfile.open(archive, 'w') as stream:
            entry = tarfile.TarInfo('gtsam-4.2.1/CMakeLists.txt')
            entry.size = len(content)
            stream.addfile(entry, io.BytesIO(content))
            link = tarfile.TarInfo('gtsam-4.2.1/link.txt')
            link.type = tarfile.SYMTYPE
            link.linkname = 'CMakeLists.txt'
            stream.addfile(link)
        source = root / 'source'
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter='data')
        return archive, source

    def test_changed_source_is_rejected_before_build_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, source = self.fixture(Path(temporary))
            verify_source(archive, source)
            (source / 'gtsam-4.2.1/CMakeLists.txt').write_text('project(modified)\n')
            with self.assertRaisesRegex(ValueError, 'source file modified'):
                verify_source(archive, source)

    def test_extra_source_files_cannot_enter_the_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, source = self.fixture(Path(temporary))
            (source / 'gtsam-4.2.1/extra.cpp').write_text('int extra;\n')
            with self.assertRaisesRegex(ValueError, 'unexpected files'):
                verify_source(archive, source)

    def test_retargeted_source_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, source = self.fixture(Path(temporary))
            link = source / 'gtsam-4.2.1/link.txt'
            link.unlink()
            link.symlink_to('missing.txt')
            with self.assertRaisesRegex(ValueError, 'source link modified'):
                verify_source(archive, source)

    def test_wrong_archive_never_creates_output_or_invokes_cmake(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'untrusted.tar'
            archive.write_bytes(b'not the pinned GTSAM archive')
            output = root / 'output'
            process = subprocess.run([sys.executable, str(Path(__file__).with_name('build_native_reference.py')),
                                      '--archive', str(archive), '--output', str(output), '--cmake', '/nonexistent-cmake'],
                                     capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('source archive checksum differs', process.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
