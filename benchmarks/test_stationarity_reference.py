"""Archived reference integrity and complete-stream checks on small fixtures."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paired_stationarity_se2 import reference


class ReferenceTests(unittest.TestCase):
    def fixture(self,path,gradient=1e-8):
        dataset={'source_sha256':'source','stream_sha256':'stream'}
        (path/'metadata.json').write_text(json.dumps({'dataset':dataset}))
        (path/'report.json').write_text(json.dumps({'eligible':True}))
        with gzip.open(path/'native-poses.csv.gz','wt') as f:
            f.write('frame,key,x,y,angle\n1,0,0,0,0\n1,1,1,0,0\n2,0,0,0,0\n2,1,1,0,0\n2,2,2,0,0\n')
        with (path/'quality.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['variant','round','cost','gradient_inf'])
            writer.writerows([['native',0,0,gradient],['native',0,1,gradient],['ropt',0,99,9]])
        self.manifest(path)
        return dataset

    def manifest(self,path):
        hashes={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('metadata.json','report.json','quality.csv','native-poses.csv.gz')}
        (path/'artifacts_sha256.json').write_text(json.dumps(hashes))

    def test_complete_hashed_native_rows_are_selected_without_ropt_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);dataset=self.fixture(path)
            poses,quality,hashes=reference(path,dataset,2)
            self.assertEqual([len(p) for p in poses],[2,3]);self.assertEqual([q['cost'] for q in quality],[0.,1.]);self.assertEqual(len(hashes),4)

    def test_mutated_artifact_different_stream_and_partial_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);dataset=self.fixture(path)
            with self.assertRaisesRegex(ValueError,'stream'):reference(path,{**dataset,'stream_sha256':'other'},2)
            with self.assertRaisesRegex(ValueError,'incomplete'):reference(path,dataset,3)
            (path/'quality.csv').write_text('mutated')
            with self.assertRaisesRegex(ValueError,'reference changed'):reference(path,dataset,2)

    def test_ineligible_and_nonstationary_native_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);dataset=self.fixture(path,1e-4)
            with self.assertRaisesRegex(ValueError,'gradient'):reference(path,dataset,2)
            (path/'report.json').write_text(json.dumps({'eligible':False}));self.manifest(path)
            with self.assertRaisesRegex(ValueError,'ineligible'):reference(path,dataset,2)


if __name__=='__main__':unittest.main()
