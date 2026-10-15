import csv
from pathlib import Path
import tempfile
import unittest

from check_real_replay import assess


class ReplayAssessmentTests(unittest.TestCase):
    def write(self, path, bad_cost=False, incomplete=False):
        poses=[(0,0.,0.,0.),(1,1.,0.,0.),(2,2.,0.,0.)]
        with (path/'poses.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['frame','key','x','y','angle'])
            for frame in (1,2):
                for key,x,y,t in poses[:frame+1]:
                    if not (incomplete and frame==2 and key==2):w.writerow([frame,key,x,y,t])
        with (path/'updates.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['frame','elapsed_ns','iterations','cost','allocations','requested_bytes'])
            for frame in (1,2):w.writerow([frame,10,1,1e-4 if bad_cost else 0.,0,0])
        (path/'samples.csv').write_text('updates,elapsed_ns\n2,20\n')
        edge=lambda a,b:(a,b,1.,0.,0.,1.,0.,0.,1.,0.,1.)
        return ((0.,0.,0.),[((1.,0.,0.),[edge(0,1)]),((2.,0.,0.),[edge(1,2)])])

    def test_every_failed_prefix_is_retained_without_marking_replay_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);problem=self.write(path,bad_cost=True)
            _,quality,report=assess(path,problem)
            self.assertFalse(report['eligible_quality'])
            self.assertEqual(report['failed_arrivals']['reported_cost_pass'],[1,2])
            self.assertEqual(len(quality),2)
            self.assertEqual(report['failed_arrivals']['gradient_pass'],[])

    def test_complete_exact_means_pass_and_truncated_means_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);problem=self.write(path)
            self.assertTrue(assess(path,problem)[2]['eligible_quality'])
            problem=self.write(path,incomplete=True)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                assess(path,problem)


if __name__=='__main__':unittest.main()
