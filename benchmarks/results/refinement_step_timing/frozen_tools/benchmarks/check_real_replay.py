"""Keep every independent quality result, including ineligible recorded replays."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from real_se2_stream import load, identity, column_normalized_gradient
from se2_stream import energy_gradient, wrap
from pose_graph import require


def assess(directory, problem):
    root, arrivals = problem
    count = len(arrivals)
    poses = [[] for _ in arrivals]
    path = directory / 'poses.csv'
    stream = path.open() if path.exists() else gzip.open(directory / 'poses.csv.gz', 'rt')
    with stream:
        for row in csv.DictReader(stream):
            k, key = int(row['frame']), int(row['key'])
            require(1 <= k <= count and key == len(poses[k-1]), 'pose layout differs')
            value = tuple(float(row[c]) for c in ('x', 'y', 'angle'))
            require(all(map(math.isfinite, value)), 'nonfinite pose')
            poses[k-1].append(value)
    with (directory / 'updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    require(len(updates) == count and all(len(v) == k+2 for k,v in enumerate(poses)),
            'incomplete recorded replay')
    require([int(r['frame']) for r in updates] == list(range(1,count+1)), 'update layout differs')
    with (directory / 'samples.csv').open() as stream:
        samples = list(csv.DictReader(stream))
    require(len(samples) == 1 and int(samples[0]['updates']) == count, 'missing complete sample')
    require(int(samples[0]['elapsed_ns']) == sum(int(r['elapsed_ns']) for r in updates), 'timers differ')
    quality, edges = [], []
    for k, ((_, added), values, row) in enumerate(zip(arrivals,poses,updates),1):
        edges.extend(added)
        cost, raw, g = energy_gradient(values, edges)
        norm = column_normalized_gradient(values, edges, g)
        reported = float(row['cost'])
        require(all(map(math.isfinite, (cost, raw, norm, reported))), 'nonfinite energy/gradient')
        require(int(row['elapsed_ns']) > 0 and 1 <= int(row['iterations']) <= 200, 'invalid work record')
        require(int(row['allocations']) == 0 and int(row['requested_bytes']) == 0, 'allocation counters enabled')
        error, tolerance = abs(cost-reported), 1e-10*max(1.,cost)
        quality.append({'frame':k,'cost':cost,'reported_cost':reported,'gradient_inf':norm,
                        'raw_gradient_inf':raw,'cost_difference':error,'cost_tolerance':tolerance,
                        'root_exact':values[0]==root,'gradient_pass':norm<=1e-7,
                        'reported_cost_pass':error<=tolerance})
    failures = {key:[r['frame'] for r in quality if not r[key]]
                for key in ('root_exact','gradient_pass','reported_cost_pass')}
    summary = {'complete_replay':True,'arrivals':count,'eligible_quality':not any(failures.values()),
               'failed_arrivals':failures,'max_normalized_gradient':max(r['gradient_inf'] for r in quality),
               'max_raw_gradient':max(r['raw_gradient_inf'] for r in quality),
               'max_reported_cost_difference':max(r['cost_difference'] for r in quality),
               'timing_scope':'single diagnostic replay; no comparative speed claim'}
    return poses, quality, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','candidate','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args(); args.output.mkdir()
    problem=load(args.input)
    poses,quality,summary=assess(args.candidate,problem)
    report={'dataset':identity(args.input),'candidate':summary,
            'gates':{'original_column_normalized_gradient_inf':1e-7,'reported_cost_relative':1e-10,
                     'fixed_root':'exact','world_pose_norm_difference':2e-6,'original_cost_relative':1e-9}}
    records=[{'variant':'candidate',**row} for row in quality]
    if args.reference:
        other,other_quality,other_summary=assess(args.reference,problem)
        differences=[]
        for k,(a,b,qa,qb) in enumerate(zip(poses,other,quality,other_quality),1):
            gap=max(math.hypot(x[0]-y[0],x[1]-y[1],wrap(x[2]-y[2])) for x,y in zip(a,b))
            delta=abs(qa['cost']-qb['cost']);tolerance=1e-9*max(1.,qa['cost'])
            differences.append({'frame':k,'world_pose_norm_difference':gap,'original_cost_difference':delta,
                                'pose_pass':gap<=2e-6,'original_cost_pass':delta<=tolerance})
        report.update({'reference':other_summary,'agreement':{'all_prefixes_pass':all(r['pose_pass'] and r['original_cost_pass'] for r in differences),
                       'max_world_pose_norm':max(r['world_pose_norm_difference'] for r in differences),
                       'max_original_cost_difference':max(r['original_cost_difference'] for r in differences)}})
        records.extend({'variant':'reference',**row} for row in other_quality)
        with (args.output/'agreement.csv').open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(differences[0]));w.writeheader();w.writerows(differences)
    with (args.output/'quality.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
