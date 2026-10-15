"""Render archived CSAIL geometry and all paired timings; no synthetic samples."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from real_se2_stream import load


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root/'benchmarks/results/csail_real_isam2')
    parser.add_argument('--output', type=Path, default=root/'docs/assets/csail-comparison.png')
    args = parser.parse_args()
    source = args.results/'csail-se-sync1045-original.g2o'
    initial, arrivals = load(source)
    seeds = [initial] + [seed for seed, _ in arrivals]
    campaigns = {}
    inputs = [source]
    for name in ('primary', 'confirmation'):
        directory = args.results/name
        metadata = json.loads((directory/'metadata.json').read_text())
        report = json.loads((directory/'report.json').read_text())
        if not report['eligible'] or metadata['dataset']['source_sha256'] != sha256(source):
            raise ValueError('complete verified CSAIL campaign required')
        if not metadata['dataset']['complete_dataset'] or metadata['rounds'] != 6:
            raise ValueError('six complete pairs required')
        with (directory/'samples.csv').open() as f:
            rows = list(csv.DictReader(f))
        samples = {variant: {} for variant in ('ropt', 'native')}
        for row in rows:
            variant, index = row['variant'], int(row['round'])
            if index in samples[variant] or int(row['updates']) != len(arrivals):
                raise ValueError('duplicate or incomplete sample')
            samples[variant][index] = int(row['elapsed_ns'])/1e9
        if any(set(values) != set(range(6)) for values in samples.values()):
            raise ValueError('missing paired sample')
        campaigns[name] = report, samples
        inputs += [directory/file for file in ('metadata.json', 'report.json', 'samples.csv')]
    poses_path = args.results/'primary/ropt-poses.csv.gz'
    poses = []
    with gzip.open(poses_path, 'rt') as f:
        for row in csv.DictReader(f):
            if int(row['frame']) == len(arrivals):
                if int(row['key']) != len(poses):
                    raise ValueError('final pose ordering differs')
                poses.append(tuple(float(row[k]) for k in ('x', 'y', 'angle')))
    if len(poses) != len(seeds) or poses[0] != initial:
        raise ValueError('incomplete final estimates or moved root')
    inputs.append(poses_path)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, (geometry, timing) = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={'width_ratios': [1, 1.15]})
    fig.suptitle('CSAIL · recorded SE2 pose graph', fontsize=19, fontweight='bold', y=.98)
    geometry.plot([p[0] for p in seeds], [p[1] for p in seeds], color='#929aa5', linewidth=1.4,
                  linestyle='--', label='Original odometry seeds')
    geometry.plot([p[0] for p in poses], [p[1] for p in poses], color='#00796b', linewidth=1.8,
                  label='ropt final estimates')
    geometry.scatter([initial[0]], [initial[1]], color='#263238', s=40, marker='s', zorder=4, label='Fixed root')
    geometry.set(title='1,045 poses · 1,172 original constraints', xlabel='World x (m)', ylabel='World y (m)')
    geometry.set_aspect('equal', adjustable='datalim')
    geometry.legend(loc='best', fontsize=8, frameon=False)

    colors = {'primary': '#00796b', 'confirmation': '#6253a3'}
    for campaign, (report, samples) in campaigns.items():
        offset = -.10 if campaign == 'primary' else .10
        for index in range(6):
            shift = offset + (index-2.5)*.014
            values = [samples[v][index] for v in ('ropt', 'native')]
            timing.plot([shift, 1+shift], values, color=colors[campaign], alpha=.40, linewidth=.9)
            timing.scatter([shift, 1+shift], values, color=colors[campaign], s=24, zorder=3,
                           label=campaign.capitalize() if index == 0 else None)
        for x, variant in enumerate(('ropt', 'native')):
            median = report['cases'][variant]['timing']['median_ms']/1000
            timing.plot([x+offset-.075, x+offset+.075], [median, median], color='#17212b', linewidth=2.8, zorder=4)
    primary = campaigns['primary'][0]['median_speedup']
    confirmation = campaigns['confirmation'][0]['median_speedup']
    timing.set(title=f'Complete replay · {primary:.2f}× / {confirmation:.2f}×', ylabel='Timed full replay (s)',
               xticks=[0, 1], xticklabels=['ropt', 'GTSAM iSAM2\ncorrected reference'], xlim=(-.35, 1.35), ylim=(0, None))
    timing.legend(loc='upper left', frameon=False, fontsize=9)
    timing.text(.02, .04, 'Each line joins one measured pair.\nBlack bars: campaign medians.', transform=timing.transAxes, fontsize=8, color='#53616f')
    for axes in (geometry, timing):
        axes.grid(alpha=.16)
        axes.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .035, 'All 1,044 arrivals pass unchanged original-information accuracy gates · forced full refresh · CPU 0, one thread',
             ha='center', fontsize=8.5, color='#53616f')
    fig.subplots_adjust(top=.82, bottom=.19, left=.07, right=.97, wspace=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor='white', metadata={'Software': 'ropt archived CSAIL comparison renderer'})
    plt.close(fig)
    manifest = {'renderer_sha256': sha256(Path(__file__)), 'inputs_sha256': {str(p.relative_to(root) if p.is_relative_to(root) else p): sha256(p) for p in inputs},
                'image_sha256': sha256(args.output), 'source_sha256': metadata['source_sha256'],
                'scope': 'archived full CSAIL recorded replay, shared accuracy full-refresh policy; no synthetic timing samples',
                'matplotlib_version': matplotlib.__version__}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
