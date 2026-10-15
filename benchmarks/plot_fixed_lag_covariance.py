"""Plot the adaptive fixed-lag example's window and local covariance summaries."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError('expected finite complete records')
    frames = [r['frame'] for r in rows]
    if frames != sorted(set(frames)):
        raise ValueError('frames must strictly increase')
    if any(r['lag'] <= 0 or r['live_variables'] < 1 or r['retired_count'] < 0 for r in rows):
        raise ValueError('invalid window summary')
    if any(r[f'std_body_{axis}'] <= 0 for r in rows for axis in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')):
        raise ValueError('expected positive standard deviations')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    colors = ['#2563eb', '#059669', '#d97706']
    axes[0, 0].step(frames, [r['lag'] for r in rows], where='post', color=colors[0], linewidth=2)
    axes[0, 0].set(title='Configured window duration', ylabel='Seconds', ylim=(0, max(r['lag'] for r in rows) + 1))
    axes[0, 1].step(frames, [r['live_variables'] for r in rows], where='post', color=colors[1], linewidth=2, label='Live variables (including fixed root)')
    axes[0, 1].bar(frames, [r['retired_count'] for r in rows], color=colors[2], alpha=.6, width=.55, label='Retired in this update')
    axes[0, 1].set(title='Live window and retirement', ylabel='Count', ylim=(0, max(r['live_variables'] for r in rows) + 2))
    axes[0, 1].legend(fontsize=8, loc='upper left')
    for ax, prefix, title, unit in [
        (axes[1, 0], 't', 'Latest pose: body translation uncertainty', 'Standard deviation (m)'),
        (axes[1, 1], 'r', 'Latest pose: body rotation uncertainty', 'Standard deviation (rad)'),
    ]:
        for axis, color in zip('xyz', colors):
            ax.plot(frames, [r[f'std_body_{prefix}{axis}'] for r in rows], color=color, marker='o', markersize=3, label=axis)
        ax.set(title=title, ylabel=unit, xlabel='Update (one second apart)')
        ax.legend(ncol=3, fontsize=8)
    changes = [rows[k]['frame'] for k in range(1, len(rows)) if rows[k]['lag'] != rows[k-1]['lag']]
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for frame in changes:
            ax.axvline(frame, color='#64748b', linestyle=':', linewidth=1, alpha=.7)
    fig.suptitle('Adaptive fixed-lag smoothing · SE(3)', fontsize=16, fontweight='bold', y=.99)
    fig.text(.5, .01, 'Exact synthetic odometry; specified noise; local Gaussian covariance via dense SVD. No timing or coverage claim.', ha='center', fontsize=8, color='#475569')
    fig.tight_layout(rect=(0, .035, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
