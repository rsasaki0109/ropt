#!/usr/bin/env python3
"""Generate scientific README graphics from archived measurements and actual LM states."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ropt-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'ropt-font-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BG, PANEL, GRID = '#080f1c', '#101b2e', '#253247'
TEXT, MUTED, TEAL, CORAL = '#eaf1fa', '#8295ad', '#55e2c4', '#ff8b77'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.facecolor': PANEL, 'axes.edgecolor': GRID,
                     'text.color': TEXT, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'grid.color': GRID})


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def pixels(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def gif(frames, path, duration):
    # Shared palette avoids frame-by-frame colour changes; reuse static pixels.
    swatch = Image.new('RGB', (frames[0].width, frames[0].height * 2))
    swatch.paste(frames[0], (0, 0)); swatch.paste(frames[-1], (0, frames[0].height))
    palette = swatch.quantize(colors=256)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [duration] * len(encoded); durations[0] = 800; durations[-1] = 1800
    encoded[0].save(path, save_all=True, append_images=encoded[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)


def header(fig, title, subtitle):
    fig.text(.035, .92, 'ropt', fontsize=36, weight='bold')
    fig.text(.175, .952, title, fontsize=15, weight='bold')
    fig.text(.175, .915, subtitle, fontsize=10, color=MUTED)


def style(ax):
    ax.grid(alpha=.55, linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)


def recorded(out, primary, confirmation, inputs):
    metadata = json.loads((primary / 'metadata.json').read_text())
    report = json.loads((primary / 'report.json').read_text())
    second = json.loads((confirmation / 'report.json').read_text())
    assert report['eligible'] and second['eligible']
    assert metadata['dataset']['complete_dataset'] and metadata['dataset']['total_source_poses'] == 943
    source = primary.parent / 'intel-vertigo943-original.g2o'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['dataset']['source_sha256']
    inputs.extend([primary / 'metadata.json', primary / 'report.json', confirmation / 'report.json',
                   primary / 'ropt-poses.csv.gz', primary / 'quality.csv', primary / 'samples.csv',
                   confirmation / 'samples.csv', source])
    poses = [[] for _ in range(942)]
    with gzip.open(primary / 'ropt-poses.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame']); key = int(row['key'])
            assert key == len(poses[frame-1])
            poses[frame-1].append((float(row['x']), float(row['y'])))
    states = [np.asarray(p) for p in poses]
    assert all(len(p) == i+2 for i,p in enumerate(states))
    original, edges = [], []
    for line in source.read_text().splitlines():
        f = line.split()
        if f and f[0] == 'VERTEX_SE2': original.append((float(f[2]), float(f[3])))
        elif f and f[0] == 'EDGE_SE2': edges.append((int(f[1]), int(f[2])))
    original = np.asarray(original); final = states[-1]
    quality = [r for r in rows(primary / 'quality.csv') if r['variant']=='ropt' and r['round']=='0']
    costs = np.maximum([float(r['cost']) for r in quality], 1e-28)
    lo = np.minimum(original.min(axis=0), final.min(axis=0)) - 2
    hi = np.maximum(original.max(axis=0), final.max(axis=0)) + 2
    fig = plt.figure(figsize=(11, 6.4), dpi=100)
    header(fig, 'Rust-native optimization for robotics', 'Recorded Intel SE2 graph  /  943 poses  /  1,837 measured constraints')
    ax = fig.add_axes((.06,.14,.54,.67)); style(ax)
    ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), xlabel='World X / m', ylabel='World Y / m')
    ax.set_aspect('equal', adjustable='box')
    fig.text(.065,.875,'FULL-HISTORY POSE GRAPH',fontsize=10,color=TEAL)
    path, = ax.plot([], [], color=TEAL, linewidth=1.6)
    tip, = ax.plot([], [], 'o', color=CORAL, markersize=5)
    ax.plot(final[0,0], final[0,1], '*', color=TEXT, markersize=9)
    lines = LineCollection([], colors=MUTED, linewidths=.45, alpha=.18); ax.add_collection(lines)
    readout = fig.text(.065,.835,'', fontsize=10, color=MUTED, family='DejaVu Sans Mono')
    bar = fig.add_axes((.69,.56,.265,.20)); style(bar)
    medians = [report['cases'][v]['timing']['median_ms']/1000 for v in ['ropt','native']]
    bar.barh([1,0], medians, color=[TEAL,CORAL], height=.48)
    bar.set(yticks=[1,0], yticklabels=['ropt','iSAM2'], xlim=(0,23), xlabel='Complete replay / seconds')
    bar.set_title(f"{report['median_speedup']:.2f}x  /  primary median", loc='left', fontsize=11)
    for y,value in zip([1,0],medians): bar.text(value+.3,y,f'{value:.2f}', va='center', fontsize=9)
    fig.text(.69,.44,f"Independent confirmation: {second['median_speedup']:.2f}x\n12 / 12 paired runs faster",color=TEAL,fontsize=10,linespacing=1.7)
    cost_ax = fig.add_axes((.69,.17,.265,.23)); style(cost_ax)
    cost_ax.set(yscale='log',xlim=(1,942),ylim=(1e-28,max(costs)*10),xlabel='Arrival / inferred pose-ID order',ylabel='Original cost')
    curve, = cost_ax.plot([], [],color=TEAL,linewidth=1.5)
    cost_ax.set_title('Energy after each update', loc='left', fontsize=10)
    fig.text(.04,.045,'Measured constraints. Inferred arrival order. Selected actual prefixes; playback is not real time.',color=MUTED,fontsize=9)
    fig.text(.69,.09,'Equal-accuracy full refresh policies; not defaults.',color=MUTED,fontsize=8)
    frames = []
    for index in sorted(set(np.linspace(0,941,80,dtype=int))):
        current = states[index]
        path.set_data(current[:,0],current[:,1]); tip.set_data([current[-1,0]],[current[-1,1]])
        lines.set_segments([(current[a],current[b]) for a,b in edges if b<len(current) and b-a>1])
        readout.set_text(f'Arrival {index+1:03d} / 942    |    {len(current):03d} stored poses')
        curve.set_data(np.arange(1,index+2),costs[:index+1]); frames.append(pixels(fig))
    gif(frames,out/'ropt-hero.gif',100)
    frames[-1].save(out/'ropt-hero.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,7),dpi=130)
    fig.subplots_adjust(left=.09,right=.96,top=.81,bottom=.14)
    header(fig,'Recorded constraints. Optimized geometry.','Intel Research Lab / complete 943-pose variant / fixed original root')
    style(ax); ax.set_aspect('equal',adjustable='datalim')
    ax.add_collection(LineCollection([(final[a],final[b]) for a,b in edges if b-a>1],colors=MUTED,linewidths=.5,alpha=.15))
    ax.plot(original[:,0],original[:,1],color=CORAL,linewidth=1,alpha=.65,label='Provided original vertex seeds')
    ax.plot(final[:,0],final[:,1],color=TEAL,linewidth=1.7,label='ropt after all 942 arrivals')
    ax.scatter(*final[0],color=TEXT,marker='*',s=100,label='Fixed pose zero')
    ax.set(xlabel='World X / m',ylabel='World Y / m'); ax.legend(loc='upper right',facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID)
    fig.text(.06,.05,'All 1,837 original constraints retained. No truth or optimized seeds. No acquisition timestamps in the source.',color=MUTED,fontsize=9)
    fig.savefig(out/'intel943-trajectory.png'); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,6.4),dpi=130)
    fig.subplots_adjust(left=.08,right=.95,top=.77,bottom=.21,wspace=.28)
    header(fig,'Measured performance, with matching accuracy','Complete recorded Intel943 replay / 6 AB-BA pairs per campaign / single CPU')
    for ax,directory,title,summary in zip(axes,[primary,confirmation],['Primary','Independent confirmation'],[report,second]):
        data = rows(directory/'samples.csv'); style(ax)
        for r in range(6):
            values = [int(next(s for s in data if s['variant']==v and int(s['round'])==r)['elapsed_ns'])/1e9 for v in ['ropt','native']]
            ax.plot([0,1],values,color=MUTED,alpha=.5,linewidth=.8)
            ax.scatter([0,1],values,color=[TEAL,CORAL],s=35,zorder=3)
        for i,v in enumerate(['ropt','native']):
            median=summary['cases'][v]['timing']['median_ms']/1000
            ax.plot([i-.18,i+.18],[median,median],color=TEXT,linewidth=2.5)
            ax.text(i,12.75,f'{median:.3f} s median',ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['ropt','GTSAM iSAM2'],xlim=(-.5,1.5),ylim=(12,22),ylabel='Complete replay / seconds')
        ax.set_title(f"{title}  /  {summary['median_speedup']:.3f}x",fontsize=13,pad=15)
    fig.text(.06,.105,'Every arrival: original-energy, normalized-gradient, fixed-root and pose-agreement gates pass.',fontsize=10,color=TEAL)
    fig.text(.06,.055,'Full refresh accuracy policies, not library defaults. Correct-Jacobian GTSAM 4.2.1. MIT1228 variant is ineligible.',fontsize=9,color=MUTED)
    fig.savefig(out/'isam2-comparison.png'); plt.close(fig)


def rosenbrock(out, trace, inputs):
    inputs.append(trace); data = rows(trace)
    points = np.asarray([[float(r['x']),float(r['y'])] for r in data]); costs = np.asarray([float(r['cost']) for r in data])
    original = .5*(100*(points[:,1]-points[:,0]**2)**2+(1-points[:,0])**2)
    assert np.all(np.isfinite(points)) and np.allclose(costs,original,rtol=1e-12,atol=1e-12)
    assert costs[-1]<1e-12 and np.linalg.norm(points[-1]-1)<1e-6
    fig = plt.figure(figsize=(9,6),dpi=100)
    header(fig,'Follow the nonlinear valley','Rosenbrock / forward automatic differentiation + Levenberg–Marquardt')
    ax = fig.add_axes((.09,.22,.49,.58)); style(ax)
    x,y = np.meshgrid(np.linspace(-1.65,1.4,220),np.linspace(-.6,1.9,220))
    z=.5*(100*(y-x*x)**2+(1-x)**2)
    ax.contour(x,y,z,levels=[.01,.1,1,5,20,80,200],colors=[GRID],linewidths=.9)
    ax.plot(np.linspace(-1.35,1.35,100),np.linspace(-1.35,1.35,100)**2,color=MUTED,alpha=.4,linewidth=.8)
    ax.scatter(1,1,marker='*',s=110,color=TEXT,label='Minimum (1, 1)')
    ax.set(xlim=(-1.65,1.4),ylim=(-.6,1.9),xlabel='x',ylabel='y')
    path, = ax.plot([],[],color=TEAL,linewidth=1.6,marker='o',markersize=3)
    tip, = ax.plot([],[],'o',color=CORAL,markersize=7)
    ax.legend(facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID,fontsize=8,loc='lower left')
    cost_ax=fig.add_axes((.7,.3,.26,.4));style(cost_ax)
    cost_ax.set(yscale='log',xlim=(0,len(data)-1),ylim=(max(costs[-1]/10,1e-28),100),xlabel='Actual LM trial',ylabel='Original cost')
    curve, = cost_ax.plot([],[],color=TEAL,linewidth=1.5)
    label = fig.text(.1,.125,'',family='DejaVu Sans Mono',fontsize=10)
    fig.text(.1,.055,'Actual solver checkpoints; rejected trials retain the accepted state. Synthetic function; no speed claim.',color=MUTED,fontsize=8)
    frames=[]
    for i,row in enumerate(data):
        path.set_data(points[:i+1,0],points[:i+1,1]);tip.set_data([points[i,0]],[points[i,1]])
        curve.set_data(np.arange(i+1),costs[:i+1])
        status='INITIAL' if i==0 else ('ACCEPTED' if row['accepted']=='true' else 'REJECTED')
        label.set_text(f'Trial {i:02d}   |   {status:8s}   |   cost {costs[i]:.3e}')
        frames.append(pixels(fig))
    gif(frames,out/'rosenbrock-convergence.gif',180)
    frames[-1].save(out/'rosenbrock-convergence.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets')
    parser.add_argument('--rosenbrock-trace',type=Path,default=ROOT/'docs/assets/rosenbrock_trace.csv')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);inputs=[]
    base=ROOT/'benchmarks/results/real_isam2'
    recorded(args.output,base/'intel943-primary',base/'intel943-confirmation',inputs)
    rosenbrock(args.output,args.rosenbrock_trace,inputs)
    outputs=['ropt-hero.gif','ropt-hero.png','intel943-trajectory.png','isam2-comparison.png','rosenbrock-convergence.gif','rosenbrock-convergence.png']
    manifest={'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'versions':{'matplotlib':matplotlib.__version__,'numpy':np.__version__,'Pillow':Image.__version__},
              'images':{name:{'bytes':(args.output/name).stat().st_size,'sha256':hashlib.sha256((args.output/name).read_bytes()).hexdigest()} for name in outputs},
              'scope':'Real Intel943 measured constraints with inferred ID arrivals; actual selected prefixes, not real-time. Rosenbrock is synthetic and uses actual LM states, without interpolation. Comparison data comes from both eligible archived campaigns; MIT1228 remains ineligible.'}
    (args.output/'showcase_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['images'],indent=2))


if __name__=='__main__':main()
