"""fig_axis_trajectories.py — the manifold's trajectories projected on the NAMED axes (Leon 2026-09-18: "can we get from
these the actual sample axis so that we can project trajectory on it, and then same for gng and choice?").

A UMAP map has no inverse, so an axis cannot be read out of it. The axes here are the condition-mean contrast directions
in neuron space — sample = mean(A) - mean(B), gng = mean(Go) - mean(NoGo), choice = mean(match) - mean(nonmatch) — each
fitted at its own window and unit-normalised, and each fitted on one half of the trials while the trajectories projected
on it come from the OTHER half (12 splits, both directions averaged; results.pkl['CODE_TRAJ'], exp_code_trajectories.py).
Without that split the projection is inflated by self-inclusion, the leak that once faked a mid-delay lick separation.

Layout: rows = trial set x stage; columns = the three axes' time courses plus the sample x choice plane the trajectories
live in. Traces are in z units along the unit axis; the shaded band is the s.d. over splits.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python fig_axis_trajectories.py
Output: figures/pseudo/dimensionality/{png,svg}/axis_trajectories.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.linewidth': 0.7,
})
TITLE_FS = 8
NB = 84
T = np.arange(NB) / 6.0
SAMPC = {0: '#332288', 1: '#44AA99'}; GNGC = {'DualGo': '#023eff', 'DualNoGo': '#1ac938'}; MATCHC = {True: '#4daf4a', False: '#377eb8'}
EPOCHS = [(2.0, 3.0, '#332288', 'sample'), (4.5, 5.5, '#cc3311', 'GNG'), (6.5, 7.0, '#ee7733', 'cue'), (9.0, 10.0, '#377eb8', 'test')]
SETS = ['DPA', 'dual']
STAGES = ['Naive', 'Expert']
AXES = ['sample', 'gng', 'choice']
GROUP = {'sample': (lambda c: c[1] == 0, SAMPC, {0: 'sample A', 1: 'sample B'}),
         'gng': (lambda c: c[0] == 'DualGo', GNGC, {'DualGo': 'Go', 'DualNoGo': 'NoGo'}),
         'choice': (lambda c: c[1] == c[2], MATCHC, {True: 'match', False: 'nonmatch'})}

C = pickle.load(open('figures/pseudo/dimensionality/results.pkl', 'rb'))['CODE_TRAJ']
CONDS = {s: sorted({k[3] for k in C if k[1] == s}) for s in SETS}


def colour(ax, cd):
    if ax == 'sample':
        return SAMPC[cd[1]]
    if ax == 'gng':
        return GNGC[cd[0]]
    return MATCHC[cd[1] == cd[2]]


def shade(a):
    for t0, t1, col, nm in EPOCHS:
        a.axvspan(t0, t1, color=col, alpha=0.10, lw=0)


rows = [(s, st) for s in SETS for st in STAGES]
fig, axs = plt.subplots(len(rows), 4, figsize=(13.0, 2.9 * len(rows)))
for r, (setname, stage) in enumerate(rows):
    conds = CONDS[setname]
    for c, axname in enumerate(AXES):
        ax = axs[r, c]
        if setname == 'DPA' and axname == 'gng':
            ax.axis('off')
            ax.text(0.5, 0.5, 'no Go/NoGo odor\non DPA trials', transform=ax.transAxes, ha='center', va='center', fontsize=7, color='0.5')
            continue
        shade(ax)
        for cd in conds:
            k = (stage, setname, axname, cd)
            if k not in C:
                continue
            m = np.asarray(C[k]['raw'], float); sd = np.asarray(C[k]['raw_sd'], float)
            ls = '-' if cd[1] == cd[2] else '--'
            ax.plot(T, m, ls, color=colour(axname, cd), lw=1.2, alpha=0.95, zorder=3)
            ax.fill_between(T, m - sd, m + sd, color=colour(axname, cd), alpha=0.12, lw=0, zorder=2)
        ax.axhline(0, ls=':', color='0.6', lw=0.7)
        ax.set_xlim(0, 13.5)
        ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · {axname} axis', loc='left', fontsize=TITLE_FS)
        if c == 0:
            ax.set_ylabel('projection (z)')
        if r == len(rows) - 1:
            ax.set_xlabel('time (s)')
    # the plane the trajectories live in: sample x choice
    ax = axs[r, 3]
    for cd in conds:
        ks, kc = (stage, setname, 'sample', cd), (stage, setname, 'choice', cd)
        if ks not in C or kc not in C:
            continue
        x = np.asarray(C[ks]['raw'], float); y = np.asarray(C[kc]['raw'], float)
        ls = '-' if cd[1] == cd[2] else '--'
        ax.plot(x, y, ls, color=SAMPC[cd[1]], lw=1.2, alpha=0.9, zorder=3)
        ax.scatter(x[0], y[0], s=16, color=SAMPC[cd[1]], marker='o', lw=0, zorder=5)
        for t0, _, col, nm in EPOCHS:
            b = int(t0 * 6)
            ax.scatter(x[b], y[b], s=13, color=col, marker='s', lw=0, zorder=6)
        if setname == 'dual':
            ax.scatter(x[-1], y[-1], s=52, color=GNGC[cd[0]], marker='*', edgecolors='k', linewidths=0.4, zorder=7)
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xlabel('sample axis (z)'); ax.set_ylabel('choice axis (z)')
    ax.set_title(f'{setname} · {"naïve" if stage == "Naive" else "expert"} · sample × choice plane', loc='left', fontsize=TITLE_FS)
hs = [mlines.Line2D([0], [0], color=SAMPC[s], label=f'sample {"A" if s == 0 else "B"}') for s in SAMPC] + \
     [mlines.Line2D([0], [0], color=GNGC[g], label='Go' if g == 'DualGo' else 'NoGo') for g in GNGC] + \
     [mlines.Line2D([0], [0], color=MATCHC[m], label='match' if m else 'nonmatch') for m in MATCHC] + \
     [mlines.Line2D([0], [0], color='0.4', ls='-', label='match (line)'), mlines.Line2D([0], [0], color='0.4', ls='--', label='nonmatch (line)')] + \
     [mlines.Line2D([0], [0], marker='s', ls='none', color=col, label=nm) for _, _, col, nm in EPOCHS]
fig.legend(handles=hs, loc='lower center', ncol=7, frameon=False, fontsize=6.5)
fig.suptitle('Trajectories projected on the named axes (contrast directions fitted on held-out trials; 12 splits, band = s.d. over splits)', fontsize=9)
fig.tight_layout(rect=(0, 0.035, 1, 0.975))
OUT = 'figures/pseudo/dimensionality'
for sub in ('png', 'svg'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
fig.savefig(f'{OUT}/png/axis_trajectories.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/axis_trajectories.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/axis_trajectories.png')

# ── how far the trajectory travels along each axis (the geometry, quantified) ──
print('\npeak separation along each axis (held-out, z units): |mean(group1) - mean(group2)| at its window')
WIN = {'sample': slice(33, 39), 'gng': slice(33, 39), 'choice': slice(54, 63)}
for setname in SETS:
    for axname in AXES:
        if setname == 'DPA' and axname == 'gng':
            continue
        line = f'  {setname:4s} {axname:6s}'
        for stage in STAGES:
            g1, g2 = [], []
            for cd in CONDS[setname]:
                k = (stage, setname, axname, cd)
                if k not in C:
                    continue
                v = np.asarray(C[k]['raw'], float)[WIN[axname]].mean()
                (g1 if GROUP[axname][0](cd) else g2).append(v)
            line += f'   {stage[:3]} {abs(np.mean(g1) - np.mean(g2)):5.2f}'
        print(line, flush=True)
