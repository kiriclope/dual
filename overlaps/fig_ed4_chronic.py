"""(RENUMBERED 2026-09-15 to citation order: this script draws Extended Data Fig. 5.)
fig_ed4_chronic.py — Extended Data Fig. 5: chronic silencing of the two control projections during
training (companion to Fig. 6b,c). Built 2026-09-15 (Leon: "keep only what is essential") — Results §6
cites it once: "Silencing ACC cell bodies instead of their terminals in mPFC did not produce this deficit,
and silencing the reverse projection, from prelimbic cortex to ACC, impaired the GNG task instead".

Two training batches (behaviour-only cohorts, every-trial silencing, opto against control-illumination
mice), each drawn exactly as Fig. 6b,c draws the ACC→mPFC batch: DPA, GNG and DPA-unpaired learning
curves (mean ± SEM across mice) and the between-group mixed model perf ~ group × day + (1 | mouse).
  a  ACC cell bodies (DualTask-Silencing-ACC)
  b  prelimbic → ACC terminals (DualTask-Silencing-Prl-ACC)
Reads the batch .mat sessions directly (fast; no tensors).
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python fig_ed4_chronic.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig5.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, glob, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.formula.api as smf
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.0
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = PS*8
NOCAP = '--nocap' in sys.argv[1:]
OFF_C, ON_C = '#888888', '#332288'          # control grey · opto indigo (as Fig. 6)
DATA_ROOT = '/storage/leon/dual_task/data/behavior'
BATCHES = [('DualTask-Silencing-ACC', 'ACC cell bodies'), ('DualTask-Silencing-Prl-ACC', 'prelimbic → ACC terminals')]
METRICS = [('DPA', 'performance', lambda df: df.tasks == 'DPA'),
           ('GNG', 'odr_perf', lambda df: df.tasks.isin(['DualGo', 'DualNoGo'])),
           ('DPA unpaired', 'performance', lambda df: (df.tasks == 'DPA') & (df.pair == 0))]


def load_session(path, mouse, day):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    Tr = np.atleast_2d(m['Trials']); out = Tr[:, 2].astype(int); n = len(out)
    perf = np.isin(out, [1, 4]).astype(float); pair = np.isin(out, [1, 2]).astype(int)
    has_gng = 'Trials1' in m and np.size(m['Trials1']) > 0
    if has_gng and 'SampleP' in m and np.size(m['SampleP']) > 0:
        S = np.atleast_2d(m['Sample'])[:, 0].astype('int64'); SP = np.atleast_2d(m['SampleP'])[:, 0].astype('int64')
        isP = np.isin(S, SP)
    else:
        isP = np.ones(n, bool) if not has_gng else np.zeros(n, bool)
    tasks = np.where(isP, 'DPA', '').astype(object); odr = np.full(n, np.nan)
    if has_gng:
        gout = np.atleast_2d(m['Trials1'])[:, 2].astype(int); dual_idx = np.where(~isP)[0]
        k = min(len(dual_idx), len(gout)); dual_idx, gout = dual_idx[:k], gout[:k]
        tasks[dual_idx] = np.where(np.isin(gout, [1, 2]), 'DualGo', 'DualNoGo'); odr[dual_idx] = np.isin(gout, [1, 4]).astype(float)
    return pd.DataFrame({'mouse': mouse, 'day': day, 'tasks': tasks, 'performance': perf, 'odr_perf': odr, 'pair': pair})


def load_batch(batch, group):
    folders = sorted(glob.glob(f'{DATA_ROOT}/{batch}/{group}_mouse_*'), key=lambda p: int(p.rsplit('_', 1)[1]))
    rows = []
    for fol in folders:
        mouse = os.path.basename(fol)
        for f in glob.glob(f'{fol}/session_*.mat'):
            day = int(os.path.basename(f).split('_')[1].split('.')[0]) + 1
            rows.append(load_session(f, mouse, day))
    return pd.concat(rows, ignore_index=True)


def pmd(df, col, mask_fn):
    return df[mask_fn(df)].groupby(['mouse', 'day'])[col].mean().reset_index().rename(columns={col: 'perf'})


def lmm(p1, p2):
    g = pd.concat([p1.assign(grp='control'), p2.assign(grp='opto')], ignore_index=True)
    g['dayc'] = g['day'] - g['day'].mean()
    res = smf.mixedlm("perf ~ C(grp, Treatment('control'))*dayc", g, groups=g['mouse']).fit()
    ci = res.conf_int()
    gn = [i for i in res.params.index if i.startswith('C(grp') and ':' not in i][0]
    it = [i for i in res.params.index if i.startswith('C(grp') and ':' in i][0]
    return (float(res.params[gn]), float(ci.loc[gn, 0]), float(ci.loc[gn, 1]), float(res.pvalues[gn]),
            float(res.params[it]), float(ci.loc[it, 0]), float(ci.loc[it, 1]), float(res.pvalues[it]))


def row(fig, gs, batch, label):
    d1, d2 = load_batch(batch, 'control'), load_batch(batch, 'opto')
    days = list(range(1, int(max(d1.day.max(), d2.day.max())) + 1))
    xt = days if len(days) <= 10 else list(range(2, len(days) + 1, 2))
    axes, coefs = [], []
    for k, (short, col, mask_fn) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        p1, p2 = pmd(d1, col, mask_fn), pmd(d2, col, mask_fn)
        for p, color, lab in [(p1, OFF_C, f'control (n = {p1.mouse.nunique()})'), (p2, ON_C, f'opto (n = {p2.mouse.nunique()})')]:
            m = np.array([p.loc[p.day == dd, 'perf'].mean() for dd in days])
            s = np.array([p.loc[p.day == dd, 'perf'].std(ddof=1) / np.sqrt((p.day == dd).sum()) if (p.day == dd).sum() > 1 else 0 for dd in days])
            ok = np.isfinite(m); x = np.array(days, float)
            ax.plot(x[ok], m[ok], '-o', color=color, lw=1.3, ms=3, label=lab, zorder=3)
            ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)
        coefs.append((short, *lmm(p1, p2)))
        ax.axhline(0.5, ls=':', color='0.5', lw=0.8)
        ax.set_xticks(xt); ax.set_xlabel('training day'); ax.set_ylim(0, 1.05)
        ax.set_title(f'{label}: {short}', loc='left', fontsize=TITLE_FS)
        if k == 0:
            ax.set_ylabel('accuracy'); ax.legend(frameon=False, loc='lower right')
        else:
            ax.tick_params(labelleft=False)
    axF = fig.add_subplot(gs[0, 4]); axes.append(axF)
    for i, (short, gb, glo, ghi, gp, ib, ilo, ihi, ip) in enumerate(coefs):
        for dx, val, vlo, vhi, pv, mk in [(-0.14, gb, glo, ghi, gp, 'o'), (0.14, ib, ilo, ihi, ip, 's')]:
            cc = 'k' if pv < 0.05 else '0.6'
            axF.errorbar(i + dx, val, yerr=[[val - vlo], [vhi - val]], fmt=mk, color=cc, ms=4.5, capsize=2.5, lw=1.1, zorder=3)
            if pv < 0.05:
                axF.text(i + dx, vhi + 0.004, '∗', ha='center', va='bottom', fontsize=PS*12, fontweight='bold')
        print(f'{label:26s} {short:13s} opto−control β={gb:+.3f} [{glo:+.3f}, {ghi:+.3f}] p={gp:.3f}   ×day β={ib:+.4f} p={ip:.3f}')
    axF.axhline(0, ls='--', color='0.4', lw=0.8)
    axF.set_xticks(range(len(coefs))); axF.set_xticklabels([c[0] for c in coefs], rotation=15, ha='right')
    axF.set_xlim(-0.6, len(coefs) - 0.4); axF.set_ylabel('opto − control\n(Δ accuracy)')
    axF.set_title('mixed model', loc='left', fontsize=TITLE_FS)
    axF.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=4.5, label='group (mean day)'),
                        mlines.Line2D([0], [0], marker='s', color='k', ls='none', ms=4.5, label='group × day')],
               frameon=False, fontsize=PS*6.0, loc='lower left')
    return axes[0]


fig = plt.figure(figsize=(10.0, 5.2))
outer = fig.add_gridspec(2, 1, hspace=0.55, left=0.06, right=0.985, top=0.94, bottom=0.10)
axA = row(fig, outer[0].subgridspec(1, 5, wspace=0.22, width_ratios=[1, 1, 1, 0.22, 0.85]), *BATCHES[0])
axB = row(fig, outer[1].subgridspec(1, 5, wspace=0.22, width_ratios=[1, 1, 1, 0.22, 0.85]), *BATCHES[1])
for ax, s in ((axA, 'a'), (axB, 'b')):
    ax.text(-0.22, 1.06, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')

CAP = [
    'Extended Data Fig. 5 | Chronic silencing of the two control projections during training (companion to '
    'Fig. 6b,c). Every-trial silencing throughout training in two further between-group cohorts, drawn as Fig. 6b,c '
    'draws the ACC→mPFC batch: DPA, GNG and DPA-unpaired accuracy against training day (mean ± SEM across mice; '
    'grey, control illumination; indigo, opsin) and the between-group mixed model, accuracy ~ group × day with a '
    'random intercept per mouse (circle, group effect at the mean day; square, group × day slope; 95% CI; ∗ p < 0.05). '
    'a, Silencing ACC cell bodies produced no deficit. b, Silencing the reverse projection, from prelimbic cortex to '
    'ACC, impaired the GNG task and spared DPA. The DPA-selective deficit of Fig. 6b,c is therefore specific to the '
    'ACC→mPFC projection.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig5.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig5.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig5.png')
