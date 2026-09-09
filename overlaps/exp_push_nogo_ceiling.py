"""exp_push_nogo_ceiling.py — ΔNoGo accuracy against the no-lick push (Leon, 2026-09-09: "delta nogo vs delta depth").
Three per-mouse scatters (n = 9): (a) ΔNoGo accuracy vs Δdepth (the +0.6 'opposite-sign' trend of Fig 4c's NoGo arm);
(b) ΔNoGo vs NAIVE NoGo accuracy (the ceiling: mice already withholding well cannot gain); (c) Δdepth vs naive NoGo
accuracy (the deep pushers are the good naive withholders). Prints the partial rank correlation of Δdepth with ΔNoGo
controlling for naive NoGo accuracy. Same estimator/units as Fig 4c (main_panels: per-mouse means, DPA delay depth).
Run from overlaps/:  /home/leon/mambaforge/envs/dual/bin/python exp_push_nogo_ceiling.py"""
import sys, os
sys.path.insert(0, '/home/leon/dual/'); os.chdir('/home/leon/dual/overlaps'); sys.path.insert(0, '/home/leon/dual/overlaps')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
import main_panels as MP
import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 400, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7})
TITLE_FS = 8
MCOL = dict(zip(MP.ALL_MICE, sns.color_palette('tab10')))

y = MP.y
nogo = [t for t in y.tasks.unique() if 'nogo' in t.lower()][0]
rows = []
for mo in MP.ALL_MICE:
    dd = np.nanmean([MP.delta_choice_sample[(mo, c)] for c, _ in MP.D_SAMPLE_CLASSES])
    acc = {}
    for st in MP.STAGES:
        m = (y.mouse == mo) & (y.stage == st) & MP.idx_laser & MP.idx_choice & (y.tasks == nogo)
        acc[st] = y.loc[m, 'odr_perf'].mean()
    rows.append(dict(mouse=mo, ddepth=dd, nogo_N=acc['Naive'], nogo_E=acc['Expert'], dNoGo=acc['Expert'] - acc['Naive']))
df = pd.DataFrame(rows)


def partial_rank(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    ea = ra - np.polyval(np.polyfit(rc, ra, 1), rc); eb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    return pearsonr(ea, eb)


fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
specs = [('ddepth', 'dNoGo', 'Δ DPA choice-code depth (Exp−Naive)', 'Δ NoGo accuracy (Exp−Naive)', 'ΔNoGo vs push'),
         ('nogo_N', 'dNoGo', 'naive NoGo accuracy', 'Δ NoGo accuracy (Exp−Naive)', 'the ceiling'),
         ('nogo_N', 'ddepth', 'naive NoGo accuracy', 'Δ DPA choice-code depth (Exp−Naive)', 'deep pushers withheld well already')]
for ax, (xk, yk, xl, yl, ttl) in zip(axes, specs):
    for _, r in df.iterrows():
        ax.scatter(r[xk], r[yk], s=34, color=MCOL[r['mouse']], edgecolors='w', linewidths=0.5, zorder=3)
    rho, p = spearmanr(df[xk], df[yk])
    ax.text(0.03, 0.03, f'n=9 Spearman\nρ={rho:+.2f}, p={p:.3f}', transform=ax.transAxes, va='bottom', fontsize=6.5, color='0.3')
    ax.axhline(0, ls=':', color='k', lw=0.7)
    if xk == 'ddepth':
        ax.axvline(0, ls=':', color='k', lw=0.7)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
rp, pp = partial_rank(df.ddepth, df.dNoGo, df.nogo_N)
fig.suptitle(f'ΔNoGo vs Δdepth is a ceiling effect: partial rank correlation Δdepth ~ ΔNoGo | naive NoGo accuracy  r={rp:+.2f}, p={pp:.2f}',
             fontsize=8, y=1.02)
fig.tight_layout()
OUT = 'figures/overlaps/controls'
os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ['png', 'svg']:
    fig.savefig(f'{OUT}/{ext}/push_nogo_ceiling.{ext}', bbox_inches='tight')
print(df.sort_values('ddepth').round(3).to_string(index=False))
for a, b, lab in [('ddepth', 'dNoGo', 'Δdepth vs ΔNoGo'), ('nogo_N', 'dNoGo', 'naive NoGo vs ΔNoGo'), ('nogo_N', 'ddepth', 'naive NoGo vs Δdepth')]:
    print(f'{lab:22s} Spearman ρ={spearmanr(df[a], df[b])[0]:+.2f} p={spearmanr(df[a], df[b])[1]:.3f}')
print(f'partial rank Δdepth ~ ΔNoGo | naive NoGo: r={rp:+.2f} p={pp:.3f}')
print('saved', os.path.abspath(f'{OUT}/png/push_nogo_ceiling.png'))
