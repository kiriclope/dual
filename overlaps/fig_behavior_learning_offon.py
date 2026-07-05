"""
fig_behavior_learning_offon.py — laser OFF vs ON learning curves for the RECORDED
cohort (interleaved laser), same 4-panel layout as the batch --ctrlopto figure.

  A  DPA performance   (OFF vs ON)          vs day
  B  GNG performance   (OFF vs ON)          vs day
  C  DPA unpaired      (OFF vs ON)          vs day
  D  within-mouse LMM laser effect (ON−OFF) per metric — laser β (○, at mean day)
     + laser×day slope (□), 95% CI, from  perf ~ laser*day + (1|mouse).

Because the laser is interleaved trial-by-trial within the same sessions, OFF vs ON
is a WITHIN-mouse contrast (unlike the behavioural batches, which are between-group
opto-vs-control every-trial silencing — see docs/behavior.md).

7 laser mice: 5 Jaws (INHIBITION) + 2 ChR (EXCITATION) — OPPOSITE manipulations, so
the pooled figure cancels real effects; --jaws / --chr give the interpretable ones.
Default produces all three (all / jaws / chr).

Curves: mean ± SEM across mice.  Top stars: per-day one-sample t-test of the per-mouse
ON−OFF Δ vs 0 (exploratory, uncorrected).  Colours: OFF grey / ON indigo.

Output: figures/overlaps/behavior/{png,svg}/behavior_learning_offon[_jaws|_chr].{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_learning_offon.py [--jaws|--chr]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, pickle, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

LASER_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23']
GROUP = {m: ('Jaws' if m.startswith('Jaws') else 'ChR') for m in LASER_MICE}
OFF_C, ON_C = '#888888', '#332288'          # OFF grey / ON indigo
N_MIN = 3

LAB = ('../data/overlaps/'
       'labels_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl')
y = pickle.load(open(LAB, 'rb'))
d = y[y.target == 'choice'].copy()
d = d[d.mouse.isin(LASER_MICE)]
DAYS = sorted(int(x) for x in d.day.unique())

IS_DPA = d.tasks == 'DPA'
IS_DUAL = d.tasks.isin(['DualGo', 'DualNoGo'])
UNP = d.pair == 0
METRICS = [('A  DPA performance', 'DPA',      'performance', IS_DPA),
           ('B  GNG performance', 'GNG',      'odr_perf',    IS_DUAL),
           ('C  DPA unpaired',    'DPA unp.', 'performance', IS_DPA & UNP)]


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def per_mouse_day_laser(col, mask):
    """Per-(mouse, day, laser) accuracy proportion over the `mask` trial subset."""
    m = mask.values
    df = pd.DataFrame({'v': d.loc[m, col].values, 'mouse': d.loc[m, 'mouse'].values,
                       'day': d.loc[m, 'day'].values, 'laser': d.loc[m, 'laser'].values}).dropna()
    return df.groupby(['mouse', 'day', 'laser'], observed=True).v.mean().reset_index(name='perf')


SEL = (['Jaws'] if '--jaws' in sys.argv[1:] else ['ChR'] if '--chr' in sys.argv[1:]
       else ['all', 'Jaws', 'ChR'])
SPEC = {
    'all':  (LASER_MICE, 'behavior_learning_offon',
             '7 laser mice (5 Jaws + 2 ChR — opposite manipulations, pooled)'),
    'Jaws': ([m for m in LASER_MICE if GROUP[m] == 'Jaws'], 'behavior_learning_offon_jaws',
             'Jaws — optogenetic INHIBITION (n=5)'),
    'ChR':  ([m for m in LASER_MICE if GROUP[m] == 'ChR'], 'behavior_learning_offon_chr',
             'ChR — optogenetic EXCITATION (n=2)'),
}
OUT = 'figures/overlaps/behavior'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)


def make(mice, out_name, subtitle):
    n = len(mice)
    print(f'\n=== {out_name}  ({subtitle}) ===')
    fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(19.5, 4.4))
    coefs = []
    for ax, (title, short, col, mask) in zip((axA, axB, axC), METRICS):
        g = per_mouse_day_laser(col, mask)
        g = g[g.mouse.isin(mice)]
        lo, hi = [], []
        for lasval, color, lab in [(0, OFF_C, 'laser OFF'), (1, ON_C, 'laser ON')]:
            sub = g[g.laser == lasval]
            m, s = [], []
            for day in DAYS:
                v = sub.loc[sub.day == day, 'perf'].dropna().values
                m.append(v.mean() if len(v) else np.nan)
                s.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else (0.0 if len(v) else np.nan))
            m, s, x = np.array(m), np.array(s), np.array(DAYS, float)
            ok = ~np.isnan(m)
            ax.plot(x[ok], m[ok], '-o', color=color, lw=2, ms=5, label=lab, zorder=3)
            ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)
            if ok.any():
                lo.append(np.nanmin((m - s)[ok])); hi.append(np.nanmax((m + s)[ok]))
        ylo = max(0.0, min(lo) - 0.05) if lo else 0.3
        yhi = min(1.06, max(hi) + 0.06) if hi else 1.05
        ax.set_ylim(ylo, yhi)
        # per-day within-mouse ON−OFF stars
        piv = g.pivot_table(index=['mouse', 'day'], columns='laser', values='perf')
        piv = piv.dropna()
        if {0, 1}.issubset(piv.columns):
            delta = (piv[1] - piv[0]).reset_index(name='d')
            for day in DAYS:
                dv = delta.loc[delta.day == day, 'd'].values
                if len(dv) >= N_MIN and not np.allclose(dv, dv[0]):
                    pv = float(ttest_1samp(dv, 0.0).pvalue)
                    if star(pv):
                        ax.text(day, yhi - 0.02 * (yhi - ylo), star(pv), ha='center', va='top',
                                fontsize=10, fontweight='bold')
        # within-mouse LMM laser effect: perf ~ laser*day + (1|mouse)
        gg = g.copy(); gg['dayc'] = gg['day'] - gg['day'].mean()
        try:
            if gg.mouse.nunique() < 3:
                raise ValueError('n<3')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = smf.mixedlm('perf ~ laser*dayc', gg, groups=gg['mouse']).fit()
            ci = res.conf_int()
            coefs.append((short, float(res.params['laser']), float(ci.loc['laser', 0]),
                          float(ci.loc['laser', 1]), float(res.pvalues['laser']),
                          float(res.params['laser:dayc']), float(ci.loc['laser:dayc', 0]),
                          float(ci.loc['laser:dayc', 1]), float(res.pvalues['laser:dayc'])))
            print(f'  {short:9s} ON-OFF β={res.params["laser"]:+.3f} p={res.pvalues["laser"]:.4f}'
                  f'  laser×day p={res.pvalues["laser:dayc"]:.4f}')
        except Exception:                                   # n<3 (ChR): mean Δ, no CI/test
            piv2 = g.pivot_table(index=['mouse', 'day'], columns='laser', values='perf').dropna()
            b = float((piv2[1] - piv2[0]).mean()) if {0, 1}.issubset(piv2.columns) else np.nan
            coefs.append((short, b, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            print(f'  {short:9s} ON-OFF β={b:+.3f}  (n<3, no test)')
        if ylo < 0.5 < yhi:
            ax.axhline(0.5, ls=':', color='0.5', lw=1)
        ax.set_xticks(DAYS); ax.set_xlabel('training day')
        ax.legend(frameon=False, fontsize=9, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    axA.set_ylabel('performance')

    # Panel D — within-mouse laser effect per metric: laser β (○) + laser×day β (□)
    for i, (short, lb, llo, lhi, lp, ib, ilo, ihi, ip) in enumerate(coefs):
        for dx, val, vlo, vhi, pv, mk in [(-0.14, lb, llo, lhi, lp, 'o'),
                                          (0.14, ib, ilo, ihi, ip, 's')]:
            if not np.isfinite(val):
                continue
            has_ci = np.isfinite(vlo) and np.isfinite(vhi)
            cc = 'k' if (np.isfinite(pv) and pv < 0.05) else '0.6'
            axD.errorbar(i + dx, val, yerr=([[val - vlo], [vhi - val]] if has_ci else None),
                         fmt=mk, color=cc, ms=6, capsize=3, lw=1.5, zorder=3)
            if np.isfinite(pv) and star(pv):
                axD.text(i + dx, (vhi if has_ci else val) + 0.004, star(pv), ha='center',
                         va='bottom', fontsize=9, fontweight='bold')
    axD.axhline(0, ls='--', color='0.4', lw=1)
    axD.set_xticks(range(len(coefs)))
    axD.set_xticklabels([c[0] for c in coefs], rotation=20, ha='right')
    axD.set_xlim(-0.6, len(coefs) - 0.4)
    axD.set_ylabel('ON − OFF   (β, Δ performance)')
    axD.set_title('D  within-mouse LMM laser effect (95% CI)', loc='left', fontweight='bold', fontsize=11)
    axD.spines[['top', 'right']].set_visible(False)
    axD.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=6, label='laser (at mean day)'),
                        mlines.Line2D([0], [0], marker='s', color='k', ls='none', ms=6, label='laser×day (slope)')],
               frameon=False, fontsize=8, loc='best')

    test = 'within-mouse LMM perf ~ laser×day + (1|mouse)' if n >= 3 else 'n=2 — mean Δ only, no test'
    fig.suptitle(f'Laser OFF vs ON learning curves · {subtitle}', fontsize=13, y=0.99)
    fig.text(0.5, 0.005,
             f'Within-mouse (interleaved laser), mean ± SEM across mice ({n}).  Top stars: per-day one-sample '
             f't-test of per-mouse ON−OFF Δ vs 0 (exploratory).  Panel D: {test}.  * p<0.05  ** p<0.01  *** p<0.001',
             ha='center', va='bottom', fontsize=8, color='0.35')
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/{out_name}.{ext}'
        fig.savefig(p, bbox_inches='tight'); print('  saved', os.path.abspath(p))
    plt.close(fig)


for key in SEL:
    make(*SPEC[key])
