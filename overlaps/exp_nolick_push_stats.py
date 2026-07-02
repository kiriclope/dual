"""
exp_nolick_push_stats.py — make the no-lick push AIRTIGHT (the STRONG headline).

Claim under test (Expert DPA, late delay, choice code):
  (1) Both A and B Expert-DPA delay states sit in the no-lick (negative) half of
      the choice axis   -> one-sample tests vs 0, per sample class.
  (2) The push DEEPENS with learning (Naive -> Expert)   -> paired test.
  (3) It is ASYMMETRIC: A strong, B weak   -> paired A-vs-B test in Expert.

Depth = mean over BINS_LATE (27-53, ~4.3-9 s) of the choice-code decision function
        (target=='choice', train-epoch averaged, per-mouse BL-std normalised) — the
        SAME quantity as `plot_scatter_perf.py` ("choice loc.").

We report, per test: n, mean, 95 % bootstrap CI over mice (10 000 resamples),
Wilcoxon signed-rank (exact, the honest n=9 test), paired/one-sample t, and Cohen's
dz effect size. CIs and dz are emphasised over p at n=9. Everything is run for:
  - train epoch:  trainTEST (canonical, headline) + trainDELAY (robustness)
  - trial set:    correct only (headline) + all laser-off (robustness; push must survive)

Also saves a per-mouse paired-line figure (Naive->Expert, A and B, DPA) with group
mean +/- bootstrap CI.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python exp_nolick_push_stats.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_1samp, ttest_rel

from src.common.options import set_options
from src.pca.io import pkl_load

# ── Config ────────────────────────────────────────────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
FIG_BASE = './figures/overlaps/nolick_push'

ALL_MICE   = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
              'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES     = ['Naive', 'Expert']

options = set_options(
    mice=ALL_MICE, tasks=['Dual'], mouse=ALL_MICE[0], laser=0,
    trials='', data_type='dF', prescreen=None, pval=0.05,
    preprocess=None, scaler_BL='standard_BL', avg_noise=False, unit_var_BL=False,
    random_state=None, T_WINDOW=0.0, l1_ratio=0.95, n_comp=3, pca='pca', scaler=None,
    bootstrap=1, n_boots=128, n_splits=5, n_repeats=10,
    class_weight=0, multilabel=0, mne_estimator='generalizing', n_jobs=64,
    days=['first', 'last'],
)
BINS_BL   = options['bins_BL']
BINS_LATE = np.arange(27, 54)

TRAIN_EPOCHS = [('trainTEST', options['bins_TEST']),      # canonical / headline
                ('trainDELAY', options['bins_DELAY'])]    # robustness

# Sample A = odor_pairs [0,1] (indigo), Sample B = [2,3] (teal)  — CLAUDE.md convention
SAMPLE_CLASSES = [('A', [0, 1]), ('B', [2, 3])]

N_BOOT = 10_000
RNG    = np.random.default_rng(0)

# ── Load ──────────────────────────────────────────────────────────────────────

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)

idx_laser   = (y_single.laser == 0)
idx_correct = (
    idx_laser &
    (y_single.performance == 1) &
    ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1))
)

TRIAL_SETS = [('correct', idx_correct),          # headline
              ('all-laser-off', idx_laser)]      # robustness (push must survive)

# ── Depth extraction ──────────────────────────────────────────────────────────

def choice_axis(bins_train):
    """Per-mouse BL-normalised choice-code decision function, train-epoch averaged.
    Returns X_ep (n_trials, 84)."""
    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd
    return X_ep

def depths(bins_train, trial_mask, cond='DPA'):
    """{(mouse, stage, cls): depth}  — late-delay choice-code mean, correct/laser trials."""
    X_ep = choice_axis(bins_train)
    out = {}
    for mouse in ALL_MICE:
        for stage in STAGES:
            for cls, pairs in SAMPLE_CLASSES:
                m = (
                    (y_single.mouse  == mouse) &
                    (y_single.tasks  == cond)  &
                    (y_single.stage  == stage) &
                    (y_single.target == 'choice') &
                    trial_mask &
                    y_single.odor_pair.isin(pairs)
                ).values
                out[(mouse, stage, cls)] = (X_ep[m][:, BINS_LATE].mean()
                                            if m.sum() else np.nan)
    return out

# ── Stats helpers ─────────────────────────────────────────────────────────────

def boot_ci(vals, func=np.mean, alpha=0.05):
    """Percentile bootstrap CI over the mouse axis (resample mice with replacement)."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    stat = func(v[idx], axis=1)
    return np.percentile(stat, 100 * alpha / 2), np.percentile(stat, 100 * (1 - alpha / 2))

def cohen_dz(diffs):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    return d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan

def one_sample(vals, label, side='less'):
    """H0: median=0. Directional H1: depth < 0 (pushed into no-lick), so side='less'."""
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    lo, hi = boot_ci(v)
    try:
        w2 = wilcoxon(v).pvalue
        w1 = wilcoxon(v, alternative=side).pvalue
    except ValueError:
        w2 = w1 = np.nan
    t2 = ttest_1samp(v, 0).pvalue
    print(f'  {label:22s} n={len(v)}  mean={v.mean():+.3f}  '
          f'95%CI[{lo:+.3f},{hi:+.3f}]  '
          f'Wilcoxon p2={w2:.3f}/p1={w1:.3f}  t p2={t2:.3f}  dz={cohen_dz(v):+.2f}')
    return v.mean(), (lo, hi)

def paired(a, b, label, side='less'):
    """Paired test on d = b - a. Directional H1 default side='less' (d<0, i.e. deeper)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = b - a
    lo, hi = boot_ci(d)
    try:
        w2 = wilcoxon(d).pvalue
        w1 = wilcoxon(d, alternative=side).pvalue
    except ValueError:
        w2 = w1 = np.nan
    t2 = ttest_rel(b, a).pvalue
    n_dir = int((d < 0).sum()) if side == 'less' else int((d > 0).sum())
    tag = 'deeper' if side == 'less' else 'A-deeper'
    print(f'  {label:22s} n={len(d)}  Δ(mean)={d.mean():+.3f}  '
          f'95%CI[{lo:+.3f},{hi:+.3f}]  '
          f'Wilcoxon p2={w2:.3f}/p1={w1:.3f}  t p2={t2:.3f}  dz={cohen_dz(d):+.2f}  '
          f'{n_dir}/{len(d)} {tag}')
    return d

# ── Run the battery ───────────────────────────────────────────────────────────

def per_mouse(dep, stage, cls):
    return np.array([dep[(m, stage, cls)] for m in ALL_MICE], float)

for train_tag, bins_train in TRAIN_EPOCHS:
    for ts_tag, ts_mask in TRIAL_SETS:
        print(f'\n{"="*78}\n{train_tag}  |  {ts_tag} trials  |  DPA  |  late delay '
              f'(bins {BINS_LATE[0]}-{BINS_LATE[-1]})\n{"="*78}')
        dep = depths(bins_train, ts_mask)

        A_exp = per_mouse(dep, 'Expert', 'A'); A_nai = per_mouse(dep, 'Naive', 'A')
        B_exp = per_mouse(dep, 'Expert', 'B'); B_nai = per_mouse(dep, 'Naive', 'B')
        pool_exp = np.nanmean([A_exp, B_exp], axis=0)
        pool_nai = np.nanmean([A_nai, B_nai], axis=0)

        print('\n(1) Expert DPA depth in no-lick half  (H0: depth = 0):')
        one_sample(A_exp,    'A (Expert)')
        one_sample(B_exp,    'B (Expert)')
        one_sample(pool_exp, 'A&B pooled (Expert)')

        print('\n(2) Deepening with learning  (Expert - Naive, paired):')
        paired(A_nai, A_exp,       'A  Naive->Expert')
        paired(B_nai, B_exp,       'B  Naive->Expert')
        paired(pool_nai, pool_exp, 'pooled Naive->Expert')

        print('\n(3) A/B asymmetry in Expert  (B - A, paired; H1: A deeper => B-A>0):')
        paired(A_exp, B_exp, 'B vs A (Expert)', side='greater')

# ── Figure: per-mouse Naive->Expert paired lines — A, B, and A&B POOLED ───────
#    Axis + trial set selectable:  python exp_nolick_push_stats.py [test|delay] [correct|all]
#    Default = delay axis, correct trials (the preferred axis; A/B pushed comparably).

FIG_TRAIN = {'test': ('trainTEST', options['bins_TEST']),
             'delay': ('trainDELAY', options['bins_DELAY'])}
fig_axis, fig_ts = 'delay', 'correct'
for a in sys.argv[1:]:
    if a in FIG_TRAIN:            fig_axis = a
    elif a in ('correct', 'all'): fig_ts = a
fig_train_tag, fig_bins = FIG_TRAIN[fig_axis]
fig_mask = idx_correct if fig_ts == 'correct' else idx_laser
ts_label = 'DPA correct' if fig_ts == 'correct' else 'DPA all trials'

dep = depths(fig_bins, fig_mask)

def panel_arrays(key):
    """Naive, Expert per-mouse depth arrays. key in {'A','B','pooled'}."""
    if key in ('A', 'B'):
        return per_mouse(dep, 'Naive', key), per_mouse(dep, 'Expert', key)
    nai = np.nanmean([per_mouse(dep, 'Naive', 'A'), per_mouse(dep, 'Naive', 'B')], axis=0)
    exp = np.nanmean([per_mouse(dep, 'Expert', 'A'), per_mouse(dep, 'Expert', 'B')], axis=0)
    return nai, exp

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

sns.set_context('talk'); sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)
matplotlib.rcParams['svg.fonttype'] = 'none'

PANELS = [('A', '#332288', 'Sample A'),
          ('B', '#44AA99', 'Sample B'),
          ('pooled', '#444444', 'A & B pooled')]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)

for ax, (key, color, title) in zip(axes, PANELS):
    nai, exp = panel_arrays(key)
    for n, e in zip(nai, exp):
        if not (np.isnan(n) or np.isnan(e)):
            ax.plot([0, 1], [n, e], '-', color=color, lw=1, alpha=0.4)
            ax.scatter([0, 1], [n, e], color=color, s=28,
                       zorder=5, edgecolors='w', linewidths=0.5)
    # group mean +/- bootstrap CI
    for x, vals in [(0, nai), (1, exp)]:
        v = vals[~np.isnan(vals)]
        lo, hi = boot_ci(v)
        ax.errorbar(x + 0.06, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                    fmt='o', color='k', ms=7, capsize=4, lw=1.6, zorder=6)
    # paired Naive->Expert deepening test. Star = TWO-SIDED Wilcoxon (conservative
    # default, post-hoc analysis); one-sided reported alongside for transparency.
    keep = ~(np.isnan(nai) | np.isnan(exp))
    d = exp[keep] - nai[keep]
    try:
        p2 = wilcoxon(d).pvalue
        p1 = wilcoxon(d, alternative='less').pvalue
    except ValueError:
        p2 = p1 = np.nan
    dz = cohen_dz(d); n_deep = int((d < 0).sum())
    ax.text(0.5, 0.98, stars(p2), transform=ax.transAxes, ha='center', va='top',
            fontsize=20, fontweight='bold', color=color)
    ax.text(0.5, 0.88,
            f'p={p2:.3f} (2-sided; 1-sided {p1:.3f})\n'
            f'dz={dz:+.2f}   {n_deep}/{keep.sum()} deeper',
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color='0.3')
    ax.axhline(0, ls='--', color='k', lw=0.8)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1]); ax.set_xticklabels(['Naive', 'Expert'])
    ax.set_title(title)
    if ax is axes[0]:
        ax.set_ylabel(f'choice-code depth\nlate delay, {ts_label}')

fig.suptitle(f'No-lick push: late-delay choice code deepens Naive→Expert '
             f'({fig_train_tag})   *=2-sided Wilcoxon p<0.05', fontsize=12)
fig.tight_layout()
os.makedirs(os.path.join(FIG_BASE, 'png'), exist_ok=True)
os.makedirs(os.path.join(FIG_BASE, 'svg'), exist_ok=True)
stem = f'{DUM}_nolick_push_paired_{fig_axis}_{fig_ts}'
fig.savefig(os.path.join(FIG_BASE, 'png', f'{stem}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_BASE, 'svg', f'{stem}.svg'), bbox_inches='tight')
plt.close(fig)
print(f'\nsaved {FIG_BASE}/png/{stem}.png  ({fig_train_tag}, {ts_label})')
