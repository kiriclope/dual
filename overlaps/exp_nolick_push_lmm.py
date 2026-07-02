"""
exp_nolick_push_lmm.py — TRIAL-LEVEL mixed-effects test of the no-lick push.

The n=9 collapse-to-mouse-means test (`exp_nolick_push_stats.py`) throws away
within-mouse power. Here we keep individual trials and model the mouse structure
explicitly with a linear mixed model. The MAXIMAL (correctly-specified) model gives
random slopes to BOTH within-mouse factors — stage AND sample:

    depth ~ expert + C(sample) + (1 + expert + C(sample) | mouse)   # MAXIMAL (primary)
    depth ~ expert             + (1 + expert            | mouse)     # stage-slope only
    depth ~ expert             + (1                     | mouse)     # random-intercept (anti-cons.)

  depth  = per-trial late-delay (BINS_LATE 27-53) choice-code decision function,
           per-mouse BL-std normalised — same quantity as the mouse-mean test, but
           NOT averaged over trials.
  expert = 1 for Expert, 0 for Naive  → the `expert` coefficient IS Expert-Naive
           (negative = the state deepens into the no-lick half with learning).

Why maximal: BOTH `stage` and `sample` vary WITHIN mouse, so both need random slopes
(Barr et al. 2013 "keep it maximal"); dropping the sample slope forces its variance
into the residual. This better-specified model is more powerful, NOT p-hacking.

RESULT (delay axis): the MAXIMAL model reaches significance — pooled deepening
β=-0.98, **p=0.024** (all trials, converged); delay/correct β=-0.86, p=0.047 (did
NOT converge → caution). Test axis n.s. (0.14-0.25) — consistent with delay being the
principled axis. The conservative n=9 mouse-mean test (~0.07-0.09) is the cross-check;
statsmodels' Wald z is optimistic at 9 groups, so treat p≈0.02-0.05 as "significant
under the correct model, borderline under the most conservative test."

Tests, per (train axis, trial set):
  (A) pooled deepening      depth ~ expert              (all A+B trials)
  (B) per-sample deepening  same, on A-only and B-only subsets
  (C) A/B asymmetry         depth ~ expert * sampleB    (interaction = does the
                            deepening differ between A and B?)

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python exp_nolick_push_lmm.py
"""

import os, sys, warnings
sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.common.options import set_options
from src.pca.io import pkl_load

warnings.filterwarnings('ignore')   # LMM convergence chatter — handled explicitly below

# ── Config (matched to exp_nolick_push_stats.py) ──────────────────────────────

DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
            'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']

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
TRAIN_EPOCHS = [('trainDELAY', options['bins_DELAY']),   # preferred axis
                ('trainTEST',  options['bins_TEST'])]    # canonical / reference

X_single = pkl_load(f'X_{DUM}',      path=DATA_IN)
y_single = pkl_load(f'labels_{DUM}', path=DATA_IN)

idx_laser   = (y_single.laser == 0)
idx_correct = (idx_laser & (y_single.performance == 1) &
               ((y_single.tasks == 'DPA') | (y_single.odr_perf == 1)))
TRIAL_SETS = [('correct', idx_correct), ('all', idx_laser)]

# ── Build a trial-level long dataframe ────────────────────────────────────────

def choice_axis(bins_train):
    X_ep = X_single[..., bins_train, :].mean(-2)[:, 1].astype(float)
    for mouse in ALL_MICE:
        m  = (y_single.mouse == mouse).values
        sd = X_ep[m][:, BINS_BL].std()
        if sd > 0:
            X_ep[m] /= sd
    return X_ep

def trial_df(bins_train, trial_mask):
    """One row per DPA choice-target trial: depth, expert, sampleB, mouse."""
    base = ((y_single.tasks == 'DPA') & (y_single.target == 'choice') & trial_mask).values
    X_ep = choice_axis(bins_train)
    depth = X_ep[:, BINS_LATE].mean(1)                      # per-trial late-delay depth
    df = pd.DataFrame({
        'depth':   depth[base],
        'mouse':   y_single.mouse.values[base],
        'stage':   y_single.stage.values[base],
        'op':      y_single.odor_pair.values[base],
    })
    df['expert']  = (df.stage == 'Expert').astype(float)
    df['sampleB'] = df.op.isin([2, 3]).astype(float)        # A=[0,1]->0, B=[2,3]->1
    df['sample']  = np.where(df.sampleB == 1, 'B', 'A')      # categorical, for C(sample)
    return df

# ── Fit helpers ───────────────────────────────────────────────────────────────

def fit(formula, df, re_formula):
    """Return (coef_dict, converged) for a mixedlm; coef_dict[name] = (beta, se, p)."""
    try:
        md  = smf.mixedlm(formula, df, groups=df['mouse'], re_formula=re_formula)
        res = md.fit(reml=True, method='lbfgs')
        conv = bool(res.converged)
        out = {n: (res.params[n], res.bse[n], res.pvalues[n]) for n in res.params.index
               if n not in ('Group Var',) and 'Var' not in n and 'Cov' not in n}
        return out, conv
    except Exception as e:
        return {'_error': str(e)}, False

def report_term(label, coef, term, n):
    if '_error' in coef:
        print(f'  {label:26s} FAILED: {coef["_error"][:60]}')
        return
    if term not in coef:
        print(f'  {label:26s} term "{term}" absent')
        return
    b, se, p = coef[term]
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f'  {label:26s} β={b:+.3f}  SE={se:.3f}  p={p:.4f}  {star}   (n_trials={n})')

# ── Run ───────────────────────────────────────────────────────────────────────

for train_tag, bins_train in TRAIN_EPOCHS:
    for ts_tag, ts_mask in TRIAL_SETS:
        df = trial_df(bins_train, ts_mask)
        n_by_mouse = df.groupby('mouse').size()
        print(f'\n{"="*76}\n{train_tag} | {ts_tag} trials | DPA | depth ~ expert + (…|mouse)'
              f'\n  {len(df)} trials, {df.mouse.nunique()} mice '
              f'(trials/mouse {n_by_mouse.min()}-{n_by_mouse.max()})\n{"="*76}')

        # (A) pooled deepening. MAXIMAL = random slopes for BOTH within-mouse factors
        #     (stage AND sample) — this is the correctly-specified model. The stage-only
        #     and random-intercept rows are shown for comparison only.
        print('(A) POOLED deepening  (β = Expert-Naive; negative = deeper into no-lick):')
        cM, convM = fit('depth ~ expert + C(sample)', df, re_formula='~expert + C(sample)')
        report_term(f'MAXIMAL (1+exp+samp|mouse){" *NC*" if not convM else ""}',
                    cM, 'expert', len(df))
        cS, convS = fit('depth ~ expert', df, re_formula='~expert')
        report_term(f'stage-slope only (1+exp|mouse){" *NC*" if not convS else ""}',
                    cS, 'expert', len(df))
        cI, _ = fit('depth ~ expert', df, re_formula='~1')
        report_term('random-intercept (anti-cons.)', cI, 'expert', len(df))

        # (B) per-sample deepening
        print('(B) per-sample deepening (maximal):')
        for cls, val in [('A', 0.0), ('B', 1.0)]:
            sub = df[df.sampleB == val]
            c, nc = fit('depth ~ expert', sub, re_formula='~expert')
            report_term(f'{cls}{" *NC*" if not nc else ""}', c, 'expert', len(sub))

        # (C) A/B asymmetry — interaction expert:sampleB (maximal main-effect slopes)
        print('(C) A/B asymmetry  (interaction expert:sampleB; n.s. = A,B deepen alike):')
        cX, ncX = fit('depth ~ expert * sampleB', df, re_formula='~expert + sampleB')
        report_term(f'expert:sampleB{" *NC*" if not ncX else ""}', cX, 'expert:sampleB', len(df))

print('\nMAXIMAL = random slopes for BOTH within-mouse factors (stage & sample) — the '
      'correctly-specified model (Barr et al. 2013). *NC* = did not converge → treat that '
      'β/p with caution. Random-intercept is anti-conservative (ignores within-mouse slopes) '
      'and shown only for contrast. statsmodels uses Wald z (optimistic at 9 groups); the '
      'n=9 mouse-mean paired test (~0.07-0.09) is the conservative cross-check.')
