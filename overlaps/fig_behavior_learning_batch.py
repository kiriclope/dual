"""
fig_behavior_learning_batch.py — behaviour_learning curves for the training-batch
data (behaviour-only cohorts, no recordings) in
    /storage/leon/dual_task/data/behavior/<batch>/<group>_mouse_*/session_N.mat

Same 5-panel design as overlaps/fig_behavior_learning.py (A DPA-vs-GNG, B Go-vs-NoGo,
C paired-vs-unpaired, D unpaired-by-task, E LMM coefficient forest; per-mouse/day
accuracy proportions + random-intercept LMM), but reads the batch .mat format and
produces one figure PER GROUP (control / opto / DPA / Dual).

Extraction recipe (verified) — every trial array is (N,4); the OUTCOME is column
index 2 (col3 is a stimulus label, NOT the response):
    Trials[:,2] : 1=hit 2=miss 3=FA 4=CR
    performance = col2 in {1,4};  paired = col2 in {1,2}, unpaired = {3,4}
    pure vs dual: isP = Sample[:,0] in SampleP[:,0]  (pure → tasks='DPA')
    GNG (odr_perf): Trials1[:,2] in {1,4} on the dual trials (row-aligned, chronological);
                    Go = Trials1[:,2] in {1,2} → 'DualGo', else 'DualNoGo'
Single-DPA mice: only Trials (all pure DPA, no GNG). Dual-only mice: all dual, no pure.
Laser is group-level only (opto vs control by folder), so this is a between-group
reproduction, not a within-mouse ON−OFF contrast.

Output: figures/overlaps/behavior/batch/behavior_learning_batch_<batch>_<group>.{png,svg}

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_behavior_learning_batch.py \
            [--batch DualTask-Silencing-ACC] [--group control|opto|DPA|Dual]
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, glob, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.io as sio
import statsmodels.formula.api as smf

matplotlib.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
})

RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
N_MIN = 4
DATA_ROOT = '/storage/leon/dual_task/data/behavior'
SHORT = {'DualTask-Silencing-ACC': 'ACC', 'DualTask-Silencing-ACC-Prl': 'ACCPrl',
         'DualTask-Silencing-Prl-ACC': 'PrlACC', 'DualTask_DPA_vs_Single_DPA': 'DPAvsSingle'}

BATCH = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--batch'),
             'DualTask-Silencing-ACC')
GROUP = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--group'), 'control')


# ── Load one session → per-trial rows ─────────────────────────────────────────
def load_session(path, mouse, day):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    Tr = np.atleast_2d(m['Trials'])
    out = Tr[:, 2].astype(int)
    n = len(out)
    perf = np.isin(out, [1, 4]).astype(float)        # DPA correct
    pair = np.isin(out, [1, 2]).astype(int)          # 1=paired, 0=unpaired
    has_gng = 'Trials1' in m and np.size(m['Trials1']) > 0
    if has_gng and 'SampleP' in m and np.size(m['SampleP']) > 0:
        S = np.atleast_2d(m['Sample'])[:, 0].astype('int64')
        SP = np.atleast_2d(m['SampleP'])[:, 0].astype('int64')
        isP = np.isin(S, SP)
    else:                                            # single-DPA (all pure) or dual-only
        isP = np.ones(n, bool) if not has_gng else np.zeros(n, bool)
    tasks = np.where(isP, 'DPA', '').astype(object)
    odr = np.full(n, np.nan)
    if has_gng:
        Tr1 = np.atleast_2d(m['Trials1'])
        gout = Tr1[:, 2].astype(int)
        dual_idx = np.where(~isP)[0]
        k = min(len(dual_idx), len(gout))            # align chronological dual order
        dual_idx, gout = dual_idx[:k], gout[:k]
        tasks[dual_idx] = np.where(np.isin(gout, [1, 2]), 'DualGo', 'DualNoGo')
        odr[dual_idx] = np.isin(gout, [1, 4]).astype(float)
    return pd.DataFrame({'mouse': mouse, 'day': day, 'tasks': tasks,
                         'performance': perf, 'odr_perf': odr, 'pair': pair})


def load_batch(batch, group):
    folders = sorted(glob.glob(f'{DATA_ROOT}/{batch}/{group}_mouse_*'),
                     key=lambda p: int(p.rsplit('_', 1)[1]))
    rows = []
    for fol in folders:
        mouse = os.path.basename(fol)
        for f in glob.glob(f'{fol}/session_*.mat'):
            day = int(os.path.basename(f).split('_')[1].split('.')[0]) + 1   # 1-indexed
            try:
                rows.append(load_session(f, mouse, day))
            except Exception as e:
                print(f'  !! {mouse} {os.path.basename(f)}: {e}')
    if not rows:
        sys.exit(f'no sessions found for {batch}/{group}')
    return pd.concat(rows, ignore_index=True)


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ── --compare : single-DPA vs dual-DPA vs dual-GNG (DPA_vs_Single batch only) ──
if '--compare' in sys.argv[1:]:
    ds = load_batch(BATCH, 'DPA')       # single-DPA mice (performance = pure DPA)
    dd = load_batch(BATCH, 'Dual')      # dual mice (DPA judgment + GNG)
    days = list(range(1, int(max(ds.day.max(), dd.day.max())) + 1))

    def _curve(df, col):
        pm = df.groupby(['mouse', 'day'])[col].mean().reset_index()
        m, s = [], []
        for day in days:
            v = pm.loc[pm.day == day, col].dropna().values
            m.append(v.mean() if len(v) else np.nan)
            s.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else (0.0 if len(v) else np.nan))
        return np.array(days, float), np.array(m), np.array(s)

    C_SINGLE, C_DUALDPA, C_GNG = '#d62728', '#ff7f0e', '#1f77b4'
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for df, col, color, lab, ls in [
            (ds, 'performance', C_SINGLE,  'single-DPA (DPA task)', '-'),
            (dd, 'performance', C_DUALDPA, 'dual-DPA',              '-'),
            (dd, 'odr_perf',    C_GNG,     'dual-GNG',              '-')]:
        x, m, s = _curve(df, col)
        ok = ~np.isnan(m)
        ax.plot(x[ok], m[ok], ls=ls, color=color, lw=2, marker='o', ms=5, label=lab, zorder=3)
        ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)

    # LMM: single-DPA vs dual-DPA performance (between-mouse), perf ~ group*day + (1|mouse)
    a = ds.groupby(['mouse', 'day']).performance.mean().reset_index(); a['grp'] = 'single'
    b = dd.groupby(['mouse', 'day']).performance.mean().reset_index(); b['grp'] = 'dual'
    g = pd.concat([a, b], ignore_index=True).rename(columns={'performance': 'perf'})
    g['dayc'] = g['day'] - g['day'].mean()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm("perf ~ C(grp, Treatment('single'))*dayc", g, groups=g['mouse']).fit()
        wt = res.wald_test_terms(scalar=True).table
    gp = float(wt.loc[[i for i in wt.index if i.startswith('C(grp') and ':' not in i][0], 'pvalue'])
    ip = float(wt.loc[[i for i in wt.index if i.startswith('C(grp') and ':' in i][0], 'pvalue'])
    print(f'single vs dual DPA: group p={gp:.4f}{star(gp) or " ns"} | group×day p={ip:.4f}{star(ip) or " ns"}')
    ax.text(0.03, 0.03, f'single vs dual DPA  (LMM):\n group p={gp:.3f}{star(gp)}   group×day p={ip:.3f}{star(ip)}',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', lw=0.6, alpha=0.9))

    ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(range(2, len(days) + 1, 2))
    ax.set_xlabel('training day'); ax.set_ylabel('performance')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Single-DPA vs Dual — DPA & GNG performance', loc='left', fontweight='bold', fontsize=12)
    fig.suptitle(f'{BATCH}: single-DPA (n={ds.mouse.nunique()}) vs dual-trained (n={dd.mouse.nunique()})',
                 fontsize=11, y=0.99)
    fig.tight_layout()
    OUT = 'figures/overlaps/behavior/batch'
    os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/behavior_learning_batch_{SHORT.get(BATCH, BATCH)}_compare.{ext}'
        fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
    plt.close(fig)
    sys.exit(0)


# ── --delta : between-group silencing effect Δ(opto − control) per condition ───
if '--delta' in sys.argv[1:]:
    from scipy.stats import ttest_ind
    g1, g2 = ('control', 'opto') if 'Silencing' in BATCH else ('DPA', 'Dual')
    d1, d2 = load_batch(BATCH, g1), load_batch(BATCH, g2)
    days = list(range(1, int(max(d1.day.max(), d2.day.max())) + 1))
    xt = days if len(days) <= 10 else list(range(2, len(days) + 1, 2))

    def _dual(df):
        return df.tasks.isin(['DualGo', 'DualNoGo'])

    LINES = [
        ('A', 'DPA',  RED,   '-',  'performance', lambda df: df.tasks == 'DPA'),
        ('A', 'GNG',  BLUE,  '-',  'odr_perf',    _dual),
        ('B', 'Go',   BLUE,  '-',  'odr_perf',    lambda df: df.tasks == 'DualGo'),
        ('B', 'NoGo', GREEN, '-',  'odr_perf',    lambda df: df.tasks == 'DualNoGo'),
        ('C', 'paired',   RED, '-',  'performance', lambda df: (df.tasks == 'DPA') & (df.pair == 1)),
        ('C', 'unpaired', RED, '--', 'performance', lambda df: (df.tasks == 'DPA') & (df.pair == 0)),
        ('D', 'DPA only', RED,   '--', 'performance', lambda df: (df.pair == 0) & (df.tasks == 'DPA')),
        ('D', 'Go',       BLUE,  '--', 'performance', lambda df: (df.pair == 0) & (df.tasks == 'DualGo')),
        ('D', 'NoGo',     GREEN, '--', 'performance', lambda df: (df.pair == 0) & (df.tasks == 'DualNoGo')),
    ]

    def daily_vals(df, col, mask_fn):
        pm = df[mask_fn(df)].groupby(['mouse', 'day'])[col].mean().reset_index()
        return {day: pm.loc[pm.day == day, col].dropna().values for day in days}

    def grp_effect(col, mask_fn):
        a = d1[mask_fn(d1)].groupby(['mouse', 'day'])[col].mean().reset_index(); a['grp'] = g1
        b = d2[mask_fn(d2)].groupby(['mouse', 'day'])[col].mean().reset_index(); b['grp'] = g2
        g = pd.concat([a, b], ignore_index=True).rename(columns={col: 'perf'}).dropna(subset=['perf'])
        if g.grp.nunique() < 2 or g.mouse.nunique() < 4:
            return None
        g['dayc'] = g['day'] - g['day'].mean()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = smf.mixedlm(f"perf ~ C(grp, Treatment('{g1}'))*dayc", g, groups=g['mouse']).fit()
        except Exception:
            return None
        nm = [i for i in res.params.index if i.startswith('C(grp') and ':' not in i][0]
        ci = res.conf_int()
        return float(res.params[nm]), float(ci.loc[nm, 0]), float(ci.loc[nm, 1]), float(res.pvalues[nm])

    fig, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(22, 4.3))
    AX = {'A': axA, 'B': axB, 'C': axC, 'D': axD}
    TITLES = {'A': 'A  Δ DPA vs GNG', 'B': 'B  Δ Go vs NoGo',
              'C': 'C  Δ paired vs unpaired', 'D': 'D  Δ unpaired, by task'}
    effects, pcount = [], {}
    for panel, label, color, ls, col, mask_fn in LINES:
        ax = AX[panel]
        cv, ov = daily_vals(d1, col, mask_fn), daily_vals(d2, col, mask_fn)
        x, dm, ds = [], [], []
        for day in days:
            c, o = cv[day], ov[day]
            if len(c) and len(o):
                x.append(day); dm.append(o.mean() - c.mean())
                ds.append(np.sqrt((c.std(ddof=1) ** 2 / len(c) if len(c) > 1 else 0)
                                  + (o.std(ddof=1) ** 2 / len(o) if len(o) > 1 else 0)))
        if not x:
            continue
        x, dm, ds = np.array(x, float), np.array(dm), np.array(ds)
        ax.plot(x, dm, ls=ls, color=color, lw=2, marker='o', ms=5, label=label, zorder=3)
        ax.fill_between(x, dm - ds, dm + ds, color=color, alpha=0.18, lw=0, zorder=1)
        idx = pcount.get(panel, 0); pcount[panel] = idx + 1
        for day in days:                                 # per-day between-group stars
            c, o = cv[day], ov[day]
            if len(c) >= N_MIN and len(o) >= N_MIN and not (np.all(c == c[0]) and np.all(o == o[0])):
                pv = float(ttest_ind(o, c, equal_var=False).pvalue)
                if star(pv):
                    ax.text(day, 0.225 - idx * 0.02, star(pv), ha='center', va='center',
                            fontsize=9, fontweight='bold', color=color)
        eff = grp_effect(col, mask_fn)
        if eff is not None:
            effects.append((f'{panel} {label}', *eff))

    for panel, ax in AX.items():
        ax.axhline(0, ls='--', color='0.4', lw=1)
        ax.set_ylim(-0.24, 0.24)
        ax.set_xticks(xt); ax.set_xlabel('training day')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(TITLES[panel], loc='left', fontweight='bold', fontsize=11)
    axA.set_ylabel(f'Δ performance  ({g2} − {g1})')

    for i, (lab, b, lo, hi, pv) in enumerate(effects):          # panel E — per-condition effect
        cc = 'k' if pv < 0.05 else '0.6'
        axE.errorbar(i, b, yerr=[[b - lo], [hi - b]], fmt='o', color=cc, ms=6, capsize=3, lw=1.5, zorder=3)
        if star(pv):
            axE.text(i, hi + 0.004, star(pv), ha='center', va='bottom', fontsize=10, fontweight='bold')
    axE.axhline(0, ls='--', color='0.4', lw=1)
    axE.set_xticks(range(len(effects)))
    axE.set_xticklabels([lab for lab, *_ in effects], rotation=45, ha='right', fontsize=8)
    axE.set_xlim(-0.6, len(effects) - 0.4)
    axE.set_ylabel(f'mean Δ  ({g2} − {g1})')
    axE.set_title(f'E  Silencing effect per condition (95% CI)', loc='left', fontweight='bold', fontsize=11)
    axE.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Silencing effect Δ({g2}−{g1}) vs day — {BATCH} '
                 f'({g1} n={d1.mouse.nunique()}, {g2} n={d2.mouse.nunique()})', fontsize=13, y=0.99)
    fig.text(0.5, 0.005,
             f'Between-group Δ = {g2} − {g1} (different animals), mean ± SEM of the difference.  '
             'Top stars: per-day Welch t-test between groups (exploratory, uncorrected).  '
             'Panel E: LMM perf ~ group×day + (1|mouse), group effect at mean day.  * p<0.05  ** p<0.01  *** p<0.001',
             ha='center', va='bottom', fontsize=8, color='0.35')
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    OUT = 'figures/overlaps/behavior/batch'
    os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/behavior_learning_batch_{SHORT.get(BATCH, BATCH)}_delta.{ext}'
        fig.savefig(p, bbox_inches='tight'); print('saved', os.path.abspath(p))
    plt.close(fig)
    sys.exit(0)


d = load_batch(BATCH, GROUP)
MICE = sorted(d.mouse.unique(), key=lambda s: int(s.rsplit('_', 1)[1]))
DAYS = list(range(1, int(d.day.max()) + 1))
short = SHORT.get(BATCH, BATCH)
OUT_NAME = f'behavior_learning_batch_{short}_{GROUP}'
TITLE = f'Behavioural learning curves — {BATCH} · {GROUP} (n={len(MICE)} mice, {len(DAYS)} days)'
print(f'{OUT_NAME}: {len(d)} trials, {len(MICE)} mice, days {DAYS[0]}–{DAYS[-1]}')


# ── Shared helpers (mirrors fig_behavior_learning.py) ─────────────────────────
def per_mouse_day(col, mask):
    sel = d[mask]
    out = {day: {} for day in DAYS}
    for day in DAYS:
        for m in MICE:
            v = sel[(sel.mouse == m) & (sel.day == day)][col].dropna()
            if len(v):
                out[day][m] = float(v.mean())
    return out


def agg(pmd):
    mean, sem, n = [], [], []
    for day in DAYS:
        pm = np.array(list(pmd[day].values()))
        if len(pm):
            mean.append(pm.mean())
            sem.append(pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0.0)
            n.append(len(pm))
        else:
            mean.append(np.nan); sem.append(np.nan); n.append(0)
    return np.array(DAYS, float), np.array(mean), np.array(sem), np.array(n)


def plot_line(ax, pmd, color, label, ls='-'):
    x, m, s, n = agg(pmd)
    ok = ~np.isnan(m)
    if not ok.any():
        return False                                 # nothing to draw (missing condition)
    ax.plot(x[ok], m[ok], ls=ls, color=color, lw=2, marker='o', ms=5,
            mfc=color, mec=color, label=label, zorder=3)
    ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0, zorder=1)
    return float(np.nanmin((m - s)[ok])), float(np.nanmax((m + s)[ok]))


def _prop(mask, cond, correct, by_day=True):
    df = pd.DataFrame({
        'correct': np.asarray(correct)[mask.values],
        'cond':    np.asarray(cond)[mask.values],
        'mouse':   d.loc[mask, 'mouse'].values,
        'day':     d.loc[mask, 'day'].values,
    }).dropna()
    keys = ['mouse', 'day', 'cond'] if by_day else ['mouse', 'cond']
    return df.groupby(keys, observed=True).correct.mean().reset_index(name='perf')


def lmm(mask, cond, correct, ref, tag):
    """Trajectory LMM perf ~ condition*day_centred + (1|mouse). Returns β records
       for panel E; skips gracefully if <2 conditions have data."""
    g = _prop(mask, cond, correct, by_day=True)
    if g.cond.nunique() < 2 or ref not in set(g.cond):
        return []
    g['dayc'] = g['day'] - g['day'].mean()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))*dayc",
                              g, groups=g['mouse']).fit()
            wt = res.wald_test_terms(scalar=True).table
    except Exception as e:
        print(f'  [{tag}] LMM skipped ({e})')
        return []
    ct = [i for i in wt.index if i.startswith('C(cond') and ':' not in i][0]
    it = [i for i in wt.index if i.startswith('C(cond') and ':' in i][0]
    cp, cdf, ip = float(wt.loc[ct, 'pvalue']), int(wt.loc[ct, 'df_constraint']), float(wt.loc[it, 'pvalue'])
    print(f'  [{tag}] LMM  condition p={cp:.4f}{f" (df{cdf})" if cdf > 1 else ""} '
          f'{star(cp) or "ns"} | cond×day p={ip:.4f} {star(ip) or "ns"}')
    ci, pv, pr = res.conf_int(), res.pvalues, res.params
    recs = []
    for name in pr.index:
        if not name.startswith('C(cond'):
            continue
        lvl = name.split('[T.')[1].split(']')[0]
        recs.append(dict(model=tag.split()[0], contrast=f'{lvl}−{ref}',
                         kind='cond×day' if ':dayc' in name else 'condition',
                         beta=float(pr[name]), lo=float(ci.loc[name, 0]),
                         hi=float(ci.loc[name, 1]), p=float(pv[name])))
    return recs


def perday_stars(ax, mask, cond, correct, ref, tag, y_star=1.03):
    g_all = _prop(mask, cond, correct, by_day=True)
    if g_all.cond.nunique() < 2 or ref not in set(g_all.cond):
        return
    ps = {}
    for day in DAYS:
        sub = mask & (d.day == day)
        if d.loc[sub, 'mouse'].nunique() < N_MIN:
            continue
        g = _prop(sub, cond, correct, by_day=False)
        if g.cond.nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = smf.mixedlm(f"perf ~ C(cond, Treatment('{ref}'))", g, groups=g['mouse']).fit()
                wt = r.wald_test_terms(scalar=True).table
            ps[day] = float(wt.loc[[i for i in wt.index if i.startswith('C(cond')][0], 'pvalue'])
        except Exception:
            pass
    if not ps:
        return
    print(f'  [{tag}] per-day LMM (uncorrected): '
          + '  '.join(f'd{k} p={ps[k]:.3f}{star(ps[k]) or "ns"}' for k in sorted(ps)))
    for k in sorted(ps):
        if star(ps[k]):
            ax.text(k, y_star, star(ps[k]), ha='center', va='top', fontsize=11,
                    fontweight='bold', color='k')


IS_DPA  = d.tasks == 'DPA'
IS_GO   = d.tasks == 'DualGo'
IS_NOGO = d.tasks == 'DualNoGo'
IS_DUAL = IS_GO | IS_NOGO
UNP     = d.pair == 0
CORR_DPA = d.performance
CORR_GNG = d.odr_perf
CORR_A   = np.where(IS_DPA, d.performance, d.odr_perf)
COND_A   = np.where(IS_DPA, 'DPA', 'GNG')
COND_PAIR = np.where(d.pair == 1, 'paired', 'unpaired')
COND_TASK = d.tasks.map({'DPA': 'DPA', 'DualGo': 'Go', 'DualNoGo': 'NoGo'})

fig, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(22, 4.3))
betas = []
xticks = DAYS if len(DAYS) <= 10 else list(range(2, len(DAYS) + 1, 2))


def build_panel(ax, title, lines, stat_args):
    """Draw a panel's lines, fit its LMM, set data-adaptive y-limits, then stars."""
    ranges = [r for pmd, color, label, ls in lines
              if (r := plot_line(ax, pmd, color, label, ls=ls))]
    betas.extend(lmm(*stat_args))
    lo = min(r[0] for r in ranges) if ranges else 0.4
    hi = max(r[1] for r in ranges) if ranges else 1.0
    ylo, yhi = max(0.0, lo - 0.05), min(1.06, hi + 0.06)
    ax.set_ylim(ylo, yhi)
    if ylo < 0.5 < yhi:
        ax.axhline(0.5, ls=':', color='0.5', lw=1)
    ax.set_xticks(xticks)
    ax.set_xlabel('training day')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    perday_stars(ax, *stat_args, y_star=yhi - 0.02 * (yhi - ylo))


build_panel(axA, 'A  DPA vs GNG performance',
            [(per_mouse_day('performance', IS_DPA),  RED,  'DPA',  '-'),
             (per_mouse_day('odr_perf',    IS_DUAL), BLUE, 'GNG',  '-')],
            (IS_DPA | IS_DUAL, COND_A, CORR_A, 'DPA', 'A DPA vs GNG'))
build_panel(axB, 'B  GNG: Go vs NoGo',
            [(per_mouse_day('odr_perf', IS_GO),   BLUE,  'Go',   '-'),
             (per_mouse_day('odr_perf', IS_NOGO), GREEN, 'NoGo', '-')],
            (IS_DUAL, COND_TASK, CORR_GNG, 'Go', 'B Go vs NoGo'))
build_panel(axC, 'C  DPA: paired vs unpaired',
            [(per_mouse_day('performance', IS_DPA & (d.pair == 1)), RED, 'paired',   '-'),
             (per_mouse_day('performance', IS_DPA & UNP),           RED, 'unpaired', '--')],
            (IS_DPA, COND_PAIR, CORR_DPA, 'paired', 'C paired vs unpaired'))
build_panel(axD, 'D  DPA unpaired, by task',
            [(per_mouse_day('performance', UNP & IS_DPA),  RED,   'DPA only', '--'),
             (per_mouse_day('performance', UNP & IS_GO),   BLUE,  'Go',       '--'),
             (per_mouse_day('performance', UNP & IS_NOGO), GREEN, 'NoGo',     '--')],
            (UNP, COND_TASK, CORR_DPA, 'DPA', 'D unpaired by task'))
axA.set_ylabel('performance')

# ── Panel E — LMM coefficients (β on y) ───────────────────────────────────────
cond_recs = [r for r in betas if r['kind'] == 'condition']
int_map = {(r['model'], r['contrast']): r for r in betas if r['kind'] == 'cond×day'}
xlabels = []
for i, c in enumerate(cond_recs):
    it = int_map[(c['model'], c['contrast'])]
    cc = 'k'  if c['p'] < 0.05 else '0.65'
    ci = BLUE if it['p'] < 0.05 else '0.65'
    axE.errorbar(i - 0.16, c['beta'], yerr=[[c['beta'] - c['lo']], [c['hi'] - c['beta']]],
                 fmt='o', color=cc, ms=7, capsize=3, lw=1.6, zorder=3)
    axE.errorbar(i + 0.16, it['beta'], yerr=[[it['beta'] - it['lo']], [it['hi'] - it['beta']]],
                 fmt='s', mfc='white', mec=ci, color=ci, ms=6, capsize=3, lw=1.4, zorder=3)
    for r, dx in ((c, -0.16), (it, 0.16)):
        if star(r['p']):
            axE.text(i + dx, r['hi'] + 0.02, star(r['p']), ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    xlabels.append((i, f"{c['model']}  {c['contrast']}"))
axE.axhline(0, ls='--', color='0.4', lw=1)
axE.set_xticks([x for x, _ in xlabels])
axE.set_xticklabels([lab for _, lab in xlabels], rotation=40, ha='right', fontsize=8.5)
axE.set_xlim(-0.6, max(len(cond_recs) - 0.4, 0.6))
axE.set_ylabel('LMM coefficient  β  (Δ performance)')
axE.set_title('E  LMM fixed-effect coefficients (95% CI)', loc='left', fontweight='bold', fontsize=11)
axE.spines[['top', 'right']].set_visible(False)
axE.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', ls='none', ms=7, label='condition (β at mean day)'),
                    mlines.Line2D([0], [0], marker='s', color=BLUE, mfc='white', ls='none', ms=6, label='condition × day (slope)')],
           frameon=False, fontsize=8, loc='upper right')

fig.suptitle(TITLE, fontsize=13, y=0.99)
fig.text(0.5, 0.005,
         f'Curves: mean ± SEM across mice ({len(MICE)}).  '
         'Top stars: per-day LMM condition effect, random intercept mouse, UNCORRECTED (exploratory; days with <4 mice untested).  '
         'Panel E: trajectory LMM, perf ~ condition×day + (1|mouse).  * p<0.05  ** p<0.01  *** p<0.001',
         ha='center', va='bottom', fontsize=8, color='0.35')
fig.tight_layout(rect=(0, 0.05, 1, 0.94))

OUT = 'figures/overlaps/behavior/batch'
os.makedirs(f'{OUT}/png', exist_ok=True)
os.makedirs(f'{OUT}/svg', exist_ok=True)
for ext in ('png', 'svg'):
    p = f'{OUT}/{ext}/{OUT_NAME}.{ext}'
    fig.savefig(p, bbox_inches='tight')
    print('saved', os.path.abspath(p))
plt.close(fig)
