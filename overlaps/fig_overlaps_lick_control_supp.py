"""SUPPLEMENT — anticipatory-lick / movement control (Musall/Stringer). Is the no-lick push / depth↔behaviour
coupling explained by anticipatory DELAY LICKING?

Per-trial LATE-DELAY lick rate (behavior/licks.org: data['lickTime'] vs data['Sample'] onsets) aligned to
the RAW neural trials in acquisition order (100% task-label validated). WINDOWS MATCH THE MAIN FIGURE:
depth read at options['bins_LD'] (=[48..53]) on DPA trials; lick window = the licks.org LD window
[7.0,7.5]s post sample onset (aligned to bins_LD once the odor-onset vs GCaMP-response offset is
accounted for). depth = per-stage choice(lick) axis @57-63 from raw activity, pooled-evoked norm.
(An earlier version used a too-broad [4.5,9.0]s window that leaked test/response licking into the
"delay" measure and spuriously showed the push attenuating; with the aligned window it does not.)

Four panels:
  A  trial-level depth vs late-delay lick rate — depth does NOT track delay licking (ρ≈+0.07); the little
     licking that happens (11% of trials) is unrelated to the readout
  B  per-mouse Naive→Expert Δ(late-delay lick rate) — modest decrease with learning
  C  PUSH deepening LMM with vs without the lick covariate — UNCHANGED (β−0.74→−0.75, both sig): the push is
     not a delay-licking artifact
  D  COUPLING Δdepth↔ΔDPA-acc — partial|Δlick unchanged (r=−0.87), Δlick↔Δacc null → licking is NOT the mechanism
Output: figures/overlaps/controls/{png,svg}/overlaps_lick_control.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, glob
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib.pyplot as plt, seaborn as sns
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from src.pca.io import pkl_load
from src.common.options import set_options

sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({          # shared house style (matches fig_overlaps_main_native.py; see CLAUDE.md)
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
PATH = '/storage/leon/dual_task/data/2Samples-DualTask-BehavioralData'
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
ACT = np.arange(57, 63); LD = np.asarray(set_options()['bins_LD']); BL = np.arange(0, 12); DELAY = (7.0, 7.5)
SAMPLES = [('A', [0, 1]), ('B', [2, 3])]
_pal = sns.color_palette('tab10', n_colors=len(MICE)); MC = {m: _pal[i] for i, m in enumerate(MICE)}


def find_mat(folder):
    m = glob.glob(os.path.join(folder, '*.mat')); return m[0] if m else None


# ── behavioural per-trial delay lick rate ──
beh = []
for mouse in MICE:
    opt = set_options(mouse=mouse)
    for day in range(1, opt['n_days'] + 1):
        f = find_mat(f'{PATH}/{mouse}-DualTask-BehavioralData/day_{day}/')
        if f is None:
            continue
        try:
            d = loadmat(f); onsets = d['Sample'][:, 0] / 1e3; licks = d['lickTime'][:, 0] / 1e3
            tl = [licks[(licks >= onsets[i]) & (licks < onsets[i + 1])] - onsets[i] for i in range(len(onsets) - 1)]
            tl.append(licks[licks >= onsets[-1]] - onsets[-1])
            rate = np.array([np.sum((r >= DELAY[0]) & (r < DELAY[1])) / (DELAY[1] - DELAY[0]) for r in tl])
            tr = d['AllTrials'][0][0][-1]
            df = pd.DataFrame(tr, columns=['sample', 'test', 'outcome', 'pair', 'distractor', 'cue', 'odr_outcome', 'odr_pair', 'laser'])
            df['tasks'] = df['distractor'].map({0: 'DPA', 1: 'DualGo', 2: 'DualNoGo'})
            df['lick_delay'] = rate; df['day'] = day; df['mouse'] = mouse
            beh.append(df)
        except Exception:
            pass
beh = pd.concat(beh, ignore_index=True)

# ── raw neural depth (per-stage choice axis) ──
Wb = pkl_load('weights_log_generalizing_overlaps_none_l1_ratio_0.0_raw_targets_choice-gng-sample-test', path='../data/overlaps')
W, VALID = Wb['weights'], Wb['valid']
print('loading X_all …', flush=True)
Xall = np.asarray(pkl_load('X_all_nan_', path='../data/pca')); yall = pkl_load('y_all_nan_', path='../data/pca').reset_index(drop=True)
depth = np.full(len(yall), np.nan)
for m in MICE:
    both = VALID[(m, 'Naive')] & VALID[(m, 'Expert')]
    axis = {st: np.asarray(W[(m, st, 'all', 'choice')])[ACT].mean(0)[VALID[(m, 'Naive')][VALID[(m, st)]]] for st in ('Naive', 'Expert')}
    P, sel = {}, {}
    for st in ('Naive', 'Expert'):
        s = ((yall.mouse == m) & (yall.tasks == 'DPA') & (yall.laser == 0) & (yall.learning == st)).to_numpy()
        P[st] = np.nansum(Xall[s][:, both, :] * axis[st][None, :, None], axis=1); sel[st] = s
    Pall = np.vstack([P['Naive'], P['Expert']]); chall = np.concatenate([yall.loc[sel['Naive'], 'choice'], yall.loc[sel['Expert'], 'choice']]).astype(float)
    sgn = np.where(chall == 1, 1.0, -1.0); vbar = (sgn[:, None] * Pall).mean(0)
    sd = (vbar - vbar[BL].mean()).std() + 1e-9; mu = Pall[:, BL].mean()
    for st in ('Naive', 'Expert'):
        depth[sel[st]] = (P[st][:, LD].mean(1) - mu) / sd
yall['depth'] = depth

# ── align lick to raw neural (acquisition order), validated ──
yall['lick_delay'] = np.nan; agree = tot = 0
for m in MICE:
    for day in sorted(yall[yall.mouse == m].day.unique()):
        yi = yall[(yall.mouse == m) & (yall.day == day) & (yall.laser == 0)].index
        bb = beh[(beh.mouse == m) & (beh.day == day) & (beh.laser == 0)]
        if len(yi) != len(bb) or len(bb) == 0:
            continue
        agree += int((yall.loc[yi, 'tasks'].to_numpy() == bb['tasks'].to_numpy()).sum()); tot += len(yi)
        yall.loc[yi, 'lick_delay'] = bb['lick_delay'].to_numpy()
print(f'alignment tasks agree {100*agree/tot:.1f}%')
d = yall[(yall.tasks == 'DPA') & (yall.laser == 0) & yall.depth.notna() & yall.lick_delay.notna()].copy()
d['sample'] = np.where(d.odor_pair.isin([0, 1]), 'A', 'B'); d['st'] = (d.learning == 'Expert').astype(int)

# ── stats ──
g = d.groupby(['mouse', 'sample', 'learning']).agg(depth=('depth', 'mean'), lick=('lick_delay', 'mean')).reset_index()
g['st'] = (g.learning == 'Expert').astype(int)
m0 = smf.mixedlm('depth ~ st + C(sample)', g, groups=g['mouse']).fit()
m1 = smf.mixedlm('depth ~ st + C(sample) + lick', g, groups=g['mouse']).fit()
lr = d.groupby(['mouse', 'learning']).lick_delay.mean().unstack()
gd = g.groupby('mouse').apply(lambda x: pd.Series(dict(dd=x[x.st == 1].depth.mean() - x[x.st == 0].depth.mean(),
                                                        dl=x[x.st == 1].lick.mean() - x[x.st == 0].lick.mean())))
acc = d.groupby(['mouse', 'learning']).performance.mean().unstack()
gd['da'] = acc['Expert'] - acc['Naive']
gd = gd.dropna()
r0, p0 = spearmanr(gd.dd, gd.da)
def rank_resid(u, z):
    ur, zr = pd.Series(u).rank().to_numpy(), pd.Series(z).rank().to_numpy()
    return ur - np.polyval(np.polyfit(zr, ur, 1), zr)
rp, pp = pearsonr(rank_resid(gd.dd, gd.dl), rank_resid(gd.da, gd.dl))
rl, pl = spearmanr(gd.dl, gd.da)

# ── figure ──
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
for (r_, c_), L in {(0, 0): 'A', (0, 1): 'B', (1, 0): 'C', (1, 1): 'D'}.items():
    ax[r_, c_].text(-0.16, 1.04, L, transform=ax[r_, c_].transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')
# A: trial-level depth vs lick
a = ax[0, 0]
for st, c in [('Naive', '#4477AA'), ('Expert', '#CC3311')]:
    ds = d[d.learning == st]; a.scatter(ds.lick_delay, ds.depth, s=5, color=c, alpha=0.12, lw=0)
rr, prr = spearmanr(d.depth, d.lick_delay)
a.set_xlabel('late-delay lick rate [7.0–7.5 s] (Hz)'); a.set_ylabel('choice-code depth\n← no lick    lick →')
a.set_title('depth does NOT track late-delay licking', loc='left', fontsize=TITLE_FS)
a.text(0.96, 0.06, f'trial-level ρ={rr:+.2f}\n(p={prr:.3f}, n={len(d)})', transform=a.transAxes, ha='right', va='bottom', fontsize=6)
a.axhline(0, ls=':', color='0.6', lw=0.7)
h = [plt.Line2D([], [], marker='o', ls='none', color=c, ms=4, label=s) for s, c in [('Naive', '#4477AA'), ('Expert', '#CC3311')]]
a.legend(handles=h, frameon=False, loc='upper left')
# B: per-mouse Δ lick
b = ax[0, 1]
for m in MICE:
    if m in lr.index and np.isfinite(lr.loc[m]).all():
        b.plot([0, 1], [lr.loc[m, 'Naive'], lr.loc[m, 'Expert']], '-o', color=MC[m], lw=0.8, ms=4, mec='w', mew=0.4)
b.set_xticks([0, 1]); b.set_xticklabels(['Naive', 'Expert']); b.set_xlim(-0.4, 1.4)
b.set_ylabel('late-delay lick rate (Hz)')
ndec = int((lr['Expert'] < lr['Naive']).sum())
b.set_title('late-delay licking falls modestly with learning', loc='left', fontsize=TITLE_FS)
b.text(0.5, 0.96, f'{ndec}/9 mice lick less (mean Δ={(lr["Expert"]-lr["Naive"]).mean():+.2f} Hz)', transform=b.transAxes, ha='center', va='top', fontsize=6)
# C: push with/without covariate
c = ax[1, 0]
vals = [('no covariate', m0.params['st'], m0.bse['st'], m0.pvalues['st']),
        ('+ lick covariate', m1.params['st'], m1.bse['st'], m1.pvalues['st'])]
for i, (lab, bta, se, p) in enumerate(vals):
    col = '#CC3311' if p < 0.05 else '0.6'
    c.errorbar(i, bta, se, fmt='o', color=col, ms=6, capsize=4, lw=1.2)
    c.text(i + 0.14, bta, f'β={bta:+.2f}\np={p:.3f}', ha='left', va='center', fontsize=6.5, color=col)
c.axhline(0, ls=':', color='0.6', lw=0.8); c.set_xticks([0, 1]); c.set_xticklabels([v[0] for v in vals]); c.set_xlim(-0.5, 1.7)
c.set_ylabel('push: stage β (depth ~ stage)')
c.set_title('push is robust to late-delay licking', loc='left', fontsize=TITLE_FS)
# D: coupling scatter + partial
e = ax[1, 1]
for m in gd.index:
    e.scatter(gd.loc[m, 'dd'], gd.loc[m, 'da'], color=MC[m], s=32, zorder=4)
z = np.polyfit(gd.dd, gd.da, 1); xx = np.array([gd.dd.min(), gd.dd.max()]); e.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.3)
e.axhline(0, ls=':', color='k', lw=0.7); e.axvline(0, ls=':', color='k', lw=0.7)
e.set_xlabel('Δ choice-code depth (Exp−Naive)'); e.set_ylabel('Δ DPA accuracy (Exp−Naive)')
e.set_title('coupling is not explained by licking', loc='left', fontsize=TITLE_FS)
e.text(0.04, 0.04, f'Δdepth↔Δacc ρ={r0:+.2f} p={p0:.3f}\npartial | Δlick  r={rp:+.2f} p={pp:.3f}\nΔlick↔Δacc ρ={rl:+.2f} p={pl:.2f} (null)',
       transform=e.transAxes, ha='left', va='bottom', fontsize=6)
fig.suptitle('Late-delay-aligned lick control: neither the push nor the coupling is a delay-licking artifact',
             fontsize=9, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT = 'figures/overlaps/controls'
for s in ('png', 'svg'):
    os.makedirs(f'{OUT}/{s}', exist_ok=True)
p = f'{OUT}/png/overlaps_lick_control.png'
fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('/png/', '/svg/').replace('.png', '.svg'), bbox_inches='tight')
print('saved', p)
