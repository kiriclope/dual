"""exp_xmouse_plane.py — is the sample x choice plane a CONSERVED, cross-animal solution?
(idea #2, 2026-09-01 analysis menu; cf. Safaie et al. 2023 preserved dynamics across animals.)

Each mouse's plane is defined functionally (coord 1 = its own sample axis, coord 2 = its own
choice axis; decoders.fit_axis on half 1, QR-orthonormalised with signs fixed to the axes), so
every animal's trials land in one COMMON 2-D reference frame. If the geometry is conserved,
a decoder trained on OTHER mice's plane coordinates should read a held-out mouse's memory and
choice near its own within-mouse ceiling.

Per (mouse, stage), NREP half-splits: axes fit on half 1; held-out half-2 trials projected to
2-D coords; coords standardised per mouse by its half-1 coordinate stats. Within = LR trained
on the mouse's own half-1 coords. Cross = LR trained on the pooled half-1 coords of the OTHER
8 mice, tested on this mouse's half-2 coords (leave-one-mouse-out). Balanced accuracy.
Variables: sample @ md (correct, all tasks) - choice @ decision (DPA, behavioural lick).

Merge-dumps {'XMOUSE_PLANE'+SUF} into results.pkl; preview PNG.
Run:  cd /home/leon/dual/pca && /home/leon/mambaforge/envs/dual/bin/python exp_xmouse_plane.py --nopca
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
from decoders import fit_axis, SUF

MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
STAGES = ['Naive', 'Expert']
NREP = 10

_c = pickle.load(open('figures/pseudo/dimensionality/fits_inputs.pkl', 'rb'))
AW = _c['AW']; VALIDIX = _c['VALIDIX']
MOUSE, LEARN, LAS, TSK, SAMP, TESTO, PERF = (_c['L'][k] for k in
                                             ['MOUSE', 'LEARN', 'LAS', 'TSK', 'SAMP', 'TESTO', 'PERF'])
MATCH = (SAMP == TESTO)
LICK = np.where(PERF == 1, MATCH, ~MATCH)


def sel(mouse, stage, **kw):
    m = (MOUSE == mouse) & (LEARN == stage) & (LAS == 0)
    for k, v in kw.items():
        arr = {'task': TSK, 'samp': SAMP, 'perf': PERF, 'lick': LICK}[k]
        m &= (arr == v)
    return np.where(m)[0]


def zscale(M, val, idx):
    sd = np.nanstd(M[np.ix_(idx, val)], axis=0)
    return np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def halves(rng, idx):
    p = rng.permutation(idx); h = len(p) // 2
    return p[:h], p[h:]


def bacc(pred, yv):
    return float(np.mean([np.mean(pred[yv == c] == c) for c in np.unique(yv)]))


# ── phase 1: per (mouse, stage, rep) — axes on half 1, coords for both halves ──
COORDS = {}                              # (mouse, stage, rep, var) -> (Ctr, ytr, Cte, yte)
rng = np.random.RandomState(1234)
for stage in STAGES:
    for mo in MICE:
        val = VALIDIX[(mo, stage)]
        if not len(val):
            continue
        allc = sel(mo, stage, perf=1)
        Mmd, Mdc = AW['md'], AW['decision']
        sdm, sdd = zscale(Mmd, val, allc), zscale(Mdc, val, allc)
        for rep in range(NREP):
            sP1, sP2 = halves(rng, sel(mo, stage, perf=1, samp=1))
            sN1, sN2 = halves(rng, sel(mo, stage, perf=1, samp=0))
            lP1, lP2 = halves(rng, sel(mo, stage, task='DPA', lick=True))
            lN1, lN2 = halves(rng, sel(mo, stage, task='DPA', lick=False))
            if min(len(sP1), len(sN1), len(lP1), len(lN1)) < 3:
                continue
            Xs = np.nan_to_num(np.vstack([Mmd[np.ix_(sP1, val)] / sdm, Mmd[np.ix_(sN1, val)] / sdm]))
            w_s, _ = fit_axis(Xs, np.r_[np.ones(len(sP1), int), np.zeros(len(sN1), int)])
            Xl = np.nan_to_num(np.vstack([Mdc[np.ix_(lP1, val)] / sdd, Mdc[np.ix_(lN1, val)] / sdd]))
            w_l, _ = fit_axis(Xl, np.r_[np.ones(len(lP1), int), np.zeros(len(lN1), int)])
            Q = np.linalg.qr(np.stack([w_s, w_l], 1))[0]
            Q[:, 0] *= np.sign(Q[:, 0] @ w_s)                 # common orientation across mice
            Q[:, 1] *= np.sign(Q[:, 1] @ w_l)
            for var, M, sd_, (p1, p2), (n1, n2) in [
                    ('sample', Mmd, sdm, (sP1, sP2), (sN1, sN2)),
                    ('choice', Mdc, sdd, (lP1, lP2), (lN1, lN2))]:
                Ctr = np.nan_to_num(np.vstack([M[np.ix_(p1, val)] / sd_,
                                               M[np.ix_(n1, val)] / sd_])) @ Q
                Cte = np.nan_to_num(np.vstack([M[np.ix_(p2, val)] / sd_,
                                               M[np.ix_(n2, val)] / sd_])) @ Q
                mu, sg = Ctr.mean(0), Ctr.std(0) + 1e-9       # standardise by the TRAIN half
                COORDS[(mo, stage, rep, var)] = ((Ctr - mu) / sg,
                                                 np.r_[np.ones(len(p1), int), np.zeros(len(n1), int)],
                                                 (Cte - mu) / sg,
                                                 np.r_[np.ones(len(p2), int), np.zeros(len(n2), int)])

# ── phase 2: within-mouse ceiling vs leave-one-mouse-out cross decoding ────────
XM = {}
print(f'══ XMOUSE_PLANE (SUF="{SUF}") ══', flush=True)
for stage in STAGES:
    for var in ['sample', 'choice']:
        for mo in MICE:
            wi, xa = [], []
            for rep in range(NREP):
                if (mo, stage, rep, var) not in COORDS:
                    continue
                Ctr, ytr, Cte, yte = COORDS[(mo, stage, rep, var)]
                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                wi.append(bacc(clf.fit(Ctr, ytr).predict(Cte), yte))
                oc = [COORDS[(o, stage, rep, var)] for o in MICE
                      if o != mo and (o, stage, rep, var) in COORDS]
                if len(oc) >= 5:
                    Xo = np.vstack([c[0] for c in oc]); yo = np.concatenate([c[1] for c in oc])
                    clf2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                    xa.append(bacc(clf2.fit(Xo, yo).predict(Cte), yte))
            if wi and xa:
                XM[(mo, stage, var)] = (float(np.mean(wi)), float(np.mean(xa)))
        vals = [XM[(m, stage, var)] for m in MICE if (m, stage, var) in XM]
        w = np.array([v[0] for v in vals]); x = np.array([v[1] for v in vals])
        rat = (x - 0.5).sum() / max((w - 0.5).sum(), 1e-9)
        try:
            p = wilcoxon(w, x).pvalue
        except Exception:
            p = np.nan
        print(f'{stage:6s} {var:6s}: within {w.mean():.3f}  cross-mouse {x.mean():.3f}  '
              f'chance-ref ratio {rat:.2f}  (within-vs-cross p={p:.3f}, n={len(w)})', flush=True)

RES = 'figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['XMOUSE_PLANE' + SUF] = XM
pickle.dump(d, open(RES, 'wb'))
print('merged XMOUSE_PLANE' + SUF)

# ── preview figure ─────────────────────────────────────────────────────────────
import seaborn as sns, matplotlib.pyplot as plt
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
})
TITLE_FS = 8
_pal = sns.color_palette('tab10', n_colors=len(MICE))
MC = {m: _pal[i] for i, m in enumerate(MICE)}
fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True, sharey=True)
for ax, var in zip(axs, ['sample', 'choice']):
    for st, mk in [('Naive', 'o'), ('Expert', 's')]:
        for m in MICE:
            if (m, st, var) in XM:
                wv, xv = XM[(m, st, var)]
                ax.scatter(wv, xv, s=34, marker=mk, color=MC[m], edgecolors='w',
                           linewidths=0.6, zorder=3)
    ax.plot([0.45, 1.0], [0.45, 1.0], ls='--', color='0.5', lw=1, zorder=1)
    ax.axhline(0.5, ls=':', color='0.6', lw=0.8); ax.axvline(0.5, ls=':', color='0.6', lw=0.8)
    ax.set_xlabel('within-mouse accuracy'); ax.set_title(var, loc='left', fontsize=TITLE_FS)
axs[0].set_ylabel('cross-mouse accuracy\n(trained on the other 8 mice)')
import matplotlib.lines as mlines
axs[1].legend(handles=[mlines.Line2D([0], [0], marker='o', color='0.4', ls='none', ms=5, label='Naive'),
                       mlines.Line2D([0], [0], marker='s', color='0.4', ls='none', ms=5, label='Expert')],
              frameon=False, fontsize=6.5, loc='lower right')
fig.suptitle('The sample × choice plane is a conserved, cross-animal frame', x=0.02, ha='left',
             fontsize=TITLE_FS)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUT = 'figures/pseudo/dimensionality'
fig.savefig(f'{OUT}/png/exp_xmouse_plane{SUF}.png', bbox_inches='tight')
fig.savefig(f'{OUT}/svg/exp_xmouse_plane{SUF}.svg', bbox_inches='tight')
print('saved', os.path.abspath(f'{OUT}/png/exp_xmouse_plane{SUF}.png'))
