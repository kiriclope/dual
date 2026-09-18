"""(RENUMBERED 2026-09-18 to citation order, ten figures after the unsupervised-geometry page was inserted as Extended Data Fig. 4: this script draws Extended Data Fig. 7.)
fig_ed3_coupling.py — Extended Data Fig. 7: the learning coupling and the push under other units, a fixed
axis, other decoders, and a lick covariate (companion to Fig. 4b,c). Built 2026-09-15 (Leon: "keep only
what is essential") — the four controls Results §4 cites: normalizations and the fixed common axis
(ED 5a,b in the old numbering), the decoder variants (old 6c), and the lick control (old 5d).

  a  the push (within-mouse LMM β) and the coupling (per-mouse Spearman ρ, n = 9) under six units of the
     same depth; raw is the unit of Fig. 4
  b  the coupling under the ridge (Fig. 4c), L1 and shrinkage-LDA decoders
  c  REINSTATED on CCGD held-out decision functions (exp_common_axis_ccgd.py): the push and the coupling on the
     per-stage axes (Fig. 4's tensor) and on one axis fitted to both stages pooled (run_overlaps --pool-stages)
  d  REINSTATED on the CCGD depth (exp_lick_control_ccgd.py): late-delay licking per mouse x stage against the
     depth; the push and the coupling with a per-mouse lick covariate

HISTORY 2026-09-15 (Leon: "make sure all the results are cross validated"): the fixed-common-axis panel and
the lick-covariate panel inherited from the old supplement projected TRAINING trials on the fold-averaged CCGD
weights (fig_overlaps_common_axis_supp.py / fig_overlaps_lick_control_supp.py). A held-out re-fit of the axis
from scratch (5-fold, the CCGD regularisation) is a different and unstable estimator (one mouse's evoked-s.d.
normalisation explodes; per-stage coupling rho -0.57 p .11 where the tensor gives -0.80 p .010), so neither
panel is shown. Reinstating them needs the CCGD pipeline itself: a pooled-stage run_overlaps build for the
common axis, and the tensor depth aligned to per-trial licks for the covariate. Panels a and b read the
cross-validated CCGD tensor (main_panels / the *_supp norm script), like Fig. 4.

Reads caches only: figures/overlaps/controls/ed3_cache.pkl (written by fig_overlaps_norm_robustness_supp.py,
fig_overlaps_common_axis_supp.py, fig_overlaps_lick_control_supp.py) and coupling_variants_cache.pkl
(exp_coupling_variants.py).
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python fig_ed3_coupling.py [--nocap]
Output: /home/leon/dual/figures/ed/{png,svg}/ed_fig7.{png,svg}
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import spearmanr
from figcaption import draw_justified

sns.set_context('notebook'); sns.set_style('ticks')
PS = 1.2      # 10-in canvas -> 183 mm is x0.72: 1.2 keeps every literal (5.5-8 pt) at >= 5 pt in print (review 2026-09-15)
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
MICE = ['JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18', 'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04']
MC = dict(zip(MICE, sns.color_palette('tab10', n_colors=len(MICE))))
SIGC, NSC = '#CC3311', '0.6'
C = pickle.load(open('figures/overlaps/controls/ed3_cache.pkl', 'rb'))
CV = pickle.load(open('figures/overlaps/controls/coupling_variants_cache.pkl', 'rb'))


def plabel(ax, s, dx=-0.10):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=PS*11, fontweight='bold', va='bottom', ha='right')


def verdict(ax, p, x=0.97, y=0.96):
    sig = p < 0.05
    ax.text(x, y, '∗' if sig else 'n.s.', transform=ax.transAxes, ha='right', va='top',
            fontsize=PS*(12 if sig else 8), fontweight='bold', color='k' if sig else '0.55')


# ══ a — six units of the same depth ══════════════════════════════════════════════════════════
NORM_LAB = {'raw': 'raw (Fig. 4)', 'baseline-std': 'baseline s.d.', 'eqnorm': 'whole-trial s.d.',
            'pooled-evoked': 'evoked s.d.', "d'-action": 'd′ (action window)', 'gap-action': 'lick − no-lick gap'}


def panel_a(fig, gs):
    N = C['norm']; norms = N['norms']; yv = np.arange(len(norms))[::-1]
    axes = []
    for k, (dat, ttl, xlab) in enumerate([(N['push'], 'push (within mouse)', 'LMM β (depth ~ stage)'),
                                          (N['coup'], 'coupling (between mice)', 'Spearman ρ')]):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        for i, nm in enumerate(norms):
            val, p = dat[nm]; sig = p < 0.05
            ax.scatter(val, yv[i], s=34, color=SIGC if sig else NSC, zorder=3, edgecolors='k', linewidths=0.4)
            ax.text(val, yv[i] + 0.26, f'p = {p:.4f}' if 0.045 < p < 0.055 else f'p = {p:.3f}', ha='center', va='bottom', fontsize=PS*6.0,
                    color='k' if sig else '0.45')
            print(f'a: {ttl[:8]} {nm:14s} {val:+.3f} p={p:.3f}')
        ax.axvline(0, ls=':', color='k', lw=0.8)
        ax.set_yticks(yv); ax.set_yticklabels([NORM_LAB[n] for n in norms] if k == 0 else [])
        ax.set_xlabel(xlab); ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        ax.margins(x=0.25); ax.set_ylim(-0.6, len(norms) - 0.1)
        if k == 1:
            ax.set_xlim(-1.05, 0.15); ax.set_xticks([-1, -0.5, 0])
    return axes[0]


# ══ b — per-stage axes against one fixed pooled axis ════════════════════════════════════════
def panel_b(fig, gs):
    M = C['common']['modes']; mice = C['common']['mice']
    axes = []
    for ci, (mode, ttl) in enumerate([('perstage', 'per-stage axes, held out'), ('commonPool', 'one axis, both stages')]):
        E = M[mode]; df = E['df']
        ax = fig.add_subplot(gs[0, ci]); axes.append(ax)
        piv = df.pivot_table(index=['mouse', 'sample'], columns='st', values='depth')
        for (m, sl), r in piv.iterrows():
            ax.plot([0, 1], [r[0], r[1]], '-o', color=MC[m], lw=0.7, ms=3.2, mec='w', mew=0.3,
                    mfc=(MC[m] if sl == 'A' else 'w'), zorder=3, alpha=0.9)
        for x, k in ((-0.18, 0), (1.18, 1)):
            v = df[df.st == k].depth.values
            ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=5, capsize=3, lw=1.1, zorder=5)
        b, p = E['push']
        ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert']); ax.set_xlim(-0.5, 1.5)
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        ax.text(0.03, 0.03, f'β = {b:+.2f}, p = {p:.3f}', transform=ax.transAxes, va='bottom', fontsize=PS*6.5, color='0.3')
        verdict(ax, p)
        if ci == 0:
            ax.set_ylabel('choice-code depth\n← no lick    lick →')
        ax2 = fig.add_subplot(gs[1, ci]); axes.append(ax2)
        ddm, dam = np.asarray(E['ddm']), np.asarray(E['dam']); ok = np.isfinite(ddm) & np.isfinite(dam)
        for i, m in enumerate(mice):
            ax2.scatter(ddm[i], dam[i], color=MC[m], s=30, edgecolors='w', linewidths=0.5, zorder=4)
        z = np.polyfit(ddm[ok], dam[ok], 1); xx = np.array([ddm[ok].min(), ddm[ok].max()])
        ax2.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.2, zorder=3)
        ax2.axhline(0, ls=':', color='0.6', lw=0.7); ax2.axvline(0, ls=':', color='0.6', lw=0.7)
        ax2.text(0.03, 0.03, f'ρ = {E["rho"]:+.2f}, p = {E["p"]:.3f}', transform=ax2.transAxes, va='bottom', fontsize=PS*6.5, color='0.3')
        verdict(ax2, E['p'])
        lo, hi = ax2.get_ylim(); ax2.set_ylim(lo - 0.22 * (hi - lo), hi)
        ax2.set_xlabel('Δ choice-code depth (expert − naïve)')
        if ci == 0:
            ax2.set_ylabel('Δ DPA accuracy\n(expert − naïve)')
        print(f'b: {mode:10s} push β={b:+.2f} p={p:.3f}   coupling ρ={E["rho"]:+.2f} p={E["p"]:.3f}')
    return axes[0]


# ══ c — the coupling under three decoders ═══════════════════════════════════════════════════
def panel_c(fig, gs):
    axes = []
    for k, (key, ttl) in enumerate([('l2', 'ridge logistic (Fig. 4c)'), ('l1', 'L1 logistic'), ('lda', 'shrinkage LDA')]):
        E = CV[key]; ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        dd, dp = np.asarray(E['dd']), np.asarray(E['dp_dpa'])
        for m, x, yv in zip(E['mice'], dd, dp):
            ax.scatter(x, yv, color=MC[m], s=30, edgecolors='w', linewidths=0.5, zorder=4)
        z = np.polyfit(dd, dp, 1); xx = np.array([dd.min(), dd.max()])
        ax.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.2, zorder=3)
        ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
        ax.text(0.03, 0.03, f'ρ = {E["rho"]:+.2f}, p = {E["p"]:.3f}', transform=ax.transAxes, va='bottom', fontsize=PS*6.5, color='0.3')
        verdict(ax, E['p'])
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.22 * (hi - lo), hi)
        if k == 1:
            ax.set_xlabel('Δ choice-code depth (expert − naïve)')
        if k == 0:
            ax.set_ylabel('Δ DPA accuracy\n(expert − naïve)')
        print(f'c: {key:4s} ρ={E["rho"]:+.2f} p={E["p"]:.3f} n={len(dd)}')
    return axes[0]


# ══ d — late-delay licking ═══════════════════════════════════════════════════════════════════
def panel_d(fig, gs):
    L = C['lick']; d = L['trial']
    axes = []
    ax = fig.add_subplot(gs[0, 0]); axes.append(ax)
    for st, col in [('Naive', '0.55'), ('Expert', '#332288')]:
        ds = d[d.learning == st]; ax.scatter(ds.lick_delay, ds.depth, s=4, color=col, alpha=0.15, lw=0, label=st.lower() if st == 'Expert' else 'naïve')
    rr, prr = L['trial_rho']
    ax.set_xlabel('late-delay lick rate (Hz)'); ax.set_ylabel('choice-code depth')
    ax.set_title('depth vs licking', loc='left', fontsize=TITLE_FS)
    ax.text(0.97, 0.04, f'ρ = {rr:+.2f}\n{len(d)} trials', transform=ax.transAxes, ha='right', va='bottom', fontsize=PS*6.5, color='0.3')
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.legend(frameon=False, loc='upper right', markerscale=3, handletextpad=0.2)
    ax = fig.add_subplot(gs[0, 1]); axes.append(ax)
    ticks = []
    for i, (lab, (bta, se, p)) in enumerate([('none', L['m0']), ('+ lick', L['m1'])]):
        col = SIGC if p < 0.05 else NSC
        ax.errorbar(i, bta, se, fmt='o', color=col, ms=5, capsize=3, lw=1.1)
        ticks.append(lab)
        print(f'd: push covariate={lab:7s} β={bta:+.3f} ± {se:.3f} p={p:.3f}')
    ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(ticks, fontsize=PS*6.2); ax.set_xlim(-0.9, 1.9)
    ax.set_ylabel('push: LMM β, depth ~ stage'); ax.set_title('push | lick', loc='left', fontsize=TITLE_FS); ax.set_ylim(-1.0, 0.12)
    ax = fig.add_subplot(gs[0, 2]); axes.append(ax)
    gd = L['gd']
    for m in gd.index:
        ax.scatter(gd.loc[m, 'dd'], gd.loc[m, 'da'], color=MC[m], s=30, edgecolors='w', linewidths=0.5, zorder=4)
    z = np.polyfit(gd.dd, gd.da, 1); xx = np.array([gd.dd.min(), gd.dd.max()]); ax.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.2)
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
    (r0, p0), (rp, pp), (rl, pl) = L['r0'], L['rp'], L['rl']
    ax.text(0.03, 0.03, f'ρ = {r0:+.2f}, p = {p0:.3f}\npartial | Δlick: r = {rp:+.2f}, p = {pp:.3f}\nΔlick vs Δaccuracy: ρ = {rl:+.2f}, p = {pl:.2f}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=PS*6.2, color='0.3')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.32 * (hi - lo), hi)
    ax.set_xlabel('Δ choice-code depth (expert − naïve)'); ax.set_ylabel('Δ DPA accuracy')
    ax.set_title('coupling | Δlick', loc='left', fontsize=TITLE_FS)
    print(f'd: coupling ρ={r0:+.2f} p={p0:.3f}; partial r={rp:+.2f} p={pp:.3f}; lick-acc ρ={rl:+.2f} p={pl:.2f}')
    return axes[0]


# ══ c — the fixed common axis, CCGD held-out (exp_common_axis_ccgd.py) ═════════════════════════
def panel_cc(fig, gs):
    CC = C['common_ccgd']
    axes = []
    for ci, (key, ttl) in enumerate([('perstage', 'per-stage axes (Fig. 4)'), ('pooled', 'one axis, both stages')]):
        E = CC[key]; g = E['g']
        ax = fig.add_subplot(gs[0, ci]); axes.append(ax)
        piv = g.pivot_table(index=['mouse', 'sample'], columns='st', values='depth')
        for (m, sl), r in piv.iterrows():
            ax.plot([0, 1], [r[0], r[1]], '-o', color=MC[m], lw=0.7, ms=3.2, mec='w', mew=0.3,
                    mfc=(MC[m] if sl == 'A' else 'w'), zorder=3, alpha=0.9)
        for x, k in ((-0.18, 0), (1.18, 1)):
            v = g[g.st == k].depth.values
            ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=5, capsize=3, lw=1.1, zorder=5)
        b, p = E['push']
        ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert']); ax.set_xlim(-0.5, 1.5)
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS)
        ax.text(0.03, 0.03, f'β = {b:+.3f}, p = {p:.3f}', transform=ax.transAxes, va='bottom', fontsize=PS*6.5, color='0.3')
        verdict(ax, p)
        if ci == 0:
            ax.set_ylabel('choice-code depth (log-odds)\n← no lick    lick →')
        ax2 = fig.add_subplot(gs[1, ci]); axes.append(ax2)
        ddm, dam = np.asarray(E['ddm']), np.asarray(E['dam'])
        for m, x, yv in zip(E['mice'], ddm, dam):
            ax2.scatter(x, yv, color=MC[m], s=30, edgecolors='w', linewidths=0.5, zorder=4)
        z = np.polyfit(ddm, dam, 1); xx = np.array([ddm.min(), ddm.max()])
        ax2.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.2, zorder=3)
        ax2.axhline(0, ls=':', color='0.6', lw=0.7); ax2.axvline(0, ls=':', color='0.6', lw=0.7)
        lo, hi = ax2.get_ylim(); ax2.set_ylim(lo - 0.22 * (hi - lo), hi)
        ax2.text(0.03, 0.03, f'ρ = {E["rho"]:+.2f}, p = {E["p"]:.3f}', transform=ax2.transAxes, va='bottom', fontsize=PS*6.5, color='0.3')
        verdict(ax2, E['p'])
        ax2.set_xlabel('Δ depth (expert − naïve)')
        if ci == 0:
            ax2.set_ylabel('Δ DPA accuracy\n(expert − naïve)')
        print(f'c: {key:9s} push β={b:+.3f} p={p:.3f}   coupling ρ={E["rho"]:+.2f} p={E["p"]:.3f}')
    return axes[0]


# ══ d — late-delay licking on the CCGD depth (exp_lick_control_ccgd.py) ═══════════════════════
def panel_dd(fig, gs):
    """ED 4d (2026-09-15, trial level): exp_lick_control_trial.py — every held-out CCGD decision function aligned to
    its behavioural trial via the session trial index written by run_overlaps.py (`--tag trial` re-run)."""
    L = C['lick_trial']; R = L['rho']
    axes = []
    # left: per mouse x stage Spearman(depth, lick rate) over trials
    ax = fig.add_subplot(gs[0, 0]); axes.append(ax)
    for _, r in R.iterrows():
        ax.scatter(r.st + np.random.RandomState(MICE.index(r.mouse)).uniform(-0.12, 0.12), r.rho, s=30, color=MC[r.mouse],
                   marker='o', facecolors=MC[r.mouse] if r.st == 1 else 'none', linewidths=1.0, zorder=3)
    ax.axhline(0, ls=':', color='0.6', lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['naïve', 'expert']); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('Spearman ρ (depth, lick rate)\nover trials, per mouse')
    ax.set_title('depth vs licking, per trial', loc='left', fontsize=TITLE_FS)
    pw = L['wilcoxon_p']; sig = pw < 0.05
    ax.text(0.5, 0.97, '∗' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top', fontsize=PS*(12 if sig else 8), fontweight='bold', color='k' if sig else '0.55')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.40 * (hi - lo), hi)
    ax.text(0.97, 0.03, f'median ρ = {L["per"].median():+.2f}\np = {pw:.2f}, {len(L["per"])} mice\n{L["n_trials"]} trials', transform=ax.transAxes, ha='right', va='bottom', fontsize=PS*6.5, color='0.3')
    # middle: the push on all trials, on no-lick trials, and with a trial-level lick covariate
    ax = fig.add_subplot(gs[0, 1]); axes.append(ax); ticks = []
    for i, (lab, (bta, se, p)) in enumerate([('all', L['push_all'][:3]), ('no-lick', L['push_nolick'][:3]), ('lick', L['push_lick'][:3])]):
        col = SIGC if p < 0.05 else NSC
        ax.errorbar(i, bta, se, fmt='o', color=col, ms=5, capsize=3, lw=1.1); ticks.append(lab)
    ax.axhline(0, ls=':', color='0.6', lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['all', 'no-\nlick', 'lick'], fontsize=PS*6.2); ax.set_xlim(-0.7, 2.7)
    ax.set_ylabel('push: LMM β, depth ~ stage'); ax.set_title('push by trial set', loc='left', fontsize=TITLE_FS)
    # right: the coupling with Δdepth from no-lick trials only
    ax = fig.add_subplot(gs[0, 2]); axes.append(ax)
    gd = L['gd']
    for m in gd.index:
        ax.scatter(gd.loc[m, 'dd_nolick'], gd.loc[m, 'da'], color=MC[m], s=30, edgecolors='w', linewidths=0.5, zorder=4)
    z = np.polyfit(gd.dd_nolick, gd.da, 1); xx = np.array([gd.dd_nolick.min(), gd.dd_nolick.max()]); ax.plot(xx, np.polyval(z, xx), '-', color='0.3', lw=1.2)
    ax.axhline(0, ls=':', color='0.6', lw=0.7); ax.axvline(0, ls=':', color='0.6', lw=0.7)
    (r0, p0), (rn, pn) = L['r_all'], L['r_nolick']
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.32 * (hi - lo), hi)
    ax.text(0.03, 0.03, f'no-lick trials: ρ = {rn:+.2f}, p = {pn:.3f}\nall trials: ρ = {r0:+.2f}, p = {p0:.3f}', transform=ax.transAxes, ha='left', va='bottom', fontsize=PS*6.2, color='0.3')
    ax.set_xlabel('Δ choice-code depth, no-lick trials'); ax.set_ylabel('Δ DPA accuracy')
    ax.set_title('coupling, no-lick trials', loc='left', fontsize=TITLE_FS)
    print(f'd: licks on {L["frac_lick"]:.1%} of trials; per-mouse ρ median {L["per"].median():+.2f} p={pw:.3f}; push all {L["push_all"][0]:+.3f} p={L["push_all"][2]:.3f}, '
          f'no-lick {L["push_nolick"][0]:+.3f} p={L["push_nolick"][2]:.3f}, lick {L["push_lick"][0]:+.3f} p={L["push_lick"][2]:.3f}; trial LMM +lick covariate β_st {L["lmm_lick"][0]:+.3f} p={L["lmm_lick"][2]:.3f} (lick β {L["lmm_lick"][3]:+.3f} p={L["lmm_lick"][4]:.3f}); coupling no-lick ρ={rn:+.2f} p={pn:.3f} (all ρ={r0:+.2f} p={p0:.3f})')
    return axes[0]


# ══ ASSEMBLE ═══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.0, 7.4))
outer = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.25], hspace=0.45, wspace=1.0, left=0.09, right=0.985, top=0.95, bottom=0.075)
gsA = outer[0, 0:5].subgridspec(1, 2, wspace=0.25, width_ratios=[1, 1])
axA = panel_a(fig, gsA)
gsB = outer[0, 6:12].subgridspec(1, 3, wspace=0.42)
axB = panel_c(fig, gsB)
gsCC = outer[1, 0:5].subgridspec(2, 2, wspace=0.30, hspace=0.45)
axCC = panel_cc(fig, gsCC)
gsDD = outer[1, 6:12].subgridspec(1, 3, wspace=0.75, width_ratios=[1, 0.6, 1])
axDD = panel_dd(fig, gsDD)
plabel(axA, 'a', dx=-0.62); plabel(axB, 'b', dx=-0.40); plabel(axCC, 'c', dx=-0.48); plabel(axDD, 'd', dx=-0.40)

CAP = [
    'Extended Data Fig. 7 | The push and the learning coupling under other units, other decoders, a fixed axis and a '
    'lick covariate (companion to Fig. 4b,c). All panels read cross-validated decision functions of the CCGD '
    'pipeline; the depth is the raw log-odds of Fig. 4 unless stated. a, The push (left; within-mouse mixed model, '
    'depth ~ stage + sample, random intercept per mouse, 36 observations) and the coupling (right; per-mouse Spearman '
    'ρ between Δdepth and ΔDPA accuracy on the GNG-free DPA trials, n = 9) under six units of the same late-delay '
    'depth. Red, p < 0.05. The coupling holds under every unit (ρ = −0.67 to −0.80; the whole-trial-s.d. unit sits '
    'at the boundary, p = .0499); the push, tested here with the across-animal mixed model, is a trend in raw units and reaches significance only in evoked-s.d. and whole-trial-s.d. units (the within-animal permutation test of Fig. 4b gives p = .006). '
    'b, The coupling under three decoders: the ridge logistic decoder of Fig. 4c (ρ = −0.80, bootstrap 95% CI over '
    'mice [−0.98, −0.24]), an L1-regularized logistic decoder (−0.73 [−1.00, −0.11]) and a shrinkage linear '
    'discriminant (−0.45 [−0.89, +0.29]).',
    'c, The same two statistics on the per-stage decoder axes of Fig. 4 (left) and on one choice axis fitted per '
    'mouse to the naïve and expert trials pooled (right; neurons registered in both stages; every trial read from '
    'the fold that held it out; pooled-axis coupling ρ = +0.65 [−0.15, +1.00]). d, Late-delay licking, trial by trial: '
    'every held-out decision function aligned to its behavioural trial (seeded folds and a session trial index; lick '
    'rate over 5.5–7.0 s after sample-odor onset, the 7.5–9.0 s late-delay window of the depth on the imaging clock). Left, '
    'Spearman ρ between depth and lick rate over the trials of each mouse and stage (open, naïve; filled, expert; '
    'Wilcoxon over the per-mouse means, eight mice with a late-delay lick at both stages: median ρ = +0.05, p = .20; licks on '
    '6.9% of the 1,824 matched trials). '
    'Middle, the push (mixed model on mouse × sample × stage means, as in a) on all trials (β = −0.077, p = .14), on the '
    'trials without a late-delay lick (−0.072, p = .16) and on the trials with one (−0.156, p = .14, 28 cells); a '
    'trial-level model with the lick rate as a covariate leaves the stage term unchanged (lick β = +0.06, p = .28). '
    'Right, the coupling with Δdepth computed from no-lick trials only (ρ = −0.80, p = .009; all trials of this run, '
    'ρ = −0.69, p = .038). This panel reads a re-run of the choice decoder with seeded folds, a new cross-validation '
    'draw of the same pipeline. Mouse colours as in Fig. 4; ∗ p < 0.05, n.s. '
    'otherwise.',
]
if not NOCAP:
    draw_justified(fig, CAP, fontsize=PS*7.2)
OUT = '/home/leon/dual/figures/ed'
fig.savefig(f'{OUT}/png/ed_fig7.png', bbox_inches='tight'); fig.savefig(f'{OUT}/svg/ed_fig7.svg', bbox_inches='tight')
print('saved', f'{OUT}/png/ed_fig7.png')
