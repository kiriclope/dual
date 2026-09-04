"""fig_overlaps_main_native.py — Fig 4: WHAT LEARNING DOES on the manifold.

RESTRUCTURED 2026-08-30 (user decision, "redistribute"): this is now the full LEARNING figure —
the dist code aligns onto the choice axis, the memory state is pushed along it, the push predicts
behaviour, and fidelity is unchanged. Panels:
  A  learning couples dist ↔ choice (MOVED FROM Fig 3): cross-decode matrices (Naive | Expert) +
     per-mouse raw |cos|(choice,dist) ∗ + per-mouse cross-decode ∗. BOTH per-mouse tests are
     knob-robust (raw-cos p=.008/.008, cross-dec p=.0078/.0039 across pca20/nopca) — whitelisted ∗.
     Drawn from the CANONICAL no-PCA caches regardless of this build's flags (pipeline-level claim).
  B  the no-lick push — sample×choice trajectory planes (Naive | Expert) + delay choice-code KDE strips
     + per-mouse late-delay choice-code depth deepening Naive→Expert (LMM p=.046 ∗, kept per user
     decision 2026-08-30; caption discloses the per-animal trend p=.098).
  C  Δdepth ↔ Δaccuracy coupling (ΔDPA | ΔGNG) — the deepening predicts DPA (not GNG) accuracy (n=9 Spearman).
  D  Naive nonpaired trials — the no-lick well depth predicts correct-reject vs false-alarm (control).
  E  choice-code d′ unchanged (control: position moves, fidelity does not).

Output filename is unchanged (`fig_overlaps_main_ab{FILE_SUF}`) so existing gallery/doc references
keep working.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main_native.py
Output: figures/overlaps/main/{png,svg}/fig_overlaps_main_ab_dpaact.{png,svg}  (default = bins 57-62
pooled-evoked)

ALTERNATIVE build: --antact (single anticipatory+action axis, choice @ bins 48-62) + --robust (robust
sample-separation units + trial-level odor-A random-slope LMM) → fig_overlaps_main_ab_dpaact_antact_robust.
On that axis push p=.032 and coupling ρ=-.68 p=.042 — but its FA/CR star is GONE: the old p=.045 ★ was
computed on the winsorised per-mouse medians; on the unclipped values (2026-08-30) it is p=.056 n.s.
"All three panels ★" is no longer true for that build.

See main_panels.py for the full analysis/method docstring and the --eqnorm/--testwin/--l1/--lda/--cv10/
--ld/--ld05/--gngact/--antact/--robust flags (parsed at import from sys.argv).
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

# Import the module: loads the tensors, applies the uniform normalisation, and defines every panel builder
# (renders nothing). Pull ALL of its module-level names into this script's globals so the assembly reads as before.
# DEFAULT (no flags) = bins 57-62 pooled-evoked (the canonical Fig 4). --antact (48-62 axis) + --robust (sample-sep
# units + trial-level odor-A random-slope LMM) produce the ALTERNATIVE → fig_overlaps_main_ab_dpaact_antact_robust.
import main_panels as _MP
PS = _MP.PS                       # print-scale factor (defined beside the shared rcParams)
globals().update({k: v for k, v in vars(_MP).items() if not k.startswith("__")})


if __name__ == '__main__':
    # ══════════════════════════════════════════════════════════════════════════════
    # FIGURE — push / repositioning of the working-memory state on the manifold (Fig 4)
    # ══════════════════════════════════════════════════════════════════════════════
    # keep rows tight: every row carries aspect-locked axes, so extra height becomes dead bands
    fig = plt.figure(figsize=(10.0, 9.0))
    gs = fig.add_gridspec(3, 12, height_ratios=[0.95, 1.25, 1.5], hspace=0.38, wspace=0.9,
                          left=0.06, right=0.985, top=0.955, bottom=0.045)

    def panel_letter(ax, L, x=0.008, dy=0.014):
        p = ax.get_position()
        fig.text(x, p.y1 + dy, L.lower(), fontsize=PS*10, fontweight='bold', va='top', ha='left')

    # ── A: learning couples the dist code to the choice axis (moved from Fig 3, 2026-08-30) ──
    # CANONICAL no-PCA caches, fixed across the --pca/--robust/--antact build variants: this panel
    # is a pipeline-level claim, not an axis-variant one (see the header docstring).
    import pickle as _pkl
    from scipy.stats import wilcoxon as _wilc
    _MATC = _pkl.load(open('figures/overlaps/ccgp/matrices_cache_acc_nopca.pkl', 'rb'))
    _RESP = _pkl.load(open('/home/leon/dual/pca/figures/pseudo/dimensionality/results.pkl', 'rb'))
    _PMC, _PMA = _RESP['PM_COS_nopca'], _RESP['PM_ACT_nopca']
    gsAL = gs[0, 0:12].subgridspec(1, 4, width_ratios=[3.0, 3.0, 3.6, 3.6], wspace=0.55)
    axAL = []
    for _j, _stage in enumerate(STAGES):
        _ax = fig.add_subplot(gsAL[0, _j]); axAL.append(_ax)
        _M = np.asarray(_MATC['ACT_Mms'][_stage])
        _ax.imshow(_M, cmap='Reds', vmin=0.5, vmax=1.0, aspect='equal')
        for _i in range(2):
            for _k in range(2):
                _ax.text(_k, _i, f'{_M[_i, _k]:.2f}', ha='center', va='center', fontsize=PS*6.6,
                         color='w' if _M[_i, _k] > 0.82 else 'k')
        _ax.set_xticks([0, 1]); _ax.set_xticklabels(['dist', 'choice'], fontsize=PS*6.0)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(['dist', 'choice'] if _j == 0 else [], fontsize=PS*6.0)
        _ax.set_title(_stage, loc='left', fontsize=TITLE_FS)
        if _j == 0:
            _ax.set_ylabel('dist ↔ choice\ncross-dec. (bal. acc.)', fontsize=PS*7)
        _S = _MATC['ACT_SUMM'][_stage]
        _ax.text(0.5, -0.30, f"off/within {_S['offdiag']:.2f}\n[{_S['offdiag_lo']:.2f}, {_S['offdiag_hi']:.2f}]",
                 transform=_ax.transAxes, ha='center', va='top', fontsize=PS*6.0, color='0.3')
        for _sp in _ax.spines.values():
            _sp.set_visible(True)
        print(f"A[align] {_stage} off/within {_S['offdiag']:.2f} [{_S['offdiag_lo']:.2f},{_S['offdiag_hi']:.2f}]")

    def _pm_scatter(ax, nv, ev, lo, hi, xlab, ylab, title):
        """Per-mouse Naive->Expert scatter (house idiom). ∗ policy: both tests drawn here are
        knob-robust across the pca20/nopca pipelines (raw-cos p=.008/.008, cross-dec .0078/.0039)
        — whitelisted. The raw-cos star deliberately REVERSES the old 'never star the cosine test'
        verdict, which was about the retired attenuation-corrected estimator (logged 2026-08-31)."""
        ax.plot([lo, hi], [lo, hi], ls='--', color='0.6', lw=0.8, zorder=0)
        n = np.array([nv[m] for m in ALL_MICE if m in nv and m in ev])
        e = np.array([ev[m] for m in ALL_MICE if m in nv and m in ev])
        for m in ALL_MICE:
            if m in nv and m in ev:
                ax.scatter(nv[m], ev[m], s=34, color=MOUSE_COLOR[m], marker=GMARKER[GROUP[m]],
                           edgecolors='w', linewidths=0.5, zorder=3)
        p = float(_wilc(e, n).pvalue)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        ax.set_xlabel(xlab, fontsize=PS*7); ax.set_ylabel(ylab, fontsize=PS*7)
        ax.set_title(f'{title}  ({"∗" if p < .05 else "n.s."})', loc='left', fontsize=TITLE_FS)
        ax.text(0.05, 0.96, f'Δ={e.mean() - n.mean():+.2f}\np={p:.3f}', transform=ax.transAxes,
                va='top', ha='left', fontsize=PS*6, color='0.3')
        print(f'A[align] {title}: {n.mean():.3f} -> {e.mean():.3f}  p={p:.4f}')

    _axc = fig.add_subplot(gsAL[0, 2])
    _pm_scatter(_axc,
                {m: _PMC[(m, 'Naive')]['ad_raw'] for m in ALL_MICE if (m, 'Naive') in _PMC},
                {m: _PMC[(m, 'Expert')]['ad_raw'] for m in ALL_MICE if (m, 'Expert') in _PMC},
                0.0, 0.3, 'raw |cos| Naive', 'raw |cos| Expert', 'choice × dist')
    _axd = fig.add_subplot(gsAL[0, 3])
    _axd.axhline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
    _axd.axvline(0.5, ls=':', color='0.85', lw=0.6, zorder=0)
    _pm_scatter(_axd,
                {m: np.nanmean([_PMA[(m, 'Naive')]['g2l'], _PMA[(m, 'Naive')]['l2g']])
                 for m in ALL_MICE if (m, 'Naive') in _PMA},
                {m: np.nanmean([_PMA[(m, 'Expert')]['g2l'], _PMA[(m, 'Expert')]['l2g']])
                 for m in ALL_MICE if (m, 'Expert') in _PMA},
                0.40, 0.85, 'cross-dec. Naive', 'cross-dec. Expert', 'dist ↔ choice')

    # ── B: the no-lick push (full row: Naive traj|kde, Expert traj|kde, per-mouse depth deepening) ──
    gsB = gs[1, 0:12].subgridspec(1, 5, width_ratios=[5, 1.2, 5, 1.2, 4.4], wspace=0.3)
    _valsx, _valsy_traj, _valsy_kde = [], [], []
    for _stg in STAGES:
        for _slab, _p, _c in SAMPLE_TRAJ:
            _xs, _ys = trajB[_stg][_slab]
            if not _xs or not _ys:
                continue
            _ax = np.stack(_xs, 0)[:, :TRAJ_END]; _ay = np.stack(_ys, 0)[:, :TRAJ_END]
            _nm = _ax.shape[0]
            _valsx.append(np.abs(_ax.mean(0)) + _ax.std(0, ddof=1) / np.sqrt(_nm))
            _valsy_traj.append(np.abs(_ay.mean(0)) + _ay.std(0, ddof=1) / np.sqrt(_nm))
            _valsy_kde += [np.abs(np.asarray(_yt)[BINS_LATE]) for _yt in _ys]   # same window as the KDE strips
    _kA = np.concatenate(_valsy_kde) if _valsy_kde else np.array([4.0])
    _rBx = float(np.concatenate(_valsx).max()) * 1.12 if _valsx else 4.0
    _rBy = float(np.concatenate(_valsy_traj).max()) * 1.28 if _valsy_traj else 4.0
    xlimB, ylimB = (-_rBx, _rBx), (-_rBy, _rBy)
    print(f'A traj limits: x=±{_rBx:.2f}  y=±{_rBy:.2f}  (kde tail dropped to taper inside; p90={np.percentile(_kA,90):.2f})')
    axB_traj, axB_hist = [], []
    ax0 = None
    for ci, stage in enumerate(STAGES):
        at = fig.add_subplot(gsB[0, ci * 2])
        ah = fig.add_subplot(gsB[0, ci * 2 + 1], sharey=at)
        ax0 = at if ax0 is None else ax0
        _draw_traj_B(at, stage, xlimB, ylimB)
        at.set_title(stage, pad=4, fontsize=TITLE_FS)
        at.set_xlabel('Sample code\n← odor A            odor B →')
        if ci == 0:
            at.set_ylabel('Choice code\n← no lick            lick →')
        _draw_hist_B(ah, stage, ylimB)
        axB_traj.append(at); axB_hist.append(ah)
    pair_handles = [Line2D([0], [0], color=_c, lw=1.3, label=f'Sample {_l}') for _l, _p, _c in SAMPLE_TRAJ]
    axB_traj[-1].legend(handles=pair_handles, frameon=False, loc='upper right',
                        handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, fontsize=PS*8)

    # depth panel: per-mouse late-delay choice-code depth, Naive → Expert (deepening mixed model)
    axB_sc = fig.add_subplot(gsB[0, 4])
    GX_B = (0.0, 1.0)
    for _slab, _fill in (('A', True), ('B', False)):
        P = pushB[_slab]
        for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
            _mc = MOUSE_COLOR[mo]
            axB_sc.plot(GX_B, [xn, ye], '-', color=_mc, lw=0.7, alpha=0.5, zorder=2)
            axB_sc.scatter(GX_B[0], xn, s=34, zorder=3, linewidths=1.0, facecolors=_mc if _fill else 'w', edgecolors=_mc)
            axB_sc.scatter(GX_B[1], ye, s=34, zorder=3, linewidths=1.0, facecolors=_mc if _fill else 'w', edgecolors=_mc)
    # group mean/SEM at the MOUSE level (A/B averaged within mouse, n=9) — concatenating A+B gave
    # an n=18 SEM with every mouse counted twice (anti-conservative; display-only, stat is the LMM)
    _bym = {'naive': {}, 'expert': {}}
    for _s in ('A', 'B'):
        for mo, xn, ye in zip(pushB[_s]['mice'], pushB[_s]['naive'], pushB[_s]['expert']):
            _bym['naive'].setdefault(mo, []).append(xn)
            _bym['expert'].setdefault(mo, []).append(ye)
    _naive_all = np.array([np.mean(v) for v in _bym['naive'].values()])
    _expert_all = np.array([np.mean(v) for v in _bym['expert'].values()])
    for _xx, _vals in ((GX_B[0], _naive_all), (GX_B[1], _expert_all)):
        _mu = _vals.mean(); _se = _vals.std(ddof=1) / np.sqrt(len(_vals))
        axB_sc.plot([_xx - 0.14, _xx + 0.14], [_mu, _mu], color='k', lw=1.3, zorder=4)
        axB_sc.errorbar(_xx, _mu, yerr=_se, color='k', capsize=2.5, lw=1.2, zorder=4)
    axB_sc.axhline(0, ls=':', color='0.6', lw=0.7)
    _dfp = pd.DataFrame([dict(mouse=mo, sample=_s, st=_st, depth=_v)
                         for _s in ('A', 'B') for _st, _k in ((0, 'naive'), (1, 'expert'))
                         for mo, _v in zip(pushB[_s]['mice'], pushB[_s][_k])])
    _nmB, _noB = _dfp['mouse'].nunique(), len(_dfp)
    if _MP.ROBUST:                                                        # settled stat (2026-08-06): trial-level odor-A RANDOM-SLOPE LMM (p≈.007), not the aggregated random-intercept
        _dep = _MP.LICK_D[:, _MP.BINS_LATE].mean(1)
        _selA = ((_MP.Lm.laser == 0) & (_MP.Lm.tasks == 'DPA') & _MP.Lm.odor_pair.isin([0, 1])).to_numpy()
        _dfA = pd.DataFrame(dict(depth=_dep[_selA], mouse=_MP.Lm.mouse.to_numpy()[_selA],
                                 st=(_MP.Lm.stage.to_numpy()[_selA] == 'Expert').astype(int)))
        _mA = smf.mixedlm('depth ~ st', _dfA, groups='mouse', re_formula='~st').fit(reml=False)
        _bpush, _ppush = float(_mA.params['st']), float(_mA.pvalues['st'])
        _statlbl = f'odor-A random-slope LMM\n({_nmB} mice, {_selA.sum()} A-trials)'
    else:
        _pfit = smf.mixedlm('depth ~ st + C(sample)', _dfp, groups=_dfp['mouse']).fit()
        _bpush, _ppush = float(_pfit.params['st']), float(_pfit.pvalues['st'])
        _statlbl = f'mixed model ({_nmB} mice, {_noB} obs)'
    _sigB = _ppush < 0.05
    print(f'A depth [{_statlbl.splitlines()[0]}] β={_bpush:+.3f} p={_ppush:.3f} ({_nmB} mice)')
    # ── the SAMPLE-SPECIFICITY test, shown rather than left to be inferred (2026-08-30) ──────────
    # "A moved, B did not" is NOT evidence that A differs from B: the direct paired comparison
    # across the same 9 mice is n.s. (p≈.055), and it is not a decoder artefact either — the sample
    # and choice axes are orthogonal (per-mouse |cos| = 0.04) and sample-axis leakage would push A
    # and B in OPPOSITE directions, which only 3/9 mice show. Printed to stdout, NOT drawn: an
    # effect-size inset was tried here and removed — the panel already carries traces, KDEs and
    # per-mouse lines. Keep the claim out of the figure and state it in the text.
    from scipy.stats import wilcoxon as _wcx
    _pv = _dfp.pivot_table(index=['mouse', 'sample'], columns='st', values='depth')
    _dlt = {_s: (_pv.xs(_s, level='sample')[1] - _pv.xs(_s, level='sample')[0]).to_numpy()
            for _s in ('A', 'B')}
    _pA, _pB = _wcx(_dlt['A']).pvalue, _wcx(_dlt['B']).pvalue
    _pAB = _wcx(_dlt['A'], _dlt['B']).pvalue
    _spec = ''
    print(f'A depth per-mouse: ΔA={_dlt["A"].mean():+.2f} (p={_pA:.3f})  '
          f'ΔB={_dlt["B"].mean():+.2f} (p={_pB:.3f})  A-vs-B p={_pAB:.3f}')
    axB_sc.set_xlim(-0.5, 1.5); axB_sc.set_xticks(GX_B); axB_sc.set_xticklabels(['Naive', 'Expert'])
    axB_sc.set_box_aspect(1)
    axB_sc.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=PS*7.5)
    axB_sc.text(0.03, 0.03, f'{_statlbl}\nβ={_bpush:+.3f}, p={_ppush:.3f}\n{_spec}',
                transform=axB_sc.transAxes, ha='left', va='bottom', fontsize=PS*6.5, color='0.3')
    axB_sc.text(0.06, 0.96, '*' if _sigB else 'n.s.', transform=axB_sc.transAxes, ha='left', va='top',
                fontsize=PS*12 if _sigB else 8, fontweight='bold', color='k' if _sigB else '0.55')
    axB_sc.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
                           mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')],
                  frameon=False, loc='upper right', fontsize=PS*6.5, handletextpad=0.3,
                  borderaxespad=0.2, labelspacing=0.3)

    # ── C: Δdepth ↔ Δperf (Expert−Naive), A&B independent (ΔDPA | ΔGNG) ──
    gsC = gs[2, 0:6].subgridspec(1, 2, wspace=0.55)
    axC = [fig.add_subplot(gsC[0, 0]), fig.add_subplot(gsC[0, 1])]
    C_specs = [(delta_dpa_perf_sample, 'Δ DPA accuracy (Exp−Naive)', 'Δ depth vs Δ DPA accuracy',
                _panelC_coupling(delta_dpa_perf_sample)),
               (delta_gng_perf_sample, 'Δ GNG accuracy (Exp−Naive)', 'Δ depth vs Δ GNG accuracy',
                _panelC_coupling(delta_gng_perf_sample))]
    _allyC = np.array([d[(m, c)] for d, _, _, _ in C_specs for m in ALL_MICE for c in (0, 1)], float)
    _allyC = _allyC[~np.isnan(_allyC)]
    _padC = (_allyC.max() - _allyC.min()) * 0.15 or 0.05
    ylimC = (_allyC.min() - _padC, _allyC.max() + _padC)
    for ax, (yv_dict, ylabel, msg, (rho, pv, n_mice, mx, my)) in zip(axC, C_specs):
        for mouse in ALL_MICE:
            px, py = [], []
            for cls, pairs in D_SAMPLE_CLASSES:
                xx = delta_choice_sample[(mouse, cls)]
                yy = yv_dict.get((mouse, cls), np.nan)
                px.append(xx); py.append(yy)
                if not (np.isnan(xx) or np.isnan(yy)):
                    face = MOUSE_COLOR[mouse] if cls == 0 else 'w'
                    ax.scatter(xx, yy, facecolors=face, edgecolors=MOUSE_COLOR[mouse],
                               marker='o', s=34, linewidths=1.0, zorder=5)
            ax.plot(px, py, '-', color=MOUSE_COLOR[mouse], lw=0.7, alpha=0.5, zorder=3)
        # (the per-mouse mean diamonds were removed 2026-09-01, user: they hid the points; the
        #  band, ρ and p are still computed on the 9 per-mouse means — the caption states the
        #  inferential unit instead of drawing it)
        regression_band(ax, mx, my)
        ax.axhline(0, ls=':', color='k', lw=0.7); ax.axvline(0, ls=':', color='k', lw=0.7)
        ax.set_ylim(ylimC)
        sig = pv < 0.05
        ax.text(0.03, 0.03, f'per-mouse (n={n_mice})\nSpearman ρ={rho:+.2f}, p={pv:.3f}',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=PS*6.5, color='0.3')
        ax.text(0.92, 0.94, '*' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top',
                fontsize=PS*12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
        ax.set_xlabel('Δ DPA choice-code depth'); ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
        print(f'B[{ylabel[:6]}] per-mouse n={n_mice} Spearman ρ={rho:+.3f} p={pv:.3f}')
    _C_leg = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
              mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')]
    axC[0].legend(handles=_C_leg, frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.0),
                  ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)

    # ── D: Naive nonpaired corr-rej vs false-alarm depth, sample A | sample B ──
    axD = fig.add_subplot(gs[2, 6:9])
    GX_FACR = {'AD': (0.0, 0.8), 'BC': (1.9, 2.7)}
    for lab, samp, odor_pair, col in FA_CR_SPEC:
        xc, xe = GX_FACR[lab]; r = facr[lab]
        for ya, yb, mouse in zip(r['cr'], r['fa'], r['used']):
            mc = MOUSE_COLOR[mouse]
            axD.plot([xc, xe], [ya, yb], '-', color=mc, lw=0.7, alpha=0.5, zorder=2)
            axD.scatter(xc, ya, s=34, facecolors=mc, edgecolors=mc, linewidths=1.0, zorder=3)
            axD.scatter(xe, yb, s=34, facecolors='w', edgecolors=mc, linewidths=1.1, zorder=3)
        for xx, vals in ((xc, r['cr']), (xe, r['fa'])):
            if len(vals):
                mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                axD.plot([xx - 0.18, xx + 0.18], [mu, mu], color='k', lw=1.3, zorder=4)
                axD.errorbar(xx, mu, yerr=se, color='k', capsize=2.5, lw=1.2, zorder=4)
        # stat on the UNCLIPPED per-mouse medians: the 10-90% clip in main_panels is display-only
        # (testing the clipped values would be silent winsorisation, with clip bounds pooled
        # across AD+BC so one pair's outliers would set the other's test)
        n = len(r['cr']); d_mean = float((r['cr_raw'] - r['fa_raw']).mean()) if n else np.nan
        tp = float(ttest_rel(r['cr_raw'], r['fa_raw']).pvalue) if n >= 3 else np.nan
        sig = (tp == tp and tp < 0.05)
        axD.text((xc + xe) / 2, 0.99, f'{lab} (sample {samp})', transform=axD.get_xaxis_transform(),
                 ha='center', va='top', fontsize=PS*7, fontweight='bold', color=col)
        axD.text((xc + xe) / 2, 0.87, '*' if sig else 'n.s.', transform=axD.get_xaxis_transform(),
                 ha='center', va='top', fontsize=PS*12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
        axD.text((xc + xe) / 2, 0.02, f'p={tp:.3f}', transform=axD.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=PS*6.5, color='0.3')
        print(f'C(FA/CR)[Naive {lab} sample {samp}] Δ(cr−fa)={d_mean:+.3f} paired-t p={tp:.3f} n={n}')
    axD.axhline(0, ls=':', color='0.6', lw=0.7)
    _y0D, _y1D = axD.get_ylim()
    axD.set_ylim(_y0D, _y1D + 0.30 * (_y1D - _y0D))   # headroom so the pair titles/stars clear the data
    axD.set_xticks([0.0, 0.8, 1.9, 2.7])
    axD.set_xticklabels(['corr.\nrej.', 'false\nalarm', 'corr.\nrej.', 'false\nalarm'], fontsize=PS*6.5)
    axD.set_xlim(-0.5, 3.2)
    axD.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=PS*7.5)
    axD.set_title('Naive nonpaired trials', loc='left', fontsize=TITLE_FS)
    axD.set_box_aspect(1)

    # ── E: within-task choice-code d′ (Naive vs Expert) — decodability UNCHANGED ⇒ the push (B) is a
    #    POSITION shift, not a fidelity change. (Same d′ that used to live with the code-geometry panels.) ──
    axDp = fig.add_subplot(gs[2, 9:12])
    _dN = np.array([_lick_dprime(m, 'Naive') for m in ALL_MICE]); _dE = np.array([_lick_dprime(m, 'Expert') for m in ALL_MICE])
    _ok = np.isfinite(_dN) & np.isfinite(_dE)
    _av = np.concatenate([_dN[_ok], _dE[_ok]]); _lim = (min(_av.min(), -0.1), _av.max() * 1.12)
    axDp.plot(_lim, _lim, ls='--', color='0.6', lw=0.8, zorder=1)
    axDp.axhline(0, ls=':', color='0.8', lw=0.6); axDp.axvline(0, ls=':', color='0.8', lw=0.6)
    for _m, _xn, _ye in zip(np.array(ALL_MICE)[_ok], _dN[_ok], _dE[_ok]):
        axDp.scatter(_xn, _ye, s=34, facecolors=MOUSE_COLOR[_m], edgecolors=MOUSE_COLOR[_m], linewidths=0.6, zorder=4)
    _dp_t = float(ttest_rel(_dE[_ok], _dN[_ok]).pvalue); _dp_d = float((_dE[_ok] - _dN[_ok]).mean()); _dp_sig = _dp_t < 0.05
    axDp.set_xlim(_lim); axDp.set_ylim(_lim); axDp.set_box_aspect(1)
    # canonical code name ("choice", as in Fig 3); the verdict lives in the n.s./Δ/p annotations,
    # not hardcoded in the title
    axDp.set_title('choice-code d′', fontsize=TITLE_FS, loc='left')
    axDp.set_xlabel('Naive d′', fontsize=PS*7.5); axDp.set_ylabel('Expert d′', fontsize=PS*7.5)
    axDp.text(0.06, 0.95, '*' if _dp_sig else 'n.s.', transform=axDp.transAxes, ha='left', va='top',
              fontsize=PS*11 if _dp_sig else 8, fontweight='bold', color='k' if _dp_sig else '0.55')
    axDp.text(0.97, 0.08, f'Δ={_dp_d:+.2f}, p={_dp_t:.3f}', transform=axDp.transAxes,
              ha='right', va='bottom', fontsize=PS*6, color='0.3')   # clear of the y=0 dotted line
    print(f'D action-code d′ Naive={np.nanmean(_dN):+.2f} Expert={np.nanmean(_dE):+.2f} Δ={_dp_d:+.2f} p={_dp_t:.3f}')

    # ── panel letters ──
    panel_letter(axAL[0], 'A')
    panel_letter(axB_traj[0], 'B')
    panel_letter(axC[0], 'C')
    panel_letter(axD, 'D', x=0.5)
    panel_letter(axDp, 'E', x=0.72)

    # ── CAPTION (justified, drawn below — same mechanism as Figs 2/3) ──
    CAP_PARAS = [
        'Figure 4 | Learning edits the geometry, not the code. The distractor code rotates onto the '
        'choice axis, and the memory state is pushed along that axis to an output-suppressing no-lick '
        'set-point whose depth predicts each animal’s memory gain. Code depth is the projection onto '
        'the choice (lick) decoder axis, per mouse, baseline-zeroed, in units of evoked s.d.; '
        'negative values lie toward no-lick.',
        'a, The action dimension reorganizes: the distractor code aligns with the choice axis. Cross- '
        'decoding between the two codes (balanced accuracy; diagonal, within-code; off-diagonal, '
        'transfer). The chance-referenced transfer grows from 0.33 [−0.03, 0.60] in naïve to 0.57 '
        '[0.36, 0.75] in expert mice. Right, the same convergence within each animal, naïve against '
        'expert: per-mouse |cos| 0.073 → 0.114 (∗ p = .008) and cross-decode 0.53 → 0.61 (∗ p = '
        '.004), both robust across decoder variants and drawn from fixed canonical caches in every '
        'build. The distractor’s demand becomes readable as what it is for the animal, a lick '
        'decision.',
        'b, The no-lick push: the memory state is repositioned along the choice axis. DPA delay '
        'trajectories in the sample × choice plane (naïve | expert; strips, distributions of late- '
        'delay depth) and per-mouse late-delay depth. With learning the delay state sinks into the '
        'half of the axis whose readout is “do not lick”, away from the lick boundary, the geometric '
        'counterpart of the vanishing lick chain in Fig. 1g. Mixed model β = −0.74, p = .046 ∗ (9 '
        'mice, 36 observations); per-animal trend, Wilcoxon p = .098, carried by sample A (Δ = −1.42, '
        'p = .098; sample B ≈ 0; the A-versus-B difference itself is n.s., p = .055).',
        'c, The push predicts behavior across animals. Each mouse’s change in depth against its '
        'change in accuracy (circles, the two sample classes per mouse, joined; the regression band, '
        'ρ and p are computed on the nine per-mouse means). The deeper a mouse pushes its memory '
        'state, the more its DPA accuracy improves (ρ = −0.83, p = .005 ∗), whereas the same change '
        'predicts nothing for GNG (ρ = +0.20, p = .61). The coupling is specific to the memory task.',
        'd, The push is a between-animal learning effect, not a trial-level readout of accuracy. '
        'Within a stage (naïve nonpaired trials), single-trial depth does not separate correct '
        'rejections from false alarms (sample A, Δ(CR−FA) = −1.16, p = .27; sample B, +0.73, p = '
        '.47).',
        'e, Position, not fidelity. The discriminability of the choice code (d′, lick against no- '
        'lick) is unchanged by learning (0.80 → 1.07, p = .25). Learning moves where the memory state '
        'sits on the axis (b), not how well the axis reads out.',
    ]
    if FILE_SUF != '_dpaact':
        CAP_PARAS[0] += (f' [BUILD VARIANT {FILE_SUF}: panel annotations carry this build’s own '
                         'statistics; the numbers quoted here are the canonical action-axis '
                         'pooled-evoked build.]')
    sys.path.insert(0, '/home/leon/dual/pca')
    from figcaption import draw_justified              # shared with Figs 2/3
    if '--nocap' not in sys.argv[1:]:   # submission build: legend goes below the figure
        draw_justified(fig, CAP_PARAS, fontsize=PS*7.2)

    OUT = 'figures/overlaps/main/eqnorm' if EQNORM else 'figures/overlaps/main'
    os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/fig_overlaps_main_ab{FILE_SUF}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        print('saved', os.path.abspath(p))
    plt.close(fig)
