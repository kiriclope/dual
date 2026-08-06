"""fig_overlaps_main_native.py — the overlaps PUSH / REPOSITIONING paper figure (Fig 4).

Learning repositions the working-memory state on the (reused) manifold. Rendered from the importable
helpers/data in `main_panels.py`. Panels:
  A  the no-lick push — sample×choice trajectory planes (Naive | Expert) + delay choice-code KDE strips
     + per-mouse late-delay choice-code depth deepening Naive→Expert.
  B  Δdepth ↔ Δaccuracy coupling (ΔDPA | ΔGNG) — the deepening predicts DPA (not GNG) accuracy (n=9 Spearman).
  C  Naive nonpaired trials — the no-lick well depth predicts correct-reject vs false-alarm.

The CODE GEOMETRY / abstraction half (code traces, within-task d′, shared-action axis, cross-task
generalization) now lives in the manifold figure `fig_overlaps_manifold.py` (Fig 3). Output filename is
unchanged (`fig_overlaps_main_ab{FILE_SUF}`) so existing gallery/doc references keep working.

Run:  cd /home/leon/dual/overlaps
      /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main_native.py
Output: figures/overlaps/main/{png,svg}/fig_overlaps_main_ab_dpaact.{png,svg}

DEFAULT build (2026-08-06) = single anticipatory+action axis (choice @ bins 48-62) + robust sample-separation
units + trial-level odor-A random-slope LMM. This is the settled Fig-4 convention and matches Fig 3's axis
(fig_overlaps_manifold.py also defaults to 48-62). Pass --legacy for the old 57-63 pooled-evoked build
(→ fig_overlaps_main_ab_dpaact_legacy).

See main_panels.py for the full analysis/method docstring and the --eqnorm/--testwin/--l1/--lda/--cv10/
--ld/--ld05/--gngact/--antact/--robust flags (parsed at import from sys.argv).
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/')

# Fig 4 DEFAULTS to the settled build (2026-08-06): single anticipatory+action axis (bins 48-62) + robust
# sample-separation units + trial-level odor-A random-slope LMM. Pass --legacy for the old 57-63 pooled-evoked
# build. Injected here because main_panels parses sys.argv AT IMPORT; scoped to THIS figure so the shared
# main_panels / manifold figure defaults are unchanged.
_LEGACY = '--legacy' in sys.argv[1:]
if not _LEGACY:
    for _f in ('--antact', '--robust'):
        if _f not in sys.argv:
            sys.argv.append(_f)

# Import the module: loads the tensors, applies the uniform normalisation, and defines every panel builder
# (renders nothing). Pull ALL of its module-level names into this script's globals so the assembly reads as before.
import main_panels as _MP
globals().update({k: v for k, v in vars(_MP).items() if not k.startswith("__")})
# Keep the CANONICAL Fig-4 filename for the (new) default build; --legacy writes a distinct _legacy file.
FILE_SUF = (FILE_SUF + '_legacy') if _LEGACY else FILE_SUF.replace('_antact', '').replace('_robust', '')


if __name__ == '__main__':
    # ══════════════════════════════════════════════════════════════════════════════
    # FIGURE — push / repositioning of the working-memory state on the manifold (Fig 4)
    # ══════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(10.0, 7.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.25, 1.5], hspace=0.42, wspace=0.9,
                          left=0.06, right=0.985, top=0.93, bottom=0.06)

    def panel_letter(ax, L, x=0.008, dy=0.014):
        p = ax.get_position()
        fig.text(x, p.y1 + dy, L, fontsize=11, fontweight='bold', va='top', ha='left')

    # ── A: the no-lick push (full row: Naive traj|kde, Expert traj|kde, per-mouse depth deepening) ──
    gsB = gs[0, 0:12].subgridspec(1, 5, width_ratios=[5, 1.2, 5, 1.2, 4.4], wspace=0.3)
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
            _valsy_kde += [np.abs(np.asarray(_yt)[BINS_DELAY]) for _yt in _ys]
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
    pair_handles = [Line2D([0], [0], color=_c, lw=2.0, label=f'Sample {_l}') for _l, _p, _c in SAMPLE_TRAJ]
    axB_traj[-1].legend(handles=pair_handles, frameon=False, loc='upper right',
                        handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, fontsize=8)

    # depth panel: per-mouse late-delay choice-code depth, Naive → Expert (deepening mixed model)
    axB_sc = fig.add_subplot(gsB[0, 4])
    GX_B = (0.0, 1.0)
    for _slab, _fill in (('A', True), ('B', False)):
        P = pushB[_slab]
        for mo, xn, ye in zip(P['mice'], P['naive'], P['expert']):
            _mc = MOUSE_COLOR[mo]
            axB_sc.plot(GX_B, [xn, ye], '-', color=_mc, lw=0.7, alpha=0.5, zorder=2)
            axB_sc.scatter(GX_B[0], xn, s=30, zorder=3, linewidths=1.0, facecolors=_mc if _fill else 'w', edgecolors=_mc)
            axB_sc.scatter(GX_B[1], ye, s=30, zorder=3, linewidths=1.0, facecolors=_mc if _fill else 'w', edgecolors=_mc)
    _naive_all = np.concatenate([pushB[s]['naive'] for s in ('A', 'B')])
    _expert_all = np.concatenate([pushB[s]['expert'] for s in ('A', 'B')])
    for _xx, _vals in ((GX_B[0], _naive_all), (GX_B[1], _expert_all)):
        _mu = _vals.mean(); _se = _vals.std(ddof=1) / np.sqrt(len(_vals))
        axB_sc.plot([_xx - 0.14, _xx + 0.14], [_mu, _mu], color='k', lw=1.8, zorder=4)
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
    axB_sc.set_xlim(-0.5, 1.5); axB_sc.set_xticks(GX_B); axB_sc.set_xticklabels(['Naive', 'Expert'])
    axB_sc.set_box_aspect(1)
    axB_sc.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
    axB_sc.set_title('Choice-code depth', loc='left', fontsize=TITLE_FS)
    axB_sc.text(0.03, 0.03, f'{_statlbl}\nβ={_bpush:+.3f}, p={_ppush:.3f}',
                transform=axB_sc.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
    axB_sc.text(0.06, 0.96, '*' if _sigB else 'n.s.', transform=axB_sc.transAxes, ha='left', va='top',
                fontsize=12 if _sigB else 8, fontweight='bold', color='k' if _sigB else '0.55')
    axB_sc.legend(handles=[mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
                           mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')],
                  frameon=False, loc='upper right', fontsize=6.5, handletextpad=0.3,
                  borderaxespad=0.2, labelspacing=0.3)

    # ── B: Δdepth ↔ Δperf (Expert−Naive), A&B independent (ΔDPA | ΔGNG) ──
    gsC = gs[1, 0:6].subgridspec(1, 2, wspace=0.55)
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
                               marker='o', s=42, linewidths=1.0, zorder=5)
            ax.plot(px, py, '-', color=MOUSE_COLOR[mouse], lw=0.7, alpha=0.5, zorder=3)
        regression_band(ax, mx, my)
        ax.axhline(0, ls=':', color='k', lw=0.7); ax.axvline(0, ls=':', color='k', lw=0.7)
        ax.set_ylim(ylimC)
        sig = pv < 0.05
        ax.text(0.03, 0.03, f'per-mouse (n={n_mice})\nSpearman ρ={rho:+.2f}, p={pv:.3f}',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color='0.3')
        ax.text(0.92, 0.94, '*' if sig else 'n.s.', transform=ax.transAxes, ha='center', va='top',
                fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
        ax.set_xlabel('Δ DPA choice-code depth'); ax.set_ylabel(ylabel)
        ax.set_title(msg, loc='left', fontsize=TITLE_FS)
        ax.set_box_aspect(1)
        print(f'B[{ylabel[:6]}] per-mouse n={n_mice} Spearman ρ={rho:+.3f} p={pv:.3f}')
    _C_leg = [mlines.Line2D([0], [0], marker='o', color='k', mfc='k', ls='none', ms=5, label='sample A'),
              mlines.Line2D([0], [0], marker='o', color='k', mfc='w', ls='none', ms=5, label='sample B')]
    axC[0].legend(handles=_C_leg, frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.0),
                  ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)

    # ── C: Naive nonpaired corr-rej vs false-alarm depth, sample A | sample B ──
    axD = fig.add_subplot(gs[1, 6:9])
    GX_FACR = {'AD': (0.0, 0.8), 'BC': (1.9, 2.7)}
    for lab, samp, odor_pair, col in FA_CR_SPEC:
        xc, xe = GX_FACR[lab]; r = facr[lab]
        for ya, yb, mouse in zip(r['cr'], r['fa'], r['used']):
            mc = MOUSE_COLOR[mouse]
            axD.plot([xc, xe], [ya, yb], '-', color=mc, lw=0.7, alpha=0.5, zorder=2)
            axD.scatter(xc, ya, s=30, facecolors=mc, edgecolors=mc, linewidths=1.0, zorder=3)
            axD.scatter(xe, yb, s=30, facecolors='w', edgecolors=mc, linewidths=1.1, zorder=3)
        for xx, vals in ((xc, r['cr']), (xe, r['fa'])):
            if len(vals):
                mu = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                axD.plot([xx - 0.18, xx + 0.18], [mu, mu], color='k', lw=1.8, zorder=4)
                axD.errorbar(xx, mu, yerr=se, color='k', capsize=2.5, lw=1.2, zorder=4)
        n = len(r['cr']); d_mean = float((r['cr'] - r['fa']).mean()) if n else np.nan
        tp = float(ttest_rel(r['cr'], r['fa']).pvalue) if n >= 3 else np.nan
        sig = (tp == tp and tp < 0.05)
        axD.text((xc + xe) / 2, 0.99, f'{lab} (sample {samp})', transform=axD.get_xaxis_transform(),
                 ha='center', va='top', fontsize=7, fontweight='bold', color=col)
        axD.text((xc + xe) / 2, 0.88, '*' if sig else 'n.s.', transform=axD.get_xaxis_transform(),
                 ha='center', va='top', fontsize=12 if sig else 8, fontweight='bold', color='k' if sig else '0.55')
        axD.text((xc + xe) / 2, 0.02, f'p={tp:.3f}', transform=axD.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=6.5, color='0.3')
        print(f'C(FA/CR)[Naive {lab} sample {samp}] Δ(cr−fa)={d_mean:+.3f} paired-t p={tp:.3f} n={n}')
    axD.axhline(0, ls=':', color='0.6', lw=0.7)
    axD.set_xticks([0.0, 0.8, 1.9, 2.7])
    axD.set_xticklabels(['corr.\nrej.', 'false\nalarm', 'corr.\nrej.', 'false\nalarm'], fontsize=6.5)
    axD.set_xlim(-0.5, 3.2)
    axD.set_ylabel('choice-code depth\n← no lick               lick →', fontsize=7.5)
    axD.set_title('Naive nonpaired trials', loc='left', fontsize=TITLE_FS)
    axD.set_box_aspect(1)

    # ── D: within-task action-code d′ (Naive vs Expert) — decodability UNCHANGED ⇒ the push (A) is a
    #    POSITION shift, not a fidelity change. (Same d′ that used to live with the code-geometry panels.) ──
    axDp = fig.add_subplot(gs[1, 9:12])
    _dN = np.array([_lick_dprime(m, 'Naive') for m in ALL_MICE]); _dE = np.array([_lick_dprime(m, 'Expert') for m in ALL_MICE])
    _ok = np.isfinite(_dN) & np.isfinite(_dE)
    _av = np.concatenate([_dN[_ok], _dE[_ok]]); _lim = (min(_av.min(), -0.1), _av.max() * 1.12)
    axDp.plot(_lim, _lim, ls='--', color='0.6', lw=0.8, zorder=1)
    axDp.axhline(0, ls=':', color='0.8', lw=0.6); axDp.axvline(0, ls=':', color='0.8', lw=0.6)
    for _m, _xn, _ye in zip(np.array(ALL_MICE)[_ok], _dN[_ok], _dE[_ok]):
        axDp.scatter(_xn, _ye, s=28, facecolors=MOUSE_COLOR[_m], edgecolors=MOUSE_COLOR[_m], linewidths=0.6, zorder=4)
    _dp_t = float(ttest_rel(_dE[_ok], _dN[_ok]).pvalue); _dp_d = float((_dE[_ok] - _dN[_ok]).mean()); _dp_sig = _dp_t < 0.05
    axDp.set_xlim(_lim); axDp.set_ylim(_lim); axDp.set_box_aspect(1)
    axDp.set_title('action-code d′ (unchanged)', fontsize=TITLE_FS, loc='left')
    axDp.set_xlabel('Naive d′', fontsize=7.5); axDp.set_ylabel('Expert d′', fontsize=7.5)
    axDp.text(0.06, 0.95, '*' if _dp_sig else 'n.s.', transform=axDp.transAxes, ha='left', va='top',
              fontsize=11 if _dp_sig else 8, fontweight='bold', color='k' if _dp_sig else '0.55')
    axDp.text(0.5, 0.02, f'Δ={_dp_d:+.2f}, p={_dp_t:.3f}', transform=axDp.transAxes, ha='center', va='bottom', fontsize=6, color='0.3')
    print(f'D action-code d′ Naive={np.nanmean(_dN):+.2f} Expert={np.nanmean(_dE):+.2f} Δ={_dp_d:+.2f} p={_dp_t:.3f}')

    # ── panel letters ──
    panel_letter(axB_traj[0], 'A')
    panel_letter(axC[0], 'B')
    panel_letter(axD, 'C', x=0.5)
    panel_letter(axDp, 'D', x=0.72)

    OUT = 'figures/overlaps/main/eqnorm' if EQNORM else 'figures/overlaps/main'
    os.makedirs(f'{OUT}/png', exist_ok=True); os.makedirs(f'{OUT}/svg', exist_ok=True)
    for ext in ('png', 'svg'):
        p = f'{OUT}/{ext}/fig_overlaps_main_ab{FILE_SUF}.{ext}'
        fig.savefig(p, bbox_inches='tight')
        print('saved', os.path.abspath(p))
    plt.close(fig)
