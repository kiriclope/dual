#!/usr/bin/env python3
"""Live figure gallery for /home/leon/dual — replaces sshfs for looking at PNGs.

Grouped, low-clutter navigation. The landing page has one PROJECT tab per top-level dir
(overlaps / pca / rnn / …). Under the active tab, figures are grouped into collapsible SECTIONS
(e.g. "overlaps / cosine", "pca / traj", "rnn / sweep_nonolick"), collapsed by default with a
figure + folder count. Expand a section to see its folder cards; click a folder to render just that
folder's figures. The filter box searches folders across ALL projects and auto-expands the matches.

Nothing heavy loads until you open a section or folder (thumbnails inside a collapsed <details> are
not fetched), so the ~5000-figure repo stays snappy. Regenerate a figure, refresh, it's there.

Usage (on the remote box, once):
    /home/leon/mambaforge/envs/dual/bin/python serve_figures.py            # port 8000
    /home/leon/mambaforge/envs/dual/bin/python serve_figures.py --port 9001

Then on your LAPTOP (once per ssh session, or via LocalForward in ~/.ssh/config):
    ssh -L 8000:localhost:8000 <this-box>
and open http://localhost:8000.

Binds 127.0.0.1 only — reachable exclusively through the SSH tunnel.
"""
import argparse, html, os
from collections import defaultdict, Counter
from urllib.parse import quote, unquote, urlsplit
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
EXTS = ('.png',)                       # PNG only; SVG twins share the folder and view rasters here
HIDE_PROJECTS = {'root', 'figures'}    # stray repo-root PNGs + the tiny top-level figures/ dir

# ── Curated paper figures ──────────────────────────────────────────────────────────────────────
# Two pinned tabs ("Main" / "Supp") at the front of the gallery. Each entry is (label, repo-relative
# PNG path). Paths point at the REAL figure files, so they always show the current regenerated PNG —
# nothing is copied. Edit these lists to curate the paper; order here = display order. A path that
# doesn't exist yet shows a greyed "missing" card (harmless placeholder).
MAIN = [
    ('Fig 1 — task & behaviour (learning curves, DPA↔GNG balance)',
     'overlaps/figures/overlaps/behavior/png/behavior_main.png'),
    # ── RESTRUCTURED 2026-08-31 ("redistribute"): Fig 2 += cross-task generalisation (E);
    #    Fig 3 = frame only (traces + storyboard + cosines); Fig 4 = the LEARNING figure
    #    (alignment ★★ + push ∗ + coupling ★ + controls). CCGP + per-mouse gen → ED supp.
    ('Fig 2 — the geometry: spectra+CI · decoding power · η² · cross-task generalisation (E) · learning-stability (F) · captioned',
     'pca/figures/pseudo/dimensionality/png/fig_dimensionality_main.png'),
    ('Fig 3 — ONE manifold  [canonical no-PCA; panel A = task-split traces DPA | Go | NoGo (ADOPTED 2026-08-31, same data as B) · storyboard · sufficiency C/D · cosines · cross-stage F]',
     'pca/figures/pseudo/dimensionality/png/fig_manifold_main.png'),
    ('Fig 4 — LEARNING · ACTION axis · pooled-evoked  [canonical: align p=.008/.004 ★★, push p=.046 ∗, coupling ρ=−0.83 p=.005 ★]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact.png'),
    ('Fig 2/3 — 3-D condition-means embedding PREVIEW  [per TASK SET (DPA | dual) × window; parallel A–B edges = the PS≈1 result; reliable-rank guard per panel; DECIDE: main or supp]',
     'pca/figures/pseudo/dimensionality/png/fig_embed_preview.png'),
    ('Fig 6 — opto · trainLD_TEST (45–59)  [canonical: trade-off r=+0.53 p=.016]',
     'overlaps/figures/overlaps/behavior/png/behavior_opto_main.png'),
]

# ── "Variants" — alternative builds of the mains (axis × normalisation × pipeline knobs).
#    Moved out of MAIN 2026-09-01 (user: one canonical card per figure in Main).
VARIANTS = [
    ('Fig 3 — PCA-20 pipeline variant  [decoder-knob robustness companion]',
     'pca/figures/pseudo/dimensionality/png/fig_manifold_main_pca20.png'),
    ('Fig 3 — ANTACT axis variant  [choice axis trained 48–62 in A+B; centred storyboard near axis-invariant; C–F unchanged]',
     'pca/figures/pseudo/dimensionality/png/fig_manifold_main_antact.png'),
    # Fig 4/5 — full grid: axis (ACTION 57–62 vs ANTICIPATORY+ACTION 48–62) × normalisation (pooled-evoked vs robust)
    ('Fig 4 — LEARNING · ACTION axis · robust  [coupling ρ=−0.83 p=.005]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact_robust.png'),
    ('Fig 4 — LEARNING · ANTACT axis · pooled-evoked  [strongest push β=−1.69 p=.003, coupling n.s.]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact_antact.png'),
    ('Fig 4 — LEARNING · ANTACT axis · robust  [push p=.032, coupling ρ=−0.68 p=.042, FA/CR p=.056 n.s. on raw]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact_antact_robust.png'),
    ('Fig 6 — opto · ACTION axis · pooled-evoked  [ΔGNG r=−0.38 p=.100]',
     'overlaps/figures/overlaps/behavior/png/behavior_opto_main_action.png'),
    ('Fig 6 — opto · ACTION axis · robust  [ΔGNG LMM p=.040]',
     'overlaps/figures/overlaps/behavior/png/behavior_opto_main_action_robust.png'),
    ('Fig 6 — opto · ANTACT axis · pooled-evoked  [ΔGNG r=−0.59 p=.006]',
     'overlaps/figures/overlaps/behavior/png/behavior_opto_main_antact.png'),
    ('Fig 6 — opto · ANTACT axis · robust  [ΔGNG r=−0.54 p=.014]',
     'overlaps/figures/overlaps/behavior/png/behavior_opto_main_antact_robust.png'),
]
SUPP = [
    ('ED — per-mouse CCGP (abstraction unchanged) + per-mouse cross-task generalisation  [left the mains 2026-08-31]',
     'pca/figures/pseudo/dimensionality/png/fig_manifold_supp.png'),
    ('ED — Fig 3 PCA-20 robustness variant (canonical = no-PCA)',
     'pca/figures/pseudo/dimensionality/png/fig_manifold_main_pca20.png'),
    # ── moved out of MAIN 2026-08-30: ED material + the Fig-4 cv10 (denoised-decoder) builds
    ('Fig 4 — push · ACTION axis · robust · cv10 ★MOST RIGOROUS  [coupling ρ=−0.833 p=.005; push n.s.]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact_robust_cv10.png'),
    ('Fig 4 — push · ANTACT axis · robust · cv10  [push p=.055, coupling p=.058, FA/CR p=.021 (raw)]',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact_antact_robust_cv10.png'),
    ('ED 9 (was Fig 2) — dPCA axes: trajectories, mixing, linking plane',
     'pca/figures/pseudo/story/png/fig_dpca_story_main.png'),
    # Fig-4 ALTERNATIVE (not adopted): renders Fig 4's push in Fig 3's memory x action frame.
    # Per-mouse calibrated pipeline — the push is NOT recoverable in the pseudo-population.
    ('Fig 4 alternative — push in the memory × action plane (per-mouse calibrated)',
     'overlaps/figures/overlaps/main/png/fig_push_bridge.png'),
    # ── SUPERSEDED Fig 3 (replaced 2026-08-30 by pca/fig_manifold_main.py) — the axis x
    # normalisation grid of the old build, kept for its panels b/d and its settled cross-decode stats
    ('old Fig 3 [superseded] — manifold · ACTION axis · pooled-evoked (was canonical)',
     'overlaps/figures/overlaps/manifold/png/fig_overlaps_manifold.png'),
    ('old Fig 3 [superseded] — manifold · ACTION axis · robust',
     'overlaps/figures/overlaps/manifold/png/fig_overlaps_manifold_robust.png'),
    ('old Fig 3 [superseded] — manifold · ANTACT axis · pooled-evoked',
     'overlaps/figures/overlaps/manifold/png/fig_overlaps_manifold_antact.png'),
    ('old Fig 3 [superseded] — manifold · ANTACT axis · robust',
     'overlaps/figures/overlaps/manifold/png/fig_overlaps_manifold_antact_robust.png'),
    # ── Fig 1 (Behaviour) ──────────────────────────────────────────────────────
    ('S1 · Learning curves — pooled 9 mice (Fig 1)',
     'overlaps/figures/overlaps/behavior/png/behavior_learning.png'),
    ('S1 · Learning curves — Jaws',
     'overlaps/figures/overlaps/behavior/png/behavior_learning_jaws.png'),
    ('S1 · Learning curves — ChR',
     'overlaps/figures/overlaps/behavior/png/behavior_learning_chr.png'),
    ('S1 · Learning curves — ACC',
     'overlaps/figures/overlaps/behavior/png/behavior_learning_acc.png'),
    ('S1 · Learning curves — laser ON',
     'overlaps/figures/overlaps/behavior/png/behavior_learning_laseron.png'),
    ('S2 · DPA↔GNG balance, not a trade-off (Fig 1h)',
     'overlaps/figures/overlaps/behavior/png/behavior_dpa_vs_gng_off.png'),
    ('S2 · Pareto front',
     'overlaps/figures/overlaps/behavior/png/behavior_pareto.png'),
    ('S2 · Dual cost + within-trial coupling',
     'overlaps/figures/overlaps/behavior/png/behavior_dual_cost.png'),
    ('S2 · Dual cost — trial-level GEE',
     'overlaps/figures/overlaps/behavior/png/behavior_dual_cost_trials.png'),
    ('S3 · Trial-history (recorded, Fig 1e/g)',
     'overlaps/figures/overlaps/behavior/png/behavior_history.png'),
    ('S3 · Switch-cost (batch)',
     'overlaps/figures/overlaps/behavior/png/behavior_history_batch.png'),
    ('G3 · Trial counts per mouse (balanced)',
     'overlaps/figures/overlaps/behavior/png/behavior_trialcounts.png'),
    # ── Fig 2 (dPCA) ───────────────────────────────────────────────────────────
    ('ED 3(a1) · Fig 2 previous build — all-tasks spectra + PR ladder (variance-weighted summary)',
     'pca/figures/pseudo/dimensionality/png/fig_dimensionality_main_pr.png'),
    ('ED 3 candidate · reliable variance per variable, TIME-RESOLVED (compression + gng phases + variables-come-online)',
     'pca/figures/pseudo/dimensionality/png/factor_variance_time.png'),
    ('ED 3 candidate · CODE TRAJECTORIES — held-out projection on each fixed Fig-2c axis across the trial',
     'pca/figures/pseudo/dimensionality/png/code_trajectories.png'),
    ('Diagnostic · choice-axis POSITION (push view) — sample-UNIFORM Expert lick-ward offset; axis is outcome-contaminated (match≡reward on correct trials) → NOT a push assay',
     'pca/figures/pseudo/dimensionality/png/code_push_traj.png'),
    ('ED 3(h) · learning removes the premature choice/bias signal from the dual delay (Naive decodable ED→LD, Expert chance; DPA control)',
     'pca/figures/pseudo/dimensionality/png/fig_bias_cleanup_ed.png'),
    ('Diagnostic · ANTICIPATORY axes (choice@LD, choice@LD→dec) — LD axis INVALID in DPA (no anticipatory direction exists); dual-Naive bias direction is REAL and VANISHES with learning; still no push',
     'pca/figures/pseudo/dimensionality/png/antact_traj.png'),
    ('Fig 2 panel-E CANDIDATE · memory manifold vs decision manifold — 12 held-out states: DPA on the sample LINE, dual in the plane; decision organizes by MATCH (task-invariant choice axis)',
     'pca/figures/pseudo/dimensionality/png/state_manifolds.png'),
    ('ED candidate · tSNE of WINDOW-averaged pseudo-trial states — unsupervised: mid-delay clusters split task→sample only (m/n coincide); decision adds the match split',
     'pca/figures/pseudo/dimensionality/png/tsne_window_states.png'),
    ('ED candidate · the THREE task axes in one frame (sample × distractor × choice), 12 held-out states MD→decision — DPA midway on the distractor axis; the decision = a choice-axis displacement',
     'pca/figures/pseudo/dimensionality/png/state_axes3d.png'),
    ('ED candidate · states in the fixed-axes plane, FOUR windows (ED→MD→LD→decision, held-out ellipse view) — task bands ABSENT at ED, appear at MD, choice claims the axis at decision',
     'pca/figures/pseudo/dimensionality/png/axes_plane.png'),
    ('ED candidate · TRAJECTORIES in the fixed-axes plane (held-out; DPA | Go | NoGo × Naive | Expert) — L-shaped paths: out along sample, hold, lick-axis sweep at test; Go bumps at the cue',
     'pca/figures/pseudo/dimensionality/png/axplane_traj.png'),
    ('ED candidate · window-CENTROID trajectories E→M→L→D connected (held-out, one panel per set) — DPA: hold on the sample axis then the decision FORK; dual: Go climbs / NoGo dips the lick axis in-delay, then forks',
     'pca/figures/pseudo/dimensionality/png/axes_plane_centers.png'),
    ('BRIDGE candidate (Fig 2→4) · the PUSH in the memory × action plane — per-mouse Naive→Expert arrows on Fig 4 calibrated depths: sample A slides into the no-lick region (Δ −1.4), B unmoved',
     'overlaps/figures/overlaps/main/png/fig_push_bridge.png'),
    ('Diagnostic · tSNE maps + fixed-axis overlays — axes drawn only where the map carries them (R²≥0.25): the sample axis survives in 1 of 4 maps → tSNE cannot display these axes; superseded by the axes-plane figure',
     'pca/figures/pseudo/dimensionality/png/scatter_axes.png'),
    ('Diagnostic · CONTINUOUS embedding trajectories (smoothed tSNE perp-80 | Isomap) — continuity achieved but still 12 separate arms + chord artifacts; Isomap ramp-dominated — linear displays remain the right tool',
     'pca/figures/pseudo/dimensionality/png/tsne_traj_continuous.png'),
    ('Diagnostic · tSNE manifold + Two-NN ID — tSNE fragments the manifold into per-condition tubes (anti-message); Two-NN returns the NOISE dimension (~46) → why cvPCA is needed; NOT for the main',
     'pca/figures/pseudo/dimensionality/png/manifold_embed.png'),
    ('S4 · Rank sufficiency — no elbow at 2 (Fig 2b)',
     'pca/figures/pseudo/flow/png/rank_sufficiency_Expert.png'),
    ('S4 · Task-manifold rank-2',
     'pca/figures/pseudo/flow/png/rank_task_manifold.png'),
    # (S5 demixed axes — loadings/mixing/EVR — CUT 2026-08-03 as redundant with Fig 2e + ED6 cosine.)
    ('S6 · Push — raw ΔF/F deepening',
     'pca/figures/pseudo/flow/lowrank/png/dpca_descent_rawdff.png'),
    ('S6 · Push — CI/time-ramp q-sweep',
     'pca/figures/pseudo/flow/lowrank/png/dpca_nolick_ci_qsweep.png'),
    ('S6 · Push — pooled-basis',
     'pca/figures/pseudo/flow/lowrank/png/dpca_nolick_pooledbasis.png'),
    ('S6 · Depth↔performance null (n=9)',
     'pca/figures/pseudo/flow/lowrank/png/dpca_depth_vs_perf.png'),
    # (S7 dPCA flows & bistability REMOVED from the paper 2026-08-03 — kept as "extra"
    #  in the gallery pca tab, not part of the submission.)
    # ── Fig 3 (Overlaps) ───────────────────────────────────────────────────────
    ('S8 · Coupling — normalisation robustness (Fig 3d)',
     'overlaps/figures/overlaps/controls/png/overlaps_norm_robustness.png'),
    ('S8 · Common-axis / decoder-sharpening control',
     'overlaps/figures/overlaps/controls/png/overlaps_common_axis_control.png'),
    ('S8 · Coupling resampling battery',
     'overlaps/figures/overlaps/controls/png/overlaps_coupling_battery.png'),
    ('S9 · Movement / anticipatory-lick control',
     'overlaps/figures/overlaps/controls/png/overlaps_lick_control.png'),
    ('S10 · Cosine orthogonality matrices (Fig 3a,b)',
     'overlaps/figures/overlaps/cosine/correct/l2/png/overlaps_cosine_matrices_expert.png'),
    ('S11 · Mixed vs modular selectivity',
     'overlaps/figures/overlaps/controls/png/overlaps_mixed_selectivity.png'),
    ('S12 · Decoder variant — L1 (lasso)',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_l1.png'),
    ('S12 · Decoder variant — LDA (whitened)',
     'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_lda.png'),
    ('S13 · Codes robust to Go/NoGo distractor',
     'overlaps/figures/overlaps/controls/png/overlaps_codes_gng_trials.png'),
    # ── Fig 6 (Opto) ───────────────────────────────────────────────────────────
    ('S14 · Batch ACC→Prl controls',
     'overlaps/figures/overlaps/behavior/batch/png/behavior_learning_batch_ACCPrl_ctrlopto.png'),
    ('S14 · Batch ACC (null)',
     'overlaps/figures/overlaps/behavior/batch/png/behavior_learning_batch_ACC_ctrlopto.png'),
    ('S14 · Batch Prl→ACC (GNG)',
     'overlaps/figures/overlaps/behavior/batch/png/behavior_learning_batch_PrlACC_ctrlopto.png'),
    ('S15 · Transient laser OFF/ON — spared',
     'overlaps/figures/overlaps/behavior/png/behavior_learning_offon.png'),
    ('S16 · Laser ON−OFF coupling — 7 mice',
     'overlaps/figures/overlaps/scatter_laser/png/log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice_onoff_ld_test_expert_ab.png'),
    # (S17 d′-spared standalone — CUT 2026-08-03: already shown as main Fig 6k,l.)
]

# ── "Extra" — parked, not in the paper ─────────────────────────────────────────
# Flow-field / low-rank attractor-dynamics work, REMOVED from the main figures and supplements
# (2026-08-03) but kept here for reference. Not part of the submission.
EXTRA = [
    ('Extra — overlaps flow story (per-regime decoders)',
     'overlaps/figures/overlaps/story/png/fig_overlaps_story_main.png'),
    ('Extra — dPCA low-rank flows (partial pooling)',
     'pca/figures/pseudo/flow/lowrank/png/dpca_lowrank_partial_Expert.png'),
    ('Extra — dPCA low-rank flows (shared)',
     'pca/figures/pseudo/flow/lowrank/png/dpca_lowrank_shared_Expert.png'),
    ('Extra — no-lick flow push (Expert)',
     'pca/figures/pseudo/flow/lowrank/png/dpca_lowrank_independent_Expert_push.png'),
    ('Extra — bistability survey (4/9)',
     'pca/figures/pseudo/flow/bistability_summary.png'),
    ('Extra — defensible manifold (measured vs modelled)',
     'pca/figures/pseudo/flow/wm_manifold_defensible.png'),
    ('Extra — slow-manifold → double-well under CI',
     'pca/figures/pseudo/flow/slowmanifold_test_pooled.png'),
    ('Extra — ★ Fig-2 FOLD PANEL: cross-task generalization (balanced acc., N vs E, colorbar@0.5)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_foldpanel_acc.png'),
    ('Extra — CCGP abstraction (pseudo-population, correct trials) ★',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_pseudo.png'),
    ('Extra — CCGP abstraction (per-mouse n=9 companion / stat)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp.png'),
    ('Extra — generalization index Naive vs Expert (d′, bootstrap CI + Δ test) ★',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_summary.png'),
    ('Extra — generalization index Naive vs Expert (balanced accuracy, bootstrap CI + Δ test)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_acc_summary.png'),
    ('Extra — cross-task d′ matrices (pseudo-pop, Naive vs Expert, raw + ÷within)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo.png'),
    ('Extra — cross-task balanced-acc. matrices (pseudo-pop, Naive vs Expert, raw + ÷within)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_acc.png'),
    ('Extra — [sample@TEST] generalization index N vs E (d′, bootstrap CI + Δ test)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_test_summary.png'),
    ('Extra — [sample@TEST] generalization index N vs E (balanced accuracy, CI + Δ test)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_test_acc_summary.png'),
    ('Extra — [sample@TEST] cross-task d′ matrices (pseudo-pop, N vs E, raw + ÷within)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_test.png'),
    ('Extra — [sample@TEST] cross-task balanced-acc. matrices (pseudo-pop, N vs E)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_pseudo_test_acc.png'),
    ('Extra — cross-task generalization matrices (per-mouse)',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices.png'),
    ('Extra — cross-task generalization matrices @ TEST epoch',
     'overlaps/figures/overlaps/ccgp/png/overlaps_ccgp_matrices_test.png'),
]


def scan():
    """dir_rel -> [(name, mtime)] for every figure directory under BASE."""
    groups = defaultdict(list)
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'svg']
        for f in files:
            if f.lower().endswith(EXTS):
                try:
                    mt = os.stat(os.path.join(root, f)).st_mtime
                except OSError:
                    continue
                groups[os.path.relpath(root, BASE)].append((f, mt))
    return groups


def dedup_parts(dir_rel):
    """Path parts with noise ('figures'/'png') dropped and adjacent repeats collapsed."""
    parts = [p for p in dir_rel.split(os.sep) if p not in ('figures', 'png')]
    return [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]


def project_of(dir_rel):
    p = dir_rel.split(os.sep)[0]
    return 'root' if p == '.' else p


def label(dir_rel):
    """Friendly folder label: last 2-3 meaningful path parts."""
    dedup = dedup_parts(dir_rel)
    return ' / '.join(dedup[-3:]) if dedup else dir_rel


def build_sections(groups):
    """project -> {section_label -> [dir_rel, …]}, ordered by recency.

    Section token = first meaningful part after the project name, descending one extra level for a
    project whose folders overwhelmingly share that part (e.g. pca/pseudo/* -> group by 'traj')."""
    by_proj = defaultdict(list)
    for d in groups:
        proj = project_of(d)
        if proj not in HIDE_PROJECTS:
            by_proj[proj].append(d)

    def newest(dir_list):
        return max(m for d in dir_list for _, m in groups[d])

    tree = {}
    for proj in sorted(by_proj, key=lambda p: newest(by_proj[p]), reverse=True):
        dirs = by_proj[proj]
        idx1 = Counter(dedup_parts(d)[1] for d in dirs if len(dedup_parts(d)) >= 2)
        descend = bool(idx1) and max(idx1.values()) / len(dirs) >= 0.6
        secs = defaultdict(list)
        for d in dirs:
            dp = dedup_parts(d)
            if descend and len(dp) >= 3:
                tok = dp[2]
            elif len(dp) >= 2:
                tok = dp[1]
            elif dp:
                tok = dp[-1]
            else:
                tok = proj
            secs[f'{proj} / {tok}'].append(d)
        tree[proj] = dict(sorted(secs.items(), key=lambda kv: newest(kv[1]), reverse=True))
    return tree


HEAD = """<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1113;color:#dde}
a{color:inherit;text-decoration:none}
header{position:sticky;top:0;background:#16191ccc;backdrop-filter:blur(8px);padding:11px 16px;
  border-bottom:1px solid #262b30;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600;color:#fff;letter-spacing:.01em}
header h1 .crumb{color:#5b8;font-weight:500}
#q{flex:1;min-width:180px;padding:7px 10px;background:#1c2024;border:1px solid #333a40;border-radius:6px;color:#eee;font-size:14px}
#q:focus{outline:none;border-color:#3a8}
#count{color:#7a8590;font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}
.back{padding:6px 11px;background:#1c2024;border:1px solid #333a40;border-radius:6px;color:#cde;font-size:13px}
.back:hover{border-color:#3a8}
.pills{display:flex;gap:8px;flex-wrap:wrap;padding:12px 16px 2px}
.pill{padding:6px 13px;border-radius:999px;background:#1c2024;border:1px solid #333a40;color:#ccd;cursor:pointer;font-size:13px}
.pill:hover{border-color:#3a8}
.pill.on{background:#2f9e6f;border-color:#2f9e6f;color:#04120a;font-weight:600}
.pill .pc{opacity:.6;font-size:11px;font-variant-numeric:tabular-nums}
.sections{display:flex;flex-direction:column;gap:8px;padding:12px 16px 24px;max-width:1400px}
details.sec{background:#141719;border:1px solid #24292e;border-radius:9px;overflow:hidden}
details.sec[open]{border-color:#2c343a}
details.sec>summary{list-style:none;cursor:pointer;padding:11px 14px;display:flex;align-items:center;gap:10px}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary:hover{background:#171b1e}
.chev{color:#5b8;transition:transform .12s;display:inline-block;width:12px}
details.sec[open] .chev{transform:rotate(90deg)}
.stitle{color:#6ec7a0;font-weight:600}
.scount{color:#7a8590;font-size:12px;font-variant-numeric:tabular-nums;margin-left:auto;white-space:nowrap}
.secbody{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;padding:4px 14px 16px}
.figs{display:grid;gap:14px;padding:16px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.pinned .figs{padding:16px 16px 24px;max-width:1400px}
.card{margin:0;background:#171a1d;border:1px solid #24292e;border-radius:9px;overflow:hidden;display:flex;flex-direction:column}
.card:hover{border-color:#356}
.card img{width:100%;display:block;background:#fff;min-height:60px}
.folder .thumb{height:130px;background:#fff center/cover no-repeat}
.card.missing{opacity:.5}
.card.missing .thumb.miss{height:130px;background:#1c2024;display:flex}
.empty{padding:40px 20px;color:#7a8590;font-size:13px;text-align:center}
figcaption,.meta{padding:8px 11px;display:flex;flex-direction:column;gap:2px;font-size:12px}
.lbl{color:#6ec7a0;font-weight:600}
.nm{color:#aab3bb;word-break:break-all}
.sub{color:#6b757e;font-size:11px;font-variant-numeric:tabular-nums}
time{color:#6b757e;font-size:11px}
.hidden{display:none}
</style>"""


def folder_card(d, items):
    newest_name, newest_mt = max(items, key=lambda t: t[1])
    cover = quote(f'{d}/{newest_name}') + f'?v={int(newest_mt)}'       # mtime busts the browser cache
    return (
        f'<a class="card folder" href="/view?d={quote(d)}" data-h="{html.escape((label(d)+" "+d).lower())}">'
        f'<div class="thumb" style="background-image:url(/{cover})"></div>'
        f'<div class="meta"><span class="lbl">{html.escape(label(d))}</span>'
        f'<span class="sub">{len(items)} figures · <time data-ts="{newest_mt:.0f}"></time></span></div></a>')


def figure_card(lbl, rel):
    """A curated single-figure card pointing at the real PNG (greyed placeholder if missing)."""
    path = os.path.join(BASE, rel)
    dh = html.escape((lbl + ' ' + rel).lower())
    if not os.path.isfile(path):
        return (f'<figure class="card fig missing" data-h="{dh}"><div class="thumb miss"></div>'
                f'<figcaption><span class="lbl">{html.escape(lbl)}</span>'
                f'<span class="nm">missing · {html.escape(rel)}</span></figcaption></figure>')
    mt = os.stat(path).st_mtime
    src = quote(rel) + f'?v={int(mt)}'                                # mtime busts the browser cache
    return (f'<figure class="card fig" data-h="{dh}">'
            f'<a href="/{src}" target="_blank"><img loading="lazy" src="/{src}"></a>'
            f'<figcaption><span class="lbl">{html.escape(lbl)}</span>'
            f'<span class="nm">{html.escape(os.path.basename(rel))}</span>'
            f'<time data-ts="{mt:.0f}"></time></figcaption></figure>')


def pill(name, count, active):
    return (f'<button class="pill{" on" if active else ""}" data-proj="{html.escape(name)}">'
            f'{html.escape(name)} <span class=pc>{count}</span></button>')


def curated_panel(name, entries, active):
    inner = (f'<div class="figs">{"".join(figure_card(l, r) for l, r in entries)}</div>' if entries
             else '<div class="empty">No figures yet — add (label, path) lines to '
                  f'{name.upper()} at the top of serve_figures.py.</div>')
    return (f'<div class="project pinned{"" if active else " hidden"}" data-proj="{html.escape(name)}">'
            f'{inner}</div>')


def scanned_panel(proj, secs, groups, active):
    blocks = []
    for slabel, dirs in secs.items():
        dirs = sorted(dirs, key=lambda d: max(m for _, m in groups[d]), reverse=True)
        newest_mt = max(m for d in dirs for _, m in groups[d])
        n_fig = sum(len(groups[d]) for d in dirs)
        cards = ''.join(folder_card(d, groups[d]) for d in dirs)
        short = slabel.split(' / ', 1)[1] if ' / ' in slabel else slabel
        blocks.append(
            f'<details class="sec" data-proj="{html.escape(proj)}">'
            f'<summary><span class="chev">▸</span><span class="stitle">{html.escape(short)}</span>'
            f'<span class="scount">{n_fig} figs · {len(dirs)} folders · '
            f'<time data-ts="{newest_mt:.0f}"></time></span></summary>'
            f'<div class="secbody">{cards}</div></details>')
    return (f'<div class="project{"" if active else " hidden"}" data-proj="{html.escape(proj)}">'
            f'<div class="sections">{"".join(blocks)}</div></div>')


def render_index(groups):
    tree = build_sections(groups)
    pills, panels, first = [], [], True
    for name, entries in (('Main', MAIN), ('Variants', VARIANTS),  # pinned curated tabs, first
                          ('Supp', SUPP), ('Extra', EXTRA)):
        pills.append(pill(name, len(entries), first))
        panels.append(curated_panel(name, entries, first))
        first = False
    for proj, secs in tree.items():                               # scanned project tabs
        pills.append(pill(proj, sum(len(d) for d in secs.values()), False))
        panels.append(scanned_panel(proj, secs, groups, False))
    body = (f'<header><h1>dual figures</h1>'
            f'<input id=q placeholder="filter figures / folders across all tabs (e.g. cosine, opto, sweep)" autofocus>'
            f'<span id=count></span></header>'
            f'<div class="pills">{"".join(pills)}</div>'
            f'{"".join(panels)}')
    return page(body, total=0, unit='', index=True)


def render_folder(d, items):
    items = sorted(items, key=lambda t: t[1], reverse=True)
    cards = []
    for name, mt in items:
        src = quote(f'{d}/{name}') + f'?v={int(mt)}'                   # mtime busts the browser cache
        cards.append(
            f'<figure class=card data-h="{html.escape(name.lower())}">'
            f'<a href="/{src}" target="_blank"><img loading="lazy" src="/{src}"></a>'
            f'<figcaption><span class="nm">{html.escape(name)}</span>'
            f'<time data-ts="{mt:.0f}"></time></figcaption></figure>')
    body = (f'<header><a class="back" href="/">← projects</a>'
            f'<h1><span class="crumb">{html.escape(label(d))}</span></h1>'
            f'<input id=q placeholder="filter figures in this folder" autofocus>'
            f'<span id=count></span></header>'
            f'<main class="figs" id=grid>{"".join(cards)}</main>')
    return page(body, total=len(items), unit='figures', index=False)


def page(body, total, unit, index):
    if index:
        js = """
const pills=[...document.querySelectorAll('.pill')],
      panels=[...document.querySelectorAll('.project')],
      secs=[...document.querySelectorAll('details.sec')],
      cards=[...document.querySelectorAll('.card')],
      q=document.getElementById('q'), cnt=document.getElementById('count');
const fmt=ts=>{const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');
  return (d.getMonth()+1)+'/'+d.getDate()+' '+p(d.getHours())+':'+p(d.getMinutes());};
document.querySelectorAll('time[data-ts]').forEach(t=>t.textContent=fmt(+t.dataset.ts));
function activeProj(){return (pills.find(b=>b.classList.contains('on'))||pills[0]).dataset.proj;}
function setProject(p){
  pills.forEach(b=>b.classList.toggle('on',b.dataset.proj===p));
  panels.forEach(el=>el.classList.toggle('hidden',el.dataset.proj!==p));}
function shown(){return cards.filter(c=>!c.classList.contains('hidden')&&
  !c.closest('.project').classList.contains('hidden')).length;}
function apply(){
  const term=q.value.trim();
  if(!term){
    cards.forEach(c=>c.classList.remove('hidden'));
    secs.forEach(s=>{s.classList.remove('hidden');s.open=false;});
    setProject(activeProj());
    cnt.textContent=shown()+' figures';
    return;}
  let re=null;try{re=new RegExp(term,'i');}catch(e){}
  const t=term.toLowerCase();
  cards.forEach(c=>{const h=c.dataset.h||'';const ok=re?re.test(h):h.includes(t);
    c.classList.toggle('hidden',!ok);});
  secs.forEach(s=>{const vis=[...s.querySelectorAll('.card')].some(c=>!c.classList.contains('hidden'));
    s.classList.toggle('hidden',!vis);s.open=vis;});
  panels.forEach(el=>{const vis=[...el.querySelectorAll('.card')].some(c=>!c.classList.contains('hidden'));
    el.classList.toggle('hidden',!vis);});
  cnt.textContent=shown()+' matches';}
pills.forEach(b=>b.addEventListener('click',()=>{q.value='';setProject(b.dataset.proj);apply();}));
q.addEventListener('input',apply);apply();"""
    else:
        js = f"""
const items=[...document.querySelectorAll('#grid>*')], q=document.getElementById('q'),
      cnt=document.getElementById('count');
const fmt=ts=>{{const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');
  return (d.getMonth()+1)+'/'+d.getDate()+' '+p(d.getHours())+':'+p(d.getMinutes());}};
document.querySelectorAll('time[data-ts]').forEach(t=>t.textContent=fmt(+t.dataset.ts));
function apply(){{let re=null;try{{re=new RegExp(q.value.trim(),'i');}}catch(e){{}}
  let n=0;items.forEach(el=>{{const h=el.dataset.h||'';
    const ok=!q.value.trim()||(re?re.test(h):h.includes(q.value.toLowerCase()));
    el.classList.toggle('hidden',!ok);if(ok)n++;}});
  cnt.textContent=n+' / {total} {unit}';}}
q.addEventListener('input',apply);apply();"""
    return f"<!doctype html><html><head>{HEAD}<title>dual figures</title></head><body>{body}<script>{js}</script></body></html>"


class Handler(SimpleHTTPRequestHandler):
    def _send(self, body):
        b = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlsplit(self.path)
        if u.path in ('/', '/index.html'):
            self._send(render_index(scan()))
            return
        if u.path == '/view':
            d = unquote(u.query[2:]) if u.query.startswith('d=') else ''
            self._send(render_folder(d, scan().get(d, [])))
            return
        super().do_GET()

    def translate_path(self, path):
        rel = unquote(urlsplit(path).path).lstrip('/')
        return os.path.join(BASE, *[p for p in rel.split('/') if p not in ('', '.', '..')])

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    g = scan()
    t = build_sections(g)
    print(f'serving {sum(len(v) for v in g.values())} figures in {len(g)} folders '
          f'({len(t)} projects) from {BASE}')
    print(f'  remote:  http://127.0.0.1:{a.port}  (localhost only)')
    print(f'  laptop:  ssh -L {a.port}:localhost:{a.port} <this-box>  then open http://localhost:{a.port}')
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
