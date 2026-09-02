"""make_ed_figures.py — compose the 9 Extended Data figures + the SI trial-count figure from the
already-rendered component PNGs (2026-09-02).

Design: every component was rendered by its own script at ~400 dpi with 6-9 pt fonts, so panels are
pasted at NATIVE pixel scale wherever possible (downscale-only when a row must fit the page width)
— a composed page is exactly as legible as its standalone parts. Captions are typeset with the
shared justified-caption machinery (pca/figcaption.py) on a matplotlib strip rendered at the same
400 dpi and pasted below the mosaic; pages wider than ~9 in get a 2-3 column caption. Outer panel
letters are bold lowercase (13 pt ~ 72 px) so they read above the components' internal uppercase
letters; blocks that are self-contained figures (ED 1/7 rows, ED 8/9) are identified by row in the
caption instead. Output is PNG only (a raster mosaic gains nothing from SVG; the share PDF is made
from the PNG).

Caption text mirrors docs/paper/results_draft.md "Extended Data Figures" (numbers verbatim from the
verified entries) — edit BOTH together. ED 3 letters here are the canon (a-g; in-text refs use 3c
per-mouse cvPCA and 3g bias cleanup).

Run:  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python make_ed_figures.py
Output: figures/ed/png/ed_fig{1..9}.png + figures/ed/png/si_trialcounts.png
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/'); sys.path.insert(0, '/home/leon/dual/pca')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from figcaption import draw_justified

DPI = 400
MARGIN = 30                 # top/right/bottom page border, px
MARGIN_L = 125              # left border — letters live HERE, never on the panels
HGAP = 105                  # between panels in a row (letters of non-first panels live in the gap)
VGAP = 90                   # between rows
LETTER_PT = 13              # outer letters: bold, above the components' internal 11pt letters
_boldfont = font_manager.findfont(font_manager.FontProperties(family=['Arial', 'DejaVu Sans'],
                                                              weight='bold'))
LFONT = ImageFont.truetype(_boldfont, int(round(LETTER_PT * DPI / 72)))


def compose_rows(rows, W):
    """rows = list of [(path, letter|None), ...]; W = usable panel width. Page = W + margins.
    Letters are drawn in the left margin (first panel) or the inter-panel gap (later panels)."""
    placed, y = [], MARGIN
    letters = []
    PW = W + MARGIN_L + MARGIN                             # full page width
    for row in rows:
        imgs = [Image.open(p) for p, _ in row]
        if len(imgs) == 1:
            s = min(1.0, W / imgs[0].width)
            sizes = [(int(round(imgs[0].width * s)), int(round(imgs[0].height * s)))]
        else:
            ars = [im.width / im.height for im in imgs]
            h = (W - HGAP * (len(imgs) - 1)) / sum(ars)
            h = min(h, min(im.height for im in imgs))          # downscale-only
            sizes = [(int(round(ar * h)), int(round(h))) for ar in ars]
        x = MARGIN_L
        rh = max(sz[1] for sz in sizes)
        for (p, letter), im, sz in zip(row, imgs, sizes):
            placed.append((im.resize(sz, Image.LANCZOS) if sz != im.size else im, x, y))
            if letter:
                letters.append((letter, x - 100, y + 2))
            x += sz[0] + HGAP
        y += rh + VGAP
    H = y - VGAP + MARGIN
    W = PW
    canvas = Image.new('RGB', (W, H), 'white')
    for im, x, yy in placed:
        canvas.paste(im, (x, yy))
    d = ImageDraw.Draw(canvas)
    for ch, x, yy in letters:
        d.text((x, yy), ch, font=LFONT, fill='black')
    return canvas


def caption_strip(paras, W):
    """Typeset `paras` justified at 7.2 pt/400 dpi across ncols columns; return a PIL image W px wide."""
    win = W / DPI
    ncols = 1 if win <= 9 else (2 if win <= 15 else 3)
    # distribute paragraphs greedily by length
    cols = [[] for _ in range(ncols)]
    tgt = sum(len(p) for p in paras) / ncols
    ci, acc = 0, 0
    for p in paras:
        cols[ci].append(p); acc += len(p)
        if acc > tgt * (ci + 1) and ci < ncols - 1:
            ci += 1
    hgen = max(2.0, sum(len(p) for p in paras) / max(1, ncols) / 100 * 0.35 + 1.0)
    fig = plt.figure(figsize=(win, hgen), dpi=DPI)
    x0f, x1f = MARGIN_L / W, 1 - MARGIN / W
    colw = (x1f - x0f - (ncols - 1) * 0.02) / ncols
    ymin = 1.0
    for i, cp in enumerate(cols):
        if not cp:
            continue
        cx0 = x0f + i * (colw + 0.02)
        yend = draw_justified(fig, cp, fontsize=7.2, x0=cx0, x1=cx0 + colw, y0=0.985)
        ymin = min(ymin, yend)
    buf = f'{OUT}/.caption_tmp.png'
    fig.savefig(buf, dpi=DPI); plt.close(fig)
    im = Image.open(buf).convert('RGB')
    arr = np.asarray(im.convert('L'))
    nz = np.where((arr < 250).any(axis=1))[0]
    im = im.crop((0, 0, im.width, (int(nz[-1]) + 20) if len(nz) else 20))
    if im.width != W:
        im = im.resize((W, int(round(im.height * W / im.width))), Image.LANCZOS)
    os.remove(buf)
    return im


def build(name, rows, paras, W):
    canvas = compose_rows(rows, W)
    cap = caption_strip(paras, canvas.width)
    page = Image.new('RGB', (canvas.width, canvas.height + cap.height + 10), 'white')
    page.paste(canvas, (0, 0)); page.paste(cap, (0, canvas.height + 10))
    out = f'{OUT}/{name}.png'
    page.save(out)
    print(f'{name}: {page.width}x{page.height}  ({page.width/DPI:.1f} x {page.height/DPI:.1f} in)')
    return out


OUT = 'figures/ed/png'
os.makedirs(OUT, exist_ok=True)
B = 'overlaps/figures/overlaps/behavior/png'
BB = 'overlaps/figures/overlaps/behavior/batch/png'
D = 'pca/figures/pseudo/dimensionality/png'
FL = 'pca/figures/pseudo/flow'
C = 'overlaps/figures/overlaps/controls/png'
M = 'overlaps/figures/overlaps/main/png'

# ── ED 1 — behaviour learning curves by cohort ────────────────────────────────
build('ed_fig1', [
    [(f'{B}/behavior_learning.png', None)],
    [(f'{B}/behavior_learning_jaws.png', None)],
    [(f'{B}/behavior_learning_chr.png', None)],
    [(f'{B}/behavior_learning_acc.png', None)],
    [(f'{B}/behavior_learning_laseron.png', None)],
], [
    'Extended Data Fig. 1 | Behavioural learning curves by cohort (companion to Fig. 1). Each row '
    'shows the behavioural learning-curve panels (A, DPA vs GNG performance; B, GNG Go vs NoGo; C, '
    'DPA paired vs unpaired; D, DPA unpaired by task context) and the LMM fixed-effect forest (E) '
    'for one cohort.',
    'Rows, top to bottom: all 9 recorded mice (laser-off trials); the Jaws subgroup (n = 5); the '
    'ChR subgroup (n = 2); the ACC-implant subgroup (n = 2); and the interleaved laser-ON trials '
    'of the 7 laser mice (5 Jaws + 2 ChR).',
    'The condition effects reproduce across splits (pooled row: GNG−DPA β = +0.037, '
    'p = 0.045; NoGo−Go +0.072; unpaired−paired −0.185; Go−DPA −0.073; LMM '
    'performance ~ condition × day, random intercept per mouse), and learning is comparable '
    'across cohorts.',
], W=8739)

# ── ED 2 — the DPA↔GNG balance is not a trade-off ─────────────────────────────
build('ed_fig2', [
    [(f'{B}/behavior_dpa_vs_gng_off.png', 'a'), (f'{B}/behavior_pareto.png', 'b')],
    [(f'{B}/behavior_dual_cost.png', 'c'), (f'{B}/behavior_dual_cost_trials.png', 'd')],
    [(f'{B}/behavior_history.png', 'e')],
    [(f'{B}/behavior_history_batch.png', 'f')],
], [
    'Extended Data Fig. 2 | The DPA↔GNG balance is not a trade-off (companion to '
    'Fig. 1e,g,h). a, Per-animal DPA vs GNG accuracy: the two co-vary in naïve mice '
    '(r ≈ 0.67) and decouple with training (Expert r = +0.10). b, Pareto front — no '
    'animal occupies the both-optimal corner. c, The fixed dual cost is small (Δ ≈ '
    '−0.03; per-mouse view) and the within-trial DPA × GNG coupling positive '
    '(Δ = +0.097, p = 0.025). d, The trial-level GEE companion: the dual-vs-pure cost '
    'is n.s. within stage, while GNG-correct trials carry better DPA memory (OR = 2.03, '
    'p = 0.001, Expert) — performing the distractor task well predicts a better, not worse, '
    'memory outcome.',
    'e, Trial-history effects (sub-panels A–H): a preceding dual trial lowers current-Go DPA '
    'accuracy (OR = 0.81, p = 0.047); GNG is history-independent. f, The blocked-design switch '
    'cost in the training batches mirrors it (into-dual OR = 0.90, p < 0.001).',
], W=7500)

# ── ED 3 — dimensionality provenance & robustness ─────────────────────────────
build('ed_fig3', [
    [(f'{D}/fig_dimensionality_main_pr.png', 'a')],
    [(f'{FL}/png/rank_sufficiency_Expert.png', 'b'), (f'{D}/fig_permouse_cvpca.png', 'c')],
    [(f'{D}/dim_all.png', 'd')],
    [(f'{D}/dim_DPA_altwin.png', 'e')],
    [(f'{D}/dim_DPA_gng.png', 'f')],
    [(f'{D}/fig_bias_cleanup_ed.png', 'g')],
], [
    'Extended Data Fig. 3 | Dimensionality: provenance and robustness (companion to Fig. 2b–d). '
    'a, The previous build of Fig. 2 (sub-panels A–D): the cvPCA method schematic, the full '
    '12-condition all-tasks spectra, and the participation-ratio ladder — memory 1.0 '
    '[1.0, 1.1] → delay 2.0 [1.6, 2.5] → decision 3.3 [2.8, 3.8] (jackknife 95% CI '
    'across mice); the full delay state’s two large dimensions are its context contrasts '
    '(distractor presence and identity). b, Reduced-rank test: held-out fit rises smoothly with no '
    'elbow at 2 (rank-2 = 62–67% of full) — the geometry, not the dynamics, is rank-2. '
    'c, The per-mouse cvPCA companion: the memory spectrum is one-dimensional animal by animal on '
    'each mouse’s own simultaneously recorded population (top-1 reliable fraction, medians '
    '0.90 naïve / 0.93 expert at mid-delay; expert memory-vs-decision Wilcoxon p = 0.047, '
    '6/7 mice; naïve directional, p = 0.22; noise-limited cells, reliable total < 5, drawn '
    'open and excluded from the test).',
    'd, The full per-fit grid for the all-tasks set (cvPCA scree, cross-validated participation '
    'ratio and shattering per window, and the per-PC η² coding matrices, Naïve and '
    'Expert). Condition-mean PCs beyond the reliable ones carry apparent η² for '
    'variables undetermined at that point in the trial — sampling noise stripped by cvPCA, '
    'not anticipatory coding (the gotcha flagged in Fig. 2d). e, Window robustness: the same DPA '
    'fit on full-delay / test windows — the DPA-delay participation ratio stays 1.0–1.1. '
    'f, The Go/NoGo cross-decode column from the DPA subspace, per window: the distractor is '
    'partially decodable from the DPA geometry (close to, but not fully, orthogonal).',
    'g, Learning removes the premature choice signal from the dual delay: in naïve mice the '
    'upcoming match/nonmatch choice is decodable from the dual delay state from early through late '
    'delay (0.64–0.66 vs shuffle ≈ 0.59), and in Expert the same signal sits at chance '
    'throughout (0.47–0.49) while post-test decoding is intact (0.96); DPA shows no such '
    'signal at either stage (control). Decodability already before the distractor marks a '
    'trial-history/bias state rather than premature deliberation. Caveats: on correct trials '
    'choice ≡ trial completion, so state-dependent selection contributes to the naïve '
    'separation; and the learning difference is pooled-level (per-mouse jackknife CI '
    '[−0.09, +0.46], n = 9). The descriptive dPCA scree, per-marginal variance, and '
    'shared-memory d′ scatter are in Extended Data Fig. 9.',
], W=4400)

# ── ED 4 — dPCA no-lick push robustness ───────────────────────────────────────
build('ed_fig4', [
    [(f'{FL}/lowrank/png/dpca_descent_rawdff.png', 'a'),
     (f'{FL}/lowrank/png/dpca_nolick_ci_qsweep.png', 'b')],
    [(f'{FL}/lowrank/png/dpca_nolick_pooledbasis.png', 'c'),
     (f'{FL}/lowrank/png/dpca_depth_vs_perf.png', 'd')],
], [
    'Extended Data Fig. 4 | The no-lick push is robust in the dPCA pipeline (corroborates '
    'Fig. 4b). a, The Naïve→Expert deepening is present in raw ΔF/F (raw vs '
    'z-scored r ≈ 0.997) — not a normalisation artifact. b, It survives removal of the '
    'condition-independent time ramp (deepening at q = 0/1/2 ramp components removed: '
    '−0.59/−0.60/−0.61). c, It holds when both stages are projected on a '
    'Naïve-defined pooled basis (8/9 mice; bootstrap CI [−0.56, −0.08]) — not '
    'a basis-rotation artifact. d, In this pipeline the depth↔accuracy link is population- '
    'rather than individual-level (per-mouse null, r = +0.46, p = 0.21); the calibrated overlaps '
    'pipeline of Fig. 4c is the individual-level assay.',
], W=5548)

# ── ED 5 — coupling/push robustness + movement control ────────────────────────
build('ed_fig5', [
    [(f'{C}/overlaps_norm_robustness.png', 'a')],
    [(f'{C}/overlaps_common_axis_control.png', 'b'), (f'{C}/overlaps_coupling_battery.png', 'c')],
    [(f'{C}/overlaps_lick_control.png', 'd')],
], [
    'Extended Data Fig. 5 | The learning coupling and push are estimator-robust, and not movement '
    '(companion to Fig. 4b,c). a, Normalisation robustness: the between-mouse '
    'Δdepth↔ΔDPA-accuracy coupling is significant under every normalisation '
    '(Spearman ρ = −0.83 to −0.90, including raw), while the within-mouse push is '
    'normalisation-sensitive. b, Fixed common axis: projecting both stages on one axis preserves '
    'the coupling (ρ = −0.72) while the push attenuates to a trend — the coupling '
    'is not a decoder-rotation artifact. c, Resampling battery for the coupling: Mundlak '
    'β = −0.041 (p = 0.006), jackknife 9/9, bootstrap CI [−1.00, −0.26], '
    'permutation p = 0.008; the ΔGNG arm is null throughout. d, Movement control: late-delay '
    'licking is rare, the choice-code depth does not track it (ρ = +0.07), and the push and '
    'coupling are unchanged with a lick covariate.',
], W=7200)

# ── ED 6 — the factorised geometry is robust ──────────────────────────────────
build('ed_fig6', [
    [('overlaps/figures/overlaps/cosine/correct/l2/png/overlaps_cosine_matrices_expert.png', 'a')],
    [(f'{C}/overlaps_mixed_selectivity.png', 'b')],
    [(f'{M}/fig_overlaps_main_ab_l1.png', 'c'), (f'{M}/fig_overlaps_main_ab_lda.png', None)],
    [(f'{C}/overlaps_codes_gng_trials.png', 'd')],
], [
    'Extended Data Fig. 6 | The factorised geometry is robust (companion to Fig. 3e and Fig. 2g). '
    'a, Cross-temporal cosine matrices: cross-code |cos| sits at the ≈0.05 chance floor at '
    'all time pairs, within-code diagonals 0.4–0.9; choice × GNG is the one '
    'least-orthogonal pair (≈0.29). b, Modular, not mixed, selectivity: per-neuron '
    'permutation tuning (sample 10 / GNG 39 / test 3 / choice 10%), with cross-variable co-tuning '
    'at chance.',
    'c, Decoder-variant robustness: the push-figure build under L1-regularised logistic '
    'regression (left) and whitened LDA (right) — the geometry and orthogonality are '
    'decoder-invariant; the push/coupling statistics are clearest under the canonical L2 '
    'logistic. d, The codes are robust to the Go/NoGo distractor: the code time courses split by '
    'Go vs NoGo trials — sample and test codes unperturbed; the action code carries the '
    'distractor lick.',
], W=6800)

# ── ED 7 — opto: chronic vs transient behaviour ───────────────────────────────
build('ed_fig7', [
    [(f'{BB}/behavior_learning_batch_ACCPrl_ctrlopto.png', 'a')],
    [(f'{BB}/behavior_learning_batch_ACC_ctrlopto.png', 'b')],
    [(f'{BB}/behavior_learning_batch_PrlACC_ctrlopto.png', 'c')],
    [(f'{B}/behavior_learning_offon_jaws.png', 'd')],
], [
    'Extended Data Fig. 7 | Chronic silencing during training impairs learning; transient '
    'silencing in trained mice spares behaviour (companion to Fig. 6b–e). a, ACC→Prl '
    'batch, control vs opto learning curves: silencing on every trial throughout training impairs '
    'DPA (β = −0.06, p = 0.009) and its unpaired trials (β = −0.12, '
    'p = 0.014). b, ACC-somata batch: null. c, Prl→ACC batch: impairs GNG. d, Transient '
    'within-mouse laser OFF vs ON learning curves in the recorded cohort (Jaws, n = 5): DPA '
    'p = 0.40, GNG p = 0.24 — acute silencing in trained animals moves the code (Fig. 6), '
    'not behaviour.',
], W=7739)

# ── ED 8 — laser ON−OFF coupling, 7 mice ─────────────────────────────────────
SL = 'overlaps/figures/overlaps/scatter_laser/png/log_generalizing_overlaps_none_l1_ratio_0.0_laser'
build('ed_fig8', [
    [(f'{SL}_targets_choice_onoff_ld_test_expert.png', 'a')],
    [(f'{SL}_targets_choice_onoff_ld_test_expert_ab.png', 'b')],
], [
    'Extended Data Fig. 8 | The acute laser ON−OFF coupling over all 7 laser mice (companion '
    'to Fig. 6g–i). a, The within-mouse laser-ON−OFF change in code depth against the '
    'change in accuracy, one point per mouse, for all 7 mice carrying interleaved laser trials '
    '(5 Jaws inhibition + 2 ChR excitation, no ACC-implant mice): the GNG arm is robust between '
    'animals (Spearman ρ = −0.90, p = 0.006, n = 7), while the DPA arm is not '
    'significant under the rank-based test (ρ = +0.55, p = 0.21). b, The same coupling '
    'with sample A and B kept as independent points (n = 14): GNG ρ = −0.60 '
    '(p = 0.024), DPA rank-n.s. Backs the Jaws-only axis choice and the alternative-n '
    'disclosure in Fig. 6.',
], W=2668)

# ── ED 9 — dPCA demixed axes story ────────────────────────────────────────────
build('ed_fig9', [
    [('pca/figures/pseudo/story/png/fig_dpca_story_main.png', None)],
], [
    'Extended Data Fig. 9 | Demixed (dPCA) axes: trajectories, mixing, and the shared plane '
    '(companion to Figs. 2 and 3). The dPCA story build: the demixing schematic, descriptive '
    'scree and marginal contrasts (time 54 / tasks 31 / sample 7 / choice 7 / test 1% of demixed '
    'variance; the scree is computed on 4 condition means inside the demixed subspace — '
    'near-circular, hence retired from Fig. 2 in favour of the cross-validated estimators); the '
    '2 × 4 Naïve/Expert trajectory grid (single-axis time courses sharpen without '
    'reorganising); the full pairwise axis-mixing slopegraph (choice–task binds, '
    '0.147 → 0.222, p < 0.001; sample–test demixes, 0.098 → 0.033, p = 0.008 '
    '— neuron-bootstrap resampling, not an across-animal test); the per-mouse shared-memory '
    'd′ scatter (Naïve +0.61 → Expert +0.54, Δ = −0.07, p = 0.91 — '
    'the memory code is present in naïve mice and preserved); and the sample × action '
    'linking plane (one shared sample axis across DPA/Go/NoGo, orthogonal to the pre-existing '
    'action axis).',
], W=3046)

# ── SI — trial counts ─────────────────────────────────────────────────────────
build('si_trialcounts', [
    [(f'{B}/behavior_trialcounts.png', None)],
], [
    'Supplementary Fig. 1 | Trial counts per mouse. Per-mouse × stage × task trial '
    'counts entering the pseudo-population (balanced by design; 5,568 laser-OFF trials total). '
    'Analysis-balanced counts, not raw behavioural trial numbers — see Methods.',
], W=3821)

print('done.')
