# Honest dimensionality — cvPCA + shattering + PC coding (Fig 2)

**The non-circular dimensionality analysis of the dual-task pseudo-population, and — since 2026-08-10 —
main Fig 2** (`pca/fig_dimensionality_main.py`, which replaced `fig_dpca_story_main.py`; that dPCA build
is now ED Fig 9). **Fig 2 is built around ONE message: "one dedicated axis per task variable — the
working memory is a line."** Panels (the DECODE build, adopted 2026-08-10): **a** trial-timeline +
split-half cvPCA schematic (mid-delay bracket at 5.5–6.3 s) · **b** 2×2 per-set reliable spectra —
DPA | dual (columns) × mid-delay | decision (rows), Naive + Expert overlaid, leave-one-mouse-out
jackknife 95% CIs (`SPEC_JK`), xlim 6 components, LINEAR fraction axis (log axis rejected) ·
**c** per-variable DECODING POWER (held-out pseudo-trials along each variable's demixed axis vs
shuffle-null 95th pct, `DPCA_COUNT`; hatched gng× bar = Go/NoGo cross-decoded from the DPA subspace,
`DPA_GNG_C`) · **d** η² PC-coding matrices, DPA-first, PC1–4, mid-delay + decision, with the boxed
gng× cross-decode column on DPA (`DPA_GNG`). [Historical: the pre-adoption composition — three
all-tasks mini spectra + PR bars 1→2→3 (`PR_JK`) — is the `--pr` legacy build, kept as the ED/caption
source for the PR numbers.] Off-message and therefore OUT of the
figure: shattering (→ ED 3 + cited from Results §3, where it pairs with CCGP), dPCA trajectories /
axis-mixing / linking plane (→ ED 9), Naive matrices + per-task-set fits (→ ED 3), PR bars (→ ED).

## Why it exists
Fig 2's old dPCA scree was near-tautological: computed on 4 condition-means inside the demixed
sample+sample:test subspace, so "top-2 ≈ 94% / PR ≈ 2.2" was ~built in. These methods measure
dimensionality on the raw pseudo-population with cross-validation, so only structure that replicates
across independent trial halves counts.

## Scripts & data flow (all under `pca/`, run from `pca/`)
- `exp_dimensionality.py` — the compute (~12 min, reloads the 20 GB `X_all_nan_`): cvPCA (A), shattering
  (B), per-variable coding (C), PC×factor η² (D). MERGE-dumps into
  `figures/pseudo/dimensionality/results.pkl`; saves a quick-look `dimensionality_qc.png` only.
- `exp_dimensionality_fits.py [--altwin]` — per-(task-set × window × stage) fits (DPA/dual/all ×
  delay/decision/delay+dec). Caches window matrices in `fits_inputs[_altwin].pkl` (~700 MB) so re-runs
  skip the 20 GB reload; merges `FITDATA` into `results[_altwin].pkl`.
- `exp_dimensionality_ci.py` — hardening (2026-08-10, cache-only): shattering over **all 462**
  dichotomies + per-resample CIs (`SD_FULL`), and split-level PR CIs (`PR_CI` — DEMOTED to a
  trial-split stability check; do NOT use as the figure's error bar, it is anti-conservative).
- `exp_dimensionality_jk.py` — **the Fig-2c error bars** (`PR_JK`): leave-one-mouse-out jackknife of
  the averaged-spectrum PR (mice = the exchangeable unit; neurons partition by mouse), 95% CI
  clipped at the PR floor of 1. Values: memory 1.0 [1.0, 1.1] Expert / 1.2 [1.0, 1.8] Naive · delay
  2.0 [1.6, 2.5] / 2.0 [1.4, 2.7] · decision 3.3 [2.8, 3.8] / 3.3 [2.3, 4.3]. The Expert memory CI
  hugging the floor is itself informative: every leave-out stays ≈1.
- `exp_dpca_count.py` — **significant-axis COUNT** (2026-08-10, cache-only ~2 min, from
  `fits_inputs.pkl`): Kobak-style dPCA-marginalization significance per (set × window ED/MD/LD/TEST/
  decision × stage) — each design contrast's demixed axis (from leakage-free train condition-means),
  single held-out pseudo-trials decoded along it, significant if balanced accuracy > 95th pct of a
  within-mouse label-shuffle null. Merges `DPCA_COUNT` into `results.pkl`. The amplitude-free COUNT
  companion to the variance-weighted PR (built after the user challenged PR as a metric for
  outside-response-window comparisons; window-averaged binary factors → each marginalization is
  rank-1, so "components per marginalization" = one test per contrast). KEY NUMBERS: the C states are
  memory 1 (sample) → delay 3 (sample+tasks+gng) → decision 5 (all), IDENTICAL Naive & Expert; dual
  Expert ladder ED 1 → MD 2 (sample+gng) → LD 2 → TEST 4 agrees cell-by-cell with the ≥3σ
  factor-variance count (incl. the Naive-only future-'choice' axis at ED/MD/LD, acc ≈ 0.64–0.66);
  DPA 'test' n.s. at the narrow test window 57–59 (0.62–0.63 vs n95 0.65) but sig at decision 57–65.
- `exp_cdec_support.py` — variant support (2026-08-10, cache-only ~3 min): **`SPEC_JK`** =
  leave-one-mouse-out jackknife 95% CIs (+ the point fractions, one source) for the B spectra
  (DPA/dual × md/decision), **`SPEC_NULL`** (added 2026-08-12) = within-mouse label-shuffle null
  spectra for B, ÷ the REAL positive total (self-normalised null would be meaningless; values ≤0.026
  everywhere), and **`DPA_GNG_C`** = the panel-C "gng ×" bar: Go/NoGo cross-decoded from
  the DPA-STATE SUBSPACE (top-3 DPA condition-mean PCs, LDA, held-out, shuffle null, 100 shuffles). KEY: at
  MID-delay the DPA subspace carries the distractor code only WEAKLY — Expert 0.61* / Naive 0.47 n.s.
  (decision 0.63*/0.61*); the older "~0.7 from PC2" figure came from the LATE-delay window and is
  consummatory-inflated.
- Renderers (no recompute, read `results.pkl`): `fig_dimensionality_main.py` — **the DECODE build IS
  main Fig 2 (ADOPTED 2026-08-10)**: B = 2×2 per-set cvPCA spectra (SPEC_JK: point + leave-one-mouse-out
  jackknife 95% CI from one source; common 6-component x-axis), C = per-variable decoding power
  (DPCA_COUNT + the hatched DPA "gng ×" from DPA_GNG_C), D = η² matrices DPA-first, PC1–4 both sets
  (DPA PC4 = degenerate ~0% row, shown for symmetry) + boxed gng × column (DPA_GNG).
  **Impact pass (2026-08-12, from the external content review):** A brackets carry window labels
  (5.5–6.3 s / post-test); B adds the SPEC_NULL grey-dashed null floor + in-panel geometry callouts
  with cartoons (the line/plane cartoon glyphs were REMOVED 2026-09-01, text callouts kept)
  ("1 reliable axis — the sample line" A–B line glyph; "+ 1 distractor axis (0.07)"
  plane glyph with orange gng arrow; "≈3 reliable axes" on the decision row; legend moved to
  DPA-decision, center right); C gets an in-panel 2×2 legend (Expert bar / Naive circle / null 95% /
  gng× ← DPA PCs) + the orange "weak transfer (dual gng = 1.0)" annotation on the md gng× bar;
  D FADES rows beyond B's reliable rank (white veil + grey text/ticklabels + dashed boundary;
  rank = # leading comps with jackknife CI lo > 1%: DPA-md 1, dual-md 2, both decisions 3 — the
  `_rank_b()` helper; the boxed gng× column is EXEMPT from the veil so C's cross-decode mechanism
  stays visible) and the gng× header now reads "gng × (cross-dec)".
  **SUBMISSION FORM (2026-08-12, Leon: "this is a paper figure, there should be no footnote, no
  title … keep that for the methods"):** the `fig.suptitle` and BOTH footnote blocks are GONE, as is
  C's interpretive panel title — all that prose now lives in `docs/paper/methods_notes.md` (η²
  chance = 1/3 + Beta(½,1) null, the fade criterion, the mid-delay sensory-tail caveat, the gng×
  definition, the † bias state). What STAYS on the figure: bold panel letters, descriptive panel
  titles, axis labels, in-panel legends/callouts. Layout consequences (don't regress): figsize
  10.6×6.2 and outer `wspace=1.0` — each block's gap must hold a shared y-label + tick labels
  (≥ ~0.035 fig width) or row 1 collides; B y-ticks PINNED to [0, 0.5, 1] (taller axes auto-add
  0.25 steps that hit the y-label); A's decision bracket label kept SHORT ("decision state", no
  "(post-test)") for the same reason; C's rotated tick reads "gng cross" NOT "gng ×" (a rotated ×
  renders as +). Grid = DPA vs
  dual × MID-delay (bins 36–38, pre-cue/pre-lick) vs decision — **all of B–D on the same windows since
  the MD-η² recompute (`exp_pceta_md.py` → FITDATA[(set,'md',stage)] + DPA_GNG[('md',stage)]; Expert:
  DPA PC1 sample .93 41%, gng× [0,.16,.21,0]; dual PC1 gng .98 37%, PC2 sample .91 14%)**;
  the 'all tasks' set is OFF the main figure. NOTE the ED full η² grid
  (`plot_dimensionality_main.py`) still renders the LATE-delay matrices — regenerate at MD before
  submission if window consistency with the main is wanted there too. **`--pr` renders the PREVIOUS build**
  (`fig_dimensionality_main_pr.png`: all-tasks spectra + PR bars + jackknife CIs, dual-first D) —
  now ED 3(a1). A dot-strip-over-PR-bars variant was built first and REJECTED — don't rebuild.
  Draft §2, ED 3 list and `methods_notes.md` rewritten for the adoption (2026-08-10),
  `plot_dimensionality_main.py` (curated dual-vs-DPA composite `dimensionality.png`),
  `plot_dimensionality_fits.py [--gng] [--altwin]` (`dim_{DPA,dual,all}*.png`),
  `plot_dimensionality_scree.py`, `fig_bias_cleanup_ed.py` (**ED 3(h)**, cache-only render from
  ANTACT_TRAJ + DPCA_COUNT: learning removes the dual-Naive premature-choice/bias signal — Naive
  choice decodable 0.64–0.66 ED→LD vs Expert chance 0.47–0.49, both * at decision; separation trace
  on the reward-free LD-defined axis climbs to ~+2 z in Naive, flat in Expert; DPA control flat;
  decodable already at ED ⇒ trial-history/bias state, not deliberation).

## Methods (one paragraph each)
**cvPCA (Stringer 2019).** Each (mouse, condition) correct-trial pool is split into two disjoint halves →
two independent condition-mean pseudo-populations (neurons partition disjointly by mouse, so the CV is
per-mouse-independent). PCA basis from one half, variance evaluated by cross-projection onto the other
(both directions averaged, 30 splits): signal replicates, noise averages to ~0. Per-neuron z by a
stage-level condition-agnostic std (no leak). PR = (Σλ)²/Σλ² on the positive-clipped reliable spectrum.
Null = condition labels shuffled within mouse. **Why repeated 2-fold and not k-fold (user question,
settled 2026-08-10):** the estimator is a CROSS-PRODUCT of two independent condition-mean estimates —
noise cancels in expectation only across exactly two independent copies; Var of the cross-term ∝
σ₁²+σ₂², minimised at n₁=n₂=n/2, so equal halves are variance-OPTIMAL and k>2 folds strictly waste
trials per copy (n/k-trial means are noisier AND a noisier basis deflates the spectrum via
misalignment). The 30 random halvings play the "repeats" role of repeated k-fold. Full paragraph in
`docs/paper/methods_notes.md`.

**Shattering dimension (Bernardi/Fusi 2020).** All 462 balanced 6-vs-6 dichotomies of the 12 conditions,
each decoded by a leakage-free pseudo-population decoder (disjoint train/test trial halves → K=24
pseudo-trials/cond → StandardScaler+PCA(30) fit on train → LDA), at the post-test decision window
(bins 57–65). SD = mean balanced accuracy over dichotomies; shuffle null = 0.50.

**PC coding (η²).** Condition-mean PCA (neurons std-normalised across the condition means), each PC's
across-condition variance decomposed onto orthogonal factor contrasts (sample / gng / test / choice
[/tasks]) — balanced 2×2×3 design so the η² are exhaustive per PC.

## Settled numbers (results.pkl, verified 2026-08-10)
| quantity | Naive | Expert |
|---|---|---|
| delay PR (12 conds) | 2.04 [1.96, 2.12] | 2.03 [2.00, 2.06] |
| decision PR (12 conds) | 3.28 [3.06, 3.44] | 3.29 [3.26, 3.43] |
| **DPA-delay PR (memory)** | 1.11 [1.00, 1.76] | **1.00 [1.00, 1.47]** |
| shattering (462 dich.) | 0.687 [.671, .698] | 0.697 [.688, .711] (null 0.50) |

PC coding → **Fig 2d = the η² MATRICES, Expert row of four** (dual-delay · dual-decision · DPA-delay ·
DPA-decision), with the message carried by the real data: dual-delay PC1 (40%) = gng .99, PC2 (12%) =
sample .90; dual-decision PC1 = gng .94, PC2 = choice .92, PC3 = test .71, PC4 = sample .71; **DPA-delay
PC1 = sample .98 (the 1-D memory line IS the sample axis; its PC2/PC3 η² lands on test/choice —
undetermined at delay BY DESIGN (test drawn independently of sample), at chance in held-out decoding,
failing cvPCA → SAMPLING NOISE, not anticipatory coding; gotcha 4 — flagged in the panel footnote)**; DPA-decision PC1 = choice .99. The FULL 2×4 grid (Naive &
Expert, `plot_dimensionality_main.py` → `dimensionality.png`) is ED material. Design settled 2026-08-10
after iterations: Expert/dual-only heatmaps hid the DPA/Naive evidence; the full 2×4 grid in the main
buried the message; a derived "chips" summary panel was REJECTED (Leon: keep the matrices — show real
data, not derived graphics); the final message-first pass then CUT everything off-message (shattering,
dPCA panels) rather than shrinking it. Final = 4-panel message figure; Naive + fits in ED.
Coding decodability at delay: sample 0.80, task ~1.0, test/choice at chance (future variables).
Everything ~stable Naive→Expert (the Naive grid shows the same one-variable-per-PC pattern).

## Gotchas (each has bitten — don't relearn)
1. **Compare the null on reliable VARIANCE, never on PR** — the null's PR (3.6–4.3) exceeds the real PR
   because PR of near-zero noise is meaningless.
2. **PR is variance-weighted**: the sample axis decodes 0.80 yet contributes little PR (cm-var ~12%).
   Low-variance ≠ unreliable; never write "sample is one of the top-2 delay dims" (they are distractor
   presence + identity in the 12-cond set / gng in the dual set).
3. **DPA-delay PR ≈ 1 is partly definitional** (only the binary sample is encoded during maintenance) —
   phrase as "the memory is a line", not as an independent discovery.
4. **Condition-mean PCs beyond ~PR are noise** — DPA-delay "PC2 = test / PC3 = choice" code FUTURE
   variables; cvPCA strips them. Read every η² heatmap together with the PR.
5. **Shattering window must be post-test (57–65)** — earlier windows leave test undecodable.
6. **Trajectory (time-resolved) PR stays excluded** — its shuffle null retains ~46–50% of the variance
   (the condition-independent time ramp).
7. **Clobber protection (fixed 2026-08-10)**: `exp_dimensionality.py` merge-dumps `results.pkl` and
   writes `dimensionality_qc.png` — earlier versions replaced the pkl (losing FITDATA/DPA_GNG keys) and
   overwrote the curated `dimensionality.png`. Don't reintroduce.
8. Window robustness (`--altwin`, full-delay / test windows): DPA-delay PR stays 1.0–1.1, delay 2.3–2.6,
   decision ≈2.5 — conclusions direction-stable.

Paper Methods paragraphs: `docs/paper/methods_notes.md` (Fig-2 block, incl. the 2-fold-vs-k-fold
justification). Memory: [[project_dimensionality]] (analysis) + [[project_main_figs_review_2026-08-10]] (the Fig-2
replacement decision). Manuscript: `docs/paper/results_draft.md` §2 + ED 3.

## 2026-08-30 review fixes (Figs 2 & 3 — see memory [[project_main_figs_review_2026-08-30]])

- **Panel-C nulls at 1000 draws** (`exp_cdec_support.py`; was 100, where the Expert-md margin of
  0.011 was seed-flippable). Verdict SURVIVES: dist-from-DPA-PCs Expert-md acc 0.61, permutation
  **p=.031**; decision-Naive p=.029, decision-Expert p=.013; md-Naive p=.67. `DPA_GNG_C` now stores
  `p` and the full `null` array; the "weak transfer" annotation is verdict-conditional.
- **Each stage vs ITS OWN null** in panel C (Expert solid / Naive dashed tick) — the single Expert
  line made one Naive dot (DPA-decision sample, sig vs own null) read as n.s.
- **Reliable-rank rule replaced** (`_rank_b`): now cumulative-95%-of-reliable-variance with a
  2×shuffle-floor guard, reproducing ranks 1/3/2/3; the old `jackknife lo > 1%` rule was knife-edge
  (dual-md rank 2 by 0.002, and t8-vs-z flipped it).
- **Jackknife CIs use t(8)=2.306** (n=9 mice), producer-side in `exp_cdec_support`, renderer-side for
  the `--pr` PR bars. ~15% wider than the old z=1.96 bars.
- **Canonical naming**: panel C/D display 'dist' (cache key stays 'gng'); cartoon values now read
  from `SPEC_JK` (0.92/0.07, was hardcoded 0.93); GNG-cue box drawn at its true 6.5–7.0 s.
- **η² caveat**: dual rows do NOT sum to 1 (4 of 7 centred contrasts displayed; dual-md PC2 leaks
  ~6% to unshown interactions) — exact only for DPA. Never caption "rows sum to 1" unqualified.
- **`decoders.py` scope corrected**: it is the Fig-3 (+overlaps caches) decoder; Fig 2's panels use
  contrast axes (`exp_dpca_count`) and LDA (`exp_cdec_support`/`exp_pceta_md`) — Methods must not
  claim one decoder across Fig 2/3.
- **Fig 3 star policy**: title verdicts only for the whitelisted knob-robust test (per-mouse
  dist↔choice cross-decode, p=.0078 pca20 / .0039 nopca after aligning PM_ACT windows/filters to the
  pooled matrices: lick @ bins_TEST, GNG side all dual trials). CCGP titles carry no verdict (test-CCGP
  flips .04/.73 with the PCA knob). **Per-mouse cosine scatter now plots RAW split-half cosines**
  (attenuation correction is unusable per animal: rel down to 0.03–0.07 inflated |cos|~0.1 → ~0.9;
  corrected values are NaN'd below REL_FLOOR=0.15 in `exp_permouse_frame.py`). NB the raw increase is
  knob-robust (p=.008 both variants) but deliberately UNSTARRED pending an explicit decision.
- Fig 3 renderer guards: `--npc N≠20` refuses to run (would mix caches and overwrite `_pca20` files);
  `exp_axis_frame` seeded (panel-C pooled cosines no longer drift per rebuild); panel-E hatches any
  ratio>1 cell; panel-A trace columns share y across Naive/Expert; x-axis cropped to 12 s.

## 2026-08-31 RESTRUCTURE ("redistribute" — user decision; supersedes the panel lists above)

The three geometry mains were REBALANCED so each carries one full message:
- **Fig 2** (`fig_dimensionality_main.py`) **gained panel E** = the cross-task generalisation
  matrices (from old Fig 3 E; canonical no-PCA cache `matrices_cache_acc_nopca.pkl`, hardcoded).
  Message: few axes, one per variable, and the SAME axes in every task. 3 rows now (A-D + E).
- **Fig 3** (`fig_manifold_main.py`) **slimmed to the frame only**: A traces (2×4), B the state
  scatters as a HORIZONTAL storyboard (DPA·md → DPA·decision → dual·md → dual·late → dual·decision),
  C axis-cosine matrices (Naive|Expert, centred). The learning panels left; panel_gen/panel_d/
  _mouse_scatter etc. were DELETED from this file (new homes below).
- **Fig 4** (`overlaps/fig_overlaps_main_native.py`) is now the LEARNING figure: new top row A =
  dist↔choice cross-decode matrices + per-mouse raw-|cos| scatter ∗ + per-mouse cross-decode
  scatter ∗ (both knob-robust; drawn from the CANONICAL no-PCA caches regardless of build flags);
  B = the push (∗ kept per user decision, caption discloses per-animal p=.098); C coupling;
  D FA/CR; E choice-d′. Letters shifted A→B→C→D→E.
- **ED**: new `pca/fig_manifold_supp.py` = per-mouse CCGP (no title verdicts) + per-mouse
  generalisation companions; plus the Fig 3 PCA-20 variant. CCGP stated in text.
- **STAR REVERSAL (logged)**: the per-mouse raw-cosine increase is now STARRED in Fig 4-A
  (p=.008/.008 across pipelines — meets the same whitelist criterion as the cross-decode star).
  The old "never star the cosine test" verdict targeted the retired attenuation-corrected estimator.
- Canonical pipeline everywhere = **no-PCA**; PCA-20 = ED robustness. NB `fig_manifold_main.py`'s
  default CLI run still builds the _pca20 file; the canonical PNG needs `--nopca`.

## 2026-08-31 (later): Fig 2 panel E — timeline order + the dist matrix

- Panel E matrices now follow the TASK TIMELINE: **sample → dist → test → choice** (user request).
- **New dist matrix** (`pca/exp_dist_task.py` → `DIST_TASK[(wn, stage)]` in results.pkl, wn='md'):
  Go-vs-NoGo has no within-task training, so the matrix reads the dist code THROUGH each task's
  state geometry — rows = source subspace (top-3 PCs of DPA/Go/NoGo condition means @ md, the
  panel-C `DPA_GNG_C` construction generalized), LDA always fit on held-in dual pseudo-trials.
  Columns = test set: **DPA†** = fraction of DPA trials classified to the NoGo side (descriptive,
  no ground truth; train-DPA × test-DPA cell BLANK by design), **Go** = Go-class accuracy,
  **NoGo** = NoGo-class accuracy (held-out halves). Cells are RAW accuracies (no ceiling → no
  ratio normalisation; the in-figure key explains both constructions).
- Expert values: src-DPA Go .61 / NoGo .58 (consistent with panel C's dist-cross bar 0.61 ✓);
  src-Go .46/.76; src-NoGo .77/.64. DPA† fractions .45–.59 — DPA trials sit AMBIGUOUSLY on the
  dist axis (they do not clearly fall on the NoGo side), an honest, quotable observation.
- NREP=16, seeds RandomState(700+row); merge-dump (results.pkl safe).

### 2026-08-31 follow-up: dist matrix OUT of the figure; panel F (stability scatters) IN
The dist matrix was REMOVED from panel E after one render (user decision): with no within-task
training possible for Go-vs-NoGo, every cell is a transfer through a geometry built without the
contrast and none can reach 1 — it read as broken next to the ratio-normalised matrices. The
analysis + cache survive (`exp_dist_task.py` / `DIST_TASK`) for ED/text use — notably the DPA†
fractions (.45–.59): DPA trials sit AMBIGUOUSLY on the dist axis. Panel E is now sample → test →
choice (timeline minus dist); the in-figure ratio key moved to the caption. Its cell hosts the new
**panel F**: per-mouse mean cross-task accuracy Naive vs Expert (PM_GEN_nopca, raw off-diagonal
mean; ratio unusable per animal) for sample/test/choice — **generalisation is STABLE across
learning** (Δ≤.013, p=.91/.36/1.00; both pipelines agree, pooled bootstrap Δ n.s.) — the foil for
Fig 4's learning effects. No verdicts drawn (star policy).

### 2026-08-31 final Fig-2 state (aesthetics + caption + review fixes) — CURRENT
- Layout: 4-row gridspec (row 1 = thin spacer separating the top row from D), outer wspace 1.5
  (air between A|B|C), fig 10.6×7.9; panel-C task group labels at the axes' OUTER edges (centred
  labels collide with D's title at any depth); E (matrices) and F (scatters) share a top line.
- **In-figure JUSTIFIED caption** (panels A–F, drawn values): matplotlib has no native
  justification, so the caption is typeset word-by-word from `CAP_PARAS` (measure word widths via
  the Agg renderer, spread slack across gaps; paragraph-final lines flush-left). Edit `CAP_PARAS`
  and re-render; NB the SVG carries caption words as separate text elements.
- **figure-review verdict (2026-08-31): every drawn number reproduces; two caption fixes applied**:
  F now states the equivalence bound (Δ 95% CIs sample [−.031,+.023] / test [−.009,+.035] /
  choice [−.033,+.054] — all within ±0.05), and C's "only when in play" names its own † exception
  (anticipatory Naive dual-choice 0.66* at mid-delay, gone with learning). Reviewer-pocket answer
  for E's heavily-hatched test matrix: weak per-task ceiling; sample+choice matrices carry the
  claim (or move test → ED).

### 2026-08-31: Fig 3 finishing pass (same treatment as Fig 2) — CURRENT
- **Justified caption added** (panels A/B/C, drawn values; the choice×dist numbers are DYNAMIC from
  `AXIS_FRAME` so the `_pca20` ED variant captions its own matrices). Justification code extracted
  to the shared **`pca/figcaption.py`** (`draw_justified`) — Fig 2 now uses it too (identical
  output, one implementation).
- Layout: row C enlarged (cols 3:9, row ratio 0.85, fig 12.4×8.6) — the two cosine matrices no
  longer float in an empty band. Panel-A sample legend moved to the EXPERT panel's lower-right
  (in the Naive panel it collided with the epoch-name band up top and the Odor-A tail below).
- Caption's panel-B sentence records the storyboard reading, incl. the honest pre-cue point: the
  dual Go/NoGo states already sit apart along the choice axis at mid-delay because the DISTRACTOR
  precedes that window.

### 2026-08-31: Fig 3 FILLED BACK OUT (user: "seems empty next to Figs 2/4") — CURRENT
Two additions, both chosen by the user from proposals:
- **Panel B is now a 2×5 storyboard (Naive | Expert)** — `build_frame(stage)` parameterised (sets
  the module STAGE global; helpers read it; per-stage axes re-fit on that stage's own independent
  trial half → per-stage units; the fixed-axis quantitative push stays Fig 4's). New `frame_states`
  precomputes the clouds per stage; all TEN frames share one x/y range. RENDER-CONFIRMED and now
  in the caption: (i) the Expert DPA delay states sit below the choice-axis baseline where the
  Naive ones do not (geometric preview of the Fig-4 push); (ii) the dual Go/NoGo mid-delay split
  is weak in Naive, strong in Expert.
- **New panel D** — per-mouse raw |cos| strip (PM_COS raw values; open=Naive, filled=Expert,
  per-mouse tab10, black mean bars): sample×choice and sample×dist hug the floor in ALL 9 mice at
  both stages (~0.05); choice×dist higher and growing (0.073→0.114) — deliberately NO stats drawn
  (the increase is tested/starred in Fig 4A; drawing it here would double-report).
Layout: fig 12.4×11.2, rows [1.45, 2.0, 0.85]; row 2 = C matrices (cols 1:7) + D strip (8:12).
Letters A–D; caption updated (B per-stage caveat + the two confirmed observations; D paragraph).
Both variants re-rendered. Fig 3 now reads at comparable density to Figs 2/4.

### 2026-08-31 Fig 3 FINAL STATE (supersedes the two blocks above where they differ)
- Panels: **A** traces 2×4 · **B** storyboard 2×5 (Naive|Expert, per-stage axes) · **C** cosine
  matrices (left-aligned, col 0) · **D** per-mouse raw-|cos| **Naive-vs-Expert SCATTERS** (one per
  axis pair — replaced the paired-dot strip at user preference; house idiom: mouse colours, opsin
  markers, unity line (the mean diamonds were removed 2026-09-01 figure-wide), shared 0–0.25
  limits; NO stats — choice×dist is starred in
  Fig 4A) · justified caption (figcaption.draw_justified).
- Layout: fig 12.4×10.8, rows [1.45, 1.55, 1.0]. Row B deliberately SHORT — its shared y-range is
  set by the +9 z decision licks, so tall frames read empty at mid-delay; compressing fills the
  data band. Row C/D taller (aspect-locked panels size by row height); C and D NW-anchored to
  share a top line.
- Render-confirmed caption claims: Expert DPA delay states below the choice baseline where Naive's
  are not (push preview → Fig 4B); dual Go/NoGo mid-delay split weak in Naive / strong in Expert;
  choice×dist above unity in 8/9 mice (D).

### 2026-08-31 (later): Fig 3 gains panels E/F — plane sufficiency WITH stats (user decision)
- **E** = per-mouse 3×3 block (`exp_permouse_plane.py` → `PM_PLANE`+SUF): each variable decoded
  from the mouse's own 2-D plane / the out-of-plane residual / the full space (held-out halves,
  canonical windows; (plane, full, out) triples). Double dissociation per animal: sample & choice
  in-plane ≡ full and collapse out-of-plane; test at chance in-plane, untouched out-of-plane.
  Learning Δs n.s. everywhere in BOTH pipelines → annotations only.
- **F** = the E averages ± SEM with paired Wilcoxons DRAWN. Verdicts (knob-robust, nopca/pca20):
  sample out-vs-full ∗ (.0039/.0039) · test plane-vs-full ∗ (.0039/.0039) · choice out-vs-full ∗
  (.0195/.0078); robust n.s.: sample plane-vs-full (.22/.50), test out-vs-full (.36/.82).
  **choice plane-vs-full flips with the knob (.94/.012) → not drawn; caption discloses.**
- Fig 3 = 4 rows (12.4×14.6): A traces · B storyboard · C+D geometry · E+F sufficiency; caption
  paragraphs E/F added. Companion analyses still pending a home: XSTAGE_DEC (cross-stage frame
  identity, transfer/within 0.90/0.87), AXIS_TIME (axes stable while live), PLANE_TRAJ (traces
  through the plane) — in `fig_manifold_addons_preview.py` / tmp gallery.
- DEAD-ENDS logged in memory: PLANE_VAR variance fractions (axis-noise-attenuated) and corrected
  cross-stage cosines (explode at rel~0.15) — decode, don't cosine.

### 2026-08-31 (final): Fig 3 E/F swapped; E carries the COMPLETE bracket set
E = summary bars first (user), with all nine pairwise Wilcoxons drawn: plane-vs-out ∗∗∗ (all
variables, knob-robust), out-vs-full ∗/n.s./∗ (sample/test/choice), plane-vs-full n.s./∗/†
(† = the choice pair, pipeline-dependent .94/.012 — bracket drawn, no verdict, caption defines).
F = the per-mouse 3×3 block (annotations only; learning Δs n.s. both pipelines).

### 2026-08-31 (last): dist added to Fig 3 E/F — completes the timeline + a new starred result
E bars = 4 groups (sample/dist/test/choice), adaptive bracket heights. dist: plane 0.61 < out
0.82 ≡ full — partial in-plane share (all verdicts knob-robust). **F dist/plane cell carries the
figure's one starred learning effect: plane-only dist accuracy 0.57→0.65 (p=.020/.027, 8/9 & 7/9
mice) — per animal, learning pulls the distractor code into the manifold (independent echo of
Fig 4A's alignment stars).** Whitelist updated accordingly.

### 2026-08-31 Fig 3 structure after the message review (supersedes panel lists above; ITSELF
### PARTIALLY SUPERSEDED by the later same-day blocks below — panel A is now the TASK-SPLIT 2×6
### row, B is the centred CCGD-replay storyboard, E/F gained per-mouse scatters; read to the end)
**A codes · B frame (2×5) · C sufficiency bars (all stats) · D per-mouse 3×4 (dist-plane learning
∗) · E cosine matrices · F cross-stage decoding 2×2s (transfer/within 0.90/0.87 — one frame across
learning, knob-robust).** Per-mouse cosine scatters → fig_manifold_supp.py panel C (duplicated
Fig 4A's data). Caption title = the proven claim (necessary & sufficient + dist pulled in).
Reading order: frame → proof → geometry/identity. exp_plane_frame.py now run for BOTH pipelines.

### 2026-08-31 storyboard REPLACED by a CCGD REPLAY (user: "B is not consistent with A")
Two fresh-axis origin conventions failed in one day, and the failure is structural: the
storyboard's freshly-fit single-window axes carry the trial's condition-independent ramp (the
documented 29–45% contamination), so NO single origin works — **baseline-zero** dragged every
window to one side of the crosshair ("dots not centered"), and the **boundary-zero** fix (b_s/b_l
class midpoints) put the mid-delay states 3–5 z below the lick line while panel A's traces showed
them AT baseline ("B not consistent with A"). The fix is the same one panel A already uses: panel
B now REPLAYS `overlaps/main_panels`' validated per-mouse CCGD projections — new cache
`exp_frame_states.py` → `FRAME_STATES`/`_pca20` in results.pkl: per-trial window means of
SAMPLE_D (x) and LICK_D (y), laser-off correct trials, reduced to per-mouse condition means
(≥3 trials/mouse, ≥3 mice per cell; windows = overlaps bins md 33–38 / late 45–53 / decision
57–62). NB Y_SAM and Y_LCK are the same trials but NOT row-aligned — cells are selected
independently per table (means don't need trial pairing). Storyboard glyphs: faint dots =
per-mouse means, ellipse = 1 SD across mice, marker = grand mean; **crosshair = per-mouse
baseline zero = exactly A's dashed line** — A and B are literally the same coordinates and units
(shared across stages too). All fresh-axis machinery (build_frame/sample_axis/lick_axis/cloud/
pseudo-trials, the fits_inputs.pkl load) is deleted from `fig_manifold_main.py`.

### 2026-08-31 storyboard FINAL: per-window per-mouse RE-CENTRING (user: "worse than ever — just
### readout the mid delay clouds and decision clouds correctly")
The raw replay was consistent with A but unreadable: between-mouse offsets + the shared ramp
inflated the 1-SD-across-mice ellipses into an overlapping soup, and the absolute positions
(both decision classes riding the ramp up) hid the splits. FINAL design: `_centered()` in
`fig_manifold_main.py` subtracts, per mouse per window, that mouse's cross-condition mean state
(cache now keeps mouse ids), so each panel shows ONLY the condition geometry at that moment —
which codes are separated and along which axis. Crosshair = the window-mean state; the absolute
displacement (ramp, no-lick push) belongs to panel A's traces and Fig 4B, and B's caption makes
no absolute-position claims anymore. Shared axes limits from the 2–98 percentiles of the
per-mouse dots (outliers clip); scale bar 2 z. Settled per-window numbers (canonical): sample
sep at md ≈ 4.8/4.8 (Naive/Expert); DPA·decision lick−no-lick split 3.2/4.6 (= A's trace gaps
exactly); **dual Go−NoGo choice-axis split md +1.2 → +4.2, late delay +1.5 → +6.2 with
learning** — the "weak in Naive, strong in Expert" caption claim, now printed by the script.
Consistency with A holds at the level of SEPARATIONS (identical projections ⇒ identical gaps);
trace values are not the cloud offsets (those are centred away).

### 2026-08-31 ANTACT axis variant (`--antact`, user request)
`fig_manifold_main.py --nopca --antact` → `fig_manifold_main_antact.png`: the choice axis in A
(choice trace) and B (y-axis, relabelled "antic. action axis") is the ANTICIPATORY action axis —
main_panels' `--antact` (decoders trained over overlaps bins 48–62) passed through
`exp_traj_orig.py --antact` / `exp_frame_states.py --antact` (cache keys `ORIG_TRACES_antact` /
`FRAME_STATES_antact`; the composed `--pca --antact` suffixes exist in the scripts but only the
canonical antact caches are built). Panels C–F unchanged (pca-side axis); caption carries an
[AXIS VARIANT] note. What it shows vs the action axis: the Expert choice trace DIPS below
baseline through the delay (the anticipatory push signature, cf. Fig 5's antact preference);
DPA·decision lick−no-lick split SMALLER (±1.5 both stages vs ±1.6/±2.3 action) and the Expert
DPA·md micro-split inverts (lick −0.18 vs no-lick +0.18, a wash) — antact trades decision
discriminability for the anticipatory signal, same trade-off as the Fig 4/5 axis grid; dual
Go−NoGo splits unchanged-strong (md +1.4→+4.3, late +1.3→+6.6).

### 2026-08-31 storyboard polish (user: "best course of action" after the dispersion/push questions)
Three decisions, all applied:
1. **Ellipses SD → SEM across mice** (cov/n): the storyboard's question is "where do the condition
   means sit"; animal-to-animal spread is panel D's job. Ellipses ~3× tighter; caption updated.
2. **The push is back in B, honestly**: the PRE-CUE (md) panels draw the pre-trial baseline as a
   grey line at −offset (the removed window mean, printed as "mean-vs-BL"). There the common ramp
   is negligible (offsets: Naive DPA +0.03, Expert DPA −0.44, dual +0.40/−0.06), so
   baseline-vs-crosshair IS the mean displacement — in Expert DPA the baseline sits visibly ABOVE
   the window mean (delay states 0.4 z on the no-lick side = the push), in Naive it coincides.
   Post-cue windows omit the line (the shared lick ramp dominates: +1.7–3.2) — caption discloses.
3. **Antact full-figure card retired from the gallery Main tab** (the centred storyboard is
   axis-invariant — a robustness finding; the `--antact` flag, caches and PNG remain).
Literature review (2026-08-31 session): the whole preprocessing chain (CV decoder projections →
per-mouse baseline+evoked normalisation → per-window cross-condition mean removal → per-animal
summary, n=9) is standard — CI-component removal is the dPCA/Mante/Panichello canonical move
(Kobak 2016 eLife; Kaufman 2016 eNeuro "largest component = timing"; Aarts 2014 Nat Neurosci for
the nested-data unit). Caveats to keep disclosed: balanced (unweighted-cell) common-mean
estimation with occasional missing cells; centring forfeits absolute claims (A + Fig 4B carry
them).

### 2026-08-31 (amendments to the polish block above, user)
- **Baseline lines REMOVED** from the pre-cue panels ("what is the point... remove them") — the
  push story stays with panel A + Fig 4B; the removed offsets still print as "mean-vs-BL".
- **Decision READ window moved 57–62 → 60–66** (10.0–11.0 s, the first second of the response
  window; the choice AXIS is still trained at the 57–62 lick moment). Motivated by "dual Expert
  decision is bad": at 57–62 the learned Go/NoGo cue-history (+3.2 z) rivals the lick split
  (+4.7) on the same axis and the 8 conditions interleave. A window scan (57–62/57–68/60–66/
  63–69/66–72/60–72) picked 60–66: lick split +4.7→+6.0 (dual E) and +4.6→+6.0 (DPA E),
  Go/NoGo superposition +3.2→+2.5, at a small sample-sep cost (2.0→1.7). The JawsM12 Go-B-lick
  outlier (+25 z raw, 15 trials — real per-mouse lick-amplitude heterogeneity, the known
  0.1–43× tensor scale spread) is window-invariant; SEM ellipses absorb it. All THREE caches
  rebuilt (canonical/_pca20/_antact); caption B now defines all three windows.

### 2026-08-31 code/analysis review of Fig 3 — verdict + fixes applied
Deep review (all 5 scripts read, both pipelines re-run, every drawn stat re-verified): NO blocking
rigor issue — no train/test leakage anywhere (PM_PLANE fits axes+classifiers on half 1, tests
half 2, out-of-plane uses the train-estimated Q; XSTAGE trains part-0 / tests disjoint part-2);
"registered neurons" literally true (VALIDIX identical across stages, all 9 mice); 9/9 mice
complete for every variable; Wilcoxon floor .0039 = all-9-one-direction; (plane,full,out) index
mapping consistent; caption numbers reproduce in BOTH pipelines; seeds fixed (bit-reproducible).
FINDINGS + FIXES (all applied):
1. **Panel E attenuation correction runs at low rel** — sample .23–.24, choice .24 (Naive)/.39
   (Expert); only dist healthy (.52–.61); denominators for the 0.32→0.47 growth differ 2×
   between stages. FIX: rel now printed top-right of each matrix; caption says the pooled values
   are "estimates, not tests" and sources the growth to Fig 4A's RAW-cosine per-animal star.
2. **No multiple-comparison correction** (C: 12 brackets, D: 12 tests; the dist-plane star
   .020/.027 would not survive Holm×12). FIX: caption D reframes it as "the per-animal test
   predicted by Fig 4A's starred alignment increase (a directional confirmation, not a
   discovery; p uncorrected)"; caption C adds "p values uncorrected".
3. **Sample/choice out-of-plane collapse is by construction** (the plane is built FROM those
   axes). FIX: caption C states it; informative results = plane=full + the test/dist contrasts.
4. Docstring rot fixed (header B description, fits_inputs claim, "1 SD"→SEM, stale PCA-20
   comment, duplicate assert); C sig markers to house 12/8; sub-floor fonts bumped (5.2–5.8 →
   5.8–6.2); caption F adds cross-stage 0.88 ± 0.03–0.05 across resamples; caption B disclosés
   "read windows chosen for display — no statistics drawn"; title drops "of both tasks" (choice
   sufficiency is DPA-tested; the dual evidence is B/F/Fig 4A).
Verified-stable: XSTAGE cross 0.880±0.046 (sample) / ±0.027 (choice) across 8 reps.

### 2026-08-31 E/F gain PER-MOUSE scatters (user request); supp back to A+B
- **E right** = 3 per-mouse raw-|cos| Naive-vs-Expert scatters (PM_COS; sa .05→.06, sd .05→.05,
  ad .07→.11 nopca) — RETURNED to the main from the supp (user reversal of the earlier
  "duplicates Fig 4A" call); no tests drawn, caption sources the ad increase to Fig 4A's star.
- **F right** = 2 per-mouse cross-stage transfer scatters from the NEW cache
  `exp_permouse_xstage.py` → `PM_XSTAGE`+SUF (per mouse: own decoder trained on one stage's
  trial half, tested held-out on BOTH stages; registered neurons asserted per mouse; per-stage
  feature sd; NREP=10, rng 500). Within-vs-cross unity scatters; annotation = mean
  chance-referenced T/W (ratio only for mice with within>0.52): **sample 0.86/0.92, choice
  0.73/0.67 (nopca/pca20)** — lower than the pooled 0.90/0.87 as expected (within-mouse decoders
  noisier); descriptive only, no verdicts. JawsM06 choice sits at chance (0.49) — honest.
- Layout: bottom row = 12-slot gsBot with width_ratios + spacers (E mats | E scatters | F mats |
  F scatters); rel annotation moved above the stage-title line ('rel .24/.24/.52' superscript
  style) after two collision iterations (below-matrix hit the caption; title-line overlapped).
- `fig_manifold_supp.py` panel C REMOVED again (would duplicate main E); supp = A ccgp + B
  per-mouse generalisation, fig 9.4×5.2.

### 2026-08-31 panel-A TASK-SPLIT variant (`fig_traj_tasksplit.py` + `--tasksplit` full figure)
2×6 companion to Fig 3 panel A (user request, built same day): **DPA | Go | NoGo × sample/choice
code**, rows Naive/Expert, y shared per CODE across all task columns so amplitudes compare
directly. Cache: `exp_traj_orig.py` now stores per-task trace keys — `sample@dual`/`lick@dual`
(pooled) plus `sample@go`/`lick@go`/`sample@nogo`/`lick@nogo` — alongside the untouched canonical
panel-A keys (n=9 everywhere; canonical rebuilt, pca20 NOT yet — run `exp_traj_orig.py --pca`
before rendering the `--pca` variant of this figure). WHAT IT SHOWS (display-level, no stats):
- **dual sample code DECAYS post-distractor in BOTH Go and NoGo** (B: +2.5 at the distractor →
  ~0 by test Naive / −1.5 to −2 Expert; deepest NoGo·A −4.4) where DPA sample holds its plateau
  — the distractor odor, not the cue response, erodes the readout;
- **Go·choice**: BOTH classes ride up together at the GNG cue (~7 s, to +6–7 z Expert — every
  correct Go trial licks the cue: motor/reward transient, not choice coding); the lick/no-lick
  split opens only at test;
- **NoGo·choice**: Expert traces dip below baseline through the late delay (withholding), then
  split at test; max class gaps at test ≈ DPA's (5.5–6.7 z).
Same replayed CCGD projections and house style as panel A; sample/distractor/GNG-cue/test bands;
staggered epoch labels. Output `fig_traj_tasksplit[_pca20].png` (standalone preview, folder
listing only).

**FULL-FIGURE variant (same day, user):** `fig_manifold_main.py --nopca --tasksplit` →
`fig_manifold_main_tasksplit.png` — the complete Fig 3 with panel A replaced by the 2×6
task-split row (`panel_traj_tasksplit`; caption A swapped, states that the dist/test code
columns are omitted); panels B–F identical to canonical. THIS is the build pinned in the
gallery Main tab (5th Fig-3 card family member: canonical / pca20 / antact / task-split).
If adopted as THE panel A: run `exp_traj_orig.py --pca` first (the @go/@nogo keys exist only
in the canonical ORIG_TRACES so far), and find a home for the dist + test code traces (ED, or
a wider merged row).

### 2026-08-31 TASK-SPLIT panel A ADOPTED (user: "same data as panel B — links to it")
The 2×6 task-split row (DPA | Go | NoGo × sample/choice) IS now Fig 3 panel A — the `--tasksplit`
flag is GONE (the review's structural objection was answered by the user's design argument: rows
A and B are the SAME per-mouse CCGD projections with the same task split — A = time courses,
B = window snapshots — which canonical A, with its dist/test axes absent from B, never had).
Executed with the review's wording fixes baked into caption A: the dual decay names its per-mouse
counts (9/9 Naive, 8/9 Expert — the Expert side is a trend, p=.098 if ever tested), is phrased as
"the DPA-trained sample READOUT decays" with the memory-survival question pointed at Fig 2E, and
the NoGo dip is "below baseline on average (7/9 mice)". The four-code 2×4 trace row (dist + test
axes — the definitional reference for C–F's variables) moved to `fig_manifold_supp.py` panel A
(supp now A traces / B ccgp / C generalisation, 9.4×9.8). ORIG_TRACES @go/@nogo keys now exist
in ALL THREE cache variants (canonical / _pca20 / _antact — all rebuilt). The interim
fig_manifold_main_tasksplit.png is deleted (canonical IS task-split); `fig_traj_tasksplit.py`
stays as a standalone preview renderer only. Gallery Main back to three Fig-3 cards.

### 2026-09-01 Codex cross-review of Figs 2–4 + verification + applied fixes
External Codex review (session 01a059c1; full text in the session log): NO new leakage or
statistical-unit errors in any cache producer; five propositions. Outcomes:
- **Scaling sensitivity (the one new analytic concern) RESOLVED**: `exp_xstage_scale_check.py`
  re-runs XSTAGE_DEC's protocol (same seeds) scoring the test stage in the TRAIN stage's
  per-neuron scaling instead of its own — transfer/within 0.919→0.899 (sample), 0.888→0.883
  (choice), i.e. ≤0.02: the Fig 3F frame-identity claim is not a renormalisation artifact.
  Cache `XSTAGE_SCALECHK_nopca`; caption F now states the robustness.
- **Applied**: Fig 2 title narrowed ("the memory and choice axes are shared across tasks") + E
  lead reworded (test matrix = the honest boundary; kept, not demoted); Fig 3B in-panel
  'window-centred' label; Fig 4C mouse-mean diamonds — REVERSED 2026-09-01 (user: diamonds
  hide the points; removed figure-wide, the caption states the animal unit in words) (A/B
  circles stay) + caption glyph clause.
- **Rejected with reasons**: Fig 3D→ED (carries the only starred learning effect + explicit
  user preference for per-mouse mains; fallback = slim to 2×4); "Fig 2 caption too long"
  (conflicts with the caption standard; manuscript captions at submission); Fig 4 headline
  recalibration (already satisfied — A leads, push trend disclosed, ∗ kept by user decision).

### 2026-09-01 craft-menu EXECUTION (user routing: PS+biplot main · TGM supp · embed=decide · AI attempted)
- **Fig 2 gains** (canonical PNG updated): PS annotations under each panel-E matrix
  (`exp_parallelism.py` → PS cache, pipeline-invariant — condition-mean vectors, no decoder:
  raw sample .28 / test .14 / choice .39, ALL ≫ null95 ≈ .04–.05; rel-corrected sample .99 /
  choice 1.00 — the task-wise coding directions are essentially PERFECTLY parallel; test corr
  1.21 = low-rel estimate, caption discloses) + **panel G** per-neuron selectivity biplot
  (`exp_neuron_sel.py` NEURON_SEL: d′ sample@md vs d′ choice@decision per neuron, n=3319,
  model-free; |d′| corr r=−0.03, both-selective 6.2% vs independence 6.4% — the cross-shaped
  cloud; row-3 layout now E 0:5 / F 5:10 / G 10:12; caption title roadmap + G paragraph added).
- **Supp gains panel D** = temporal-generalisation matrices (King & Dehaene;
  `overlaps/exp_tgm_cache.py` → TGM, computed from the tensor's train-bin × test-bin decision
  functions, no refitting): sample = ONE stable block spanning the delay (both stages), dist =
  post-cue block, test/choice compact late blocks — the standard "stable code" visual
  (supp now A traces / B ccgp / C gen / D TGM, 9.4×12.6).
- **ALIGNMENT INDEX = DEAD-END** (`exp_alignment_index.py`, cached AI_nopca v1): Elsayed-style
  subspace AI sits AT its covariance-matched null (cross-stage K=2 0.59 vs null95 0.63; K=3
  0.54 vs 0.67; v2 with per-neuron scaling + rank-matching: md K=2 0.34 vs null mean 0.36,
  run then terminated — verdict clear). Same disease as PLANE_VAR: top-K condition-mean
  subspaces carry unreliable dimensions, and the covariance-shaped null is extremely
  conservative. DO NOT annotate Fig 3F with it — the decoding transfer (+ scaling check)
  remains the cross-stage evidence. Logged so it is not re-attempted.
- **Embedding preview** (`fig_embed_preview.py`, gallery Main "DECIDE" card): 12 condition
  means, z-scaled, PCA-3D per stage × window — tasks separate on PC1/2, the A–B sample edges
  run PARALLEL across all three tasks (the PS result made visual). Awaiting main-vs-supp call.
- **Fig 4 `--polish` variant** (`fig_overlaps_main_ab_dpaact_polish.png`, canonical untouched):
  set-point WELL schematic insets on the B planes (deeper minimum in Expert, green lick
  boundary) + axis-rotation glyph inset in A's raw-|cos| scatter (dist axis at 71°→62° from
  the choice axis, angles from the corrected Fig 3E cosines); caption [POLISH VARIANT] tag.
- TODO (user: "not now"): orofacial motion-energy regression if video exists (ED 5d lick
  covariate already in place).

### 2026-09-01 (later): embedding split per task set; Fig 4 insets PROMOTED to canonical
- `fig_embed_preview.py` now 2×4: stage × (DPA·md | DPA·decision | dual·md | dual·decision),
  each panel its OWN per-set PCA. Titles carry the Fig-2b RELIABLE RANK and the suptitle warns
  that PCs beyond it are noise (with 4 DPA conditions, PC2/3 are unreliable — do not read the
  offset between the two A–B edges). Still a DECIDE card in gallery Main.
- Fig 4's two schematic insets (set-point well in B, axis-rotation glyph in A) are now part of
  the CANONICAL build in all four variants — the --polish flag and the _polish PNG are gone;
  caption A/B carry the inset descriptions inline.

### 2026-09-01 (final): Fig 4 schematic insets REMOVED (user: "catastrophic, unreadable")
The well and rotation-glyph insets were reverted the same day they were promoted — at inset
scale (~0.3 of a small panel, 5 pt text) they were illegible. Canonical Fig 4 is back to the
pre-inset build (all four variants re-rendered; caption clauses removed; gallery card reverted).
DESIGN LESSON (log with the real-data-panels rule): schematic explainers either get a DEDICATED
schematic slot at full panel size (the Fig 2a precedent) or they don't go in — corner insets at
inset scale do not work. The set-point/rotation ideas remain available as full-size schematic
candidates if a slot ever opens.

### 2026-09-01 (late): polish batch + style coherence + gallery Variants tab
- **Style-coherence canon** enforced across ALL figure scripts (one rc block; letters-only
  bold; sig stars 12/8; per-mouse dots s=34; lines 1.3; savefig 400) — the pass also caught
  Fig 5's stale caption/header stats (corrected to the rendered truth; see `docs/behavior.md`).
- **Fig 2**: B's cartoon glyphs OUT (text callouts kept — same lesson as the Fig 4 insets:
  no micro-drawings inside panels); bottom row rebuilt as equal-width slots (gsE/gsF internal
  `wspace=0.28` both) with E/F/G on ONE shared centre line (panel F's stale `set_anchor('NW')`
  → `'C'`). Fig 1's banner sentence removed; Fig 3 D grid tightened (wspace 0.24→0.08).
- **Mean diamonds removed from every per-mouse scatter in the paper set** (5 mains +
  `fig_manifold_supp`; user: they hide the points) — including Fig 4C's mouse-coloured ones
  from the Codex round: the caption now states the fitted unit in words ("the fitted unit is
  the animal, not the 18 points"). Diamonds survive only in non-paper scripts (standalone
  `fig_ccgp.py`, superseded `fig_overlaps_manifold.py`, `fig_manifold_addons_preview.py`).
- **Gallery**: new pinned **Variants** tab — Main = one canonical card per figure (+ the
  embed-preview DECIDE card); the 9 alternative builds (Fig 3 pca20/antact, Fig 4 + Fig 5
  axis×norm grids) moved there. Curated tabs are module-level lists in `serve_figures.py`:
  restart the server after editing them.
