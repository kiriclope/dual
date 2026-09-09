# Honest dimensionality — cvPCA + shattering + PC coding (Fig 2)

> **NAMING CANON since 2026-09-09 (Leon).** The Go/NoGo trials are NOT called "distractors" any more: prose says the
> **Go/NoGo odor** / **Go/NoGo task**, and **GNG** for the code, axis and compact labels; every figure label that read
> `dist` now reads **GNG**, and the trial-timeline event in Figs 2a/3a is **GNG**. CACHE KEYS ARE UNCHANGED — `dist`
> (PM_PLANE, E_VARS) and `gng` (dPCA / CCGD caches) are still the keys; scripts map key → label at draw time
> (`CODE_NAME`/`CODE_ORDER` in the manifold scripts, `E_LABEL` in Fig 3c, the `'gng'→'GNG'` maps in Fig 2). Older
> "distractor" wording below is pre-rename; only the two sentences citing distractor studies (Jacob 2014,
> Parthasarathy 2017) and that reference title keep the word on purpose.


> **CANONICAL AXIS WINDOWS since 2026-09-08 (Leon: "same windows for all panels").** Sample and distractor decoder axes = bins **36–38** (6.0–6.5 s, post-distractor pre-cue); choice and test axes = bins **54–62** (9.0–10.5 s, test onset → 0.5 s after test offset) — the SAME bin indices in the overlaps (CCGD) and pseudo-population pipelines. Every "57–62 / 16–47 / 58–83 / 33–38 / 57–65 / 45–59 (opto depth)" axis window quoted below is the pre-flip convention (reachable with `--legacyaxes`); read-out windows (mid-delay 33–38 or 36–38, late delay 45–53 or 48–53, Fig 3b decision read 60–66) are unchanged. Numbers under the new axes: Fig 4 push β = −1.15 p = .007, coupling ρ = −0.72 p = .030 (norm battery all p ≤ .05, leave-one-out 4/9); Fig 3c choice out-vs-full a trend (p = .098); Fig 3e T/W 0.90 / 0.72; Fig 6 ΔGNG clustered p = .009. Variant table (windows A–E) and the flip recipe: memory `project_axis_windows.md`; text audit = draft v12.20 banner in `docs/paper/results_draft.md`.

> **2026-09-08 (later) — choice-decoder TRIAL SETS.** Fig 3c–e per-mouse choice classes (lick vs no-lick at the test) now pool ALL trial types (`--dpachoice` = former DPA-only); Fig 4a's choice side stays on the distractor-free DPA trials (Leon, option 2, after the dual-trial version proved heterogeneous across animals even with Go/NoGo-balanced classes; `--dualact` / `--allact` / `--strat` reproduce the alternatives). Fig 3c choice brackets ∗ (p = .012); Fig 4a per-mouse |cos| 0.063 → 0.104 (9/9, p = .004), cross-decode 0.53 → 0.60 (p = .020). Details: memory `project_dimensionality.md` / `project_overlaps_main_native.md`; draft banner v12.21.


> **2026-09-09 — Fig. 3 panel annotations (Leon).** Panel d no longer prints the `rel .xx/.xx` split-half
> reliabilities above the cosine matrices (they stay in the printed output and in Methods; the legend says the
> correction uses them), panel e no longer prints `T/W` in the matrix titles or on the per-mouse scatters (the
> transfer/within ratios stay in the legend), and panel b labels the sample axis under ALL FOUR expert-row panels
> instead of only the third.

**The non-circular dimensionality analysis of the dual-task pseudo-population, and — since 2026-08-10 —
main Fig 2** (`pca/fig_dimensionality_main.py`, which replaced `fig_dpca_story_main.py`; that dPCA build
is now ED Fig 9). **Fig 2 is built around ONE message: "one dedicated axis per task variable — the
working memory is a line."** Panels (the DECODE build, adopted 2026-08-10): **a** trial-timeline +
split-half cvPCA schematic (mid-delay bracket labelled 6.0–6.5 s = bins 36–38; the old 5.5–6.3 s label was wrong) · **b** 2×2 per-set reliable spectra —
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
  now ED 3a (was "3(a1)" pre-composition — see the 2026-09-02 ED-pages block below). A dot-strip-over-PR-bars variant was built first and REJECTED — don't rebuild.
  Draft §2, ED 3 list and `methods_notes.md` rewritten for the adoption (2026-08-10),
  `plot_dimensionality_main.py` (curated dual-vs-DPA composite `dimensionality.png`),
  `plot_dimensionality_fits.py [--gng] [--altwin]` (`dim_{DPA,dual,all}*.png`),
  `plot_dimensionality_scree.py`, `fig_bias_cleanup_ed.py` (**ED 3g** — was "3(h)", cache-only render from
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


### 2026-09-07: plane-vs-full "sufficiency" is BY CONSTRUCTION — text recast (analysis kept)
- Leon (artifact comment): "worried we are overselling… isn't it straightforward that we can decode from the
  plane and not outside it?" Verified in `exp_permouse_plane.py` (lines ~86–114): for sample and choice the
  FULL classifier (`make_clf`) is fit on the SAME half-split trials as the axis (`fit_axis` = `make_clf`) that
  spans the plane (the HH dict reuses the axis pools' split), so plane coordinate 1 IS the full decoder's
  output and the 2-feature plane classifier reproduces it by construction; the |Δ| ≤ 0.012 "bounded
  equivalence" is refit noise. Collapse without the plane is expected for a binary variable (one Δμ
  direction) — a weak "no further linear information in the residual" check.
- What panel C/D still shows: test code entirely OUTSIDE the plane; distractor PARTLY inside and drawn in with
  learning (0.57→0.65 ∗). The "one plane" claim rests on Fig 2b/2e/3e/3f.
- Applied 2026-09-07 (draft v12.6): §3 paragraph = consistency check + where the other variables live;
  "necessary and sufficient" removed from the Discussion and from the Fig 3 legend (CAP_PARAS title, panel c
  "What lives in the plane", panel d "The same pattern holds"); figure re-rendered. Same failure family as the
  retracted cross-mouse plane (2026-09-01): exact plane = full ⇒ suspect construction.
- **OUT-OF-CONTEXT plane test — BUILT 2026-09-07** (`exp_ooc_plane_pseudo.py` pooled, `exp_ooc_plane.py`
  per-mouse companion, `fig_ooc_plane.py`; caches `OOC_PLANE_PSEUDO_nopca` / `OOC_PLANE_nopca` in results.pkl;
  figure `figures/pseudo/dimensionality/png/fig_ooc_plane_nopca.png`). Design: plane = QR[sample axis @md,
  choice axis @decision] fitted on DPA trials of ONE stage (reference Naive-DPA | Expert-DPA), then in every
  other context (stage × trial type × window; sample ed/md/decision, choice decision) a 2-feature LR is refit
  on the fixed plane's coordinates and compared with a plane fitted IN that context (`iplane`, = the full
  decoder by construction) — ratio_ic = (fixed − .5)/(in-context − .5); `transfer` = the reference axis
  decoder with NO refit; `resid` = decoding with the plane projected out (stays ≈ full everywhere: population
  codes are redundant → NOT a necessity measure, reported only). Pooled (20 reps, K=24/48 pseudo-trials):
  out-of-context median ratio_ic sample 0.98 (ref Naive) / 1.00 (ref Expert), choice 1.01 / 1.02; no-refit
  median 0.86 / 0.86 (sample), 0.87 / 0.89 (choice); cells with ceiling < 0.60 (choice @md, sample @decision on
  dual trials) masked. NO-REFIT DROPS where the 2-D refit does not: expert NoGo sample @md 0.33–0.43 and
  expert DPA sample @decision 0.30 vs ratio_ic 0.85–1.06 → the sample code after the distractor / at the test
  is DISPLACED INSIDE the reference plane (the morph stays in the plane; the boundary moves). Per-mouse
  companion (20 reps): median ratio_ic over out-of-context cells 0.72 [IQR .66–.90] sample (n=9), 0.78
  [.67–.89] choice (n=8); fixed − in-context = −0.052 ± 0.011 (p<.01) / −0.042 ± 0.007 (p=.01); same-stage
  other-task cells ≈ 0.9–1.0, cross-stage cells 0.67–0.80 (per-mouse planes are fitted on 15–50 trials per
  class, so part of the shortfall is estimation noise; the pooled build is the clean statement, the per-mouse
  build the honest companion). Reference DPA-only pools make the other-task cells genuinely out of context.
  STATUS 2026-09-07 (late): Leon — "no new panel". Written in as ONE §3 paragraph (after the recast plane
  paragraph), a Methods paragraph (plane subsection) and ED 6e (`fig_ooc_plane_ed.py` → composed into
  `figures/ed/png/ed_fig6.png` by make_ed_figures.py; caption there + draft ED section). The full audit figure
  (`fig_ooc_plane.py`) stays in the repo/gallery `tmp/` only. Leon judged the full 4-matrix version "too much
  information" for a main panel — do not re-propose it for Fig 3.

### 2026-08-31 (later): Fig 3 gains panels E/F — plane sufficiency WITH stats (user decision)
> **STALE 2026-09-07:** the plane-vs-full "sufficiency"/"necessity" reading of these panels is BY CONSTRUCTION for
> sample and choice (see the 2026-09-07 block above); the panels stay but the text/legend now call them a
> consistency check and the non-circular statement is the out-of-context test (ED 6e).

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
Fig 4A's data). Caption title = the proven claim (necessary & sufficient + dist pulled in). **[STALE 2026-09-07: title no longer says necessary/sufficient — see the 2026-09-07 block.]**
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
baseline through the delay (the anticipatory push signature, cf. the opto figure's antact
preference — Fig 6 since the 2026-09-01 renumbering);
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
  the opto figure's stale caption/header stats (corrected to the rendered truth; see `docs/behavior.md`).
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
  embed-preview DECIDE card); the 9 alternative builds (Fig 3 pca20/antact, Fig 4 + opto (Fig 6)
  axis×norm grids) moved there. Curated tabs are module-level lists in `serve_figures.py`:
  restart the server after editing them.

### 2026-09-01/02: exploratory analyses (pca side) — containment WIN, cross-mouse RETRACTED
- **`exp_manifold_containment.py`** (cache `MANIFOLD_CONTAIN_nopca`; BCI within-manifold framing):
  the PUSH displacement vector is contained in the naive intrinsic manifold (top-10 naive
  single-trial PCs) at f=0.13 md — 4× the random-direction null (p=.0039) and indistinguishable
  from the held-out-naive ceiling 0.19 (p=.20, n=9) → consistent with within-manifold
  repositioning. The ROTATION vector sits below ceiling but that is NOT interpretable yet
  (axis-estimation noise biases containment toward the null — needs the within-stage split-half
  Δw noise floor). Ceiling low overall (calcium noise; FA upgrade queued). Full log in memory
  `project_main_figs_review_2026-08-30`.
- **`exp_xmouse_plane.py` — RETRACTED** (docstring carries the full reason): the cross-mouse
  plane "conservation" was circular by construction (own supervised axis + per-mouse
  standardisation force the same ±1 dumbbell in every mouse; within/cross classifiers make
  98.6%-identical predictions). Figure removed; do not cite.

### 2026-09-02: per-mouse cvPCA companion (ED 3c — was "3i" pre-composition) — built at user request
User asked "why don't we do the analysis at a per mouse level" (from the artifact-comment round
on the pseudo-population limitation). `exp_permouse_cvpca.py` (cache `PM_CVPCA`, estimator-free)
= Fig 2b's estimator run inside each mouse's own simultaneously recorded neurons (DPA 4-cond
set, md vs decision, 30 halvings). RESULT: wherever the reliable variance is resolvable
(reliable-total ≥ 5; 7/9 mice per stage), the memory spectrum is 1-D per animal — top-1 median
0.90 Naive / 0.93 Expert; within-mouse memory-vs-decision spread: Expert 0.93→0.61 Wilcoxon
p=.047 (6/7), Naive directional p=.22. Noise-limited cells (JawsM01-E, JawsM06 both, ACCM04-N
+ md-only cells) drawn open + excluded — the disclosure rule IS the honest answer to "why
pooled": those mice have <5 units of replicating variance at their trial counts. Wired in:
§2 sentence + Methods block + ED 3c entry + Discussion limitation (i) now cites it; figure
`fig_permouse_cvpca.png` pinned to gallery Supp. Do NOT star the Naive contrast.

### 2026-09-02 (later): the ED pages are COMPOSED — ED 3 relettered a–g
`/home/leon/dual/make_ed_figures.py` composes all 9 ED figures + the SI trial-count figure from
the existing component renders → `figures/ed/png/ed_fig{1..9}.png` + `si_trialcounts.png`
(native-resolution PIL mosaics; justified captions via `pca/figcaption.py` at 7.2 pt/400 dpi,
multi-column on wide pages; bold lowercase page letters in a 125 px left margin; PNG-only).
**ED 3 lettering canon since composition: a = previous PR build (was a1) · b = reduced-rank ·
c = per-mouse cvPCA (was i) · d = dim_all per-fit grid · e = dim_DPA_altwin window robustness ·
f = dim_DPA_gng cross-decode · g = bias cleanup (was h).** The old ED 3 (a)/(b)/(f) dPCA
descriptives (scree / marginal variance / shared-memory d′) have NO standalone renders — they
live in ED 9 (`fig_dpca_story_main.png`) and the ED 3 caption points there. In-text refs updated
(draft ×3, discussion ×1). Captions live in `make_ed_figures.py` AND the draft ED section — edit
together. Share: `figures/paper_share/ED_Fig*.{png,pdf}` + zip; artifacts: new "mPFC Dual-Task
Extended Data" page + figures-artifact footer + draft-artifact appendix all refreshed.

### 2026-09-03 — Nature-style pass (applies to every main figure; supersedes older layout notes above)
- **No claim-sentence panel titles** on the mains any more (user decision "go for it", 5-figure sweep,
  commit 22d0d4a): the legend's first sentence and its `a,` entries carry the claims; panels keep only
  orientation labels (condition / window / axis names). Fig 6's row banners are gone as well; its b/d/e
  panels carry bare "DPA"/"GNG" labels so they self-identify. Older mentions of "message-based titles"
  or "row banners" in this file are historical.
- **Panel letters are lowercase bold** (Nature final-artwork spec); in-text refs use `Fig. 1g` style.
- **Legends are Nature-format** ("Figure N | title." then `a, …` entries), geometry-voiced and in the
  human-voice register; they live in each script's `CAP_PARAS` AND in the draft's "## Figure legends"
  section — edit both together. The captioned PNG is the working/gallery build; `--nocap` is the
  submission build and `make_submission_figs.py` exports 183 mm vector PDFs.
- **Print scale**: each main defines `PS` (Fig 1 1.45 · Fig 2 1.15 · Fig 3 1.30 · Fig 4 1.10 via
  `main_panels.py` · Fig 6 1.35) so printed text lands at 5–7 pt. Convention in CLAUDE.md.

## 2026-09-07 (late) — reviewer-readiness fixes on Figs 2 and 3 + ED 6f (draft v12.9/v12.10)
Two review rounds (internal rigor, figure review, outside-referee simulation); the figure-side fixes that
touch this area, all cosmetic/labelling — NO statistic changed:
- **Fig 2a bracket** now prints the true mid-delay window `6.0–6.5 s` (bins 36–38). The PS ("parallelism
  score") text under panel e no longer carries the reliability-corrected "≈0.96–1.0 parallel" line: the
  corrected value rests on split-half reliabilities the panel does not show and read as an over-claim next
  to a sample transfer of only 0.06–0.56. The PS cache is untouched; the raw PS vs label-shuffle null stays.
- **Fig 2e legend + Results §2**: the sample transfer is stated as PARTIAL at mid-delay (0.06–0.56 of a
  small ceiling 0.60–0.66 on Go/NoGo trials, where the readout has faded after the distractor — Fig 3a) with
  the choice at 0.65–0.99; the "memory is not lost" claim in §3 now rests on ED 6e (refit-in-plane readout),
  not on Fig 2e. **Fig 2g** prints r = −0.02 (the caption/text said −0.03; the script's number wins).
- **Fig 3b window labels** corrected to `5.5–6.5 / 7.5–9.0 / 10.0–11.2 s`; **Fig 3c legend** gives the
  residual numbers (sample 0.70→0.56 p=.004; choice 0.67→0.61 p=.020) instead of "collapsed"; **Fig 3f**
  legend carries the per-animal transfer/within ratios (sample 0.86, choice 0.73).
- **Distractor plane SHARE** (Results §3 new sentence): chance-referenced share of the distractor signal
  carried by the plane, (plane−0.5)/(full−0.5), grows 0.19 → 0.35 (Wilcoxon p = .039, 7/9 mice), i.e. the
  plane-only rise (0.57→0.65) outpaces the full-population rise (0.79→0.86). Computed in-session from the
  per-mouse plane cache (`exp_permouse_plane.py` output) — not yet its own script.
- **Orthogonality floor**: §3 now quotes |cos| ≈ 0.07–0.09 against the ≈0.05 floor expected for random
  directions at these population sizes (was quoted bare).
- **ED 6f** = `pca/fig_manifold_supp.py` (four-code traces / per-mouse CCGP / per-mouse cross-task
  generalization) is now COMPOSED into the ED 6 page (`make_ed_figures.py` row f + caption) — it was
  referenced from the text but never on a page. ED 6 is now 6 rows (6955×20686 px, 51.7 in tall) — will need
  splitting for submission.
- The Fig 2e/3 changes are mirrored in the draft's "Figure legends" section (edit CAP_PARAS + draft together).

## 2026-09-08 — Fig 2c DECLUTTERED (Leon: "panel c has too much information")
Removed from `panelC_decode`: the hatched DPA-subspace distractor cross-decode bar (+ orange "weak
transfer" callout), the naïve null (dashed) lines, and the legend entries for both. Kept: expert bars, naïve
open circles, ONE null mark per bar (expert 95th percentile), the † for the naïve-only anticipatory choice
(judged against the naïve null, which is printed but not drawn), chance line, DPA | dual divider. The
cross-decode result (md expert 0.61, null95 0.59, p = .031; decision 0.63 p = .013; naïve md 0.47 n.s.)
is still printed (`C-dec: DPA-subspace …` lines) and now lives in panel d's orange column + its legend and
the §2 sentence "(Fig. 2d, orange column; permutation p = 0.031)". DPA group has 3 bars (sample/test/choice),
dual 4. No statistic changed. Draft v12.13 mirrors legends c/d.

**2026-09-08 (later) — Fig 2b/d edits (Leon):** panel b prints the window tag (mid-delay / decision) on
the DPA column too (callouts moved down to y=0.84 in all four spectra); panel d shows PC1–3 only
(`nk = 3` in `panelD_mats`; DPA PC4 was the degenerate 0 % direction, dual PC4 sits below the reliable
rank). No numbers changed. Panel c's shuffle null is being switched to the MATCHED null (see the next block
once `exp_dpca_count.py` finishes).

## 2026-09-08 (late) — Fig 2: MATCHED shuffle null (panel c), e/f moved to mid-delay/decision, c in b-format, g coloured
**Matched null (`exp_dpca_count.py`, Leon: "why is shuffle accuracy so high?").** The old null was the 95th pct
of SINGLE-SPLIT accuracies (40 DPA / 80 dual held-out pseudo-trials per shuffle → SD 0.06–0.10, null95
0.61–0.70) while the bar is a 15-split MEAN. Each shuffle is now scored as the same 15-split mean → null95
0.52–0.55 (SD ≈0.02); the cache keeps `null95_split` (old), `null_mean`, `null_sd`, `p`. Backup of the
pre-change cache: scratchpad `results.pkl.bak_before_matched_null`. Verdict changes in the drawn (md/decision)
cells: only naïve dual sample @ decision 0.63 → sig. Other windows (ed/delay/test) gained several sig flags
(e.g. naïve DPA choice @ ed 0.59, naïve dual test @ ed 0.56) — ED 3g re-rendered (its naïve-choice stars
unchanged in substance, null ≈0.53). Methods sentence + §2 "≈0.59" → "≈0.53" updated.
**e/f windows (Leon: "use mid-delay to be consistent with the rest of the figure").** Panel e now reads
`overlaps/figures/overlaps/ccgp/matrices_cache_mddec_acc_nopca.pkl` (`fig_ccgp_matrices_pseudo.py --mddec
--acc --nopca`: sample @ bins_MD 36–38, choice/test @ 57–65 via the new `W_DEC`/`AW['DEC']`) and panel f
`PM_GEN_mddec_nopca` (`exp_permouse_frame.py --nopca --mddec`, section d only, merged under the new key; the
old LD/TEST caches and keys are untouched for Figs 3/4 + ED). PS was already md/decision. RESULT: choice
0.73–1.01, test ≥0.72 (4 cells >1 hatched), sample 0.27–0.90 ASYMMETRIC (dual-trained → DPA 0.76–0.80;
DPA-trained → dual 0.27–0.44); within ceilings sample 0.94/0.78/0.80, test 0.79/0.70/0.68, choice
0.96/0.91/0.95. Panel f: sample Δ 0.00 [−.05,+.05] p=1.00; test +0.03 [−.00,+.06] p=.04 (8/9 up); choice
+0.04 [−.02,+.10] p=.20; within-task test 0.58→0.59, choice 0.64→0.69 (p=.16) → the per-mouse transfer
FRACTION is stable (medians sample .94→.92, test .87→.97, choice .99→.99; all p≥.31). The "all n.s. & bounded
±0.05" equivalence claim is RETIRED (draft v12.15). ED 6f (fig_manifold_supp.py panel C) repointed to the new
key and recomposed. Old-window numbers for the record: choice 0.65–0.99, sample 0.06–0.56, f all n.s.
**Layout:** panel c = 2×2 grid (DPA|dual × mid-delay|decision, `gsC` width_ratios [3,4]), legend (Expert /
Naive / null 95%) inside DPA mid-delay upper-right, † explained in the caption; panel e plain task labels
(ceilings → legend) and "PS x.xx" only (null → legend); panel g coloured by selectivity class (neither 50 % /
sample only 10 % / choice only 33 % / both 6 %; legend above the axes, short stats text).

**2026-09-08 (latest) — Fig 2 layout (Leon):** panel d matrices are SQUARE boxes (`aspect='auto'` +
`set_box_aspect(1)`, anchor C; row-d height ratio 0.84→0.70, `gsD` wspace 0.70→0.50) so the three rows
(a–c | d | e–g) share the same left/right extent; panel g now shows TWO clouds — every neuron coloured by the
axis with the larger |d′| (sample indigo 33 %, choice green 61 %) with the 207 both-selective neurons (6.2 %)
in orange on top; no separate "neither" colour; key stacked above the axes within the panel width, stats
text lower-left. Legend g updated in script + draft.

**2026-09-08 (latest+1) — DPA cross-decode column OUT of Fig 2d (Leon: "remove dist from panel d dpa, keep the
cross dec for figure 3").** `DCROSS = False` in `fig_dimensionality_main.py`: DPA matrices are 3×3
(sample/test/choice), no orange box; `DPA_GNG` / `DPA_GNG_C` caches unchanged and still printed by panel c.
The pooled number (0.61 @ md, p = .031) is quoted in the §3 plane paragraph next to the per-mouse Fig 3c
plane-share result; Fig 3 untouched. Draft v12.16.

## 2026-09-08 — Fig 3 RESTRUCTURED (Leon; draft v12.17)
- **b storyboard 2×4**: `B_SPECS` lost `('dual','delay')` — columns DPA mid-delay | DPA decision | dual mid-delay |
  dual decision. Legend/text: "two moments, mid-delay and decision".
- **old d (per-mouse plane grid) → ED 6g**: `panel_e_plane` is rendered standalone at the end of
  `fig_manifold_main.py` → `fig_manifold_permouse_plane{FIGSUF}.png/svg`; composed as ED 6 row g
  (`make_ed_figures.py`, caption = the old d legend). Text refs: "Fig. 3c,d" → "Fig. 3c and Extended Data Fig. 6g".
- **old e → d** (cosine matrices keep the dist axis; the per-mouse raw-|cos| scatters are the two SAMPLE pairs only —
  the choice × dist scatter duplicated Fig 4a's starred test and was dropped); **old f → e**. D and E are STACKED in
  the block beside C (`gsR` 2×1 → 4 squares per row), figsize (12.4, 12.4), height_ratios [1.45, 1.55, 1.55].
- References relettered (3e→3d, 3f→3e) in results/discussion/Methods/ED 6 title; ED 6 f-caption "Fig. 3c–e".
  No statistics changed. ED 6 page is now 58 in tall (7 rows) — must be split before submission.

**2026-09-08 (Fig 3d, v12.18):** the cosine matrices are 2×2 (sample × choice; `C3[:2,:2]`, rel label shows the
first two reliabilities; the full 3×3 is still printed). Per-mouse scatters unchanged (memory axis vs choice, vs
dist). The pooled 0.32→0.47 choice×dist overlap is quoted in §3 text only; its per-animal test is Fig 4a.
GOTCHA: figure references wrapped across lines ("Fig.\n3e") escape plain-string relettering — use
`re.sub(r'Fig\.\s*\n\s*3e', …)`; one "Fig. 3c–f" survived to v12.18 and was fixed.

**2026-09-08 (Fig 3d, v12.19):** the per-mouse scatter block is ONE panel (sample × choice raw |cos|); `PAIRS`
has one entry, `gsE2 = gsDrow[0, 3:4]` (slot 5 empty so the square aligns with E's third square). §3 cites Fig 3d
for sample×choice (0.07–0.08) and gives sample×dist (0.09) as text.
