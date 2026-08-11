# Honest dimensionality — cvPCA + shattering + PC coding (Fig 2)

**The non-circular dimensionality analysis of the dual-task pseudo-population, and — since 2026-08-10 —
main Fig 2** (`pca/fig_dimensionality_main.py`, which replaced `fig_dpca_story_main.py`; that dPCA build
is now ED Fig 9). **Fig 2 is built around ONE message: "one dedicated axis per task variable — the
working memory is a line."** Panels: **a** trial-timeline + split-half cvPCA schematic · **b** THREE mini reliable spectra mapping
1:1 onto c's bars — memory (DPA delay; a single component = the line, visually) · delay (all tasks) ·
decision (all tasks) — Naive/Expert + shuffled null (÷ real total), LINEAR fraction axis (log axis
rejected), condition set named in every title (user: B's set was ambiguous vs D's dual/DPA split) ·
**c** PR bars 1→2→3 with leave-one-mouse-out jackknife 95% CIs, Naive + Expert overlaid (= the
stability claim shown, not asserted) · **d** the η² PC-coding matrices, square cells, Expert row of
four (dual-delay · dual-decision · DPA-delay · DPA-decision). Off-message and therefore OUT of the
figure: shattering (→ ED 3 + cited from Results §3, where it pairs with CCGP), dPCA trajectories /
axis-mixing / linking plane (→ ED 9), Naive matrices + per-task-set fits (→ ED 3).

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
  (DPA/dual × md/decision), and **`DPA_GNG_C`** = the panel-C "gng ×" bar: Go/NoGo cross-decoded from
  the DPA-STATE SUBSPACE (top-3 DPA condition-mean PCs, LDA, held-out, shuffle null). KEY: at
  MID-delay the DPA subspace carries the distractor code only WEAKLY — Expert 0.61* / Naive 0.47 n.s.
  (decision 0.63*/0.61*); the older "~0.7 from PC2" figure came from the LATE-delay window and is
  consummatory-inflated.
- Renderers (no recompute, read `results.pkl`): `fig_dimensionality_main.py` — **the DECODE build IS
  main Fig 2 (ADOPTED 2026-08-10)**: B = 2×2 per-set cvPCA spectra (SPEC_JK: point + leave-one-mouse-out
  jackknife 95% CI from one source; common 6-component x-axis), C = per-variable decoding power
  (DPCA_COUNT + the hatched DPA "gng ×" from DPA_GNG_C), D = η² matrices DPA-first, PC1–4 both sets
  (DPA PC4 = degenerate ~0% row, shown for symmetry) + boxed gng × column (DPA_GNG). Grid = DPA vs
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
  `plot_dimensionality_scree.py`.

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
PC1 = sample .98 (the 1-D memory line IS the sample axis; its PC2/PC3 code future variables = noise,
gotcha 4 — flagged in the panel footnote)**; DPA-decision PC1 = choice .99. The FULL 2×4 grid (Naive &
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
