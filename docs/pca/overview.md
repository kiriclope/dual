# PCA Subproject Overview

## What PCA measures

Pseudo-population PCA projects all 9 mice into a shared low-dimensional subspace
to visualise how population geometry evolves across learning and tasks.

Three models, all saving `traj / labels / weights / evr` PKLs to `data/pca/`:

| Model | Script | Data format | Description |
|---|---|---|---|
| **single** | `run_single.py` | NaN-padded (`X_all_nan_`) | Per-mouse cv PCA; Procrustes-aligned across mice |
| **meta** | `run_meta.py` | Zero-padded (`X_all_<scale>`) | Joint cv PCA on pooled matrix |
| **pseudo** | `run_pseudo.py` | Zero-padded (`X_all_<scale>`) | Condition-averaged pseudo-pop PCA; no trial-count dilution |

The paper's **Figure 2E** uses the single-mouse pipeline.

---

## Running models — `run_pca.py`

Single entry point for all three models:

```bash
cd /home/leon/dual/pca

python run_pca.py single                           # default single run
python run_pca.py single --rebuild --scale std     # rebuild + std norm
python run_pca.py single --epoch DELAY --n-comp 6

python run_pca.py meta                             # default meta run
python run_pca.py meta --rebuild                   # build X_all_center + run
python run_pca.py meta --epoch DELAY --n-comp 6

python run_pca.py pseudo                           # default pseudo run
python run_pca.py pseudo --norm mad --n-comp 10

python run_pca.py --help                           # all models summary
python run_pca.py single --help                    # model-specific args
```

Each model can also be run directly (`python run_single.py`, etc.).

---

## Parameters and DUM construction

All three models share Phase 1 (`--rebuild`) and Phase 2 (cv PCA) args.

### Phase 1 — build data from raw (all models, `--rebuild` only)

| Arg | Default (single) | Default (meta/pseudo) | Effect |
|---|---|---|---|
| `--scale` | `''` | `center` | Per-neuron/day normalisation. Single: `''`/`std`. Meta/pseudo: `center`/`std`/`mad`/`none`. Tags X_all filename. |
| `--scaler-bl` | `center_BL` | `standard_BL` | BL correction inside `get_X_y_days` |
| `--days` | `first last` | `first last` | Days to include |

Single builds `X_all_nan_<scale>.pkl` (NaN-padded, per-mouse slots).
Meta/pseudo build `X_all_<scale>.pkl` (zero-padded) + `mouse_slices.pkl` via `build_padded_X`.

### Phase 2 — cv PCA (always applied)

| Arg | Default single | Default meta | Default pseudo | Effect |
|---|---|---|---|---|
| `--epoch` | `TEST` | `TEST` | `DELAY` | Fit window (pseudo also accepts `ED`, `CHOICE`, `POST_GNG`, `ALL`; `POST_GNG` is written `POSTGNG` in the DUM since `_` is the field separator. `ALL` = whole-timeline basis — `epoch=None`, used for the latent-dynamics analysis) |
| `--stage` | `Expert` | `Expert` | `Expert` | Learning stage |
| `--n-comp` | `10` | `10` | `6` | Number of PCs |
| `--n-splits` | `-1` (LOO) | `5` | `5` | CV folds |
| `--n-repeats` | `1` | `10` | `10` | CV repeats |
| `--cv-scale` | `standard` | — | — | Within-fold z-score (single only) |
| `--correct` | `True` | — | — | Correct trials only (single only) |
| `--norm` | — | — | `zscore` | Per-neuron norm in condition averages (pseudo only) |
| `--mouse-gain` | — | `None` | — | Mouse contribution normalisation (meta only) |

### Default DUMs (match existing result files)

| Model | Default DUM |
|---|---|
| single | `pca_TEST_Expert_standard_loo_correct_odor_pair` |
| meta | `meta_TEST_Expert_center_5x10` |
| pseudo | `pseudo_DELAY_Expert_zscore_5x10` |

Non-default `--scale` appends `_scale_<val>` to the DUM; non-default `--scaler-bl` appends `_<val>`.
For pseudo, `--preprocess 0` appends `_raw`, non-default `--factors` (≠ `odor_pair tasks`)
appends `_f-<a>-<b>-…` (e.g. `_f-sample-test-choice`), `--pert-ref` appends `_pertref`, and
`--remove-ci Q` appends `_ci<Q>` (plus the window when `--ci-epoch` ≠ `all`, e.g. `_ci2test`), and
`--dpca` appends `_dpca` (plus `_q<N>` when `--dpca-q` ≠ 2) — so every basis- or projection-changing option is distinct on disk and
in the figure folder tree.

Build-time scaling (`--scale`): `center` subtracts the per-day mean PSTH **at every time bin**
(removes condition-independent *and* condition-dependent structure — pushes e.g. DPA negative on
the choice axis during the GNG cue, since the grand mean carries the Go/NoGo distractor that DPA
lacks); `blcenter` subtracts the per-day **baseline** mean only (no artifact, but the common-mode
ramp survives); `std`/`mad` add clipped scaling; `none` is raw.

### Perturbed-trial projection (pseudo)

The basis is fit on clean trials only; clean scores are cross-validated (held-out
fold projection, Procrustes-aligned to the reference, averaged over repeats —
one out-of-sample row per trial, no duplication). Perturbed / non-clean trials
never enter any fit, so they have no in-sample bias either way:

- **default (fold-wise)** — projected through *every* fold's basis, aligned and
  averaged, exactly like the clean trials → clean and perturbed share one frame.
- **`--pert-ref`** — projected once through the reference (all-clean) basis;
  marginally lower-variance but in a slightly different frame than the clean rows.

The two differ only modestly (group-mean trajectories correlate ~0.999, up to
~10% local amplitude at the peak; the gap shrinks with more repeats). Fold-wise
is the consistency-preferring default; `--pert-ref` reproduces the older behaviour.

---

## Data files (`/home/leon/dual/data/pca/`)

### Input (build phase)

| File | Used by | Description |
|---|---|---|
| `X_all_nan_<scale>.pkl` | single | NaN-padded (trials, n_neurons_total, 84); NaN outside each mouse's neurons |
| `X_all_<scale>.pkl` | meta, pseudo | Zero-padded (trials, n_neurons_total, 84) |
| `y_all_nan_<scale>.pkl` | single | Trial metadata DataFrame |
| `y_all_<scale>.pkl` | meta, pseudo | Trial metadata DataFrame |
| `mouse_slices.pkl` | meta, pseudo | `{mouse: slice}` neuron index map |

`X_all_center.pkl` and `mouse_slices.pkl` currently live at
`/home/leon/dual_task/dual_data/data/pca/` (old path). Run with `--rebuild`
to regenerate in `../data/pca/`, or point `--data-dir` at the old path.

### Output (result files)

| File | Shape | Description |
|---|---|---|
| `single_traj_<dum>.pkl` | (9216, 10, 84) | Single-mouse PC projections — (trials, n_comp, n_time) |
| `single_labels_<dum>.pkl` | (9216, 19) | Trial metadata DataFrame |
| `single_weights_<dum>.pkl` | (n_comp, n_neurons_total) | Fold-averaged loadings |
| `single_evr_<dum>.pkl` | (9, n_comp) | EVR per mouse |
| `meta_traj_<dum>.pkl` | (n_trials, n_comp, 84) | Meta PC projections |
| `meta_labels_<dum>.pkl` | DataFrame | Trial metadata |
| `meta_weights_<dum>.pkl` | (n_comp, n_neurons_total) | Reference fold loadings |
| `meta_evr_<dum>.pkl` | (n_folds, n_comp) | EVR per fold |
| `pseudo_traj_<dum>.pkl` | (n_trials, n_comp, 84) | Pseudo PC projections |
| `pseudo_labels_<dum>.pkl` | DataFrame | Trial metadata |
| `pseudo_weights_<dum>.pkl` | (n_comp, n_neurons_total) | Reference loadings (decoders if `--dpca`) |
| `pseudo_evr_<dum>.pkl` | (n_folds, n_comp) | EVR per fold |
| `pseudo_marglabels_<dum>.pkl` | list[str] | dPCA only — marginal name per component (`time`/`sample`/`test`/`sample:test`) |

Key `labels` columns: `mouse`, `day`, `learning`, `stage`, `tasks`, `laser`,
`performance`, `odr_perf`, `sample`, `sample_odor`, `choice`, `odr_choice`,
`odor_pair`, `test_odor`.

---

## Sample × lick axis space (Figure 3 / `decode/fig3BF.py`)

For attractor and boundary analyses, the 10-PC trajectories are projected onto
two interpretable axes defined per mouse from the last 2 expert days:

- **Sample axis** (`sa`): LR decoder trained on sample identity (A vs B), oriented B→positive.
- **Lick axis** (`la`): LR decoder on GNG `odr_choice` (Dual trials only), orthogonalised to `sa`, oriented lick→positive.

Projections: `ps = traj @ sa` (n_trials, 84), `pl = traj @ la` (n_trials, 84).

---

## Plot scripts

| Script | Shows | Data loaded | Shared primitives |
|---|---|---|---|
| `plot_pseudo_traj.py` | 1D PC trajectories split by odor_pair / tasks / sample / choice / test | `pseudo_{traj,labels}_<dum>` from `../data/pca/` | `plot_mean_sem` |
| `plot_pseudo_traj2d.py` | 2D PC-plane trajectories (PC1-2, PC1-3, PC2-3) as time-gradient paths with arrows, same splits | `pseudo_{traj,labels}_<dum>` from `../data/pca/` | `plot_gradient_line`, `add_arrows` (via `src.pca.plot.plot_trajectories_2d`) |
| `plot_pseudo_state2d.py` | 2×3 grid (Naive/Expert × DPA/Go/NoGo) of sample-PC × choice-PC paths + choice-PC KDE strip (overlaps `plot_traj2d.py` analog) | `pseudo_{traj,labels}_<dum>` from `../data/pca/` | `sem_band`, `plot_gradient_line`, `add_arrows` |
| `plot_pseudo_loadings.py` | EVR, loadings vs θ, weight planes, per-mouse loading energy | `pseudo_{evr,weights}_<dum>` + `mouse_slices` from `../data/pca/` | — |
| `plot_pseudo_mixing.py` | Task-component mixing: coding strength per PC + variable×variable `\|cos\|` heatmap | `pseudo_{traj,labels}_<dum>` from `../data/pca/` | — |
| `plot_pseudo_cross.py` | Cross-projection: one run's trials on another run's PCs (e.g. none data on center PCs), 1D traj | `X_all_<scale>` + `pseudo_{traj,labels}_<basis-dum>` from `../data/pca/` | `plot_mean_sem` |
| `plot_single_individual.py` | Per-mouse EVR, 1D PC traces, loadings vs θ | `single_traj_<dum>` from `results/` | `plot_mean_sem` |
| `plot_meta_individual.py` | Per-mouse meta-PCA EVR, 1D PC traces | `meta_traj_<dum>` from `results/` | `plot_mean_sem` |
| `decode/fig3BF.py` | Attractor centroids (B-D) and boundary distances (E-F) | — | — |

### `plot_pseudo_*` scripts (current pseudo pipeline)

All take `--dum` (selects which `pseudo_*_<dum>.pkl` to load) and `--data-dir`
(default `../data/pca`). The **scale** is parsed from the DUM (`_scale_<x>`;
untagged = the default `center`) and used to organise output folders.

`plot_pseudo_traj.py` and `plot_pseudo_traj2d.py` share trial-selection flags:
`--stage {Expert,Naive}` (which trials to project/plot — the basis itself is
always the Expert clean fit), `--correct` / `--no-correct`, `--laser {0,1,all}`,
`--no-bl-correct`. `traj2d` also has `--t-start` / `--t-end` to window the path.
The plotted selection (stage + non-default laser/correct) is encoded in the
filename via `<SEL>`. `plot_pseudo_loadings.py` only takes `--dum`/`--data-dir`/
`--n-show` (EVR/loadings are basis properties, independent of trial selection).

`plot_pseudo_traj.py --relevant` plots the identified task PCs (Sample / Choice /
Test) **by role** instead of the first `--n-show` PCs by index — useful when the
informative PC is not in the top few (e.g. for the `none` run Choice is PC5).
These figures get a `_relPCs` filename suffix so they sit beside the index-based
ones.

`plot_pseudo_state2d.py` draws the overlaps-style 2×3 trajectory grid; its x/y
PCs default to the **identified** Sample / Choice PCs (see below) and can be
overridden with `--sample-pc` / `--choice-pc`. The KDE strip is the choice-PC
location over the delay, split by sample.

Figures are organised by **epoch** then **scale** then **ci** then **factor set**,
all parsed from the DUM (`pseudo_<EPOCH>_…` → epoch lowercased, e.g. `delay`/`test`;
`_scale_<x>` → scale, untagged = `center`; `_ci<Q>` → `ci<Q>`, untagged = `ci0`;
`_f-<factors>` → factor, untagged = `odor_pair-tasks`). So runs with different
`--epoch`, `--scale`, `--remove-ci` or `--factors` never mix.

Figure layout (PNG dpi=300 + SVG):
```
figures/pseudo/traj/<epoch>/<scale>/<ci>/<factor>/<stage>/{png,svg}/<dum>_<SEL>_<split>[_relPCs].{png,svg}
figures/pseudo/traj2d/<epoch>/<scale>/<ci>/<factor>/<stage>/{png,svg}/<dum>_<SEL>_<split>.{png,svg}
figures/pseudo/state2d/<epoch>/<scale>/<ci>/<factor>/{png,svg}/<dum>_pc<sx>x<cy>.{png,svg}
figures/pseudo/evr/<epoch>/<scale>/<ci>/<factor>/{png,svg}/<dum>_evr.{png,svg}
figures/pseudo/loadings/<epoch>/<scale>/<ci>/<factor>/{png,svg}/<dum>_{theta,2d,energy}.{png,svg}
figures/pseudo/mixing/<epoch>/<scale>/<ci>/<factor>/{png,svg}/<dum>_{mixing,mixing_time}.{png,svg}
```
EVR/loadings/state2d/mixing sit at the epoch/scale/ci/factor level only (basis
properties, stage-independent).

`plot_pseudo_traj2d.py` calls `src.pca.plot.plot_trajectories_2d`, which sets
axis limits explicitly from the trajectory extent (a `LineCollection` does not
drive autoscale, and `ax.relim()` ignores collections).

### PC identification (`src/pca/identify.py`)

Which PC carries which task variable is **run-dependent** (a DELAY-epoch fit
puts Choice on PC1 and Sample on PC3; a TEST-epoch fit orders them differently),
so it is measured from the projected trajectories, not assumed. Over the 4 odor
pairs the variables are orthogonal ±1 contrasts:

| Variable | Pairs (+) vs (−) | scored in window |
|---|---|---|
| Sample (A vs B) | {0,1} vs {2,3} | delay |
| Choice (lick vs no) | {0,2} vs {1,3} | test |
| Test (C vs D) | {0,3} vs {1,2} | test |

`identify_pcs(X, y, stage='Expert')` returns a per-PC label list by assigning
each variable 1:1 to the PC carrying the largest fraction of its contrast
energy; `pc_label(k, labels)` formats `'PC 1 (Choice)'`. All `plot_pseudo_*`
scripts call this (on the Expert basis trials) so every PC axis is annotated
with the variable it encodes. For the default DELAY run the mapping is
**PC1=Choice, PC2=Test, PC3=Sample**.

### Component mixing (`src/pca/identify.py`, `plot_pseudo_mixing.py`)

How cleanly the PCs demix the task variables. Each variable's **coding vector**
is its per-PC contrast score (the direction in PC space along which it is
expressed); the mixing between two variables is the `|cos|` angle between their
coding vectors (1 = same direction, 0 = orthogonal/demixed).

- `coding_vectors(X, y)` → `(n_var, n_comp)` contrast score matrix.
- `variable_mixing(X, y)` → `(M, C, names)`; `M` is the `n_var×n_var` `|cos|` matrix.
- `participation_ratio(C)` → per-variable effective #PCs it spreads over (1 = on one PC).
- `mixing_index(M)` → scalar = mean off-diagonal `|cos|` (0 = fully demixed).
- `variable_mixing_time(X, y)` → `(M, C, energy, names)`; per-time-bin `|cos|`
  plus per-variable coding-vector `energy` (`|cos|` is only meaningful where the
  energy is non-trivial — e.g. not pre-stimulus).

`plot_pseudo_mixing.py --dum <run>` renders two figures: `<dum>_mixing` (coding
strength per PC + variable×variable `|cos|` heatmaps, with the mixing index) and
`<dum>_mixing_time` (pairwise `|cos|` over time above, per-variable coding energy
below). Findings (DELAY fit): `center` + tasks-inclusive factors demix best
(index ≈ 0.23); `none` runs and the no-`tasks` `sample-test-choice` run mix more
(≈ 0.27–0.32). Sample↔Choice are near-orthogonal everywhere; Sample↔Test carry
the irreducible mixing (related odor identities). The time-resolved view shows
much of the demixing is **temporal** — Sample coding peaks in the delay, Choice
only at test. A **TEST-epoch fit** demixes even better (index ≈ 0.17) and sharpens
the Test axis, but spends PC1 on a non-discriminative test-evoked common mode and
pushes Choice down to PC4 (DELAY fit: PC1=Choice, PC2=Test, PC3=Sample;
TEST fit: PC1=common, PC2=Test, PC3=Sample, PC4=Choice).

### Cross-projection (`plot_pseudo_cross.py`)

Project one run's trials onto another run's PCs — e.g. the `none` (raw) trials on
the `center` PCs. The basis (`W`, per-neuron `mean`/`scale`) is re-fit on the
`--basis-scale` clean data (it reproduces the saved basis exactly, `|cos|`=1),
then the `--data-scale` trials are projected through it via `project_trials` and
the 1D trajectories are plotted.

```bash
python plot_pseudo_cross.py --basis-scale center --data-scale none
# also: --epoch --factors --norm --n-comp --stage --relevant --no-bl-correct
```
Output: `figures/pseudo/cross/<data>_on_<basis>/<epoch>/<factor>/<stage>/{png,svg}/`
(the `<factor>` level keeps different `--factors` bases from colliding).

Two subtleties: (1) the projected data is genuinely **out-of-sample** (it did not
define the basis), so no leakage. (2) The PC **identity** is read from the basis
run's saved **held-out** traj, *not* from an in-sample re-projection of the basis
data — the latter overfits e.g. Choice onto near-degenerate high PCs.

Finding (raw `none` data on `center` PCs): the Sample axis is recovered, but the
raw common-mode ramp bleeds in along whichever axes have the most variance — and
**this depends on the factor set**, because the factors decide which PC is Sample:

| factors | Sample PC | what the raw data does on it |
|---|---|---|
| `odor_pair-tasks` | PC3 | clean, ~symmetric A/B; the ramp lands on PC1/PC2 instead |
| `sample-test-choice-tasks` | PC3 | same as above (≈ odor_pair-tasks) |
| `sample-test-choice` | **PC1** | A/B separate but **both ride the ramp upward** — Sample is now the leading high-variance axis, exactly where the raw drift concentrates |

So putting Sample on the top PC (via `sample-test-choice` factors) makes it *more*
susceptible to raw-drift contamination when uncentered data is projected;
centering's main job is removing that nuisance common mode.

### Centering artifact and `--remove-ci`

The common-mode ramp and the centering interact badly. `--scale center` subtracts
the per-day mean PSTH at **every time bin**; because that mean pools DPA + Go +
NoGo, during the GNG cue it carries the distractor response (Go/NoGo only), so DPA
— which has no distractor — is pushed **negative** on any PC loading those neurons
(e.g. the Choice PC). `--scale blcenter` (baseline-only) avoids the artifact but
leaves the ramp, which then dominates the weaker task axes (Choice can be ~93%
common-mode).

`--remove-ci Q` resolves both. It projects out the top-`Q` **condition-independent**
(ramp) directions per mouse before the fit (`remove_ci_subspace` in
`src/pca/pseudo.py`): the per-mouse condition-independent marginal (equal-weighted
mean over the factor conditions, time-centred) is SVD'd, and its top-`Q`
neuron-space directions are projected out — `X' = X − U(UᵀX)`. Because this is a
**fixed-direction projection**, not a per-time subtraction, a trial only loses its
own component along the ramp directions, so DPA is never pushed negative.

Recommended: `--scale blcenter --remove-ci 2` (or `3`). Comparison (DELAY, center
basis):

| config | mixing | DPA on Choice PC | Choice common/disc | artifact? |
|---|---|---|---|---|
| `center` | 0.233 | **−1.95** | 0.03 | ❌ |
| `blcenter` | 0.289 | +1.70 | 14.6 | ✅ but ramp |
| `blcenter --remove-ci 2` | **0.216** | +0.16 | 3.1 | ✅ |
| `blcenter --remove-ci 3` | 0.223 | −0.36 | 2.1 (Test 2.4→0.56) | ✅ |

`Q` is the knob: more directions strip more ramp but eventually touch
condition-dependent structure (small DPA push); the mixing index also jitters
±0.05 run-to-run at `--n-repeats 1` (only 5 CV folds). `blcenter --remove-ci` is
the only DELAY config with **no artifact AND ramp-suppressed task axes**.

**`--ci-epoch`** (default `all`) controls the window the CI directions are
estimated from. `all` (whole timeline) is best — it captures the genuinely
condition-independent **global ramp**. Restricting it to a specific window
(e.g. `--ci-epoch TEST`) backfires: that window's marginal carries
condition-*dependent* structure (the delay distractor, or the test-evoked
response that is collinear with the C/D signal), so projecting it out degrades
demixing. A non-`all` window is tagged in the DUM (`_ci<Q><window>`).

**Epoch matters for whether to demix at all.** For a **DELAY** fit the cue is
condition-*dependent* (absent in DPA) → `center` is artifactual → use
`blcenter --remove-ci`. For a **TEST** fit the test response is
condition-*independent* (present in every condition) → `center` removes it
cleanly with no artifact and demixes best (mixing ≈ 0.17); CI-removal there only
hurts. A **POST_GNG** fit (bins 48–65, overlapping the test period) behaves like
TEST: plain `blcenter` (ci0) already demixes well (mixing ≈ 0.18, no artifact) and
CI-removal degrades it (ci0→3: mixing 0.18 → 0.33 / 0.25 / 0.29; DPA push grows).

So the rule: **`blcenter --remove-ci` only for the pure DELAY fit**; for TEST /
POST_GNG the common mode segregates onto PC1 on its own, so plain `center`
(TEST) or `blcenter` (POST_GNG) is best and CI-removal should be left off.

### Demixed PCA — `--dpca` (`src/pca/dpca.py`)

`--remove-ci` only strips the top ~half of the condition-independent component
(its spectrum is high-dimensional — top-3 ≈ 56%), so the task axes still carry
stimulus timing; and a PCA "choice" axis fit on the delay is mostly that timing
(choice separation ≈ 0.1 vs common-mode ≈ 0.2–2.5). **dPCA** (Kobak et al. 2016)
solves both: it decomposes the condition-averaged pseudo-population into
orthogonal marginalisations — `time` (condition-independent) + one per factor +
their interactions — and fits a **reduced-rank regression decoder** per marginal,
so each task axis captures only its own variance, demixed from the timing.

- `dpca_decode(P, sizes, factors, q, ridge)` — marginals + decoders. Reduced-rank
  regression via the SVD-of-X form (ridge-regularised since neurons ≫ samples);
  each decoder's top-`q` axes come from a **truncated** `randomized_svd` (only `q`
  vectors needed) — ~2.5×+ faster than a full SVD, essential for the `sample test
  tasks` design (16 components: >13 min → ~2.6 min).
- `cv_dpca(...)` — drop-in CV wrapper, same return shape as `cv_pca_pseudo` plus
  the per-component marginal-label list (saved as `pseudo_marglabels_<dum>.pkl`).

**Two correctness requirements (learned the hard way):**

1. **Fit on the whole trial, with held-out CV.** Fitting on a single window
   (e.g. DELAY) where test/choice have no signal, and projecting in-sample, makes
   the reduced-rank regression model that window's *noise* and spuriously "decode"
   test/choice *before they are presented*. `cv_dpca` therefore (a) fits the whole
   timeline and (b) projects **held-out** trials (refit per fold, per-marginal
   Procrustes-align, average) so overfit noise does not generalise. So **`--dpca`
   ignores `--epoch`** (it marginalises over time anyway) — the DUM epoch field is
   forced to `ALL` and a note is printed. A within-sample label shuffle confirms
   choice is genuinely *not* decodable in the delay (real ≈ null).

2. **Complete factorial `--factors`** (the marginals must sum to the data), e.g.
   `--factors sample test` (spans the 4 odor pairs). The **choice signal is the
   `sample:test` interaction** (lick iff sample and test match), so it gets its own
   demixed axis; `identify_pcs` independently labels it `Choice`, so all plot
   scripts work unchanged.

**Component organisation.** dPCA components are **not variance-ordered** — they are
grouped by marginalisation in a fixed order: main effects (`sample`, `test`) →
interactions (`sample:test` = Choice, then `*:tasks` …) → `time` (condition-
independent) **last**. `--dpca-q` = the rank kept *per marginal* (the dPCA analog
of "how many PCs", but per variable). `q=1` → one axis each (PC1=sample, PC2=test,
PC3=choice, PC4=time); `q=2` (default) → two each (e.g. a transient onset + a
sustained memory axis for sample). The component number tells you the marginal,
not which has most variance — `time` is usually the largest yet sits last.
`--dpca-ridge` = ridge fraction.

Because of this, the EVR figure for a dPCA run is **not** a scree line:
`plot_pseudo_loadings.py` detects the `pseudo_marglabels` sidecar and draws a bar
chart of **variance explained per marginalisation** (summed over its `q` axes,
annotated), so you read off how much signal each variable accounts for (typically
`time` ≫ task variables).

```bash
python run_pseudo.py --scale blcenter --factors sample test --dpca           # q=2 (default)
python run_pseudo.py --scale blcenter --factors sample test --dpca --dpca-q 1 # one axis per variable
python run_pseudo.py --scale blcenter --factors sample test tasks --dpca      # + a clean tasks axis
```

Result: each axis separates exactly its variable **at its real time**, with no
stimulus timing — Sample is sustained through the delay (memory), while **Choice
and Test are flat until test onset** then separate (delay sep ≈ 0.03–0.23 ≈ the
shuffle null; was ≈ 1.8 with the in-sample, single-window fit). The `time`
marginal absorbs most of the variance. Figures land under
`figures/pseudo/<type>/all/<scale>/ci0/<factors>_dpca/…`.

**Note**: the older `plot_single_individual.py` / `plot_meta_individual.py`
still load from `pca/results/` (old path); update their `RESULTS` when switching
to the consolidated `../data/pca/` layout.

---

## Latent dynamics (LDS) — `src/pca/dynamics.py` + `pca/plot_pseudo_dynamics.py`

Asks **how low-dimensional and how fast** the population dynamics are by fitting a
first-order linear flow to the saved latent trajectories:

    z_{t+1} = A z_t + B u_t + b

(`fit_lds`, ridge LS over all clean trials × time steps; bias unregularised). It
is a two-stage *latents-then-dynamics* linear model — the average linear flow, not
a single-trial latent-inference model (GPFA / LDS-EM).

**API.** `fit_lds(Z, U=None)` → `A, B, b`; `lds_modes(A, dt)` → eigenvalues +
time constant τ (s, from `|λ|`) + frequency (Hz, from the phase); `cv_predict_r2`
= held-out **h-step** predictive R² (rolls the flow forward h steps, feeding the
known `u_t` at each step); `dimensionality_curve` = R² vs number of leading
latent dims; `boxcar_inputs` builds onset regressors.

**Use a whole-timeline basis.** Fit the PCA with **`--epoch ALL`** (added for this)
so the latents span the full trajectory. A single-epoch basis (e.g. DELAY) biases
the directions toward that epoch and makes the flow look *artificially slower and
lower-dim*. `center` scale also distorts the flow (the demean artifact) — use
**`blcenter`**. The canonical run is `pseudo_ALL_Expert_zscore_5x1_scale_blcenter`.

**Why multi-step horizons.** A 1-step R² is saturated (`z_{t+1}≈z_t` when `|λ|≈1`),
so it always looks ~1.0. The dimension where the **2 s-horizon** R² plateaus (or
peaks then *drops* as extra dims overfit) is the effective dynamical
dimensionality. On the whole-timeline blcenter PCA it **peaks at ~2 dims** — the
predictable dynamics are low-dimensional even on a basis that includes the
stimulus transients.

**Two modes (auto-detected from the DUM):**

- **PCA run** → eigenvalue spectrum + dimensionality curve (autonomous solid,
  `+inputs` dashed) over the leading PCs.
- **dPCA run** (a `pseudo_marglabels` sidecar exists) → the axes are grouped by
  marginal, so a "top-k" curve is meaningless; instead the LDS is fit **separately
  in the task subspace** (sample / test / sample:test axes) **and the time
  subspace** (the condition-independent timing axes), comparing their flow.

**The input term `B u_t`.** Timing-locked stimulus regressors test whether
imperfect long-horizon prediction is unmodeled stimulus drive vs genuine dynamics.
Two flavours, applied per subspace: a **shared** condition-independent boxcar
(`boxcar_inputs`, sample/distractor/test onsets) for the `time` axis, and
**per-trial SIGNED** inputs (`signed_inputs`: ±1 for which sample/test/choice the
trial saw, at the onset window) for the `task` axes — because dPCA removes the
time marginal from the task axes, so they respond only to condition-*dependent*
drive. `U` may be `(n_inputs, n_time)` shared or `(n_trials, n_inputs, n_time)`
per-trial; `fit_lds`/`cv_predict_r2` handle both.

**Findings (canonical blcenter run).** The **task/WM** dynamics are non-rotational
(integrator / line-attractor-like) — confirmed by the nonlinear fixed-point analysis below
(all WM wells are real-eigenvalue nodes, freq=0). ⚠ **Correction (2026-06-17):** the *global*
top-6 LDS is **not** strictly f≈0 — it carries a weak slow **rotational** pair (|λ|≈0.995,
~0.038 Hz, ~½ turn/trial) that belongs to the **condition-independent timing/ramp** (the
curved arc seen in raw PCA), not the memory. So "non-rotational" holds for the *task* code,
not the global timing. The two regimes are otherwise distinct:

| subspace | dims | τ | held-out R² @ 0.17 / 0.83 / 2.0 s | inputs help? |
|---|---|---|---|---|
| **time** (common-mode ramp) | 2 | ≈22 s | 0.98 / 0.92 / 0.78 | slightly (→0.79) |
| **task** (sample/test/choice) | 6 | ≈3–6 s | 0.92 / 0.76 / 0.49 | **no** (→0.50) |

The input term **overturned the earlier "long-horizon limited by unmodeled
inputs" caveat**: feeding stimulus inputs recovers essentially *none* of the task
subspace's long-horizon gap (signed inputs: 0.489→0.500; condition-independent
boxcars: no change at all, by construction). So the task code's lower
predictability is **genuine faster dynamics + single-trial variability, not
stimulus surprise**. The slow timing/ramp subspace is the only one the inputs
(mildly) help. Net: dynamics are low-dim and slow overall, but split into a very
slow, predictable **timing/ramp** mode and a faster, autonomous **task-memory**
mode (τ≈3–6 s, matching working-memory decay).

```bash
python run_pseudo.py --scale blcenter --epoch ALL --n-comp 12   # build the basis
python plot_pseudo_dynamics.py --dum pseudo_ALL_Expert_zscore_5x1_scale_blcenter
python plot_pseudo_dynamics.py --dum pseudo_ALL_Expert_zscore_5x1_scale_blcenter_f-sample-test_dpca
```
Figures → `figures/pseudo/dynamics/<mode>/<scale>/<factor>/<stage>/{png,svg}/<dum>_dyn.{png,svg}`
(`<mode>` = `pca`/`dpca`, auto-detected; `<stage>` = `Expert`/`Naive`).

**2-D flow field — `pca/plot_pseudo_flow.py`.** Fits a 2-D flow on a chosen pair
of axes (`--dims sample choice`) and streamplots it, with the fixed points and the
condition-mean trajectories (split by sample/choice, sample-coloured) overlaid.
Axes are picked by **role name** — for a dPCA run mapped to the marginal axes
(`choice` = the `sample:test` interaction), for a PCA run resolved with
`identify_pcs` — or by raw index. Three modes:

- **Linear (default).** Best 2-D *linear* fit `Δz = (A − I) z + b` (other dims
  integrated out, **not** a 2×2 slice of the full `A`), one fixed point `z*`. The
  result is a **stable node** (`|λ|≈0.96–0.98 < 1`, `z*≈0`): a slow *memory leak*;
  where the trajectories peel off the streamlines marks **stimulus input** kicking
  the state against the decay.
- **`--per-sample`.** Two panels (A-trials / B-trials), each its own linear fit.
  The fixed point **displaces toward the sample location** (`z*≈(−0.9, ·)` for A,
  `(+0.8, ·)` for B): the A/B "attractors" are real but **condition-dependent** —
  pooling the symmetric ±1 axis averages them to `z*≈0` (one linear `A` cannot hold
  two fixed points). Equivalent to the input-driven shift `z*=(I−A)⁻¹(b+Bu)`.
- **`--nonlinear`.** The **autonomous** flow, showing **both attractors at once**
  (a linear `A` has exactly one fixed point). Fits a **neural rate-network flow**
  `Δz = L z + W φ(g·z) + b` (`--nonlin tanh`/`relu`/`logistic`; leak `L` + saturating
  recurrent feedback `W` = the textbook attractor-memory mechanism; linear in the
  params, closed-form ridge) or a polynomial flow (`--nonlin poly`).
  `flow_fixed_points` locates + classifies every fixed point by its Jacobian
  (● attractor / ✕ saddle / ○ repeller).

  The autonomous flow is **always fit on the DPA working-memory delay** — DPA-only,
  bins 21–54 (`autonomous_flow()`, shared with the `u=0` panel of `--inputs`). The
  DPA delay window matters: it has the sample-driven separation + memory but **not**
  the test-period choice divergence (over the whole trial the wells become *saddles*
  because choice is driven; DPA has no distractor to muddy the delay). `--fit-win`
  overrides the window; `--task` only subsets the overlaid trajectories, never the fit.

  **Fit data — single trials by default, `--cond-mean` to denoise** (filename `_cm`).
  ⚠ On this data single trials are **degenerate**: std≈3 noise reads as a
  leak-to-origin and the double well **collapses to one attractor** at ≈(−0.05, 0).
  **Pass `--cond-mean`** (condition means grouped by sample×test) to recover the
  bistable landscape — at the defaults (`tanh`, **gain 1.0, ridge 0.2**; the gain from
  the FP-stability sweep in `select_flow_model.py`, the ridge from a finer follow-up —
  ridge 0.2 has the best bootstrap stability, P2≈0.80, and a smoother field than 0.1)
  it gives **two attractors** at
  ≈`(−1.38, 0)` and `(+1.02, 0)` with a **saddle** at ≈`(0.24, 0)`.
- **`--inputs`.** Autonomous + input-driven **flow-field approximations** of the data, in
  an 8-panel grid (autonomous + A, B, Go, NoGo, Cue, C, D), to compare against the RNN's
  autonomous + input-driven flows. **Each panel independently fits one nonlinear rate
  flow** `Δz = Lz + Wφ(g·z) + b` (`fit_rnn_flow`, same as `--nonlinear`) to the data in
  that regime, finds its fixed points (discrete-map Jacobian), and overlays the condition
  trajectories:
  - **autonomous** — the no-input data (**DPA delay**, bins 21–53) → the bistable
    working-memory landscape (= `--nonlinear`);
  - **A/B** — `sample=0/1` trials, bins 15–29 (sample onset → into delay);
  - **Go/NoGo/Cue** — DualGo / DualNoGo / both, bins 30–51 (distractor → response peak);
  - **C/D** — `test=0/1` trials, bins 57–83 (test → response settle).

  Each window runs from the stimulus onset **through the response peak/settling** so the
  data actually reaches a near-zero-velocity state — this puts the fitted **attractors in
  the data** (a too-short window cuts off while the transient response is still moving, so
  the fit would place fixed points by extrapolation instead). With the full windows, A/B
  land at the sample wells, NoGo at the no-lick state, and **C/D give two attractors at
  the lick/no-lick endpoints (bimodal, in-support)**. (Go keeps one spurious extrapolated
  attractor — its response is transient and asymmetric.) Same procedure every panel, just
  a different data slice — no input model. Fits use **regime-aware condition means**
  (grouped by the factor actually manipulated in the window: sample alone pre-test → ~2×
  trials/mean, sample×test once the test odor is present; the **Cue panel groups by `tasks`**
  = Go/NoGo, its defining variable — otherwise it averages Go+NoGo and misses the cue);
  shared 92nd-pct colour scale; trajectories truncated at each window end. Each panel is
  **self-validating** (see below). `--inputs` requires `--task all`.

**Selection flags**: `--stage Expert|Naive`; `--task DPA|DualGo|DualNoGo` (subsets the
overlaid trajectories — `--inputs` requires `--task all`; the autonomous fit is always
DPA); `--correct` (restrict to correct trials, `performance==1`; **default is ALL
trials, correct + incorrect** — incorrect trials muddy the structure, so `--correct`
is the cleaner view, `_correct` in the filename); `--cond-mean` (fit on condition
means vs single trials, `_cm`); `--fit-win LO HI` (overrides the fit window; linear
defaults to the whole trial, the nonlinear autonomous to the DPA delay 21–54);
`--smooth SIGMA` (Gaussian temporal denoise of the latents at the source, tag `_sm<σ>` —
cosmetic for the pooled DUM, matters only for thin per-mouse DUMs); `--rescale STD`
(normalise each latent axis to this std — **required for per-mouse DUMs**, whose latents
come out at std≈6–15 vs the pooled ≈2.8, else the FP box / `gain=1` miss the data; use
`--rescale 2.8`). With `--rescale`, `--nonlinear`/`--inputs` use a box floored at the
pooled-calibrated `±2.5/±3.2` but grown to the data so per-mouse wells aren't clipped;
`--mask-support` (`--mask-bw`, `--mask-alpha`) fades the field where it's far from the fit
trajectories, flagging off-manifold **extrapolation** (the field is fit to a few 1-D curves,
so structure away from them — incl. fixed points there — is unconstrained), tag `_msk`.
Stage/task/flags are folded into the figure path/filename. Aesthetic matches the RNN (small fixed-point markers, sparse
white streamlines). The basis is whatever the `<dum>` was fit on; for a genuine Naive
analysis use a **Naive `<dum>`** (e.g. `pseudo_ALL_Naive_…_f-sample-test_dpca`), not
`--stage Naive` on an Expert basis.

The **DPA delay autonomous dynamics** are simply `--nonlinear` (the autonomous fit is
always DPA working memory): **two A/B attractors** at sample ≈ −1.37 / +1.07 with a
saddle ≈ 0.24 between — the bistable working-memory landscape.

```bash
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice                 # linear
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --per-sample    # A/B displaced fps
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --nonlinear     # DPA-WM bistable autonomous
python plot_pseudo_flow.py --dum <dpca_dum> --dims sample choice --inputs --cond-mean   # autonomous + per-regime input panels
python plot_pseudo_flow.py --dum <naive_dpca_dum> --dims sample choice --inputs --stage Naive
```
Figures → `figures/pseudo/flow/<mode>/<scale>/<factor>/<stage>/{png,svg}/<dum>_flow_<a>x<b>[…][_b<lo>-<hi>][_<task>].{png,svg}`
(`<mode>` = `linear`/`persample`/`nonlinear`/`inputs`; `<stage>` = `Expert`/`Naive`).

All three modes render in the `~/rnn/src/dynamics.py` house style: a **`magma`
speed (`‖Δz‖`) heatmap** + **white streamlines**, fixed points as white ● (stable)
/ white ✕ (saddle) / red ○ (unstable), a `‖Δz‖` colorbar, and the condition-mean
trajectories with a white halo so they read on the dark background. Attractors sit
in the dark (low-speed) basins by construction.

The flow model lives in `src/pca/dynamics.py`: `fit_lds` (linear), `fit_rnn_flow`
(neural `Δz=Lz+Wφ(g·z)+b`, returns `flow, M=[b|L|W]` — used for both the autonomous and
each input-driven panel, fit per regime), `fit_poly_flow`
(polynomial), `flow_fixed_points` (locate/classify), `cv_flow_r2` (single-trial velocity
R² — the ~1% noise floor), `cv_condmean_flow_r2` (**held-out condition-mean velocity R²**,
the correct metric — see below), `bootstrap_fixed_points` (FP reproducibility).

### Single trials, validation & per-mouse flows

**Single-trial flow fields don't work here, and it's not a fixable estimator bug.** A flow
fit conditions on each trial's instantaneous (noisy) position; the per-bin within-condition
noise std (≈3) exceeds the sample-well separation (≈1.4) and is a **slow per-trial offset**
(averaging the whole delay drops its variance only ~1.3×, ≈1 effective sample/trial), so a
single trial is only ~68% informative about its well — the field washes the two wells out.
The condition mean works because it averages across many trials whose offsets are
independent across trials, so it is the **minimal sufficient statistic** for well identity;
the bistable field is an across-trial ensemble property. (The slow *linear* dynamics do
transfer to single trials; only the nonlinear bistability is below the noise floor.)
Diagnostics: `pca/exp_single_trial_{bias,sep}.py`. **The condition mean is the denoising
ceiling** (`exp_ceiling.py`): the dominant noise is a *slow per-trial offset* (RMS 3.5 ≫ well
sep 1.4) removed **only by averaging across trials** (bundling ∝ 1/√k); temporal Gaussian
smoothing (any σ) and a **Kalman/RTS smoother leave it intact** (slow → treated as signal), and
bundled estimates converge to the condition mean as k→∞ — so **no single-trial denoiser (GPFA,
Kalman, shrinkage) beats the condition mean here.**

**Validation (panels self-annotate).** `cv_condmean_flow_r2` splits trials, builds means
from the train half, fits, and scores the **held-out** half's condition-mean velocity R² —
matching what we display (single-trial `cv_flow_r2` tests the wrong object). High-velocity
input regimes get this R² on the panel (pooled C +0.22, NoGo +0.19, Go +0.16, D +0.11,
Cue +0.23); the **slow autonomous has no velocity signal** so it's validated by FP
reproducibility — bootstrap **P(2 attractors)** (pooled 0.72). Tuning (`exp_tune_global.py`):
the defaults gain 1.0 / ridge 0.2 are already near-optimal.

**Per-mouse & sub-pool DUMs.** `pca/build_mouse_dpca.py` builds a single mouse's dPCA
(real co-recorded population) and `pca/build_subpool_dpca.py` a pool over any mouse subset,
both reusing `X_all` without clobbering it; run the flow with `--rescale 2.8`. The **9-mouse
bistability survey** (autonomous P(2 att), correct trials): only **3/9 are bistable**
(JawsM18 0.97, JawsM15 0.95, ChRM23 0.90), JawsM12 borderline, the rest monostable —
per-mouse bistability **tracks sample-memory strength** (JawsM15 d′1.31 vs ChRM04 d′0.75).
Dropping the two weakest, lowest-data mice (`build_subpool_dpca --exclude JawsM01 JawsM06
--tag pool7`) lifts pooled P(2) 0.72→0.95 and sharpens C/D — but this is **partly
outcome-selection** (the mice were dropped because they were monostable), so the honest
reading is "the pooled structure is carried by the animals that have it," not a free win.
Per-mouse **input** panels are not velocity-validated (thin trials); only the autonomous
FP-reproducibility is trustworthy per mouse — the pooled population is the object for the
input flows.

**Manifold geometry — is it a "slow circular manifold with two wells"?** Tested directly
(`exp_manifold.py`, `exp_manifold_pca.py`): (1) the **dPCA sample×choice** plane shows a
clean **cross** — two wells on the sample axis, choice orthogonal at test — but dPCA
orthogonalises sample⊥choice, so it *cannot* reveal a ring; (2) **raw PCA** is dominated by
a shared **curved arc = the condition-independent timing/ramp** (Sample demoted to PC6), so
the dominant slow manifold is *timing*, not WM; (3) **ramp-removed PCA** shows the task
geometry *does* curve (the cross is partly a demixing artifact) but as a **fan to the test
corners, not a closed ring with two wells**. **Model-free radial/angular geometry — canonical
`exp_bootstrap_polar.py`** (bootstrap-CI on the condition-mean trajectory): the sample-axis
`v_rad(r)` goes outward→through-zero→inward → a **significant preferred radius (well)
r₀=1.02 [0.59–1.36]**, and **angular speed brackets 0 → no rotation** — and being non-rotational,
the autonomous flow *is* gradient-compatible (no along-manifold drift to miss). Verdict: defensible = a slow timing
manifold + a **non-rotational, line-like, elongated** two-well sample geometry + orthogonal
choice; a **"circular manifold with two wells" is over-reading** (ring-vs-cross underdetermined
at this SNR). **Headline figure `fig_manifold.py` → `wm_manifold_defensible.png`**: (A) single-
trial delay occupancy = one contractive 2-D blob (A/B wells sub-noise); (B) condition-mean 2-D
geometry ± SEM (sample-coded delay → choice-resolved corners) — separating measured from modelled.
The raw single-trial radial profile was EIV/offset-biased (all-inward); the **ceiling demo**
(`exp_ceiling.py`) shows that bias is removed only by averaging across trials (not by temporal
or Kalman smoothing), so the **condition mean is the denoising ceiling** and the bootstrap
condition-mean profile above is the clean version.

**Per-mouse polar (reworked 2026-06-18).** The resampling methods are **equivalent** — *bundling*,
*bootstrap-of-condmean*, and *CV-of-condmean* give the same radial profile at matched averaging
unit; only **small-k bundling (k=6) is EIV-biased** (manufactures a spurious inward branch), which
invalidated the old `analyze_polar_permouse_profile.py` "every mouse has a well" claim
(`exp_polar_bug.py`, `exp_polar_resample.py`, `fig_polar_methods*.py`). Canonical per-mouse
estimator = **bootstrap-of-condmean** (`fig_polar_radang_permouse.py` → `polar_radang_permouse_boot.png`,
radial+angular ±CI). Honest result: the **+→0→− well shape is NOT reliably resolved per animal**
(wide CIs, 15–60 trials/sample) — solid only **pooled** (r₀≈1.0–1.1); **non-rotation is the one
per-mouse-robust feature** (angular CI brackets 0). Rescue for the well (`fig_polar_wellfit_permouse.py`):
fit the restoring flow `v_rad=b·(r₀−r)` and test **slope b<0** (the outward approach alone fixes r₀)
— **4/9 mice individually significant** (JawsM15, JawsM18, ChRM23, ChRM04), 3 more negative-leaning
but under-powered, 2 flat. Well *existence* detectable per mouse; *location* and bistability-tracking
remain pooled-only.

**Cleanest polar view = read v_rad/v_ang from the FITTED FLOW FIELD** (`fig_polar_flowfield_permouse.py
[sample|choice]`, `fig_polar_flowfield_pooled.py`): fit the rate-net flow to the 2 DPA-delay
condition means, then evaluate `v_rad(r)=flow(r·û)·r̂` (and angular) along the axis at every radius —
smooth, denoised, and defined even at radii the trajectory never reaches (the field interpolates the
inward branch), with bootstrap CI. **Pooled:** sample-axis well **r₀=1.15**, angular≈0; **choice axis
flat / no well, extent r≲1.0** ⇒ sample ⟂ choice, well is sample-specific. **Per-mouse:** clear wells
in JawsM15/M18/ChRM23/JawsM12/ChRM04, weak/flat in the rest. Caveat: field fit from 2 trajectories, so
the ends lean on the model form and **beyond max data radius (grey in the figures) is extrapolation**.

**Nonlinear fixed-point eigenvalue analysis** (`exp_eig_nonlinear.py`, `analyze_bistability.py`;
summary `figures/pseudo/flow/bistability_summary.png`). Linearising the **nonlinear** flow at
each fixed point (`flow_fixed_points` returns the map Jacobian `eig(I+J)`; convert to τ/freq
like `lds_modes`) gives the local modes: every fixed point has a **slow** sample-memory
direction (τ≈5–8 s, |λ|≈0.97–0.98) + a **fast** off-manifold one (τ≈0.9 s, |λ|≈0.84) — fast
contraction onto a slow manifold — and all wells are **non-rotational nodes**. The **barrier
height** (saddle's unstable |eig(I+J)|) measures bistability *depth*: pooled 1.018 (nearly
flat), bistable mice 1.15–1.24 — shallow even when robust. **9-mouse survey:** 4 bistable
(ChRM23, JawsM18 robust; JawsM12, JawsM15 borderline), 5 monostable; P(2) tracks sample-memory
d′ **positively but imperfectly** — **ACCM03 has the highest d′ yet is monostable**, so strong
sample memory is necessary-ish but not sufficient. P(2) is box/bootstrap-sensitive — report the
bistable *set*, not exact values.

**Slow manifold vs 2D double-well** (`exp_slowmanifold_test.py`, `fig_flowfield_panels.py`,
`fig_flowfield_binned.py [k]`). Discriminator = dynamics BETWEEN the wells: walk the inter-well
(sample) axis, split the local Jacobian eigen-rates into longitudinal vs transverse + read the
speed. **Pooled = decisively a SLOW 1-D MANIFOLD carrying 2 shallow wells** (not a deep 2D
double-well): mean inter-well speed only **8% of max**, persistent **transverse≫longitudinal gap
≈4×** (transverse τ≈1 s fast contraction onto the axis, longitudinal τ≈26 s near-marginal),
near-flat saddle. The three-panel field (speed/radial/angular) and the binned (model-free, k=10
bundled) field both confirm the two-well + non-rotational quadrupole structure independent of the
fit. **Honest scope:** the longitudinal facts (low along-axis speed, shallow wells, flat saddle)
are data-supported; the transverse fast-contraction leans on the rate-net form (choice barely
explored in the delay). **Per-mouse: underdetermined** — only ChRM23 cleanly replicates; the
verdict is a pooled result. **CI-removal robustness:** re-running on CI-removed DUMs
(`exp_slowmanifold_test.py pooled {1,2}`) shows the slow-manifold reading is **ramp-driven** — the
transverse contraction collapses as the timing ramp is stripped (τ 1 s → 12 s; verdict
SLOW-MANIFOLD@q0 → intermediate@q1 → 2D-double-well@q2). **Bistability (two wells + saddle) is
robust to CI removal; the "slow manifold" is not** (it was largely the condition-independent ramp,
not the sample-memory subspace).

---

## Legacy scripts (old inline versions)

| Script | Status |
|---|---|
| `run_meta_all.py` | Old combined compute+plot; loads from `/home/leon/dual_task/dual_data/data/pca`; no argparse |
| `run_pseudo_all.py` | Old compute-only (in-sample, `choice` factor, `meta_` prefix); superseded by `run_pseudo.py` + `plot_pseudo_traj.py` / `plot_pseudo_loadings.py` |
| `run_single_all.py` | Old combined compute+plot; now fixed to save to `../data/pca/` |

`plot_pseudo_all.py` was **deleted** — its EVR/trajectory/loadings figures are
now produced by `plot_pseudo_traj.py` + `plot_pseudo_loadings.py`.

---

## Shared plotting primitives (`src/plot/traj.py`)

All trajectory and 1D trace plots across both overlaps and PCA use `src/plot/traj.py`.

```python
from src.plot.traj import (
    plot_mean_sem,             # mean line + ± SEM band
    plot_gradient_line,        # time-coloured 2D path
    add_arrows,                # direction arrowheads
    sem_band,                  # cross-mouse SEM tube on path normal
    make_time_cmap,            # light→dark colormap from a base colour
    colored_path,              # path coloured by arbitrary scalar
    truncate_cmap,             # clip sequential colormap
    velocity_points,           # centred finite-difference velocity
    transition_velocity_points,
    bin_velocity,              # Nadaraya-Watson binned field
    raw_counts,                # raw 2D position histogram
    panel_fields,              # WTA combined flow field per label
    draw_panel,                # full panel renderer
)
```
