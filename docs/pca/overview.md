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
| `--epoch` | `TEST` | `TEST` | `DELAY` | Fit window |
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
`--remove-ci Q` appends `_ci<Q>` — so every basis- or projection-changing option is distinct on
disk and in the figure folder tree.

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
| `pseudo_weights_<dum>.pkl` | (n_comp, n_neurons_total) | Reference loadings |
| `pseudo_evr_<dum>.pkl` | (n_folds, n_comp) | EVR per fold |

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

`Q` is the knob: more directions strip more ramp (q=3 also removes the test-evoked
common mode) but eventually touch condition-dependent structure (small DPA push).
`blcenter --remove-ci` is the only config with **no artifact AND ramp-suppressed
task axes**, at the lowest mixing index.

**Note**: the older `plot_single_individual.py` / `plot_meta_individual.py`
still load from `pca/results/` (old path); update their `RESULTS` when switching
to the consolidated `../data/pca/` layout.

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
