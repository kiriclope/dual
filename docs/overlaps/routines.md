# Overlaps Scripts & Routines

## Working directory & Python
```bash
cd /home/leon/dual/overlaps
/home/leon/mambaforge/envs/dual/bin/python <script>.py
```

## Shared constants (every script)
```python
DUM      = 'log_generalizing_overlaps_none_l1_ratio_0.0'
DATA_IN  = '../data/overlaps'
ALL_MICE = ['JawsM01','JawsM06','JawsM12','JawsM15','JawsM18','ChRM04','ChRM23','ACCM03','ACCM04']
STAGES, CONDITIONS = ['Naive','Expert'], ['DPA','DualGo','DualNoGo']
SAMPLE_SPLITS = [('A',[0,1],'#332288'), ('B',[2,3],'#44AA99')]

TRAIN_EPOCHS = [('trainTEST', options['bins_TEST']),   ('trainDELAY', options['bins_DELAY']),
                ('trainCHOICE', options['bins_CHOICE']), ('trainED',  options['bins_ED'])]
```
X_epoch normalization and idx_correct: see `docs/shared_data.md`.

---

## `plot_traj2d.py`
**Shows:** 2D mean trajectory (sample×choice) over time + KDE histogram strip per panel.
**Trials:** correct only. **Layout:** 2×3 + histogram strip each panel.

```python
# local (overlaps-specific data collection)
mouse_trajectories(X_ep, cond, stage, target, odor_pairs=None)
# → list of per-mouse mean trajectories (T,)

# from src.plot.traj
plot_gradient_line(ax, x, y, color, lw=2.1)   # time-gradient LineCollection
sem_band(ax, mx, my, ex, ey, color)            # SEM tube projected onto path normal
add_arrows(ax, x, y, color, n_arrows=3)        # direction arrowheads
make_time_cmap(base_color, lo=0.22)            # light→dark cmap
```
Key params: `xlim=(-4,4)`, `ylim=(-2,6)`, `yticks=[-2,0,2,4,6]`, KDE `bw_method=0.4` over `BINS_DELAY`.
Output: `figures/overlaps/traj2d/correct/{png,svg}/`

---

## `plot_traj2d_planes.py`
**Shows:** Three code-plane trajectories — `sample×choice`, `sample×test`, `choice×test` —
as time-gradient paths with arrows + SEM-over-mice band, coloured by a split variable.
The overlaps analog of PCA `pca/plot_pseudo_traj2d.py` (three panels per figure, using the
three CCGD decoder codes as the axes instead of three PCs).
**Trials:** correct only. **Layout:** 1×3 panels.

Splits (`SPLITS`), each its own subfolder. `fold` = codes folded for that split (see below):

| split | levels | scope | fold |
|---|---|---|---|
| `pair`   | AC/AD/BD/BC      | per (stage, condition) | — |
| `sample` | Odor A / B       | per (stage, condition) | choice, test |
| `choice` | No lick / Lick   | per (stage, condition) | sample, test |
| `test`   | Odor C / D       | per (stage, condition) | sample, choice |
| `task`   | DPA / Go / NoGo  | per stage (conditions pooled, coloured by task) | sample, choice, test |

**Folding** (default on; `--no-fold` to disable). When a split pools across a code's
discriminated variable (e.g. `task` pools A and B), the signed code averages to ~0
because the two classes have opposite sign. Folding multiplies each trial's code by
`2·label − 1` (`SIGNS[code]`, label from `CODE_VAR`) before averaging, so classes align
and the axis shows coding *magnitude* (annotated `(folded)`). A code is folded only when
its variable is pooled, so e.g. the `sample` split keeps Odor A vs B separated on the
sample axis while folding the pooled choice/test axes.

```python
# local (overlaps-specific data collection)
per_mouse(trial_mask, signs)
# → {mouse: mean (folded) decision-function trajectory (T,)}; signs = SIGNS[code] or ONES
draw_planes(data, levels, fold, title, out_dir, stem)   # renders one 3-panel figure

# from src.plot.traj
plot_gradient_line, add_arrows, sem_band
```
Paths truncated at the test offset (`bins_TEST[-1]+1`); per-panel limits from the group-mean extent.
Output: `figures/overlaps/traj2d_planes/{fold|nofold}/{train_tag}/{split}/{png,svg}/{stem}.{png,svg}`
(`{stem}` = `{stage}_{cond}`, or `{stage}` for the pooled `task` split). 104 figures per fold-mode × 4 train epochs.

---

## `plot_flow2d.py`
**Shows:** Empirical flow field — magma speed heatmap + streamlines + star fixed points + overlay.
**Trials:** all laser-off, per-(mouse,odor_pair) means. **Layout:** 2×3 × 4 epochs × 2 variants × 3 modes = 24 figs.

```python
# local (overlaps-specific data collection)
group_mean_trajs(X_ep, cond, stage, pairs)
# → list of (sx,cy) per (mouse,odor_pair); used to BUILD the field

grand_mean_traj(X_ep, cond, stage, pairs, target)
# → single (T,) array; equal-weight mean; used for OVERLAY + fixed-point location

# from src.plot.traj
velocity_points(trajs, segments, step=5, smooth=0)
# → (px,py,du,dv); centred diff h=step//2 within each segment

transition_velocity_points(trajs, segments, step=5)
# → (px,py,du,dv); forward diff x[t+step]-x[t]

bin_velocity(px, py, du, dv, x_edges, y_edges, sigma=1.2)
# → (U, V, count_raw); Nadaraya-Watson

panel_fields(trajs_by_label, x_edges, y_edges, segments, bins_late, sigma=1.2, min_raw_count=1)
# → dict with: U,V, U_trans,V_trans, count,count_smooth,supported,
#              speed, speed_late,supported_late, per_sample
# Returns None if no data

draw_panel(ax, fields, xi, yi, xlim, ylim, occ_vmax, field_mode, traj_overlay,
           bins_delay, bins_late, xtime, cmap_speed)
# field_mode: 'velocity'|'flux'|'transition'
# Heatmap = hybrid: speed_late where supported_late, speed elsewhere
```
Key params: `N_BINS=14`, `SIGMA=1.2`, `MIN_RAW_COUNT=1`, `VEL_STEP=5`, `TRAJ_SMOOTH=0`, `AXIS_PAD=0.4`.
`CMAP_SPEED = magma` (slow=dark). Variants: `all_trials=[BINS_DELAY]`, `all_trials_bl=[BINS_BL,BINS_DELAY]`.
Output: `figures/overlaps/flow2d/{train_tag}/{variant}/{mode}/{png,svg}/`

---

## `plot_codes.py`
**Shows:** 1D mean ± SEM decision-value traces for sample / choice / test codes, split by label (A vs B, lick vs no-lick, C vs D).
**Trials:** correct only. **Layout:** 1×3 panels per (condition, stage), one figure per train epoch.
Uses `plot_mean_sem` from `src.plot.traj`.
Output: `figures/overlaps/codes/{train_tag}/correct/{png,svg}/`

---

## `plot_pairs.py`
**Shows:** 1D mean ± SEM decision-value traces split by odor pair (AC/AD/BD/BC); match pairs solid, non-match dashed.
**Trials:** correct only. **Layout:** tasks × stages grid, one figure per target (sample/choice/test).
Uses `plot_mean_sem` from `src.plot.traj`.
Output: `figures/overlaps/pairs/{png,svg}/`

---

## `plot_scatter_perf.py`
**Shows:** Δ choice loc. (correct, BINS_LATE) vs Δ DPA/GNG performance.
**Trials:** x-axis = correct only; y-axis (Δ perf) = all laser-off. **Layout:** 1×3, 4 figs per epoch.

```python
perf_delta(perf_col, task_mask)          # → {mouse: Expert−Naive}
annotate_stats(ax, xs, ys)               # Pearson r, Spearman ρ, 1-samp t — placed ABOVE panel
regression_band(ax, xs, ys)             # regression line + 95% CI band
```
Per-panel xlim centred at 0; shared ylim; titles via `.replace('DualGo','Go').replace('DualNoGo','NoGo')`.
Output: `figures/overlaps/scatter_perf/{train_tag}/{png,svg}/`
Files: `*_dpa_perf`, `*_dpa_perf_by_sample`, `*_gng_perf`, `*_gng_perf_by_sample`

---

## `plot_geometry.py`
**Shows:** Per-(mouse,odor_pair) BINS_LATE mean in sample×choice plane; Naive open / Expert filled; strip plot below.
**Trials:** correct. **Layout:** GridSpec 2 rows × 3 cols.
Output: `figures/overlaps/geometry/{train_tag}/correct/{png,svg}/`

---

## `plot_marginal.py`
**Shows:** 1D sample/choice code vs time during delay; Naive (#9ecae1) / Expert (#2171b5) ± SEM.
**Trials:** all laser-off. **Layout:** 2 rows (sample/choice) × 3 cols.
Uses `plot_mean_sem` from `src.plot.traj`.
Output: `figures/overlaps/marginal/{train_tag}/{png,svg}/`

---

## `plot_occupancy.py`
**Shows:** 2D KDE occupancy at BINS_LATE; A/B centre-of-mass circles.
**Trials:** all laser-off. **Layout:** 2×3. `N_BINS=30`, `SIGMA=0.8`, `AXIS_LIM=(-4,4)`, `CMAP=viridis`.

```python
collect_delay_points(X_ep, cond, stage)
# → (all_x, all_y, sample_pts); per-(mouse,odor_pair) means at each BINS_LATE bin
```
Output: `figures/overlaps/occupancy/{train_tag}/{png,svg}/`

---

## `plot_scatter_ab.py`
**Shows:** Per-animal A/B late-delay positions; Naive open / Expert filled; grand-mean centroid (black edge).
**Trials:** all laser-off. **Layout:** 1×3.
Output: `figures/overlaps/scatter_ab/{train_tag}/{png,svg}/`

---

## Shared plotting primitives (`src/plot/traj.py`)

All overlaps scripts import drawing helpers from `src.plot.traj` rather than defining them locally.

| Function | Used by | Purpose |
|---|---|---|
| `plot_mean_sem(ax, xtime, mu, sem, color, alpha=0.2, **kw)` | `plot_codes`, `plot_pairs`, `plot_marginal` | Mean line + ± SEM band |
| `plot_gradient_line(ax, x, y, color, lw=2.1)` | `plot_traj2d`, `plot_traj2d_planes` | Time-coloured 2D path |
| `add_arrows(ax, x, y, color, n_arrows=3)` | `plot_traj2d`, `plot_traj2d_planes` | Direction arrowheads |
| `sem_band(ax, mx, my, ex, ey, color)` | `plot_traj2d`, `plot_traj2d_planes` | SEM tube on path normal |
| `make_time_cmap(base_color, lo=0.22)` | `plot_traj2d` | Light→dark colormap |
| `colored_path(ax, x, y, t, cmap)` | `plot_flow2d` | Path coloured by scalar |
| `truncate_cmap(name, lo=0.35, hi=1.0)` | `plot_flow2d` | Clip sequential colormap |
| `velocity_points(trajs, segments, step, smooth)` | `plot_flow2d` | Centred finite-difference velocity |
| `transition_velocity_points(trajs, segments, step)` | `plot_flow2d` | Forward-difference displacement |
| `bin_velocity(px, py, du, dv, x_edges, y_edges, sigma)` | `plot_flow2d` | Nadaraya-Watson binned field |
| `raw_counts(px, py, x_edges, y_edges)` | `plot_flow2d` | Raw 2D histogram |
| `panel_fields(trajs_by_label, x_edges, y_edges, segments, bins_late, ...)` | `plot_flow2d` | WTA combined flow field |
| `draw_panel(ax, fields, ..., bins_delay, bins_late, xtime, cmap_speed)` | `plot_flow2d` | Full panel renderer |

---

## Regenerate all figures
```bash
cd /home/leon/dual/overlaps
for s in plot_traj2d plot_flow2d plot_scatter_perf plot_geometry plot_marginal plot_occupancy plot_scatter_ab; do
    echo "=== $s ===" && /home/leon/mambaforge/envs/dual/bin/python ${s}.py
done
```
