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
**Trials:** `correct` (default) or `all` laser-off (correct + incorrect) — pass `all`/`--all` on
the command line (`idx_trials` swaps `idx_correct`↔`idx_laser`; output folder swaps too). The
no-lick push survives the all-trials version, so it is not an artifact of conditioning on correct
trials. **Layout:** 2×3 + histogram strip each panel.

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
Output: `figures/overlaps/traj2d/{correct|all}/{png,svg}/`

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

## Plotting conventions (shared with dPCA — source: `pca/plot_pseudo_traj.py`)

Match these for any trajectory/1D-code figure so overlaps and dPCA panels are interchangeable.

**Matplotlib / style**
```python
matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_context('notebook'); sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)
W, H = 3.5, 2.6                       # per-panel inches
```
Always save **PNG dpi=300 + SVG** (`svg.fonttype='none'`), to `{...}/png/` and `{...}/svg/`.

**Time axis (every 1D trace)**
```python
xtime = np.linspace(0, 14, 84)
ax.set_xlim([0, 14]); ax.set_xticks([0, 2, 4.5, 6.5, 9, 11, 14])
add_vlines(ax, if_dpa=0)             # src.common.plot_utils — epoch shading
ax.axhline(0, ls='--', color='k', lw=0.6)
```

**Mean ± SEM** — never plot bare means. Use `src.plot.traj.plot_mean_sem(ax, xtime, mu, sem, color, lw=1.6, label=…)`
with `mu = Xs.mean(0)`, `sem = Xs.std(0)/sqrt(n)`, after per-trial baseline subtraction
(`Xs -= Xs[:, BL].mean`, `BL = slice(0,12)`). Pool across mice only after per-mouse BL z-score.

**Colours per split** (exact hexes from `plot_pseudo_traj.py`)

| Split | levels → labels | colours |
|---|---|---|
| sample | A / B | `#332288` (indigo) / `#44AA99` (teal) |
| choice | No-lick / Lick | `#377eb8` / `#4daf4a` |
| test | Odor C / Odor D | `#377eb8` / `#4daf4a` |
| task | DPA / Go / NoGo | `sns.color_palette('muted')` `[3] / [0] / [2]` |
| odor_pair | AC / AD / BD / BC | `#332288` / `#88CCEE` / `#117733` / `#44AA99` |

---

## Codes / trajectories / flows (sample · choice · test · task)

Three CCGD target codes (`y['target']` ∈ `sample`/`choice`/`test`), decision function `X[:,1]`,
train-epoch averaged, read across test time. See `overview.md` for the method/rationale.

| Script | Command | Shows | Output |
|---|---|---|---|
| `fig_overlaps_codes_1d.py` | `python fig_overlaps_codes_1d.py` | **1D codes**: sample/choice/**test** each on its own code + task on the choice code. Loops **nine train epochs** sweeping the whole trial (`stim` S1 sample ~2.5 s → `ed` early-delay ~3.5 s → Go/NoGo distractor odor ~5 s → `md` mid-delay ~6 s → `gng_rwd` GNG outcome ~7 s → `delay`/`ld` late-delay ~8 s → `test` S2 ~9.5 s → `choice` DPA response → `dpa_rwd` DPA outcome ~11.5 s) × **both stages**. **CRUCIAL caveat**: the Go/NoGo distractor odor *and* the GNG reward exist **only on Dual trials** (DPA has no distractor; `odr_choice` NaN). Since sample/choice/test panels are **DPA-only**, ed/md/gng_rwd/ld are just **uninterrupted delay-maintenance timepoints** there — the distractor/GNG-reward events live in the **task panel** (all tasks), where md/gng_rwd bracket the GNG odor + outcome. Interpretation: **sample** epoch-invariant (ed≈md≈ld → one stable memory; robustness = *stability*, not surviving a within-trial distractor); **choice (DPA-only)** non-flat in the delay = a **maintained no-lick action set** held alongside the memory (dual-coding; deepens with learning), firming into the executed lick at choice/dpa_rwd — **not** foreknowledge of the answer (correct lick undetermined until S2). sample/choice/test panels are **DPA-only** (avoids the Dual 2nd-odor distractor contaminating the trace ~7–8 s); task panel keeps all tasks. **Any pre-test anchor (stim/ed/md/gng_rwd/delay/ld) flags its test panel ⚠ pre-test confound** (a test axis trained before the test odor is present is spurious; test/choice/dpa_rwd are valid; see overview.md). Three error-bar variants per (train,stage): **`grandmean`** = per-mouse mean → mean±SEM over mice (the honest one); **`permouse`** 9×4 grid (SEM over trials); **`pooled`** 1×4 (trials pooled over mice, SEM over trials — **reference only**, tight but pseudo-replicated) | `figures/overlaps/codes1d/{stim,ed,md,gng_rwd,delay,ld,test,choice,dpa_rwd}/{png,svg}/` → `overlaps_codes1d_{grandmean,permouse,pooled}_{naive,expert}` (train epoch = subdir) |
| `plot_marginal.py` | `python plot_marginal.py` | 1D code vs time, Naive vs Expert (incl. **test**) | `figures/overlaps/marginal/` |
| `plot_traj2d_planes.py` | `python plot_traj2d_planes.py [--no-fold]` | **2D planes**: `sample×choice`, `sample×test`, `choice×test` (3 panels), per (stage,cond); `TRAIN_EPOCHS` loop | `figures/overlaps/traj2d_planes/<fold>/<train>/<split>/{png,svg}/` |
| `plot_traj2d.py` | `python plot_traj2d.py` | `sample×choice` 2D trajectory + KDE strip | `figures/overlaps/traj2d/correct/{png,svg}/` |
| `fig_overlaps_traj.py` | `python fig_overlaps_traj.py` | `sample×choice` condition-mean trajectories (per-mouse+pooled) + sample-memory(t) | `figures/overlaps/flow/png/` |
| `fig_overlaps_flow_empirical.py` | `python fig_overlaps_flow_empirical.py [mouse\|pooled] [--pooled]` | **Autonomous flow** (recommended): empirical binned, `sample×choice`, trainTEST, **DELAY-only**; A/B winner-take-all default (`--pooled` = average) | `figures/overlaps/flow_emp/{png,svg}/` |
| `fig_overlaps_flow_planes.py` | `python fig_overlaps_flow_planes.py [mouse\|pooled]` | rate-net flow on same plane (reference; **5-fold CV-tunes ridge**, prints held-out vel-R² — ≈0 = noise floor) | `figures/overlaps/flow_planes/{png,svg}/` |
| `fig_overlaps_flow_inputs.py` | `python fig_overlaps_flow_inputs.py [mouse\|pooled]` | autonomous + input-driven 8-panel grid (A/B/Go/NoGo/Cue/C/D), **empirical binned** | `figures/overlaps/flow_inputs/{png,svg}/` |
| `fig_overlaps_flow_lowrank_shared.py` | `python fig_overlaps_flow_lowrank_shared.py [--shared\|--independent\|--partial] [--stage Expert\|Naive] [--train delay\|test\|ld\|wide\|diag] [--fit vel\|traj] [--compare] [--anchor W] [--margin M] [--smooth σ] [--vstep K]` | **rank-2 gain-modulated flows ported VERBATIM from `pca/fig_dpca_flow_lowrank_shared.py`** (model `ż=−z+S(z)·Az+c`, CV-tuned `(a,δ[,λ])`, 8-panel autonomous+inputs). Data section swapped: sample×choice plane built from the CCGD codes (trainTEST, per-mouse BL-norm), with the two `target` blocks matched by a label-key sort (row-identical on sample/test/tasks/choice) → per-trial `[sample,choice]`, rescaled to the dPCA std≈2.8 scale. Modes: pooled→independent (default), `--partial` (CV-tunes ridge λ, generalizes best), `--shared`. **Fixed points marked at the condition-mean TRAJECTORY ENDPOINTS** (last-5-bin mean) — subproject convention, so trajectories terminate at the fps by construction (root-finding is displaced at low SNR: at the CV gain the field's fp sits on the A endpoint but B's shallow well is unclassified). `--smooth`/`--vstep` do NOT improve the pooled fit (velocity-SNR ceiling; `--vstep` default 1, K>1 hurts). **IMPROVED 2026-07-01**: **single train axis** (`--train`, default **delay** — faithful bistable autonomous, moves B +0.46→+1.25 = a real well) + **per-regime input READ windows = [odor onset, offset + `--margin` (default 12 bins ≈2 s calcium tail)]** (A/B 15–30, Go/NoGo/Cue 30–45, C/D 57–72; autonomous = delay maintenance). The **window** is what fixes C/D: on the single delay axis, C/D 57–84→57–72 takes vel-R² −0.25→**+0.46/+0.30** (post-response tail was noise). **Anchor only the autonomous** (`--anchor 8`; inputs are driven) → 2 attractors AT the A/B endpoints. (A per-regime plane — test for C/D — was tried & reverted; makes C/D slightly cleaner (+0.65/+0.48) but breaks single-axis comparability.) Pooled CV drops (−0.17) as C/D are delay-confounded on this axis (don't read response regimes on a delay axis). **RETRACTED**: an earlier gain-raise to force P₂≥0.5 bistability manufactured a 2nd attractor off the B endpoint (the "fp not where the data goes" bug); #1 anchoring alone on the TEST plane also can't place B (too near the saddle) — the plane change (#2) is what makes B a real well. **`--fit traj` (descriptive objective, 2026-07-01):** the GOAL is to fit a rank-2 dynamics that *describes* the trajectories — so fit each regime by **integrating the flow from the trajectory start and minimising POSITION error** (`least_squares`; independent mode) and score the **in-sample trajectory R²** (does rank-2 reproduce the trajectories?). Result: **it does — pooled in-sample traj-R² +0.70** (Expert/delay; A/B ~1.0, autonomous +0.63, Go/NoGo +0.82/+0.75, C/D +0.64/+0.60; only Cue poor +0.15). **Robust across the grid** (independent × delay/test/ld/diag × Naive/Expert): pooled descriptive traj-R² **0.53–0.80, diag best (+0.80/+0.79)** — the rank-2 description holds in both stages. The bistable autonomous + input-driven reshaping fall out of the fit. Held-out generalization (rank-2 vs linear a=0) is printed as a *secondary* note and is poor & ≈linear — i.e. a good low-D **description**, not out-of-sample prediction (that's the intent). Output → `figures/overlaps/flow_lowrank_traj/`. | `figures/overlaps/flow_lowrank_shared/{png,svg}/overlaps_lowrank_{mode}_{stage}_train{axis}.*` |
| `fig_overlaps_traincmp.py` | `python fig_overlaps_traincmp.py [sample\|choice\|test]` | **stability check**: code geometry across train epochs + diagonal, + code(level1−level0) corr matrix (sample=trial-wide stable; choice=late-delay→response; test=response-only/delay-confound) | `figures/overlaps/traincmp/{png,svg}/` |
| `exp_nolick_push_stats.py` | `python exp_nolick_push_stats.py [test\|delay] [correct\|all]` | **rigorous no-lick-push stats (2026-07-02)**: late-delay choice-code depth per (mouse,stage,sample A/B), DPA. Prints (1) one-sample Expert-depth-in-no-lick, (2) paired Naive→Expert **deepening**, (3) A/B asymmetry — each with **bootstrap 95% CI over mice, Wilcoxon two-sided & one-sided (directional), t, Cohen's dz**; the printed battery always loops trainTEST+trainDELAY × correct+all-laser-off. **Honest verdict:** the push is a **pooled, medium-large POPULATION TREND** (Δ −0.5 to −1.0 BLσ, dz −0.5 to −0.64, 7–8/9 mice) — by the conservative **two-sided tests it does NOT reach p<0.05** (pooled Wilcoxon p2≈0.074, paired t p2≈0.090, A p2≈0.10, B p2≈0.13); directional one-sided p1≈0.037. (The bootstrap CI excluding 0 is **anti-conservative at n=9** — ~21% narrower than the honest t-CI [−2.14,+0.19] which includes 0; don't cite it as significance. See `exp_nolick_push_reconcile.py`.) **Not** significant per-class or per-mouse; **A-strong/B-weak asymmetry is trainTEST-axis-specific, vanishes on the delay axis** (there A/B push comparably). **Figure** (args pick axis+trial set, default **delay correct**): 3-panel per-mouse Naive→Expert paired lines — **Sample A, Sample B, and the A&B POOLED single metric** — group mean±bootstrap-CI + **significance stars driven by the per-panel MAXIMAL-LMM p** (pooled = both random slopes `(1+exp+samp\|mouse)`; A/B-only = stage slope `(1+exp\|mouse)`), with **`n/c` shown instead of a star when the fit did not converge**; dz, n/9 deeper, and the conservative mouse-mean Wilcoxon p annotated alongside. On delay/all the **pooled** panel is * (LMM p=0.024), A n.s. (0.080), B `n/c` (fit does not converge). The pooled metric recovers the power lost by per-sample splitting. Stem `..._nolick_push_paired_{axis}_{ts}` | `figures/overlaps/nolick_push/{png,svg}/` |
| `exp_nolick_push_lmm.py` | `python exp_nolick_push_lmm.py` | **trial-level mixed-effects test of the no-lick push (2026-07-02, updated)**: individual DPA choice-target trials (delay+test × correct+all). **MAXIMAL model = random slopes for BOTH within-mouse factors** `depth ~ expert + C(sample) + (1+expert+C(sample)\|mouse)` (Barr et al. 2013 — dropping the sample slope leaks its variance to residual). **VERDICT: the correctly-specified maximal model IS significant on the delay axis** — pooled deepening **β=−0.98, p=0.024** (delay/all, converged), β=−0.86 p=0.047 (delay/correct, but `*NC*` did NOT converge → caution); **TEST axis n.s.** (0.14–0.25), consistent with delay being the principled axis. Prints maximal + stage-slope-only (p≈0.062, more conservative) + random-intercept side by side. ⚠ The **random-INTERCEPT** p<0.0001 is **PSEUDO-REPLICATION** (ignores within-mouse slopes → anti-conservative SE); **never report it**. statsmodels' Wald z is optimistic at 9 groups; the n=9 mouse-mean test (~0.07–0.09) is the conservative cross-check. No figure — stdout only | — |

The `test` code is covered by `fig_overlaps_codes_1d.py` (its own 1D panel) and by the
`sample×test` / `choice×test` planes in `plot_traj2d_planes.py`.

### Decoder weights & cosine similarity (axis alignment)

`run_overlaps.py --save-weights` writes `weights_<DUM>.pkl` =
`{'weights': {(mouse,stage,context,target): ws (n_train_times, n_neurons)}, 'valid': {(mouse,stage): mask}}`,
the **fold-averaged discriminant axis in raw neuron space** (back-projected via `raw=True`).
Plumbed through `ccgd_validation(..., return_weights=True)` → `null_info["weights"]`
(`src/overlaps/ccgd.py`; uses `get_space_params_timewise`). The canonical run is
`run_overlaps.py --scaler none --save-weights` → DUM `log_generalizing_overlaps_none_l1_ratio_0.0_raw`
(a separate `_raw` fileset; does NOT overwrite the canonical `..._0.0` tensors). ~19 min.

| `fig_overlaps_cosine.py` | `python fig_overlaps_cosine.py` | **cosine of the discriminant axes** (neuron space, per mouse then averaged): within-code epoch×epoch **stability** heatmaps (per code, per stage) + between-code **alignment** vs epoch (sample–choice/sample–test/choice–test, vs the `±1/√N` chance floor) | `figures/overlaps/cosine/{png,svg}/` |

**Result (2026-06-24):** between-code alignment sits **at the ±1/√N chance floor at every epoch, both stages** → sample/choice/test axes are **mutually orthogonal** the whole trial (small Naive sample–choice anti-alignment ~−0.08 collapses to ~0 in Expert — codes sharpen to orthogonal with learning). Within-code cosines 0.4–0.9 ≫ floor → each axis is temporally stable (sample most stable through the delay; choice/test cohere at response). = the dual-coding geometry, measured directly from weights.

| `fig_overlaps_orthogonalization.py` | `python fig_overlaps_orthogonalization.py` | **cross-stage Naive→Expert orthogonalization test**: per-mouse delay-window between-code \|cos\|, paired Wilcoxon, + within-code ed·ld self-cosine **reliability control** | `figures/overlaps/cosine/{png,svg}/orthogonalization_naive_expert.*` |

**Orthogonalization result (2026-06-24):** only the **sample–choice (memory×action) pair orthogonalizes with learning** — signed cos **−0.068→−0.010** (mildly anti-aligned in Naive → orthogonal in Expert), \|cos\| 0.083→0.029, **Wilcoxon p=0.020, 7/9 mice**. sample–test null (p=0.074, wrong way), choice–test trend (p=0.098). **Confound ruled out:** within-code self-stability (ed·ld) is **identical across stages** (sample 0.574/0.582, choice 0.422/0.426, test 0.465/0.468; all p>0.8) → Expert axes are not noisier, the drop is real orthogonalization not noise-toward-floor.

---

## Regenerate all figures
```bash
cd /home/leon/dual/overlaps
for s in plot_traj2d plot_traj2d_planes plot_flow2d plot_scatter_perf plot_geometry plot_marginal plot_occupancy plot_scatter_ab; do
    echo "=== $s ===" && /home/leon/mambaforge/envs/dual/bin/python ${s}.py
done
# codes / trajectories from the overlaps decoders:
for s in fig_overlaps_codes_1d fig_overlaps_traj; do
    /home/leon/mambaforge/envs/dual/bin/python ${s}.py
done
# empirical autonomous flows (pooled + per-mouse):
for w in pooled JawsM01 JawsM06 JawsM12 JawsM15 JawsM18 ChRM04 ChRM23 ACCM03 ACCM04; do
    /home/leon/mambaforge/envs/dual/bin/python fig_overlaps_flow_empirical.py $w
done
```

## Decoder weights & cosine analysis — full reproduce path
```bash
cd /home/leon/dual/overlaps
# 1) re-fit decoders and SAVE weights (raw neuron space). ~19 min; reuses the existing
#    data/pca/X_all_nan_.pkl (no --rebuild). Writes a SEPARATE _raw fileset + weights pkl;
#    does NOT touch the canonical X_..._0.0 tensors the other scripts read.
/home/leon/mambaforge/envs/dual/bin/python run_overlaps.py --scaler none --save-weights
#    -> data/overlaps/{X,labels,weights}_log_generalizing_overlaps_none_l1_ratio_0.0_raw.pkl
# 2) cosine figures (need the weights pkl above):
/home/leon/mambaforge/envs/dual/bin/python fig_overlaps_cosine.py            # stability + alignment
/home/leon/mambaforge/envs/dual/bin/python fig_overlaps_orthogonalization.py # Naive→Expert test
#    -> figures/overlaps/cosine/{png,svg}/
```

## Git / data hygiene (2026-06-30)
- **Never commit data.** `.gitignore` excludes `*.pkl`, `*.pth`, `*.svg`, `*.pdf`, `*.pyc`,
  `data/`. The pseudo-population (`data/pca/X_all_nan_.pkl`, ~20 GB), the CCGD tensors and the
  weights pkls all live only on disk. Commit **code + docs + PNG** only.
- History was cleaned with `git-filter-repo --path pca/results/ --invert-paths --force` to remove
  two oversized pkls (177 MB + 59 MB) that the first "upload local repo" commit had baked in before
  the ignore existed (they blocked `git push`). The on-disk copies were preserved; only history
  changed. `origin` = `git@github.com:kiriclope/dual.git` (filter-repo drops the remote — re-add it
  after any future rewrite). If a push is ever rejected for large files again, scan history with:
  `git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | awk '$1=="blob" && $2>40000000'`.
