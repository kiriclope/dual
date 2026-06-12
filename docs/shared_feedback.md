# Shared Feedback & Coding Principles

## Verify empirically — never infer from code reading alone

Always verify data structure and code behaviour by running Python checks, not by reading code alone.

**Why:** Two confirmed past errors from code-reading inferences:
1. Claimed `cv_pca_meta` creates a "block-diagonal covariance structure" making joint PCA illusory — **wrong**. The condition-averaging step (`cv_avg_cond`) before PCA makes the matrix dense across mice via shared task structure.
2. Misidentified `X_all_center` as per-trial baseline subtracted (`center_BL`) when it is actually per-day mean PSTH subtracted.

**How to apply:**
- Before asserting what a preprocessing step does: `np.nanmean(X[mask], axis=0)` and inspect the result
- Before asserting what a PCA fit sees: print the shape of the matrix actually passed to `pca.fit()`
- Before asserting a column's values: `y_single['col'].value_counts()` or `y_single['col'].unique()`

---

## Run commands
```bash
/home/leon/mambaforge/envs/dual/bin/python script.py
```
Always `cd` into the script's directory before running (scripts use relative paths like `../data/`).

---

## Shared plotting primitives live in `src/plot/traj.py`

Do not add new plotting helpers inline to overlaps or PCA scripts. Put them in `src/plot/traj.py` and import from there.

Current shared functions:
- `plot_mean_sem` — mean line + ± SEM band; used by all 1D trace scripts
- `plot_gradient_line`, `add_arrows`, `sem_band`, `make_time_cmap` — 2D trajectory drawing
- `colored_path`, `truncate_cmap` — flow field drawing
- `velocity_points`, `transition_velocity_points`, `bin_velocity`, `raw_counts` — flow field computation
- `panel_fields`, `draw_panel` — flow field panel rendering

Data-collection functions that are overlaps-specific (`group_mean_trajs`, `grand_mean_traj`, `mouse_trajectories`) stay in the overlaps scripts because they depend on `y_single` with the `target` column.

---

## Security
**Never print, display, or repeat the value of any variable or file containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD`** — extract and use silently only.
