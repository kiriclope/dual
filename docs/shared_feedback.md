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

## 2026-09-08 — axis-window unification: lessons
- **One axis definition, both pipelines.** Decoder axes are sample/dist bins 36–38 and choice/test bins 54–62 everywhere
  (`--legacyaxes` restores the old ones). Read-out windows (late delay 45–53, Fig 3b decision read 60–66) are a separate
  thing; state both when a panel mixes them.
- **Build variants before flipping.** Five candidate windows (A–E) were built as sandboxed caches + variant artifact pages
  and compared on ONE table (push, coupling + its robustness battery, Fig 3c, Fig 6, d′) before the canonical flip; the
  trade-off (early-test bins buy the push, cost the choice code) only became visible on that table. Table in memory
  `project_axis_windows.md`.
- **Supplements must share the main figure's windows.** The four ED-5 supplements had their own late-delay window
  (set_options 48–53, battery 45–52) and their "figure default" row did not reproduce Fig 4; aligned to 45–53 it does.
  Any "reproduces the main" row must be checked numerically against the main's print.
- **Never insert a `# comment` into a replacement that can land mid-line** (`a = ...; b = ...` lines) — it swallowed
  three variable definitions and crashed four scripts silently inside a long batch. Also define flags before first use
  (`LEGACY` in `fig_behavior_opto_main.py`).
- **Patch text with assert-counted, whitespace-tolerant matches** (scratchpad `flip_text_patch.py`): the wrapped .md needs
  `\s+` between words; caption literals need exact matches; every anchor asserted, then grep-verified.
- **Do not consult decoder variants Leon has dropped** (pca20): verdicts come from the canonical no-PCA build only.
