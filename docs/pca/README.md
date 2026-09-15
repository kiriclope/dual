# PCA Subproject

Pseudo-population dPCA on the dual working-memory task (9 mice). Latent dynamics, flow fields, and the
no-lick learning push. See `docs/meta_project.md` for the paper overview.

## Docs
- **`dimensionality.md`** — ★ the CURRENT main Fig 2 (`fig_dimensionality_main.py`, since 2026-08-10):
  honest cvPCA + shattering + PC-coding dimensionality; methods, settled numbers, gotchas. ALSO carries
  main **Fig 3** (`fig_manifold_main.py`, since 2026-08-30; dated blocks — read "FINAL Fig 3 structure"
  and the storyboard blocks that follow it): the one-manifold figure (traces · storyboard of REPLAYED
  per-mouse CCGD states, per-window centred, SEM ellipses, decision read 10–11 s · plane sufficiency
  bars + per-mouse · cosines · cross-stage decoding) and its ED companion `fig_manifold_supp.py`;
  caches from `exp_permouse_plane.py` / `exp_plane_frame.py` / `exp_axis_time.py` /
  `exp_frame_states.py` / `exp_permouse_xstage.py` / `exp_xstage_scale_check.py` (+ `--antact` / `--pca` variants) and the
  the task-split trace row (DPA | Go | NoGo) which IS panel A since 2026-08-31 (standalone
  preview `fig_traj_tasksplit.py`; the four-code 2×4 row is now the supp's panel A); three
  Fig-3 cards pinned in the gallery Main tab (canonical / pca20 / antact).
- **`story_figure_reproduction.md`** — the dPCA story figure (`fig_dpca_story_main.py`; its trajectory grid + axis mixing are **ED Fig 3** since 2026-09-15 (citation order) via `fig_ed6_dpca.py`, the rest is cut; it was ED 9 before:
  trajectory grid, axis-mixing, linking plane, shared-memory scatter): full reproduction guide
  (hypotheses, data, every routine, exact math/windows, results, caveats).
- `overview.md` — subproject overview & results.
- `flows_handoff.md` — design history / handoff for the flows work (read its SETTLED block first).
- `story_figure_methods.md` — condensed how-to for `pca/fig_dpca_story_main.py`.
- **`story_figure_review.md`** — review log: bugs fixed (gated-deformation push, panel J; sec-3 flows
  switched to partial pooling → CV now positive, two shared landscapes for the two epochs) & standing
  caveats (variance is a proxy; sec-4 push depth is fit from data, gate profile is a modeling choice).
- **Per-animal shattering + refit-dPCA bootstrap (2026-09-15):** `exp_shatter_permouse.py` (0.7 min; SD_LOO leave-one-mouse-out
  jackknife of the 462-dichotomy shattering + SD_MOUSE own-population → ED 2c) and `exp_dpca_refit_boot.py` (6.9 min; dPCA
  re-fitted on 1,000 resampled nine-mouse sets → DPCA_REFIT_BOOT, the interval ED 3b draws; the fixed-axis bootstrap was
  over-confident). Both merge into `figures/pseudo/dimensionality/results.pkl`; numbers in `dimensionality.md` top notes.
- **Out-of-context plane test (2026-09-07, ED 6e → ED 1c since 2026-09-15 (citation order), drawn by `fig_ed2_plane.py`):** `exp_ooc_plane_pseudo.py` (pooled) + `exp_ooc_plane.py`
  (per-mouse) → caches `OOC_PLANE_PSEUDO_nopca` / `OOC_PLANE_nopca` in results.pkl (**since 2026-09-15 (later) the
  `--alltrials` caches `*_nopca_all` — all laser-off trials, not correct-only — are CANONICAL for ED 1c and, via
  `exp_permouse_plane.py --nopca --alltrials` → `PM_PLANE_nopca_all`, for Fig 3c/ED 1d; `--correctonly` on the two
  figure scripts = the old build; `dimensionality.md` top block**); `fig_ooc_plane_ed.py` = the
  ED 6e component, `fig_ooc_plane.py` = the full audit (gallery `tmp/` only). Design, numbers and the
  by-construction finding it replaces: `dimensionality.md` 2026-09-07 block.
