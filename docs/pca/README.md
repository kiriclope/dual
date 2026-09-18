> **2026-09-18 — A TENTH EXTENDED DATA FIGURE, THE UNSUPERVISED GEOMETRY PAGE (ED 4):** `pca/fig_ed_geometry.py`
> enters as **ED 4** (the same geometry read with no axes chosen: t-SNE maps + 20-draw kNN purity, per-mouse manifold
> geometry at MATCHED trial counts, cross-validated unsupervised axes, UMAP trajectories on DPA trials), first cited in
> §2 beside Fig 2d. Everything after it shifts: plane 4 → **5**, dPCA 5 → **6**, coupling 6 → **7**, opto placeholder
> 7 → **8**, chronic 8 → **9**, laser 9 → **10**. Scripts keep their file names. Dated notes below keep the numbering
> of their day. Renumbering checklist: `make_ed_figures.py` FIGS, `build_ed_artifact.py` SPECS, `serve_figures.py`
> SUPP, each script's `ed_fig{N}` path AND any `ED_N = <n>` constant (`fig_ed_behavior.py`, `fig_ed_imaging.py`,
> `fig_ed_opto_validation.py` have one), the `Extended Data Fig. N |` caption, cross-references in OTHER figures'
> captions, and in `docs/paper/` the citations, the `**ED Fig. N |` legend headings and the plural "Figs 7 and 10".

> **2026-09-16 — EXTENDED DATA RENUMBERED AGAIN (nine figures, first-citation order):** behaviour = ED 1 (new, `overlaps/fig_ed_behavior.py`),
> imaging/drift = ED 2 (new, `pca/fig_ed_imaging.py`), dimensionality 2 → **3**, plane 1 → **4**, dPCA 3 → **5**, coupling 4 → **6**,
> opto validation placeholder = ED 7 (new), chronic 5 → **8**, laser 6 → **9**. Dated notes below keep the numbering of their day.

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
- **ED 4 unsupervised geometry (2026-09-18):** `fig_ed_geometry.py` (0.2 min, caches only; `--nocap` for the submission
  build, `--printcap` dumps the caption so the `results_draft.md` legend can be kept in step). Reads four caches:
  `geometry_cache.pkl` (`fig_geometry_main.py --recompute`, 0.6 min — the t-SNE maps and the UMAP DPA trajectories),
  `manifold_purity_draws.pkl` (`exp_manifold_purity_draws.py` — kNN purity averaged over 20 pseudo-trial draws with paired
  shuffle nulls, UMAP and t-SNE), `manifold_matched.pkl` (`exp_manifold_matched.py`, 1.6 min — the per-mouse manifold
  geometry at matched trial counts) and `unsup_axes_cv.pkl` (`fig_unsup_axes_cv.py`, needs `cmbin_splits.pkl` from
  `exp_cmbin_splits.py`, 0.5 min and one 20 GB X pass). **Two traps this page exists to avoid.** (i) A single 2-D embedding
  of ~150 pseudo-trials is a random variable: the same data gave sample purity 0.48 on one draw and 0.61 on the next, so
  every purity quoted is a 20-draw mean against a draw-paired null, never one map. (ii) `exp_manifold_geometry.py` (the
  first pass) compared clouds of different size and scaled each by its own s.d.; that made the dual manifold look
  higher-dimensional (PR 37 → 49, p .004) and fixed the radius at sqrt(nneu) in every set. `exp_manifold_matched.py` matches
  n within mouse/stage/window and uses one condition-agnostic scale: **the PR effect disappears** (all p ≥ .20) and what
  survives is the lower sample separation on every dual set and the rise of the Go/NoGo separation with learning. Use the
  matched script for any per-mouse manifold claim.
- **ED 2 imaging/drift (2026-09-16):** `fig_ed_imaging.py` (~1 min; loads the 2.5 GB canonical CCGD tensor) — neurons per mouse
  (113–693, 3,319) and per-session held-out decodability of the sample (mid-delay) and choice (decision) codes: sample 0.74–0.77,
  choice 0.60–0.66 in every session, per-mouse s.d. 0.03–0.06 — the drift check behind Fig. 3e's cross-stage transfer.
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
