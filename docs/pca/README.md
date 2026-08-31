# PCA Subproject

Pseudo-population dPCA on the dual working-memory task (9 mice). Latent dynamics, flow fields, and the
no-lick learning push. See `docs/meta_project.md` for the paper overview.

## Docs
- **`dimensionality.md`** — ★ the CURRENT main Fig 2 (`fig_dimensionality_main.py`, since 2026-08-10):
  honest cvPCA + shattering + PC-coding dimensionality; methods, settled numbers, gotchas. ALSO carries
  main **Fig 3** (`fig_manifold_main.py`, since 2026-08-30; dated blocks — read "FINAL Fig 3 structure"
  + the storyboard re-origin note): the one-manifold figure (traces · boundary-centred storyboard ·
  plane sufficiency bars + per-mouse · cosines · cross-stage decoding) and its ED companion
  `fig_manifold_supp.py`; caches from `exp_permouse_plane.py` / `exp_plane_frame.py` / `exp_axis_time.py`.
- **`story_figure_reproduction.md`** — the dPCA story figure (`fig_dpca_story_main.py`, now **ED Fig 9**:
  trajectory grid, axis-mixing, linking plane, shared-memory scatter): full reproduction guide
  (hypotheses, data, every routine, exact math/windows, results, caveats).
- `overview.md` — subproject overview & results.
- `flows_handoff.md` — design history / handoff for the flows work (read its SETTLED block first).
- `story_figure_methods.md` — condensed how-to for `pca/fig_dpca_story_main.py`.
- **`story_figure_review.md`** — review log: bugs fixed (gated-deformation push, panel J; sec-3 flows
  switched to partial pooling → CV now positive, two shared landscapes for the two epochs) & standing
  caveats (variance is a proxy; sec-4 push depth is fit from data, gate profile is a modeling choice).
