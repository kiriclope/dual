# Rebuild timings

Append-only record of what each rebuild actually costs, written by `log_timings.py` (see its docstring).
Consult it before estimating; do not guess. `source` says whether the number came from a stopwatch in
place or from somewhere less direct.

| finished | run | step | minutes | source |
|---|---|---|---|---|
| 2026-09-08 13:10 | axis variant _t1 (2026-09-08 build) | step 0: overlaps traces/states | 0.3 | measured, parsed from the original run log |
| 2026-09-08 13:30 | axis variant _t1 (2026-09-08 build) | step 1: pooled cross-task matrices | 19.7 | measured, parsed from the original run log |
| 2026-09-08 13:30 | axis variant _t1 (2026-09-08 build) | step 2: sandbox pca caches | 0.1 | measured, parsed from the original run log |
| 2026-09-08 13:33 | axis variant _t1 (2026-09-08 build) | running exp_dpca_count.py | 3.2 | measured, parsed from the original run log |
| 2026-09-08 13:34 | axis variant _t1 (2026-09-08 build) | running exp_axis_frame.py --nopca | 0.8 | measured, parsed from the original run log |
| 2026-09-08 13:34 | axis variant _t1 (2026-09-08 build) | running exp_permouse_frame.py --nopca | 0.4 | measured, parsed from the original run log |
| 2026-09-08 13:35 | axis variant _t1 (2026-09-08 build) | running exp_permouse_frame.py --nopca --mddec | 0.4 | measured, parsed from the original run log |
| 2026-09-08 13:35 | axis variant _t1 (2026-09-08 build) | running exp_permouse_plane.py --nopca | 0.2 | measured, parsed from the original run log |
| 2026-09-08 13:35 | axis variant _t1 (2026-09-08 build) | running exp_permouse_xstage.py --nopca | 0.1 | measured, parsed from the original run log |
| 2026-09-08 13:36 | axis variant _t1 (2026-09-08 build) | running exp_plane_frame.py --nopca | 1.3 | measured, parsed from the original run log |
| 2026-09-08 13:36 | axis variant _t1 (2026-09-08 build) | running exp_parallelism.py --nopca | 0.2 | measured, parsed from the original run log |
| 2026-09-08 13:37 | axis variant _t1 (2026-09-08 build) | running exp_neuron_sel.py --nopca | 0.1 | measured, parsed from the original run log |
| 2026-09-08 13:42 | axis variant _t1 (2026-09-08 build) | running exp_cdec_support.py | 5.5 | measured, parsed from the original run log |
| 2026-09-08 13:42 | axis variant _t1 (2026-09-08 build) | variant caches saved to results_t1.pkl | 0.0 | measured, parsed from the original run log |
| 2026-09-08 13:42 | axis variant _t1 (2026-09-08 build) | restoring canonical caches | 0.0 | measured, parsed from the original run log |
| 2026-09-08 13:43 | axis variant _t1 (2026-09-08 build) | step 3: renders | 0.7 | measured, parsed from the original run log |
| 2026-09-08 13:43 | axis variant _t1 (2026-09-08 build) | **TOTAL** | 33.0 | measured, parsed from the original run log |
| 2026-09-08 13:10 | axis variant _te (2026-09-08 build) | **TOTAL** | 16.0 | measured, first and last step mark of the run log |
| 2026-09-08 14:33 | axis variant _tc (2026-09-08 build) | **TOTAL** | 21.0 | measured, first and last step mark of the run log |
| 2026-09-08 15:10 | axis variant _td (2026-09-08 build) | **TOTAL** | 27.0 | measured, first and last step mark of the run log |
| 2026-09-09 13:5 | stale decision-window refresh | exp_dimensionality_jk.py | 2.0 | approximate, reported by the agent that ran it |
| 2026-09-09 13:5 | stale decision-window refresh | exp_dimensionality_ci.py | 10.0 | approximate, reported by the agent that ran it |
| 2026-09-09 14:0 | stale decision-window refresh | exp_learning_delta.py | 4.0 | approximate, reported by the agent that ran it |
| 2026-09-09 14:12 | axis variant _t1 (2026-09-09 rebuild) | step 0: overlaps traces/states | 0.3 | measured, parsed from the run log |
| 2026-09-09 14:18 | axis variant _t1 (2026-09-09 rebuild) | step 1: pooled cross-task matrices | 5.3 | measured, parsed from the run log |
| 2026-09-09 14:18 | axis variant _t1 (2026-09-09 rebuild) | step 2: sandbox pca caches | 0.1 | measured, parsed from the run log |
| 2026-09-09 14:22 | axis variant _t1 (2026-09-09 rebuild) | running exp_dpca_count.py | 4.5 | measured, parsed from the run log |
| 2026-09-09 14:24 | axis variant _t1 (2026-09-09 rebuild) | running exp_axis_frame.py --nopca | 1.8 | measured, parsed from the run log |
| 2026-09-09 14:25 | axis variant _t1 (2026-09-09 rebuild) | running exp_permouse_frame.py --nopca | 0.5 | measured, parsed from the run log |
| 2026-09-09 14:25 | axis variant _t1 (2026-09-09 rebuild) | running exp_permouse_frame.py --nopca --mddec | 0.4 | measured, parsed from the run log |
| 2026-09-09 14:25 | axis variant _t1 (2026-09-09 rebuild) | running exp_permouse_plane.py --nopca | 0.3 | measured, parsed from the run log |
| 2026-09-09 14:26 | axis variant _t1 (2026-09-09 rebuild) | running exp_permouse_xstage.py --nopca | 0.1 | measured, parsed from the run log |
| 2026-09-09 14:27 | axis variant _t1 (2026-09-09 rebuild) | running exp_plane_frame.py --nopca | 1.3 | measured, parsed from the run log |
| 2026-09-09 14:27 | axis variant _t1 (2026-09-09 rebuild) | running exp_parallelism.py --nopca | 0.2 | measured, parsed from the run log |
| 2026-09-09 14:27 | axis variant _t1 (2026-09-09 rebuild) | running exp_neuron_sel.py --nopca | 0.1 | measured, parsed from the run log |
| 2026-09-09 14:33 | axis variant _t1 (2026-09-09 rebuild) | running exp_cdec_support.py | 5.8 | measured, parsed from the run log |
| 2026-09-09 14:33 | axis variant _t1 (2026-09-09 rebuild) | running exp_pceta_cv.py | 0.4 | measured, parsed from the run log |
| 2026-09-09 14:33 | axis variant _t1 (2026-09-09 rebuild) | variant caches saved to results_t1.pkl | 0.0 | measured, parsed from the run log |
| 2026-09-09 14:33 | axis variant _t1 (2026-09-09 rebuild) | restoring canonical caches | 0.0 | measured, parsed from the run log |
| 2026-09-09 14:34 | axis variant _t1 (2026-09-09 rebuild) | step 3: renders | 0.7 | measured, parsed from the run log |
| 2026-09-09 14:34 | axis variant _t1 (2026-09-09 rebuild) | **TOTAL** | 21.9 | measured, parsed from the run log |
| 2026-09-09 14:38 | cross-task matrices, variant _t1 (--acc --nopca) | COLD: cache empty, 20.5 GB tensor loaded | 3.1 | measured |
| 2026-09-09 14:40 | cross-task matrices, variant _t1 (--acc --nopca) | WARM: cache hit, tensor NOT loaded | 2.3 | measured |
| 2026-09-09 15:37 | axis variant _fmd (sample = full mid-delay) | step 0: overlaps traces/states | 0.2 | measured, parsed from the run log |
| 2026-09-09 15:40 | axis variant _fmd (sample = full mid-delay) | step 1: pooled cross-task matrices | 2.3 | measured, parsed from the run log |
| 2026-09-09 15:40 | axis variant _fmd (sample = full mid-delay) | step 2: sandbox pca caches | 0.0 | measured, parsed from the run log |
| 2026-09-09 15:40 | axis variant _fmd (sample = full mid-delay) | decision window unchanged (FK '-'): canonical AW/FITDATA kept | 0.0 | measured, parsed from the run log |
| 2026-09-09 15:43 | axis variant _fmd (sample = full mid-delay) | running exp_dpca_count.py | 3.5 | measured, parsed from the run log |
| 2026-09-09 15:44 | axis variant _fmd (sample = full mid-delay) | running exp_axis_frame.py --nopca | 0.8 | measured, parsed from the run log |
| 2026-09-09 15:44 | axis variant _fmd (sample = full mid-delay) | running exp_permouse_frame.py --nopca | 0.5 | measured, parsed from the run log |
| 2026-09-09 15:45 | axis variant _fmd (sample = full mid-delay) | running exp_permouse_frame.py --nopca --mddec | 0.4 | measured, parsed from the run log |
| 2026-09-09 15:45 | axis variant _fmd (sample = full mid-delay) | running exp_permouse_plane.py --nopca | 0.3 | measured, parsed from the run log |
| 2026-09-09 15:45 | axis variant _fmd (sample = full mid-delay) | running exp_permouse_xstage.py --nopca | 0.1 | measured, parsed from the run log |
| 2026-09-09 15:46 | axis variant _fmd (sample = full mid-delay) | running exp_plane_frame.py --nopca | 1.3 | measured, parsed from the run log |
| 2026-09-09 15:47 | axis variant _fmd (sample = full mid-delay) | running exp_parallelism.py --nopca | 0.1 | measured, parsed from the run log |
| 2026-09-09 15:47 | axis variant _fmd (sample = full mid-delay) | running exp_neuron_sel.py --nopca | 0.1 | measured, parsed from the run log |
| 2026-09-09 15:53 | axis variant _fmd (sample = full mid-delay) | running exp_cdec_support.py | 5.9 | measured, parsed from the run log |
| 2026-09-09 15:53 | axis variant _fmd (sample = full mid-delay) | running exp_pceta_cv.py | 0.4 | measured, parsed from the run log |
| 2026-09-09 15:53 | axis variant _fmd (sample = full mid-delay) | variant caches saved to results_fmd.pkl | 0.0 | measured, parsed from the run log |
| 2026-09-09 15:53 | axis variant _fmd (sample = full mid-delay) | restoring canonical caches | 0.0 | measured, parsed from the run log |
| 2026-09-09 15:54 | axis variant _fmd (sample = full mid-delay) | step 3: renders | 0.7 | measured, parsed from the run log |
| 2026-09-09 15:54 | axis variant _fmd (sample = full mid-delay) | **TOTAL** | 16.7 | measured, parsed from the run log |
| 2026-09-10 17:01 | Fig 2 panel b -> design-contrast basis | exp_contrast_var.py (CONTRAST_VAR + CONTRAST_NULL; 2 sets x 2 windows x 2 stages, 30 splits, leave-one-mouse-out jackknife + shuffle null; cache-only, no X reload) | 12.0 | measured |
| 2026-09-10 17:01 | Fig 2 panel b -> design-contrast basis | fig_dimensionality_main.py re-render (any flag) | 2.0 | measured |
