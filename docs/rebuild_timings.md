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
| 2026-09-10 17:26 | Fig 2b contrast basis REVERTED | revert + re-render + verify byte-identical to d81c59c | 6.0 | measured |
| 2026-09-14 12:51 | Fig 2b cvPCA bias disclosure | exp_cvpca_bias_check.py (all six checks: noise, sim, folds, trials --figure, scaling, align; cache-only) | 1.0 | measured |
| 2026-09-15 11:55 | ED rebuild 2026-09-15 | exp_coupling_variants.py (Fig 4c coupling under l2/l1/lda; 3 main_panels subprocesses) | 0.4 | measured (wall clock in the run log) |
| 2026-09-15 11:55 | ED rebuild 2026-09-15 | fig_overlaps_norm_robustness_supp.py (+ ed3_cache dump) | 0.1 | measured (wall clock in the run log) |
| 2026-09-15 11:55 | ED rebuild 2026-09-15 | fig_overlaps_common_axis_supp.py (loads X_all; + ed3_cache dump) | 0.6 | measured (wall clock in the run log) |
| 2026-09-15 11:55 | ED rebuild 2026-09-15 | fig_overlaps_lick_control_supp.py (loads X_all; + ed3_cache dump) | 0.4 | measured (wall clock in the run log) |
| 2026-09-15 11:56 | ED rebuild 2026-09-15 | make_ed_figures.py (6 native EDs x 2 builds + 183 mm PDFs + share copies) | 1.1 | measured |
| 2026-09-15 12:22 | ED rebuild 2026-09-15 | fig_ccgp.py --canon (per-mouse CCGP on the canonical windows, 30 shuffles) | 2.0 | measured |
| 2026-09-15 12:30 | ED rebuild 2026-09-15 | fig_overlaps_common_axis_supp.py, HELD-OUT 5-fold inner-CV axis fits (loads X_all) | 1.0 | measured (wall clock in the run log) |
| 2026-09-15 12:30 | ED rebuild 2026-09-15 | fig_overlaps_lick_control_supp.py, HELD-OUT 5-fold inner-CV axis fits (loads X_all) | 0.6 | measured (wall clock in the run log) |
| 2026-09-15 12:54 | ED 3 reinstatement 2026-09-15 | run_overlaps.py --pool-stages --scaler none --targets choice --contexts all (9 mice, one pooled fit each) | 7.6 | measured |
| 2026-09-15 12:56 | ED 3 reinstatement 2026-09-15 | exp_common_axis_ccgd.py (two tensors, ~1 GB each) | 0.7 | approx, wall clock observed |
| 2026-09-15 12:56 | ED 3 reinstatement 2026-09-15 | exp_lick_control_ccgd.py (tensor + behaviour .mat files) | 0.6 | approx, wall clock observed |
| 2026-09-15 15:40 | ED 1 canonical windows 2026-09-15 | exp_ed1_spectra.py (12-cond + DPA spectra, PR + LOO jackknife, shuffle null, md + decision) | 1.3 | measured |
| 2026-09-15 15:47 | ED 1f selection test 2026-09-15 | exp_dpca_count.py --alltrials (4 windows) | 6.9 | measured |
| 2026-09-15 16:46 | all-trial replications 2026-09-15 | exp_permouse_plane.py --nopca --alltrials | 0.3 | measured |
| 2026-09-15 16:49 | all-trial replications 2026-09-15 | exp_ooc_plane.py --nopca --alltrials | 2.2 | measured |
| 2026-09-15 16:49 | all-trial replications 2026-09-15 | exp_ooc_plane_pseudo.py --nopca --alltrials | 0.4 | measured |
| 2026-09-15 17:04 | all-trial replications 2026-09-15 | fig_ccgp.py --canon --alltrials | 19.0 | measured |
| 2026-09-15 17:42 | per-animal shattering 2026-09-15 | exp_shatter_permouse.py (LOO jackknife 18x8 + own-population 18x20, 462 dich., 36 workers) | 0.7 | measured |
| 2026-09-15 17:58 | refit-dPCA bootstrap 2026-09-15 | exp_dpca_refit_boot.py (X_all_blcenter load + 1000 mouse draws x 2 stages refit, 36 workers) | 6.9 | measured |
| 2026-09-15 18:03 | trial-level lick control 2026-09-15 | exp_lick_control_ccgd.py re-run with the corrected .mat sample mapping | 0.1 | measured |
| 2026-09-15 18:07 | trial-level lick control 2026-09-15 | run_overlaps.py --scaler none --targets choice --tag trial (seeded folds, trial index column; 9 mice x 2 stages) | 8.1 | measured |
| 2026-09-15 18:07 | trial-level lick control 2026-09-15 | exp_lick_control_trial.py (tensor + .mat per-trial licks) | 0.1 | measured |
| 2026-09-16 12:56 | supplementary review 2026-09-16 | fig_ed_behavior.py (47 .mat files, 5 GEE fits x 2 stages) | 1.0 | measured in-session |
| 2026-09-16 12:56 | supplementary review 2026-09-16 | fig_ed_imaging.py (loads the 2.5 GB canonical tensor) | 1.0 | measured in-session |
