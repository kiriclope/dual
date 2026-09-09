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
