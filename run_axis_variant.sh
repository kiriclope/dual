#!/bin/bash
# run_axis_variant.sh — rebuild ONE axis-window variant end to end (caches + renders + log).
#
# Durable home (2026-09-09): this and build_variant_page.py used to live in a Claude scratchpad, which is
# cleaned up between sessions. They are now tracked in the repo root beside make_submission_figs.py.
#
#   VARIANT_LOG=.variant_tmp/variant_t1.log ./run_axis_variant.sh _t1 54-65 decision_t1 _tewin 2>&1 \
#     | tee .variant_tmp/variant_t1.log
#   (VARIANT_LOG is optional; when set, the per-step timings are appended to docs/rebuild_timings.md
#    on exit. Never estimate a rebuild cost — read that file.)
#
#   ./run_axis_variant.sh _t1 54-65 decision_t1 _tewin
#     $1 V       file/key suffix                 (_te | _tc | _t1 | _td | _f2)
#     $2 CB      choice/test axis bins 'a-b'     (54-59 | 54-62 | 54-65 | 57-62 | 57-65)
#     $3 FK      window key inside fits_inputs$FITSUF.pkl   (e.g. decision_t1); '-' = leave the
#                decision window canonical (for a SAMPLE-only variant, with SEED_MD below)
#     $4 FITSUF  which candidate-window fits cache to read  (default _tewin)
#     env SEED_MD  optional: pickle {'bins','AW'} replacing the SAMPLE window AW['md']
#                  (build one with a nanmean over the bins you want; see the block below)
#
# The five variants (sample/GNG axis is 36-38 in every one; test odor is 9.0-10.0 s):
#     A _te  54-59  9.0-10.0  test odor only
#     C _tc  54-62  9.0-10.5  test odor + 0.5 s      <- CANONICAL since 2026-09-08
#     B _t1  54-65  9.0-11.0  test odor + 1 s
#     D _td  57-62  9.5-10.5
#     E _f2  57-65  9.5-11.0
#
# WARNING — this SANDBOXES the canonical caches: it copies results.pkl and fits_inputs.pkl aside, seeds
# them with the variant's decision window, runs the cache jobs, saves results$V.pkl, then restores the
# originals (also on error, via trap). NEVER run it while anything else is reading or writing
# pca/figures/pseudo/dimensionality/results.pkl — one job at a time.
#
# Then build and publish the page:
#   /home/leon/mambaforge/envs/dual/bin/python build_variant_page.py _t1 <log> "<title>" <out.html> "<defn>"
#   Artifact: file_path = <out.html>, url = the variant's artifact (B = e2134a9b-74d1-4cdc-baf9-53c3929768af)
set -u
V=$1; CB=$2; FK=$3; FITSUF=${4:-_tewin}
PY=/home/leon/mambaforge/envs/dual/bin/python
R=/home/leon/dual
S=${VARIANT_TMP:-$R/.variant_tmp}
D=$R/pca/figures/pseudo/dimensionality
mkdir -p "$S"
export DUAL_AXSUF=$V DUAL_SAMPLE_BINS=36-38 DUAL_CHOICE_BINS=$CB
cd "$R"

echo "== $(date) [$V] step 0: overlaps traces/states"
(cd pca && $PY exp_traj_orig.py 2>&1 | grep -v Warning | tail -1 && $PY exp_frame_states.py 2>&1 | grep -v Warning | tail -1)

echo "== $(date) [$V] step 1: pooled cross-task matrices"
(cd overlaps && $PY fig_ccgp_matrices_pseudo.py --acc --nopca 2>&1 | grep -E "cached|Traceback|Error" | tail -2)

echo "== $(date) [$V] step 2: sandbox pca caches"
cp "$D/results.pkl" "$S/results_canon.pkl"; cp "$D/fits_inputs.pkl" "$S/fits_inputs_canon.pkl"
restore() { echo "== $(date) [$V] restoring canonical caches"; cp "$S/results_canon.pkl" "$D/results.pkl"; cp "$S/fits_inputs_canon.pkl" "$D/fits_inputs.pkl"; }
trap restore EXIT
if [ "$FK" = "-" ]; then
  echo "== $(date) [$V] decision window unchanged (FK '-'): canonical AW/FITDATA kept"
else
FK=$FK FITSUF=$FITSUF $PY - <<'EOF'
import pickle, os
D = '/home/leon/dual/pca/figures/pseudo/dimensionality'; FK = os.environ['FK']; FITSUF = os.environ['FITSUF']
fi = pickle.load(open(f'{D}/fits_inputs.pkl', 'rb')); fv = pickle.load(open(f'{D}/fits_inputs{FITSUF}.pkl', 'rb'))
fi['AW']['decision'] = fv['AW'][FK]; pickle.dump(fi, open(f'{D}/fits_inputs.pkl', 'wb'))
R = pickle.load(open(f'{D}/results.pkl', 'rb')); A = pickle.load(open(f'{D}/results{FITSUF}.pkl', 'rb'))['FITDATA']
n = 0
for (s, w, st), v in A.items():
    if w == FK: R['FITDATA'][(s, 'decision', st)] = v; n += 1
pickle.dump(R, open(f'{D}/results.pkl', 'wb')); print(f'seeded: AW decision:={FK}; FITDATA decision from results{FITSUF} ({n} keys)')
EOF
fi

# Optional SAMPLE-window change (SEED_MD=<pickle with {'bins','AW'}>, e.g. figures/.../aw_md_full.pkl).
# The decision seeding above pulls its window from a prebuilt candidate cache; the sample axis has no such
# cache, so pass one. AW['md'] is replaced and every downstream job in the loop recomputes on it: panel b
# (exp_cdec_support), panel c (exp_dpca_count), panel d (exp_pceta_cv), the plane/per-mouse caches, and the
# overlaps side follows DUAL_SAMPLE_BINS on its own. NOTE the raw FITDATA md keys (cm_var / pceta, the
# UNCROSS-VALIDATED fallback and the ED PR ladder) are NOT recomputed here and stay on the old window.
if [ -n "${SEED_MD:-}" ]; then
  SEED_MD=$SEED_MD $PY - <<'EOF'
import pickle, os, numpy as np
D = '/home/leon/dual/pca/figures/pseudo/dimensionality'
src = pickle.load(open(os.environ['SEED_MD'], 'rb')); b = np.asarray(src['bins'])
fi = pickle.load(open(f'{D}/fits_inputs.pkl', 'rb'))
assert fi['AW']['md'].shape == src['AW'].shape, (fi['AW']['md'].shape, src['AW'].shape)
fi['AW']['md'] = src['AW']; pickle.dump(fi, open(f'{D}/fits_inputs.pkl', 'wb'))
print(f'seeded: AW md := bins {b[0]}-{b[-1]} ({b[0]/6:.2f}-{(b[-1]+1)/6:.2f} s)')
EOF
  # A failed seed must ABORT: continuing would silently produce a variant on the canonical window.
  if [ $? -ne 0 ]; then echo "== [$V] SEED_MD FAILED — aborting"; restore; exit 1; fi
fi

cd pca
export DUAL_DPCA_WINS=md,decision
# exp_pceta_cv.py added 2026-09-09: panel d is cross-validated, so the variant needs its own
# pceta_cv / cm_var_cv keys or fig_dimensionality_main falls back to the raw (uncross-validated) ones.
for job in "exp_dpca_count.py" "exp_axis_frame.py --nopca" "exp_permouse_frame.py --nopca" \
           "exp_permouse_frame.py --nopca --mddec" "exp_permouse_plane.py --nopca" \
           "exp_permouse_xstage.py --nopca" "exp_plane_frame.py --nopca" "exp_parallelism.py --nopca" \
           "exp_neuron_sel.py --nopca" "exp_cdec_support.py" "exp_pceta_cv.py"; do
  echo "== $(date) [$V] running $job"; $PY $job 2>&1 | grep -v Warning | tail -2
done
unset DUAL_DPCA_WINS
cp "$D/results.pkl" "$D/results$V.pkl"
echo "== $(date) [$V] variant caches saved to results$V.pkl"
restore; trap - EXIT

echo "== $(date) [$V] step 3: renders"
DUAL_RES=$D/results$V.pkl $PY fig_dimensionality_main.py 2>&1 | grep -E "^E-gen|^F:|^C-dec: decision|saved|Traceback|Error"
DUAL_RES=$D/results$V.pkl $PY fig_manifold_main.py --nopca 2>&1 | grep -E "^b:|^e-pm|^f: plane only|saved|Traceback|Error"
cd ../overlaps
$PY fig_overlaps_main_native.py 2>&1 | grep -E "^A depth|^B\[|^C|^D|saved|Traceback|Error"
$PY fig_behavior_opto_main.py 2>&1 | grep -E "corr r=|LMM|trade-off|saved|Traceback|Error" | head -12
echo "== $(date) [$V] VARIANT_DONE"

# Record what this actually cost, so nobody has to guess next time (see docs/rebuild_timings.md).
if [ -n "${VARIANT_LOG:-}" ] && [ -f "$VARIANT_LOG" ]; then
  $PY "$R/log_timings.py" parse "$VARIANT_LOG" --run "axis variant $V" || true
fi
