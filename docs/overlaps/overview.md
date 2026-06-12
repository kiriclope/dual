# Overlaps Subproject Overview

## What overlaps measures

Cross-generalising decision codes (CCGD): one decoder trained on sample identity (sample code), another on lick choice (choice code), each cross-generalising across training and test time. The resulting time×time generalisation matrices are averaged over the diagonal to get a scalar per trial per time bin.

Projecting onto sample and choice codes simultaneously gives a 2D state space where:
- **x-axis** = sample code (separates A from B)
- **y-axis** = choice code (separates lick from no-lick)

The paper hypothesis predicts that Expert delay-period states occupy the **lower** (no-lick) half of the choice axis for DPA, and that A/B separate cleanly along the sample axis.

---

## Active figures

| Script | Shows | Trials |
|---|---|---|
| `plot_traj2d.py` | 2D trajectory over time + KDE strip | correct only |
| `plot_flow2d.py` | Empirical flow field (speed + streamlines) | all laser-off |
| `plot_scatter_perf.py` | Δ choice loc. vs Δ performance | x: correct; y: all |
| `plot_geometry.py` | Late-delay positions per (mouse, odor_pair) | correct only |
| `plot_marginal.py` | 1D code vs time (Naive/Expert) | all laser-off |
| `plot_occupancy.py` | 2D KDE occupancy at BINS_LATE | all laser-off |
| `plot_scatter_ab.py` | Per-animal A/B endpoints | all laser-off |

See `docs/overlaps/routines.md` for run commands and output paths.

---

## Locked-in design choices

- **Colors**: sample A = `#332288` (indigo), sample B = `#44AA99` (teal)
- **Condition titles**: DPA / Go / NoGo (strip "Dual" prefix)
- **Trajectory limits**: xlim=(-4,4), ylim=(-2,6), yticks=[-2,0,2,4,6]
- **BINS_LATE** = `BINS_DELAY[int(0.6*len):]` ≈ bins 39–53 (last 40% of delay)
- **Stage labels**: Naive / Expert (from `days=['first','last']`)
- **Fixed points**: mean of grand_mean_traj over BINS_LATE (NOT speed minimum)
- **Flow field heatmap**: magma colormap, hybrid speed (BINS_LATE at fixed points, all-delay elsewhere)
- **Condition naming**: `cond.replace('DualGo','Go').replace('DualNoGo','NoGo')`
