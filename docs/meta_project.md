# Project: mPFC Population Geometry (Dual Task)

## Paper hypothesis

mPFC encodes sample identity and lick action on near-orthogonal axes. With learning (Naive→Expert), DPA delay-period activity moves along the **lick (choice) axis** — away from the lick sector — without disrupting sample discriminability along the **sample axis**. Distance to the lick boundary predicts within-animal DPA accuracy.

---

## Three subprojects

### 1. Overlaps (`/home/leon/dual/overlaps/`)
Cross-generalising decision codes (CCGD): decoders trained on sample identity and lick choice cross-generalise across time. Measures how the sample and choice codes co-evolve during the delay in the sample×choice plane.
- See `docs/overlaps/overview.md`, `docs/overlaps/routines.md`, `docs/overlaps/feedback.md`

### 2. PCA (`/home/leon/dual/decode/`)
Pseudo-population PCA on delay-period activity across mice. Figure 2E of the paper.
- Script: `decode/singleCrossCond.org` (Jupyter/org notebook), `decode/fig3BF.py`
- Docs to be created when work begins: `docs/pca/`

### 3. Decoders (`/home/leon/dual/decode/`)
Single-neuron and population decoders tracking sample and lick axes across learning days. Figure 3 of the paper.
- Script: `decode/fig3BF.py`
- Key outputs: sample axis, lick axis, Δlick per mouse, distance to lick boundary
- Docs to be created when work begins: `docs/decoders/`

---

## How the subprojects relate

All three use the same 9 mice, same recording sessions, same trial structure — see `docs/shared_data.md`.
- **Overlaps** works on `X_single` (CCGD decision values, cross-time generalisation matrix)
- **PCA** works on `traj_all` (per-trial PCA projections)
- **Decoders** projects onto sample/lick axes defined from single-neuron weights

The sample and choice axes in overlaps correspond to the sample and lick axes in decoders — they capture the same geometry from different analysis angles.
