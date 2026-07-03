# Overlaps MAIN figure — hypothesis, method & reproduction

The overlaps-only main paper figure: a 5-panel arc (A→E) showing the **dual code**, its
**learning-driven no-lick push**, and the **acute-optogenetic test** of the same link, all read on
one axis. Assembled by `overlaps/fig_overlaps_main.py` from five per-analysis panel scripts
(E = the laser ON−OFF causal analog of D).

Everything runs from `/home/leon/dual/overlaps` with
`/home/leon/mambaforge/envs/dual/bin/python`.

---

## 1. Hypothesis / scientific arc

The mPFC population carries, *simultaneously in the delay*, two orthogonal codes in the
sample × choice CCGD frame:
- a **sample memory** (which odor was shown — held stably across the delay), and
- a **choice / lick code** (the lick vs no-lick action set), non-flat already in the delay =
  a *maintained action set*, not test foreknowledge (the correct lick still needs the future test).

With learning, the DPA delay state is pulled into the **no-lick half** of the choice axis
("no-lick push" / "well deepening"), and — across mice — **how far a mouse pushes correlates with
how much DPA accuracy it gains**. The push is DPA-specific (no matching Go/NoGo relationship). This
is the acute-perturbation counterpart to the laser ON−OFF scatter (`laser_onoff.md`).

**Codes / conventions:** Sample A = odor_pairs [0,1] (#332288 indigo), Sample B = [2,3] (#44AA99
teal). "Depth" = late-delay position on the choice code; negative = no-lick half.

## 2. What the figure shows (5 panels, all on `trainLD_TEST`)

| panel | content | headline stat |
|---|---|---|
| **A** | 1-D codes over the trial: sample / choice / test / task (task rides the choice axis). Read-out spans the test epoch → the **test code is valid** (no pre-test confound). | sample epoch-invariant; choice code non-flat in delay |
| **B** | The no-lick push in the sample × choice plane: DPA state Naive→Expert (focused 1×2, Naive \| Expert). | Expert DPA state sits deeper in no-lick |
| **C** | Well deepening: per-mouse late-delay depth Naive→Expert + maximal-LMM stars. | pooled dz=−0.53, 8/9 mice, LMM **p=0.098** (trend) |
| **D** | Δ accuracy vs Δ depth (**Expert−Naive**, learning), 1×2: ΔDPA and ΔGNG. | ΔDPA ρ=−0.67 **p=0.050** *; ΔGNG null → DPA-specific |
| **E** | **Causal analog of D**: Δ accuracy vs Δ depth (**laser ON−OFF**, Expert), 1×2 DPA & GNG, 7 laser mice (● Jaws inhibit / ▲ ChR excite). Full method: `laser_onoff.md`. | ΔGNG ρ=−0.90 **p=0.006** *; ΔDPA null (mirror of D) |

**Read D↔E together:** same Δdepth↔Δaccuracy relationship, learning (D) vs acute perturbation (E).
In D the **DPA** column is significant; in E the **GNG** column is — consistent with the choice
code being the lick axis (the causal lick-axis perturbation tracks Go/NoGo; the learning correlate
tracks DPA).

**D and E share an identical style** so they read as a matched pair: same figsize (9×3.7) @300 dpi,
per-animal tab10 colors (same key as `plot_scatter_perf.py`), group markers ● Jaws / ▲ ChR (D also
has ■ ACC for its 2 extra mice, n=9 vs E's n=7), thin axis lines, the same
`all (n=N): r=… p=…  ρ=… p=…` stat line with a Spearman-driven `*`/`n.s.`, and a per-mouse legend on
the right. `plot_scatter_perf.py --dpa-panel` achieves this by applying E's rcParams via a local
`rc_context` (so the script's other, larger-font figures are untouched); its x-label drops the
"(Exp−Naive)" tag to avoid the two labels overlapping (the contrast is in the panel title).

## 3. The canonical CCGD tensor (build once, ~19 min)

All four panels read the canonical (laser-off) tensor `log_generalizing_overlaps_none_l1_ratio_0.0`
from `data/overlaps/`. Build it (reuses `data/pca/X_all_nan_.pkl`; DUM = **generalizing**, scaler
`none`, l1_ratio 0.0, `null_type=None`, correct=False, contexts=all, both stages, all targets):

```bash
cd /home/leon/dual/overlaps
/home/leon/mambaforge/envs/dual/bin/python run_overlaps.py --scaler none --no-raw
#  -> data/overlaps/{X,labels}_log_generalizing_overlaps_none_l1_ratio_0.0.pkl  (~1.9 GB)
```

The tensor is `X_single[trial, {proba, decision_fn}, train_time(84), test_time(84)]`; panels use the
decision function (`[:,1]`). "Depth" = per-mouse baseline-std-normalized choice-code decision
function, averaged over the **train** bins of the read-out axis, then over the **test-time**
late-delay window `BINS_LATE = arange(27,54)`.

Read-out axes (train bins): `trainLD_TEST` = bins **45–59** (`bins_LD` 45–53 ⊕ `bins_TEST` 54–59) —
the locked axis. Others: `trainDELAY` 18–53, `trainLD` 45–53, `trainTEST` 54–59.

## 4. Build the four panels

Each panel script writes its own PNG+SVG under `figures/overlaps/…`; the assembler picks them up.

```bash
cd /home/leon/dual/overlaps

# A — 1-D codes (regenerates the FULL epoch sweep: ~4 min, many epochs × stages × variants)
/home/leon/mambaforge/envs/dual/bin/python fig_overlaps_codes_1d.py
#   -> figures/overlaps/codes1d/ld_test/png/overlaps_codes1d_grandmean_expert.png

# B — no-lick push, focused DPA Naive|Expert (all laser-off trials)
/home/leon/mambaforge/envs/dual/bin/python plot_traj2d.py --all --dpa-only
#   -> figures/overlaps/traj2d/all/png/{DUM}_trainLD_TEST_dpaonly.png

# C — well deepening + maximal-LMM stars (ld_test axis, all laser-off trials)
/home/leon/mambaforge/envs/dual/bin/python exp_nolick_push_stats.py ld_test all
#   -> figures/overlaps/nolick_push/png/{DUM}_nolick_push_paired_ld_test_all.png

# D — Δdepth ↔ Δperf (Expert−Naive), 1×2 DPA & GNG specificity panel (styled to match E)
/home/leon/mambaforge/envs/dual/bin/python plot_scatter_perf.py --dpa-panel
#   -> figures/overlaps/scatter_perf/trainLD_TEST/png/{DUM}_trainLD_TEST_dpa_panel.png

# E — laser ON−OFF causal analog (Expert, ld_test). Needs the _laser tensor first
#     (see docs/overlaps/laser_onoff.md §4: run_overlaps.py --with-laser --targets choice)
/home/leon/mambaforge/envs/dual/bin/python plot_scatter_laser.py expert ld_test
#   -> figures/overlaps/scatter_laser/png/{DUM}_laser_targets_choice_onoff_ld_test_expert.png
```
(`DUM` = `log_generalizing_overlaps_none_l1_ratio_0.0`. Scripts B and D loop over all TRAIN_EPOCHS,
so one run produces every axis variant; A regenerates the whole codes sweep; C takes
`<axis> <correct|all>` positional tokens.)

## 5. Assemble the figure

```bash
cd /home/leon/dual/overlaps
/home/leon/mambaforge/envs/dual/bin/python fig_overlaps_main.py            # default = ld_test (LOCKED)
#   -> figures/overlaps/main/{png,svg}/fig_overlaps_main.png
```

`fig_overlaps_main.py` is a **layout proof** — it stacks the already-rendered panel PNGs into one
lettered figure and prints the panel→source map. **Final publication assembly = vector-edit the
per-panel SVG twins** listed in its output. The `AXES` dict maps each variant to
`(codes_epoch, push_ax, deepset, scatter_ax)`. **Panel E** is appended only on axes with a laser
twin (`LASER_AX = {ld_test, ld, delay}`, Expert stage via `LASER_MODE`); on `test`/`mixed`/`choice…`
it is skipped (the assembler drops any missing panel), so those variants stay 4-panel.

Axis variants (flag → filename tag):

| flag | panels-on | file suffix |
|---|---|---|
| *(none)* | `ld_test` — **locked** (A valid, D sig, C strong trend) | *(none)* |
| `--mixed` | C on `trainDELAY` (p=0.024) + D on `trainTEST` (ρ=−0.67) | `_mixed` |
| `--delay` | all `trainDELAY` | `_trainDELAY` |
| `--test` | all `trainTEST` | `_trainTEST` |
| `--ld` | all `trainLD` | `_trainLD` |
| `--choice` / `--test_choice` / `--ld_test_choice` | choice-inclusive axes | `_trainCHOICE` etc. |

## 6. Why `trainLD_TEST` is the locked axis

**No single axis makes both C (deepening) and D (scatter) formally significant:**
- **C** (Naive→Expert deepening) is significant only on the full **`trainDELAY`** window
  (maximal-LMM p=0.024); on `trainLD_TEST` it is a strong **trend** (dz=−0.53, 8/9 mice, p=0.098).
- **D** (depth↔perf) is significant on `trainLD`/`trainTEST`/`trainLD_TEST` (ρ≈−0.67–0.72,
  p≈0.03–0.05) but **null on `trainDELAY`** (ρ=−0.17).
- **A**'s test code is only valid when the read-out window spans the test epoch (rules out
  delay-/ld-only).

`trainLD_TEST` is the best **coherent single axis**: A valid, D significant, C a strong trend. The
`--mixed` variant (C on delay, D on test) is the only one with **both** C and D formally
significant, but it mixes axes across panels. Report `trainLD_TEST` as the main figure and cite
`--mixed` / `--delay` for robustness.

## 7. Stats summary (n = 9 mice)

- **Deepening (C).** Pooled (A&B) late-delay choice-code deepening Δ ≈ −0.5 to −1.0 BLσ, dz ≈
  −0.5 to −0.64, 7–8/9 mice. Significant under the **correctly-specified maximal LMM**
  (`depth ~ expert + C(sample) + (1+expert+C(sample)|mouse)`, Barr et al. 2013): **p=0.024** on
  `trainDELAY`; p=0.098 on `trainLD_TEST`. Conservative n=9 mouse-mean test → trend (Wilcoxon
  p≈0.074). **Strength = cross-method convergence** (dPCA + overlaps + raw ΔF/F), not a single p.
  The "A strong / B weak" asymmetry is **trainTEST-axis-specific** — on the delay axis A and B
  deepen comparably; do NOT headline the asymmetry.
- **Depth↔performance (D).** ΔDPA vs Δdepth ρ=−0.67, p=0.050 (*); ΔGNG null → DPA-specific. Fragile
  individual-difference correlation (borderline; null on the dPCA side) — report as suggestive.

## 8. Caveats

- **n = 9 mice.** Treat every p near 0.05 as suggestive; lean on cross-method convergence.
- Panel **C** deepening is a **population** effect; not significant per-sample-class or per-mouse.
- Panel **D** is a fragile individual-difference correlation (borderline, axis-dependent).
- Depth is a projection through the learned choice axis, read at late delay; the sample code is
  invariant across LD→test, which is what makes the combined `trainLD_TEST` axis clean.

## 9. Files

- Assembler: `overlaps/fig_overlaps_main.py`
- Panels: `fig_overlaps_codes_1d.py` (A), `plot_traj2d.py --all --dpa-only` (B),
  `exp_nolick_push_stats.py <axis> all` (C), `plot_scatter_perf.py --dpa-panel` (D),
  `plot_scatter_laser.py expert ld_test` (E — needs the `_laser` tensor, see `laser_onoff.md`)
- LMM detail: `exp_nolick_push_lmm.py`
- Tensor: `data/overlaps/{X,labels}_log_generalizing_overlaps_none_l1_ratio_0.0.pkl` (gitignored —
  rebuild §3)
- Output: `overlaps/figures/overlaps/main/{png,svg}/fig_overlaps_main[_TAG].{png,svg}`
- Caption: `figure_captions.org` ("Figure X (Overlaps / CCGD)")
- Sibling docs: `overview.md` (arc + caveats), `routines.md` (run commands),
  `laser_onoff.md` (the causal analog).
