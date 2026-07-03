# Laser ON−OFF causal scatter — method & reproduction

**Figure:** Δ(laser ON − laser OFF) choice-code *depth* vs Δ(ON − OFF) *behavioral accuracy*,
per mouse, for DPA and Go/NoGo. The acute-optogenetic analog of the Expert−Naive brain↔behavior
scatter (`plot_scatter_perf.py`): *does the perturbation that moves the delay-period choice-code
state also move accuracy?* (cf. Fig 7 — delay-period ACC→mPFC inhibition displaces activity along
the lick axis and impairs DPA.)

Everything runs from `/home/leon/dual/overlaps` with
`/home/leon/mambaforge/envs/dual/bin/python`.

---

## 1. Scientific setup

- **Depth** = position along the *choice code* (the lick / no-lick decision axis), read out at
  late delay. Negative = pushed toward the no-lick half.
- **Manipulation** = optogenetic laser during the trial. We compare laser-ON vs laser-OFF trials
  *within each mouse* and ask whether the ON−OFF change in depth co-varies (across mice) with the
  ON−OFF change in accuracy.
- **Two behavioral read-outs:** DPA accuracy (`performance`, on DPA trials) and Go/NoGo accuracy
  (`odr_perf`, on Dual-Go/Dual-NoGo trials). The x-axis is always the **DPA** choice-code depth;
  the GNG panel is the specificity comparison.

## 2. Data facts (verified)

- Laser-ON trials exist for **7 mice only**, in **both learning stages** (Naive & Expert, ~288
  trials each per stage):
  - **Jaws (n=5) — inhibition:** JawsM01, JawsM06, JawsM12, JawsM15, JawsM18
  - **ChR (n=2) — excitation:** ChRM04, ChRM23
  - **ACCM03 / ACCM04 have ZERO laser trials** → excluded.
- Because Jaws (inhibition) and ChR (excitation) are opposite-signed manipulations, points are
  colored per animal and marked by group (● Jaws / ▲ ChR); the regression is pooled over all 7
  with **no sign flip** (descriptive — see caveats).
- JawsM01 & JawsM18 have **exactly 0** Δaccuracy on/off (near-ceiling performers, 31/32 and 95/96
  in both conditions). This is real, confirmed against the raw `data/pca/y_all_nan_.pkl` — not a
  pipeline bug.

## 3. The key methodological gotcha

The canonical CCGD pipeline **drops laser-ON trials** at the dataloader
(`src/overlaps/data.py` hard-masks `laser==0`), so laser-ON depth **does not exist** in the
canonical tensor `X_log_generalizing_overlaps_none_l1_ratio_0.0.pkl`.

To recover it we **project laser-ON trials through the laser-OFF-trained decoders** — a held-out
projection, exactly like the cross-condition set. The decoders, CV folds and axis are unchanged;
only extra rows are added. Because ON and OFF come from the *same* run (same decoders,
`random_state=0`), the per-mouse ON−OFF difference is valid.

Pipeline changes (all additive & backward-compatible; the canonical laser-off tensor is untouched):

| file | change |
|---|---|
| `src/overlaps/data.py` | `dataloader(..., with_laser=True)` returns a **7-tuple** with an extra laser-ON held-out set (`laser==1`, same stage/context, **not** correctness-filtered — laser impairs accuracy, so correct-only would be survivor-biased). |
| `src/overlaps/ccgd.py` | `ccgd_validation` projects that set through each fold's decoder (mirrors the cross-condition block), fold-averages, and appends `laser==1` rows to `y_cv`/`dfs`. **Not** null-calibrated → consistent only when `null_type=None` (the canonical build). |
| `src/overlaps/analysis.py` | 7-tuple unpack. |
| `overlaps/run_overlaps.py` | `--with-laser` flag; writes a **separate `_laser` fileset**. |

## 4. Build the `_laser` tensor (~6 min)

Omit `--stages` so **both** Naive & Expert laser rows are projected. Reuses the existing
`data/pca/X_all_nan_.pkl` (no `--rebuild`); `contexts=all` (default) → the all-task choice axis,
matching the locked main figure.

```bash
cd /home/leon/dual/overlaps
/home/leon/mambaforge/envs/dual/bin/python run_overlaps.py \
    --scaler none --no-raw --with-laser --targets choice
# ->  data/overlaps/X_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl
#     data/overlaps/labels_...same....pkl
```

Sanity checks after building:
- `laser` column has both `{0,1}` for the 7 laser mice (0 laser-ON for ACC).
- decision function finite on `laser==1` rows.
- `stage` column has both `Naive` and `Expert`.

## 5. Make the figure

```bash
cd /home/leon/dual/overlaps
/home/leon/mambaforge/envs/dual/bin/python plot_scatter_laser.py {pooled|expert} {ld_test|ld|delay}
# default (no tokens) = expert ld_test
```

Two CLI tokens, order-independent:
- **stage mode** — `expert` (Expert only) | `pooled` (Naive + Expert).
- **read-out axis** — `ld_test` (bins 45–59, the locked main-figure axis) | `ld` (45–53) |
  `delay` (18–53).

Outputs (PNG @300 dpi + SVG with `svg.fonttype=none`):
```
figures/overlaps/scatter_laser/{png,svg}/
    log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice_onoff_{axis}_{mode}.{png,svg}
```

## 6. What the script computes

- **Depth axis:** `X[..., BINS_TRAIN, :].mean(-2)[:, 1]` — the choice-code decision function
  (`[:,1]`) averaged over the train-epoch bins, giving a per-trial value over test-time; then
  per-mouse divided by its baseline-bin (`BINS_BL`) std.
- **Depth value** = mean over `BINS_LATE = arange(27, 54)` (test-time late delay).
- **Per-mouse Δdepth** = **equal-weight A&B pooled**: average the sample-A (odor_pairs [0,1]) and
  sample-B ([2,3]) means, then ON − OFF. (Matches the main-figure deepening panel; identical to a
  trial-weighted mean here because DPA A/B counts are balanced.)
- **Δaccuracy** = mean `performance` (DPA) / `odr_perf` (GNG), ON − OFF, per mouse.
- **Stat on the figure = 7-mouse Spearman** ρ (and Pearson r shown alongside); `*` if Spearman
  p<0.05 else `n.s.`. Chosen because it is **stable** across Expert/pooled and matches the 7
  plotted dots (honest about n = animals).
- Colors: `tab10` keyed to the 9-mouse `ALL_MICE` list, **identical to `plot_scatter_perf.py`**.

## 7. Results (preliminary, n = 7 animals)

| panel | Expert | Naive + Expert (pooled) |
|---|---|---|
| **Δdepth ↔ ΔDPA** | ρ=+0.55, p=0.21 (n.s.); Pearson r≈+0.8 is the JawsM15+ChR spread | **ρ=+0.00, p=1.0** — flat |
| **Δdepth ↔ ΔGNG** | **ρ=−0.90, p=0.006** | **ρ=−0.89, p=0.007** |

- **DPA: no relationship.** The Expert Pearson r is driven by one sub-ceiling mouse (JawsM15) and
  the two ChR points; pooling Naive dilutes it to nothing.
- **GNG: robust between-animal correlation**, essentially identical Expert vs pooled. Coherent
  because the choice code *is* the lick axis, so moving it most directly affects lick-based
  Go/NoGo behavior (DPA needs the sample–test match, a different computation).
- **Pooling Naive+Expert weakens, it does not boost** — Naive adds near-chance-behavior noise.

## 8. Caveats — read before quoting a number

- **n = 7 animals.** No method changes that. Report GNG as a *between-animal* correlation,
  descriptive.
- **LMM does NOT belong on the figure.** A day-paired mixed model
  (`Δperf ~ Δdepth + mouse REs`, unit = (mouse, stage, day)) was tried to add power but is
  **unstable**: with ≤3 days/mouse and 7 clusters the maximal model does not converge and the
  random-intercept model is singular at Expert (mouse RE variance ≈ 0). The auto-fallback then
  selects different models per panel and the headline p swings from **0.0003 (OLS)** to
  **0.73 (random-intercept)** for the *same* relationship. It is printed to **stdout only**
  (`max`/`ri`/`ols` breakdown). Honest reading: the random-intercept LMM — which factors out the
  mouse — is n.s., i.e. the GNG effect is between-animal, not a within-animal day-to-day coupling.
- **Depth is a projection through the fixed laser-off axis** — it measures how far the acute
  manipulation moves the state along the *learned* code, not a re-fit axis.
- The `_laser` tensor's per-mouse baseline std is computed over all of that mouse's rows in the
  tensor; in `expert` mode this spans Expert rows, in `pooled` mode Naive+Expert — so Pearson r can
  shift slightly between modes, but the reported Spearman is invariant.

## 9. Files

- Analysis: `overlaps/plot_scatter_laser.py`
- Pipeline: `src/overlaps/data.py`, `src/overlaps/ccgd.py`, `src/overlaps/analysis.py`,
  `overlaps/run_overlaps.py` (`--with-laser`)
- Tensor: `data/overlaps/{X,labels}_log_generalizing_overlaps_none_l1_ratio_0.0_laser_targets_choice.pkl`
  (gitignored — rebuild with §4)
- Figures: `overlaps/figures/overlaps/scatter_laser/{png,svg}/`
- Sibling docs: `routines.md` (§ "Laser ON−OFF causal scatter"), memory
  `project_overlaps_laser_onoff.md`.
