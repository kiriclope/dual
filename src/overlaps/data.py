import pandas as pd


def dataloader(
    X, y_df,
    target="sample", stage="Expert", context="DPA",
    correct=False, strata=False, with_laser=False,
):
    """Split trials into within-condition (train/test) and cross-condition sets.

    Parameters
    ----------
    target  : column in y_df used as the decode label
    stage   : learning stage filter ('Expert', 'Naive')
    context : task context filter; 'all' skips context filtering
    correct : if True, restrict to performance==1 (and odr_perf==1 for GNG)
    strata  : if True, return stratification vector over (odor_pair, tasks, day);
              otherwise stratify on target labels
    with_laser : if True, also return a laser-ON held-out set (laser==1, same
              stage/context, NOT correctness-filtered) to be projected through the
              laser-off-trained decoders. Empty set when False.

    Returns
    -------
    X_within, y_within, X_cross, y_cross, X_laser, y_laser, strata_vec
    """
    idx_correct = True
    if correct:
        idx_correct = (y_df.performance == 1) & (
            (y_df.tasks == "DPA") | (y_df.odr_perf == 1)
        )

    # idx_cross_context is the context mask for the cross-condition set.
    # It must be a pandas Series (not a Python bool scalar) so that ~ works correctly.
    idx_context = True
    idx_cross_context = pd.Series(False, index=y_df.index)  # empty by default

    if target in ("distractor", "odr_choice", "licks"):
        context = "Dual"
        idx_context = (y_df.tasks != "DPA")
        idx_cross_context = (y_df.tasks == "DPA")
    elif context in ("DPA", "DualGo", "DualNoGo"):
        idx_context = (y_df.tasks == context)
        idx_cross_context = (y_df.tasks != context)
    # context == 'all': idx_cross_context stays False (no cross-condition set)

    m_within = ((y_df.laser == 0) & (y_df.learning == stage)
                & idx_context & idx_correct)
    m_cross = ((y_df.laser == 0) & (y_df.learning == stage)
               & idx_cross_context & idx_correct)

    X_within = X[m_within]
    y_within = y_df.loc[m_within].reset_index(drop=True).copy()
    y_within["labels"] = y_within[target].to_numpy()

    X_cross = X[m_cross]
    y_cross = y_df.loc[m_cross].reset_index(drop=True).copy()
    y_cross["labels"] = y_cross[target].to_numpy()

    # Laser-ON held-out set: same stage/context, NOT correctness-filtered (laser
    # impairs accuracy, so correct-only would be survivor-biased). Projected — never
    # trained on. Empty when with_laser=False so downstream guards skip it.
    m_laser = ((y_df.laser == 1) & (y_df.learning == stage) & idx_context
               if with_laser else pd.Series(False, index=y_df.index))
    X_laser = X[m_laser]
    y_laser = y_df.loc[m_laser].reset_index(drop=True).copy()
    y_laser["labels"] = y_laser[target].to_numpy() if len(y_laser) else []

    print(f"X_within {X_within.shape}  y_within {y_within.shape}")
    print(f"X_cross  {X_cross.shape}  y_cross  {y_cross.shape}")
    if with_laser:
        print(f"X_laser  {X_laser.shape}  y_laser  {y_laser.shape}")

    if strata:
        strata_vec = (
            y_within["odor_pair"].astype(str) + "_"
            + y_within["tasks"].astype(str) + "_"
            + y_within["day"].astype(str)
        ).to_numpy()
    else:
        strata_vec = y_within[target].astype(str).to_numpy()

    return X_within, y_within, X_cross, y_cross, X_laser, y_laser, strata_vec
