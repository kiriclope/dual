import numpy as np
from scipy.linalg import orthogonal_procrustes


def procrustes_rotation(W, W_ref):
    """
    Find orthogonal R such that W.T @ R ≈ W_ref.T (least-squares sense).

    W, W_ref : (n_comp, n_neurons)
    Returns R : (n_comp, n_comp)
    """
    if W_ref is None:
        return np.eye(W.shape[0], dtype=float)
    R, _ = orthogonal_procrustes(W.T, W_ref.T)
    return R


def apply_rotation_to_scores(Z, R):
    """
    Rotate latent scores.

    Z : (n_trials, n_time, n_comp)
    R : (n_comp, n_comp)
    Returns Z_rot same shape.
    """
    return (Z.reshape(-1, Z.shape[-1]) @ R).reshape(Z.shape)


def apply_rotation_to_loadings(W, R):
    """
    Rotate loading matrix to match a reference space.

    W : (n_comp, n_neurons)
    R : (n_comp, n_comp)
    Returns W_rot : (n_comp, n_neurons)
    """
    return R.T @ W


def compute_gain_vector(n_neurons_total, mouse_slices, mode="equal_mouse"):
    """
    Per-neuron gain vector for weighted PCA across mice.

    mode
    ----
    "equal_mouse"  : gain = 1/sqrt(N_m) per mouse block — each mouse
                     contributes equally regardless of neuron count
    "equal_neuron" : all ones (standard unweighted PCA)
    None           : all ones

    Returns g : (n_neurons_total,)
    """
    g = np.ones(n_neurons_total, dtype=np.float32)
    if mode is None or mode == "equal_neuron":
        return g
    if mode != "equal_mouse":
        raise ValueError(f"Unknown mode: {mode!r}")
    for _, sl in mouse_slices.items():
        Nm = sl.stop - sl.start
        g[sl] = 1.0 / np.sqrt(Nm)
    return g
