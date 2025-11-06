import numpy as np

from scipy.spatial.distance import cdist


def _as_2d(x):
    """Ensure x is a 2D array (n_samples, n_features)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("Input must be 2D (n_samples, n_features) or 1D (n_features,).")
    return x


def _validate_and_scale(x, betas):
    """Validate betas and scale x by positive lengthscales betas.

    betas can be a positive scalar or a 1D array of positive values with
    length equal to x.shape[1].
    """
    x = _as_2d(x)
    betas = np.asarray(betas, dtype=float)
    if betas.ndim == 0:
        if not np.isfinite(betas) or betas <= 0:
            raise ValueError("betas (lengthscale) must be a positive finite scalar.")
        scale = betas
    elif betas.ndim == 1:
        if betas.shape[0] != x.shape[1]:
            raise ValueError(f"betas must have length equal to n_features ({x.shape[1]}).")
        if not np.all(np.isfinite(betas)) or np.any(betas <= 0):
            raise ValueError("All betas (lengthscales) must be positive and finite.")
        scale = betas
    else:
        raise ValueError("betas must be a positive scalar or 1D array of positive values.")
    return x / scale


def get_obs_cov(x, betas, marg_var):
    """
    Squared-exponential (RBF) covariance on X:
    K[i,j] = marg_var * exp(-0.5 * ||(x_i - x_j)/betas||^2)

    Parameters
    - x: array-like shape (n, d) or (d,)
    - betas: positive scalar or array-like shape (d,), treated as lengthscales
    - marg_var: positive scalar (signal variance)
    """
    if not np.isfinite(marg_var) or marg_var <= 0:
        raise ValueError("marg_var must be a positive finite scalar.")
    xs = _validate_and_scale(x, betas)
    D2 = cdist(xs, xs, metric="sqeuclidean")
    K = float(marg_var) * np.exp(-0.5 * D2)
    # Enforce symmetry for numerical robustness
    K = 0.5 * (K + K.T)
    return K


def get_cross_cov(x_new, x, betas, marg_var):
    """
    Cross-covariance between X_new and X under the RBF kernel:
    K[i,j] = marg_var * exp(-0.5 * ||(x_new_i - x_j)/betas||^2)

    Parameters
    - x_new: array-like shape (m, d) or (d,)
    - x: array-like shape (n, d) or (d,)
    - betas: positive scalar or array-like shape (d,), treated as lengthscales
    - marg_var: positive scalar (signal variance)
    """
    if not np.isfinite(marg_var) or marg_var <= 0:
        raise ValueError("marg_var must be a positive finite scalar.")
    xs_new = _validate_and_scale(x_new, betas)
    xs = _validate_and_scale(x, betas)
    D2 = cdist(xs_new, xs, metric="sqeuclidean")
    K = float(marg_var) * np.exp(-0.5 * D2)
    return K
