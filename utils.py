import numpy as np

from scipy.special import erfc, erfcx


def log1mexp(x):
    """
    Numerically stable computation of log(1 - exp(x)).
    Accepts scalars or numpy arrays; x should be <= 0 for real-valued results.
    Returns a numpy array with the same shape as x, or a Python float for scalar input.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x <= -np.log(2.0)
    out[mask] = np.log1p(-np.exp(x[mask]))
    out[~mask] = np.log(-np.expm1(x[~mask]))
    return out.item() if out.shape == () else out


def log_h(z):
    """
    Stable log of h(z) = phi(z) + z * Phi(z), where phi and Phi are the
    standard normal PDF and CDF, respectively.

    Accepts scalars or numpy arrays and returns the same shape (or a Python
    float for scalar input).
    """
    z = np.asarray(z, dtype=float)
    c1 = 0.5 * np.log(2.0 * np.pi)  # log(sqrt(2*pi))
    c2 = 0.5 * np.log(np.pi / 2.0)  # log(sqrt(pi/2))
    eps = np.sqrt(np.finfo(float).eps)

    out = np.empty_like(z)

    # Regions
    m1 = z > -1.0
    m2 = (z <= -1.0) & (z > -1.0 / eps)
    m3 = ~(m1 | m2)

    if np.any(m1):
        # Direct formula: log(phi(z) + z * Phi(z))
        pdf = np.exp(-0.5 * z[m1] ** 2 - c1)  # phi(z)
        cdf = 0.5 * erfc(-z[m1] / np.sqrt(2.0))  # Phi(z)
        out[m1] = np.log(pdf + z[m1] * cdf)

    if np.any(m2):
        # Use erfcx and log1mexp for moderately negative z
        a = erfcx(-z[m2] / np.sqrt(2.0)) * np.abs(z[m2])  # |z| * erfcx(-z/sqrt(2))
        out[m2] = (-0.5 * z[m2] ** 2 - c1 + log1mexp(np.log(a) + c2))

    if np.any(m3):
        # Asymptotic for very negative z: log h ≈ log phi − 2 log|z|
        out[m3] = -0.5 * z[m3] ** 2 - c1 - 2.0 * np.log(np.abs(z[m3]))

    return out.item() if out.shape == () else out


def running_max(y):
    """
    Running (prefix) maximum of a 1D array-like y.
    Returns an array of the same shape as y.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("running_max expects a 1D array-like input.")
    return np.maximum.accumulate(y)
