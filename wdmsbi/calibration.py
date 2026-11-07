"""Credible levels at which the truth enters a posterior, for the calibration test."""
import numpy as np
from scipy import stats


def marginal_levels(samples, truth):
    """Credible level of the central interval whose edge is the truth, per parameter.

    With rank r = fraction of samples below the truth, the level is |2r - 1|;
    for a calibrated posterior it is uniform on (0, 1).
    """
    rank = np.mean(np.asarray(samples) < np.asarray(truth)[None, :], axis=0)
    return np.abs(2 * rank - 1)


def joint_level(samples, truth, dims=(0, 1, 2)):
    """Credible level of the Gaussian ellipsoid (sample mean and covariance) whose edge is the truth.

    The squared Mahalanobis distance D^2 is mapped through the predictive
    Hotelling distribution, n(n-p) D^2 / [p(n+1)(n-1)] ~ F(p, n-p), which
    accounts for estimating the mean and covariance from n draws.
    """
    x = np.asarray(samples, dtype=np.float64)[:, dims]
    delta = np.asarray(truth, dtype=np.float64)[list(dims)] - x.mean(0)
    d2 = max(float(delta @ np.linalg.solve(np.cov(x, rowvar=False, ddof=1), delta)), 0.0)
    n, p = len(x), len(dims)
    return float(stats.f.cdf(n * (n - p) * d2 / (p * (n + 1) * (n - 1)), p, n - p))


def binomial_band(levels, n, confidence=0.95):
    """Pointwise interval of the empirical coverage expected from n calibrated observations."""
    a = (1 - confidence) / 2
    return stats.binom.ppf(a, n, levels) / n, stats.binom.ppf(1 - a, n, levels) / n
