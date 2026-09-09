from __future__ import annotations

from math import lgamma

import numpy as np

from pertussis_global_dynamics.model2_constants import NB_DISPERSION


def negative_binomial_sample(rng: np.random.Generator, mean: np.ndarray, dispersion: float = NB_DISPERSION) -> np.ndarray:
    mu = np.clip(np.asarray(mean, dtype=float), 0.0, None)
    r = max(float(dispersion), 1e-6)
    gamma_scale = np.where(mu > 0, mu / r, 1e-12)
    latent_rate = rng.gamma(shape=r, scale=gamma_scale)
    return rng.poisson(latent_rate)


def nb_log_likelihood(observed: float, mean: float, dispersion: float = NB_DISPERSION) -> float:
    if not np.isfinite(observed) or not np.isfinite(mean):
        return np.nan
    y = max(float(observed), 0.0)
    mu = max(float(mean), 1e-9)
    r = max(float(dispersion), 1e-6)
    p = r / (r + mu)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(lgamma(y + r) - lgamma(r) - lgamma(y + 1.0) + r * np.log(p) + y * np.log1p(-p))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])
