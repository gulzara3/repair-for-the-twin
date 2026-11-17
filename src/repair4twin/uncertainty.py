"""Rubin's rules and split-conformal calibration (NumPy only)."""

from __future__ import annotations

import numpy as np


def rubin_pool(
    draw_means: np.ndarray, draw_vars: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool ``m`` imputation draws with Rubin's rules.

    ``draw_means`` and ``draw_vars`` have shape ``[m, n]``: the twin's
    predictive mean and within-variance for each imputed copy.

    Returns ``(pooled_mean, T, W_bar)`` where ``T = W_bar + (1 + 1/m) B``.
    ``W_bar`` alone is returned for the B-term ablation.
    """
    m = draw_means.shape[0]
    pooled = draw_means.mean(0)
    w_bar = draw_vars.mean(0)
    b = draw_means.var(0, ddof=1) if m > 1 else np.zeros_like(pooled)
    total = w_bar + (1.0 + 1.0 / m) * b
    return pooled, total, w_bar


def conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal multiplier: the ceil((n+1)(1-alpha))-th smallest score.

    With normalised scores ``s = |y - mu| / sigma`` on a clean calibration
    split, ``mu +/- q * sigma`` covers at least ``1 - alpha`` of exchangeable
    test points.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def interval_metrics(
    mean: np.ndarray, var: np.ndarray, true: np.ndarray, q: float
) -> tuple[float, float]:
    """Empirical coverage and mean width of ``mean +/- q * sqrt(var)``."""
    half = q * np.sqrt(np.maximum(var, 1e-9))
    lo, hi = mean - half, mean + half
    return float(np.mean((true >= lo) & (true <= hi))), float(np.mean(hi - lo))
