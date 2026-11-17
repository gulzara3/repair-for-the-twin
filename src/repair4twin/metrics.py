"""Repair, prediction and decision metrics.

Decision regret is the maintenance cost of an alarm policy minus the cost of
the perfect-information policy. With the default costs (planned 1,
corrective 10, false alarm 2), regret equals ``2 * FP + 9 * FN``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COST = {"C_pm": 1.0, "C_cm": 10.0, "C_fa": 2.0}
DEFAULT_TAU = 20.0


def imputation_rmse(x_true: np.ndarray, x_imp: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over the injected (held-out) positions only."""
    return float(np.sqrt(np.mean((x_true[mask] - x_imp[mask]) ** 2)))


def imputation_mae(x_true: np.ndarray, x_imp: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true[mask] - x_imp[mask])))


def rul_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def phm08_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric PHM08 challenge score; late predictions cost more."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def cost_from_alarm(
    alarm: np.ndarray,
    true: np.ndarray,
    tau: float = DEFAULT_TAU,
    cost: dict[str, float] | None = None,
) -> float:
    """Fleet maintenance cost of a boolean alarm vector."""
    cost = cost or DEFAULT_COST
    crit = true <= tau
    return float(
        cost["C_pm"] * (alarm & crit).sum()
        + cost["C_cm"] * ((~alarm) & crit).sum()
        + cost["C_fa"] * (alarm & ~crit).sum()
    )


def decision_regret(
    pred: np.ndarray,
    true: np.ndarray,
    tau: float = DEFAULT_TAU,
    cost: dict[str, float] | None = None,
) -> float:
    """Regret of the point policy: alarm iff predicted RUL <= tau."""
    return cost_from_alarm(pred <= tau, true, tau, cost) - cost_from_alarm(
        true <= tau, true, tau, cost
    )


def risk_aware_regret(
    mean: np.ndarray,
    var: np.ndarray,
    true: np.ndarray,
    q: float,
    tau: float = DEFAULT_TAU,
    cost: dict[str, float] | None = None,
) -> float:
    """Regret of the risk-aware policy: alarm iff mu - q * sqrt(T) <= tau."""
    alarm = (mean - q * np.sqrt(np.maximum(var, 1e-9))) <= tau
    return cost_from_alarm(alarm, true, tau, cost) - cost_from_alarm(true <= tau, true, tau, cost)


def regret_decomposition(alarm: np.ndarray, true: np.ndarray, tau: float = DEFAULT_TAU) -> dict:
    """Counts of false alarms (FP) and missed failures (FN)."""
    crit = true <= tau
    return {"false_alarms": int((alarm & ~crit).sum()), "missed": int((~alarm & crit).sum())}
