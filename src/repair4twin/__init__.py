"""Repair for the Twin: decision-aware evaluation of sensor-data imputation.

The package is split so that the light modules (corruption, uncertainty,
metrics, stats) need only NumPy, pandas, SciPy and scikit-learn. The deep
modules (twin, imputers with PyPOTS, harness) import PyTorch and PyPOTS
lazily, so analysis of saved results works on any machine.
"""

from repair4twin.config import Config
from repair4twin.corruption import corruption_rng, inject_missing
from repair4twin.metrics import (
    cost_from_alarm,
    decision_regret,
    imputation_mae,
    imputation_rmse,
    phm08_score,
    risk_aware_regret,
    rul_rmse,
)
from repair4twin.uncertainty import conformal_quantile, interval_metrics, rubin_pool

__version__ = "1.0.0"

__all__ = [
    "Config",
    "conformal_quantile",
    "corruption_rng",
    "cost_from_alarm",
    "decision_regret",
    "imputation_mae",
    "imputation_rmse",
    "inject_missing",
    "interval_metrics",
    "phm08_score",
    "risk_aware_regret",
    "rubin_pool",
    "rul_rmse",
]
