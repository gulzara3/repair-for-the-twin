"""Experiment configuration.

The defaults reproduce the paper exactly. ``Config.quick()`` gives the smoke
test used during development (FD001 only, about 25 minutes on one GPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MECHANISMS = ("MCAR", "MAR", "MNAR")


def _default_cost() -> dict[str, float]:
    # Planned maintenance, corrective maintenance (missed failure), false alarm.
    return {"C_pm": 1.0, "C_cm": 10.0, "C_fa": 2.0}


@dataclass(frozen=True)
class Config:
    """All settings of one experimental run."""

    # reproducibility
    seed: int = 2026

    # scope
    subsets: tuple[str, ...] = SUBSETS
    anchor: str = "FD001"
    mechanisms: tuple[str, ...] = MECHANISMS
    rates: tuple[float, ...] = (0.1, 0.3, 0.5)
    seeds: tuple[int, ...] = tuple(range(10))
    gap_len: int = 3

    # data
    window: int = 30
    rul_cap: int = 125

    # models
    ensemble_k: int = 5
    twin_epochs: int = 60
    imputer_epochs: int = 40  # SAITS
    usgan_epochs: int = 150
    csdi_epochs: int = 60
    mi_draws_csdi: int = 20
    mi_draws_mice: int = 5

    # uncertainty and decision
    nominal: float = 0.95
    tau: float = 20.0
    cost: dict[str, float] = field(default_factory=_default_cost)

    # paths
    data_dir: Path = Path("data/cmapss")
    out_dir: Path = Path("runs/default")

    @classmethod
    def quick(cls, **overrides) -> Config:
        """Smoke-test configuration (FD001, two seeds, one rate)."""
        base = cls(
            subsets=("FD001",),
            mechanisms=("MCAR", "MNAR"),
            rates=(0.3,),
            seeds=(0, 1),
            ensemble_k=3,
            twin_epochs=15,
            imputer_epochs=8,
            usgan_epochs=10,
            csdi_epochs=10,
            mi_draws_csdi=5,
            mi_draws_mice=3,
            out_dir=Path("runs/quick"),
        )
        return replace(base, **overrides)

    @property
    def alpha(self) -> float:
        return 1.0 - self.nominal
