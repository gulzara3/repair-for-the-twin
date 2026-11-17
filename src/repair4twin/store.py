"""Checkpoint and result persistence, so an interrupted run resumes.

Layout under ``out_dir``::

    models/<FD00x>/      twin_0..K-1.pt, histories.json, conformal.json,
                         saits.pypots, usgan.pypots, csdi.pypots
    results/all_results.csv          one row per configuration, de-duplicated
    preds/<FD00x>_<mech>_<rate>_<seed>.npz   per-engine mean and variances
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLS = ["dataset", "mech", "rate", "seed", "imputer"]


class ResultStore:
    def __init__(self, out_dir: Path):
        self.base = Path(out_dir)
        self.models_dir = self.base / "models"
        self.results_csv = self.base / "results" / "all_results.csv"
        self.preds_dir = self.base / "preds"
        for p in (self.models_dir, self.results_csv.parent, self.preds_dir):
            p.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.results_csv) if self.results_csv.exists() else pd.DataFrame()

    def append(self, rows: list[dict]) -> pd.DataFrame:
        """Append rows, keep the latest per key, and write atomically."""
        df = pd.concat([self.load(), pd.DataFrame(rows)], ignore_index=True)
        df = df.drop_duplicates(subset=KEY_COLS, keep="last")
        tmp = self.results_csv.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(self.results_csv)
        return df

    def preds_path(self, dataset: str, mech: str, rate: float, seed: int) -> Path:
        return self.preds_dir / f"{dataset}_{mech}_{int(rate * 100)}_{seed}.npz"

    def model_path(self, dataset: str, name: str) -> Path:
        d = self.models_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def block_done(self, dataset, mech, rate, seed, imputers) -> bool:
        """True iff every imputer row and the prediction archive exist."""
        df = self.load()
        if df.empty:
            return False
        sub = df[
            (df.dataset == dataset) & (df.mech == mech) & (df.rate == rate) & (df.seed == seed)
        ]
        return (
            set(imputers) <= set(sub.imputer)
            and self.preds_path(dataset, mech, rate, seed).exists()
        )
