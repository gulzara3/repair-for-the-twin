"""Statistical analysis of the results table (``results/all_results.csv``).

Every headline number in the paper comes from a function in this module.
None of them needs a GPU or the raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

from repair4twin.imputers import REAL_IMPUTERS

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def _subset(df: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    return df[df.dataset == dataset] if dataset else df


# ------------------------------------------------------- ranking divergence
def two_layer_taus(df: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau between imputation RMSE and each downstream metric.

    One row per (subset, mechanism, rate) cell, computed on seed-averaged
    means of the seven real repair methods.
    """
    out = []
    real = df[df.imputer.isin(REAL_IMPUTERS)]
    for (ds, mech, rate), g in real.groupby(["dataset", "mech", "rate"]):
        a = g.groupby("imputer")[["imp_rmse", "rul_rmse", "phm08", "regret", "regret_risk"]].mean()
        row = {"dataset": ds, "mech": mech, "rate": rate}
        for col in ("rul_rmse", "phm08", "regret", "regret_risk"):
            row[f"tau_{col}"] = kendalltau(a.imp_rmse, a[col])[0]
        out.append(row)
    return pd.DataFrame(out)


def divergence_test(df: pd.DataFrame) -> dict:
    """Does reconstruction error agree more with accuracy than with decisions?

    One-sided Wilcoxon signed-rank test on the per-cell gap
    ``tau(RMSE, PHM08) - tau(RMSE, point regret)`` (and the same with RUL RMSE).
    """
    t = two_layer_taus(df)
    gap_phm = t.tau_phm08 - t.tau_regret
    gap_rul = t.tau_rul_rmse - t.tau_regret
    return {
        "n_cells": len(t),
        "mean_tau_rul_rmse": t.tau_rul_rmse.mean(),
        "mean_tau_phm08": t.tau_phm08.mean(),
        "mean_tau_regret": t.tau_regret.mean(),
        "mean_tau_regret_risk": t.tau_regret_risk.mean(),
        "cells_accuracy_above_decision": int((gap_phm > 0).sum()),
        "p_phm08": wilcoxon(gap_phm, alternative="greater").pvalue,
        "cells_rul_above_decision": int((gap_rul > 0).sum()),
        "p_rul_rmse": wilcoxon(gap_rul, alternative="greater").pvalue,
    }


def ranking_divergence(df, dataset=None, n_boot=2000, rng_seed=0) -> pd.DataFrame:
    """Kendall tau (imputation RMSE vs point regret) per (mechanism, rate),
    with seed-bootstrap 95% confidence intervals (notebook Section 6c)."""
    df = _subset(df, dataset)
    real = df[df.imputer.isin(REAL_IMPUTERS)]
    rng = np.random.default_rng(rng_seed)
    out = []
    for (mech, rate), g in real.groupby(["mech", "rate"]):
        agg = g.groupby("imputer")[["imp_rmse", "regret", "phm08"]].mean()
        tau, p = kendalltau(agg.imp_rmse.rank(), agg.regret.rank())
        tau_phm, _ = kendalltau(agg.imp_rmse.rank(), agg.phm08.rank())
        rho, _ = spearmanr(agg.imp_rmse, agg.regret)
        boots, seeds = [], g.seed.unique()
        for _ in range(n_boot):
            take = rng.choice(seeds, size=len(seeds), replace=True)
            gb = pd.concat([g[g.seed == s] for s in take])
            ab = gb.groupby("imputer")[["imp_rmse", "regret"]].mean()
            boots.append(kendalltau(ab.imp_rmse.rank(), ab.regret.rank())[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(
            {
                "mech": mech,
                "rate": rate,
                "tau_rmse_vs_regret": tau,
                "tau_ci_lo": lo,
                "tau_ci_hi": hi,
                "p_value": p,
                "rho_rmse_vs_regret": rho,
                "tau_rmse_vs_phm08": tau_phm,
            }
        )
    return pd.DataFrame(out)


# ------------------------------------------------------- decision policies
def policy_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Point-rule regret divided by risk-aware regret, per subset and method."""
    keep = REAL_IMPUTERS + ["oracle_clean"]
    m = df[df.imputer.isin(keep)].groupby(["dataset", "imputer"])[["regret", "regret_risk"]].mean()
    m["ratio"] = m.regret / m.regret_risk
    return m.reset_index()


def h2_wilcoxon(df, dataset=None, a="csdi", b="saits") -> pd.DataFrame:
    """One-sided paired Wilcoxon over seeds: is risk-aware regret of ``a`` lower?"""
    df = _subset(df, dataset)
    out = []
    for (mech, rate), g in df.groupby(["mech", "rate"]):
        av = g[g.imputer == a].sort_values("seed").regret_risk.values
        bv = g[g.imputer == b].sort_values("seed").regret_risk.values
        p = (
            wilcoxon(av, bv, alternative="less").pvalue
            if len(av) >= 5 and (av != bv).any()
            else np.nan
        )
        out.append(
            {
                "mech": mech,
                "rate": rate,
                f"{a}_mean": av.mean(),
                f"{b}_mean": bv.mean(),
                "one_sided_p": p,
            }
        )
    return pd.DataFrame(out)


def seed_ci(df, dataset=None, metric="regret_risk", z=1.96) -> pd.DataFrame:
    df = _subset(df, dataset)
    g = df.groupby(["imputer", "seed"])[metric].mean().reset_index()
    agg = g.groupby("imputer")[metric].agg(["mean", "std", "count"])
    agg["ci95"] = z * agg["std"] / np.sqrt(agg["count"])
    return agg


# ------------------------------------------------------- B-term ablation
def b_term_effect(df: pd.DataFrame) -> dict:
    """How the between-imputation term B changes risk-aware regret.

    ``dB`` is regret with B minus regret without B (negative = B helps).
    The effect is related to the coverage reached without B.
    """
    mi = df[df.imputer.isin(["csdi", "mice"])].copy()
    mi["dB"] = mi.regret_risk - mi.regret_risk_noB
    cell = mi.groupby(["dataset", "imputer", "mech", "rate"])[["coverage_noB", "dB"]].mean()
    agg = mi.groupby(["dataset", "imputer"])[["coverage_noB", "dB"]].mean()
    r_agg = spearmanr(agg.coverage_noB, agg.dB)
    r_cell = spearmanr(cell.coverage_noB, cell.dB)
    terc = pd.qcut(cell.coverage_noB, 3, labels=["low", "mid", "high"])
    helps = (cell.dB < 0).groupby(terc, observed=True).mean()
    return {
        "spearman_subset_means": r_agg.statistic,
        "p_subset_means": r_agg.pvalue,
        "n_subset_means": len(agg),
        "spearman_cells": r_cell.statistic,
        "p_cells": r_cell.pvalue,
        "n_cells": len(cell),
        "share_B_helps_low_coverage": helps["low"],
        "share_B_helps_high_coverage": helps["high"],
    }


# ------------------------------------------------------- selection criterion
def repair_for_the_twin(df, dataset=None) -> dict:
    """In-sample choice of repair method: by RMSE versus by decision cost."""
    df = _subset(df, dataset)
    agg = df[df.imputer.isin(REAL_IMPUTERS)].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
    by_rmse, by_twin = agg.imp_rmse.idxmin(), agg.regret_risk.idxmin()
    c_r, c_t = agg.loc[by_rmse, "regret_risk"], agg.loc[by_twin, "regret_risk"]
    return {
        "dataset": dataset or "all",
        "selected_by_rmse": by_rmse,
        "cost_rmse_choice": c_r,
        "selected_by_twin": by_twin,
        "cost_twin_choice": c_t,
        "saving_pct": 100 * (1 - c_t / c_r),
    }


def out_of_sample_selection(df: pd.DataFrame, dataset: str, split: int = 5) -> dict:
    """Select on seeds ``< split`` and score on the remaining seeds.

    Reports the subset-level saving and the pooled per-cell saving, where a
    method is chosen separately in every (mechanism, rate) cell.
    """
    d = df[(df.dataset == dataset) & df.imputer.isin(REAL_IMPUTERS)]
    sel = d[d.seed < split].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
    ev = d[d.seed >= split].groupby("imputer").regret_risk.mean()
    pick_r, pick_t = sel.imp_rmse.idxmin(), sel.regret_risk.idxmin()
    tr = tt = 0.0
    better = worse = 0
    for _, g in d.groupby(["mech", "rate"]):
        s = g[g.seed < split].groupby("imputer")[["imp_rmse", "regret_risk"]].mean()
        e = g[g.seed >= split].groupby("imputer").regret_risk.mean()
        cr, ct = e[s.imp_rmse.idxmin()], e[s.regret_risk.idxmin()]
        tr, tt = tr + cr, tt + ct
        better, worse = better + (ct < cr), worse + (ct > cr)
    return {
        "dataset": dataset,
        "pick_by_rmse": pick_r,
        "cost_rmse_pick": ev[pick_r],
        "pick_by_decision": pick_t,
        "cost_decision_pick": ev[pick_t],
        "saving_pct": 100 * (1 - ev[pick_t] / ev[pick_r]),
        "per_cell_saving_pct": 100 * (1 - tt / tr),
        "cells_better": int(better),
        "cells_worse": int(worse),
    }


def summary(df: pd.DataFrame) -> dict:
    """The headline numbers reported in the abstract and conclusions."""
    ratios = policy_ratios(df).ratio
    return {
        "rows": len(df),
        "divergence": divergence_test(df),
        "policy_ratio_median": ratios.median(),
        "policy_ratio_iqr": (ratios.quantile(0.25), ratios.quantile(0.75)),
        "policy_ratio_range": (ratios.min(), ratios.max()),
        "b_term": b_term_effect(df),
        "selection": [out_of_sample_selection(df, ds) for ds in SUBSETS],
    }
