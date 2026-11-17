"""Resumable corrupt -> repair -> predict -> decide harness."""

from __future__ import annotations

import numpy as np

from repair4twin.corruption import corruption_rng, inject_missing
from repair4twin.imputers import IMPUTERS, get_csdi, get_saits, get_usgan, repair
from repair4twin.metrics import (
    decision_regret,
    imputation_mae,
    imputation_rmse,
    phm08_score,
    risk_aware_regret,
    rul_rmse,
)
from repair4twin.uncertainty import interval_metrics, rubin_pool


def evaluate_block(dataset, mech, rate, seed, x_te, y_te, twin, models, q, cfg, store, device=None):
    """Evaluate every imputer on one (mechanism, rate, seed) corruption block."""
    from repair4twin.twin import ensemble_predict

    rng = corruption_rng(mech, rate, seed)
    xc, mask = inject_missing(x_te, mech, rate, gap_len=cfg.gap_len, rng=rng)
    rows, arch = [], {"y": y_te, "mask_rate": np.float32(mask.mean())}
    kw = {"tau": cfg.tau, "cost": cfg.cost}
    for imp in IMPUTERS:
        if imp == "oracle_clean":
            xr, draws = x_te, None
        else:
            xr, draws = repair(imp, xc, models, cfg)

        if imp in ("no_repair", "oracle_clean"):
            i_rmse = i_mae = np.nan
        else:
            i_rmse, i_mae = imputation_rmse(x_te, xr, mask), imputation_mae(x_te, xr, mask)

        if draws is not None:  # multiple imputation: pool with Rubin's rules
            d_means, d_vars = [], []
            for k in range(draws.shape[1]):
                mu, v = ensemble_predict(twin, draws[:, k], device)
                d_means.append(mu)
                d_vars.append(v)
            mean, var, w_only = rubin_pool(np.stack(d_means), np.stack(d_vars))
        else:
            mean, var = ensemble_predict(twin, xr, device)
            w_only = None

        cov, wid = interval_metrics(mean, var, y_te, q)
        row = {
            "dataset": dataset,
            "mech": mech,
            "rate": rate,
            "seed": seed,
            "imputer": imp,
            "imp_rmse": i_rmse,
            "imp_mae": i_mae,
            "rul_rmse": rul_rmse(mean, y_te),
            "phm08": phm08_score(mean, y_te),
            "coverage": cov,
            "width": wid,
            "regret": decision_regret(mean, y_te, **kw),
            "regret_risk": risk_aware_regret(mean, var, y_te, q, **kw),
            "coverage_noB": np.nan,
            "width_noB": np.nan,
            "regret_risk_noB": np.nan,
        }
        arch[f"{imp}_mean"] = mean.astype(np.float32)
        arch[f"{imp}_var"] = var.astype(np.float32)
        if w_only is not None:  # B-term ablation: same draws, T = W_bar only
            cov0, wid0 = interval_metrics(mean, w_only, y_te, q)
            row.update(
                coverage_noB=cov0,
                width_noB=wid0,
                regret_risk_noB=risk_aware_regret(mean, w_only, y_te, q, **kw),
            )
            arch[f"{imp}_var_noB"] = w_only.astype(np.float32)
        rows.append(row)
    np.savez_compressed(store.preds_path(dataset, mech, rate, seed), **arch)
    return rows


def run_dataset(name, cfg, store, device=None, generator=None):
    """Train or load all models for one subset, then run every missing block."""
    from repair4twin.data import load_dataset
    from repair4twin.twin import get_device, get_twin

    device = device or get_device()
    d = load_dataset(name, cfg)
    print(f"{name}: train {d['x_tr'].shape}  test {d['x_te'].shape}")
    twin, hist, q = get_twin(store, name, d["x_tr"], d["y_tr"], cfg, device, generator)
    models = {
        "saits": get_saits(store.model_path(name, "saits.pypots"), d["x_tr"], cfg, device),
        "usgan": get_usgan(store.model_path(name, "usgan.pypots"), d["x_tr"], cfg, device),
        "csdi": get_csdi(store.model_path(name, "csdi.pypots"), d["x_tr"], cfg, device),
    }
    total = len(cfg.mechanisms) * len(cfg.rates) * len(cfg.seeds)
    done = 0
    for mech in cfg.mechanisms:
        for rate in cfg.rates:
            for seed in cfg.seeds:
                done += 1
                tag = f"{name} {mech} r={rate} s={seed}"
                if store.block_done(name, mech, rate, seed, IMPUTERS):
                    print(f"[skip {done}/{total}] {tag} (already saved)")
                    continue
                rows = evaluate_block(
                    name,
                    mech,
                    rate,
                    seed,
                    d["x_te"],
                    d["y_te"],
                    twin,
                    models,
                    q,
                    cfg,
                    store,
                    device,
                )
                store.append(rows)
                print(f"[done {done}/{total}] {tag}")
    return hist, q
