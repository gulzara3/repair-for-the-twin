"""The nine-method repair spectrum.

Classical imputers need scikit-learn only. SAITS, US-GAN and CSDI come from
PyPOTS 1.5 and are imported on first use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from repair4twin.data import N_FEAT

IMPUTERS = [
    "no_repair",
    "mean",
    "ffill",
    "knn",
    "mice",
    "usgan",
    "saits",
    "csdi",
    "oracle_clean",
]
REAL_IMPUTERS = [i for i in IMPUTERS if i not in ("no_repair", "oracle_clean")]
MULTIPLE_IMPUTERS = ("mice", "csdi")


def _flatten(x):
    n, t, f = x.shape
    return x.reshape(n * t, f).copy(), (n, t, f)


def _unflatten(x2d, shape):
    return x2d.reshape(*shape).astype(np.float32)


def impute_mean(xc: np.ndarray) -> np.ndarray:
    flat, sh = _flatten(xc)
    col_mean = np.nanmean(flat, axis=0)
    idx = np.where(np.isnan(flat))
    flat[idx] = np.take(col_mean, idx[1])
    return _unflatten(flat, sh)


def impute_ffill(xc: np.ndarray) -> np.ndarray:
    out = xc.copy()
    for i in range(out.shape[0]):
        out[i] = pd.DataFrame(out[i]).ffill().bfill().values
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def impute_knn(xc: np.ndarray, k: int = 5) -> np.ndarray:
    flat, sh = _flatten(xc)
    return _unflatten(KNNImputer(n_neighbors=k).fit_transform(flat), sh)


def impute_mice_multi(xc: np.ndarray, m: int = 5, seed: int = 2026) -> np.ndarray:
    """``m`` posterior draws, shape ``[n, m, t, f]``."""
    flat, sh = _flatten(xc)
    draws = []
    for d in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + d)
        draws.append(_unflatten(imp.fit_transform(flat), sh))
    return np.stack(draws, axis=1)


# ------------------------------------------------------------ PyPOTS models
def _fit_or_load(model, ckpt: Path, x_tr, label: str):
    if ckpt.exists():
        try:
            model.load(str(ckpt))
            print(f"[resume] {label} loaded from {ckpt.name}")
            return model
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {label} load failed ({exc}); retraining.")
    model.fit({"X": x_tr})
    try:
        model.save(str(ckpt), overwrite=True)
    except TypeError:  # older PyPOTS signature
        model.save(str(ckpt))
    print(f"[ckpt] {label} saved -> {ckpt.name}")
    return model


def get_saits(ckpt: Path, x_tr, cfg, device):
    from pypots.imputation import SAITS

    m = SAITS(
        n_steps=cfg.window,
        n_features=N_FEAT,
        n_layers=2,
        d_model=256,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ffn=128,
        dropout=0.1,
        epochs=cfg.imputer_epochs,
        patience=5,
        batch_size=256,
        num_workers=2,
        device=device,
    )
    return _fit_or_load(m, ckpt, x_tr, "SAITS")


def get_usgan(ckpt: Path, x_tr, cfg, device):
    from pypots.imputation import USGAN

    m = USGAN(
        n_steps=cfg.window,
        n_features=N_FEAT,
        rnn_hidden_size=128,
        hint_rate=0.7,
        lambda_mse=1.0,
        epochs=cfg.usgan_epochs,
        patience=15,
        batch_size=256,
        num_workers=2,
        device=device,
    )
    return _fit_or_load(m, ckpt, x_tr, "US-GAN")


def get_csdi(ckpt: Path, x_tr, cfg, device):
    from pypots.imputation import CSDI

    m = CSDI(
        n_steps=cfg.window,
        n_features=N_FEAT,
        n_layers=4,
        n_heads=8,
        n_channels=64,
        d_time_embedding=64,
        d_feature_embedding=32,
        d_diffusion_embedding=128,
        n_diffusion_steps=50,
        target_strategy="random",
        schedule="quad",
        epochs=cfg.csdi_epochs,
        patience=8,
        batch_size=64,
        num_workers=2,
        device=device,
    )
    return _fit_or_load(m, ckpt, x_tr, "CSDI")


def impute_point(model, xc: np.ndarray) -> np.ndarray:
    return model.impute({"X": xc}).astype(np.float32)


def impute_csdi_multi(model, xc: np.ndarray, m: int = 20) -> np.ndarray:
    out = np.asarray(model.predict({"X": xc}, n_sampling_times=m)["imputation"], np.float32)
    assert out.ndim == 4 and out.shape[1] == m, f"unexpected {out.shape}"
    return out  # [n, m, t, f]


def repair(imp: str, xc: np.ndarray, models: dict, cfg):
    """Return ``(x_repaired, draws_or_None)`` for one imputer."""
    if imp == "no_repair":
        return np.nan_to_num(xc, nan=0.0).astype(np.float32), None
    if imp == "mean":
        return impute_mean(xc), None
    if imp == "ffill":
        return impute_ffill(xc), None
    if imp == "knn":
        return impute_knn(xc), None
    if imp in ("usgan", "saits"):
        return impute_point(models[imp], xc), None
    if imp == "mice":
        draws = impute_mice_multi(xc, m=cfg.mi_draws_mice, seed=cfg.seed)
        return draws.mean(axis=1), draws
    if imp == "csdi":
        draws = impute_csdi_multi(models["csdi"], xc, m=cfg.mi_draws_csdi)
        return np.median(draws, axis=1), draws
    raise ValueError(f"unknown imputer: {imp}")
