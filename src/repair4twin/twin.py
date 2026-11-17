"""Heteroscedastic 1D-CNN + GRU digital twin and its deep ensemble (PyTorch)."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from repair4twin.uncertainty import conformal_quantile

DTYPE = torch.bfloat16


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_determinism(seed: int = 2026) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] deterministic algorithms partial: {exc}")


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RULTwin(nn.Module):
    """1D-CNN + GRU twin with a heteroscedastic head returning (mu, log_var)."""

    def __init__(self, n_feat: int = 14, hid: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, hid, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hid, hid, 5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(hid, hid, batch_first=True, num_layers=2, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hid, hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 2)
        )

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(h)
        out = self.head(hn[-1])
        return out[:, 0], out[:, 1].clamp(-6.0, 8.0)


def gaussian_nll(mu, log_var, y):
    return 0.5 * (log_var + (y - mu) ** 2 / log_var.exp()).mean()


@torch.no_grad()
def ensemble_predict(members, x, device=None, batch: int = 4096):
    """Uniform-mixture ensemble: mean and within-variance.

    The within-variance is the mean aleatoric sigma^2 plus the epistemic
    variance of the member means.
    """
    device = device or get_device()
    mus, sig2s = [], []
    for m in members:
        m.eval()
        mu_o, s2_o = [], []
        for i in range(0, len(x), batch):
            xt = torch.tensor(x[i : i + batch]).to(device)
            with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=device == "cuda"):
                mu, lv = m(xt)
            mu_o.append(mu.float().cpu().numpy())
            s2_o.append(lv.float().exp().cpu().numpy())
        mus.append(np.concatenate(mu_o))
        sig2s.append(np.concatenate(s2_o))
    mus, sig2s = np.stack(mus), np.stack(sig2s)  # [K, n]
    var_epi = mus.var(0, ddof=1) if len(members) > 1 else 0.0
    return mus.mean(0), sig2s.mean(0) + var_epi


def conformal_multiplier(members, x_val, y_val, alpha: float = 0.05, device=None) -> float:
    """Split-conformal q on the clean validation split."""
    mu, var = ensemble_predict(members, x_val, device)
    scores = np.abs(y_val - mu) / np.sqrt(np.maximum(var, 1e-9))
    return conformal_quantile(scores, alpha)


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def make_loader(x, y, generator, bs: int = 512, shuffle: bool = True, device: str = "cpu"):
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    workers = 4 if device == "cuda" else 0
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers else None,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle,
    )


def train_member(
    x_tr,
    y_tr,
    x_val,
    y_val,
    member: int,
    ckpt: Path,
    cfg,
    device=None,
    generator=None,
    base_lr: float = 1e-3,
):
    """Train one member: MSE for the first 20% of epochs, then Gaussian NLL.

    AdamW, cosine schedule with 5% warmup, gradient clipping at 1.0, and early
    stopping on validation RMSE (patience 8). Reloads from ``ckpt`` if present.
    As in the notebook, one shuffling ``generator`` is shared by all members.
    """
    device = device or get_device()
    generator = generator or torch.Generator().manual_seed(cfg.seed)
    epochs = cfg.twin_epochs
    torch.manual_seed(cfg.seed + member)
    model = RULTwin(n_feat=x_tr.shape[-1]).to(device)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[resume] twin member {member} (val RMSE {state['best_val']:.3f})")
        return model, state["best_val"], state["history"]

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    loader = make_loader(x_tr, y_tr, generator, device=device)
    total_steps = epochs * len(loader)
    warmup = int(0.05 * total_steps)
    mse_epochs = max(1, int(0.2 * epochs))
    mse = nn.MSELoss()

    best_val, best_state, bad, step, history = float("inf"), None, 0, 0, []
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = cosine_warmup(step, total_steps, warmup, base_lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=device == "cuda"):
                mu, lv = model(xb)
                loss = mse(mu, yb) if ep < mse_epochs else gaussian_nll(mu, lv, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        mu_v, _ = ensemble_predict([model], x_val, device)
        vrmse = float(np.sqrt(np.mean((mu_v - y_val) ** 2)))
        history.append(vrmse)
        if vrmse < best_val:
            best_val, best_state, bad = vrmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 8 and ep >= mse_epochs + 2:
                break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "best_val": best_val, "history": history}, ckpt)
    print(f"[ckpt] twin member {member} saved (val RMSE {best_val:.3f})")
    return model, best_val, history


def get_twin(
    store, dataset: str, x_tr, y_tr, cfg, device=None, generator=None, val_frac: float = 0.1
):
    """Train or load the K-member ensemble and its conformal multiplier."""
    device = device or get_device()
    generator = generator or torch.Generator().manual_seed(cfg.seed)
    idx = np.random.default_rng(cfg.seed).permutation(len(x_tr))
    cut = int(len(x_tr) * (1 - val_frac))
    tr_i, val_i = idx[:cut], idx[cut:]
    members, histories = [], []
    for s in range(cfg.ensemble_k):
        m, _, h = train_member(
            x_tr[tr_i],
            y_tr[tr_i],
            x_tr[val_i],
            y_tr[val_i],
            member=s,
            ckpt=store.model_path(dataset, f"twin_{s}.pt"),
            cfg=cfg,
            device=device,
            generator=generator,
        )
        members.append(m)
        histories.append(h)
    store.model_path(dataset, "histories.json").write_text(json.dumps(histories))

    cpath = store.model_path(dataset, "conformal.json")
    if cpath.exists():
        q = json.loads(cpath.read_text())["q"]
        print(f"[resume] conformal q = {q:.3f}")
    else:
        q = conformal_multiplier(members, x_tr[val_i], y_tr[val_i], cfg.alpha, device)
        cpath.write_text(json.dumps({"q": q}))
        print(f"[conformal] q = {q:.3f} (Gaussian value would be 1.96)")
    return members, histories, q
