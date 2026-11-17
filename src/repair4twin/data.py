"""NASA C-MAPSS download and preprocessing (DQ-Twin-Bench conventions).

14 informative sensors, window length 30, piecewise RUL capped at 125,
min-max scaling for FD001/FD003 and condition-wise min-max scaling over six
k-means operating regimes for FD002/FD004.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

COL_NAMES = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
KEEP_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
FEATURE_COLS = [f"s{i}" for i in KEEP_SENSORS]
N_FEAT = len(FEATURE_COLS)
MULTI_REGIME = {"FD002", "FD004"}

MIRRORS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]
EXPECTED_FILES = [f"{s}_FD00{i}.txt" for i in range(1, 5) for s in ("train", "test")] + [
    f"RUL_FD00{i}.txt" for i in range(1, 5)
]


def _extract(blob: bytes, data_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(data_dir)
    for inner in list(data_dir.rglob("*.zip")):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(data_dir)
        inner.unlink()
    for txt in data_dir.rglob("*FD00*.txt"):
        if txt.parent != data_dir:
            txt.replace(data_dir / txt.name)


def have_cmapss(data_dir: Path) -> bool:
    return all((Path(data_dir) / f).exists() for f in EXPECTED_FILES)


def fetch_cmapss(data_dir: Path, cache_zip: Path | None = None) -> None:
    """Download C-MAPSS into ``data_dir`` unless it is already there.

    The raw text files may also be placed in ``data_dir`` by hand, for
    example after downloading them from the NASA Prognostics Data Repository.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if have_cmapss(data_dir):
        print(f"[cache] C-MAPSS found in {data_dir}")
        return
    if cache_zip and Path(cache_zip).exists():
        _extract(Path(cache_zip).read_bytes(), data_dir)
        if have_cmapss(data_dir):
            return
    last = None
    for url in MIRRORS:
        try:
            print(f"[fetch] {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "repair4twin"})
            blob = urllib.request.urlopen(req, timeout=180).read()
            print(f"[sha256] {hashlib.sha256(blob).hexdigest()} ({len(blob) / 1e6:.1f} MB)")
            if cache_zip:
                Path(cache_zip).write_bytes(blob)
            _extract(blob, data_dir)
            if have_cmapss(data_dir):
                print("[ok] FD001-FD004 files verified.")
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[warn] mirror failed: {exc}")
    raise RuntimeError(
        f"Could not download C-MAPSS ({last}). Place the files {EXPECTED_FILES} in {data_dir}."
    )


def load_subset(data_dir: Path, name: str):
    data_dir = Path(data_dir)
    kw = {"sep": r"\s+", "header": None}
    train = pd.read_csv(data_dir / f"train_{name}.txt", names=COL_NAMES, **kw)
    test = pd.read_csv(data_dir / f"test_{name}.txt", names=COL_NAMES, **kw)
    rul = pd.read_csv(data_dir / f"RUL_{name}.txt", **kw).values.ravel()
    return train, test, rul


def add_piecewise_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=rul_cap)
    return df


def condition_normalize(train: pd.DataFrame, test: pd.DataFrame, name: str, seed: int = 2026):
    train, test = train.copy(), test.copy()
    if name in MULTI_REGIME:
        km = KMeans(n_clusters=6, n_init=10, random_state=seed)
        train["regime"] = km.fit_predict(train[["op1", "op2", "op3"]])
        test["regime"] = km.predict(test[["op1", "op2", "op3"]])
        for reg in range(6):
            sc = MinMaxScaler()
            m_tr, m_te = train["regime"] == reg, test["regime"] == reg
            sc.fit(train.loc[m_tr, FEATURE_COLS])
            train.loc[m_tr, FEATURE_COLS] = sc.transform(train.loc[m_tr, FEATURE_COLS])
            if m_te.any():
                test.loc[m_te, FEATURE_COLS] = sc.transform(test.loc[m_te, FEATURE_COLS])
    else:
        sc = MinMaxScaler().fit(train[FEATURE_COLS])
        train[FEATURE_COLS] = sc.transform(train[FEATURE_COLS])
        test[FEATURE_COLS] = sc.transform(test[FEATURE_COLS])
    return train, test


def make_windows(df, is_test, rul_vec=None, window: int = 30, rul_cap: int = 125):
    """Sliding windows for training; the last window per engine for testing."""
    xs, ys = [], []
    for uid, grp in df.groupby("unit"):
        arr = grp[FEATURE_COLS].values
        if is_test:
            if len(arr) < window:
                arr = np.pad(arr, ((window - len(arr), 0), (0, 0)), "edge")
            xs.append(arr[-window:])
            ys.append(min(rul_vec[int(uid) - 1], rul_cap))
        else:
            rul = grp["RUL"].values
            for i in range(len(arr) - window + 1):
                xs.append(arr[i : i + window])
                ys.append(rul[i + window - 1])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def load_dataset(name: str, cfg) -> dict[str, np.ndarray]:
    """Preprocessed windows ``x_tr, y_tr, x_te, y_te`` for one subset."""
    tr, te, rul = load_subset(cfg.data_dir, name)
    tr = add_piecewise_rul(tr, cfg.rul_cap)
    tr, te = condition_normalize(tr, te, name, cfg.seed)
    x_tr, y_tr = make_windows(tr, False, window=cfg.window, rul_cap=cfg.rul_cap)
    x_te, y_te = make_windows(te, True, rul, window=cfg.window, rul_cap=cfg.rul_cap)
    return {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te}
