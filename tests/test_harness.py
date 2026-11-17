"""End-to-end harness test with stand-in models (no GPU, PyTorch or PyPOTS).

The twin and the three deep imputers are replaced by light fakes, so the test
checks the plumbing: corruption, all nine repair paths, Rubin pooling, the
B-term ablation, the results store and resume behaviour.
"""

import sys
import types

import numpy as np
import pytest

from repair4twin.config import Config
from repair4twin.imputers import IMPUTERS, MULTIPLE_IMPUTERS
from repair4twin.store import ResultStore


class FakePointModel:
    def impute(self, data):
        x = data["X"].copy()
        x[np.isnan(x)] = 0.5
        return x


class FakeCSDI:
    def predict(self, data, n_sampling_times):
        rng = np.random.default_rng(0)
        x = data["X"]
        draws = [np.where(np.isnan(x), rng.random(x.shape), x) for _ in range(n_sampling_times)]
        return {"imputation": np.stack(draws, axis=1)}


def fake_ensemble_predict(members, x, device=None, batch=4096):  # noqa: ARG001
    mean = 100.0 * x.mean(axis=(1, 2))
    return mean, np.full_like(mean, 25.0)


@pytest.fixture
def harness(monkeypatch):
    fake = types.ModuleType("repair4twin.twin")
    fake.ensemble_predict = fake_ensemble_predict
    monkeypatch.setitem(sys.modules, "repair4twin.twin", fake)
    from repair4twin import harness as h

    return h


def test_evaluate_block_and_resume(harness, tmp_path):
    cfg = Config.quick(out_dir=tmp_path)
    store = ResultStore(tmp_path)
    rng = np.random.default_rng(1)
    x_te = rng.random((12, 30, 14)).astype(np.float32)
    y_te = rng.uniform(0, 60, 12).astype(np.float32)
    models = {"saits": FakePointModel(), "usgan": FakePointModel(), "csdi": FakeCSDI()}

    rows = harness.evaluate_block(
        "FD001", "MNAR", 0.3, 0, x_te, y_te, None, models, 1.6, cfg, store
    )
    assert [r["imputer"] for r in rows] == IMPUTERS
    for r in rows:
        has_b = r["imputer"] in MULTIPLE_IMPUTERS
        assert np.isnan(r["regret_risk_noB"]) != has_b
        assert r["regret"] >= 0 and r["regret_risk"] >= 0
        assert 0 <= r["coverage"] <= 1
    clean = next(r for r in rows if r["imputer"] == "oracle_clean")
    assert np.isnan(clean["imp_rmse"])

    assert not store.block_done("FD001", "MNAR", 0.3, 0, IMPUTERS)
    store.append(rows)
    store.append(rows)  # duplicates are dropped
    assert store.block_done("FD001", "MNAR", 0.3, 0, IMPUTERS)
    assert len(store.load()) == len(IMPUTERS)
    arch = np.load(store.preds_path("FD001", "MNAR", 0.3, 0))
    assert {"y", "csdi_var_noB", "mice_var_noB"} <= set(arch.files)
