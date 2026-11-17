"""Shape and determinism checks for the PyTorch twin (skipped without torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from repair4twin.twin import RULTwin, cosine_warmup, ensemble_predict  # noqa: E402


def test_forward_shapes():
    mu, lv = RULTwin(n_feat=14)(torch.zeros(8, 30, 14))
    assert mu.shape == lv.shape == (8,)
    assert float(lv.min()) >= -6.0 and float(lv.max()) <= 8.0


def test_ensemble_variance_includes_disagreement():
    torch.manual_seed(0)
    members = [RULTwin(n_feat=14) for _ in range(3)]
    x = np.random.default_rng(0).random((5, 30, 14)).astype(np.float32)
    mean, var = ensemble_predict(members, x, device="cpu")
    assert mean.shape == var.shape == (5,) and (var > 0).all()


def test_cosine_warmup():
    assert cosine_warmup(0, 100, 10, 1.0) == 0.0
    assert cosine_warmup(10, 100, 10, 1.0) == pytest.approx(1.0)
    assert cosine_warmup(100, 100, 10, 1.0) == pytest.approx(0.0, abs=1e-12)
