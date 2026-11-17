import numpy as np

from repair4twin.uncertainty import conformal_quantile, interval_metrics, rubin_pool


def test_rubin_total_variance():
    means = np.array([[10.0, 20.0], [12.0, 20.0], [14.0, 20.0]])
    vars_ = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    pooled, total, w = rubin_pool(means, vars_)
    np.testing.assert_allclose(pooled, [12.0, 20.0])
    np.testing.assert_allclose(w, [1.0, 2.0])
    # B = var(10, 12, 14) = 4, T = W + (1 + 1/3) * B
    np.testing.assert_allclose(total, [1.0 + 4.0 * 4 / 3, 2.0])


def test_rubin_single_draw_has_no_b():
    pooled, total, w = rubin_pool(np.array([[5.0]]), np.array([[3.0]]))
    assert total[0] == w[0] == 3.0


def test_conformal_quantile_rank():
    scores = np.arange(1, 20, dtype=float)  # n = 19
    # ceil(20 * 0.95) = 19 -> the largest score
    assert conformal_quantile(scores, 0.05) == 19.0
    assert conformal_quantile(scores, 0.5) == 10.0


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal, test = rng.standard_normal(2000), rng.standard_normal(20000)
    q = conformal_quantile(np.abs(cal), 0.05)
    cov, _ = interval_metrics(np.zeros_like(test), np.ones_like(test), test, q)
    assert 0.93 < cov < 0.97 and abs(q - 1.96) < 0.15


def test_interval_width():
    cov, wid = interval_metrics(np.zeros(3), np.full(3, 4.0), np.array([0.0, 3.0, 5.0]), 1.5)
    assert wid == 6.0 and cov == 2 / 3
