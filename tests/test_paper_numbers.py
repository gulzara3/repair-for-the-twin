"""Regression tests: the released results reproduce the paper's claims."""

import pytest

from repair4twin import stats


def test_results_table_is_complete(results):
    assert len(results) == 4 * 3 * 3 * 10 * 9
    assert results.groupby(["dataset", "mech", "rate", "seed"]).size().eq(9).all()


def test_two_layer_divergence(results):
    d = stats.divergence_test(results)
    assert d["n_cells"] == 36
    assert d["mean_tau_rul_rmse"] == pytest.approx(0.77, abs=0.005)
    assert d["mean_tau_regret"] == pytest.approx(0.41, abs=0.005)
    assert d["cells_accuracy_above_decision"] == 32
    assert d["p_phm08"] == pytest.approx(3.9e-7, rel=0.01)


def test_policy_ratio(results):
    r = stats.policy_ratios(results).ratio
    assert len(r) == 32
    assert r.median() == pytest.approx(2.27, abs=0.005)
    assert r.quantile(0.25) == pytest.approx(1.59, abs=0.005)
    assert r.quantile(0.75) == pytest.approx(2.70, abs=0.005)
    assert (r >= 2).sum() == 19


def test_b_term_is_conditional(results):
    b = stats.b_term_effect(results)
    assert b["spearman_subset_means"] == pytest.approx(0.98, abs=0.005)
    assert b["spearman_cells"] == pytest.approx(0.46, abs=0.005)
    assert b["share_B_helps_low_coverage"] == pytest.approx(0.46, abs=0.01)
    assert b["share_B_helps_high_coverage"] == pytest.approx(0.04, abs=0.01)


@pytest.mark.parametrize(
    ("ds", "saving", "per_cell"),
    [("FD001", 4.9, 1.6), ("FD002", 0.0, 1.4), ("FD003", 15.8, 8.7), ("FD004", 0.0, -5.2)],
)
def test_out_of_sample_selection(results, ds, saving, per_cell):
    r = stats.out_of_sample_selection(results, ds)
    assert r["saving_pct"] == pytest.approx(saving, abs=0.05)
    assert r["per_cell_saving_pct"] == pytest.approx(per_cell, abs=0.05)
