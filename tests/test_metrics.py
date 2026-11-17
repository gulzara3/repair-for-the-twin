import numpy as np

from repair4twin.metrics import (
    cost_from_alarm,
    decision_regret,
    phm08_score,
    regret_decomposition,
    risk_aware_regret,
)

TRUE = np.array([5.0, 15.0, 30.0, 60.0])  # two critical engines at tau = 20


def test_perfect_policy_has_zero_regret():
    assert decision_regret(TRUE, TRUE) == 0.0


def test_regret_is_2fp_plus_9fn():
    pred = np.array([25.0, 15.0, 10.0, 18.0])  # one miss, two false alarms
    d = regret_decomposition(pred <= 20, TRUE)
    assert d == {"false_alarms": 2, "missed": 1}
    assert decision_regret(pred, TRUE) == 2 * 2 + 9 * 1


def test_cost_values():
    alarm = np.array([True, False, True, False])
    assert cost_from_alarm(alarm, TRUE) == 1 + 10 + 2


def test_risk_aware_margin_catches_late_prediction():
    mean = np.array([24.0, 15.0, 30.0, 60.0])
    var = np.array([9.0, 9.0, 9.0, 9.0])
    assert decision_regret(mean, TRUE) == 9.0
    assert risk_aware_regret(mean, var, TRUE, q=2.0) == 0.0


def test_phm08_penalises_late_more():
    true = np.array([50.0])
    assert phm08_score(true + 10, true) > phm08_score(true - 10, true) > 0
