import numpy as np
import pytest

from repair4twin.corruption import corruption_rng, inject_missing


@pytest.fixture
def windows():
    return np.random.default_rng(0).random((40, 30, 14)).astype(np.float32)


def test_mask_matches_nans(windows):
    xc, mask = inject_missing(windows, "MCAR", 0.3, gap_len=3, rng=corruption_rng("MCAR", 0.3, 0))
    assert np.array_equal(np.isnan(xc), mask)
    assert np.array_equal(xc[~mask], windows[~mask])


@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_rate_is_plausible(windows, mech, rate):
    _, mask = inject_missing(windows, mech, rate, gap_len=3, rng=corruption_rng(mech, rate, 1))
    # Bursts overlap, so the realised rate sits below the nominal rate.
    assert 0.4 * rate < mask.mean() <= rate + 0.05


def test_same_seed_same_mask(windows):
    a = inject_missing(windows, "MNAR", 0.3, 3, corruption_rng("MNAR", 0.3, 7))[1]
    b = inject_missing(windows, "MNAR", 0.3, 3, corruption_rng("MNAR", 0.3, 7))[1]
    c = inject_missing(windows, "MNAR", 0.3, 3, corruption_rng("MNAR", 0.3, 8))[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_mnar_hits_high_values(windows):
    _, mask = inject_missing(windows, "MNAR", 0.3, gap_len=1, rng=corruption_rng("MNAR", 0.3, 0))
    med = np.median(windows, axis=1, keepdims=True)
    assert mask[windows <= np.broadcast_to(med, windows.shape)].mean() == 0.0


def test_unknown_mechanism(windows):
    with pytest.raises(ValueError):
        inject_missing(windows, "XYZ", 0.1)
