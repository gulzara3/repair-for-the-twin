"""Missing-data injection under Rubin's MCAR, MAR and MNAR mechanisms.

Gaps are injected as bursts of ``gap_len`` consecutive cycles, on the test
(inference) stream only. The random generator is seeded from integers alone,
so a resumed run reproduces identical masks.
"""

from __future__ import annotations

import numpy as np

# Session-stable mechanism ids. Python's hash() is salted per interpreter
# session and must not be used here.
MECH_ID = {"MCAR": 1, "MAR": 2, "MNAR": 3}


def corruption_rng(mech: str, rate: float, seed: int) -> np.random.Generator:
    """Generator for one (mechanism, rate, seed) corruption block."""
    return np.random.default_rng(100000 * MECH_ID[mech] + int(rate * 1000) * 100 + seed)


def inject_missing(
    x: np.ndarray,
    mechanism: str = "MCAR",
    rate: float = 0.2,
    gap_len: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_with_nans, mask)`` for windows of shape ``[n, t, f]``.

    * MCAR: every cycle starts a gap with probability ``rate / gap_len``.
    * MAR: the probability doubles where the *neighbouring* channel
      ``(j + 1) % f`` is above its window median, and is zero elsewhere.
    * MNAR: the same rule driven by the channel's *own* value.

    ``mask`` is True at every injected position.
    """
    rng = rng or np.random.default_rng(2026)
    xc = x.copy()
    n, t, f = xc.shape
    mask = np.zeros_like(xc, dtype=bool)
    for i in range(n):
        for j in range(f):
            if mechanism == "MCAR":
                p = np.full(t, rate)
            elif mechanism == "MAR":
                driver = xc[i, :, (j + 1) % f]
                p = rate * 2.0 * (driver > np.median(driver))
            elif mechanism == "MNAR":
                p = rate * 2.0 * (xc[i, :, j] > np.median(xc[i, :, j]))
            else:
                raise ValueError(f"unknown mechanism: {mechanism}")
            starts = np.where(rng.random(t) < p / max(gap_len, 1))[0]
            for s in starts:
                mask[i, s : min(s + gap_len, t), j] = True
    xc[mask] = np.nan
    return xc, mask
