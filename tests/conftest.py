from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def results():
    return pd.read_csv(ROOT / "results" / "all_results.csv")
