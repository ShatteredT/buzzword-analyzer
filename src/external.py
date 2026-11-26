import pandas as pd

from src.config import PROJECT_ROOT

EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_test.csv"


def load_external():
    return pd.read_csv(EXTERNAL_PATH)