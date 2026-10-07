"""Load and validate the raw credit card fraud dataset."""
import pandas as pd

from src.config import DATA_PATH, FEATURES, ID_COL, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the xlsx, validate the schema and drop the ID column."""
    df = pd.read_excel(path, engine="openpyxl")
    if df.columns[-1] != TARGET:
        raise ValueError(f"Expected last column to be '{TARGET}', got '{df.columns[-1]}'")
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df = df.drop(columns=[ID_COL], errors="ignore")
    return df[FEATURES + [TARGET]]


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].astype(int)
