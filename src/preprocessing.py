"""Preprocessing: imputation, IQR outlier capping, cyclical hour encoding, scaling, one-hot."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BINARY, CATEGORICAL, CYCLIC, NUMERIC


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorise each column to [Q1 - k*IQR, Q3 + k*IQR], with bounds learned at fit time."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Map a periodic value (e.g. hour of day) to sin/cos so 23:00 and 00:00 are neighbours."""

    def __init__(self, period: float = 24.0):
        self.period = period

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        angle = 2 * np.pi * X / self.period
        return np.hstack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None):
        names = list(input_features)
        return np.asarray([f"{n}_sin" for n in names] + [f"{n}_cos" for n in names], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap", IQRCapper(k=1.5)),
        ("scale", StandardScaler()),
    ])
    cyclic = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", CyclicalEncoder(period=24)),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("hour", cyclic, CYCLIC),
            ("bin", binary, BINARY),
            ("cat", categorical, CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def outlier_summary(df: pd.DataFrame, cols=NUMERIC, k: float = 1.5, z: float = 3.0) -> pd.DataFrame:
    """Compare IQR vs Z-score outlier counts (for EDA only; the pipeline uses IQR capping)."""
    rows = []
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_mask = (s < q1 - k * iqr) | (s > q3 + k * iqr)
        z_mask = ((s - s.mean()) / s.std()).abs() > z
        row = {"feature": c, "iqr_outliers": int(iqr_mask.sum()), "zscore_outliers": int(z_mask.sum())}
        if "is_fraud" in df:
            row["fraud_among_iqr_outliers"] = int(df.loc[iqr_mask, "is_fraud"].sum())
        rows.append(row)
    return pd.DataFrame(rows)
