import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import FEATURES, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_raw, split_xy
from src.preprocessing import IQRCapper, build_preprocessor


@pytest.fixture(scope="module")
def data():
    return split_xy(load_raw())


def test_preprocessor_handles_missing_values(data):
    X, _ = data
    X = X.copy()
    X.iloc[::7, 0] = np.nan
    X.iloc[::11, 3] = np.nan
    X.iloc[::13, -1] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_iqr_capper_clips_and_uses_fit_bounds():
    cap = IQRCapper(k=1.5).fit(np.arange(1, 101).reshape(-1, 1))
    out = cap.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == cap.lower_[0] and out[2, 0] == cap.upper_[0] and out[1, 0] == 50.0


def test_unseen_category_does_not_crash(data):
    X, _ = data
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["merchant_category"] = "Spaceships"
    assert pre.transform(row).shape[0] == 1


def test_split_preserves_fraud_ratio(data):
    X, y = data
    _, _, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    assert abs(ytr.mean() - y.mean()) < 0.002 and abs(yte.mean() - y.mean()) < 0.002


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_predict_one(data):
    from src.predict import load_artifacts, predict_one
    X, _ = data
    model, meta = load_artifacts()
    proba, label = predict_one(model, X.iloc[0].to_dict(), meta["threshold"])
    assert 0.0 <= proba <= 1.0 and label in (0, 1)
