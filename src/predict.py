"""Load the trained pipeline and score single transactions."""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return model, metadata


def predict_one(model, record: dict, threshold: float):
    """Return (fraud_probability, label) for one transaction given as {feature: value}."""
    row = pd.DataFrame([{f: record.get(f, np.nan) for f in FEATURES}])
    proba = float(model.predict_proba(row)[0, 1])
    return proba, int(proba >= threshold)


def explain_one(model, record: dict) -> pd.DataFrame:
    """Per-feature contribution to the log-odds: coefficient x transformed value."""
    row = pd.DataFrame([{f: record.get(f, np.nan) for f in FEATURES}])
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    x = pre.transform(row)[0]
    contrib = clf.coef_[0] * x
    return (
        pd.DataFrame({"feature": pre.get_feature_names_out(), "contribution": contrib})
        .assign(abs=lambda d: d.contribution.abs())
        .sort_values("abs", ascending=False)
        .drop(columns="abs")
        .reset_index(drop=True)
    )
