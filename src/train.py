"""Train the fraud model: compare imbalance strategies, tune threshold, evaluate, save artifacts.

Run with:  python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from src.config import (
    BINARY, CATEGORICAL, CV_FOLDS, C_GRID, CYCLIC, F_BETA, FEATURES, METADATA_PATH, MIN_PRECISION,
    MODEL_PATH, MODELS_DIR, NUMERIC, RANDOM_STATE, TEST_PREDS_PATH, TEST_SIZE, TARGET,
)
from src.data_loader import load_raw, split_xy
from src.evaluate import metrics_at_threshold, tune_threshold
from src.preprocessing import build_preprocessor


def build_pipeline(strategy: str) -> ImbPipeline:
    steps = [("preprocess", build_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=RANDOM_STATE)))
    class_weight = "balanced" if strategy == "class_weight" else None
    steps.append(("clf", LogisticRegression(
        class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)))
    return ImbPipeline(steps)


def main():
    df = load_raw()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean():.2%}) | "
          f"Test: {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean():.2%})")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall",
               "precision": "precision", "f1": "f1"}
    strategies = {"class_weight": "LR + class_weight='balanced'",
                  "smote": "LR + SMOTE (train folds only)",
                  "none": "LR, no imbalance handling"}

    results, best = [], None
    for key, label in strategies.items():
        gs = GridSearchCV(build_pipeline(key), {"clf__C": C_GRID}, scoring=scoring,
                          refit="pr_auc", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        i = gs.best_index_
        row = {"strategy": key, "label": label, "best_C": gs.best_params_["clf__C"]}
        for m in scoring:
            row[f"cv_{m}_mean"] = float(gs.cv_results_[f"mean_test_{m}"][i])
            row[f"cv_{m}_std"] = float(gs.cv_results_[f"std_test_{m}"][i])
        results.append(row)
        if best is None or row["cv_pr_auc_mean"] > best[0]["cv_pr_auc_mean"]:
            best = (row, gs.best_estimator_)

    comp = pd.DataFrame(results)
    print("\nStrategy comparison (5-fold stratified CV on train; threshold 0.5 for P/R/F1):")
    print(comp[["label", "best_C", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
                "cv_precision_mean", "cv_recall_mean", "cv_f1_mean"]].round(3).to_string(index=False))

    best_row, _ = best
    print(f"\nSelected: {best_row['label']} (C={best_row['best_C']}) by CV PR-AUC")

    # Threshold tuning on out-of-fold probabilities (train only; test stays untouched).
    pipe = build_pipeline(best_row["strategy"]).set_params(clf__C=best_row["best_C"])
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = tune_threshold(y_train, oof, beta=F_BETA, min_precision=MIN_PRECISION)
    print(f"Tuned threshold (max F{F_BETA:g}, precision >= {MIN_PRECISION}): {threshold:.3f}")

    # Final fit + one-shot test evaluation.
    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    m_default = metrics_at_threshold(y_test, test_proba, 0.5)
    m_tuned = metrics_at_threshold(y_test, test_proba, threshold)
    for name, m in [("threshold=0.5", m_default), (f"tuned threshold={threshold:.3f}", m_tuned)]:
        print(f"\nTest metrics @ {name}")
        print(f"  ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | Precision {m['precision']:.3f} | "
              f"Recall {m['recall']:.3f} | F1 {m['f1']:.3f} | F2 {m['f2']:.3f}")
        print(f"  Confusion: {m['confusion_matrix']}")

    # Coefficients with feature names.
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    coef_list = sorted(({"feature": n, "coef": float(c)} for n, c in zip(names, coefs)),
                       key=lambda d: abs(d["coef"]), reverse=True)

    # UI metadata from the training data.
    feature_info = {}
    for c in NUMERIC + CYCLIC:
        s = df[c]
        feature_info[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
    for c in BINARY:
        feature_info[c] = {"mode": int(df[c].mode()[0])}
    for c in CATEGORICAL:
        feature_info[c] = {"categories": sorted(df[c].unique().tolist()), "mode": df[c].mode()[0]}
    fraud = df[df[TARGET] == 1]
    legit = df[df[TARGET] == 0]
    presets = {
        "legitimate": {**{c: float(legit[c].median()) for c in NUMERIC + CYCLIC},
                       **{c: int(legit[c].mode()[0]) for c in BINARY},
                       "merchant_category": legit["merchant_category"].mode()[0]},
        "fraud": {**{c: float(fraud[c].median()) for c in NUMERIC + CYCLIC},
                  **{c: int(fraud[c].mode()[0]) for c in BINARY},
                  "merchant_category": fraud["merchant_category"].mode()[0]},
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    pd.DataFrame({"y_true": y_test.values, "proba": test_proba}).to_csv(TEST_PREDS_PATH, index=False)
    metadata = {
        "strategy": best_row["strategy"], "strategy_label": best_row["label"], "C": best_row["best_C"],
        "threshold": threshold, "beta": F_BETA, "min_precision": MIN_PRECISION,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_fraud": int(y_train.sum()), "test_fraud": int(y_test.sum()),
        "test_metrics_default": m_default, "test_metrics_tuned": m_tuned,
        "cv_comparison": results, "coefficients": coef_list,
        "features": FEATURES, "feature_info": feature_info, "presets": presets,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {MODEL_PATH.name}, {METADATA_PATH.name}, {TEST_PREDS_PATH.name} to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
