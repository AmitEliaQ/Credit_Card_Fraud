"""Streamlit dashboard: EDA, model performance, threshold explorer and live prediction.

Run with:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import MODEL_PATH, NUMERIC, TARGET, TEST_PREDS_PATH
from src.data_loader import load_raw
from src.evaluate import metrics_at_threshold
from src.predict import explain_one, load_artifacts, predict_one
from src.preprocessing import outlier_summary

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")


@st.cache_data
def get_data():
    return load_raw()


@st.cache_resource
def get_model():
    return load_artifacts()


@st.cache_data
def get_test_preds():
    return pd.read_csv(TEST_PREDS_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
    st.stop()

df = get_data()
model, meta = get_model()
preds = get_test_preds()
threshold = meta["threshold"]

st.title("🛡️ Credit Card Fraud Detection")
tab_eda, tab_perf, tab_thr, tab_pred = st.tabs(
    ["📊 Overview / EDA", "📈 Model Performance", "🎚️ Threshold Explorer", "🔍 Predict a Transaction"])

# ---------------------------------------------------------------- EDA
with tab_eda:
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraudulent", f"{int(df[TARGET].sum()):,}")
    c3.metric("Fraud rate", f"{df[TARGET].mean():.2%}")

    left, right = st.columns(2)
    by_hour = df.groupby("transaction_hour")[TARGET].mean().reset_index()
    left.plotly_chart(px.line(by_hour, x="transaction_hour", y=TARGET, markers=True,
                              title="Fraud rate by hour of day", labels={TARGET: "fraud rate"}),
                      width='stretch')
    by_cat = df.groupby("merchant_category")[TARGET].mean().reset_index()
    right.plotly_chart(px.bar(by_cat, x="merchant_category", y=TARGET,
                              title="Fraud rate by merchant category", labels={TARGET: "fraud rate"}),
                       width='stretch')

    flags = []
    for col in ["foreign_transaction", "location_mismatch"]:
        g = df.groupby(col)[TARGET].mean().reset_index()
        g["flag"] = col
        g["value"] = g[col].map({0: "No", 1: "Yes"})
        flags.append(g[["flag", "value", TARGET]])
    st.plotly_chart(px.bar(pd.concat(flags), x="flag", y=TARGET, color="value", barmode="group",
                           title="Fraud rate by risk flag", labels={TARGET: "fraud rate"}),
                    width='stretch')

    cols = st.columns(3)
    for col, name in zip(cols, ["amount", "device_trust_score", "velocity_last_24h"]):
        fig = px.histogram(df.assign(cls=df[TARGET].map({0: "Legitimate", 1: "Fraud"})), x=name,
                           color="cls", barmode="overlay", histnorm="probability density",
                           opacity=0.6, title=name)
        col.plotly_chart(fig, width='stretch')

    corr = df.drop(columns="merchant_category").corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation matrix"), width='stretch')
    st.subheader("Outlier summary")
    st.caption("The pipeline caps (winsorises) IQR outliers rather than dropping rows, because "
               "extreme values are often genuine fraud signals and fraud cases are scarce.")
    st.dataframe(outlier_summary(df), hide_index=True, width='stretch')

# ---------------------------------------------------------------- Performance
with tab_perf:
    m = meta["test_metrics_tuned"]
    d = meta["test_metrics_default"]
    st.caption(f"Model: **{meta['strategy_label']}**, C={meta['C']}. Held-out test set: "
               f"{meta['n_test']:,} transactions, {meta['test_fraud']} fraud. "
               f"Tuned threshold = {threshold:.3f} (default 0.5 shown for comparison).")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
                                         ("Recall", "recall"), ("Precision", "precision"),
                                         ("F1", "f1")]):
        delta = None if key in ("roc_auc", "pr_auc") else f"{m[key] - d[key]:+.3f} vs 0.5"
        col.metric(label, f"{m[key]:.3f}", delta)

    y, p = preds["y_true"].values, preds["proba"].values
    left, right = st.columns(2)
    fpr, tpr, _ = roc_curve(y, p)
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC"))
    roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    roc.update_layout(title=f"ROC curve (AUC {m['roc_auc']:.3f})", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    left.plotly_chart(roc, width='stretch')

    prec, rec, _ = precision_recall_curve(y, p)
    pr = go.Figure(go.Scatter(x=rec, y=prec, name="PR"))
    pr.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="tuned threshold",
                            marker=dict(size=12, color="red")))
    pr.update_layout(title=f"Precision-Recall curve (AP {m['pr_auc']:.3f})",
                     xaxis_title="Recall", yaxis_title="Precision")
    right.plotly_chart(pr, width='stretch')

    left, right = st.columns(2)
    cm = m["confusion_matrix"]
    left.plotly_chart(px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                                x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                                color_continuous_scale="Blues", title="Confusion matrix (tuned threshold)"),
                      width='stretch')
    coefs = pd.DataFrame(meta["coefficients"]).sort_values("coef")
    right.plotly_chart(px.bar(coefs, x="coef", y="feature", orientation="h",
                              title="Logistic regression coefficients (scaled features)"),
                       width='stretch')

    st.subheader("Imbalance strategy comparison (5-fold CV on train)")
    comp = pd.DataFrame(meta["cv_comparison"])
    comp["PR-AUC"] = comp.apply(lambda r: f"{r.cv_pr_auc_mean:.3f} ± {r.cv_pr_auc_std:.3f}", axis=1)
    comp["ROC-AUC"] = comp.apply(lambda r: f"{r.cv_roc_auc_mean:.3f} ± {r.cv_roc_auc_std:.3f}", axis=1)
    comp["Precision@0.5"] = comp.cv_precision_mean.round(3)
    comp["Recall@0.5"] = comp.cv_recall_mean.round(3)
    st.dataframe(comp[["label", "best_C", "PR-AUC", "ROC-AUC", "Precision@0.5", "Recall@0.5"]],
                 hide_index=True, width='stretch')
    st.caption("Differences in PR-AUC are within one standard deviation: with ~121 training frauds, "
               "the strategies rank similarly. Class weighting and SMOTE shift the default 0.5 "
               "operating point towards recall; threshold tuning achieves the same effect directly.")

    with st.expander("Why Recall and Precision matter for fraud"):
        st.markdown(
            "- **Accuracy is misleading**: labelling everything legitimate scores ~98.5% and catches no fraud.\n"
            "- **Recall** = fraction of real frauds caught. Low recall means fraud slips through and costs money.\n"
            "- **Precision** = fraction of alerts that are real fraud. Low precision means blocked "
            "legitimate customers, support cost and churn.\n"
            "- The threshold trades one for the other. **F1** balances them; **PR-AUC** summarises the "
            "trade-off across all thresholds and is not inflated by the many true negatives.")

# ---------------------------------------------------------------- Threshold explorer
with tab_thr:
    t = st.slider("Decision threshold", 0.01, 0.99, float(round(threshold, 2)), 0.01)
    mt = metrics_at_threshold(preds["y_true"], preds["proba"], t)
    cols = st.columns(4)
    cols[0].metric("Recall", f"{mt['recall']:.3f}")
    cols[1].metric("Precision", f"{mt['precision']:.3f}")
    cols[2].metric("F1", f"{mt['f1']:.3f}")
    cols[3].metric("Flagged", f"{mt['confusion_matrix']['tp'] + mt['confusion_matrix']['fp']}")
    cm = mt["confusion_matrix"]
    st.plotly_chart(px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                              x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                              color_continuous_scale="Blues"), width='stretch')
    ts = np.linspace(0.01, 0.99, 99)
    sweep = pd.DataFrame([metrics_at_threshold(preds["y_true"], preds["proba"], x) for x in ts])
    long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"])
    fig = px.line(long, x="threshold", y="value", color="variable",
                  title="Precision / Recall / F1 vs threshold (test set)")
    fig.add_vline(x=t, line_dash="dash")
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- Predict
with tab_pred:
    info = meta["feature_info"]
    st.write("Enter transaction details, or load a preset, then run the model.")
    pc1, pc2, _ = st.columns([1, 1, 4])
    if pc1.button("Typical legitimate"):
        st.session_state["vals"] = dict(meta["presets"]["legitimate"])
    if pc2.button("Typical fraud"):
        st.session_state["vals"] = dict(meta["presets"]["fraud"])
    default = {**{k: v["median"] for k, v in info.items() if "median" in v},
               **{k: v["mode"] for k, v in info.items() if "mode" in v}}
    v = {**default, **st.session_state.get("vals", {})}

    with st.form("predict_form"):
        a, b = st.columns(2)
        amount = a.number_input("Amount ($)", 0.0, 100000.0, float(round(v["amount"], 2)), 1.0)
        hour = a.slider("Transaction hour", 0, 23, int(v["transaction_hour"]))
        cats = info["merchant_category"]["categories"]
        category = a.selectbox("Merchant category", cats, index=cats.index(v["merchant_category"]))
        age = a.slider("Cardholder age", 18, 100, int(v["cardholder_age"]))
        foreign = b.radio("Foreign transaction?", ["No", "Yes"], index=int(v["foreign_transaction"]),
                          horizontal=True)
        mismatch = b.radio("Location mismatch?", ["No", "Yes"], index=int(v["location_mismatch"]),
                           horizontal=True)
        trust = b.slider("Device trust score", 0, 100, int(v["device_trust_score"]))
        velocity = b.slider("Transactions in last 24h", 0, 20, int(v["velocity_last_24h"]))
        submitted = st.form_submit_button("Run model", type="primary")

    if submitted:
        record = {"amount": amount, "transaction_hour": hour, "merchant_category": category,
                  "foreign_transaction": int(foreign == "Yes"), "location_mismatch": int(mismatch == "Yes"),
                  "device_trust_score": trust, "velocity_last_24h": velocity, "cardholder_age": age}
        proba, label = predict_one(model, record, threshold)
        if label:
            st.error(f"🚨 **FRAUD**: fraud probability {proba:.1%} (threshold {threshold:.1%})")
        else:
            st.success(f"✅ **LEGITIMATE**: fraud probability {proba:.1%} (threshold {threshold:.1%})")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100, number={"suffix": "%"},
            title={"text": "Fraud probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "crimson" if label else "seagreen"},
                   "threshold": {"line": {"color": "black", "width": 4}, "value": threshold * 100}}))
        gauge.update_layout(height=280)
        g, e = st.columns(2)
        g.plotly_chart(gauge, width='stretch')
        contrib = explain_one(model, record).head(6).sort_values("contribution")
        e.plotly_chart(px.bar(contrib, x="contribution", y="feature", orientation="h",
                              title="Top drivers (log-odds; right = towards fraud)",
                              color=contrib.contribution > 0,
                              color_discrete_map={True: "crimson", False: "seagreen"})
                       .update_layout(showlegend=False), width='stretch')
