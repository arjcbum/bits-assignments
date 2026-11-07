import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
TARGET = "diagnosis"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
st.set_page_config(page_title="Classifier Lab | AIML Assignment 2", page_icon="🧪", layout="wide")
st.title("Classifier Lab")
st.caption("BITS Pilani · M.Tech AIML · Machine Learning Assignment 2")
st.markdown("Evaluate five pre-trained classifiers on a labelled CSV test set. The app **does not retrain** models on uploaded data.")

with st.sidebar:
    st.header("Evaluation setup")
    selected = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption("CSV must include all 30 feature columns and the target column `diagnosis`.")
    if st.button("Use bundled sample", use_container_width=True):
        st.session_state["use_sample"] = True

@st.cache_resource
def load_model(filename):
    return joblib.load(ROOT / "model" / filename)

@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "test_data.csv")

if uploaded is not None:
    try:
        eval_df = pd.read_csv(uploaded)
        source_label = "Uploaded test data"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif st.session_state.get("use_sample", True):
    eval_df = load_sample()
    source_label = "Bundled held-out test sample"
else:
    st.info("Upload a labelled test CSV or select **Use bundled sample**.")
    st.stop()

if TARGET not in eval_df.columns:
    st.error("The uploaded CSV must contain the target column `diagnosis`.")
    st.stop()

model = load_model(MODEL_FILES[selected])
expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
X = eval_df.drop(columns=[TARGET])
y = eval_df[TARGET]
if expected is not None:
    missing = sorted(set(expected) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected))
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    X = X[expected]
try:
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    auc = roc_auc_score(y, score)
except Exception as exc:
    st.error(f"Evaluation failed. Check column names, values and target labels. Details: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": auc,
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}
st.subheader(f"{selected} · {source_label}")
st.caption(f"{len(eval_df):,} labelled rows evaluated")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(set(y))))
    ax.set_yticks(range(len(set(y))))
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    st.code(classification_report(y, pred, zero_division=0), language="text")

st.markdown("#### Preview")
st.dataframe(eval_df.head(12), use_container_width=True)

st.markdown("#### Compare all five saved models")
comparison = []
for model_name, filename in MODEL_FILES.items():
    m = load_model(filename)
    p = m.predict(X)
    s = m.predict_proba(X)[:, 1]
    comparison.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y, p),
        "AUC": roc_auc_score(y, s),
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
        "MCC": matthews_corrcoef(y, p),
    })
st.dataframe(pd.DataFrame(comparison).set_index("Model").style.format("{:.4f}"), use_container_width=True)
st.download_button("Download evaluation results", pd.DataFrame(comparison).to_csv(index=False), "evaluation_results.csv", "text/csv")
