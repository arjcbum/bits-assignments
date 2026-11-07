"""
Train five classifiers for BITS Pilani AIML ML Assignment 2.
Dataset: Wisconsin Diagnostic Breast Cancer (569 rows, 30 features).
Run: python train_models.py
"""
from pathlib import Path
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "test_data.csv")
# test_data.csv is a held-out evaluation sample. For reproducible training,
# reconstruct the full public dataset from scikit-learn's bundled copy.
from sklearn.datasets import load_breast_cancer
full = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "diagnosis"})
X, y = full.drop(columns="diagnosis"), full["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=42))]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
    "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=7))]),
    "gaussian_naive_bayes": Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB())]),
    "svm": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
}
(ROOT / "model").mkdir(exist_ok=True)
rows = []
for key, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    joblib.dump(model, ROOT / "model" / f"{key}.joblib")
    rows.append({
        "Model": key, "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "outputs" / "model_comparison.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

metadata = {
    "dataset": "Wisconsin Diagnostic Breast Cancer",
    "total_samples": len(full),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "number_of_features": X.shape[1],
    "target_column": "diagnosis",
    "test_size": 0.25,
    "random_state": 42,
    "models": list(models.keys())
}

output_dir = ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = output_dir / "experiment_metadata.json"

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print(f"\nExperiment metadata saved: {metadata_path}")