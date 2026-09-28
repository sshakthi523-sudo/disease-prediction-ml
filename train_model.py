"""
Disease Prediction using Machine Learning
Trains and evaluates a Random Forest model to predict breast cancer
(malignant vs benign) from the Wisconsin Diagnostic dataset.
Run:  python train_model.py
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

RANDOM_STATE = 42
for folder in ("data", "models", "reports"):
    os.makedirs(folder, exist_ok=True)

# 1. Load data --------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# In sklearn: 0 = malignant, 1 = benign. We flip so 1 = disease (malignant).
df["target"] = 1 - df["target"]
df.to_csv("data/breast_cancer.csv", index=False)

print(f"Rows: {df.shape[0]}  Features: {df.shape[1] - 1}")
print("Missing values:", int(df.isnull().sum().sum()))
print("Class balance (1 = malignant):")
print(df["target"].value_counts(normalize=True).round(3).to_string(), "\n")

X, y = df.drop(columns="target"), df["target"]

# 2. Train / test split (stratified keeps class ratio) ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# 3. Compare models with 5-fold cross-validation -------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}
cv_results = {}
print("5-fold cross-validation accuracy:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
    print(f"  {name:20s} {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4. Tune Random Forest --------------------------------------------------------
grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid={"n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]},
    cv=cv, scoring="recall", n_jobs=-1)   # recall: missing a disease is costly
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)

# 5. Evaluate on unseen test data ------------------------------------------------
pred = best.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1_score": round(f1_score(y_test, pred), 4),
    "cross_validation": cv_results,
    "best_params": grid.best_params_,
}
print("\nTest set results:")
print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 6. Charts ---------------------------------------------------------------------
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(10)
imp.plot(kind="barh", color="teal", figsize=(8, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=150); plt.close()

# 7. Save model -------------------------------------------------------------------
joblib.dump({"model": best,
             "features": list(X.columns),
             "means": X.mean().to_dict(),
             "mins": X.min().to_dict(),
             "maxs": X.max().to_dict(),
             "top_features": list(imp.index[::-1])}, "models/rf_model.joblib")
print("Saved model to models/rf_model.joblib")
