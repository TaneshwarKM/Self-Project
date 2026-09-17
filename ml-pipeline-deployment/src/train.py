"""
train.py
Trains a churn-prediction classifier, tracking every run (params,
metrics, the model artifact itself) with MLflow so training is
reproducible and comparable across experiments. Saves the winning
model to models/churn_model.joblib for the serving layer to load.
"""

from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.make_dataset import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_PATH = "models/churn_model.joblib"
MLFLOW_EXPERIMENT = "churn-prediction"


def load_data(path: str = "data/churn_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train(
    n_estimators: int = 200,
    max_depth: int = 8,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    df = load_data()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(
            {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size, "seed": seed}
        )

        # class_weight="balanced" matters here: churn is a minority class
        # (~12% of the synthetic data), and an unweighted model can get
        # high accuracy by mostly predicting "no churn" while missing
        # almost all actual churners — a classic imbalanced-classification
        # trap that F1 (not accuracy) is meant to catch.
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, class_weight="balanced"
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)
        # Use pickle serialization explicitly: mlflow's default skops format
        # refuses to (de)serialize tree-based models like RandomForest
        # without an explicit trust declaration.
        mlflow.sklearn.log_model(
            model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)

        # Also save a snapshot of the training feature distributions —
        # monitoring.py compares incoming request data against this to
        # detect drift after deployment.
        X_train.describe().to_csv("models/training_feature_summary.csv")

        return metrics


if __name__ == "__main__":
    metrics = train()
    print("Training complete.")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Model saved to {MODEL_PATH}")
