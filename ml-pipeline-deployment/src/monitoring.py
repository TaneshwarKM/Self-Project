"""
monitoring.py
Compares incoming request feature distributions against the training
distribution using the Kolmogorov-Smirnov test, per feature, to flag
data drift — a real signal that model performance may be degrading
in production even before ground-truth labels come back to confirm it.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

DRIFT_P_VALUE_THRESHOLD = 0.05  # below this, we reject "same distribution"


def load_training_summary(path: str = "models/training_feature_summary.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def check_drift(training_samples: pd.Series, incoming_samples: pd.Series) -> dict:
    """Two-sample KS test: are these two samples from the same distribution?"""
    statistic, p_value = ks_2samp(training_samples, incoming_samples)
    return {
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": bool(p_value < DRIFT_P_VALUE_THRESHOLD),
    }


def check_drift_report(training_df: pd.DataFrame, incoming_df: pd.DataFrame) -> dict:
    """Run the KS test per feature and summarize which ones have drifted."""
    report = {}
    for column in training_df.columns:
        if column not in incoming_df.columns:
            continue
        report[column] = check_drift(training_df[column], incoming_df[column])

    drifted_features = [col for col, result in report.items() if result["drift_detected"]]
    return {
        "per_feature": report,
        "drifted_features": drifted_features,
        "any_drift_detected": len(drifted_features) > 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check incoming data for drift vs training distribution")
    parser.add_argument("--training-data", default="data/churn_dataset.csv")
    parser.add_argument("--incoming-data", required=True, help="CSV of recent production requests")
    args = parser.parse_args()

    training_df = pd.read_csv(args.training_data)
    incoming_df = pd.read_csv(args.incoming_data)

    report = check_drift_report(training_df, incoming_df)
    print("Drift check results:")
    for feature, result in report["per_feature"].items():
        flag = "DRIFT" if result["drift_detected"] else "ok"
        print(f"  [{flag}] {feature}: KS={result['ks_statistic']}, p={result['p_value']}")

    if report["any_drift_detected"]:
        print(f"\n⚠ Drift detected in: {report['drifted_features']}")
    else:
        print("\nNo significant drift detected.")
