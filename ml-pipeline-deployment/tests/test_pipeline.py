"""
test_pipeline.py
Tests the full loop: dataset generation, training, and the FastAPI
serving layer via TestClient (no running server needed).
"""

import os

from fastapi.testclient import TestClient

from src.make_dataset import FEATURE_COLUMNS, TARGET_COLUMN, make_dataset
from src.monitoring import check_drift, check_drift_report


def test_make_dataset_has_expected_columns_and_signal():
    df = make_dataset(n_samples=500, seed=1)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert df[TARGET_COLUMN].isin([0, 1]).all()
    # Churn rate should be neither ~0% nor ~100% — i.e. the synthetic
    # generator actually produces a learnable, non-degenerate task.
    churn_rate = df[TARGET_COLUMN].mean()
    assert 0.05 < churn_rate < 0.95


def test_dataset_is_reproducible_with_same_seed():
    df1 = make_dataset(n_samples=100, seed=7)
    df2 = make_dataset(n_samples=100, seed=7)
    assert df1.equals(df2)


def test_check_drift_no_drift_on_identical_distributions():
    import pandas as pd

    sample = pd.Series(range(100))
    result = check_drift(sample, sample)
    assert bool(result["drift_detected"]) is False
    assert result["p_value"] == 1.0


def test_check_drift_flags_clearly_shifted_distribution():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    training = pd.Series(rng.normal(loc=0, scale=1, size=500))
    shifted = pd.Series(rng.normal(loc=10, scale=1, size=500))  # far-shifted mean

    result = check_drift(training, shifted)
    assert bool(result["drift_detected"]) is True


def test_check_drift_report_flags_only_the_shifted_column():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    training_df = pd.DataFrame(
        {"stable_feature": rng.normal(0, 1, 300), "shifting_feature": rng.normal(0, 1, 300)}
    )
    incoming_df = pd.DataFrame(
        {
            "stable_feature": rng.normal(0, 1, 300),
            "shifting_feature": rng.normal(8, 1, 300),  # shifted
        }
    )

    report = check_drift_report(training_df, incoming_df)
    assert "shifting_feature" in report["drifted_features"]
    assert "stable_feature" not in report["drifted_features"]


def test_api_health_endpoint():
    from src.serve import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


def test_api_predict_endpoint_returns_valid_response():
    if not os.path.exists("models/churn_model.joblib"):
        import pytest

        pytest.skip("Run `python -m src.train` before this test — no trained model found.")

    from src.serve import app

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "tenure_months": 12,
                "monthly_charges": 65.0,
                "support_tickets_last_90d": 2,
                "num_products": 3,
                "avg_login_frequency_per_week": 3.5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["churn_prediction"] in (0, 1)
        assert 0.0 <= body["churn_probability"] <= 1.0


def test_api_predict_rejects_invalid_input():
    from src.serve import app

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "tenure_months": -5,  # invalid: negative
                "monthly_charges": 65.0,
                "support_tickets_last_90d": 2,
                "num_products": 3,
                "avg_login_frequency_per_week": 3.5,
            },
        )
        assert response.status_code == 422
