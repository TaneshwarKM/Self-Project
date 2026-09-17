"""
make_dataset.py
Generates a synthetic customer-churn dataset with realistic-ish
structure (some features genuinely predictive, some noise) so the
pipeline has something concrete to train and monitor. Swap this for
a real dataset loader in production — the rest of the pipeline
doesn't care where the data came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_last_90d",
    "num_products",
    "avg_login_frequency_per_week",
]
TARGET_COLUMN = "churned"


def make_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.exponential(scale=24, size=n_samples).clip(0, 120)
    monthly_charges = rng.normal(loc=70, scale=25, size=n_samples).clip(10, 200)
    support_tickets = rng.poisson(lam=1.2, size=n_samples)
    num_products = rng.integers(1, 6, size=n_samples)
    login_frequency = rng.normal(loc=4, scale=2, size=n_samples).clip(0, 14)

    # Churn probability genuinely depends on tenure, tickets, and login frequency,
    # so a real model should be able to learn a meaningfully-better-than-random signal.
    churn_logit = (
        -0.04 * tenure
        + 0.35 * support_tickets
        - 0.25 * login_frequency
        - 0.15 * num_products
        + 0.01 * monthly_charges
        - 1.0
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_prob)

    return pd.DataFrame(
        {
            "tenure_months": tenure.round(1),
            "monthly_charges": monthly_charges.round(2),
            "support_tickets_last_90d": support_tickets,
            "num_products": num_products,
            "avg_login_frequency_per_week": login_frequency.round(2),
            "churned": churned,
        }
    )


if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("data/churn_dataset.csv", index=False)
    print(f"Wrote {len(df)} rows to data/churn_dataset.csv")
    print(f"Churn rate: {df['churned'].mean():.2%}")
