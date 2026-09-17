# End-to-End ML Pipeline with Production Deployment

A complete ML lifecycle for a customer-churn classifier: data
generation, tracked training (MLflow), a validated REST API
(FastAPI), containerization (Docker), CI (GitHub Actions), and
post-deployment drift monitoring — the full loop from notebook-grade
model to something actually servable in production.

## Why this exists

A model in a notebook has no business value until it's reliably
servable, tested, and monitored in production. This project
demonstrates every layer of that lifecycle on a realistic (if
synthetic) churn-prediction task, rather than stopping at
`model.fit()`.

## Architecture

```
 make_dataset.py ──▶ synthetic churn data (2000 rows, realistic signal)
        │
        ▼
   train.py      ──▶ RandomForest (class-balanced), tracked in MLflow
        │            (params, metrics, model artifact)
        │            saves models/churn_model.joblib
        │            saves models/training_feature_summary.csv
        ▼
   serve.py      ──▶ FastAPI: /health, /predict
        │            Pydantic input validation, loads model once at startup
        ▼
   Dockerfile    ──▶ containerized service, port 8000
        │
        ▼
 .github/workflows/ci.yml ──▶ generate data → train → test → build image
                               on every push/PR

   monitoring.py ──▶ KS-test drift check: incoming request features
                      vs. training distribution, per feature
```

## Features

- **Tracked training** — every run's params, metrics, and model
  artifact logged to MLflow for reproducibility and comparison across
  experiments.
- **Class-imbalance-aware** — churn is a minority class (~12% in the
  synthetic data); the model uses `class_weight="balanced"` and is
  evaluated on F1/ROC-AUC, not just accuracy, which would be
  misleadingly high on an imbalanced target.
- **Validated API** — Pydantic schemas reject out-of-range input
  (e.g. negative tenure) with a 422 before it ever reaches the model.
- **Containerized** — a `Dockerfile` builds a self-contained serving
  image.
- **CI on every push** — GitHub Actions regenerates data, retrains,
  runs the full test suite, and builds the Docker image, so a broken
  change is caught before merge.
- **Drift monitoring** — `monitoring.py` runs a Kolmogorov-Smirnov
  test per feature comparing incoming production data to the training
  distribution, flagging drift before it silently degrades accuracy.

## Setup

```bash
git clone <this-repo>
cd ml-pipeline-deployment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the dataset
python -m src.make_dataset

# 2. Train (tracked in MLflow — view with `mlflow ui`)
python -m src.train

# 3. Serve
uvicorn src.serve:app --reload
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 12, "monthly_charges": 65.0, "support_tickets_last_90d": 2, "num_products": 3, "avg_login_frequency_per_week": 3.5}'

# 4. Run tests
pytest tests/ -v

# 5. Build and run the container
docker build -t churn-api .
docker run -p 8000:8000 churn-api

# 6. Check for drift against a batch of recent production requests
python -m src.monitoring --incoming-data data/recent_requests.csv
```

## Results

Measured by actually running this pipeline end-to-end (RandomForest,
200 estimators, max_depth=8, class-balanced, 20% held-out test split,
2000 synthetic rows):

| Metric | Value |
|---|---|
| Accuracy | 0.79 |
| F1 (churn class) | 0.30 |
| ROC-AUC | 0.72 |

All 8 tests pass, including live API validation via FastAPI's
`TestClient` (health check, valid prediction, and a rejected
malformed request).

> F1 is intentionally the headline metric here, not accuracy: with
> ~12% churn prevalence, a model that always predicts "no churn"
> would score ~88% accuracy while catching zero actual churners.
> Class-balancing trades a bit of ROC-AUC for meaningfully better
> recall on the minority class — the tradeoff a real churn-prevention
> team would actually want.

## Project structure

```
ml-pipeline-deployment/
├── src/
│   ├── make_dataset.py    # synthetic churn data generator
│   ├── train.py            # MLflow-tracked training
│   ├── serve.py             # FastAPI serving layer
│   └── monitoring.py        # KS-test drift detection
├── tests/
│   └── test_pipeline.py     # dataset, drift, and live API tests
├── .github/workflows/ci.yml # test + build on every push
├── Dockerfile
└── requirements.txt
```

## Design decisions worth noting

- **Why MLflow over just printing metrics?** Once you have more than
  a couple of training runs, "which hyperparameters gave the best
  F1?" becomes unanswerable from memory. MLflow makes every run
  comparable and the winning model's provenance traceable.
- **Why explicit pickle serialization in `log_model`?** MLflow's
  newer default (`skops`) refuses to serialize tree-based models
  like RandomForest without an explicit trust declaration — a real
  compatibility issue this project hit and fixed, not a
  hypothetical one.
- **Why `class_weight="balanced"` instead of leaving it default?**
  Verified directly: the unweighted model had an F1 of ~0.04 despite
  87% accuracy — it was barely predicting the minority class at all.
  Balancing raised F1 to 0.30 at a small accuracy cost, which is the
  right tradeoff for a churn-prevention use case where missing an
  actual churner is expensive.
- **Why KS-test for drift instead of just comparing means?** A
  feature's mean can stay stable while its shape changes (e.g. a
  distribution that becomes bimodal), which a KS test catches and a
  simple mean comparison would miss.

## Possible extensions

- Add automatic redeployment when a new training run beats the
  currently-served model on validation metrics
- Wire `monitoring.py` into a scheduled job that runs against real
  request logs and alerts on drift
- Add a `/predict/batch` endpoint for scoring multiple customers per
  request
- Swap RandomForest for a gradient-boosted model (XGBoost/LightGBM)
  and compare

## License

MIT
