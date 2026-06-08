"""
MLflow run provider.

Connects to an MLflow tracking server, lists experiments, and fetches runs into
the canonical wide Polars DataFrame. All direct MLflow API access lives here.

mlflow/
├── __init__.py   # Package initialization
└── provider.py   # MLflowProvider (RunProvider) + cached experiment listing
"""
