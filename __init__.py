"""DE Power Fair Value Forecasting package.

Modules:
    data_loader: SMARD and FRED data fetching.
    data_validation: Data quality and completeness checks.
    model_training: Feature engineering, model training, and evaluation.
"""

from data_loader import build_dataset
from data_validation import validate_dataset, validate_smard_series
from model_training import run_fair_value_workflow

__all__ = [
    "build_dataset",
    "validate_dataset",
    "validate_smard_series",
    "run_fair_value_workflow",
]
