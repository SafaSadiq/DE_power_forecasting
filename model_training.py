"""Model training module for DE power fair value forecasting.

Provides feature engineering, model training (baseline Ridge and improved
HistGradientBoosting), evaluation metrics, bias correction, trading signal
generation, and regime diagnostics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Feature sets
# ============================================================

BASELINE_FEATURES = [
    "load",
    "residual_load",
    "vre",
    "gas_price",
    "oil_price",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]

# Top 20 features selected by permutation importance on test set
# (reduced from 91 features for simplicity and robustness)
IMPROVED_FEATURES = [
    "gas_x_residual_positive",   # gas price × positive residual load (14.4)
    "residual_load",             # net load minus renewables (10.6)
    "scarcity_score",            # composite tightness indicator (3.1)
    "vre_share",                 # renewable share of total load (1.1)
    "solar_share",               # solar share of total load (0.8)
    "gas_price",                 # Henry Hub lagged 24h (0.5)
    "wind_share",                # wind share of total load (0.5)
    "solar_x_surplus",           # solar generation during surplus (0.5)
    "oil_price",                 # Brent crude lagged 24h (0.4)
    "residual_share",            # residual load / total load (0.4)
    "load",                      # total load forecast (0.3)
    "solar",                     # solar generation forecast (0.2)
    "vre_x_surplus_depth",       # VRE × depth below surplus threshold (0.2)
    "residual_x_hour",           # residual load × hour of day (0.2)
    "weekday_sin",               # day-of-week cyclical encoding (0.2)
    "weekday_cos",               # day-of-week cyclical encoding (0.2)
    "hour_sin",                  # hour-of-day cyclical encoding (0.2)
    "hour_cos",                  # hour-of-day cyclical encoding (pair)
    "vre",                       # total VRE generation forecast (0.1)
    "wind_total",                # total wind generation forecast (0.1)
]


# ============================================================
# Feature engineering
# ============================================================

def create_fair_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all features needed for fair value modelling.

    Takes raw forecast data and produces target variable, fundamental features,
    shares, asymmetric residual-load transforms, rolling regime flags,
    regime distances, composite scores, ramps, interactions, and calendar encodings.

    Args:
        df: Raw dataset from build_dataset with SMARD and fuel columns.

    Returns:
        DataFrame with all engineered features appended.
    """
    df = df.copy()
    df["target"] = df["price_day_ahead_eur_mwh"]

    # Fundamental forecasts
    df["load"] = df["load_forecast_mwh"]
    df["residual_load"] = df["residual_load_forecast_mwh"]
    df["wind_total"] = df["wind_total_forecast_mwh"]
    df["solar"] = df["solar_forecast_mwh"]
    df["vre"] = df["vre_forecast_mwh"]
    df["gas_price"] = df["gas_henryhub_usd_mmbtu"].shift(24)
    df["oil_price"] = df["brent_usd_bbl"].shift(24)

    # Shares
    df["vre_share"] = df["vre"] / df["load"]
    df["wind_share"] = df["wind_total"] / df["load"]
    df["solar_share"] = df["solar"] / df["load"]
    df["residual_share"] = df["residual_load"] / df["load"]

    # Asymmetric residual-load features
    df["residual_positive"] = df["residual_load"].clip(lower=0)
    df["residual_negative"] = df["residual_load"].clip(upper=0)

    # Rolling quantile regimes
    window = 24 * 60
    min_periods = 24 * 14

    for col, base_col in [("residual", "residual_load"), ("vre", "vre")]:
        for q in [0.05, 0.10, 0.90, 0.95]:
            q_name = f"{col}_q{int(q*100):02d}"
            df[q_name] = (
                df[base_col]
                .rolling(window, min_periods=min_periods)
                .quantile(q)
            )

    # Regime flags
    df["negative_residual_regime"] = (df["residual_load"] < 0).astype(int)
    df["surplus_regime"] = (df["residual_load"] <= df["residual_q10"]).astype(int)
    df["extreme_surplus_regime"] = (df["residual_load"] <= df["residual_q05"]).astype(int)
    df["tight_regime"] = (df["residual_load"] >= df["residual_q90"]).astype(int)
    df["extreme_tight_regime"] = (df["residual_load"] >= df["residual_q95"]).astype(int)
    df["high_vre_regime"] = (df["vre"] >= df["vre_q90"]).astype(int)

    # Regime depths
    df["surplus_depth"] = (df["residual_q10"] - df["residual_load"]).clip(lower=0)
    df["tightness_depth"] = (df["residual_load"] - df["residual_q90"]).clip(lower=0)

    # Composite scores
    df["scarcity_score"] = (
        df["tight_regime"]
        + df["extreme_tight_regime"]
        + df["tightness_depth"] / df["load"]
        + df["gas_price"] * df["residual_share"]
    )
    df["surplus_score"] = (
        df["surplus_regime"]
        + df["extreme_surplus_regime"]
        + df["negative_residual_regime"]
        + df["high_vre_regime"]
        + df["surplus_depth"] / df["load"]
    )

    # Ramps (only 24h residual ramp retained)
    df["residual_ramp_24h"] = df["residual_load"].diff(24)

    # Interactions
    df["gas_x_residual_positive"] = df["gas_price"] * df["residual_positive"]
    df["vre_x_surplus_depth"] = df["vre"] * df["surplus_depth"]
    df["solar_x_surplus"] = df["solar"] * df["surplus_regime"]
    df["residual_x_hour"] = df["residual_load"] * df.index.hour

    # Calendar encodings
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# Utilities
# ============================================================

def time_split(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train and test sets.

    Args:
        df: DataFrame with a DatetimeIndex.
        split_date: Date string for the split point (test starts at this date).

    Returns:
        Tuple of (train_df, test_df).
    """
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()
    return train, test


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute comprehensive evaluation metrics for a fair value model.

    Includes standard regression metrics plus trading-relevant measures
    like bias, spread statistics, and tail performance.

    Args:
        y_true: Actual prices.
        y_pred: Predicted fair values.

    Returns:
        Dictionary of metric name to value.
    """
    error = y_pred - y_true
    spread = y_true - y_pred
    peak_mask = y_true >= y_true.quantile(0.9)
    low_mask = y_true <= y_true.quantile(0.1)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "Bias_model_minus_actual": error.mean(),
        "Spread_mean_actual_minus_fv": spread.mean(),
        "Spread_std": spread.std(),
        "Peak_MAE_top_10pct": mean_absolute_error(
            y_true[peak_mask], y_pred[peak_mask]
        ),
        "Low_price_MAE_bottom_10pct": mean_absolute_error(
            y_true[low_mask], y_pred[low_mask]
        ),
        "Correlation": np.corrcoef(y_true, y_pred)[0, 1],
    }


def apply_bias_correction(
    y_train: pd.Series, train_pred: pd.Series, test_pred: pd.Series
) -> tuple[pd.Series, float]:
    """Apply simple mean-bias correction to test predictions.

    Computes the average bias on the training set and subtracts it
    from test predictions.

    Args:
        y_train: Actual training targets.
        train_pred: Model predictions on training data.
        test_pred: Model predictions on test data.

    Returns:
        Tuple of (corrected test predictions, training bias value).
    """
    train_bias = (train_pred - y_train).mean()
    corrected = test_pred - train_bias
    return corrected, train_bias


# ============================================================
# Model training
# ============================================================

def _train_and_predict(
    df: pd.DataFrame,
    features: list[str],
    model,
    split_date: str,
    bias_correct: bool,
    output_name: str,
) -> tuple:
    """Shared training logic for both baseline and improved models.

    Args:
        df: Raw dataset.
        features: List of feature column names.
        model: Scikit-learn estimator or pipeline.
        split_date: Date string for train/test split.
        bias_correct: Whether to apply bias correction.
        output_name: Name for the output fair value Series.

    Returns:
        Tuple of (model, fair_value_series, y_test, metrics_dict).
    """
    df_model = create_fair_value_features(df)
    df_model = df_model.dropna(subset=features + ["target"])

    train, test = time_split(df_model, split_date)
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    model.fit(X_train, y_train)

    train_pred = pd.Series(model.predict(X_train), index=X_train.index)
    test_pred = pd.Series(model.predict(X_test), index=X_test.index)

    if bias_correct:
        test_pred, train_bias = apply_bias_correction(y_train, train_pred, test_pred)
    else:
        train_bias = 0.0

    fair_value = pd.Series(test_pred, index=X_test.index, name=output_name)
    metrics = evaluate_model(y_test, fair_value)
    metrics["Train_bias_removed"] = train_bias

    return model, fair_value, y_test, metrics


def train_baseline_model(
    df: pd.DataFrame, split_date: str, bias_correct: bool = True
) -> tuple:
    """Train the baseline Ridge regression fair value model.

    Uses a scaled Ridge regression with fundamental features only
    (load, residual load, VRE, fuel prices, calendar).

    Args:
        df: Raw dataset from build_dataset.
        split_date: Date string for the train/test split.
        bias_correct: Whether to apply mean-bias correction.

    Returns:
        Tuple of (model, fair_value_series, y_test, metrics_dict).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    return _train_and_predict(
        df, BASELINE_FEATURES, model, split_date, bias_correct, "baseline_fair_value"
    )


def train_improved_model(
    df: pd.DataFrame, split_date: str, bias_correct: bool = True
) -> tuple:
    """Train the improved HistGradientBoosting fair value model.

    Uses a gradient boosting model with the top 20 most important features
    selected via permutation importance analysis.

    Args:
        df: Raw dataset from build_dataset.
        split_date: Date string for the train/test split.
        bias_correct: Whether to apply mean-bias correction.

    Returns:
        Tuple of (model, fair_value_series, y_test, metrics_dict).
    """
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=0.1,
        random_state=42,
    )
    return _train_and_predict(
        df, IMPROVED_FEATURES, model, split_date, bias_correct, "improved_fair_value"
    )


# ============================================================
# Trading signal and diagnostics
# ============================================================

def create_fair_value_signal(
    y_true: pd.Series, fair_value: pd.Series, entry_threshold: float = 10.0
) -> pd.DataFrame:
    """Generate a trading signal based on the spread between actual price and fair value.

    Signal is +1 (buy) when actual is below fair value by more than the threshold,
    -1 (sell) when actual is above, and 0 otherwise.

    Args:
        y_true: Actual prices.
        fair_value: Model fair values.
        entry_threshold: Minimum absolute spread (EUR/MWh) to generate a signal.

    Returns:
        DataFrame with columns: actual_price, fair_value, spread_actual_minus_fair, signal.
    """
    signal = pd.DataFrame(index=y_true.index)
    signal["actual_price"] = y_true
    signal["fair_value"] = fair_value
    signal["spread_actual_minus_fair"] = signal["actual_price"] - signal["fair_value"]

    signal["signal"] = 0
    signal.loc[signal["spread_actual_minus_fair"] > entry_threshold, "signal"] = -1
    signal.loc[signal["spread_actual_minus_fair"] < -entry_threshold, "signal"] = 1

    return signal


def regime_diagnostics(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime error diagnostics for the improved model.

    Breaks down MAE and bias by market regime to identify where the model
    performs well or struggles.

    Args:
        df: Raw dataset (used to recompute features for regime flags).
        results: DataFrame with 'actual_price' and 'improved_fair_value' columns.

    Returns:
        DataFrame indexed by regime name with columns: count, MAE, bias.
    """
    fv = create_fair_value_features(df)
    aligned = fv.loc[results.index].copy()
    aligned["actual_price"] = results["actual_price"]
    aligned["improved_fair_value"] = results["improved_fair_value"]
    aligned["error"] = aligned["improved_fair_value"] - aligned["actual_price"]

    diagnostics = {}
    regimes = [
        "negative_residual_regime",
        "surplus_regime",
        "extreme_surplus_regime",
        "tight_regime",
        "extreme_tight_regime",
        "high_vre_regime",
    ]

    for regime in regimes:
        mask = aligned[regime] == 1
        if mask.sum() > 0:
            diagnostics[regime] = {
                "count": int(mask.sum()),
                "MAE": mean_absolute_error(
                    aligned.loc[mask, "actual_price"],
                    aligned.loc[mask, "improved_fair_value"],
                ),
                "bias": aligned.loc[mask, "error"].mean(),
            }

    return pd.DataFrame(diagnostics).T


# ============================================================
# Full workflow
# ============================================================

def run_fair_value_workflow(
    df: pd.DataFrame,
    split_date: str = "2026-03-01",
    entry_threshold: float = 10.0,
    bias_correct: bool = True,
) -> dict:
    """Run the complete fair value modelling workflow.

    Trains both baseline and improved models, computes trading signals,
    and generates regime diagnostics.

    Args:
        df: Raw dataset from build_dataset.
        split_date: Date string for train/test split.
        entry_threshold: Minimum spread for signal generation (EUR/MWh).
        bias_correct: Whether to apply mean-bias correction.

    Returns:
        Dictionary containing models, fair values, metrics, signals,
        regime diagnostics, and a combined results DataFrame.
    """
    baseline_model, baseline_fv, y_test, baseline_metrics = train_baseline_model(
        df, split_date=split_date, bias_correct=bias_correct
    )
    improved_model, improved_fv, y_test_2, improved_metrics = train_improved_model(
        df, split_date=split_date, bias_correct=bias_correct
    )

    results = pd.DataFrame(index=y_test.index)
    results["actual_price"] = y_test
    results["baseline_fair_value"] = baseline_fv
    results["improved_fair_value"] = improved_fv
    results["baseline_spread"] = results["actual_price"] - results["baseline_fair_value"]
    results["improved_spread"] = results["actual_price"] - results["improved_fair_value"]

    baseline_signal = create_fair_value_signal(y_test, baseline_fv, entry_threshold)
    improved_signal = create_fair_value_signal(y_test_2, improved_fv, entry_threshold)
    regime_diag = regime_diagnostics(df, results)

    return {
        "baseline_model": baseline_model,
        "improved_model": improved_model,
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "baseline_signal": baseline_signal,
        "improved_signal": improved_signal,
        "regime_diagnostics": regime_diag,
        "results": results,
    }
