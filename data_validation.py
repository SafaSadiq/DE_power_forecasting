"""Data validation module for DE power fair value forecasting.

Provides functions to verify SMARD API availability, check dataset
completeness, detect data quality issues, and visualize validation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

from data_loader import SMARD_SERIES


# Plausible value ranges for all API-sourced columns.
# Bounds are based on historical DE market data and physical constraints.
EXPECTED_RANGES = {
    # SMARD series
    "price_day_ahead_eur_mwh": (-1000.0, 1000.0),
    "load_forecast_mwh": (15_000.0, 95_000.0),
    "residual_load_forecast_mwh": (-100_000.0, 100_000.0),
    "wind_onshore_forecast_mwh": (0.0, 100_000.0),
    "wind_offshore_forecast_mwh": (0.0, 100_000.0),
    "solar_forecast_mwh": (0.0, 100_000.0),
    # Derived SMARD features
    "wind_total_forecast_mwh": (0.0, 100_000.0),
    "vre_forecast_mwh": (0.0, 500_000.0),
    "forecast_vre_share": (0.0, 3.0),
    "forecast_wind_share": (0.0, 2.0),
    "forecast_solar_share": (0.0, 2.0),
    # FRED fuel prices
    "gas_henryhub_usd_mmbtu": (0.0, 100.0),
    "brent_usd_bbl": (0.0, 500.0),
}


def validate_smard_series(
    series_map: dict | None = None, resolution: str = "hour"
) -> pd.DataFrame:
    """Verify that each configured SMARD series index file is reachable and valid.

    Sends a lightweight GET request to each series' index endpoint and checks
    whether the response contains timestamps.

    Args:
        series_map: Dictionary mapping series names to (filter_id, region) tuples.
            Defaults to SMARD_SERIES from data_loader.
        resolution: Temporal resolution to check ('hour' or 'quarterhour').

    Returns:
        DataFrame with columns: series, filter_id, region, status_code, ok, error.
    """
    if series_map is None:
        series_map = SMARD_SERIES

    rows = []
    for name, (filter_id, region) in series_map.items():
        url = (
            f"https://www.smard.de/app/chart_data/"
            f"{filter_id}/{region}/index_{resolution}.json"
        )
        try:
            r = requests.get(url, timeout=20)
            ok = r.status_code == 200 and "timestamps" in r.text
            rows.append({
                "series": name,
                "filter_id": filter_id,
                "region": region,
                "status_code": r.status_code,
                "ok": ok,
                "error": None,
            })
        except Exception as exc:
            rows.append({
                "series": name,
                "filter_id": filter_id,
                "region": region,
                "status_code": None,
                "ok": False,
                "error": str(exc),
            })

    return pd.DataFrame(rows)


def check_completeness(df: pd.DataFrame, expected_freq: str = "h") -> pd.DataFrame:
    """Check temporal completeness of the dataset.

    Identifies missing timestamps by comparing the actual index against
    a full date range at the expected frequency.

    Args:
        df: Dataset with a DatetimeIndex.
        expected_freq: Expected frequency string (default: 'h' for hourly).

    Returns:
        DataFrame summarizing gaps: total_expected, total_actual,
        missing_count, missing_pct, and the first/last missing timestamps.
    """
    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=expected_freq, tz=df.index.tz
    )
    missing = full_range.difference(df.index)

    return pd.DataFrame([{
        "total_expected": len(full_range),
        "total_actual": len(df),
        "missing_count": len(missing),
        "missing_pct": round(100 * len(missing) / len(full_range), 2),
        "first_missing": missing.min() if len(missing) > 0 else None,
        "last_missing": missing.max() if len(missing) > 0 else None,
    }])


def check_column_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Check data quality for each column in the dataset.

    Reports null counts, null percentage, value range, and whether
    any infinite values are present.

    Args:
        df: Dataset to check.

    Returns:
        DataFrame with quality metrics per column.
    """
    records = []
    for col in df.columns:
        series = df[col]
        n_null = series.isna().sum()
        n_inf = 0
        if series.dtype in ["float64", "float32"]:
            n_inf = int(np.isinf(series).sum())

        records.append({
            "column": col,
            "dtype": str(series.dtype),
            "null_count": int(n_null),
            "null_pct": round(100 * n_null / len(series), 2),
            "inf_count": n_inf,
            "min": series.min() if series.dtype != "object" else None,
            "max": series.max() if series.dtype != "object" else None,
        })

    return pd.DataFrame(records)


def check_value_sanity(
    df: pd.DataFrame,
    expected_ranges: dict | None = None,
) -> pd.DataFrame:
    """Check all API-sourced columns against plausible value bounds.

    For each column with a defined expected range, counts how many values
    fall outside the bounds.

    Args:
        df: Dataset to check.
        expected_ranges: Dictionary mapping column names to (min, max) tuples.
            Defaults to EXPECTED_RANGES.

    Returns:
        DataFrame with per-column sanity results: column, expected_min,
        expected_max, actual_min, actual_max, out_of_bounds_count, out_of_bounds_pct.
    """
    if expected_ranges is None:
        expected_ranges = EXPECTED_RANGES

    records = []
    for col, (low, high) in expected_ranges.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        n_total = len(series)
        if n_total == 0:
            oob_count = 0
            oob_pct = 0.0
            actual_min = actual_max = None
        else:
            oob = (series < low) | (series > high)
            oob_count = int(oob.sum())
            oob_pct = round(100 * oob_count / n_total, 2)
            actual_min = series.min()
            actual_max = series.max()

        records.append({
            "column": col,
            "expected_min": low,
            "expected_max": high,
            "actual_min": actual_min,
            "actual_max": actual_max,
            "out_of_bounds_count": oob_count,
            "out_of_bounds_pct": oob_pct,
        })

    return pd.DataFrame(records)


def feature_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a per-feature validation summary for visual display.

    For each API-sourced column, computes:
    - completeness_pct: percentage of non-null values
    - in_bounds_pct: percentage of values within expected range
    - status: 'OK', 'WARNING', or 'CRITICAL' based on thresholds

    Status thresholds:
    - CRITICAL: completeness < 80% or in_bounds < 90%
    - WARNING: completeness < 95% or in_bounds < 98%
    - OK: otherwise

    Args:
        df: The loaded dataset to validate.

    Returns:
        DataFrame indexed by feature name with completeness, correctness,
        and status columns suitable for visual display.
    """
    records = []
    n_rows = len(df)

    for col, (low, high) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue

        series = df[col]
        n_valid = series.notna().sum()
        completeness_pct = round(100 * n_valid / n_rows, 1)

        valid_values = series.dropna()
        if len(valid_values) > 0:
            in_bounds = ((valid_values >= low) & (valid_values <= high)).sum()
            in_bounds_pct = round(100 * in_bounds / len(valid_values), 1)
        else:
            in_bounds_pct = 0.0

        # Determine status
        if completeness_pct < 80 or in_bounds_pct < 90:
            status = "CRITICAL"
        elif completeness_pct < 95 or in_bounds_pct < 98:
            status = "WARNING"
        else:
            status = "OK"

        records.append({
            "feature": col,
            "completeness_pct": completeness_pct,
            "in_bounds_pct": in_bounds_pct,
            "actual_min": valid_values.min() if len(valid_values) > 0 else None,
            "actual_max": valid_values.max() if len(valid_values) > 0 else None,
            "expected_min": low,
            "expected_max": high,
            "status": status,
        })

    return pd.DataFrame(records).set_index("feature")


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run all validation checks and return a summary dictionary.

    Combines completeness, column quality, value sanity (all API columns),
    and per-feature summary into a single validation report.

    Args:
        df: The loaded dataset to validate.

    Returns:
        Dictionary with keys: 'completeness', 'column_quality',
        'value_sanity', 'feature_summary', and 'is_valid'.
    """
    completeness = check_completeness(df)
    quality = check_column_quality(df)
    value_sanity = check_value_sanity(df)
    feature_summary = feature_validation_summary(df)

    is_valid = (
        completeness["missing_pct"].iloc[0] < 5.0
        and quality["null_pct"].max() < 20.0
        and value_sanity["out_of_bounds_pct"].max() < 5.0
    )

    return {
        "completeness": completeness,
        "column_quality": quality,
        "value_sanity": value_sanity,
        "feature_summary": feature_summary,
        "is_valid": is_valid,
    }


def plot_data_quality_report(validation: dict) -> None:
    """Render a visual data quality report showing completeness and correctness per feature.

    Displays a horizontal bar chart with color-coded completeness and in-bounds
    percentages, actual value ranges, and a status badge (OK / Warning / Critical)
    for each API-sourced feature.

    Args:
        validation: Dictionary returned by validate_dataset(), must contain
            keys 'feature_summary' and 'value_sanity'.
    """
    summary = validation["feature_summary"]

    fig, ax = plt.subplots(figsize=(12, len(summary) * 0.6 + 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(summary) - 0.5)
    ax.invert_yaxis()
    ax.set_axis_off()

    status_colors = {"OK": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}

    # Header
    ax.text(0.0, -1.2, "Feature", fontsize=10, fontweight="bold", va="center")
    ax.text(4.2, -1.2, "Completeness", fontsize=10, fontweight="bold", va="center", ha="center")
    ax.text(6.5, -1.2, "In-Bounds", fontsize=10, fontweight="bold", va="center", ha="center")
    ax.text(8.5, -1.2, "Range", fontsize=10, fontweight="bold", va="center", ha="center")
    ax.text(9.7, -1.2, "Status", fontsize=10, fontweight="bold", va="center", ha="center")

    for i, (feature, row) in enumerate(summary.iterrows()):
        # Alternate row background
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color="#f8f9fa", zorder=0)

        # Feature name (shortened for readability)
        short_name = (
            feature
            .replace("_forecast_mwh", "")
            .replace("_eur_mwh", "")
            .replace("_usd_", " $")
        )
        ax.text(0.0, i, short_name, fontsize=9, va="center", family="monospace")

        # Completeness bar
        comp = row["completeness_pct"]
        bar_width = comp / 100 * 2.0
        bar_color = "#2ecc71" if comp >= 95 else "#f39c12" if comp >= 80 else "#e74c3c"
        ax.barh(i, bar_width, left=3.2, height=0.5, color=bar_color, alpha=0.7, edgecolor="white")
        ax.text(5.4, i, f"{comp:.0f}%", fontsize=8, va="center", ha="left")

        # In-bounds bar
        ib = row["in_bounds_pct"]
        ib_width = ib / 100 * 1.5
        ib_color = "#2ecc71" if ib >= 98 else "#f39c12" if ib >= 90 else "#e74c3c"
        ax.barh(i, ib_width, left=5.9, height=0.5, color=ib_color, alpha=0.7, edgecolor="white")
        ax.text(7.5, i, f"{ib:.0f}%", fontsize=8, va="center", ha="left")

        # Actual range
        if row["actual_min"] is not None:
            range_str = f"[{row['actual_min']:.0f}, {row['actual_max']:.0f}]"
        else:
            range_str = "N/A"
        ax.text(8.5, i, range_str, fontsize=7, va="center", ha="center", color="#555")

        # Status badge
        status = row["status"]
        ax.scatter(9.7, i, s=120, color=status_colors[status], zorder=3, marker="o")
        ax.text(9.7, i, status[0], fontsize=7, va="center", ha="center",
                color="white", fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="OK"),
        mpatches.Patch(color="#f39c12", label="Warning"),
        mpatches.Patch(color="#e74c3c", label="Critical"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title("Data Quality Report: Completeness & Correctness per Feature",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()

    # Print detailed value sanity table
    print("\nDetailed Value Sanity Check:")
    display(validation["value_sanity"])
