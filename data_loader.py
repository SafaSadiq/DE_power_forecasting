"""Data loading module for DE power fair value forecasting.

Provides functions to fetch hourly time series from the SMARD API
(German electricity market data platform) and fuel price proxies from FRED.
"""

import time

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests


TZ = "Europe/Berlin"

SMARD_SERIES = {
    "price_day_ahead_eur_mwh": ("4169", "DE-LU"),
    "load_forecast_mwh": ("411", "DE"),
    "residual_load_forecast_mwh": ("4362", "DE"),
    "wind_onshore_forecast_mwh": ("123", "DE"),
    "wind_offshore_forecast_mwh": ("3791", "DE"),
    "solar_forecast_mwh": ("125", "DE"),
}


def to_ms(ts: str) -> int:
    """Convert a timezone-aware timestamp string to milliseconds since epoch.

    Args:
        ts: Date string (e.g. '2026-01-01') that will be localized to TZ.

    Returns:
        Milliseconds since Unix epoch.
    """
    return int(pd.Timestamp(ts, tz=TZ).timestamp() * 1000)


def smard_get_series(
    filter_id: str, region: str, start: str, end: str, resolution: str = "hour"
) -> pd.Series:
    """Download a single SMARD time series as a timezone-aware pandas Series.

    Args:
        filter_id: SMARD numeric series identifier (e.g. '4169' for day-ahead price).
        region: SMARD region code (e.g. 'DE', 'DE-LU').
        start: Start date string (inclusive).
        end: End date string (inclusive).
        resolution: Temporal resolution ('hour' or 'quarterhour').

    Returns:
        A pandas Series indexed by timezone-aware datetime with float values.

    Raises:
        ValueError: If the series is not found or returns no timestamps.
    """
    base = "https://www.smard.de/app/chart_data"
    idx_url = f"{base}/{filter_id}/{region}/index_{resolution}.json"

    idx = requests.get(idx_url, timeout=30)
    if idx.status_code == 404:
        raise ValueError(
            f"SMARD series not found: filter_id={filter_id}, "
            f"region={region}, resolution={resolution}"
        )
    idx.raise_for_status()

    timestamps = idx.json().get("timestamps", [])
    if not timestamps:
        raise ValueError(
            f"No timestamps returned by SMARD for "
            f"filter_id={filter_id}, region={region}"
        )

    start_ms, end_ms = to_ms(start), to_ms(end)
    frames = []

    for block_ts in timestamps:
        if block_ts > end_ms:
            break
        if block_ts < start_ms - 14 * 24 * 3600 * 1000:
            continue

        url = (
            f"{base}/{filter_id}/{region}/"
            f"{filter_id}_{region}_{resolution}_{block_ts}.json"
        )
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()

        series = r.json().get("series", [])
        if series:
            frames.append(pd.DataFrame(series, columns=["ts_ms", "value"]))

        time.sleep(0.05)

    if not frames:
        return pd.Series(dtype="float64")

    df = pd.concat(frames).drop_duplicates("ts_ms")
    df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(TZ)

    s = df.set_index("datetime")["value"].sort_index()
    s = s.loc[pd.Timestamp(start, tz=TZ) : pd.Timestamp(end, tz=TZ)]
    return s.astype(float)


def load_fuel_prices(start: str, end: str) -> pd.DataFrame:
    """Load and hourly-forward-fill fuel proxy prices from FRED.

    Fetches Henry Hub natural gas and Brent crude oil daily prices,
    then resamples to hourly frequency using forward-fill.

    Args:
        start: Start date string.
        end: End date string.

    Returns:
        DataFrame with columns 'gas_henryhub_usd_mmbtu' and 'brent_usd_bbl',
        indexed by hourly timezone-aware datetime.
    """
    gas = web.DataReader("DHHNGSP", "fred", start, end)
    oil = web.DataReader("DCOILBRENTEU", "fred", start, end)

    df = pd.concat([gas, oil], axis=1)
    df.columns = ["gas_henryhub_usd_mmbtu", "brent_usd_bbl"]
    df.index = df.index.tz_localize("UTC").tz_convert(TZ)
    return df.resample("h").ffill()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by trimming edge NaNs and interpolating interior gaps.

    Rows at the start or end of the dataset that contain any NaN values are
    dropped (since they cannot be reliably interpolated without surrounding
    data). Interior NaN values are filled using time-based linear interpolation.

    Args:
        df: Raw dataset with a DatetimeIndex.

    Returns:
        Cleaned DataFrame with edge NaNs removed and interior NaNs interpolated.
    """
    # Identify rows with any NaN
    has_nan = df.isna().any(axis=1)

    # Find first row without NaN (trim leading NaN rows)
    first_valid = has_nan[~has_nan].index.min()
    # Find last row without NaN (trim trailing NaN rows)
    last_valid = has_nan[~has_nan].index.max()

    if first_valid is None or last_valid is None:
        return df

    # Trim edges
    df = df.loc[first_valid:last_valid].copy()

    # Interpolate interior gaps (time-weighted linear interpolation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")

    return df


def build_dataset(
    start: str | None = None,
    end: str | None = None,
    include_fuels: bool = True,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """Build a combined hourly forecast dataset from SMARD and fuel prices.

    Downloads all configured SMARD series, computes derived features
    (total wind, VRE, shares), and optionally joins fuel price data.
    Returns raw data — call clean_dataset() separately after validation.

    Args:
        start: Start date string. Defaults to lookback_days before today.
        end: End date string. Defaults to today.
        include_fuels: Whether to include FRED fuel price data.
        lookback_days: Number of days to look back if start is not provided.

    Returns:
        DataFrame indexed by hourly timezone-aware datetime with all features.
    """
    if start is None:
        start = (
            pd.Timestamp.now(tz=TZ) - pd.Timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
    if end is None:
        end = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d")

    data = {}
    for name, (filter_id, region) in SMARD_SERIES.items():
        data[name] = smard_get_series(filter_id, region, start, end)

    df = pd.concat(data, axis=1).sort_index()

    # Derived features
    df["wind_total_forecast_mwh"] = (
        df["wind_onshore_forecast_mwh"] + df["wind_offshore_forecast_mwh"]
    )
    df["vre_forecast_mwh"] = df["wind_total_forecast_mwh"] + df["solar_forecast_mwh"]
    df["forecast_vre_share"] = df["vre_forecast_mwh"] / df["load_forecast_mwh"]
    df["forecast_wind_share"] = df["wind_total_forecast_mwh"] / df["load_forecast_mwh"]
    df["forecast_solar_share"] = df["solar_forecast_mwh"] / df["load_forecast_mwh"]

    # Calendar features
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    if include_fuels:
        fuels = load_fuel_prices(start, end)
        df = df.join(fuels, how="left")

    return df
