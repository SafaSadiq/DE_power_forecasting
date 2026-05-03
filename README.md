# DE Power Day-Ahead Fair Value Forecast

A modular forecasting system for German (DE-LU) day-ahead power prices. Builds a fair value model from publicly available data and generates hourly BUY/SELL/HOLD trade signals with built-in reliability filtering.

## Overview

This project fetches real-time generation forecasts and fuel prices, engineers regime-aware features, trains machine learning models, and produces actionable trade signals — all from a single notebook entry point.

**Key capabilities:**
- Automated data ingestion from SMARD API (German electricity market) and FRED (fuel prices)
- Data quality validation with visual reporting
- Baseline (Ridge) vs. improved (HistGradientBoosting) model comparison
- Dual-condition trade signal generation with model-error-aware suppression
- LLM-powered model diagnosis via OpenAI GPT-4o

---

## Repository Structure

```
DE_power_forecasting/
├── Notebook_Fair_Value_Forecast    # Main notebook (entry point)
├── data_loader.py                  # SMARD API + FRED data fetching
├── data_validation.py              # Quality checks & visual report
├── model_training.py               # Feature engineering, training, evaluation
├── evaluate_day_ahead_forecast.py  # Trade signal generation & display
├── __init__.py                     # Package marker
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## Quick Start

### 1. Prerequisites
- Databricks workspace with serverless compute (Python 3.10+)
- Internet access (for SMARD API and FRED)
- OpenAI API key (optional, for AI diagnosis cell only)

### 2. Run the notebook
Open `Notebook_Fair_Value_Forecast` and run all cells top-to-bottom. The notebook:
1. Installs `pandas_datareader` via `%pip`
2. Fetches 120 days of hourly data
3. Validates data quality
4. Trains both models
5. Displays diagnostic charts
6. Generates trade signals for the last available day
7. (Optional) Runs LLM diagnosis

### 3. Configuration
Key parameters are set in the notebook cells:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `lookback_days` | `build_dataset()` | 120 | Days of historical data to fetch |
| `split_date` | Cell 7 | 80% mark | Train/test temporal split point |
| `entry_threshold` | Cell 11 | 10.0 | Min spread (EUR/MWh) to trigger a signal |
| `error_threshold` | Cell 11 | 30.0 | Max acceptable model MAE for signal generation |
| `bias_correct` | Cell 7 | True | Whether to apply mean-bias correction |

---

## Module Reference

### `data_loader.py`
Handles all external data fetching and initial preprocessing.

| Function | Description |
|----------|-------------|
| `smard_get_series()` | Downloads a single SMARD time series as timezone-aware pandas Series |
| `load_fuel_prices()` | Fetches Henry Hub gas and Brent crude from FRED, resamples to hourly |
| `build_dataset()` | Orchestrates full dataset construction (SMARD + fuels + derived features) |
| `clean_dataset()` | Trims edge NaN rows and interpolates interior gaps |

### `data_validation.py`
Automated quality assurance before model training.

| Function | Description |
|----------|-------------|
| `validate_smard_series()` | Health-checks all SMARD API endpoints |
| `check_completeness()` | Identifies missing timestamps vs. expected hourly frequency |
| `check_column_quality()` | Reports nulls, infinities, and value ranges per column |
| `check_value_sanity()` | Checks values against physical bounds (e.g., load > 15 GW) |
| `validate_dataset()` | Runs all checks and returns a summary report |
| `plot_data_quality_report()` | Visual report with completeness and in-bounds bars |

### `model_training.py`
Feature engineering, model training, and evaluation.

| Function | Description |
|----------|-------------|
| `create_fair_value_features()` | Engineers all features (shares, regimes, interactions, calendar) |
| `train_baseline_model()` | Ridge regression with 10 fundamental features |
| `train_improved_model()` | HistGradientBoosting with 20 selected features |
| `evaluate_model()` | 9-metric evaluation (MAE, RMSE, R², bias, peak/low MAE, etc.) |
| `run_fair_value_workflow()` | Full pipeline: train both models, compute signals, regime diagnostics |

**Selected features (top 20 by permutation importance):**
1. `gas_x_residual_positive` — gas price × positive residual load (14.4)
2. `residual_load` — net load minus renewables (10.6)
3. `scarcity_score` — composite tightness indicator (3.1)
4. `vre_share`, `solar_share`, `wind_share` — renewable penetration ratios
5. `gas_price`, `oil_price` — fuel cost proxies (lagged 24h)
6. `solar_x_surplus`, `vre_x_surplus_depth` — regime interactions
7. Calendar encodings (hour, weekday sin/cos)

### `evaluate_day_ahead_forecast.py`
Trade signal generation with reliability filtering.

| Function | Description |
|----------|-------------|
| `evaluate_day_ahead()` | Generates signals for last full day with dual-condition filtering |
| `display_trade_ideas()` | Styled table + chart with buy/sell zones and suppressed markers |

---

## Data Sources

| Source | Data | Update Frequency | Access |
|--------|------|-------------------|--------|
| [SMARD](https://www.smard.de) | Day-ahead price, load, wind, solar forecasts | Hourly | Free, no key required |
| [FRED](https://fred.stlouisfed.org) | Henry Hub gas, Brent crude oil | Daily | Free via `pandas_datareader` |

---

## Model Performance (as of May 2026)

| Metric | Baseline (Ridge) | Improved (HGB) |
|--------|:-----------------:|:--------------:|
| MAE | 21.1 EUR/MWh | 20.9 EUR/MWh |
| RMSE | 40.8 EUR/MWh | 40.0 EUR/MWh |
| R² | 0.693 | 0.704 |
| Peak MAE (top 10%) | 14.3 EUR/MWh | 15.3 EUR/MWh |
| Low price MAE (bottom 10%) | 72.9 EUR/MWh | 72.4 EUR/MWh |
| Correlation | 0.882 | 0.891 |

**Known limitation:** Both models struggle with extreme negative prices (< -100 EUR/MWh) during renewable oversupply events. The error-threshold filtering in signal generation mitigates this by suppressing unreliable signals.

---

## AI Diagnosis (Optional)

Cell 13 uses GPT-4o to produce a natural-language diagnosis of model performance. To enable:
1. Set your OpenAI API key: replace `'YOUR_API_KEY'` in the cell, or set the `OPENAI_API_KEY` environment variable
2. Run the cell after model training completes

The LLM receives model metrics, error-correlated features, and worst forecast hours — then provides trader-oriented analysis of strengths, weaknesses, and improvement suggestions.

---

## Development Notes

- All modules use `importlib.reload()` in the notebook for hot-reloading during development
- The notebook is self-contained: running all cells top-to-bottom reproduces results
- Feature engineering still computes intermediate features (regime flags, quantiles) internally but only the top 20 are used by the model
- Bias correction is applied post-training by subtracting mean training-set residual

---

## License

Internal use — Uniper Energy Trading.
