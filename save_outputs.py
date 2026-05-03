"""Output saving module for DE power fair value forecasting.

Exports all notebook results (QA, model metrics, trade signals, LLM diagnosis)
to the output folder with date-stamped filenames.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = "output"


def save_qa_results(
    smard_status: pd.DataFrame,
    validation: dict,
    plot_fn,
    output_dir: str = OUTPUT_DIR,
    run_date: str | None = None,
) -> None:
    """Save QA results: SMARD API status, feature validation, and quality chart.

    Args:
        smard_status: DataFrame from validate_smard_series().
        validation: Dict from validate_dataset().
        plot_fn: The plot_data_quality_report function (to regenerate chart).
        output_dir: Directory to save outputs.
        run_date: Date string for filenames (default: today YYYYMMDD).
    """
    os.makedirs(output_dir, exist_ok=True)
    run_date = run_date or datetime.now().strftime("%Y%m%d")

    smard_status.to_csv(f"{output_dir}/qa_smard_api_status_{run_date}.csv", index=False)
    validation["feature_summary"].to_csv(f"{output_dir}/qa_feature_validation_{run_date}.csv")

    fig_qa = plot_fn(validation)
    if fig_qa:
        fig_qa.savefig(
            f"{output_dir}/qa_data_quality_report_{run_date}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_qa)

    print("[1/4] QA results saved")


def save_model_results(
    metrics_df: pd.DataFrame,
    fair_value_results: pd.DataFrame,
    regime_diagnostics: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
    run_date: str | None = None,
) -> None:
    """Save model results: metrics table, fair value predictions, and diagnostic charts.

    Args:
        metrics_df: Comparison DataFrame with Baseline/Improved/Delta columns.
        fair_value_results: Test set results with actual_price and fair values.
        regime_diagnostics: Per-regime MAE/bias DataFrame.
        df: Full cleaned dataset (for residual load and price columns).
        output_dir: Directory to save outputs.
        run_date: Date string for filenames (default: today YYYYMMDD).
    """
    os.makedirs(output_dir, exist_ok=True)
    run_date = run_date or datetime.now().strftime("%Y%m%d")

    metrics_df.to_csv(f"{output_dir}/model_metrics_comparison_{run_date}.csv", index=False)
    fair_value_results.to_csv(f"{output_dir}/model_fair_value_results_{run_date}.csv")
    regime_diagnostics.to_csv(f"{output_dir}/model_regime_diagnostics_{run_date}.csv")

    # Diagnostic charts
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    ax.plot(fair_value_results.index, fair_value_results["actual_price"], alpha=0.7, label="Actual")
    ax.plot(fair_value_results.index, fair_value_results["improved_fair_value"], alpha=0.7, label="Improved FV")
    ax.plot(fair_value_results.index, fair_value_results["baseline_fair_value"], alpha=0.4, label="Baseline FV")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Day-Ahead Price vs Fair Value Models (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    error = fair_value_results["improved_fair_value"] - fair_value_results["actual_price"]
    x = df.loc[fair_value_results.index, "residual_load_forecast_mwh"]
    ax.scatter(x, error, alpha=0.4, s=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Residual load forecast (MWh)")
    ax.set_ylabel("Fair value error (EUR/MWh)")
    ax.set_title("Improved Model Error vs Residual Load")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x = df.loc[fair_value_results.index, "price_day_ahead_eur_mwh"]
    ax.scatter(x, error, alpha=0.4, s=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Day-ahead price (EUR/MWh)")
    ax.set_ylabel("Fair value error (EUR/MWh)")
    ax.set_title("Improved Model Error vs Actual Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/model_diagnostic_charts_{run_date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[2/4] Model results saved")


def save_forecast_results(
    result: dict,
    output_dir: str = OUTPUT_DIR,
    run_date: str | None = None,
) -> None:
    """Save day-ahead forecast: trade signals CSV, text summary, and chart.

    Args:
        result: Dictionary returned by evaluate_day_ahead().
        output_dir: Directory to save outputs.
        run_date: Date string for filenames (default: today YYYYMMDD).
    """
    os.makedirs(output_dir, exist_ok=True)
    run_date = run_date or datetime.now().strftime("%Y%m%d")

    result["trade_view"].to_csv(f"{output_dir}/da_forecast_trade_signals_{run_date}.csv")

    summary = result["summary"]
    day = result["day"]
    params = result["params"]

    forecast_text = f"""DE Power Day-Ahead Forecast & Trade Recommendations
{'=' * 60}
Forecast Date: {day.strftime('%Y-%m-%d')}
Generated:     {datetime.now().strftime('%Y-%m-%d %H:%M')}
Parameters:    Entry threshold = {params['entry_threshold']:.0f} EUR/MWh, Error threshold = {params['error_threshold']:.0f} EUR/MWh

Trade Summary:
  BUY signals:    {summary['n_buy']} hours
  SELL signals:   {summary['n_sell']} hours
  HOLD:           {summary['n_hold']} hours
  SUPPRESSED:     {summary['n_suppressed']} hours (model error > {params['error_threshold']:.0f} EUR/MWh)
  Avg spread:     {summary['avg_spread']:+.2f} EUR/MWh
  Direction:      {summary['bias']}

Hourly Signals:
{result['trade_view'].to_string()}

Error by Price Bin:
{result['error_stats'].to_string(index=False)}
"""

    with open(f"{output_dir}/da_forecast_recommendations_{run_date}.txt", "w") as f:
        f.write(forecast_text)

    # Trade signal chart
    day_data = result["day_data"]
    fig, ax = plt.subplots(figsize=(12, 4))
    hours = range(len(day_data))
    ax.step(hours, day_data["actual_price"].values, where="mid", label="Actual Price", linewidth=2)
    ax.step(hours, day_data["improved_fair_value"].values, where="mid", label="Fair Value", linewidth=2, linestyle="--")

    for i, (idx, row) in enumerate(day_data.iterrows()):
        if row["signal"] == "BUY":
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color="green")
        elif row["signal"] == "SELL":
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color="red")
        if row["suppressed"]:
            ax.scatter(i, row["actual_price"], marker="x", color="orange", s=100, zorder=5)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Day-Ahead Forecast & Trade Signals \u2014 {day.strftime('%Y-%m-%d')}")
    ax.set_xticks(range(0, len(day_data), 1))
    ax.set_xticklabels([f"{h:02d}" for h in day_data.index.hour])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/da_forecast_chart_{run_date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[3/4] Day-ahead forecast saved")


def save_llm_diagnosis(
    diagnosis: str,
    improved_metrics: dict,
    top_features: pd.Series,
    worst_hours: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
    run_date: str | None = None,
) -> None:
    """Save the LLM diagnosis text output with input context.

    Args:
        diagnosis: LLM response text.
        improved_metrics: Dict of improved model metrics.
        top_features: Series of top error-correlated features.
        worst_hours: DataFrame of worst forecast hours.
        output_dir: Directory to save outputs.
        run_date: Date string for filenames (default: today YYYYMMDD).
    """
    os.makedirs(output_dir, exist_ok=True)
    run_date = run_date or datetime.now().strftime("%Y%m%d")

    llm_output = f"""AI Model Diagnosis \u2014 DE Power Fair Value Model
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model:     GPT-4o via OpenAI API

Input Context:
  - Model MAE: {improved_metrics['MAE']:.2f} EUR/MWh
  - Model RMSE: {improved_metrics['RMSE']:.2f} EUR/MWh
  - Model R\u00b2: {improved_metrics['R2']:.3f}
  - Bias: {improved_metrics['Bias_model_minus_actual']:.2f} EUR/MWh
  - Peak MAE: {improved_metrics['Peak_MAE_top_10pct']:.2f} EUR/MWh
  - Low price MAE: {improved_metrics['Low_price_MAE_bottom_10pct']:.2f} EUR/MWh

Top 10 Features Correlated with Error:
{top_features.to_string()}

Worst 5 Forecast Hours:
{worst_hours[['actual_price', 'improved_fair_value', 'improved_spread']].to_string()}

{'=' * 60}
LLM DIAGNOSIS:
{'=' * 60}
{diagnosis}
"""

    with open(f"{output_dir}/llm_diagnosis_{run_date}.txt", "w") as f:
        f.write(llm_output)

    print("[4/4] LLM diagnosis saved")


def save_all_outputs(
    smard_status: pd.DataFrame,
    validation: dict,
    plot_fn,
    metrics_df: pd.DataFrame,
    fair_value_results: pd.DataFrame,
    output: dict,
    df: pd.DataFrame,
    result: dict,
    diagnosis: str,
    top_features: pd.Series,
    worst_hours: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Save all notebook outputs to the output folder.

    Convenience function that calls all individual save functions.

    Args:
        smard_status: SMARD API validation DataFrame.
        validation: Dataset validation dict.
        plot_fn: plot_data_quality_report function.
        metrics_df: Model comparison metrics DataFrame.
        fair_value_results: Test set results DataFrame.
        output: Full workflow output dict from run_fair_value_workflow().
        df: Full cleaned dataset.
        result: Trade signal result dict from evaluate_day_ahead().
        diagnosis: LLM diagnosis text string.
        top_features: Top error-correlated features Series.
        worst_hours: Worst forecast hours DataFrame.
        output_dir: Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_date = datetime.now().strftime("%Y%m%d")

    save_qa_results(smard_status, validation, plot_fn, output_dir, run_date)
    save_model_results(
        metrics_df, fair_value_results, output["regime_diagnostics"], df, output_dir, run_date
    )
    save_forecast_results(result, output_dir, run_date)
    save_llm_diagnosis(
        diagnosis, output["improved_metrics"], top_features, worst_hours, output_dir, run_date
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 60}")
    for f_name in sorted(os.listdir(output_dir)):
        size_kb = os.path.getsize(f"{output_dir}/{f_name}") / 1024
        print(f"  {f_name:50s} ({size_kb:.1f} KB)")
