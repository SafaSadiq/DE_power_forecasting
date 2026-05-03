"""Day-ahead forecast evaluation and trade signal generation.

Provides functions to generate hourly trade signals for a given day
based on the spread between actual price and model fair value,
with signal suppression when the model error is too high for
that price level to be reliable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _compute_error_by_price_bin(
    fair_value_results: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute historical MAE per price bin from the full test set.

    Used to assess model reliability at different price levels.

    Args:
        fair_value_results: Full test set results with 'actual_price'
            and 'improved_fair_value' columns.
        n_bins: Number of equal-frequency bins to split prices into.

    Returns:
        DataFrame with columns 'price_bin', 'mae', 'std', 'count', 'bin_mid'.
    """
    df = fair_value_results.copy()
    df["error"] = (df["improved_fair_value"] - df["actual_price"]).abs()
    df["price_bin"] = pd.qcut(df["actual_price"], n_bins, duplicates="drop")

    stats = df.groupby("price_bin", observed=True)["error"].agg(["mean", "std", "count"])
    stats.columns = ["mae", "std", "count"]
    stats["bin_mid"] = stats.index.map(lambda x: x.mid)
    return stats.reset_index()


def _lookup_hist_mae(price: float, bin_mae_map: dict) -> float:
    """Find the historical MAE for a given price from the bin map."""
    for bin_interval, mae in bin_mae_map.items():
        if price in bin_interval:
            return mae
    return np.nan


def evaluate_day_ahead(
    fair_value_results: pd.DataFrame,
    df: pd.DataFrame,
    entry_threshold: float = 10.0,
    error_threshold: float = 30.0,
) -> dict:
    """Generate trade signals for the last full day with error-based filtering.

    For each hour, compares the actual price to the model fair value.
    A signal is only generated when BOTH conditions are met:
    1. The absolute spread exceeds entry_threshold.
    2. The historical model error (MAE) for that price level is below
       error_threshold.

    If condition 1 is met but condition 2 is not, the signal is suppressed
    and highlighted as unreliable.

    Args:
        fair_value_results: Test set results from run_fair_value_workflow,
            with columns 'actual_price', 'improved_fair_value'.
        df: Full cleaned dataset.
        entry_threshold: Minimum absolute spread (EUR/MWh) to consider
            a trade signal.
        error_threshold: Maximum acceptable historical MAE (EUR/MWh) for
            the price bin. Signals at price levels where the model error
            exceeds this are suppressed.

    Returns:
        Dictionary with keys:
        - 'day': the date evaluated
        - 'trade_view': display-ready DataFrame with hourly signals
        - 'day_data': raw DataFrame for the day
        - 'summary': dict with trade counts and bias
        - 'error_stats': historical error by price bin
        - 'params': dict with entry_threshold and error_threshold
    """
    # Find last full day (24 hours)
    day_counts = fair_value_results.groupby(
        fair_value_results.index.normalize()
    ).size()
    full_days = day_counts[day_counts >= 24].index
    last_day = full_days.max()

    day_data = fair_value_results[
        fair_value_results.index.normalize() == last_day
    ].copy()

    # Compute spread
    day_data["spread"] = day_data["actual_price"] - day_data["improved_fair_value"]

    # Compute historical error by price bin
    error_stats = _compute_error_by_price_bin(fair_value_results)

    # Build bin -> MAE lookup
    price_bins = pd.qcut(
        fair_value_results["actual_price"],
        q=error_stats.shape[0],
        duplicates="drop",
    )
    bin_mae_map = dict(zip(price_bins.cat.categories, error_stats["mae"].values))

    # Assign historical MAE for each hour's price level
    day_data["hist_mae"] = day_data["actual_price"].map(
        lambda p: _lookup_hist_mae(p, bin_mae_map)
    )

    # Generate signals with dual-condition filtering
    spread_exceeds = day_data["spread"].abs() > entry_threshold
    error_acceptable = day_data["hist_mae"] <= error_threshold

    day_data["signal"] = "HOLD"
    day_data.loc[spread_exceeds & error_acceptable & (day_data["spread"] > 0), "signal"] = "SELL"
    day_data.loc[spread_exceeds & error_acceptable & (day_data["spread"] < 0), "signal"] = "BUY"

    # Mark suppressed signals (spread exceeded threshold but error too high)
    day_data["suppressed"] = spread_exceeds & ~error_acceptable

    # Build display signal column
    day_data["display_signal"] = day_data["signal"]
    day_data.loc[day_data["suppressed"], "display_signal"] = "SUPPRESSED (high model error)"

    # Build trade view table
    trade_view = day_data[
        ["actual_price", "improved_fair_value", "spread", "hist_mae", "display_signal"]
    ].copy()
    trade_view.columns = [
        "Actual (EUR/MWh)",
        "Fair Value (EUR/MWh)",
        "Spread (EUR/MWh)",
        "Hist. MAE (EUR/MWh)",
        "Signal",
    ]
    trade_view.index = trade_view.index.strftime("%H:%M")
    trade_view.index.name = "Hour"

    # Summary
    n_buy = (day_data["signal"] == "BUY").sum()
    n_sell = (day_data["signal"] == "SELL").sum()
    n_suppressed = day_data["suppressed"].sum()
    avg_spread = day_data["spread"].mean()

    if avg_spread < -5:
        bias = "Market priced BELOW fair value. Lean BUY."
    elif avg_spread > 5:
        bias = "Market priced ABOVE fair value. Lean SELL."
    else:
        bias = "Market fairly priced overall."

    summary = {
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": len(day_data) - n_buy - n_sell - n_suppressed,
        "n_suppressed": int(n_suppressed),
        "avg_spread": avg_spread,
        "bias": bias,
    }

    return {
        "day": last_day,
        "trade_view": trade_view,
        "day_data": day_data,
        "summary": summary,
        "error_stats": error_stats,
        "params": {"entry_threshold": entry_threshold, "error_threshold": error_threshold},
    }


def display_trade_ideas(result: dict) -> None:
    """Display the trade ideas table, summary, and chart.

    Args:
        result: Dictionary returned by evaluate_day_ahead().
    """
    trade_view = result["trade_view"]
    day_data = result["day_data"]
    summary = result["summary"]
    last_day = result["day"]
    params = result["params"]

    # Table styling
    table_styles = [
        {"selector": "caption", "props": "font-size: 13px; font-weight: bold; text-align: left; padding: 6px 0;"},
        {"selector": "th", "props": "border: 1px solid #333; padding: 6px 10px; background-color: #f5f5f5; font-weight: bold;"},
        {"selector": "td", "props": "border: 1px solid #333; padding: 6px 10px;"},
        {"selector": "table", "props": "border-collapse: collapse; border: 2px solid #333;"},
    ]

    def color_signal(val):
        val_str = str(val)
        if "SUPPRESSED" in val_str:
            return "color: orange; font-style: italic; text-decoration: line-through;"
        elif "BUY" in val_str:
            return "color: green; font-weight: bold;"
        elif "SELL" in val_str:
            return "color: red; font-weight: bold;"
        return ""

    def highlight_high_error(val):
        if pd.notna(val) and val > params["error_threshold"]:
            return "background-color: #fff3cd;"
        return ""

    display(
        trade_view.round(2).style
        .set_caption(
            f"Trade Ideas for {last_day.strftime('%Y-%m-%d')} "
            f"(spread threshold: \u00b1{params['entry_threshold']:.0f} EUR/MWh, "
            f"max model error: {params['error_threshold']:.0f} EUR/MWh)"
        )
        .set_table_styles(table_styles)
        .map(color_signal, subset=["Signal"])
        .map(highlight_high_error, subset=["Hist. MAE (EUR/MWh)"])
        .format({
            "Actual (EUR/MWh)": "{:.2f}",
            "Fair Value (EUR/MWh)": "{:.2f}",
            "Spread (EUR/MWh)": "{:+.2f}",
            "Hist. MAE (EUR/MWh)": "{:.1f}",
        })
    )

    # Print summary
    print(f"\nTrade Summary for {last_day.strftime('%Y-%m-%d')}:")
    print(f"  Buy signals:       {summary['n_buy']} hours")
    print(f"  Sell signals:      {summary['n_sell']} hours")
    print(f"  Hold:              {summary['n_hold']} hours")
    print(f"  Suppressed:        {summary['n_suppressed']} hours (model error > {params['error_threshold']:.0f} EUR/MWh at that price)")
    print(f"  Avg spread:        {summary['avg_spread']:+.2f} EUR/MWh")
    print(f"  \u2192 {summary['bias']}")

    # Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    hours = range(len(day_data))
    threshold = params["entry_threshold"]

    ax.step(hours, day_data["actual_price"].values, where="mid", label="Actual Price", linewidth=2)
    ax.step(hours, day_data["improved_fair_value"].values, where="mid", label="Fair Value", linewidth=2, linestyle="--")

    # Shade valid signal zones
    valid_sell = (day_data["signal"].values == "SELL")
    valid_buy = (day_data["signal"].values == "BUY")
    ax.fill_between(
        hours, day_data["actual_price"].values, day_data["improved_fair_value"].values,
        where=valid_sell, alpha=0.3, color="red", label="Sell zone", step="mid",
    )
    ax.fill_between(
        hours, day_data["actual_price"].values, day_data["improved_fair_value"].values,
        where=valid_buy, alpha=0.3, color="green", label="Buy zone", step="mid",
    )

    # Mark suppressed hours
    suppressed_mask = day_data["suppressed"].values
    if suppressed_mask.any():
        supp_hours = [h for h, m in zip(hours, suppressed_mask) if m]
        supp_prices = day_data["actual_price"].values[suppressed_mask]
        ax.scatter(supp_hours, supp_prices, color="orange", s=80, zorder=5,
                   marker="x", linewidths=2, label="Suppressed (high error)")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Day-Ahead Forecast & Trade Signals \u2014 {last_day.strftime('%Y-%m-%d')}")
    ax.legend(loc="upper right")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in day_data.index.hour])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
