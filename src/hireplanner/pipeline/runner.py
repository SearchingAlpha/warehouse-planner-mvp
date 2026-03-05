"""Semi-automated pipeline orchestrator for weekly forecast generation.

Usage:
    hireplanner --config configs/clients/acme.yaml --data data/acme_volumes.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd


def run_pipeline(
    client_config_path: str,
    data_path: str,
    output_dir: str = "output",
    log_dir: str = "data/accuracy_logs",
    locales_dir: str = "configs/locales",
    dry_run: bool = False,
) -> str:
    """Run the full forecasting pipeline for a client.

    Steps:
        1. Load client config
        2. Load, clean, and validate data
        3. Load previous forecast (if exists)
        4. Run forecasting models
        5. Calculate backlog per flow
        6. Generate alerts
        7. Calculate headcount
        8. Compare previous forecast vs new actuals (accuracy)
        9. Append accuracy log
       10. Save current forecast
       11. Generate Markdown + PNG report
       12. Print summary

    Returns:
        Path to the generated report.md.
    """
    from hireplanner.config.client_config import load_client_config
    from hireplanner.ingestion.loader import load_data
    from hireplanner.ingestion.cleaner import clean_data
    from hireplanner.ingestion.validator import validate_data, validate_data_strict

    # Step 1: Load client config
    _log("Loading client config...")
    config = load_client_config(client_config_path)
    _log(f"  Client: {config.client_name}")
    _log(f"  Active flows: {config.active_flows}")
    _log(f"  Language: {config.language}")

    # Step 2: Load, clean, validate data
    _log("Loading and processing data...")
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)
    validate_data_strict(cleaned_df)
    warnings = validate_data(cleaned_df)
    for w in warnings:
        _log(f"  Warning: {w}")

    if dry_run:
        _log("Dry run complete. Data is valid.")
        return ""

    # Step 3: Load previous forecast for accuracy comparison
    _log("Checking for previous forecast...")
    from hireplanner.metrics.accuracy_tracker import (
        load_previous_forecast,
        compare_forecast_to_actual,
        calculate_accuracy_metrics,
        append_accuracy_log,
        load_accuracy_log,
        get_accuracy_trend,
        save_forecast,
    )

    prev_forecast = load_previous_forecast(config.client_name, log_dir)

    # Step 4: Run forecasting
    _log("Running forecast models...")
    forecast_df = _run_forecasting(cleaned_df, config)
    _log(f"  Forecast generated: {len(forecast_df)} rows, {config.forecast_horizon} days")

    # Step 5: Calculate backlog per flow
    _log("Calculating backlog projections...")
    from hireplanner.planning.backlog import calculate_all_backlogs

    backlog_dfs = calculate_all_backlogs(config, forecast_df)
    for flow, bdf in backlog_dfs.items():
        _log(f"  {flow}: peak end backlog = {bdf['end_backlog'].max():.0f} units")

    # Step 6: Generate alerts
    _log("Generating alerts...")
    from hireplanner.planning.alerts import generate_alert_series, summarize_alerts

    alert_series_dict = {}
    alert_summary = {"critical_days": 0, "watch_days": 0, "healthy_days": 0}
    for flow, bdf in backlog_dfs.items():
        alerts = generate_alert_series(bdf["days_of_backlog"], config)
        alert_series_dict[flow] = alerts
        summary = summarize_alerts(alerts, bdf["days_of_backlog"], bdf["date"])
        # Merge into combined summary
        alert_summary["critical_days"] += summary["critical_days"]
        alert_summary["watch_days"] += summary["watch_days"]
        alert_summary["healthy_days"] += summary["healthy_days"]
        if summary.get("first_critical_date") is not None:
            if alert_summary.get("first_critical_date") is None:
                alert_summary["first_critical_date"] = summary["first_critical_date"]
        alert_summary["peak_days_of_backlog"] = max(
            alert_summary.get("peak_days_of_backlog", 0),
            summary.get("peak_days_of_backlog", 0),
        )

    _log(f"  Critical days: {alert_summary['critical_days']}")
    _log(f"  Watch days: {alert_summary['watch_days']}")

    # Step 7: Calculate headcount
    _log("Calculating headcount plan...")
    from hireplanner.planning.labor import build_headcount_plan

    headcount_df = build_headcount_plan(forecast_df, config)
    peak_hc = int(headcount_df["hc_total"].max())
    _log(f"  Peak total headcount: {peak_hc}")

    # Step 8: Compare previous forecast vs actuals
    accuracy_data = None
    if prev_forecast is not None:
        _log("Comparing previous forecast to actuals...")
        accuracy_data = _calculate_accuracy(
            prev_forecast, cleaned_df, config, log_dir
        )
        if accuracy_data and accuracy_data.get("metrics"):
            _log(f"  WAPE: {accuracy_data['metrics']['wape']:.1%}")
            _log(f"  Trend: {accuracy_data.get('trend', 'stable')}")

    # Step 9-10: Append accuracy log and save current forecast
    if accuracy_data and accuracy_data.get("metrics"):
        append_accuracy_log(config.client_name, accuracy_data["metrics"], log_dir)
        _log("  Accuracy log updated.")

    run_date = date.today()
    save_forecast(config.client_name, forecast_df, run_date, log_dir)
    _log("  Current forecast saved.")

    # Step 11: Generate Markdown + PNG report
    _log("Generating Markdown report...")
    from hireplanner.reporting.markdown_generator import generate_markdown_report

    safe_name = config.client_name.lower().replace(" ", "_")
    report_output_dir = Path(output_dir) / f"{safe_name}_{run_date.isoformat()}"

    report_path = generate_markdown_report(
        config=config,
        forecast_df=forecast_df,
        backlog_dfs=backlog_dfs,
        headcount_df=headcount_df,
        accuracy_data=accuracy_data,
        alert_summary=alert_summary,
        alert_series_dict=alert_series_dict,
        output_dir=report_output_dir,
        locales_dir=locales_dir,
    )

    # Step 12: Print summary
    _log("")
    _log("=" * 60)
    _log(f"  Report generated: {report_path}")
    _log(f"  Client: {config.client_name}")
    _log(f"  Forecast period: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
    _log(f"  Peak headcount: {peak_hc}")
    _log(f"  Critical alert days: {alert_summary['critical_days']}")
    if accuracy_data and accuracy_data.get("metrics"):
        _log(f"  Forecast accuracy (WAPE): {accuracy_data['metrics']['wape']:.1%}")
    else:
        _log("  Forecast accuracy: N/A (first run)")
    _log("=" * 60)

    return report_path


def _run_forecasting(
    cleaned_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """Run LightGBM forecasting for each active flow.

    Returns standard forecast DataFrame with columns:
        date, flow, forecast_p50, forecast_p10, forecast_p90
    """
    from hireplanner.forecasting.lightgbm_model import LightGBMForecaster
    from hireplanner.forecasting.ensemble import build_forecast_df

    # Determine the last date in history and build forecast dates
    last_date = pd.to_datetime(cleaned_df["date"]).max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=config.forecast_horizon,
        freq="D",
    )
    history_dates = pd.DatetimeIndex(pd.to_datetime(cleaned_df["date"]))

    all_forecasts = []

    for flow in config.active_flows:
        if flow not in cleaned_df.columns:
            continue

        history = cleaned_df[flow]
        _log(f"  Training LightGBM for {flow}...")

        try:
            model = LightGBMForecaster()
            result = model.forecast(history, history_dates, horizon=config.forecast_horizon)
        except Exception as e:
            _log(f"  LightGBM failed for {flow}: {e}. Using naive fallback.")
            import numpy as np

            values = history.values
            last_week = values[-7:] if len(values) >= 7 else values
            repeats = (config.forecast_horizon // len(last_week)) + 1
            naive = np.tile(last_week, repeats)[: config.forecast_horizon]
            result = {
                "p10": naive * 0.8,
                "p50": naive,
                "p90": naive * 1.2,
            }

        flow_df = build_forecast_df(result, forecast_dates, flow)
        all_forecasts.append(flow_df)

    return pd.concat(all_forecasts, ignore_index=True)


def _calculate_accuracy(
    prev_forecast: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    config,
    log_dir: str,
) -> Optional[dict]:
    """Calculate accuracy metrics comparing previous forecast to new actuals."""
    from hireplanner.metrics.accuracy_tracker import (
        compare_forecast_to_actual,
        calculate_accuracy_metrics,
        load_accuracy_log,
        get_accuracy_trend,
    )

    accuracy_data = {"comparison": None, "metrics": None, "trend": "stable"}

    for flow in config.active_flows:
        if flow not in cleaned_df.columns:
            continue

        try:
            comparison = compare_forecast_to_actual(prev_forecast, cleaned_df, flow)
            if comparison.empty:
                continue

            metrics = calculate_accuracy_metrics(comparison)
            accuracy_data["comparison"] = comparison
            accuracy_data["metrics"] = metrics

            # Determine trend
            log = load_accuracy_log(config.client_name, log_dir)
            if log is not None:
                accuracy_data["trend"] = get_accuracy_trend(log)

            break  # Use first flow with data for accuracy
        except Exception as e:
            _log(f"  Accuracy calc failed for {flow}: {e}")

    return accuracy_data


def _log(message: str) -> None:
    """Print a log message."""
    print(message)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hireplanner",
        description="HireRobots: Weekly forecasting and labor planning for 3PL warehouses",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to client YAML config file",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to volume data CSV/Excel file",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for Markdown report (default: output/)",
    )
    parser.add_argument(
        "--log-dir",
        default="data/accuracy_logs",
        help="Directory for accuracy logs (default: data/accuracy_logs/)",
    )
    parser.add_argument(
        "--locales-dir",
        default="configs/locales",
        help="Directory for locale YAML files (default: configs/locales/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate data and config only, do not generate forecast",
    )

    args = parser.parse_args()

    try:
        report_path = run_pipeline(
            client_config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
            log_dir=args.log_dir,
            locales_dir=args.locales_dir,
            dry_run=args.dry_run,
        )
        if report_path:
            sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
