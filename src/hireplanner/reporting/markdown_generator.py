"""Markdown + PNG report generator for warehouse planning."""
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table string."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.1f}" if abs(v) >= 10 else f"{v:.2f}")
            elif isinstance(v, (pd.Timestamp, np.datetime64)):
                cells.append(pd.Timestamp(v).strftime("%Y-%m-%d"))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _alert_indicator(status: str) -> str:
    """Return a text indicator for alert status."""
    if status == "Critical":
        return "[CRITICAL]"
    elif status == "Watch":
        return "[!!]"
    return "[OK]"


def _write_executive_summary(
    config, alert_summary, accuracy_metrics, headcount_summary, t_fn, locale,
    cost_summary=None,
) -> str:
    """Generate the Executive Summary section."""
    lines = []
    lines.append(f"# {config.client_name}")
    lines.append("")
    lines.append(f"**{t_fn('labels.report_date', locale)}:** {date.today().isoformat()}")
    lines.append("")
    lines.append("## " + t_fn("tabs.executive_summary", locale))
    lines.append("")

    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")

    if accuracy_metrics:
        lines.append(f"| {t_fn('labels.wape', locale)} | {accuracy_metrics.get('wape', 0):.1%} |")

    if alert_summary:
        lines.append(
            f"| {t_fn('labels.avg_days_of_backlog', locale)} "
            f"| {alert_summary.get('peak_days_of_backlog', 0):.1f} |"
        )
        lines.append(
            f"| {t_fn('labels.critical_alert_days', locale)} "
            f"| {alert_summary.get('critical_days', 0)} |"
        )

    if headcount_summary:
        lines.append(
            f"| {t_fn('labels.peak_headcount', locale)} "
            f"| {headcount_summary.get('peak_total', 0)} |"
        )

    if cost_summary and cost_summary.get("total_savings", 0) != 0:
        lines.append(
            f"| {t_fn('labels.total_savings', locale)} "
            f"| ${cost_summary['total_savings']:,.2f} |"
        )
        lines.append(
            f"| {t_fn('labels.avg_daily_savings', locale)} "
            f"| ${cost_summary['avg_daily_savings']:,.2f} |"
        )

    lines.append("")

    if alert_summary and alert_summary.get("critical_days", 0) > 0:
        lines.append(f"**{t_fn('alerts.critical', locale)}:** "
                      f"{alert_summary['critical_days']} days in critical status")
        if alert_summary.get("first_critical_date"):
            lines.append(f"First critical date: {alert_summary['first_critical_date']}")
        lines.append("")

    trend = accuracy_metrics.get("trend", "stable") if accuracy_metrics else "stable"
    lines.append(f"**{t_fn('labels.trend', locale)}:** {t_fn(f'labels.{trend}', locale)}")
    lines.append("")

    return "\n".join(lines)


def _write_daily_forecast(config, forecast_df, figures_dir, t_fn, locale) -> str:
    """Generate the Daily Forecast section with charts."""
    from hireplanner.reporting.matplotlib_charts import save_forecast_chart

    lines = []
    lines.append("## " + t_fn("tabs.daily_forecast", locale))
    lines.append("")

    for flow in config.active_flows:
        flow_data = forecast_df[forecast_df["flow"] == flow].copy().reset_index(drop=True)
        if flow_data.empty:
            continue

        flow_label = t_fn(f"labels.{flow}", locale)
        lines.append(f"### {flow_label}")
        lines.append("")

        display_df = pd.DataFrame({
            t_fn("headers.date", locale): flow_data["date"],
            t_fn("headers.day_of_week", locale): pd.to_datetime(flow_data["date"]).dt.day_name(),
            t_fn("headers.forecast_p50", locale): flow_data["forecast_p50"].round(0),
            t_fn("headers.forecast_p10", locale): flow_data["forecast_p10"].round(0),
            t_fn("headers.forecast_p90", locale): flow_data["forecast_p90"].round(0),
        })
        lines.append(_df_to_markdown(display_df))
        lines.append("")

        chart_name = f"forecast_{flow}.png"
        save_forecast_chart(
            flow_data["date"].values,
            flow_data["forecast_p10"].values,
            flow_data["forecast_p50"].values,
            flow_data["forecast_p90"].values,
            title=f"{flow_label} - {t_fn('tabs.daily_forecast', locale)}",
            path=figures_dir / chart_name,
        )
        lines.append(f"![{flow_label} forecast](figures/{chart_name})")
        lines.append("")

    return "\n".join(lines)


def _write_backlog_projection(
    config, backlog_dfs, alert_series_dict, figures_dir, t_fn, locale,
) -> str:
    """Generate the Backlog Projection section."""
    from hireplanner.reporting.matplotlib_charts import save_backlog_chart

    lines = []
    lines.append("## " + t_fn("tabs.backlog_projection", locale))
    lines.append("")

    for flow in config.active_flows:
        if flow not in backlog_dfs:
            continue

        bdf = backlog_dfs[flow]
        alert_series = alert_series_dict.get(flow, pd.Series(["Healthy"] * len(bdf)))
        flow_label = t_fn(f"labels.{flow}", locale)
        current_staffing = getattr(config, f"current_staffing_{flow}", 0)

        lines.append(f"### {flow_label}")
        lines.append("")

        cols = {
            t_fn("headers.date", locale): bdf["date"],
            t_fn("headers.beg_backlog", locale): bdf["beg_backlog"].round(0),
            t_fn("headers.new_demand", locale): bdf["new_demand"].round(0),
        }

        # Recommended capacity columns
        cap_rec_label = t_fn("headers.capacity_recommended", locale)
        end_rec_label = t_fn("headers.end_backlog_recommended", locale)
        dob_rec_label = t_fn("headers.days_of_backlog_recommended", locale)
        cols[cap_rec_label] = bdf["capacity_recommended"].round(0)
        cols[end_rec_label] = bdf["end_backlog_recommended"].round(0)
        cols[dob_rec_label] = bdf["days_of_backlog_recommended"].round(2)

        # Actual capacity columns (only when staffing > 0)
        if current_staffing > 0:
            cap_act_label = t_fn("headers.capacity_actual", locale)
            end_act_label = t_fn("headers.end_backlog_actual", locale)
            dob_act_label = t_fn("headers.days_of_backlog_actual", locale)
            cols[cap_act_label] = bdf["capacity_actual"].round(0)
            cols[end_act_label] = bdf["end_backlog_actual"].round(0)
            cols[dob_act_label] = bdf["days_of_backlog_actual"].round(2)

        cols[t_fn("headers.alert_status", locale)] = [
            _alert_indicator(s) for s in alert_series.values
        ]

        display_df = pd.DataFrame(cols)
        lines.append(_df_to_markdown(display_df))
        lines.append("")

        # Compute target backlog for chart
        demand_arr = bdf["new_demand"].values
        reference_daily_capacity = float(np.mean(demand_arr))
        target_backlog_units = (
            config.target_backlog_ratio
            * config.backlog_threshold_critical
            * reference_daily_capacity
        )

        chart_name = f"backlog_{flow}.png"
        end_actual = bdf["end_backlog_actual"].values if current_staffing > 0 else None
        save_backlog_chart(
            bdf["date"].values,
            bdf["end_backlog_recommended"].values,
            end_actual,
            target_backlog_units,
            title=f"{flow_label} - {t_fn('tabs.backlog_projection', locale)}",
            path=figures_dir / chart_name,
        )
        lines.append(f"![{flow_label} backlog](figures/{chart_name})")
        lines.append("")

    return "\n".join(lines)


def _write_headcount_plan(config, headcount_df, figures_dir, t_fn, locale) -> str:
    """Generate the Headcount Plan section."""
    from hireplanner.reporting.matplotlib_charts import save_headcount_chart

    lines = []
    lines.append("## " + t_fn("tabs.headcount_plan", locale))
    lines.append("")

    cols = {t_fn("headers.date", locale): headcount_df["date"]}
    hc_chart_data = {}

    for flow in config.active_flows:
        rec_col = f"hc_{flow}_recommended"
        act_col = f"hc_{flow}_actual"
        current_staffing = getattr(config, f"current_staffing_{flow}", 0)

        rec_label = t_fn(f"headers.hc_{flow}", locale) + " " + t_fn("headers.hc_recommended", locale)
        cols[rec_label] = headcount_df[rec_col]
        hc_chart_data[rec_label] = headcount_df[rec_col].values

        if current_staffing > 0:
            act_label = t_fn(f"headers.hc_{flow}", locale) + " " + t_fn("headers.hc_actual", locale)
            cols[act_label] = headcount_df[act_col]
            hc_chart_data[act_label] = headcount_df[act_col].values

    total_rec_label = t_fn("headers.hc_total", locale) + " " + t_fn("headers.hc_recommended", locale)
    cols[total_rec_label] = headcount_df["hc_total_recommended"]
    hc_chart_data[total_rec_label] = headcount_df["hc_total_recommended"].values

    total_actual_staffing = config.current_staffing_outbound + config.current_staffing_inbound
    if total_actual_staffing > 0:
        total_act_label = t_fn("headers.hc_total", locale) + " " + t_fn("headers.hc_actual", locale)
        cols[total_act_label] = headcount_df["hc_total_actual"]
        hc_chart_data[total_act_label] = headcount_df["hc_total_actual"].values

    # Cost columns (only when cost_per_hour > 0 and there is actual staffing)
    if config.cost_per_hour > 0 and total_actual_staffing > 0:
        cols[t_fn("headers.daily_cost_recommended", locale)] = headcount_df["daily_cost_recommended"].round(2)
        cols[t_fn("headers.daily_cost_actual", locale)] = headcount_df["daily_cost_actual"].round(2)
        cols[t_fn("headers.daily_savings", locale)] = headcount_df["daily_savings"].round(2)

    display_df = pd.DataFrame(cols)
    lines.append(_df_to_markdown(display_df))
    lines.append("")

    # Weekly summary
    lines.append(f"### {t_fn('labels.weekly_summary', locale)}")
    lines.append("")
    hc_copy = headcount_df.copy()
    hc_copy["week"] = pd.to_datetime(hc_copy["date"]).dt.isocalendar().week
    weekly = hc_copy.groupby("week").agg(
        avg_hc=("hc_total_recommended", "mean"),
        total_hours=("hc_total_recommended", lambda x: x.sum() * config.hours_per_shift),
    ).reset_index()
    weekly["avg_hc"] = weekly["avg_hc"].round(1)
    weekly["total_hours"] = weekly["total_hours"].round(0)
    weekly.columns = [
        t_fn("headers.date", locale),
        t_fn("labels.avg_daily_hc", locale),
        t_fn("labels.total_hours", locale),
    ]
    lines.append(_df_to_markdown(weekly))
    lines.append("")

    # One chart per flow + one total chart
    for flow in config.active_flows:
        flow_label = t_fn(f"labels.{flow}", locale)
        current_staffing = getattr(config, f"current_staffing_{flow}", 0)
        flow_chart_data = {}

        rec_label = t_fn(f"headers.hc_{flow}", locale) + " " + t_fn("headers.hc_recommended", locale)
        flow_chart_data[rec_label] = headcount_df[f"hc_{flow}_recommended"].values

        if current_staffing > 0:
            act_label = t_fn(f"headers.hc_{flow}", locale) + " " + t_fn("headers.hc_actual", locale)
            flow_chart_data[act_label] = headcount_df[f"hc_{flow}_actual"].values

        chart_name = f"headcount_{flow}.png"
        save_headcount_chart(
            headcount_df["date"].values,
            flow_chart_data,
            title=f"{flow_label} - {t_fn('tabs.headcount_plan', locale)}",
            path=figures_dir / chart_name,
        )
        lines.append(f"![{flow_label} headcount](figures/{chart_name})")
        lines.append("")

    # Total headcount chart
    total_chart_data = {}
    total_rec_label_chart = t_fn("headers.hc_total", locale) + " " + t_fn("headers.hc_recommended", locale)
    total_chart_data[total_rec_label_chart] = headcount_df["hc_total_recommended"].values

    total_actual_staffing_chart = config.current_staffing_outbound + config.current_staffing_inbound
    if total_actual_staffing_chart > 0:
        total_act_label_chart = t_fn("headers.hc_total", locale) + " " + t_fn("headers.hc_actual", locale)
        total_chart_data[total_act_label_chart] = headcount_df["hc_total_actual"].values

    chart_name = "headcount_total.png"
    save_headcount_chart(
        headcount_df["date"].values,
        total_chart_data,
        title=f"{t_fn('labels.total', locale)} - {t_fn('tabs.headcount_plan', locale)}",
        path=figures_dir / chart_name,
    )
    lines.append(f"![Total headcount](figures/{chart_name})")
    lines.append("")

    # Cost savings subsection
    if config.cost_per_hour > 0 and total_actual_staffing_chart > 0:
        from hireplanner.reporting.matplotlib_charts import save_cost_savings_chart

        total_cost_rec = headcount_df["daily_cost_recommended"].sum()
        total_cost_act = headcount_df["daily_cost_actual"].sum()
        total_savings = total_cost_act - total_cost_rec

        lines.append(f"### {t_fn('labels.cost_savings', locale)}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| {t_fn('labels.total_cost_recommended', locale)} | ${total_cost_rec:,.2f} |")
        lines.append(f"| {t_fn('labels.total_cost_actual', locale)} | ${total_cost_act:,.2f} |")
        lines.append(f"| {t_fn('labels.total_savings', locale)} | ${total_savings:,.2f} |")
        lines.append("")

        daily_savings = headcount_df["daily_savings"].values
        cumulative_savings = np.cumsum(daily_savings)
        chart_name_savings = "cost_savings.png"
        save_cost_savings_chart(
            headcount_df["date"].values,
            daily_savings,
            cumulative_savings,
            title=t_fn("labels.cost_savings", locale),
            path=figures_dir / chart_name_savings,
        )
        lines.append(f"![Cost Savings](figures/{chart_name_savings})")
        lines.append("")

    return "\n".join(lines)


def _write_accuracy_report(config, accuracy_data, figures_dir, t_fn, locale) -> str:
    """Generate the Accuracy Report section."""
    from hireplanner.reporting.matplotlib_charts import save_accuracy_chart

    lines = []
    lines.append("## " + t_fn("tabs.accuracy_report", locale))
    lines.append("")

    comparison_val = accuracy_data.get("comparison") if isinstance(accuracy_data, dict) else None
    has_comparison = comparison_val is not None and (
        not isinstance(comparison_val, pd.DataFrame) or not comparison_val.empty
    )

    if accuracy_data is None or (isinstance(accuracy_data, dict) and not has_comparison):
        lines.append(t_fn("labels.no_accuracy_data", locale))
        lines.append("")
        return "\n".join(lines)

    comparison_df = accuracy_data.get("comparison") if isinstance(accuracy_data, dict) else accuracy_data
    metrics = accuracy_data.get("metrics", {}) if isinstance(accuracy_data, dict) else {}
    trend = accuracy_data.get("trend", "stable") if isinstance(accuracy_data, dict) else "stable"

    # Metrics table
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for metric_name in ["wape", "mape", "mae"]:
        if metric_name in metrics:
            fmt = f"{metrics[metric_name]:.1%}" if metric_name in ("wape", "mape") else f"{metrics[metric_name]:.0f}"
            lines.append(f"| {t_fn(f'labels.{metric_name}', locale)} | {fmt} |")
    lines.append(f"| {t_fn('labels.trend', locale)} | {t_fn(f'labels.{trend}', locale)} |")
    lines.append("")

    # Comparison table
    if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
        display_df = pd.DataFrame({
            t_fn("headers.date", locale): comparison_df["date"],
            t_fn("headers.forecast_col", locale): comparison_df["forecast"].round(0),
            t_fn("headers.actual", locale): comparison_df["actual"].round(0),
            t_fn("headers.error", locale): comparison_df["absolute_error"].round(0),
            t_fn("headers.pct_error", locale): (comparison_df["percentage_error"] * 100).round(1),
        })
        lines.append(_df_to_markdown(display_df))
        lines.append("")

        chart_name = "accuracy.png"
        save_accuracy_chart(
            comparison_df["date"].values,
            comparison_df["forecast"].values,
            comparison_df["actual"].values,
            title=t_fn("tabs.accuracy_report", locale),
            path=figures_dir / chart_name,
        )
        lines.append(f"![Accuracy comparison](figures/{chart_name})")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_report(
    config,
    forecast_df: pd.DataFrame,
    backlog_dfs: dict[str, pd.DataFrame],
    headcount_df: pd.DataFrame,
    accuracy_data: Optional[dict] = None,
    alert_summary: Optional[dict] = None,
    alert_series_dict: Optional[dict[str, pd.Series]] = None,
    output_dir: str | Path = "output",
    locales_dir: str | Path = "configs/locales",
) -> str:
    """Generate a Markdown report with embedded PNG chart references.

    Creates output_dir/report.md + output_dir/figures/*.png

    Returns path to the generated report.md.
    """
    from hireplanner.config.i18n import load_locale, t

    locale = load_locale(config.language, locales_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if alert_series_dict is None:
        alert_series_dict = {}

    headcount_summary = {
        "peak_total": int(headcount_df["hc_total_recommended"].max()) if not headcount_df.empty else 0,
    }

    accuracy_metrics = None
    if accuracy_data and isinstance(accuracy_data, dict):
        accuracy_metrics = accuracy_data.get("metrics")
        if accuracy_metrics:
            accuracy_metrics = dict(accuracy_metrics)
            accuracy_metrics["trend"] = accuracy_data.get("trend", "stable")

    # Compute cost summary if applicable
    cost_summary = None
    total_actual_staffing = config.current_staffing_outbound + config.current_staffing_inbound
    if config.cost_per_hour > 0 and total_actual_staffing > 0 and not headcount_df.empty:
        total_cost_rec = float(headcount_df["daily_cost_recommended"].sum())
        total_cost_act = float(headcount_df["daily_cost_actual"].sum())
        total_savings = total_cost_act - total_cost_rec
        n_days = len(headcount_df)
        cost_summary = {
            "total_cost_recommended": total_cost_rec,
            "total_cost_actual": total_cost_act,
            "total_savings": total_savings,
            "avg_daily_savings": total_savings / n_days if n_days else 0,
        }

    sections = []

    # Section 1: Executive Summary
    sections.append(_write_executive_summary(
        config, alert_summary, accuracy_metrics, headcount_summary, t, locale,
        cost_summary=cost_summary,
    ))

    # Section 2: Daily Forecast
    sections.append(_write_daily_forecast(
        config, forecast_df, figures_dir, t, locale,
    ))

    # Section 3: Backlog Projection
    sections.append(_write_backlog_projection(
        config, backlog_dfs, alert_series_dict, figures_dir, t, locale,
    ))

    # Section 4: Headcount Plan
    sections.append(_write_headcount_plan(
        config, headcount_df, figures_dir, t, locale,
    ))

    # Section 5: Accuracy Report
    sections.append(_write_accuracy_report(
        config, accuracy_data, figures_dir, t, locale,
    ))

    report_content = "\n---\n\n".join(sections)
    report_path = output_dir / "report.md"
    report_path.write_text(report_content, encoding="utf-8")

    return str(report_path)
