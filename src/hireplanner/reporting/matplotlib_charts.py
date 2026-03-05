"""Matplotlib chart generators for warehouse planning reports."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def save_forecast_chart(
    dates,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    title: str,
    path: str | Path,
) -> None:
    """Save a forecast line chart with P10/P50/P90 bands."""
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dates, p10, p90, alpha=0.2, color="#2E75B6", label="P10–P90 range")
    ax.plot(dates, p50, color="#2E75B6", linewidth=2, label="Forecast (P50)")
    ax.plot(dates, p10, color="#BDD7EE", linewidth=1, linestyle="--", label="P10")
    ax.plot(dates, p90, color="#BDD7EE", linewidth=1, linestyle="--", label="P90")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Volume (units)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_backlog_chart(
    dates,
    end_backlog_recommended: np.ndarray,
    end_backlog_actual: np.ndarray | None,
    target_backlog_units: float,
    title: str,
    path: str | Path,
) -> None:
    """Save a backlog area chart with recommended/actual tracks + target line."""
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dates, 0, end_backlog_recommended, alpha=0.3, color="#2E75B6",
                    label="Backlog (Recommended)")
    ax.plot(dates, end_backlog_recommended, color="#2E75B6", linewidth=1.5)

    if end_backlog_actual is not None:
        ax.fill_between(dates, 0, end_backlog_actual, alpha=0.15, color="#ED7D31",
                        label="Backlog (Actual staffing)")
        ax.plot(dates, end_backlog_actual, color="#ED7D31", linewidth=1.5)

    ax.axhline(y=target_backlog_units, color="#C00000", linestyle="--", linewidth=1.5,
               label=f"Target backlog ({target_backlog_units:,.0f})")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Units")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_headcount_chart(
    dates,
    hc_data: dict[str, np.ndarray],
    title: str,
    path: str | Path,
) -> None:
    """Save a grouped bar chart for headcount plan."""
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(12, 5))

    n_series = len(hc_data)
    width = 0.8 / max(n_series, 1)
    x = np.arange(len(dates))
    colors = ["#2E75B6", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#FF6347"]

    for i, (label, values) in enumerate(hc_data.items()):
        offset = (i - n_series / 2 + 0.5) * width
        ax.bar(x + offset, values, width=width, label=label,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Workers")
    ax.set_xticks(x[::7] if len(x) > 14 else x)
    ax.set_xticklabels(
        [d.strftime("%m-%d") for d in dates[::7]] if len(dates) > 14
        else [d.strftime("%m-%d") for d in dates],
        rotation=45,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_accuracy_chart(
    dates,
    forecast: np.ndarray,
    actual: np.ndarray,
    title: str,
    path: str | Path,
) -> None:
    """Save an overlay line chart comparing forecast to actual."""
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, forecast, color="#2E75B6", linewidth=2, label="Forecast")
    ax.plot(dates, actual, color="#ED7D31", linewidth=2, label="Actual")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Volume (units)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
