# HireRobots - Warehouse Forecasting & Labor Planning

Weekly forecasting and labor planning service for 3PL warehouses. Generates 28-day volume forecasts with confidence intervals, backlog projections, headcount recommendations, and accuracy tracking — all delivered as a formatted Excel report.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Client Configuration](#client-configuration)
- [Data Format Requirements](#data-format-requirements)
- [Pipeline Architecture](#pipeline-architecture)
- [Forecasting Model](#forecasting-model)
- [Backlog Model](#backlog-model)
- [Alert System](#alert-system)
- [Headcount Planning](#headcount-planning)
- [Excel Report Structure](#excel-report-structure)
- [Accuracy Tracking](#accuracy-tracking)
- [Localization](#localization)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Overview

HireRobots replaces gut-feel labor planning with data-driven weekly forecasts. The core workflow:

1. Client sends updated volume data (CSV/Excel) weekly
2. Founder runs the pipeline with one command
3. Pipeline generates a 5-tab Excel report with forecasts, backlog projections, headcount plans, and accuracy metrics
4. Founder reviews and emails the report to the client

Target accuracy: **< 10% WAPE** (vs. 20-30% industry average with manual methods).

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd warehouse-planner-mvp

# Install in development mode
pip install -e ".[dev]"
```

**Dependencies:**
- `pandas` >= 2.0
- `numpy` >= 1.24
- `lightgbm` >= 4.0
- `openpyxl` >= 3.1
- `pyyaml` >= 6.0

**Dev dependencies:** `pytest`, `pytest-cov`, `ruff`

## Quick Start

### 1. Create a client configuration

Copy the example config and customize it:

```bash
cp configs/clients/example_client.yaml configs/clients/my_client.yaml
```

Edit the YAML file with your client's parameters (see [Client Configuration](#client-configuration)).

### 2. Prepare historical volume data

Provide a CSV or Excel file with at least 12 months of daily volume data. Required columns:

| Column | Description |
|--------|-------------|
| `date` | Daily date (YYYY-MM-DD) |
| `outbound` | Daily outbound volume (units) |
| `inbound` | Daily inbound volume (units) |

Column names are auto-detected (see [Data Format Requirements](#data-format-requirements)).

### 3. Run the pipeline

```bash
hireplanner --config configs/clients/my_client.yaml --data data/my_client_volumes.csv
```

This generates an Excel report in the `output/` directory.

### 4. Validate data without generating a forecast

```bash
hireplanner --config configs/clients/my_client.yaml --data data/my_client_volumes.csv --dry-run
```

## CLI Reference

```
hireplanner --config <path> --data <path> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to client YAML config file |
| `--data` | *(required)* | Path to volume data CSV/Excel file |
| `--output` | `output/` | Output directory for the Excel report |
| `--log-dir` | `data/accuracy_logs/` | Directory for accuracy logs |
| `--locales-dir` | `configs/locales/` | Directory for locale YAML files |
| `--dry-run` | `false` | Validate data and config only, skip forecast |

**Output:** An Excel file named `{client_name}_{date}.xlsx` in the output directory.

## Client Configuration

Each client has a YAML configuration file. Example (`configs/clients/example_client.yaml`):

```yaml
client_name: "Logista Madrid"

active_flows:
  - outbound
  - inbound

productivity_inbound: 85.0    # units/hour/worker
productivity_outbound: 120.0  # units/hour/worker

hours_per_shift: 8
overhead_buffer: 0.15         # 15% buffer on headcount

backlog_threshold_watch: 1.0    # days of backlog -> Watch status
backlog_threshold_critical: 2.0 # days of backlog -> Critical status

initial_backlog_outbound: 3500  # units at service start
initial_backlog_inbound: 1200   # units at service start

language: "es"          # Report language: "en" or "es"
forecast_horizon: 28    # Days to forecast
```

### Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `client_name` | string | Client name, used in report headers |
| `active_flows` | list | Flows to forecast: `["outbound"]`, `["inbound"]`, or both |
| `productivity_inbound` | float | Units processed per hour per worker (inbound) |
| `productivity_outbound` | float | Units processed per hour per worker (outbound) |
| `hours_per_shift` | int | Standard shift length in hours |
| `overhead_buffer` | float | Headcount buffer (0.15 = 15%) |
| `backlog_threshold_watch` | float | Days of backlog that triggers Watch alert |
| `backlog_threshold_critical` | float | Days of backlog that triggers Critical alert |
| `initial_backlog_outbound` | int | Starting backlog at onboarding (outbound units) |
| `initial_backlog_inbound` | int | Starting backlog at onboarding (inbound units) |
| `language` | string | Report language: `"en"` or `"es"` |
| `forecast_horizon` | int | Number of days to forecast (default: 28) |

## Data Format Requirements

### Input file format

CSV (`.csv`) or Excel (`.xlsx`).

### Required columns

The loader auto-detects column names (case-insensitive). Accepted aliases:

| Concept | Accepted names |
|---------|---------------|
| **Date** | `date`, `ds`, `timestamp`, `fecha` |
| **Outbound** | `outbound`, `ob`, `shipped`, `enviado` |
| **Inbound** | `inbound`, `ib`, `received`, `recibido` |

### Minimum requirements

- **Minimum history:** 365 days (12 months). Recommended: 730+ days (24 months) for seasonal patterns.
- **Missing data:** Must not exceed 10% of total data points.

### Data cleaning (automatic)

The pipeline automatically applies:

1. **Gap filling** — Interpolates short gaps (< 3 days), forward-fills longer ones
2. **Negative clipping** — Clips negative values to 0
3. **Outlier flagging** — Flags values > 3 std devs from rolling median (flagged, not removed)
4. **Calendar features** — Adds `day_of_week`, `week_of_year`, `is_weekend`

## Pipeline Architecture

The pipeline runs 12 steps sequentially:

```
┌──────────────────────────────────────────────────────────┐
│  1. Load client config (YAML)                            │
│  2. Load, clean, and validate data                       │
│  3. Load previous forecast (if exists, for accuracy)     │
│  4. Run LightGBM forecasting per flow                    │
│  5. Calculate backlog projections per flow                │
│  6. Generate threshold alerts (Healthy/Watch/Critical)    │
│  7. Calculate headcount plan                             │
│  8. Compare previous forecast vs new actuals             │
│  9. Append accuracy log                                  │
│ 10. Save current forecast for next week's comparison     │
│ 11. Generate 5-tab Excel report                          │
│ 12. Print summary                                        │
└──────────────────────────────────────────────────────────┘
```

If LightGBM fails for any flow, the pipeline falls back to a naive repeat-last-week forecast with +/- 20% confidence intervals.

## Forecasting Model

### LightGBM Quantile Regression

The forecasting engine uses **LightGBM** with quantile regression to produce probabilistic forecasts.

**Three models are trained per flow:**

| Quantile | Output | Interpretation |
|----------|--------|----------------|
| 0.10 | P10 | Lower bound — 90% chance actual will be above this |
| 0.50 | P50 | Median forecast (point estimate) |
| 0.90 | P90 | Upper bound — 90% chance actual will be below this |

### Feature engineering

**Lag features (7):**
- `lag_1`, `lag_2`, `lag_3` — Recent trend
- `lag_7`, `lag_14`, `lag_21`, `lag_28` — Weekly seasonality

**Rolling statistics (6):**
- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` — Trend smoothing
- `rolling_std_7`, `rolling_std_14`, `rolling_std_28` — Volatility

**Calendar features (5):**
- `day_of_week`, `day_of_month`, `week_of_year`, `month`, `is_weekend`

### Recursive multi-step forecasting

Each forecast step feeds the P50 prediction back as a lag value for the next step. This allows the model to produce 28-day forecasts from a single training pass while maintaining temporal coherence.

### Model parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `num_leaves` | 31 | Maximum leaves per tree |
| `min_child_samples` | 20 | Minimum samples per leaf |

All forecast values are floored at 0 (no negative volumes).

## Backlog Model

The backlog model translates volume forecasts into operational state. This is the core differentiator — it answers "how many days behind are we?"

### Core equation

```
End Workable Backlog = max(0, Beg Workable Backlog - Capacity + New Workable Demand)
```

Where:
- **New Workable Demand** = Forecasted new volume for the day (P50)
- **Beg Workable Backlog** = Previous day's End Workable Backlog
- **Capacity** = `headcount × productivity × hours_per_shift`

### Days of backlog

```
Days of End Workable Backlog = End Backlog / mean(next 7 days capacity)
```

This metric answers: "At current pace, how many days behind are we?"

## Alert System

Three-tier threshold classification based on days of backlog:

| Days of Backlog | Status | Recommended Action |
|-----------------|--------|--------------------|
| < `threshold_watch` (default 1.0) | **Healthy** | Normal operations. Consider reducing temp staff if sustained. |
| `threshold_watch` to `threshold_critical` | **Watch** | Monitor closely. Plan for possible overtime or reinforcement. |
| >= `threshold_critical` (default 2.0) | **Critical** | Immediate action: request reinforcement, authorize overtime. |

Thresholds are configurable per client based on their SLA commitments.

## Headcount Planning

### Formula

```
headcount = ceil(volume / (productivity × hours_per_shift) × (1 + overhead_buffer))
```

- Always rounds up to whole workers
- Separate headcount for inbound and outbound flows
- Overhead buffer defaults to 15% (configurable)

### Output

The headcount plan provides per day:
- Recommended headcount (inbound)
- Recommended headcount (outbound)
- Total headcount

## Excel Report Structure

The output is a 5-tab Excel workbook with formatting and charts.

### Tab 1 — Executive Summary

Key metrics at a glance:
- Forecast accuracy (last week's WAPE)
- Average days of backlog
- Peak headcount needed
- Alert summary (critical/watch/healthy day counts)
- Accuracy trend (improving / stable / degrading)

### Tab 2 — Daily Forecast (28 days)

Per active flow:
- Date, Day of Week
- Forecast Volume: P50 (point estimate), P10 (lower bound), P90 (upper bound)
- Includes an area chart showing forecast with confidence intervals

### Tab 3 — Backlog Projection

Per active flow:
- Date, Beg Workable Backlog, New Demand, Capacity, End Workable Backlog
- Days of End Workable Backlog
- Alert status column with conditional formatting (green/yellow/red)

### Tab 4 — Headcount Plan

- Date, Headcount Outbound, Headcount Inbound, Total Headcount
- Includes a bar chart of daily headcount

### Tab 5 — Accuracy Report

- Forecast vs Actual comparison for the last measured period
- WAPE, MAPE, MAE metrics
- Accuracy trend (last 4 weeks if available)
- Chart: forecast vs actual overlay

## Accuracy Tracking

### How it works

1. Each week's forecast is saved to `data/accuracy_logs/`
2. On the next run, the previous forecast is loaded and compared against the new actual data
3. Metrics are calculated (WAPE, MAPE, MAE) and appended to a historical log
4. The trend is determined: **improving**, **stable**, or **degrading** based on the last 4 weeks

### Metrics

| Metric | Formula | Primary? |
|--------|---------|----------|
| **WAPE** | sum(\|actual - forecast\|) / sum(actual) | Yes |
| **MAPE** | mean(\|actual - forecast\| / actual) | No |
| **MAE** | mean(\|actual - forecast\|) | No |
| **RMSE** | sqrt(mean((actual - forecast)^2)) | No |

### File structure

```
data/accuracy_logs/
├── logista_madrid_forecast.csv       # Latest forecast snapshot
└── logista_madrid_accuracy_log.csv   # Historical accuracy log
```

## Localization

Reports support English (`en`) and Spanish (`es`). All labels, tab names, headers, and status strings are translated.

Locale files are stored in `configs/locales/`:

```
configs/locales/
├── en.yaml
└── es.yaml
```

Set the language in the client config:

```yaml
language: "es"  # or "en"
```

The translation system uses dot-notation keys (e.g., `tabs.executive_summary` resolves to "Resumen Ejecutivo" in Spanish). Falls back to English if a key is missing.

## Testing

Run the full test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=hireplanner
```

Skip slow tests (model training):

```bash
pytest -m "not slow"
```

The suite contains **161 tests** covering:

| Module | Coverage |
|--------|----------|
| `config/` | Client config loading, validation, i18n |
| `ingestion/` | Data loading, cleaning, validation |
| `forecasting/` | LightGBM features, forecasts, ensemble logic |
| `planning/` | Backlog calculation, alerts, labor/headcount |
| `metrics/` | WAPE, MAPE, MAE, RMSE, accuracy tracking |
| `reporting/` | Excel generation, formatters, charts |
| `pipeline/` | End-to-end pipeline runner |

## Project Structure

```
warehouse-planner-mvp/
├── configs/
│   ├── clients/
│   │   └── example_client.yaml      # Example client configuration
│   └── locales/
│       ├── en.yaml                   # English translations
│       └── es.yaml                   # Spanish translations
├── src/
│   └── hireplanner/
│       ├── config/
│       │   ├── client_config.py      # ClientConfig dataclass + YAML loader
│       │   └── i18n.py               # Multi-language support
│       ├── ingestion/
│       │   ├── loader.py             # CSV/Excel loading with auto-detect
│       │   ├── cleaner.py            # Data cleaning pipeline
│       │   └── validator.py          # Data validation rules
│       ├── forecasting/
│       │   ├── lightgbm_model.py     # LightGBM quantile regression forecaster
│       │   └── ensemble.py           # Model selection/blending utilities
│       ├── planning/
│       │   ├── backlog.py            # Backlog calculation engine
│       │   ├── alerts.py             # Threshold-based alert system
│       │   └── labor.py              # Headcount calculator
│       ├── metrics/
│       │   ├── evaluation.py         # WAPE, MAPE, MAE, RMSE
│       │   └── accuracy_tracker.py   # Forecast vs actual tracking + logging
│       ├── reporting/
│       │   ├── excel_generator.py    # 5-tab Excel report builder
│       │   ├── formatters.py         # openpyxl styles (headers, alerts, colors)
│       │   └── charts.py            # Line/Area/Bar chart builders
│       └── pipeline/
│           └── runner.py             # Main orchestrator + CLI entry point
├── tests/                            # 161 unit tests (pytest)
├── pyproject.toml                    # Project metadata and dependencies
├── prd.md                            # Product Requirements Document
└── PLAN.md                           # Implementation plan
```
