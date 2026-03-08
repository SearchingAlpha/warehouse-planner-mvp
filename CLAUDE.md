# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HireRobots Warehouse Planner MVP** — a weekly forecasting and labor planning service for 3PL warehouses. Generates 28-day volume forecasts (P10/P50/P90 via LightGBM quantile regression), dual-track backlog projections, headcount recommendations, and accuracy tracking. Output is a Markdown report with PNG charts.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run the planner
hireplanner --config configs/clients/example_client.yaml --data data/example_client.csv

# Run all tests
pytest

# Run tests with coverage
pytest --cov=hireplanner

# Skip slow tests (model training)
pytest -m "not slow"

# Run a single test file
pytest tests/test_planning/test_backlog.py

# Lint and format
ruff check .
ruff format .
```

## Architecture

The pipeline (`src/hireplanner/pipeline/runner.py`) orchestrates a 12-step flow:

**Data → Forecast → Plan → Report**

1. **Config** (`config/`): YAML client config loading + i18n (en/es)
2. **Ingestion** (`ingestion/`): CSV/Excel loader with column auto-detection, data cleaning (gap fill, outlier flagging, calendar features), and validation (≥365 days history required)
3. **Forecasting** (`forecasting/`): LightGBM trains 3 quantile models (P10/P50/P90) per flow with recursive 28-day multi-step prediction. Falls back to naive repeat-last-week if LightGBM fails
4. **Planning** (`planning/`): Backlog engine (`end = max(0, begin - capacity + demand)`), days-of-backlog metric, 3-tier alerts (healthy/watch/critical), headcount calculator with dual tracks (recommended vs actual staffing). Recommended headcount uses shift-rotation patterns (LP solver via scipy) to produce realistic weekly staffing — each rotation gets a constant HC per calendar week, and overlapping rotations create a daily curve. Uncovered days (e.g., weekends) get zero capacity.
5. **Metrics** (`metrics/`): WAPE (primary), MAPE, MAE, RMSE. Accuracy tracker compares previous forecast to new actuals and maintains historical CSV logs
6. **Reporting** (`reporting/`): Markdown report generator (5 sections) + matplotlib PNG charts

## Key Conventions

- Python 3.10+, `from __future__ import annotations` throughout
- ruff for linting/formatting, 100-char line length
- Full type annotations and docstrings with Args/Returns/Raises
- Tests in `tests/` mirror `src/hireplanner/` module structure; shared fixtures in `conftest.py`
- Client configs are YAML files in `configs/clients/`; locale translations in `configs/locales/`
- Output goes to `output/` (gitignored); accuracy logs to `data/accuracy_logs/` (gitignored)
