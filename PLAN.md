# HireRobots MVP -- Implementation Plan

## 0. Current State Assessment

The repository currently contains only `prd.md`. The PRD references components marked as "already built" (data ingestion, LightGBM forecasting, evaluation metrics, basic labor calculator, AI agent, unit tests). These exist in a separate prototype environment and must be ported into this repository as part of the initial scaffolding step before the 4-week build plan begins.

**Assumption:** The existing code will be brought in from the prototype and placed into the project structure defined below. If the prototype code does not exist as standalone modules, the plan includes building those components from scratch in Week 0 (pre-work).

---

## 1. Project Structure

```
warehouse-planner-mvp/
|-- prd.md
|-- PLAN.md
|-- README.md
|-- pyproject.toml                     # Project metadata, dependencies, scripts
|-- .gitignore
|-- configs/
|   |-- clients/
|   |   |-- _template.yaml             # Client config template
|   |   |-- example_client.yaml        # Example/demo client config
|   |-- locales/
|       |-- en.yaml                    # English labels
|       |-- es.yaml                    # Spanish labels
|-- src/
|   |-- hireplanner/
|   |   |-- __init__.py
|   |   |-- ingestion/
|   |   |   |-- __init__.py
|   |   |   |-- loader.py             # CSV/Excel loading + auto-detect columns
|   |   |   |-- cleaner.py            # Gap-fill, clip negatives, outlier flags
|   |   |   |-- validator.py          # Min history, missing data checks
|   |   |-- forecasting/
|   |   |   |-- __init__.py
|   |   |   |-- lightgbm_model.py     # LightGBM quantile regression forecaster
|   |   |   |-- ensemble.py           # Model selection / blending logic
|   |   |-- metrics/
|   |   |   |-- __init__.py
|   |   |   |-- evaluation.py         # WAPE, MAPE, MAE, RMSE
|   |   |   |-- accuracy_tracker.py   # Forecast vs actual tracking [NEW]
|   |   |-- planning/
|   |   |   |-- __init__.py
|   |   |   |-- labor.py              # Existing labor calculator
|   |   |   |-- backlog.py            # Backlog engine [NEW]
|   |   |   |-- alerts.py             # Threshold alerts [NEW]
|   |   |-- config/
|   |   |   |-- __init__.py
|   |   |   |-- client_config.py      # YAML loader + validation [NEW]
|   |   |   |-- i18n.py               # Multi-language label resolution [NEW]
|   |   |-- reporting/
|   |   |   |-- __init__.py
|   |   |   |-- excel_generator.py    # 5-tab Excel workbook [NEW]
|   |   |   |-- charts.py             # Chart builders for openpyxl [NEW]
|   |   |   |-- formatters.py         # Cell styles, conditional formatting [NEW]
|   |   |-- pipeline/
|   |   |   |-- __init__.py
|   |   |   |-- runner.py             # Semi-auto pipeline orchestrator [NEW]
|   |   |-- agent/
|   |       |-- __init__.py
|   |       |-- agent.py              # GPT-4o internal agent (existing)
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                    # Shared fixtures
|   |-- test_ingestion/
|   |   |-- __init__.py
|   |   |-- test_loader.py
|   |   |-- test_cleaner.py
|   |   |-- test_validator.py
|   |-- test_forecasting/
|   |   |-- __init__.py
|   |   |-- test_lightgbm.py
|   |   |-- test_ensemble.py
|   |-- test_metrics/
|   |   |-- __init__.py
|   |   |-- test_evaluation.py
|   |   |-- test_accuracy_tracker.py   # [NEW]
|   |-- test_planning/
|   |   |-- __init__.py
|   |   |-- test_labor.py
|   |   |-- test_backlog.py            # [NEW]
|   |   |-- test_alerts.py             # [NEW]
|   |-- test_config/
|   |   |-- __init__.py
|   |   |-- test_client_config.py      # [NEW]
|   |   |-- test_i18n.py              # [NEW]
|   |-- test_reporting/
|   |   |-- __init__.py
|   |   |-- test_excel_generator.py    # [NEW]
|   |-- test_pipeline/
|   |   |-- __init__.py
|   |   |-- test_runner.py             # [NEW]
|   |-- fixtures/
|       |-- sample_data.csv
|       |-- example_client.yaml
|-- data/
|   |-- sample/                        # Synthetic sample data for testing
|   |-- accuracy_logs/                 # Historical accuracy tracking (gitignored)
|-- output/                            # Generated reports (gitignored)
```

---

## 2. Dependencies (`pyproject.toml`)

```toml
[project]
name = "hireplanner"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "openpyxl>=3.1",
    "pyyaml>=6.0",
    "lightgbm>=4.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "ruff"]

[project.scripts]
hireplanner = "hireplanner.pipeline.runner:main"
```

---

## 3. Week-by-Week Implementation Plan

### WEEK 0: Scaffolding and Porting (Pre-work, 1-2 days)

**Goal:** Set up the project structure, port existing code into the new layout, verify all 46 existing tests pass.

| # | Task | Files | Effort |
|---|------|-------|--------|
| 0.1 | Create project structure (directories, `__init__.py` files, `pyproject.toml`, `.gitignore`) | All scaffolding files | 1h |
| 0.2 | Port existing ingestion code into `src/hireplanner/ingestion/` (loader, cleaner, validator) | `ingestion/loader.py`, `cleaner.py`, `validator.py` | 2h |
| 0.3 | Port existing forecasting code into `src/hireplanner/forecasting/` | `forecasting/lightgbm_model.py` | 2h |
| 0.4 | Port existing evaluation metrics into `src/hireplanner/metrics/evaluation.py` | `metrics/evaluation.py` | 1h |
| 0.5 | Port existing labor calculator into `src/hireplanner/planning/labor.py` | `planning/labor.py` | 1h |
| 0.6 | Port existing agent code into `src/hireplanner/agent/` | `agent/agent.py` | 1h |
| 0.7 | Port existing tests, adapt imports, verify all 46 pass | `tests/` | 2h |
| 0.8 | Create synthetic sample data for development/testing | `data/sample/`, `tests/fixtures/` | 1h |

**Exit Criteria:** `pytest` runs, all 46 ported tests pass, project installs with `pip install -e .`

---

### WEEK 1: Backlog Engine + Client Config

**Goal:** Backlog calculation working end-to-end on synthetic data with configurable client parameters.

#### 1A. Client Config System (`configs/` + `src/hireplanner/config/`)

| # | Task | Details |
|---|------|---------|
| 1.1 | Define client config YAML schema | Create `configs/clients/_template.yaml` with all parameters from PRD Section 7: `client_name`, `active_flows`, `productivity_inbound`, `productivity_outbound`, `hours_per_shift`, `overhead_buffer`, `backlog_threshold_watch`, `backlog_threshold_critical`, `initial_backlog_outbound`, `initial_backlog_inbound`, `language`, `forecast_horizon` |
| 1.2 | Implement `client_config.py` | `load_client_config(path: str) -> ClientConfig` function that reads YAML and returns a dataclass. Validate all required fields present, correct types, sensible ranges (e.g., productivity > 0, overhead_buffer 0-1, language in ["es","en"]). Raise `ConfigError` with clear messages on validation failure. Provide defaults for optional fields (hours_per_shift=8, overhead_buffer=0.15, forecast_horizon=28). |
| 1.3 | Create `ClientConfig` dataclass | Define in `config/client_config.py`. Use `@dataclass` with type hints. Include a `validate()` method. |
| 1.4 | Create example client config | `configs/clients/example_client.yaml` with realistic values for a Spanish 3PL warehouse |
| 1.5 | Write tests for config system | `tests/test_config/test_client_config.py`: test loading valid config, missing required fields, invalid types, default population, edge cases (zero productivity, negative values). Target: 8-10 tests. |

**Key Design Decisions:**
- Use Python `dataclass` for `ClientConfig` (not dict) for type safety and IDE support
- Config validation happens at load time, fail fast with clear messages
- Path to config file is the primary parameter to the pipeline

#### 1B. Backlog Calculation Engine (`src/hireplanner/planning/backlog.py`)

| # | Task | Details |
|---|------|---------|
| 1.6 | Implement core backlog equation | Function `calculate_daily_backlog(forecast_demand: pd.Series, capacity_per_day: pd.Series, initial_backlog: float) -> pd.DataFrame` that returns columns: `date`, `beg_backlog`, `new_demand`, `capacity`, `end_backlog`. Core equation: `end_backlog = max(0, beg_backlog - capacity + new_demand)`. Iterate day-by-day since each day depends on previous day's end_backlog. |
| 1.7 | Implement capacity calculation | Function `calculate_daily_capacity(headcount: pd.Series, productivity: float, hours_per_shift: float) -> pd.Series`. This converts headcount plan to units processed per day. Also implement the reverse: `headcount_from_forecast(forecast: pd.Series, productivity: float, hours_per_shift: float, overhead_buffer: float) -> pd.Series` (this may already exist in labor.py -- integrate or delegate). |
| 1.8 | Implement days-of-backlog metric | Function `calculate_days_of_backlog(end_backlog: pd.Series, capacity: pd.Series, window: int = 7) -> pd.Series`. For each day: `end_backlog[day] / mean(capacity[day+1:day+8])`. Handle edge case at end of forecast horizon (use available remaining days). |
| 1.9 | Implement per-flow backlog calculation | Function `calculate_flow_backlog(config: ClientConfig, forecast_df: pd.DataFrame, flow: str) -> pd.DataFrame` that orchestrates 1.6-1.8 for a single flow (inbound or outbound), pulling the right productivity rate and initial backlog from config. |
| 1.10 | Integration: multi-flow backlog | Function `calculate_all_backlogs(config: ClientConfig, forecast_df: pd.DataFrame) -> dict[str, pd.DataFrame]` that runs backlog calculation for each active flow in `config.active_flows`. Returns `{"outbound": df_out, "inbound": df_in}`. |
| 1.11 | Write backlog tests | `tests/test_planning/test_backlog.py`: test basic equation with known inputs/outputs (hand-calculated), backlog never goes negative, days-of-backlog calculation, edge cases (zero demand, zero capacity, initial backlog = 0), multi-day propagation. Target: 12-15 tests. |

#### 1C. Backlog Threshold Alerts (`src/hireplanner/planning/alerts.py`)

| # | Task | Details |
|---|------|---------|
| 1.12 | Implement threshold classification | Function `classify_backlog_status(days_of_backlog: float, threshold_watch: float, threshold_critical: float) -> str` returns "Healthy", "Watch", or "Critical". |
| 1.13 | Implement alert series generation | Function `generate_alert_series(days_of_backlog: pd.Series, config: ClientConfig) -> pd.Series` that applies classification to every day. Returns Series of status strings. |
| 1.14 | Implement alert summary | Function `summarize_alerts(alert_series: pd.Series) -> dict` returning `{"critical_days": int, "watch_days": int, "healthy_days": int, "first_critical_date": date_or_None, "peak_days_of_backlog": float}`. This feeds the Executive Summary tab. |
| 1.15 | Write alert tests | `tests/test_planning/test_alerts.py`: test boundary conditions (exactly at threshold), all-healthy scenario, all-critical scenario, mixed. Target: 6-8 tests. |

**Week 1 Exit Criteria:**
- `ClientConfig` loads and validates YAML configs
- Backlog engine produces correct 28-day projections on synthetic data
- Alerts classify correctly
- ~25 new tests added, all passing

---

### WEEK 2: Excel Report Generator

**Goal:** Generate a complete 5-tab Excel workbook with formatting and charts.

#### 2A. Reporting Infrastructure

| # | Task | Details |
|---|------|---------|
| 2.1 | Implement `formatters.py` | Define reusable openpyxl styles: `HEADER_STYLE` (bold, blue background, white text, border), `NUMBER_FORMAT` ("#,##0" for integers, "#,##0.0%" for percentages), `DATE_FORMAT` ("YYYY-MM-DD"), `ALERT_FORMATS` (green/yellow/red fill), `TITLE_STYLE` (large font, bold), auto-column-width helper function |
| 2.2 | Implement `charts.py` | Chart builder functions using openpyxl.chart: `create_forecast_chart` (line chart with P10/P50/P90), `create_backlog_chart` (area chart with threshold lines), `create_headcount_chart` (bar chart), `create_accuracy_chart` (overlay line chart) |

#### 2B. Tab Implementations (`reporting/excel_generator.py`)

| # | Task | Details |
|---|------|---------|
| 2.3 | Implement Tab 1 -- Executive Summary | Function `write_executive_summary(wb, config, alert_summary, accuracy_metrics, headcount_summary)`. Content: client name + report date header, key metrics table, critical alerts list, week-over-week trend indicator. |
| 2.4 | Implement Tab 2 -- Daily Forecast | Function `write_daily_forecast(wb, config, forecast_df)`. Columns: Date, Day of Week, Forecast (P50), Lower (P10), Upper (P90). Per active flow. Include forecast chart. |
| 2.5 | Implement Tab 3 -- Backlog Projection | Function `write_backlog_projection(wb, config, backlog_dfs, alert_series_dict)`. Columns: Date, Beg Backlog, New Demand, Capacity, End Backlog, Days of Backlog, Alert Status. Conditional formatting on Alert Status column. Include backlog chart. Per active flow. |
| 2.6 | Implement Tab 4 -- Headcount Plan | Function `write_headcount_plan(wb, config, headcount_df)`. Columns: Date, HC Inbound, HC Outbound, Total HC. Optional: comparison vs current staffing. Weekly aggregation summary rows. |
| 2.7 | Implement Tab 5 -- Accuracy Report | Function `write_accuracy_report(wb, config, accuracy_data)`. Columns: Date, Forecast, Actual, Error. Metrics summary: WAPE, MAPE, MAE. Trend table. Handle first-run case gracefully. |
| 2.8 | Implement main generator | Function `generate_excel_report(config, forecast_df, backlog_dfs, headcount_df, accuracy_data, output_path) -> str`. Orchestrates all 5 tabs, saves workbook, returns path. |
| 2.9 | Write reporting tests | `tests/test_reporting/test_excel_generator.py`: workbook has 5 tabs, each tab populates expected columns, conditional formatting applied, file saves as valid xlsx, handles missing accuracy data, handles single flow. Target: 10-12 tests. |

**Key Design Decisions:**
- Each tab is a separate function for testability and maintainability
- The generator accepts pre-computed DataFrames (separation of concerns)
- Tab names will be resolved through the i18n system (Week 3), hardcoded in English for Week 2

**Week 2 Exit Criteria:**
- `generate_excel_report()` produces a valid 5-tab `.xlsx` file
- Charts render correctly in Excel
- Conditional formatting works on alert columns
- ~10-12 new tests added, all passing

---

### WEEK 3: Accuracy Tracking + i18n + Integration Pipeline

**Goal:** Full pipeline working end-to-end with one command, multi-language support.

#### 3A. Accuracy Tracking (`src/hireplanner/metrics/accuracy_tracker.py`)

| # | Task | Details |
|---|------|---------|
| 3.1 | Implement forecast vs actual comparison | Function `compare_forecast_to_actual(forecast_df, actual_df) -> pd.DataFrame`. Join on date, compute per-day error. |
| 3.2 | Implement rolling accuracy metrics | Function `calculate_accuracy_metrics(comparison_df) -> dict` returning `{"wape", "mape", "mae", "period_start", "period_end"}`. Uses existing `metrics/evaluation.py`. |
| 3.3 | Implement historical accuracy log | Functions `append_accuracy_log(client_name, metrics, log_dir)` and `load_accuracy_log(client_name, log_dir) -> pd.DataFrame`. Appends to CSV, creates file if first run. |
| 3.4 | Implement accuracy degradation flag | Function `check_accuracy_degradation(current_wape, threshold=0.15) -> bool`. Function `get_accuracy_trend(log_df, last_n=4) -> str` returning "improving", "stable", or "degrading". |
| 3.5 | Implement previous forecast storage | Functions `save_forecast(client_name, forecast_df, run_date, log_dir)` and `load_previous_forecast(client_name, log_dir) -> Optional[pd.DataFrame]`. |
| 3.6 | Write accuracy tracking tests | Target: 8-10 tests. |

#### 3B. Multi-Language Support (i18n)

| # | Task | Details |
|---|------|---------|
| 3.7 | Define locale YAML files | `configs/locales/en.yaml` and `configs/locales/es.yaml`. Structure: tabs, headers, alerts, labels. |
| 3.8 | Implement `i18n.py` | Functions `load_locale(language, locales_dir) -> dict` and `t(key, locale) -> str` for dot-notation translation lookup with English fallback. |
| 3.9 | Integrate i18n into Excel generator | Update all tab-writing functions to accept a `locale` parameter. Replace hardcoded strings with `t()` calls. |
| 3.10 | Write i18n tests | Target: 6-8 tests. |

#### 3C. Semi-Auto Pipeline (`src/hireplanner/pipeline/runner.py`)

| # | Task | Details |
|---|------|---------|
| 3.11 | Implement pipeline orchestrator | Main function `run_pipeline(client_config_path, data_path, output_dir)` executing: load config → load/clean/validate data → load previous forecast → run forecasting → calculate backlog → generate alerts → calculate headcount → compare accuracy → append log → save forecast → generate Excel → print summary. |
| 3.12 | Implement CLI entry point | `main()` function using `argparse`: `hireplanner --config ... --data ... --output ...`. Optional flags: `--no-forecast`, `--dry-run`. |
| 3.13 | Implement console summary output | Print: client name, forecast dates, accuracy metrics, critical alert count, output file path. |
| 3.14 | Write pipeline integration tests | Target: 5-7 tests with mocked forecasting. |

**Week 3 Exit Criteria:**
- `hireplanner --config ... --data ...` produces a complete Excel report
- Report is in the configured language (ES or EN)
- Accuracy tracking works across runs
- ~20 new tests, all passing

---

### WEEK 4: Testing, Polish, and Real Data Validation

**Goal:** MVP alpha-ready. Can deliver a weekly forecast to a real warehouse.

| # | Task | Details |
|---|------|---------|
| 4.1 | End-to-end test with realistic data | Create or obtain realistic 3PL volume data (18-24 months). Run full pipeline. Verify all tabs make operational sense. |
| 4.2 | Excel visual polish | Review generated Excel in actual Excel/LibreOffice. Adjust column widths, chart sizes, color schemes, number formats. |
| 4.3 | Error handling hardening | Add try/except blocks in pipeline runner with user-friendly error messages. Handle: missing files, corrupt data, model failures, disk space. |
| 4.4 | Edge case testing | Test: outbound-only client, exactly 12 months history, data with many outliers, zero initial backlog, very high initial backlog, all weekend data missing. |
| 4.5 | Performance check | Target: < 5 minutes per client on a standard laptop. |
| 4.6 | README and documentation | Installation instructions, quick start, client config reference, example usage. |
| 4.7 | Create demo/example run | Script or instructions to run pipeline on sample data. |
| 4.8 | Final test suite review | Total target: 80+ tests (46 existing + ~35 new). Fix any flaky tests. |

**Week 4 Exit Criteria:**
- Pipeline runs successfully on realistic data
- Excel report looks professional
- Error messages are helpful
- README enables someone new to set up and run
- All tests pass, 80+ total

---

## 4. Dependency Graph

```
Week 0: Scaffolding
   |
   v
Week 1: Client Config ──────────────> Week 1: Backlog Engine
         (1.1-1.5)                           (1.6-1.11)
              |                                   |
              |                    Week 1: Alerts (1.12-1.15)
              |                         |
              v                         v
         Week 2: Excel Report Generator (2.1-2.9)
              |           |              |
              v           v              v
    Week 3: i18n    Week 3: Accuracy   Week 3: Pipeline
    (3.7-3.10)      Tracker (3.1-3.6)  Runner (3.11-3.14)
              |           |              |
              v           v              v
         Week 4: Integration + Polish (4.1-4.8)
```

**Critical path:** Config → Backlog → Alerts → Excel → Pipeline → Polish

**Parallelizable:**
- Within Week 1: Config and Backlog can be developed in parallel (define interface first)
- Within Week 3: i18n and Accuracy Tracker are independent; both feed into Pipeline integration

---

## 5. Module Interface Contracts

### ClientConfig dataclass

```python
@dataclass
class ClientConfig:
    client_name: str
    active_flows: list[str]           # ["outbound"], ["inbound"], or ["outbound", "inbound"]
    productivity_inbound: float       # units/hour
    productivity_outbound: float      # units/hour
    hours_per_shift: int              # default 8
    overhead_buffer: float            # default 0.15
    backlog_threshold_watch: float    # default 1.0 (days)
    backlog_threshold_critical: float # default 2.0 (days)
    initial_backlog_outbound: int     # units
    initial_backlog_inbound: int      # units
    language: str                     # "es" or "en"
    forecast_horizon: int             # default 28
```

### Backlog Engine Output (per flow)

```python
# DataFrame columns:
# date: datetime, beg_backlog: float, new_demand: float,
# capacity: float, end_backlog: float, days_of_backlog: float,
# alert_status: str ("Healthy" | "Watch" | "Critical")
```

### Forecast DataFrame Convention

```python
# DataFrame columns:
# date: datetime, forecast_p50: float, forecast_p10: float,
# forecast_p90: float, flow: str ("outbound" | "inbound")
```

### Pipeline Runner Signature

```python
def run_pipeline(
    client_config_path: str,
    data_path: str,
    output_dir: str = "output/",
    log_dir: str = "data/accuracy_logs/",
    locales_dir: str = "configs/locales/"
) -> str:  # returns path to generated Excel file
```

---

## 6. Testing Strategy

| Layer | Module | Test Type | Mock/Real |
|-------|--------|-----------|-----------|
| Config | `client_config.py` | Unit | Real (YAML fixtures) |
| Config | `i18n.py` | Unit | Real (YAML fixtures) |
| Planning | `backlog.py` | Unit | Real (hand-calculated expected values) |
| Planning | `alerts.py` | Unit | Real |
| Metrics | `accuracy_tracker.py` | Unit | Real (synthetic DataFrames) |
| Reporting | `excel_generator.py` | Unit + Integration | Real (verify file output, check cell values via openpyxl read-back) |
| Pipeline | `runner.py` | Integration | Mock forecasting models (avoid model download in CI) |
| E2E | Full pipeline | E2E (manual + automated) | Mock forecasting for CI; real models for local validation |

**Test fixtures** (`tests/conftest.py`):
- `sample_config` — returns a valid `ClientConfig` with typical values
- `sample_forecast_df` — 28-day synthetic forecast DataFrame
- `sample_actuals_df` — 28-day synthetic actual data
- `sample_backlog_df` — pre-computed backlog DataFrame
- `tmp_output_dir` — temporary directory for Excel file output (pytest `tmp_path`)

---

## 7. Risk Mitigations

| Risk | Mitigation |
|------|------------|
| LightGBM training on short history | Minimum 60 days of data required for feature warm-up; falls back to naive forecast on failure |
| openpyxl chart rendering differences across Excel versions | Test cell values, not visual rendering; manual visual QA in Week 4 |
| Backlog equation edge cases (negative values, overflow) | Clamp `end_backlog` to `max(0, ...)` always; test with extreme values |
| Client config drift (YAML fields added later) | Use dataclass with defaults; validate at load time; version field in YAML for future migration |
| First-run accuracy (no previous forecast exists) | Handle gracefully: Tab 5 shows "No historical data yet" message; pipeline does not crash |
