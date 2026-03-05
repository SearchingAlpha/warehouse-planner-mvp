_Engineering Operational Truth_: where there is mystery there is margin

**Forecasting & Labor Planning for 3PL Warehouses**

Version 1.0 | February 2026 Status: Draft | Confidential

---

## 1. Product Vision

### 1.1 Problem Statement

3PL warehouses plan labor using gut feeling, basic Excel, or last year's data. This leads to chronic over- or understaffing that costs mid-size 3PLs an estimated €100–300K/year in avoidable labor costs: wasted shifts on slow days, overtime premiums on peak days, missed SLAs, and client churn.

### 1.2 Solution

HireRobots delivers a weekly forecasting and labor planning service. Each week, the ops manager receives an email with an Excel report containing:

- 28-day volume forecast (inbound and/or outbound) with confidence intervals
- Projected backlog evolution and days-of-backlog alerts
- Recommended daily headcount based on real productivity rates
- Accuracy tracking vs. previous forecasts

Target accuracy: <10% WAPE (vs. 20–30% industry average with manual methods), representing a 4–6x improvement in forecast precision.

### 1.3 MVP Scope

**What this MVP is:** A semi-automated weekly service where the founder runs forecasts, reviews the output, and emails the Excel report to the client. Manual delivery, maximum learning.

**What this MVP is NOT:** A self-service SaaS platform. No user accounts, no web dashboard, no client-facing UI. The AI agent is an internal tool for the founder, not exposed to clients.

---

## 2. Target User & Buyer

### 2.1 The User (feels the pain)

|Attribute|Detail|
|---|---|
|**Title**|Operations Manager / Site Manager|
|**Company**|3PL provider in Spain, ≥20K units/week outbound, ~50+ employees|
|**Daily reality**|Plans staffing based on gut feel; spends weekends in Excel guessing Monday's headcount; firefighting mode instead of planning mode|
|**Core need**|Fewer surprises on Monday morning. Confidence in shift planning. Proof of competence to GM and end clients.|

### 2.2 The Buyer (signs the check)

**⚠️ Hypothesis to validate:** Does the ops manager have authority to approve a €3K/month tool, or does the decision escalate to GM / director / owner? This determines whether we sell directly or need to arm our champion with an ROI case.

_Discovery call question: "If you found a tool that helped with this, what would the process look like to bring it in? Who else would need to be involved?"_

---

## 3. Core Metrics Model

### 3.1 The Backlog Framework

The backlog model is HireRobots' core differentiator. It translates abstract volume forecasts into operational decisions. The fundamental equation:

> **End Workable Backlog = Beg Workable Backlog – Capacity + New Workable Demand**

#### Definitions

|Term|Definition|
|---|---|
|**New Workable Demand**|Volume of new units entering the pipeline each day (forecasted)|
|**Beg Workable Backlog**|Backlog at start of day = End Workable Backlog of previous day|
|**End Workable Backlog**|Backlog at end of day after processing|
|**Capacity**|Shipments (outbound) or Receipts (inbound) processed that day, driven by headcount × productivity|

#### Days of Backlog (Alert Metric)

> **Days of End Workable Backlog = End Workable Backlog / Avg Next 7 Days Capacity**

This metric answers the critical question: "At current pace, how many days behind are we?" When this exceeds a defined threshold (based on the client's SLAs), an alert is triggered in the report.

#### Threshold Logic

|Days of Backlog|Status|Recommended Action|
|---|---|---|
|< 1.0 day|✅ **Healthy**|Normal operations. Consider reducing temp staff if sustained.|
|1.0 – 2.0 days|⚠️ **Watch**|Monitor closely. Plan for possible overtime or temp reinforcement.|
|> 2.0 days|🛑 **Critical**|Immediate action required: request ETT reinforcement, authorize overtime.|

**Note:** Default thresholds above are starting values. HireRobots will propose specific thresholds per client based on their SLA commitments during onboarding. These thresholds are configurable.

### 3.2 Dual Flow Support

The model supports both outbound (shipments) and inbound (receipts) flows independently. Each flow has its own:

- Volume forecast (with confidence intervals)
- Backlog calculation
- Headcount recommendation
- Days-of-backlog alerts

For the MVP, the client chooses which flows to activate. Outbound is expected to be the priority (direct client impact), but inbound is available from day one.

---

## 4. Functional Requirements

### 4.1 Data Ingestion

#### R1: Manual Data Upload

|Attribute|Requirement|
|---|---|
|**Input format**|CSV or Excel (.xlsx) uploaded manually by the client (email or shared drive)|
|**Required columns**|Date (daily granularity), Volume (inbound and/or outbound units)|
|**Minimum history**|12 months (365 days). Recommended: 24+ months for seasonal patterns.|
|**Update frequency**|Weekly. Client sends refreshed data each week before forecast generation.|
|**Column detection**|Auto-detect common column names (date/ds/timestamp, inbound/IB/received, outbound/OB/shipped). Case-insensitive.|

#### R2: Data Cleaning & Validation

- Fill date gaps (interpolate short gaps <3 days, forward-fill longer ones)
- Clip negative values to 0
- Flag outliers (>3 std devs from rolling median) — flagged, not removed
- Add calendar features: day_of_week, week_of_year, is_weekend
- Reject datasets with >10% missing data or <12 months of history
- Provide clear error messages if validation fails

### 4.2 Forecasting Engine

#### R3: Forecast Generation

|Attribute|Requirement|
|---|---|
|**Horizon**|28 days (4 weeks)|
|**Granularity**|Daily|
|**Models**|Hybrid: Chronos-2 (foundation model, zero-shot) + Prophet (classical baseline). System selects best or blends.|
|**Outputs per day**|Point forecast (median), Lower bound (P10), Upper bound (P90)|
|**Flows**|Inbound and outbound forecasted independently|
|**Target accuracy**|WAPE < 10% (measured on rolling 4-week basis)|

### 4.3 Backlog & Labor Planning

#### R4: Backlog Projection

- Calculate daily End Workable Backlog using the backlog equation for each active flow
- Compute Days of End Workable Backlog = End Backlog / Avg Next 7 Days Capacity
- Apply threshold-based alerts (Healthy / Watch / Critical) per client SLA configuration
- Project backlog for all 28 forecast days
- Initial backlog (day 0) provided by client at onboarding, then calculated automatically from previous day

#### R5: Headcount Recommendation

- Calculate daily headcount = Forecast Volume / (Productivity Rate × Hours per Shift)
- Apply overhead buffer (default 15%, configurable per client)
- Separate headcount for inbound and outbound flows
- Productivity rate (units/hour) provided by client during onboarding
- Round up to whole workers (cannot assign 0.3 of a person)

### 4.4 Accuracy Tracking

#### R6: Forecast vs. Actual Comparison

- Each week, compare previous week's forecast against actual volumes received in the data refresh
- Calculate WAPE (primary), MAPE, and MAE for the measured period
- Include accuracy metrics in the weekly report (dedicated section)
- Maintain historical accuracy log per client for trend analysis
- Flag if accuracy degrades below 15% WAPE threshold

### 4.5 Report Output

#### R7: Weekly Excel Report

The primary deliverable is an Excel workbook (.xlsx) sent as an email attachment. The Excel must contain:

**Tab 1 — Executive Summary**

- Key metrics at a glance: forecast accuracy (last week), average days of backlog, peak headcount needed
- Alerts: any days where backlog exceeds critical threshold
- Week-over-week trend (improving / stable / degrading)

**Tab 2 — Daily Forecast (28 days)**

- Date, Day of Week
- Forecast Volume: Point estimate, Lower bound (P10), Upper bound (P90)
- Per active flow (outbound and/or inbound)

**Tab 3 — Backlog Projection**

- Date, Beg Workable Backlog, New Workable Demand (forecast), Capacity (recommended), End Workable Backlog
- Days of End Workable Backlog
- Alert status column (Healthy / Watch / Critical) with conditional formatting
- Per active flow

**Tab 4 — Headcount Plan**

- Date, Recommended Headcount (inbound), Recommended Headcount (outbound), Total Headcount
- Comparison vs. current staffing level (if provided by client)
- Weekly aggregation (total hours, average daily headcount)

**Tab 5 — Accuracy Report**

- Forecast vs. Actual for the last measured period
- WAPE, MAPE, MAE metrics
- Accuracy trend (last 4 weeks if available)
- Chart: forecast vs. actual overlay

#### R8: Email Delivery

- Founder sends email manually after reviewing the Excel
- Email body includes: brief summary (2–3 key takeaways), any critical alerts, Excel attached
- Language: configurable per client (Spanish or English)
- Excel labels, headers, and tab names in client's language

---

## 5. Client Onboarding Requirements

Before generating the first forecast, the following data and parameters must be collected from the client:

|#|Data / Parameter|Description|Format|
|---|---|---|---|
|1|**Historical volume data**|Daily inbound and/or outbound volumes, minimum 12 months|CSV or Excel|
|2|**Initial backlog**|Current workable backlog at start of service (units)|Single number per flow|
|3|**Productivity rate**|Units processed per hour per worker (inbound and outbound separately)|Units/hour|
|4|**Hours per shift**|Standard shift length|Hours (default: 8)|
|5|**SLA commitments**|Delivery timeframes to clients (used to set backlog thresholds)|Days / description|
|6|**Active flows**|Which flows to forecast: outbound only, inbound only, or both|Selection|
|7|**Report language**|Preferred language for the Excel report|Spanish or English|
|8|**Current staffing levels**|Average daily headcount per flow (optional, for comparison)|Workers/day|

---

## 6. Weekly Operating Process

The MVP operates on a weekly cycle with semi-automated delivery:

|Step|When|Action|Owner|
|---|---|---|---|
|1|Sunday/Monday|Client sends updated volume data (last week actuals)|Client|
|2|Monday|Founder ingests new data into pipeline|Founder|
|3|Monday|Run forecast script: generates 28-day forecast + backlog + headcount|Automated (script)|
|4|Monday|Script generates Excel report with all 5 tabs|Automated (script)|
|5|Monday|Founder reviews Excel for quality, accuracy, and anomalies|Founder|
|6|Monday|Founder sends email with Excel + key takeaways to client|Founder|
|7|Monday/Tuesday|Client reviews report; founder available for questions via email/call|Both|

**Target time per client per week:** 30–45 minutes (data ingestion + review + send). As clients scale, automate steps 2–4 fully.

---

## 7. Client Configuration

Each client has a YAML or JSON configuration file with the following parameters:

|Parameter|Example Value|Type|Notes|
|---|---|---|---|
|`client_name`|"Logista Madrid"|String|Used in report headers|
|`active_flows`|["outbound", "inbound"]|List|Which flows to forecast|
|`productivity_inbound`|85|Float (units/hr)|Client-provided|
|`productivity_outbound`|120|Float (units/hr)|Client-provided|
|`hours_per_shift`|8|Integer|Default: 8|
|`overhead_buffer`|0.15|Float (0–1)|Default: 15%|
|`backlog_threshold_watch`|1.0|Float (days)|SLA-based|
|`backlog_threshold_critical`|2.0|Float (days)|SLA-based|
|`initial_backlog_outbound`|3500|Integer (units)|Set at onboarding|
|`initial_backlog_inbound`|1200|Integer (units)|Set at onboarding|
|`language`|"es"|"es" \| "en"|Report language|
|`forecast_horizon`|28|Integer (days)|Default: 28|

---

## 8. Technical Architecture (MVP)

### 8.1 System Overview

```
Client (CSV/Excel) → Data Pipeline → Forecast Engine → Backlog & Labor Calc → Excel Generator → Founder Review → Email to Client
```

### 8.2 Tech Stack

|Component|Technology|Notes|
|---|---|---|
|**Language**|Python 3.10+|Tested on 3.12|
|**Foundation model**|Amazon Chronos-2 (HuggingFace)|Zero-shot, 46M params (small)|
|**Classical model**|Meta Prophet|Interpretable baseline|
|**Data processing**|pandas, numpy|Standard data stack|
|**Excel generation**|openpyxl|Conditional formatting, charts|
|**Configuration**|YAML per client|Easy to modify|
|**Testing**|pytest|46+ unit tests|
|**AI Agent (internal)**|OpenAI GPT-4o|Founder tool for ad-hoc analysis, not client-facing|

### 8.3 What's Built vs. What's Needed

|Component|Status|Notes|
|---|---|---|
|Data ingestion (load, clean, validate)|✅ Built|Auto-detect, clean, validate pipeline|
|Chronos-2 forecasting|✅ Built|28-day forecast with confidence intervals|
|Prophet forecasting|✅ Built|Classical baseline|
|Evaluation metrics (WAPE, MAPE, etc.)|✅ Built|4 metrics implemented|
|Basic labor calculator|✅ Built|Volume → hours → headcount|
|AI agent (internal tool)|✅ Built|GPT-4o with sandboxed code execution|
|Unit tests (46 tests)|✅ Built|Layers 1–3 covered|
|Backlog calculation engine|🔶 To Build|Core backlog equation + days-of-backlog|
|Backlog threshold alerts|🔶 To Build|Healthy / Watch / Critical classification|
|Client config system (YAML)|🔶 To Build|Per-client parameters|
|Excel report generator (5 tabs)|🔶 To Build|openpyxl with formatting + charts|
|Accuracy tracking (forecast vs actual)|🔶 To Build|Rolling comparison + historical log|
|Multi-language support (ES/EN)|🔶 To Build|Configurable labels and headers|
|Semi-auto pipeline script|🔶 To Build|One command: ingest → forecast → Excel|

---

## 9. MVP Success Criteria

### 9.1 Alpha Tester Success

|Metric|Target|Measurement|
|---|---|---|
|Forecast accuracy (WAPE)|**< 10%**|Rolling 4-week average|
|Report delivery|**Every Monday by noon**|Consistency of delivery|
|Client uses the report|**Weekly for staffing decisions**|Ask in weekly check-in|
|Time to generate report|**< 45 min per client**|Founder time tracking|
|Client satisfaction (NPS)|**> 50**|Monthly survey|

### 9.2 Business Validation

|Question to Validate|Signal|Timeline|
|---|---|---|
|Will they pay €3K/month?|Conversion after free pilot|Week 13|
|Do they actually use the report?|Weekly engagement|Week 9–13|
|Does backlog metric resonate?|Client references it in calls|Week 10+|
|Can we deliver <10% WAPE?|Measured accuracy|Week 13|
|Is 45 min/client sustainable?|Founder time log|Week 9–13|

---

## 10. Out of Scope (MVP)

The following are explicitly excluded from the MVP to maintain focus and speed:

- Web dashboard or client-facing UI
- User accounts, authentication, or multi-tenant platform
- Automated email sending (founder sends manually)
- WMS integration or API connectors
- Real-time or intraday forecasting (weekly cycle only)
- Client-facing AI agent or chatbot
- Mobile app
- Automated data collection from client systems
- Multi-warehouse support for a single client (one warehouse per config)

**Guiding principle:** If it doesn't directly help the alpha tester receive a better forecast on Monday, it's not in the MVP.

---

## 11. Development Timeline

Aligned with the Planner milestone of MVP alpha-ready by March 14 (Week 4):

|Week|Focus|Deliverable|
|---|---|---|
|**Week 1**|Backlog engine + client config|Backlog calculation working E2E on synthetic data|
|**Week 2**|Excel report generator|5-tab Excel output with formatting and charts|
|**Week 3**|Accuracy tracking + i18n + integration|Full pipeline: one command generates complete report|
|**Week 4**|Testing + polish on real/realistic data|MVP alpha-ready: can deliver weekly forecast to a real warehouse|

---

_This is a living document. Update after first 3 discovery calls and after first alpha tester feedback. Goal: by Week 9, this document should reflect validated requirements, not assumptions._