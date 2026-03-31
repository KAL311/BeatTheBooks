# Opening Day Launch Checklist

## Objective
This checklist is the final hardening guide for a controlled Opening Day launch of the MLB prediction engine, betting report, and quant dashboard stack.

The goal is not to prove a long-run edge before first pitch. The goal is to launch a stable, conservative, well-instrumented system that:
- produces the daily report and workbook reliably
- writes the quant store cleanly
- degrades safely when data is incomplete
- keeps bankroll policy conservative until real regular-season evidence accumulates

## Current Readiness View
As of March 18, 2026, the project is close to launch-ready if the final days are used for stability and process discipline rather than large new feature additions.

Current strength areas:
- core prediction engine is stable
- clean replay validation exists
- uncertainty, shrinkage, and calibration are live
- quant store and dashboard are operational
- market-aware policy research exists
- market-specific takeover thresholds now exist for moneyline, totals, and run line

Current unavoidable limitation:
- real-market CLV and ROI proof will remain thin until regular-season priced bets settle

## Go / No-Go Standard
A launch is `GO` if all of the following are true:
- daily report generates successfully
- season workbook generates successfully and has expected sheets
- DuckDB quant store writes successfully
- dashboard loads without runtime errors
- latest run snapshot is current for the intended report date
- core state blocks exist in `model_state.json`
- no critical file or database dependency is missing
- fallback behavior is available for missing lines, missing lineups, missing starters, and missing market proof

A launch is `WARN` if:
- one or more noncritical analytics tables are empty by design
- regular-season market proof is not ready yet
- market-specific recommendations remain shadow-driven
- live validation sample is still small

A launch is `NO-GO` if:
- the report fails to generate
- the workbook fails to generate or is structurally broken
- the quant store cannot be written
- the dashboard crashes on startup or common views
- the model state is missing critical calibration/shrinkage blocks
- required databases or output files are missing

## Final 3-Day Roadmap
### Day 1
- run the automated readiness checker
- fix all FAIL items
- fix all dashboard/runtime exceptions
- confirm report, workbook, DuckDB, and watchlist outputs write cleanly

### Day 2
- freeze conservative launch defaults
- confirm market ingestion and empty-state behavior
- verify fallback behavior for:
  - missing starter data
  - missing lineups
  - missing market lines
  - sparse real-market evidence

### Day 3
- perform one full dry run on the exact launch workflow
- rerun the readiness checker
- do not add major model features unless a clear defect is found
- treat reliability as more important than ambition

## Opening Day Operator Checklist
### Before first run
- confirm `run_daily_mlb_report.py` exists and compiles
- confirm `quant_dashboard.py` exists and compiles
- confirm `sportsbook_lines.db` exists
- confirm `mlb_quant_dashboard.duckdb` exists or can be created
- confirm `model_state.json` exists
- confirm latest schedule and probable starters are available

### After main run
- confirm `MLB_Report_<date>.txt` exists
- confirm `MLB_Season_To_Date_2026.xlsx` exists
- confirm the workbook contains:
  - `Team_Metrics`
  - `Top_20_Batters`
  - `Top_20_Pitchers`
  - `Power_Rankings`
- confirm `mlb_quant_dashboard.duckdb` updated
- confirm latest `quant_runs.report_date` matches the run date

### Before using quant recommendations
- confirm dashboard opens cleanly
- confirm latest run appears in Overview
- confirm market candidates are populated when lines exist
- confirm portfolio section does not crash on empty candidate states
- confirm policy recommendation is present
- confirm no stale-date mismatch exists between report date and latest run date

## Launch Defaults
These should remain conservative on Opening Day:
- default portfolio policy: `Flat`
- default posture: `Base`
- only promote to real-market policy by market when that market clears its own takeover threshold
- keep bankroll deployment conservative until settled priced bets accumulate
- treat moneyline as the highest-trust market first
- treat totals and run line as lower-trust until their own evidence paths strengthen

## Fail-Safe Matrix
| Trigger | Risk | System behavior | Operator action |
|---|---|---|---|
| Missing report file | launch failure | fail readiness | rerun model, inspect traceback |
| Workbook missing expected sheets | broken deliverable | fail readiness | inspect workbook export path |
| DuckDB export missing | quant stack unavailable | fail readiness | rerun model, inspect quant store path |
| Dashboard runtime error | monitoring blind spot | fail readiness | patch dashboard before launch |
| Missing market lines | no candidate sizing | warn only | use report without portfolio actions or load manual lines |
| No settled priced bets | no real-market proof | warn only | continue shadow-driven policy research |
| Missing lineup confirmations | lower certainty | degrade through uncertainty/actionability | use conservative posture |
| Missing starter certainty | side/total noise | degrade through uncertainty/actionability | do not press correlated exposure |
| Sparse totals/run-line evidence | low market trust | keep `No call yet` state | wait for market-specific takeover |
| Stale latest run date | operational mismatch | fail readiness | rerun daily model before using outputs |

## Market-Specific Real-Market Takeover Thresholds
- Moneyline:
  - primary takeover: `3 settled slates / 10 priced bets`
  - high-confidence: `5 / 20`
- Total:
  - primary takeover: `4 / 12`
  - high-confidence: `6 / 24`
- Run line:
  - primary takeover: `5 / 14`
  - high-confidence: `7 / 28`

## What Not To Do In The Final Days
- do not add major new feature families unless a clear defect is found
- do not relax fail-safes just to make the dashboard look fuller
- do not allow totals or run line to inherit trust they have not earned
- do not mistake lack of spring market proof for broken launch readiness

## Primary Success Definition For Opening Day
A successful launch means:
- the model runs cleanly
- the report is readable and actionable
- the workbook is structurally correct
- the quant dashboard is stable
- the system stays conservative where evidence is thin
- the regular season can begin accumulating real proof immediately
