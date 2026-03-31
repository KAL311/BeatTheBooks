# 2026 MLB Report

Python-based MLB reporting and market-tracking project for generating the daily report, sportsbook watchlist, workbook exports, dashboard data, and validation utilities.

## Working Directory

Run commands from:

```powershell
cd "<repo-root>"
```

## Setup

Install the dashboard/runtime dependencies if needed:

```powershell
python -m pip install -r .\quant_dashboard_requirements.txt
```

## GitHub Backup Notes

This project is designed to keep live processing local while still allowing a clean GitHub backup of the recoverable source.

- The live production copy can remain on your machine.
- The GitHub copy should be treated as a backup/source repo, not the active runtime folder.
- Runtime databases and dated output files are intentionally ignored by `.gitignore`.
- Core code, lightweight state, replay logs, and cached priors can still be committed so the project can be rebuilt later if needed.

Local-only runtime artifacts that are not intended for normal GitHub backup:

- `mlb_quant_dashboard.duckdb`
- `sportsbook_lines.db`
- dated report/watchlist/manual line files
- `MLB_Season_To_Date_2026.xlsx`
- virtual environment and cache folders

## Daily Workflow

Run the full market capture -> report -> post-capture cycle:

```powershell
pwsh -File .\run_daily_market_cycle.ps1
```

Useful variants:

```powershell
pwsh -File .\run_daily_market_cycle.ps1 -CaptureOnly
pwsh -File .\run_daily_market_cycle.ps1 -ReportOnly
pwsh -File .\run_daily_market_cycle.ps1 -SkipPostCapture
pwsh -File .\run_daily_market_cycle.ps1 -PythonExe "C:\path\to\python.exe"
```

Run the core pieces directly:

```powershell
python .\capture_market_lines.py
python .\run_daily_mlb_report.py
streamlit run .\quant_dashboard.py
```

Date-specific examples:

```powershell
python .\capture_market_lines.py --date 2026-03-23
python .\capture_market_lines.py --date 2026-03-23 --output .\Sportsbook_Watchlist_2026-03-23.txt
python .\prepare_manual_market_template.py --date 2026-03-23
python .\seed_manual_market_template_from_previous.py --date 2026-03-23 --lookback-days 7 --sportsbook ManualConsensus
```

## Main Commands

### Primary runners

```powershell
pwsh -File .\run_daily_market_cycle.ps1
python .\run_daily_mlb_report.py
python .\run_daily_mlb_report_clean.py
python .\capture_market_lines.py
streamlit run .\quant_dashboard.py
```

### Manual market template helpers

```powershell
python .\prepare_manual_market_template.py
python .\prepare_manual_market_template.py --date YYYY-MM-DD
python .\seed_manual_market_template_from_previous.py
python .\seed_manual_market_template_from_previous.py --date YYYY-MM-DD --lookback-days N --sportsbook ManualConsensus --overwrite
```

### Validation and reporting

```powershell
python .\verify_quant_stack.py
python .\opening_day_readiness.py
python .\opening_day_readiness.py --report-date YYYY-MM-DD
python .\clean_backtest_runner.py
python .\clean_backtest_runner.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --max-dates N --output PATH --summary-output PATH --rebuild
python .\verify_policy_research.py
python .\inspect_bet_journal.py
python .\tmp_verify_real_market_policy.py
```

### Maintenance and one-off checks

```powershell
python .\__check_evidence_history.py
python .\__check_market_db_schema.py
python .\__check_market_proof.py
python .\__check_market_thresholds.py
python .\__refresh_total_state.py
python .\__verify_dashboard_policy_fix.py
python .\__apply_total_sigma_calibration.py
python .\__apply_total_sigma_update.py
python .\__apply_totals_market_proof.py
```

### Debug utilities

```powershell
python .\__debug_fit_totals.py
python .\__debug_hou_wsh.py
python .\__debug_inspect_csv.py
python .\__debug_post_refresh.py
python .\__debug_replay_row.py
python .\__debug_replay_state.py
python .\__debug_sigma_check.py
python .\__debug_sigma_result.py
python .\__debug_sigma_state.py
python .\__debug_total_sigma_fit.py
python .\__debug_total_sigma_nll.py
python .\__debug_total_state.py
python .\__debug_totals_bias.py
python .\__debug_unmatched_odds.py
```

## Outputs

The main daily workflow produces or updates files like:

- `MLB_Report_YYYY-MM-DD.txt`
- `Sportsbook_Watchlist_YYYY-MM-DD.txt`
- `Manual_Sportsbook_Lines_YYYY-MM-DD.csv`
- `MLB_Season_To_Date_2026.xlsx`
- `mlb_quant_dashboard.duckdb`
- `sportsbook_lines.db`

## Notes

- `run_daily_mlb_report.py` uses the local date internally and does not currently expose CLI flags.
- `capture_market_lines.py`, `prepare_manual_market_template.py`, `seed_manual_market_template_from_previous.py`, `opening_day_readiness.py`, and `clean_backtest_runner.py` do expose CLI arguments.
- Backup files such as `.bak`, `*_backup.py`, and legacy recovery scripts are intentionally not listed as normal project commands.
