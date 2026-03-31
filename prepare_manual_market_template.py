from __future__ import annotations

import argparse
import datetime as dt
import os

import capture_market_lines as c
import market_helpers as m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the manual sportsbook entry template for a given MLB slate.")
    parser.add_argument("--date", dest="report_date", default=None, help="Report date in YYYY-MM-DD format. Defaults to local today.")
    return parser.parse_args()


def parse_report_date(raw: str | None) -> dt.date:
    if raw:
        return dt.date.fromisoformat(str(raw).strip())
    return m.today_local()


def main() -> None:
    args = parse_args()
    report_date = parse_report_date(args.report_date)
    games, warnings = c.load_games_for_date(report_date)
    path, updated = m.ensure_manual_market_template(report_date, games)
    print(f"Wrote: {path}")
    print(f"Games in template: {len(games)}")
    print(f"Template updated: {'yes' if updated else 'no'}")
    if warnings:
        for warning in warnings:
            print(f"WARN: {warning}")
    print(f"Next step: fill {os.path.basename(path)} with manual lines, then run run_daily_mlb_report.py.")


if __name__ == "__main__":
    main()
