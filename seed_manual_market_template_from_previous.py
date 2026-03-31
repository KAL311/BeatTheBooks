from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import capture_market_lines as c
import market_helpers as m

LINE_COLUMNS = [
    "home_ml",
    "away_ml",
    "home_run_line",
    "home_run_line_price",
    "away_run_line",
    "away_run_line_price",
    "total_line",
    "over_price",
    "under_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed today's manual sportsbook template from the most recent prior filled template."
    )
    parser.add_argument("--date", dest="report_date", default=None, help="Target date in YYYY-MM-DD format. Defaults to local today.")
    parser.add_argument("--lookback-days", dest="lookback_days", type=int, default=7, help="How many days back to search for a prior filled template.")
    parser.add_argument("--sportsbook", dest="sportsbook", default="ManualConsensus", help="Sportsbook label to seed from. Defaults to ManualConsensus.")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite existing filled fields in the target template.")
    return parser.parse_args()


def parse_report_date(raw: Optional[str]) -> dt.date:
    if raw:
        return dt.date.fromisoformat(str(raw).strip())
    return m.today_local()


def read_template(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=m.MANUAL_MARKET_TEMPLATE_COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=m.MANUAL_MARKET_TEMPLATE_COLUMNS)
    for col in m.MANUAL_MARKET_TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[m.MANUAL_MARKET_TEMPLATE_COLUMNS].copy()


def normalize_template(df: pd.DataFrame, sportsbook: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["report_date"] = work["report_date"].fillna("").astype(str).str.strip()
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce").fillna(0).astype(int)
    work["away_team"] = work["away_team"].fillna("").astype(str).str.upper().str.strip()
    work["home_team"] = work["home_team"].fillna("").astype(str).str.upper().str.strip()
    work["sportsbook"] = work["sportsbook"].fillna("").astype(str).str.strip()
    if sportsbook:
        work = work[work["sportsbook"].str.lower() == sportsbook.strip().lower()].copy()
    return work


def row_has_lines(row: pd.Series) -> bool:
    for col in LINE_COLUMNS:
        value = row.get(col)
        if pd.notna(value) and str(value).strip() != "":
            return True
    return False


def find_prior_template(target_date: dt.date, lookback_days: int, sportsbook: str) -> Tuple[Optional[Path], pd.DataFrame]:
    for offset in range(1, max(int(lookback_days), 1) + 1):
        candidate_date = target_date - dt.timedelta(days=offset)
        candidate_path = Path(m.manual_market_template_path(candidate_date))
        df = normalize_template(read_template(candidate_path), sportsbook)
        if df.empty:
            continue
        if any(row_has_lines(row) for _, row in df.iterrows()):
            return candidate_path, df
    return None, pd.DataFrame(columns=m.MANUAL_MARKET_TEMPLATE_COLUMNS)


def merge_notes(existing_note: object, source_date: dt.date) -> str:
    existing = str(existing_note or "").strip()
    carry_tag = f"carry-forward from {source_date.isoformat()}"
    if not existing:
        return carry_tag
    if carry_tag.lower() in existing.lower():
        return existing
    return f"{existing} | {carry_tag}"


def seed_template_from_prior(
    target_date: dt.date,
    lookback_days: int,
    sportsbook: str,
    overwrite: bool,
) -> Tuple[Path, int, int, Optional[Path], List[str]]:
    warnings: List[str] = []
    games, load_warn = c.load_games_for_date(target_date)
    warnings.extend(load_warn)
    target_path_str, _ = m.ensure_manual_market_template(target_date, games)
    target_path = Path(target_path_str)

    target_df = read_template(target_path)
    prior_path, prior_df = find_prior_template(target_date, lookback_days, sportsbook)
    if prior_path is None or prior_df.empty:
        return target_path, 0, len(target_df), None, warnings

    prior_map: Dict[Tuple[str, str, str], pd.Series] = {}
    for _, row in prior_df.iterrows():
        if not row_has_lines(row):
            continue
        key = (
            str(row["away_team"]),
            str(row["home_team"]),
            str(row["sportsbook"]).strip().lower(),
        )
        prior_map[key] = row

    seeded = 0
    untouched = 0
    target_out = target_df.copy()
    for idx, row in target_out.iterrows():
        key = (
            str(row.get("away_team", "")).strip().upper(),
            str(row.get("home_team", "")).strip().upper(),
            str(row.get("sportsbook", "")).strip().lower(),
        )
        prior_row = prior_map.get(key)
        if prior_row is None:
            untouched += 1
            continue

        changed = False
        for col in LINE_COLUMNS:
            current_val = row.get(col)
            prior_val = prior_row.get(col)
            if not overwrite and pd.notna(current_val) and str(current_val).strip() != "":
                continue
            if pd.isna(prior_val) or str(prior_val).strip() == "":
                continue
            target_out.at[idx, col] = prior_val
            changed = True

        if changed:
            source_date = dt.date.fromisoformat(str(prior_row.get("report_date") or target_date.isoformat()))
            target_out.at[idx, "snapshot_ts"] = ""
            target_out.at[idx, "source"] = "manual_template_seeded"
            target_out.at[idx, "notes"] = merge_notes(target_out.at[idx, "notes"], source_date)
            seeded += 1
        else:
            untouched += 1

    target_out.to_csv(target_path, index=False)
    return target_path, seeded, untouched, prior_path, warnings


def main() -> None:
    args = parse_args()
    target_date = parse_report_date(args.report_date)
    target_path, seeded, untouched, prior_path, warnings = seed_template_from_prior(
        target_date=target_date,
        lookback_days=args.lookback_days,
        sportsbook=args.sportsbook,
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote: {target_path}")
    print(f"Seeded rows: {seeded}")
    print(f"Unchanged rows: {untouched}")
    if prior_path is not None:
        print(f"Source template: {prior_path}")
    else:
        print("Source template: none found with filled lines")
    if warnings:
        for warning in warnings:
            print(f"WARN: {warning}")
    print("Next step: review the carried-forward rows, adjust prices as needed, then run run_daily_mlb_report.py.")


if __name__ == "__main__":
    main()
