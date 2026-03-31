from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import market_helpers as m
import run_daily_mlb_report as clean
WATCHLIST_TEMPLATE = "Sportsbook_Watchlist_{date}.txt"
LABEL_PRIORITY = {"MUST TAKE": 0, "BET": 1, "WATCH": 2, "PASS": 3}


@dataclass
class MarketGame:
    game_pk: int
    away: str
    home: str
    venue: str
    start_time_et: str
    away_pitcher: Optional[str] = None
    home_pitcher: Optional[str] = None
    venue_id: Optional[int] = None
    away_pitcher_id: Optional[int] = None
    home_pitcher_id: Optional[int] = None
    scheduled_dt_utc: Optional[dt.datetime] = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture sportsbook lines and output a model-linked watchlist.")
    parser.add_argument("--date", dest="report_date", default=None, help="Report date in YYYY-MM-DD format. Defaults to local today.")
    parser.add_argument("--output", dest="output_path", default=None, help="Optional output path for the watchlist text file.")
    return parser.parse_args()


def parse_report_date(raw: Optional[str]) -> dt.date:
    if raw:
        return dt.date.fromisoformat(str(raw).strip())
    return clean.today_local()


def load_games_for_date(report_date: dt.date) -> Tuple[List[MarketGame], List[str]]:
    client = clean.RequestClient()
    games_raw, warnings = clean.fetch_schedule_games(client, report_date)
    games: List[MarketGame] = []
    for item in games_raw:
        games.append(
            MarketGame(
                game_pk=int(item.get("game_pk") or 0),
                away=str(item.get("away_team") or "").strip().upper(),
                home=str(item.get("home_team") or "").strip().upper(),
                venue=str(item.get("venue_name") or "Unknown venue").strip(),
                start_time_et=str(item.get("start_time_ct") or "").strip(),
                away_pitcher=clean.canonical_player_name(item.get("away_pitcher")),
                home_pitcher=clean.canonical_player_name(item.get("home_pitcher")),
                venue_id=m.parse_market_int(item.get("venue_id")),
                away_pitcher_id=m.parse_market_int(item.get("away_pitcher_id")),
                home_pitcher_id=m.parse_market_int(item.get("home_pitcher_id")),
                scheduled_dt_utc=item.get("scheduled_utc"),
            )
        )
    return games, warnings


def _coerce_utc_datetime(value: object) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc) if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def mark_closing_market_snapshots_local(report_date: dt.date, games: List[MarketGame]) -> Tuple[int, List[str]]:
    m.ensure_market_db()
    now_utc = dt.datetime.now(dt.timezone.utc)
    report_iso = report_date.isoformat()
    marked = 0

    with sqlite3.connect(m.SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for g in games:
            scheduled = _coerce_utc_datetime(g.scheduled_dt_utc)
            if scheduled is None or scheduled > now_utc:
                continue
            rows = conn.execute(
                """
                SELECT id, sportsbook, snapshot_ts
                FROM odds_snapshots
                WHERE report_date = ? AND away_team = ? AND home_team = ?
                ORDER BY snapshot_ts DESC, id DESC
                """,
                (report_iso, g.away, g.home),
            ).fetchall()
            if not rows:
                continue
            conn.execute(
                "UPDATE odds_snapshots SET is_closing = 0 WHERE report_date = ? AND away_team = ? AND home_team = ?",
                (report_iso, g.away, g.home),
            )
            latest_by_book: Dict[str, int] = {}
            for row in rows:
                sportsbook = str(row["sportsbook"] or "").strip()
                if sportsbook and sportsbook not in latest_by_book:
                    latest_by_book[sportsbook] = int(row["id"])
            if latest_by_book:
                conn.executemany(
                    "UPDATE odds_snapshots SET is_closing = 1 WHERE id = ?",
                    [(row_id,) for row_id in latest_by_book.values()],
                )
                marked += len(latest_by_book)
    return marked, []


def capture_market_state(report_date: dt.date, games: List[MarketGame]) -> Tuple[Dict[str, int], Dict[str, Dict[str, object]], List[str]]:
    warnings: List[str] = []
    m.ensure_out_dir()
    m.ensure_market_db()
    m.ensure_market_csv_template()

    import_info, import_warn = m.import_all_market_lines(report_date, games)
    warnings.extend(import_warn)

    report_phase = m.season_phase_for_date(report_date)
    fetched = 0
    if report_phase == "spring" and str(os.getenv("ODDS_API_ALLOW_SPRING_FETCH") or "").strip() not in ["1", "true", "yes", "on"]:
        warnings.append(f"Live sportsbook fetch skipped for spring schedule; use {os.path.basename(str(import_info.get('template_path', '')))} for manual lines.")
    else:
        fetched, fetch_warn = m.fetch_market_lines_from_odds_api(report_date, games)
        warnings.extend(fetch_warn)

    closing_marked, closing_warn = m.mark_closing_market_snapshots(report_date, games)
    warnings.extend(closing_warn)

    settled_count, settle_warn = m.settle_bet_journal(report_date)
    warnings.extend(settle_warn)

    market_map, market_warn = m.load_market_consensus_for_date(report_date, games)
    warnings.extend(market_warn)
    auto_template_rows, auto_template_warn = m.populate_manual_template_from_consensus(report_date, games, market_map)
    warnings.extend(auto_template_warn)
    if auto_template_rows > 0:
        template_path = str(import_info.get("template_path") or m.manual_market_template_path(report_date))
        reimported_template_rows, reimport_warn = m.import_market_lines_csv(
            report_date,
            template_path,
            default_source='odds_api_consensus',
        )
        warnings.extend(reimport_warn)
        import_info["manual_template"] = int(reimported_template_rows)
        import_info["total"] = int(import_info.get("manual_csv", 0)) + int(import_info.get("manual_template", 0))

    summary = {
        "imported": int(import_info.get("total", 0)),
        "manual_csv": int(import_info.get("manual_csv", 0)),
        "manual_template": int(import_info.get("manual_template", 0)),
        "auto_template_rows": int(auto_template_rows),
        "fetched": int(fetched),
        "closing_marked": int(closing_marked),
        "settled": int(settled_count),
        "market_games": int(len(market_map)),
        "template_path": str(import_info.get("template_path", "")),
    }
    return summary, market_map, warnings


def load_watch_rows(report_date: dt.date) -> List[Dict[str, object]]:
    m.ensure_market_db()
    report_iso = report_date.isoformat()
    with sqlite3.connect(m.SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM bet_journal
            WHERE report_date = ? AND status IN ('watch', 'bet') AND result_units IS NULL
            ORDER BY
                CASE actionability_label WHEN 'MUST TAKE' THEN 0 WHEN 'BET' THEN 1 WHEN 'WATCH' THEN 2 ELSE 3 END,
                COALESCE(actionability_score, 0) DESC,
                ABS(COALESCE(edge_value, 0)) DESC,
                id ASC
            """,
            (report_iso,),
        ).fetchall()
    return [dict(row) for row in rows]


def _fmt_price(value: Optional[int]) -> str:
    parsed = m.parse_market_int(value)
    return "n/a" if parsed is None else f"{parsed:+d}"


def _fmt_float(value: Optional[float], digits: int = 1, signed: bool = False) -> str:
    parsed = m.safe_float(value)
    if parsed is None:
        return "n/a"
    if signed:
        return f"{parsed:+.{digits}f}"
    return f"{parsed:.{digits}f}"


def _market_entry_for_row(row: Dict[str, object], market_map: Dict[str, Dict[str, object]]) -> Optional[Dict[str, object]]:
    game_pk = m.parse_market_int(row.get("game_pk"))
    key = m.market_game_key(game_pk if game_pk and game_pk > 0 else None, row.get("away_team"), row.get("home_team"))
    return market_map.get(key)


def build_watch_entry(row: Dict[str, object], market: Dict[str, object]) -> Optional[Dict[str, object]]:
    away_team = str(row.get("away_team") or "").strip().upper()
    home_team = str(row.get("home_team") or "").strip().upper()
    market_type = str(row.get("market_type") or "").strip().lower()
    selection = str(row.get("selection") or "").strip()
    sportsbook_count = int(m.safe_float(market.get("sportsbook_count")) or 0)
    market_label = str(market.get("market_label") or row.get("sportsbook") or "Market")
    confidence = str(row.get("confidence_tier") or "Low")
    action_label = str(row.get("actionability_label") or "").strip().upper()
    action_score = m.safe_float(row.get("actionability_score"))
    action_notes = str(row.get("actionability_notes") or "").strip()
    report_iso = str(row.get("report_date") or "").strip()
    try:
        report_phase = m.season_phase_for_date(dt.date.fromisoformat(report_iso)) if report_iso else None
    except Exception:
        report_phase = None

    detail = ""
    edge_value: Optional[float] = None
    label = "PASS"

    if market_type == "moneyline":
        selected_team = selection.upper()
        home_ml = m.parse_market_int(market.get("home_ml"))
        away_ml = m.parse_market_int(market.get("away_ml"))
        market_home_prob, market_away_prob = m.no_vig_probabilities(home_ml, away_ml)
        model_prob = m.safe_float(row.get("model_probability"))
        fair_price = m.parse_market_int(row.get("model_price"))
        if selected_team == home_team:
            book_price = home_ml
            market_prob = market_home_prob
            matchup = f"{away_team} @ {home_team}"
        elif selected_team == away_team:
            book_price = away_ml
            market_prob = market_away_prob
            matchup = f"{away_team} @ {home_team}"
        else:
            return None
        if model_prob is None or market_prob is None or book_price is None:
            return None
        edge_value = float(model_prob - market_prob)
        raw_label = m.market_signal_label("moneyline", edge_value, sportsbook_count, model_phase=report_phase); label = action_label if raw_label != "PASS" and action_label in ["MUST TAKE", "BET", "WATCH"] else raw_label
        detail = (
            f"{selection} {_fmt_price(book_price)} | fair {_fmt_price(fair_price)} | "
            f"edge {edge_value * 100:+.1f} pts | {market_label}" + (f" | action {action_score:.0f}/100" if action_score is not None else "") + (f" | {action_notes}" if action_notes else "")
        )
        return {
            "label": label,
            "market_type": "moneyline",
            "matchup": matchup,
            "detail": detail,
            "confidence": confidence,
            "edge_sort": abs(edge_value),
        }

    if market_type == "run_line":
        match = re.match(r"^([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?)$", selection.upper())
        if not match:
            return None
        selected_team = match.group(1)
        stored_line = float(match.group(2))
        model_prob = m.safe_float(row.get("model_probability"))
        fair_price = m.parse_market_int(row.get("model_price"))
        projected_margin = m.safe_float(row.get("model_line_value"))
        home_line = m.parse_market_float(market.get("home_run_line"))
        away_line = m.parse_market_float(market.get("away_run_line"))
        home_price = m.parse_market_int(market.get("home_run_line_price"))
        away_price = m.parse_market_int(market.get("away_run_line_price"))
        market_home_prob, market_away_prob = m.no_vig_probabilities(home_price, away_price)
        if selected_team == home_team:
            current_line = home_line
            book_price = home_price
            market_prob = market_home_prob
        elif selected_team == away_team:
            current_line = away_line
            book_price = away_price
            market_prob = market_away_prob
        else:
            return None
        if model_prob is None or market_prob is None or current_line is None or book_price is None:
            return None
        if abs(float(current_line) - stored_line) > 0.01:
            label = "PASS"
            detail = (
                f"{selection} moved to {selected_team} {_fmt_float(current_line, digits=1, signed=True)} {_fmt_price(book_price)} | "
                f"proj margin {_fmt_float(projected_margin, digits=1, signed=True)} | {market_label}"
            )
        else:
            edge_value = float(model_prob - market_prob)
            raw_label = m.market_signal_label("run_line", edge_value, sportsbook_count, model_phase=report_phase); label = action_label if raw_label != "PASS" and action_label in ["MUST TAKE", "BET", "WATCH"] else raw_label
            detail = (
                f"{selection} {_fmt_price(book_price)} | fair {_fmt_price(fair_price)} | "
                f"proj margin {_fmt_float(projected_margin, digits=1, signed=True)} | "
                f"edge {edge_value * 100:+.1f} pts | {market_label}" + (f" | action {action_score:.0f}/100" if action_score is not None else "") + (f" | {action_notes}" if action_notes else "")
            )
        return {
            "label": label,
            "market_type": "run_line",
            "matchup": f"{away_team} @ {home_team}",
            "detail": detail,
            "confidence": confidence,
            "edge_sort": abs(edge_value) if edge_value is not None else 0.0,
        }

    if market_type == "total":
        match = re.match(r"^(OVER|UNDER)\s+(\d+(?:\.\d+)?)$", selection.upper())
        if not match:
            return None
        side = match.group(1)
        model_total = m.safe_float(row.get("model_line_value"))
        current_total = m.parse_market_float(market.get("total_line"))
        if model_total is None or current_total is None:
            return None
        diff = float(model_total - current_total)
        signed_edge = diff if side == "OVER" else -diff
        book_price = m.parse_market_int(market.get("over_price") if side == "OVER" else market.get("under_price"))
        current_lean = "OVER" if diff > 0.05 else ("UNDER" if diff < -0.05 else "FLAT")
        if current_lean != side:
            label = "PASS"
        else:
            raw_label = m.market_signal_label("total", signed_edge, sportsbook_count, model_phase=report_phase); label = action_label if raw_label != "PASS" and action_label in ["MUST TAKE", "BET", "WATCH"] else raw_label
        detail = (
            f"{side.title()} {current_total:.1f} {_fmt_price(book_price)} | model total {model_total:.1f} | "
            f"edge {signed_edge:+.1f} runs | {market_label}" + (f" | action {action_score:.0f}/100" if action_score is not None else "") + (f" | {action_notes}" if action_notes else "")
        )
        return {
            "label": label,
            "market_type": "total",
            "matchup": f"{away_team} @ {home_team}",
            "detail": detail,
            "confidence": confidence,
            "edge_sort": abs(signed_edge),
        }

    return None


def build_watch_entries(report_date: dt.date, market_map: Dict[str, Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    rows = load_watch_rows(report_date)
    if not rows:
        warnings.append(f"No current-day watch candidates are stored yet. Fill {os.path.basename(m.manual_market_template_path(report_date))} or load sportsbook lines, then run run_daily_mlb_report.py to seed model-based market candidates.")
        return [], warnings

    entries: List[Dict[str, object]] = []
    missing_market = 0
    for row in rows:
        market = _market_entry_for_row(row, market_map)
        if not market:
            missing_market += 1
            continue
        entry = build_watch_entry(row, market)
        if entry is not None and entry.get("label") in ["MUST TAKE", "BET", "WATCH"]:
            entries.append(entry)

    if missing_market:
        warnings.append(f"{missing_market} watch candidate(s) do not yet have a current market consensus row.")

    entries.sort(key=lambda item: (LABEL_PRIORITY.get(str(item.get("label")), 9), -float(item.get("edge_sort", 0.0)), str(item.get("matchup", ""))))
    return entries, warnings


def render_watchlist(
    report_date: dt.date,
    output_path: str,
    summary: Dict[str, int],
    warnings: List[str],
    entries: List[Dict[str, object]],
) -> str:
    report_phase = m.season_phase_for_date(report_date)
    spring_live_enabled = str(os.getenv("ODDS_API_ALLOW_SPRING_FETCH") or "").strip().lower() in ["1", "true", "yes", "on"]
    if report_phase == "spring" and not spring_live_enabled:
        operating_mode = "Operating mode: Manual spring mode | live Odds API fetch is disabled for spring, so use the manual sportsbook template."
    elif report_phase == "spring":
        operating_mode = "Operating mode: Live spring mode | Odds API spring fetch is enabled."
    else:
        operating_mode = "Operating mode: Live regular-season mode | Odds API fetch is enabled for scheduled capture runs."
    lines: List[str] = []
    lines.append(f"SPORTSBOOK WATCHLIST - {report_date.isoformat()}")
    lines.append("Built from latest stored market snapshots and today\'s model candidates.")
    lines.append(operating_mode)
    lines.append("")
    lines.append("SUMMARY")
    lines.append(f" - CSV rows imported: {summary.get('imported', 0)}")
    lines.append(f" - Manual template rows auto-filled from Odds API: {summary.get('auto_template_rows', 0)}")
    lines.append(f" - Odds API rows fetched: {summary.get('fetched', 0)}")
    lines.append(f" - Games with market consensus: {summary.get('market_games', 0)}")
    lines.append(f" - Closing snapshots marked: {summary.get('closing_marked', 0)}")
    lines.append(f" - Historical bets settled: {summary.get('settled', 0)}")
    if summary.get("template_path"):
        lines.append(f" - Manual template: {os.path.basename(str(summary.get('template_path', '')))}")
    lines.append(f" - Active watch entries: {len(entries)}")
    lines.append("")

    if warnings:
        lines.append("WARNINGS")
        for warning in warnings:
            lines.append(f" - {warning}")
        lines.append("")

    lines.append("WATCHLIST")
    if not entries:
        lines.append(" - No active MUST TAKE, BET, or WATCH candidates right now.")
    else:
        for entry in entries:
            lines.append(
                f" - {entry['label']} | {str(entry['market_type']).upper()} | {entry['matchup']} | "
                f"{entry['detail']} | confidence {entry['confidence']}"
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    args = parse_args()
    report_date = parse_report_date(args.report_date)
    output_path = args.output_path or os.path.join(m.OUT_DIR, WATCHLIST_TEMPLATE.format(date=report_date.isoformat()))

    games, game_warn = load_games_for_date(report_date)
    summary, market_map, market_warn = capture_market_state(report_date, games)
    entries, watch_warn = build_watch_entries(report_date, market_map)
    warnings = game_warn + market_warn + watch_warn

    out = render_watchlist(report_date, output_path, summary, warnings, entries)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()



