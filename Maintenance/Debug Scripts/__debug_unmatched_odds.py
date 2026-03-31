import datetime as dt
import os

import requests

import market_helpers as m
import capture_market_lines as cap
import run_daily_mlb_report as clean


def main():
    report_date = dt.date.fromisoformat(os.getenv("DEBUG_REPORT_DATE", "2026-03-23"))
    games, warnings = cap.load_games_for_date(report_date)
    print("schedule warnings:", warnings)
    print("scheduled games:")
    for g in games:
        print(f" - {g.away} @ {g.home} | {getattr(g, 'scheduled_dt_utc', None)}")

    fetched_rows, fetch_warnings = m.fetch_market_lines_from_odds_api(report_date, games)
    print("capture-path fetch rows:", fetched_rows)
    print("capture-path warnings:", fetch_warnings)
    summary, market_map, market_warn = cap.capture_market_state(report_date, games)
    print("capture_market_state summary:", summary)
    print("capture_market_state warnings:", market_warn)
    print("capture_market_state market games:", len(market_map))

    api_key = str(os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or "").strip()
    if not api_key:
        print("missing api key")
        return
    event_times = [m.parse_utc_datetime(getattr(g, "scheduled_dt_utc", None)) for g in games]
    event_times = [t for t in event_times if t is not None]
    params = {
        "apiKey": api_key,
        "regions": str(os.getenv("ODDS_API_REGIONS") or "us").strip() or "us",
        "markets": str(os.getenv("ODDS_API_MARKETS") or "h2h,spreads,totals").strip() or "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeRotationNumbers": "true",
    }
    if event_times:
        params["commenceTimeFrom"] = (min(event_times) - dt.timedelta(hours=6)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        params["commenceTimeTo"] = (max(event_times) + dt.timedelta(hours=6)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    response = requests.get(f"{m.ODDS_API_BASE_URL}/sports/{m.ODDS_API_DEFAULT_SPORT}/odds", params=params, headers=clean.HEADERS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    print("events returned:", len(payload))
    unmatched = []
    bookmaker_rows = 0
    unmatched_rows = 0
    no_market_rows = 0
    for event in payload:
        matched = m.match_market_event_to_game(event, games)
        if int(matched.get("game_pk") or 0) <= 0:
            unmatched.append(
                (
                    event.get("away_team"),
                    event.get("home_team"),
                    event.get("commence_time"),
                    event.get("sport_title"),
                )
            )
        for bookmaker in event.get("bookmakers") or []:
            bookmaker_rows += 1
            record = m.parse_odds_api_book_row(event, bookmaker, games, report_date)
            if record is None:
                unmatched_rows += 1
                markets = [str((market or {}).get("key") or "") for market in (bookmaker.get("markets") or [])]
                if not any(key in {"h2h", "spreads", "totals"} for key in markets):
                    no_market_rows += 1
                    print(
                        "NO_CORE_MARKETS:",
                        event.get("away_team"),
                        "@",
                        event.get("home_team"),
                        "| book:",
                        bookmaker.get("title") or bookmaker.get("key"),
                        "| markets:",
                        markets,
                    )
                else:
                    print(
                        "UNMATCHED_ROW:",
                        event.get("away_team"),
                        "@",
                        event.get("home_team"),
                        "| book:",
                        bookmaker.get("title") or bookmaker.get("key"),
                        "| matched:",
                        matched,
                        "| markets:",
                        markets,
                    )
    print("unmatched events:", len(unmatched))
    for away, home, commence, sport_title in unmatched:
        print(f" - {away} @ {home} | {commence} | {sport_title}")
    print("bookmaker rows:", bookmaker_rows)
    print("unmatched bookmaker rows:", unmatched_rows)
    print("rows without h2h/spreads/totals:", no_market_rows)


if __name__ == "__main__":
    main()
