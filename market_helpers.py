
from __future__ import annotations

import datetime as dt
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import run_daily_mlb_report as clean

OUT_DIR = clean.OUT_DIR
SPORTSBOOK_DB_PATH = os.path.join(OUT_DIR, 'sportsbook_lines.db')
SPORTSBOOK_CSV_PATH = os.path.join(OUT_DIR, 'sportsbook_lines.csv')
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'
ODDS_API_DEFAULT_SPORT = 'baseball_mlb'
LOCAL_TZ = clean.LOCAL_TZ
UTC_TZ = dt.timezone.utc
_OPENING_DAY_CACHE: Dict[int, Optional[dt.date]] = {}

MARKET_CSV_COLUMNS = [
    'report_date', 'game_pk', 'away_team', 'home_team', 'sportsbook',
    'home_ml', 'away_ml', 'home_run_line', 'home_run_line_price',
    'away_run_line', 'away_run_line_price', 'total_line', 'over_price', 'under_price',
    'is_closing', 'source', 'snapshot_ts',
]

MANUAL_MARKET_TEMPLATE_COLUMNS = [
    'report_date', 'game_pk', 'away_team', 'home_team', 'venue', 'start_time_et',
    'sportsbook', 'home_ml', 'away_ml', 'home_run_line', 'home_run_line_price',
    'away_run_line', 'away_run_line_price', 'total_line', 'over_price', 'under_price',
    'snapshot_ts', 'source', 'notes',
]

MLB_TEAM_NAME_ALIASES = {
    'ARIZONA DIAMONDBACKS': 'AZ', 'DIAMONDBACKS': 'AZ',
    'ATLANTA BRAVES': 'ATL', 'BRAVES': 'ATL',
    'BALTIMORE ORIOLES': 'BAL', 'ORIOLES': 'BAL',
    'BOSTON RED SOX': 'BOS', 'RED SOX': 'BOS',
    'CHICAGO CUBS': 'CHC', 'CUBS': 'CHC',
    'CHICAGO WHITE SOX': 'CWS', 'WHITE SOX': 'CWS',
    'CINCINNATI REDS': 'CIN', 'REDS': 'CIN',
    'CLEVELAND GUARDIANS': 'CLE', 'GUARDIANS': 'CLE',
    'COLORADO ROCKIES': 'COL', 'ROCKIES': 'COL',
    'DETROIT TIGERS': 'DET', 'TIGERS': 'DET',
    'HOUSTON ASTROS': 'HOU', 'ASTROS': 'HOU',
    'KANSAS CITY ROYALS': 'KC', 'ROYALS': 'KC',
    'LOS ANGELES ANGELS': 'LAA', 'LA ANGELS': 'LAA', 'ANGELS': 'LAA',
    'LOS ANGELES DODGERS': 'LAD', 'LA DODGERS': 'LAD', 'DODGERS': 'LAD',
    'MIAMI MARLINS': 'MIA', 'MARLINS': 'MIA',
    'MILWAUKEE BREWERS': 'MIL', 'BREWERS': 'MIL',
    'MINNESOTA TWINS': 'MIN', 'TWINS': 'MIN',
    'NEW YORK METS': 'NYM', 'METS': 'NYM',
    'NEW YORK YANKEES': 'NYY', 'YANKEES': 'NYY',
    'ATHLETICS': 'ATH', 'OAKLAND ATHLETICS': 'ATH', 'SACRAMENTO ATHLETICS': 'ATH', 'AS': 'ATH',
    'PHILADELPHIA PHILLIES': 'PHI', 'PHILLIES': 'PHI',
    'PITTSBURGH PIRATES': 'PIT', 'PIRATES': 'PIT',
    'SAN DIEGO PADRES': 'SD', 'PADRES': 'SD',
    'SAN FRANCISCO GIANTS': 'SF', 'GIANTS': 'SF',
    'SEATTLE MARINERS': 'SEA', 'MARINERS': 'SEA',
    'ST LOUIS CARDINALS': 'STL', 'SAINT LOUIS CARDINALS': 'STL', 'CARDINALS': 'STL',
    'TAMPA BAY RAYS': 'TB', 'RAYS': 'TB',
    'TEXAS RANGERS': 'TEX', 'RANGERS': 'TEX',
    'TORONTO BLUE JAYS': 'TOR', 'BLUE JAYS': 'TOR',
    'WASHINGTON NATIONALS': 'WSH', 'NATIONALS': 'WSH',
    'SPRINGFIELD CARDINALS': 'SPR', 'SPRINGFIELD': 'SPR',
    'SUGAR LAND SPACE COWBOYS': 'SUG', 'SPACE COWBOYS': 'SUG', 'SUGAR LAND': 'SUG',
    'SULTANES DE MONTERREY': 'MTY', 'MONTERREY SULTANES': 'MTY', 'MONTERREY': 'MTY', 'SULTANES': 'MTY',
}
MLB_TEAM_ABBREVIATIONS = sorted(set(MLB_TEAM_NAME_ALIASES.values()))


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def today_local() -> dt.date:
    return clean.today_local()


def safe_float(value) -> Optional[float]:
    return clean.safe_float(value)


def finite(value, fallback=0.0) -> float:
    out = safe_float(value)
    return fallback if out is None else float(out)


def parse_market_int(value) -> Optional[int]:
    parsed = safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return int(round(parsed))


def parse_market_float(value) -> Optional[float]:
    parsed = safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def parse_utc_datetime(value) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        return value.astimezone(UTC_TZ) if value.tzinfo else value.replace(tzinfo=UTC_TZ)
    raw = str(value or '').strip()
    if not raw:
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw.replace('Z', '+00:00'))
        return parsed.astimezone(UTC_TZ) if parsed.tzinfo else parsed.replace(tzinfo=UTC_TZ)
    except Exception:
        return None


def opening_day_for_season(season: int) -> Optional[dt.date]:
    season = int(season)
    if season not in _OPENING_DAY_CACHE:
        opening_day = None
        try:
            response = requests.get(
                clean.STATSAPI_SCHEDULE_URL,
                params={'sportId': 1, 'gameType': 'R', 'startDate': f'{season}-03-01', 'endDate': f'{season}-04-30'},
                headers=clean.HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get('dates', []):
                day_value = str(item.get('date') or '').strip()
                if day_value:
                    opening_day = dt.date.fromisoformat(day_value)
                    break
        except Exception:
            opening_day = None
        _OPENING_DAY_CACHE[season] = opening_day
    return _OPENING_DAY_CACHE[season]


def season_phase_for_date(date_: dt.date) -> str:
    opening_day = opening_day_for_season(date_.year)
    if opening_day is not None:
        return 'spring' if date_ < opening_day else 'regular'
    # Fail open around likely Opening Day windows so transient StatsAPI failures
    # do not suppress live odds fetch on regular-season dates.
    if date_.month < 3:
        return 'spring'
    if date_.month == 3 and date_.day < 20:
        return 'spring'
    return 'regular'


def market_thresholds(market_type: str, sportsbook_count: int = 1, model_phase: Optional[str] = None) -> Dict[str, float]:
    phase_name = str(model_phase or 'regular').strip().lower()
    books = max(int(finite(sportsbook_count, 1)), 1)
    kind = str(market_type or '').strip().lower()
    if kind == 'total':
        if phase_name == 'spring':
            return {'watch': 1.00, 'bet': 1.25 if books >= 2 else 1.35, 'high': 1.50 if books >= 2 else 1.65}
        return {'watch': 1.10, 'bet': 1.60 if books >= 2 else 1.75, 'high': 2.10 if books >= 2 else 2.25}
    if phase_name == 'spring':
        return {'watch': 0.040, 'bet': 0.055 if books >= 2 else 0.065, 'high': 0.075 if books >= 2 else 0.085}
    return {'watch': 0.025, 'bet': 0.040 if books >= 2 else 0.050, 'high': 0.055 if books >= 2 else 0.065}


def recent_market_actionability_profile(
    market_type: str,
    report_date: Optional[dt.date] = None,
    lookback_days: int = 7,
    model_phase: Optional[str] = None,
) -> Dict[str, object]:
    kind = str(market_type or '').strip().lower()
    phase_name = str(model_phase or '').strip().lower() or None
    profile = {
        'market_type': kind,
        'lookback_days': int(max(lookback_days, 1)),
        'sample_bets': 0,
        'sample_days': 0,
        'roi': None,
        'avg_clv': None,
        'threshold_scale': 1.0,
        'action_cap': 'MUST TAKE',
        'throttle_label': 'neutral',
        'notes': 'recent live proof is still building; no additional market throttle applied',
    }
    if not os.path.exists(SPORTSBOOK_DB_PATH):
        return profile
    target_date = report_date or today_local()
    cutoff = target_date - dt.timedelta(days=max(int(lookback_days), 1))
    try:
        with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
            query = """
                SELECT report_date, phase, market_type, stake_units, result_units, clv_value
                FROM bet_journal
                WHERE result_units IS NOT NULL
                  AND market_type = ?
                  AND report_date >= ?
                  AND report_date < ?
            """
            params: List[object] = [kind, cutoff.isoformat(), target_date.isoformat()]
            if phase_name:
                query += " AND LOWER(COALESCE(phase, '')) = ?"
                params.append(phase_name)
            journal = pd.read_sql_query(query, conn, params=params)
    except Exception:
        return profile
    if journal.empty:
        return profile
    journal['stake_units'] = pd.to_numeric(journal.get('stake_units'), errors='coerce').fillna(1.0)
    journal['result_units'] = pd.to_numeric(journal.get('result_units'), errors='coerce')
    journal['clv_value'] = pd.to_numeric(journal.get('clv_value'), errors='coerce')
    journal['report_date'] = pd.to_datetime(journal.get('report_date'), errors='coerce').dt.date
    journal = journal[journal['result_units'].notna()].copy()
    if journal.empty:
        return profile

    sample_bets = int(len(journal))
    sample_days = int(journal['report_date'].dropna().nunique())
    total_stake = float(journal['stake_units'].sum())
    units = float(journal['result_units'].sum())
    roi = (units / total_stake) if total_stake > 1e-9 else None
    avg_clv = float(journal['clv_value'].dropna().mean()) if journal['clv_value'].notna().any() else None

    min_bets = {'moneyline': 10, 'run_line': 12, 'total': 10}.get(kind, 10)
    threshold_scale = 1.0
    action_cap = 'MUST TAKE'
    throttle_label = 'neutral'
    notes = 'recent live proof is stable enough that no extra throttle is needed'
    if sample_bets < min_bets:
        notes = f"recent live proof is still building ({sample_bets}/{min_bets} settled bets), so no market-specific cap is applied yet"
    else:
        clv_value = finite(avg_clv, 0.0)
        roi_value = finite(roi, 0.0)
        severe_negative = clv_value <= {'moneyline': -0.10, 'run_line': -0.12, 'total': -0.12}.get(kind, -0.10) or roi_value <= {'moneyline': -0.05, 'run_line': -0.05, 'total': -0.04}.get(kind, -0.05)
        moderate_negative = clv_value <= {'moneyline': -0.04, 'run_line': -0.06, 'total': -0.05}.get(kind, -0.04) or roi_value <= {'moneyline': -0.02, 'run_line': -0.02, 'total': -0.01}.get(kind, -0.02)
        strong_positive = clv_value >= {'moneyline': 0.035, 'run_line': 0.045, 'total': 0.045}.get(kind, 0.035) and roi_value >= 0.02

        if kind == 'total':
            if severe_negative:
                threshold_scale = 1.45
                action_cap = 'WATCH'
                throttle_label = 'heavy throttle'
                notes = 'recent totals CLV/ROI are weak, so totals are capped at WATCH and require a materially bigger edge plus cleaner game conditions'
            elif moderate_negative:
                threshold_scale = 1.22
                action_cap = 'BET'
                throttle_label = 'soft throttle'
                notes = 'recent totals CLV/ROI are shaky, so totals need a larger edge, cleaner setup quality, and cannot be promoted past BET'
        elif kind == 'run_line':
            if severe_negative:
                threshold_scale = 1.28
                action_cap = 'BET'
                throttle_label = 'firm throttle'
                notes = 'recent run-line CLV is weak, so run lines need a larger edge, stronger starter control, and stay capped below MUST TAKE'
            elif moderate_negative:
                threshold_scale = 1.14
                throttle_label = 'soft throttle'
                notes = 'recent run-line proof is mixed, so thresholds are tightened and fragile setups should stay below BET-quality confidence'
            elif strong_positive:
                threshold_scale = 0.97
                notes = 'recent run-line proof is constructive, so thresholds are allowed to stay near baseline'
        else:
            if severe_negative:
                threshold_scale = 1.12
                action_cap = 'BET'
                throttle_label = 'soft throttle'
                notes = 'recent moneyline CLV/ROI softened, so moneyline promotions are capped below MUST TAKE'
            elif moderate_negative:
                threshold_scale = 1.06
                notes = 'recent moneyline proof is slightly soft, so thresholds are nudged tighter'
            elif strong_positive:
                threshold_scale = 0.98
                notes = 'recent moneyline proof is constructive, so baseline thresholds remain largely intact'

    profile.update({
        'sample_bets': sample_bets,
        'sample_days': sample_days,
        'roi': roi,
        'avg_clv': avg_clv,
        'threshold_scale': float(np.clip(threshold_scale, 0.90, 1.50)),
        'action_cap': action_cap,
        'throttle_label': throttle_label,
        'notes': notes,
    })
    return profile


def market_signal_label(market_type: str, edge_value: Optional[float], sportsbook_count: int = 1, model_phase: Optional[str] = None) -> str:
    edge = abs(finite(edge_value, 0.0))
    thresholds = market_thresholds(market_type, sportsbook_count=sportsbook_count, model_phase=model_phase)
    if edge >= thresholds['high']:
        return 'MUST TAKE'
    if edge >= thresholds['bet']:
        return 'BET'
    if edge >= thresholds['watch']:
        return 'WATCH'
    return 'PASS'


def american_implied_probability(price: Optional[int]) -> Optional[float]:
    american = parse_market_int(price)
    if american is None:
        return None
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def no_vig_probabilities(home_price: Optional[int], away_price: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    home_prob = american_implied_probability(home_price)
    away_prob = american_implied_probability(away_price)
    if home_prob is None or away_prob is None:
        return None, None
    total = home_prob + away_prob
    if total <= 0:
        return None, None
    return home_prob / total, away_prob / total


def market_game_key(game_pk: Optional[int], away_team: Optional[str], home_team: Optional[str]) -> str:
    try:
        if game_pk is not None and str(game_pk).strip() != '':
            return f'pk:{int(game_pk)}'
    except Exception:
        pass
    away = str(away_team or '').strip().upper()
    home = str(home_team or '').strip().upper()
    return f'teams:{away}@{home}'


def manual_market_template_path(report_date: dt.date) -> str:
    return os.path.join(OUT_DIR, f'Manual_Sportsbook_Lines_{report_date.isoformat()}.csv')


def ensure_market_csv_template() -> None:
    if os.path.exists(SPORTSBOOK_CSV_PATH) and os.path.getsize(SPORTSBOOK_CSV_PATH) > 0:
        return
    pd.DataFrame(columns=MARKET_CSV_COLUMNS).to_csv(SPORTSBOOK_CSV_PATH, index=False)


def _game_attr(game, key: str, fallback=None):
    if hasattr(game, key):
        return getattr(game, key)
    if isinstance(game, dict):
        if key in game:
            return game.get(key, fallback)
        alias_map = {
            'away': 'away_team',
            'home': 'home_team',
            'venue': 'venue_name',
            'start_time_et': 'start_time_ct',
            'scheduled_dt_utc': 'scheduled_utc',
        }
        alias = alias_map.get(key)
        if alias:
            return game.get(alias, fallback)
    return fallback

def build_manual_market_template_rows(report_date: dt.date, games) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for g in games:
        rows.append({
            'report_date': report_date.isoformat(),
            'game_pk': int(_game_attr(g, 'game_pk', 0) or 0),
            'away_team': str(_game_attr(g, 'away', '') or '').strip().upper(),
            'home_team': str(_game_attr(g, 'home', '') or '').strip().upper(),
            'venue': str(_game_attr(g, 'venue', '') or ''),
            'start_time_et': str(_game_attr(g, 'start_time_et', '') or ''),
            'sportsbook': 'ManualConsensus',
            'home_ml': '', 'away_ml': '', 'home_run_line': '', 'home_run_line_price': '',
            'away_run_line': '', 'away_run_line_price': '', 'total_line': '', 'over_price': '', 'under_price': '',
            'snapshot_ts': '', 'source': 'manual_template', 'notes': '',
        })
    return rows


def ensure_manual_market_template(report_date: dt.date, games) -> Tuple[str, bool]:
    path = manual_market_template_path(report_date)
    template_df = pd.DataFrame(build_manual_market_template_rows(report_date, games), columns=MANUAL_MARKET_TEMPLATE_COLUMNS)
    created_or_updated = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame(columns=MANUAL_MARKET_TEMPLATE_COLUMNS)
        for col in MANUAL_MARKET_TEMPLATE_COLUMNS:
            if col not in existing.columns:
                existing[col] = ''
        existing = existing[MANUAL_MARKET_TEMPLATE_COLUMNS].copy()
        existing['report_date'] = report_date.isoformat()
        existing['sportsbook'] = existing['sportsbook'].fillna('ManualConsensus').astype(str).str.strip()
        existing.loc[existing['sportsbook'] == '', 'sportsbook'] = 'ManualConsensus'
        existing['source'] = existing['source'].fillna('manual_template').astype(str).str.strip()
        existing.loc[existing['source'] == '', 'source'] = 'manual_template'
        existing['away_team'] = existing['away_team'].fillna('').astype(str).str.upper().str.strip()
        existing['home_team'] = existing['home_team'].fillna('').astype(str).str.upper().str.strip()
        existing['game_pk'] = pd.to_numeric(existing['game_pk'], errors='coerce').fillna(0).astype(int)
        existing_keys = {
            (int(row['game_pk']), str(row['away_team']), str(row['home_team']))
            for _, row in existing.iterrows()
        }
        additions = []
        for _, row in template_df.iterrows():
            key = (int(row['game_pk']), str(row['away_team']), str(row['home_team']))
            if key not in existing_keys:
                additions.append(row.to_dict())
        if additions:
            created_or_updated = True
            existing = pd.concat([existing, pd.DataFrame(additions, columns=MANUAL_MARKET_TEMPLATE_COLUMNS)], ignore_index=True, sort=False)
        final_df = existing[MANUAL_MARKET_TEMPLATE_COLUMNS].copy()
    else:
        final_df = template_df.copy()
        created_or_updated = True
    if not final_df.empty:
        final_df['game_pk_sort'] = pd.to_numeric(final_df['game_pk'], errors='coerce').fillna(0).astype(int)
        final_df['start_sort'] = final_df['start_time_et'].fillna('').astype(str)
        final_df = final_df.sort_values(['start_sort', 'away_team', 'home_team', 'sportsbook', 'game_pk_sort']).drop(columns=['game_pk_sort', 'start_sort'])
        final_df = final_df.drop_duplicates(subset=['report_date', 'game_pk', 'away_team', 'home_team'], keep='last').reset_index(drop=True)
    final_df.to_csv(path, index=False)
    return path, created_or_updated


def populate_manual_template_from_consensus(report_date: dt.date, games, market_map: Dict[str, Dict[str, object]]) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    if not market_map:
        return 0, warnings
    path, _ = ensure_manual_market_template(report_date, games)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return 0, [f"Manual template auto-population failed: {exc}"]
    for col in MANUAL_MARKET_TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    df = df[MANUAL_MARKET_TEMPLATE_COLUMNS].copy()
    df = df.drop_duplicates(subset=['report_date', 'game_pk', 'away_team', 'home_team'], keep='last').reset_index(drop=True)
    for col in [
        'sportsbook', 'home_ml', 'away_ml', 'home_run_line', 'home_run_line_price',
        'away_run_line', 'away_run_line_price', 'total_line', 'over_price', 'under_price',
        'source', 'snapshot_ts',
    ]:
        df[col] = df[col].astype(object)
    report_iso = report_date.isoformat()
    df['report_date'] = report_iso
    updated = 0

    for idx, row in df.iterrows():
        game_pk = parse_market_int(row.get('game_pk'))
        key = market_game_key(game_pk if game_pk and game_pk > 0 else None, row.get('away_team'), row.get('home_team'))
        market = market_map.get(key)
        if not market:
            continue
        before = (
            row.get('sportsbook'),
            row.get('home_ml'),
            row.get('away_ml'),
            row.get('home_run_line'),
            row.get('home_run_line_price'),
            row.get('away_run_line'),
            row.get('away_run_line_price'),
            row.get('total_line'),
            row.get('over_price'),
            row.get('under_price'),
            row.get('source'),
            row.get('snapshot_ts'),
        )
        df.at[idx, 'sportsbook'] = str(market.get('market_label') or 'OddsAPIConsensus')
        df.at[idx, 'home_ml'] = '' if parse_market_int(market.get('home_ml')) is None else parse_market_int(market.get('home_ml'))
        df.at[idx, 'away_ml'] = '' if parse_market_int(market.get('away_ml')) is None else parse_market_int(market.get('away_ml'))
        df.at[idx, 'home_run_line'] = '' if parse_market_float(market.get('home_run_line')) is None else parse_market_float(market.get('home_run_line'))
        df.at[idx, 'home_run_line_price'] = '' if parse_market_int(market.get('home_run_line_price')) is None else parse_market_int(market.get('home_run_line_price'))
        df.at[idx, 'away_run_line'] = '' if parse_market_float(market.get('away_run_line')) is None else parse_market_float(market.get('away_run_line'))
        df.at[idx, 'away_run_line_price'] = '' if parse_market_int(market.get('away_run_line_price')) is None else parse_market_int(market.get('away_run_line_price'))
        df.at[idx, 'total_line'] = '' if parse_market_float(market.get('total_line')) is None else parse_market_float(market.get('total_line'))
        df.at[idx, 'over_price'] = '' if parse_market_int(market.get('over_price')) is None else parse_market_int(market.get('over_price'))
        df.at[idx, 'under_price'] = '' if parse_market_int(market.get('under_price')) is None else parse_market_int(market.get('under_price'))
        df.at[idx, 'source'] = 'odds_api_consensus'
        df.at[idx, 'snapshot_ts'] = str(market.get('snapshot_ts') or dt.datetime.now(UTC_TZ).isoformat().replace('+00:00', 'Z'))
        after = (
            df.at[idx, 'sportsbook'],
            df.at[idx, 'home_ml'],
            df.at[idx, 'away_ml'],
            df.at[idx, 'home_run_line'],
            df.at[idx, 'home_run_line_price'],
            df.at[idx, 'away_run_line'],
            df.at[idx, 'away_run_line_price'],
            df.at[idx, 'total_line'],
            df.at[idx, 'over_price'],
            df.at[idx, 'under_price'],
            df.at[idx, 'source'],
            df.at[idx, 'snapshot_ts'],
        )
        if before != after:
            updated += 1
    if updated:
        df.to_csv(path, index=False)
        warnings.append(f"Auto-populated {updated} manual template row(s) from Odds API consensus.")
    return updated, warnings


def ensure_market_db() -> None:
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_ts TEXT NOT NULL,
                report_date TEXT NOT NULL,
                phase TEXT,
                game_pk INTEGER NOT NULL DEFAULT 0,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                sportsbook_key TEXT,
                provider_event_id TEXT,
                provider_sport_key TEXT,
                bookmaker_last_update TEXT,
                odds_api_pull_ts TEXT,
                odds_api_requests_remaining INTEGER,
                odds_api_requests_used INTEGER,
                odds_api_requests_last INTEGER,
                home_ml INTEGER,
                away_ml INTEGER,
                home_run_line REAL,
                home_run_line_price INTEGER,
                away_run_line REAL,
                away_run_line_price INTEGER,
                total_line REAL,
                over_price INTEGER,
                under_price INTEGER,
                is_closing INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual_csv',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_odds_snapshots_lookup
                ON odds_snapshots (report_date, game_pk, away_team, home_team, sportsbook, snapshot_ts);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_snapshots_dedupe
                ON odds_snapshots (snapshot_ts, report_date, game_pk, away_team, home_team, sportsbook, source);
            CREATE TABLE IF NOT EXISTS bet_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts TEXT NOT NULL,
                report_date TEXT NOT NULL,
                game_pk INTEGER NOT NULL DEFAULT 0,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                sportsbook TEXT,
                market_type TEXT NOT NULL,
                selection TEXT NOT NULL,
                line_value REAL,
                book_price INTEGER,
                model_price INTEGER,
                model_line_value REAL,
                model_probability REAL,
                market_probability REAL,
                edge_value REAL,
                edge_kind TEXT,
                confidence_tier TEXT,
                actionability_label TEXT,
                actionability_score REAL,
                actionability_notes TEXT,
                status TEXT NOT NULL DEFAULT 'watch',
                stake_units REAL,
                notes TEXT,
                result_units REAL,
                clv_value REAL,
                settled_ts TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_bet_journal_watch_unique
                ON bet_journal (
                    report_date, game_pk, away_team, home_team, sportsbook, market_type, selection,
                    IFNULL(line_value, 0), IFNULL(book_price, 0)
                );
            """
        )
        existing_cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(odds_snapshots)").fetchall()}
        for column_name, column_type in [
            ('phase', 'TEXT'),
            ('sportsbook_key', 'TEXT'),
            ('provider_event_id', 'TEXT'),
            ('provider_sport_key', 'TEXT'),
            ('bookmaker_last_update', 'TEXT'),
            ('odds_api_pull_ts', 'TEXT'),
            ('odds_api_requests_remaining', 'INTEGER'),
            ('odds_api_requests_used', 'INTEGER'),
            ('odds_api_requests_last', 'INTEGER'),
        ]:
            if column_name not in existing_cols:
                conn.execute(f'ALTER TABLE odds_snapshots ADD COLUMN {column_name} {column_type}')
        cols = {str(row[1]) for row in conn.execute('PRAGMA table_info(bet_journal)').fetchall()}
        if 'phase' not in cols:
            conn.execute('ALTER TABLE bet_journal ADD COLUMN phase TEXT')
        if 'model_line_value' not in cols:
            conn.execute('ALTER TABLE bet_journal ADD COLUMN model_line_value REAL')
        if 'actionability_label' not in cols:
            conn.execute('ALTER TABLE bet_journal ADD COLUMN actionability_label TEXT')
        if 'actionability_score' not in cols:
            conn.execute('ALTER TABLE bet_journal ADD COLUMN actionability_score REAL')
        if 'actionability_notes' not in cols:
            conn.execute('ALTER TABLE bet_journal ADD COLUMN actionability_notes TEXT')


def import_market_lines_csv(report_date: dt.date, csv_path: str = SPORTSBOOK_CSV_PATH, default_source: str = 'manual_csv') -> Tuple[int, List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    if os.path.abspath(csv_path) == os.path.abspath(SPORTSBOOK_CSV_PATH):
        ensure_market_csv_template()
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0, warnings
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return 0, warnings
    except Exception as exc:
        return 0, [f'Sportsbook CSV import failed: {exc}']
    if df.empty:
        return 0, warnings
    report_iso = report_date.isoformat()
    now_iso = dt.datetime.now(UTC_TZ).isoformat()
    if 'report_date' not in df.columns:
        df['report_date'] = report_iso
    else:
        df['report_date'] = df['report_date'].fillna(report_iso).astype(str).str.strip()
        df.loc[df['report_date'] == '', 'report_date'] = report_iso
    if 'snapshot_ts' not in df.columns:
        df['snapshot_ts'] = now_iso
    else:
        df['snapshot_ts'] = df['snapshot_ts'].fillna(now_iso).astype(str).str.strip()
        df.loc[df['snapshot_ts'] == '', 'snapshot_ts'] = now_iso
    if 'source' not in df.columns:
        df['source'] = default_source
    else:
        df['source'] = df['source'].fillna(default_source).astype(str).str.strip()
        df.loc[df['source'] == '', 'source'] = default_source
    if 'is_closing' not in df.columns:
        df['is_closing'] = 0
    target = df[df['report_date'].astype(str) == report_iso].copy()
    if target.empty:
        return 0, warnings
    required = ['away_team', 'home_team', 'sportsbook']
    missing = [col for col in required if col not in target.columns]
    if missing:
        return 0, [f"Sportsbook CSV missing required columns: {', '.join(missing)}"]
    records: List[Tuple] = []
    source_values: set[str] = set()
    skipped = 0
    for _, row in target.iterrows():
        away_team = str(row.get('away_team', '') or '').strip().upper()
        home_team = str(row.get('home_team', '') or '').strip().upper()
        sportsbook = str(row.get('sportsbook', '') or '').strip()
        if not away_team or not home_team or not sportsbook:
            skipped += 1
            continue
        home_ml = parse_market_int(row.get('home_ml'))
        away_ml = parse_market_int(row.get('away_ml'))
        home_run_line = parse_market_float(row.get('home_run_line'))
        home_run_line_price = parse_market_int(row.get('home_run_line_price'))
        away_run_line = parse_market_float(row.get('away_run_line'))
        away_run_line_price = parse_market_int(row.get('away_run_line_price'))
        total_line = parse_market_float(row.get('total_line'))
        over_price = parse_market_int(row.get('over_price'))
        under_price = parse_market_int(row.get('under_price'))
        if all(v is None for v in [home_ml, away_ml, home_run_line, home_run_line_price, away_run_line, away_run_line_price, total_line, over_price, under_price]):
            skipped += 1
            continue
        game_pk = parse_market_int(row.get('game_pk')) or 0
        source_label = str(row.get('source', default_source) or default_source).strip() or default_source
        phase_name = season_phase_for_date(report_date)
        source_values.add(source_label)
        records.append((
            str(row.get('snapshot_ts')), report_iso, phase_name, game_pk, away_team, home_team, sportsbook,
            None, None, None, None, None, None, None, None,
            home_ml, away_ml, home_run_line, home_run_line_price, away_run_line, away_run_line_price,
            total_line, over_price, under_price, int(bool(parse_market_int(row.get('is_closing')) or 0)), source_label,
        ))
    if not records:
        if skipped:
            warnings.append(f"{os.path.basename(csv_path)} skipped {skipped} incomplete or blank row(s) for {report_iso}.")
        return 0, warnings
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        for source_label in sorted(source_values):
            conn.execute('DELETE FROM odds_snapshots WHERE report_date = ? AND source = ?', (report_iso, source_label))
        conn.executemany(
            """
            INSERT INTO odds_snapshots (
                snapshot_ts, report_date, phase, game_pk, away_team, home_team, sportsbook,
                sportsbook_key, provider_event_id, provider_sport_key, bookmaker_last_update,
                odds_api_pull_ts, odds_api_requests_remaining, odds_api_requests_used, odds_api_requests_last,
                home_ml, away_ml, home_run_line, home_run_line_price,
                away_run_line, away_run_line_price, total_line, over_price, under_price,
                is_closing, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    if skipped:
        warnings.append(f"{os.path.basename(csv_path)} skipped {skipped} incomplete or blank row(s) for {report_iso}.")
    return len(records), warnings


def import_all_market_lines(report_date: dt.date, games) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    ensure_market_csv_template()
    template_path, template_updated = ensure_manual_market_template(report_date, games)
    imported_csv, csv_warn = import_market_lines_csv(report_date, SPORTSBOOK_CSV_PATH, default_source='manual_csv')
    warnings.extend(csv_warn)
    imported_template, template_warn = import_market_lines_csv(report_date, template_path, default_source='manual_template')
    if int(imported_template) == 0:
        template_warn = [w for w in template_warn if not (os.path.basename(template_path) in w and 'skipped' in w.lower())]
    warnings.extend(template_warn)
    warnings.append(f'Manual market template ready: {os.path.basename(template_path)}')
    return {
        'manual_csv': int(imported_csv), 'manual_template': int(imported_template), 'total': int(imported_csv + imported_template),
        'template_path': template_path, 'template_updated': bool(template_updated),
    }, warnings


def _median_value(items: List[Optional[float]], integer: bool = False):
    vals = [float(v) for v in items if v is not None and np.isfinite(v)]
    if not vals:
        return None
    med = float(np.median(np.array(vals, dtype=float)))
    return int(round(med)) if integer else med


def load_market_consensus_for_date(report_date: dt.date, games=None) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    report_iso = report_date.isoformat()
    desired_keys = set()
    if games:
        desired_keys = {market_game_key(_game_attr(g, 'game_pk'), _game_attr(g, 'away'), _game_attr(g, 'home')) for g in games}
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM odds_snapshots
            WHERE report_date = ?
            ORDER BY is_closing DESC, snapshot_ts DESC, id DESC
            """,
            (report_iso,),
        ).fetchall()
    grouped: Dict[str, Dict[str, sqlite3.Row]] = {}
    for row in rows:
        row_game_pk = int(row['game_pk']) if row['game_pk'] is not None else 0
        key = market_game_key(row_game_pk if row_game_pk > 0 else None, row['away_team'], row['home_team'])
        if desired_keys and key not in desired_keys:
            continue
        sportsbook = str(row['sportsbook'] or '').strip()
        if not sportsbook:
            continue
        grouped.setdefault(key, {})
        if sportsbook not in grouped[key]:
            grouped[key][sportsbook] = row
    out: Dict[str, Dict[str, object]] = {}
    for key, book_rows in grouped.items():
        selected = list(book_rows.values())
        if not selected:
            continue
        out[key] = {
            'market_label': f"Consensus ({len(selected)} book{'s' if len(selected) != 1 else ''})",
            'sportsbook_count': len(selected),
            'sportsbooks': sorted(str(r['sportsbook']) for r in selected),
            'away_team': str(selected[0]['away_team']), 'home_team': str(selected[0]['home_team']), 'game_pk': int(selected[0]['game_pk'] or 0),
            'home_ml': _median_value([parse_market_int(r['home_ml']) for r in selected], integer=True),
            'away_ml': _median_value([parse_market_int(r['away_ml']) for r in selected], integer=True),
            'home_run_line': _median_value([parse_market_float(r['home_run_line']) for r in selected]),
            'home_run_line_price': _median_value([parse_market_int(r['home_run_line_price']) for r in selected], integer=True),
            'away_run_line': _median_value([parse_market_float(r['away_run_line']) for r in selected]),
            'away_run_line_price': _median_value([parse_market_int(r['away_run_line_price']) for r in selected], integer=True),
            'total_line': _median_value([parse_market_float(r['total_line']) for r in selected]),
            'over_price': _median_value([parse_market_int(r['over_price']) for r in selected], integer=True),
            'under_price': _median_value([parse_market_int(r['under_price']) for r in selected], integer=True),
            'snapshot_ts': max(str(r['snapshot_ts']) for r in selected),
        }
    return out, warnings


def normalize_team_name_key(value: Optional[object]) -> str:
    txt = str(value or '').strip().upper().replace('&', 'AND')
    txt = re.sub(r'[^A-Z0-9 ]+', '', txt)
    txt = re.sub(r'\bSPLIT\s+SQUAD\b', ' ', txt)
    txt = re.sub(r'\bSPLIT\b', ' ', txt)
    txt = re.sub(r'\bSQUAD\b', ' ', txt)
    txt = re.sub(r'\bSS\b', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def mlb_team_abbreviation(name: Optional[object]) -> Optional[str]:
    key = normalize_team_name_key(name)
    if not key:
        return None
    if key in MLB_TEAM_ABBREVIATIONS:
        return key
    return MLB_TEAM_NAME_ALIASES.get(key)


def team_match_keys(name: Optional[object]) -> set[str]:
    keys: set[str] = set()
    abbr = mlb_team_abbreviation(name)
    norm = normalize_team_name_key(name)
    if abbr:
        keys.add(abbr)
    if norm:
        keys.add(norm)
    for alias_name, alias_abbr in MLB_TEAM_NAME_ALIASES.items():
        if abbr and alias_abbr == abbr:
            keys.add(alias_name)
    return {k for k in keys if k}

def _select_latest_market_rows(rows: List[sqlite3.Row]) -> List[sqlite3.Row]:
    grouped: Dict[str, sqlite3.Row] = {}
    for row in rows:
        sportsbook = str(row['sportsbook'] or '').strip()
        if sportsbook and sportsbook not in grouped:
            grouped[sportsbook] = row
    return list(grouped.values())


def load_market_consensus_row_from_conn(conn: sqlite3.Connection, report_iso: str, away_team: str, home_team: str, game_pk: int = 0, closing_only: bool = False) -> Optional[Dict[str, object]]:
    conn.row_factory = sqlite3.Row
    if game_pk:
        rows = conn.execute(
            """
            SELECT *
            FROM odds_snapshots
            WHERE report_date = ? AND game_pk = ?
            ORDER BY is_closing DESC, snapshot_ts DESC, id DESC
            """,
            (report_iso, int(game_pk)),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM odds_snapshots
            WHERE report_date = ? AND away_team = ? AND home_team = ?
            ORDER BY is_closing DESC, snapshot_ts DESC, id DESC
            """,
            (report_iso, str(away_team).upper(), str(home_team).upper()),
        ).fetchall()
    if not rows:
        return None
    selected_rows = rows
    if closing_only:
        closing_rows = [row for row in rows if int(row['is_closing'] or 0) == 1]
        if closing_rows:
            selected_rows = closing_rows
    selected = _select_latest_market_rows(selected_rows)
    if not selected:
        return None
    return {
        'market_label': f"Consensus ({len(selected)} book{'s' if len(selected) != 1 else ''})",
        'sportsbook_count': len(selected),
        'sportsbooks': sorted(str(r['sportsbook']) for r in selected),
        'away_team': str(selected[0]['away_team']), 'home_team': str(selected[0]['home_team']), 'game_pk': int(selected[0]['game_pk'] or 0),
        'home_ml': _median_value([parse_market_int(r['home_ml']) for r in selected], integer=True),
        'away_ml': _median_value([parse_market_int(r['away_ml']) for r in selected], integer=True),
        'home_run_line': _median_value([parse_market_float(r['home_run_line']) for r in selected]),
        'home_run_line_price': _median_value([parse_market_int(r['home_run_line_price']) for r in selected], integer=True),
        'away_run_line': _median_value([parse_market_float(r['away_run_line']) for r in selected]),
        'away_run_line_price': _median_value([parse_market_int(r['away_run_line_price']) for r in selected], integer=True),
        'total_line': _median_value([parse_market_float(r['total_line']) for r in selected]),
        'over_price': _median_value([parse_market_int(r['over_price']) for r in selected], integer=True),
        'under_price': _median_value([parse_market_int(r['under_price']) for r in selected], integer=True),
        'snapshot_ts': max(str(r['snapshot_ts']) for r in selected),
    }


def match_market_event_to_game(event: Dict[str, object], games) -> Dict[str, object]:
    away_abbr = mlb_team_abbreviation(event.get('away_team')) or normalize_team_name_key(event.get('away_team'))[:8]
    home_abbr = mlb_team_abbreviation(event.get('home_team')) or normalize_team_name_key(event.get('home_team'))[:8]
    away_keys = team_match_keys(event.get('away_team'))
    home_keys = team_match_keys(event.get('home_team'))
    candidates = []
    for g in games:
        game_away = str(_game_attr(g, 'away', '')).upper()
        game_home = str(_game_attr(g, 'home', '')).upper()
        if game_away == away_abbr and game_home == home_abbr:
            candidates.append(g)
            continue
        game_away_keys = team_match_keys(game_away)
        game_home_keys = team_match_keys(game_home)
        if away_keys.intersection(game_away_keys) and home_keys.intersection(game_home_keys):
            candidates.append(g)
    if not candidates:
        return {'game_pk': 0, 'away': away_abbr, 'home': home_abbr}
    if len(candidates) == 1:
        g = candidates[0]
        return {'game_pk': int(_game_attr(g, 'game_pk', 0) or 0), 'away': _game_attr(g, 'away', away_abbr), 'home': _game_attr(g, 'home', home_abbr)}
    event_dt = parse_utc_datetime(event.get('commence_time'))
    if event_dt is not None:
        def game_time_distance(game) -> float:
            game_dt = parse_utc_datetime(_game_attr(game, 'scheduled_dt_utc'))
            if game_dt is None:
                return 1e18
            return abs((game_dt - event_dt).total_seconds())
        g = sorted(candidates, key=game_time_distance)[0]
    else:
        g = candidates[0]
    return {'game_pk': int(_game_attr(g, 'game_pk', 0) or 0), 'away': _game_attr(g, 'away', away_abbr), 'home': _game_attr(g, 'home', home_abbr)}


def parse_odds_api_book_row(event: Dict[str, object], bookmaker: Dict[str, object], games, report_date: dt.date) -> Optional[Tuple]:
    matched = match_market_event_to_game(event, games)
    away_team = str(matched.get('away') or '').strip().upper()
    home_team = str(matched.get('home') or '').strip().upper()
    game_pk = int(matched.get('game_pk') or 0)
    if not away_team or not home_team or game_pk <= 0:
        return None
    snapshot_ts = str(bookmaker.get('last_update') or event.get('commence_time') or dt.datetime.now(UTC_TZ).isoformat())
    sportsbook = str(bookmaker.get('title') or bookmaker.get('key') or '').strip()
    sportsbook_key = str(bookmaker.get('key') or '').strip() or None
    provider_event_id = str(event.get('id') or '').strip() or None
    provider_sport_key = str(event.get('sport_key') or '').strip() or None
    bookmaker_last_update = str(bookmaker.get('last_update') or '').strip() or None
    if not sportsbook:
        return None
    row = {
        'home_ml': None, 'away_ml': None, 'home_run_line': None, 'home_run_line_price': None,
        'away_run_line': None, 'away_run_line_price': None, 'total_line': None, 'over_price': None, 'under_price': None,
    }
    for market in bookmaker.get('markets') or []:
        market_key = str((market or {}).get('key') or '').strip().lower()
        outcomes = market.get('outcomes') or []
        if market_key == 'h2h':
            for outcome in outcomes:
                team_abbr = mlb_team_abbreviation(outcome.get('name'))
                price = parse_market_int(outcome.get('price'))
                if team_abbr == home_team:
                    row['home_ml'] = price
                elif team_abbr == away_team:
                    row['away_ml'] = price
        elif market_key == 'spreads':
            for outcome in outcomes:
                team_abbr = mlb_team_abbreviation(outcome.get('name'))
                point = parse_market_float(outcome.get('point'))
                price = parse_market_int(outcome.get('price'))
                if team_abbr == home_team:
                    row['home_run_line'] = point
                    row['home_run_line_price'] = price
                elif team_abbr == away_team:
                    row['away_run_line'] = point
                    row['away_run_line_price'] = price
        elif market_key == 'totals':
            for outcome in outcomes:
                label = str(outcome.get('name') or '').strip().lower()
                point = parse_market_float(outcome.get('point'))
                price = parse_market_int(outcome.get('price'))
                if label == 'over':
                    row['total_line'] = point
                    row['over_price'] = price
                elif label == 'under':
                    if row['total_line'] is None:
                        row['total_line'] = point
                    row['under_price'] = price
    if all(row[key] is None for key in row):
        return None
    return (
        snapshot_ts, report_date.isoformat(), season_phase_for_date(report_date), game_pk, away_team, home_team, sportsbook,
        sportsbook_key, provider_event_id, provider_sport_key, bookmaker_last_update,
        None, None, None, None,
        row['home_ml'], row['away_ml'], row['home_run_line'], row['home_run_line_price'],
        row['away_run_line'], row['away_run_line_price'], row['total_line'], row['over_price'], row['under_price'],
        0, 'odds_api',
    )


def fetch_market_lines_from_odds_api(report_date: dt.date, games) -> Tuple[int, List[str]]:
    api_key = str(os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY') or '').strip()
    if not api_key:
        return 0, []
    sport = str(os.getenv('ODDS_API_SPORT') or ODDS_API_DEFAULT_SPORT).strip() or ODDS_API_DEFAULT_SPORT
    regions = str(os.getenv('ODDS_API_REGIONS') or 'us').strip() or 'us'
    markets = str(os.getenv('ODDS_API_MARKETS') or 'h2h,spreads,totals').strip() or 'h2h,spreads,totals'
    bookmakers = str(os.getenv('ODDS_API_BOOKMAKERS') or '').strip()
    event_times = [parse_utc_datetime(_game_attr(g, 'scheduled_dt_utc')) for g in games]
    event_times = [t for t in event_times if t is not None]
    commence_from = None
    commence_to = None
    if event_times:
        commence_from = (min(event_times) - dt.timedelta(hours=6)).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        commence_to = (max(event_times) + dt.timedelta(hours=6)).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'includeRotationNumbers': 'true',
    }
    if bookmakers:
        params['bookmakers'] = bookmakers
    if commence_from:
        params['commenceTimeFrom'] = commence_from
    if commence_to:
        params['commenceTimeTo'] = commence_to
    warnings: List[str] = []
    try:
        response = requests.get(f'{ODDS_API_BASE_URL}/sports/{sport}/odds', params=params, headers=clean.HEADERS, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return 0, [f'Live sportsbook fetch failed: {exc}']
    if not isinstance(payload, list):
        return 0, ['Live sportsbook fetch returned an unexpected payload.']
    pull_ts = dt.datetime.now(UTC_TZ).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    requests_remaining = parse_market_int(response.headers.get('x-requests-remaining'))
    requests_used = parse_market_int(response.headers.get('x-requests-used'))
    requests_last = parse_market_int(response.headers.get('x-requests-last'))
    records: List[Tuple] = []
    unmatched_books = 0
    for event in payload:
        if not isinstance(event, dict):
            continue
        for bookmaker in event.get('bookmakers') or []:
            if not isinstance(bookmaker, dict):
                continue
            record = parse_odds_api_book_row(event, bookmaker, games, report_date)
            if record is None:
                unmatched_books += 1
                continue
            records.append((
                record[0], record[1], record[2], record[3], record[4], record[5], record[6],
                record[7], record[8], record[9], record[10],
                pull_ts, requests_remaining, requests_used, requests_last,
                record[15], record[16], record[17], record[18], record[19], record[20], record[21], record[22], record[23], record[24], record[25],
            ))
    if not records:
        if not payload:
            warnings.append('Live sportsbook fetch returned 0 events in the filtered slate window.')
        if unmatched_books:
            warnings.append(f"Live sportsbook fetch returned {unmatched_books} bookmaker row(s), but none matched today's scheduled games.")
        return 0, warnings
    ensure_market_db()
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO odds_snapshots (
                snapshot_ts, report_date, phase, game_pk, away_team, home_team, sportsbook,
                sportsbook_key, provider_event_id, provider_sport_key, bookmaker_last_update,
                odds_api_pull_ts, odds_api_requests_remaining, odds_api_requests_used, odds_api_requests_last,
                home_ml, away_ml, home_run_line, home_run_line_price,
                away_run_line, away_run_line_price, total_line, over_price, under_price,
                is_closing, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    if unmatched_books:
        warnings.append(f'Live sportsbook fetch skipped {unmatched_books} unmatched bookmaker row(s).')
    if requests_last is not None:
        warnings.append(f'Live sportsbook fetch credit cost: {requests_last}.')
    if requests_used is not None:
        warnings.append(f'Live sportsbook fetch credits used this period: {requests_used}.')
    if requests_remaining is not None:
        warnings.append(f'Live sportsbook fetch credits remaining: {requests_remaining}.')
    if commence_from and commence_to:
        warnings.append(f'Live sportsbook fetch filtered slate window: {commence_from} to {commence_to}.')
    return len(records), warnings


def mark_closing_market_snapshots(report_date: dt.date, games) -> Tuple[int, List[str]]:
    ensure_market_db()
    now_utc = dt.datetime.now(UTC_TZ)
    report_iso = report_date.isoformat()
    marked = 0
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for game in games:
            scheduled = parse_utc_datetime(_game_attr(game, 'scheduled_dt_utc'))
            if scheduled is None or scheduled > now_utc:
                continue
            away_team = str(_game_attr(game, 'away', '') or '').upper()
            home_team = str(_game_attr(game, 'home', '') or '').upper()
            rows = conn.execute(
                """
                SELECT id, sportsbook, snapshot_ts
                FROM odds_snapshots
                WHERE report_date = ? AND away_team = ? AND home_team = ?
                ORDER BY snapshot_ts DESC, id DESC
                """,
                (report_iso, away_team, home_team),
            ).fetchall()
            if not rows:
                continue
            conn.execute('UPDATE odds_snapshots SET is_closing = 0 WHERE report_date = ? AND away_team = ? AND home_team = ?', (report_iso, away_team, home_team))
            latest_by_book: Dict[str, int] = {}
            for row in rows:
                sportsbook = str(row['sportsbook'] or '').strip()
                if sportsbook and sportsbook not in latest_by_book:
                    latest_by_book[sportsbook] = int(row['id'])
            if latest_by_book:
                conn.executemany('UPDATE odds_snapshots SET is_closing = 1 WHERE id = ?', [(row_id,) for row_id in latest_by_book.values()])
                marked += len(latest_by_book)
    return marked, []


def payout_units_from_american(price: Optional[int], stake_units: float, result: str) -> Optional[float]:
    american = parse_market_int(price)
    stake = float(finite(stake_units, 1.0))
    if american is None:
        return None
    if result == 'push':
        return 0.0
    if result == 'lose':
        return -stake
    if result != 'win':
        return None
    return stake * (american / 100.0) if american > 0 else stake * (100.0 / abs(american))


def compute_bet_clv(conn: sqlite3.Connection, row: sqlite3.Row) -> Optional[float]:
    report_iso = str(row['report_date'])
    away_team = str(row['away_team'])
    home_team = str(row['home_team'])
    game_pk = int(row['game_pk'] or 0)
    market_type = str(row['market_type'] or '')
    selection = str(row['selection'] or '')
    closing = load_market_consensus_row_from_conn(conn, report_iso, away_team, home_team, game_pk=game_pk, closing_only=True)
    if closing is None:
        closing = load_market_consensus_row_from_conn(conn, report_iso, away_team, home_team, game_pk=game_pk, closing_only=False)
    if closing is None:
        return None
    market_probability = safe_float(row['market_probability'])
    if market_type == 'moneyline':
        selected_team = selection.strip().upper()
        closing_home_prob, closing_away_prob = no_vig_probabilities(closing.get('home_ml'), closing.get('away_ml'))
        closing_prob = closing_home_prob if selected_team == home_team else closing_away_prob
        if closing_prob is None:
            return None
        if market_probability is not None and np.isfinite(market_probability):
            return float(closing_prob - market_probability)
        open_prob = american_implied_probability(parse_market_int(row['book_price']))
        return None if open_prob is None else float(closing_prob - open_prob)
    if market_type == 'run_line':
        match = re.match(r'^([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?)$', selection.strip().upper())
        if not match:
            return None
        team_code = match.group(1)
        bet_line = float(match.group(2))
        closing_home_prob, closing_away_prob = no_vig_probabilities(closing.get('home_run_line_price'), closing.get('away_run_line_price'))
        closing_line = closing.get('home_run_line') if team_code == home_team else closing.get('away_run_line')
        closing_prob = closing_home_prob if team_code == home_team else closing_away_prob
        if closing_line is not None and abs(float(closing_line) - bet_line) > 0.01:
            return float((float(closing_line) - bet_line) * (1.0 if bet_line > 0 else -1.0))
        if closing_prob is None:
            return None
        if market_probability is not None and np.isfinite(market_probability):
            return float(closing_prob - market_probability)
        open_prob = american_implied_probability(parse_market_int(row['book_price']))
        return None if open_prob is None else float(closing_prob - open_prob)
    if market_type == 'total':
        match = re.match(r'^(OVER|UNDER)\s+(\d+(?:\.\d+)?)$', selection.strip().upper())
        if not match:
            return None
        side = match.group(1)
        bet_total = float(match.group(2))
        closing_total = safe_float(closing.get('total_line'))
        if closing_total is None:
            return None
        return float(closing_total - bet_total) if side == 'OVER' else float(bet_total - closing_total)
    return None

def fetch_final_scores_for_date(target_date: dt.date) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, object]] = []
    try:
        response = requests.get(
            clean.STATSAPI_SCHEDULE_URL,
            params={'sportId': 1, 'date': target_date.isoformat(), 'hydrate': 'team,linescore'},
            headers=clean.HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return pd.DataFrame(columns=['away_team', 'home_team', 'away_score', 'home_score']), [f'Final scores unavailable for {target_date.isoformat()}: {exc}']
    for day in payload.get('dates', []):
        for game in day.get('games', []):
            status = str(((game.get('status') or {}).get('detailedState') or '')).lower()
            if 'final' not in status:
                continue
            away = (game.get('teams') or {}).get('away') or {}
            home = (game.get('teams') or {}).get('home') or {}
            away_team = str(((away.get('team') or {}).get('abbreviation')) or ((away.get('team') or {}).get('teamName')) or ((away.get('team') or {}).get('name')) or '').strip().upper()
            home_team = str(((home.get('team') or {}).get('abbreviation')) or ((home.get('team') or {}).get('teamName')) or ((home.get('team') or {}).get('name')) or '').strip().upper()
            away_score = parse_market_int(away.get('score'))
            home_score = parse_market_int(home.get('score'))
            if not away_team or not home_team or away_score is None or home_score is None:
                continue
            rows.append({'away_team': away_team, 'home_team': home_team, 'away_score': away_score, 'home_score': home_score})
    return pd.DataFrame(rows), warnings


def settle_bet_journal(as_of_date: Optional[dt.date] = None) -> Tuple[int, List[str]]:
    ensure_market_db()
    cutoff = (as_of_date or today_local()).isoformat()
    warnings: List[str] = []
    settled = 0
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM bet_journal
            WHERE result_units IS NULL AND report_date < ?
            ORDER BY report_date, id
            """,
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0, warnings
        finals_by_date: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = {}
        for iso in sorted({str(row['report_date']) for row in rows}):
            try:
                target_date = dt.date.fromisoformat(iso)
            except Exception:
                continue
            finals_df, final_warn = fetch_final_scores_for_date(target_date)
            warnings.extend(final_warn)
            keyed: Dict[Tuple[str, str], Dict[str, object]] = {}
            for _, final_row in finals_df.iterrows():
                keyed[(str(final_row['away_team']), str(final_row['home_team']))] = {'away_score': int(final_row['away_score']), 'home_score': int(final_row['home_score'])}
            finals_by_date[iso] = keyed
        for row in rows:
            report_iso = str(row['report_date'])
            away_team = str(row['away_team'])
            home_team = str(row['home_team'])
            final = finals_by_date.get(report_iso, {}).get((away_team, home_team))
            if final is None:
                continue
            away_score = int(final['away_score'])
            home_score = int(final['home_score'])
            market_type = str(row['market_type'] or '')
            selection = str(row['selection'] or '')
            result = None
            if market_type == 'moneyline':
                winner = home_team if home_score > away_score else away_team
                result = 'win' if selection.strip().upper() == winner else 'lose'
            elif market_type == 'run_line':
                match = re.match(r'^([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?)$', selection.strip().upper())
                if match:
                    team_code = match.group(1)
                    line_value = float(match.group(2))
                    selected_score = home_score if team_code == home_team else away_score
                    opp_score = away_score if team_code == home_team else home_score
                    adjusted = selected_score + line_value
                    if abs(adjusted - opp_score) < 1e-9:
                        result = 'push'
                    elif adjusted > opp_score:
                        result = 'win'
                    else:
                        result = 'lose'
            elif market_type == 'total':
                match = re.match(r'^(OVER|UNDER)\s+(\d+(?:\.\d+)?)$', selection.strip().upper())
                if match:
                    side = match.group(1)
                    total_line = float(match.group(2))
                    total_runs = away_score + home_score
                    if abs(total_runs - total_line) < 1e-9:
                        result = 'push'
                    elif side == 'OVER':
                        result = 'win' if total_runs > total_line else 'lose'
                    else:
                        result = 'win' if total_runs < total_line else 'lose'
            if result is None:
                continue
            stake_units = safe_float(row['stake_units'])
            if stake_units is None or not np.isfinite(stake_units):
                stake_units = 1.0
            result_units = payout_units_from_american(parse_market_int(row['book_price']), stake_units, result)
            status = 'graded' if str(row['status'] or 'watch') == 'watch' else 'settled'
            clv_value = compute_bet_clv(conn, row)
            conn.execute(
                """
                UPDATE bet_journal
                SET result_units = ?, clv_value = ?, settled_ts = ?, status = ?
                WHERE id = ?
                """,
                (result_units, clv_value, dt.datetime.now(UTC_TZ).isoformat(), status, int(row['id'])),
            )
            settled += 1
    return settled, warnings


