
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sqlite3
import time
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

OUT_DIR = r"C:\Users\fraun\OneDrive\Documents\2026 MLB Report"
SEASON = 2026
MODEL_STATE_PATH = os.path.join(OUT_DIR, "model_state.json")
BACKTEST_LOG_PATH = os.path.join(OUT_DIR, "backtest_log.csv")
CLEAN_BACKTEST_LOG_PATH = os.path.join(OUT_DIR, "clean_backtest_log.csv")
PARK_FACTORS_CACHE_PATH = os.path.join(OUT_DIR, "park_factors_cache.csv")
VENUE_METADATA_CACHE_PATH = os.path.join(OUT_DIR, "venue_metadata_cache.json")
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
STATSAPI_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
STATSAPI_VENUE_URL = "https://statsapi.mlb.com/api/v1/venues/{venue_id}"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SAVANT_STATCAST_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MLB-Report-Clean-Rebuild/1.0)"}
LOCAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")
ROLL_BLEND = {"season": 0.55, "d30": 0.20, "d14": 0.15, "d7": 0.10}
DEFENSE_PRIOR_REGRESSION_BIP = {"season": 650.0, "d30": 350.0, "d14": 220.0, "d7": 160.0}
TRAVEL_REST_LOOKBACK_DAYS = 5
OFFENSE_PRIOR_REGRESSION_PA = {"season": 450.0, "d30": 300.0, "d14": 200.0, "d7": 140.0}
STARTER_PRIOR_REGRESSION_BF = 60.0
LEAGUE_BASE_RUNS = {"spring": 4.95, "regular": 4.55}
REPORT_NOTES = [
    "This clean runner rebuilds live schedule, Statcast features, park/weather context, and fair lines from scratch.",
    "The current model state and backtest log are reused read-only for weights, calibrations, and validation summaries.",
    "The legacy self-updating backtest pipeline is intentionally not ported into this first clean rebuild.",
]

def today_local() -> dt.date:
    return dt.datetime.now(LOCAL_TZ).date()

def safe_float(value):
    try:
        if value is None:
            return None
        out = float(str(value).strip().replace('%', ''))
        return out if math.isfinite(out) else None
    except Exception:
        return None

def finite(value, fallback=0.0):
    out = safe_float(value)
    return fallback if out is None else float(out)

def display_number(value, fmt='.3f'):
    out = safe_float(value)
    return 'N/A' if out is None else format(out, fmt)

def canonical_player_name(name):
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    if ',' in text:
        last, first = [part.strip() for part in text.split(',', 1)]
        if first and last:
            text = f"{first} {last}"
    return re.sub(r'\s+', ' ', text).strip() or None

def sigmoid(value):
    value = max(min(finite(value, 0.0), 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-value))

def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(finite(value, 0.0) / math.sqrt(2.0)))

def zscore(series):
    numeric = pd.to_numeric(series, errors='coerce')
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not math.isfinite(finite(std, 0.0)) or finite(std, 0.0) == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (numeric - mean) / std

class RequestClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.feed_cache = {}
        self.schedule_cache = {}
        self.weather_cache = {}

    def get_json(self, url, params=None, timeout=30):
        last_exc = None
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"JSON request failed for {url}: {last_exc}")

    def get_text(self, url, params=None, timeout=60):
        last_exc = None
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"Text request failed for {url}: {last_exc}")

def load_model_state():
    if not os.path.exists(MODEL_STATE_PATH):
        return {}
    with open(MODEL_STATE_PATH, 'r', encoding='utf-8') as handle:
        return json.load(handle)

def load_backtest_log():
    if not os.path.exists(BACKTEST_LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(BACKTEST_LOG_PATH)
    except Exception:
        return pd.DataFrame()

def load_clean_backtest_log():
    if not os.path.exists(CLEAN_BACKTEST_LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(CLEAN_BACKTEST_LOG_PATH)
    except Exception:
        return pd.DataFrame()

def load_validation_log():
    clean = load_clean_backtest_log()
    return clean if not clean.empty else load_backtest_log()


def load_park_cache():
    if not os.path.exists(PARK_FACTORS_CACHE_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(PARK_FACTORS_CACHE_PATH)
    except Exception:
        return pd.DataFrame()

def load_venue_metadata_cache():
    if not os.path.exists(VENUE_METADATA_CACHE_PATH):
        return {}
    try:
        with open(VENUE_METADATA_CACHE_PATH, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}

def save_venue_metadata_cache(cache):
    with open(VENUE_METADATA_CACHE_PATH, 'w', encoding='utf-8') as handle:
        json.dump(cache, handle, indent=2)
def phase_weights_from_state(state, phase):
    weights = {
        'w_off': 0.60, 'w_starter': 0.45, 'w_bullpen': 0.20,
        'w_park': 0.03, 'w_weather': 0.02, 'w_context': 0.10,
        'home_field': 0.06, 'spread_scale': 6.0,
    }
    weights.update(((state.get('weights_by_phase') or {}).get(phase)) or (state.get('weights') or {}))
    if phase == 'spring':
        guardrail = state.get('spring_guardrail') or {}
        if guardrail.get('apply') and isinstance(guardrail.get('adjusted_weights'), dict):
            weights.update(guardrail['adjusted_weights'])
    if phase == 'regular':
        earn_back = state.get('regular_earn_back') or {}
        if earn_back.get('apply') and isinstance(earn_back.get('adjusted_weights'), dict):
            weights.update(earn_back['adjusted_weights'])
    return {key: finite(value, weights[key]) for key, value in weights.items()}

def apply_probability_calibration(probability, state, phase):
    profile = ((state.get('probability_calibration') or {}).get(phase) or {})
    p = min(max(finite(probability, 0.5), 1e-6), 1.0 - 1e-6)
    if not profile.get('apply'):
        return p
    logit = math.log(p / (1.0 - p))
    return sigmoid(finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * logit))

def apply_total_calibration(total_runs, state, phase):
    profile = ((state.get('total_calibration') or {}).get(phase) or {})
    total = finite(total_runs, 9.0)
    if not profile.get('apply'):
        return total
    adjusted = finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * total)
    threshold = finite(profile.get('tail_threshold'), 9.0)
    tail_slope = finite(profile.get('tail_slope'), 0.0)
    if tail_slope and total > threshold:
        adjusted += tail_slope * (total - threshold)
    return float(np.clip(adjusted, 5.5, 15.5))

def apply_margin_calibration(margin, state, phase):
    profile = ((state.get('margin_calibration') or {}).get(phase) or {})
    if not profile.get('apply'):
        return finite(margin, 0.0)
    return float(np.clip(finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * finite(margin, 0.0)), -7.0, 7.0))

def schedule_for_date(client, target_date):
    key = target_date.isoformat()
    if key not in client.schedule_cache:
        client.schedule_cache[key] = client.get_json(STATSAPI_SCHEDULE_URL, params={'sportId': 1, 'date': key, 'hydrate': 'probablePitcher,team'})
    return client.schedule_cache[key]

def fetch_schedule_games(client, report_date):
    games, warnings = [], []
    payload = schedule_for_date(client, report_date)
    for day in payload.get('dates', []):
        for game in day.get('games', []):
            try:
                away = (game.get('teams') or {}).get('away') or {}
                home = (game.get('teams') or {}).get('home') or {}
                away_team = away.get('team') or {}
                home_team = home.get('team') or {}
                scheduled_utc = dt.datetime.fromisoformat(str(game['gameDate']).replace('Z', '+00:00')).astimezone(UTC_TZ)
                games.append({
                    'game_pk': int(game['gamePk']),
                    'game_type': str(game.get('gameType', '')),
                    'status': str((game.get('status') or {}).get('detailedState') or ''),
                    'scheduled_utc': scheduled_utc,
                    'start_time_ct': scheduled_utc.astimezone(LOCAL_TZ).strftime('%I:%M %p CT').lstrip('0'),
                    'away_team': str(away_team.get('abbreviation') or away_team.get('teamName') or away_team.get('name') or ''),
                    'home_team': str(home_team.get('abbreviation') or home_team.get('teamName') or home_team.get('name') or ''),
                    'venue_id': (game.get('venue') or {}).get('id'),
                    'venue_name': str((game.get('venue') or {}).get('name') or 'Unknown venue'),
                    'away_pitcher': canonical_player_name((away.get('probablePitcher') or {}).get('fullName')),
                    'home_pitcher': canonical_player_name((home.get('probablePitcher') or {}).get('fullName')),
                    'away_pitcher_id': (away.get('probablePitcher') or {}).get('id'),
                    'home_pitcher_id': (home.get('probablePitcher') or {}).get('id'),
                })
            except Exception as exc:
                warnings.append(f'Schedule parse failed for one game: {exc}')
    games.sort(key=lambda item: item['scheduled_utc'])
    return games, warnings

def season_phase_for_games(games):
    game_types = {game['game_type'] for game in games if game.get('game_type')}
    return 'spring' if game_types and game_types.issubset({'S', 'E'}) else 'regular'

def season_cache_path(kind, season_year):
    return os.path.join(OUT_DIR, f'_{kind}_{season_year}.csv')

def savant_statcast_csv(client, player_type, season_year, start_date, end_date_exclusive, game_types):
    params = {
        'all': 'true', 'hfGT': game_types, 'hfSea': f'{season_year}|', 'player_type': player_type,
        'game_date_gt': (start_date - dt.timedelta(days=1)).isoformat(), 'game_date_lt': end_date_exclusive.isoformat(),
        'group_by': '', 'sort_col': 'game_date', 'sort_order': 'desc', 'min_pas': '0', 'type': 'details',
    }
    text = client.get_text(SAVANT_STATCAST_CSV_URL, params=params, timeout=90).lstrip('\ufeff')
    if not text.strip() or text.lstrip().startswith('<!DOCTYPE'):
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()

def batting_team_from_rows(df):
    if {'home_team', 'away_team', 'inning_topbot'}.issubset(df.columns):
        top = df['inning_topbot'].astype(str).str.lower().str.startswith('top')
        return pd.Series(np.where(top, df['away_team'].astype(str), df['home_team'].astype(str)), index=df.index)
    if 'team' in df.columns:
        return df['team'].astype(str)
    return pd.Series(['UNK'] * len(df), index=df.index)
def fielding_team_from_rows(df):
    if {'home_team', 'away_team', 'inning_topbot'}.issubset(df.columns):
        top = df['inning_topbot'].astype(str).str.lower().str.startswith('top')
        return pd.Series(np.where(top, df['home_team'].astype(str), df['away_team'].astype(str)), index=df.index)
    if 'fld_team' in df.columns:
        return df['fld_team'].astype(str)
    return pd.Series(['UNK'] * len(df), index=df.index)


def normalize_handedness(value):
    text = str(value or '').strip().upper()[:1]
    return text if text in {'L', 'R'} else None

def aggregate_team_offense(df, pitcher_hand=None):
    if df.empty:
        return pd.DataFrame(columns=['team', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'bb_rate', 'k_rate', 'offense_score'])
    work = df.copy()
    work['team_key'] = batting_team_from_rows(work)
    work['p_throws_key'] = work.get('p_throws', pd.Series([None] * len(work), index=work.index)).apply(normalize_handedness)
    filter_hand = normalize_handedness(pitcher_hand)
    if filter_hand:
        work = work[work['p_throws_key'] == filter_hand].copy()
        if work.empty:
            return pd.DataFrame(columns=['team', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'bb_rate', 'k_rate', 'offense_score'])
    work['launch_speed'] = pd.to_numeric(work.get('launch_speed'), errors='coerce')
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['estimated_ba_using_speedangle'] = pd.to_numeric(work.get('estimated_ba_using_speedangle'), errors='coerce')
    work['estimated_slg_using_speedangle'] = pd.to_numeric(work.get('estimated_slg_using_speedangle'), errors='coerce')
    work['woba_value'] = pd.to_numeric(work.get('woba_value'), errors='coerce')
    work['event_text'] = (work.get('events', '').astype(str).str.lower() + ' ' + work.get('description', '').astype(str).str.lower())
    grouped = work.groupby('team_key', dropna=False)
    out = pd.DataFrame({'team': grouped.size().index, 'pa': grouped.size().values})
    out['xwoba'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['xba'] = grouped['estimated_ba_using_speedangle'].mean().values
    out['xslg'] = grouped['estimated_slg_using_speedangle'].mean().values
    out['woba'] = grouped['woba_value'].mean().values
    out['avg_ev'] = grouped['launch_speed'].mean().values
    out['hardhit_rate'] = grouped['launch_speed'].apply(lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') >= 95))).values
    out['bb_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: any(token in str(text) for token in {'walk', 'intent_walk', 'hit_by_pitch'}))))).values
    out['k_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: 'strikeout' in str(text))))).values
    out['xwoba'] = out['xwoba'].fillna(out['woba'])
    for column in ['xwoba', 'xba', 'xslg', 'avg_ev', 'hardhit_rate', 'bb_rate', 'k_rate']:
        out[column] = out[column].fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['offense_score'] = (0.45 * zscore(out['xwoba']) + 0.20 * zscore(out['xslg']) + 0.10 * zscore(out['xba']) + 0.10 * zscore(out['avg_ev']) + 0.10 * zscore(out['hardhit_rate']) + 0.08 * zscore(out['bb_rate']) - 0.03 * zscore(out['k_rate'])).fillna(0.0)
    return out[['team', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'bb_rate', 'k_rate', 'offense_score']]
def previous_regular_season_team_offense(client, season_year, pitcher_hand=None):
    hand_key = normalize_handedness(pitcher_hand)
    cache_kind = 'team_offense_prior' if hand_key is None else f'team_offense_prior_vs_{hand_key.lower()}'
    cache_path = season_cache_path(cache_kind, season_year)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    out = aggregate_team_offense(
        savant_statcast_csv(client, 'batter', season_year, dt.date(season_year, 3, 1), dt.date(season_year, 11, 30), 'R|'),
        pitcher_hand=hand_key,
    )
    out.to_csv(cache_path, index=False)
    return out

def blend_team_offense(season_df, d30_df, d14_df, d7_df, prior_df):
    def indexed(df):
        return df.set_index('team') if df is not None and not df.empty else pd.DataFrame(columns=['team']).set_index(pd.Index([], name='team'))
    season_idx, d30_idx, d14_idx, d7_idx, prior_idx = indexed(season_df), indexed(d30_df), indexed(d14_df), indexed(d7_df), indexed(prior_df)
    teams = sorted(set(season_idx.index) | set(d30_idx.index) | set(d14_idx.index) | set(d7_idx.index) | set(prior_idx.index))
    rows = []
    for team in teams:
        prior_score = finite(prior_idx.get('offense_score', pd.Series(dtype=float)).get(team), 0.0)
        base_row = season_idx.loc[team].to_dict() if team in season_idx.index else {}
        scores, prior_weights = {}, {}
        for label, frame in [('season', season_idx), ('d30', d30_idx), ('d14', d14_idx), ('d7', d7_idx)]:
            if team in frame.index:
                row = frame.loc[team]
                score = finite(row.get('offense_score'), prior_score)
                pa = finite(row.get('pa'), 0.0)
                reg = OFFENSE_PRIOR_REGRESSION_PA[label]
                current_weight = pa / (pa + reg) if (pa + reg) > 0 else 0.0
                scores[label] = (current_weight * score) + ((1.0 - current_weight) * prior_score)
                prior_weights[label] = 1.0 - current_weight
            else:
                scores[label] = prior_score
                prior_weights[label] = 1.0
        rows.append({
            'team': team, 'pa': finite(base_row.get('pa'), 0.0), 'xwoba': safe_float(base_row.get('xwoba')), 'xslg': safe_float(base_row.get('xslg')),
            'xba': safe_float(base_row.get('xba')), 'avg_ev': safe_float(base_row.get('avg_ev')), 'hardhit_rate': safe_float(base_row.get('hardhit_rate')),
            'offense_score': scores['season'],
            'offense_score_blended': (ROLL_BLEND['season'] * scores['season']) + (ROLL_BLEND['d30'] * scores['d30']) + (ROLL_BLEND['d14'] * scores['d14']) + (ROLL_BLEND['d7'] * scores['d7']),
            'prior_weight': (ROLL_BLEND['season'] * prior_weights['season']) + (ROLL_BLEND['d30'] * prior_weights['d30']) + (ROLL_BLEND['d14'] * prior_weights['d14']) + (ROLL_BLEND['d7'] * prior_weights['d7']),
        })
    return pd.DataFrame(rows)

def aggregate_batter_quality(df):
    if df.empty:
        return pd.DataFrame(columns=['batter', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'batter_score'])
    work = df.copy()
    work['batter_name'] = work.get('player_name', '').apply(canonical_player_name)
    work['launch_speed'] = pd.to_numeric(work.get('launch_speed'), errors='coerce')
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['estimated_ba_using_speedangle'] = pd.to_numeric(work.get('estimated_ba_using_speedangle'), errors='coerce')
    work['estimated_slg_using_speedangle'] = pd.to_numeric(work.get('estimated_slg_using_speedangle'), errors='coerce')
    grouped = work.groupby('batter_name', dropna=False)
    out = pd.DataFrame({'batter': grouped.size().index, 'pa': grouped.size().values})
    out['xwoba'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['xba'] = grouped['estimated_ba_using_speedangle'].mean().values
    out['xslg'] = grouped['estimated_slg_using_speedangle'].mean().values
    out['avg_ev'] = grouped['launch_speed'].mean().values
    out['hardhit_rate'] = grouped['launch_speed'].apply(lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') >= 95))).values
    for column in ['xwoba', 'xba', 'xslg', 'avg_ev', 'hardhit_rate']:
        out[column] = out[column].fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['batter_score'] = (0.40 * zscore(out['xwoba']) + 0.20 * zscore(out['xslg']) + 0.10 * zscore(out['xba']) + 0.15 * zscore(out['avg_ev']) + 0.15 * zscore(out['hardhit_rate'])).fillna(0.0)
    return out

def merge_handed_offense_splits(strengths, split_strengths, split_label):
    if strengths is None or strengths.empty:
        return strengths
    renamed = (split_strengths if split_strengths is not None else pd.DataFrame(columns=['team'])).copy()
    suffix = str(split_label)
    keep = ['team', 'pa', 'offense_score', 'offense_score_blended', 'prior_weight']
    if renamed.empty:
        for column in keep[1:]:
            renamed[f'{column}_{suffix}'] = []
        renamed = renamed[['team'] + [f'{column}_{suffix}' for column in keep[1:]]]
    else:
        renamed = renamed[keep].rename(columns={column: f'{column}_{suffix}' for column in keep if column != 'team'})
    return strengths.merge(renamed, on='team', how='left')
def aggregate_team_defense(df):
    if df.empty:
        return pd.DataFrame(columns=['team', 'bip', 'xwoba_contact', 'woba_contact', 'defense_residual', 'defense_score'])
    work = df.copy()
    work['team_key'] = fielding_team_from_rows(work)
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['woba_value'] = pd.to_numeric(work.get('woba_value'), errors='coerce')
    work = work[work['estimated_woba_using_speedangle'].notna() & work['woba_value'].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=['team', 'bip', 'xwoba_contact', 'woba_contact', 'defense_residual', 'defense_score'])
    grouped = work.groupby('team_key', dropna=False)
    out = pd.DataFrame({'team': grouped.size().index, 'bip': grouped.size().values})
    out['xwoba_contact'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['woba_contact'] = grouped['woba_value'].mean().values
    out['defense_residual'] = out['xwoba_contact'] - out['woba_contact']
    out['defense_score'] = zscore(out['defense_residual']).fillna(0.0)
    return out

def previous_regular_season_team_defense(client, season_year):
    cache_path = season_cache_path('team_defense_prior', season_year)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    out = aggregate_team_defense(savant_statcast_csv(client, 'batter', season_year, dt.date(season_year, 3, 1), dt.date(season_year, 11, 30), 'R|'))
    out.to_csv(cache_path, index=False)
    return out

def blend_team_defense(season_df, d30_df, d14_df, d7_df, prior_df):
    def indexed(df):
        return df.set_index('team') if df is not None and not df.empty else pd.DataFrame(columns=['team']).set_index(pd.Index([], name='team'))
    season_idx, d30_idx, d14_idx, d7_idx, prior_idx = indexed(season_df), indexed(d30_df), indexed(d14_df), indexed(d7_df), indexed(prior_df)
    teams = sorted(set(season_idx.index) | set(d30_idx.index) | set(d14_idx.index) | set(d7_idx.index) | set(prior_idx.index))
    rows = []
    for team in teams:
        prior_score = finite(prior_idx.get('defense_score', pd.Series(dtype=float)).get(team), 0.0)
        prior_bip = finite(prior_idx.get('bip', pd.Series(dtype=float)).get(team), 0.0)
        base_row = season_idx.loc[team].to_dict() if team in season_idx.index else {}
        scores, prior_weights = {}, {}
        for label, frame in [('season', season_idx), ('d30', d30_idx), ('d14', d14_idx), ('d7', d7_idx)]:
            if team in frame.index:
                row = frame.loc[team]
                score = finite(row.get('defense_score'), prior_score)
                bip = finite(row.get('bip'), 0.0)
                reg = DEFENSE_PRIOR_REGRESSION_BIP[label]
                current_weight = bip / (bip + reg) if (bip + reg) > 0 else 0.0
                scores[label] = (current_weight * score) + ((1.0 - current_weight) * prior_score)
                prior_weights[label] = 1.0 - current_weight
            else:
                scores[label] = prior_score
                prior_weights[label] = 1.0
        rows.append({
            'team': team,
            'defense_bip': finite(base_row.get('bip'), prior_bip),
            'defense_score': scores['season'],
            'defense_score_blended': (ROLL_BLEND['season'] * scores['season']) + (ROLL_BLEND['d30'] * scores['d30']) + (ROLL_BLEND['d14'] * scores['d14']) + (ROLL_BLEND['d7'] * scores['d7']),
            'defense_prior_weight': (ROLL_BLEND['season'] * prior_weights['season']) + (ROLL_BLEND['d30'] * prior_weights['d30']) + (ROLL_BLEND['d14'] * prior_weights['d14']) + (ROLL_BLEND['d7'] * prior_weights['d7']),
        })
    return pd.DataFrame(rows)


def aggregate_pitcher_quality(df):
    if df.empty:
        return pd.DataFrame(columns=['pitcher', 'pitcher_id', 'p_throws', 'bf', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate', 'pitcher_score'])
    work = df.copy()
    work['pitcher_name'] = work.get('player_name', '').apply(canonical_player_name)
    work['pitcher_id'] = pd.to_numeric(work.get('pitcher'), errors='coerce')
    work['p_throws'] = work.get('p_throws', pd.Series([None] * len(work), index=work.index)).apply(normalize_handedness)
    work['launch_speed'] = pd.to_numeric(work.get('launch_speed'), errors='coerce')
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['estimated_slg_using_speedangle'] = pd.to_numeric(work.get('estimated_slg_using_speedangle'), errors='coerce')
    work['woba_value'] = pd.to_numeric(work.get('woba_value'), errors='coerce')
    work['event_text'] = (work.get('events', '').astype(str).str.lower() + ' ' + work.get('description', '').astype(str).str.lower())
    grouped = work.groupby(['pitcher_name', 'pitcher_id'], dropna=False)
    out = pd.DataFrame({'pitcher': [idx[0] for idx in grouped.size().index], 'pitcher_id': [idx[1] for idx in grouped.size().index], 'bf': grouped.size().values})
    out['p_throws'] = grouped['p_throws'].agg(lambda s: next((hand for hand in s.dropna().tolist() if hand in {'L', 'R'}), None)).values
    out['xwoba_allowed'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['woba_allowed'] = grouped['woba_value'].mean().values
    out['xslg_allowed'] = grouped['estimated_slg_using_speedangle'].mean().values
    out['avg_ev_allowed'] = grouped['launch_speed'].mean().values
    out['hardhit_allowed_rate'] = grouped['launch_speed'].apply(lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') >= 95))).values
    out['k_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: 'strikeout' in str(text))))).values
    out['bb_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: ('walk' in str(text)) or ('hit_by_pitch' in str(text)))))).values
    out['xwoba_allowed'] = out['xwoba_allowed'].fillna(out['woba_allowed'])
    for column in ['xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate']:
        out[column] = out[column].fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['pitcher_score'] = (-0.40 * zscore(out['xwoba_allowed']) - 0.15 * zscore(out['xslg_allowed']) - 0.10 * zscore(out['avg_ev_allowed']) - 0.10 * zscore(out['hardhit_allowed_rate']) + 0.20 * zscore(out['k_rate']) - 0.05 * zscore(out['bb_rate'])).fillna(0.0)
    return out

def previous_regular_season_pitchers(client, season_year):
    cache_path = season_cache_path('pitcher_priors', season_year)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    out = aggregate_pitcher_quality(savant_statcast_csv(client, 'pitcher', season_year, dt.date(season_year, 3, 1), dt.date(season_year, 11, 30), 'R|'))
    out.to_csv(cache_path, index=False)
    return out

def pitcher_lookup_map(df):
    by_name, by_id = {}, {}
    if df is None or df.empty:
        return by_name, by_id
    for _, row in df.iterrows():
        record = row.to_dict()
        name = canonical_player_name(record.get('pitcher'))
        pitcher_id = safe_float(record.get('pitcher_id'))
        if name:
            by_name[name] = record
        if pitcher_id is not None:
            by_id[int(pitcher_id)] = record
    return by_name, by_id

def resolve_pitcher_projection(pitcher_name, pitcher_id, current_df, prior_df):
    current_by_name, current_by_id = pitcher_lookup_map(current_df)
    prior_by_name, prior_by_id = pitcher_lookup_map(prior_df)
    name_key = canonical_player_name(pitcher_name)
    current = current_by_id.get(int(pitcher_id)) if pitcher_id is not None and int(pitcher_id) in current_by_id else current_by_name.get(name_key)
    prior = prior_by_id.get(int(pitcher_id)) if pitcher_id is not None and int(pitcher_id) in prior_by_id else prior_by_name.get(name_key)
    if current is not None:
        current_score = finite(current.get('pitcher_score'), 0.0)
        bf = finite(current.get('bf'), 0.0)
        current_hand = normalize_handedness(current.get('p_throws'))
        prior_hand = normalize_handedness((prior or {}).get('p_throws'))
        if prior is not None:
            current_weight = bf / (bf + STARTER_PRIOR_REGRESSION_BF) if (bf + STARTER_PRIOR_REGRESSION_BF) > 0 else 0.0
            return {'score': (current_weight * current_score) + ((1.0 - current_weight) * finite(prior.get('pitcher_score'), 0.0)), 'source': 'current + prior blend', 'confidence': 0.55 + (0.45 * current_weight), 'hand': current_hand or prior_hand}
        return {'score': current_score, 'source': 'current season', 'confidence': 0.60 if bf < STARTER_PRIOR_REGRESSION_BF else 0.95, 'hand': current_hand}
    if prior is not None:
        return {'score': finite(prior.get('pitcher_score'), 0.0), 'source': '2025 regular-season prior', 'confidence': 0.55, 'hand': normalize_handedness(prior.get('p_throws'))}
    return {'score': 0.0, 'source': 'neutral prior', 'confidence': 0.25, 'hand': None}

def matchup_adjusted_offense(team_row, pitcher_hand):
    base_score = finite((team_row or {}).get('offense_score_blended'), 0.0)
    hand_key = normalize_handedness(pitcher_hand)
    if hand_key is None:
        return {'score': base_score, 'adjustment': 0.0, 'split_score': base_score, 'split_weight': 0.0, 'split_label': 'vs unknown'}
    suffix = 'vs_lhp' if hand_key == 'L' else 'vs_rhp'
    split_score = safe_float((team_row or {}).get(f'offense_score_blended_{suffix}'))
    split_prior = finite((team_row or {}).get(f'prior_weight_{suffix}'), 1.0)
    split_pa = finite((team_row or {}).get(f'pa_{suffix}'), 0.0)
    if split_score is None:
        return {'score': base_score, 'adjustment': 0.0, 'split_score': base_score, 'split_weight': 0.0, 'split_label': f'vs {hand_key}HP'}
    reliability = min(split_pa / 140.0, 1.0)
    split_weight = 0.30 * reliability * (1.0 - (0.65 * split_prior))
    adjusted_score = base_score + (split_weight * (split_score - base_score))
    return {
        'score': adjusted_score,
        'adjustment': adjusted_score - base_score,
        'split_score': split_score,
        'split_weight': split_weight,
        'split_label': f'vs {hand_key}HP',
    }
def completed_games_in_range(client, start_date, end_date):
    games = []
    current = start_date
    while current <= end_date:
        payload = schedule_for_date(client, current)
        for day in payload.get('dates', []):
            for game in day.get('games', []):
                if str(((game.get('status') or {}).get('abstractGameState')) or '') == 'Final':
                    games.append(game)
        current += dt.timedelta(days=1)
    return games

def bullpen_snapshot(client, report_date, lookback_days, current_pitchers, prior_pitchers):
    start_date = report_date - dt.timedelta(days=lookback_days)
    end_date = report_date - dt.timedelta(days=1)
    if end_date < start_date:
        return pd.DataFrame(columns=['team', 'bullpen_score'])
    current_by_name, current_by_id = pitcher_lookup_map(current_pitchers)
    prior_by_name, prior_by_id = pitcher_lookup_map(prior_pitchers)
    rows = []
    for game in completed_games_in_range(client, start_date, end_date):
        game_pk = int(game['gamePk'])
        if game_pk not in client.feed_cache:
            client.feed_cache[game_pk] = client.get_json(STATSAPI_GAME_FEED_URL.format(game_pk=game_pk))
        feed = client.feed_cache[game_pk]
        box = (((feed.get('liveData') or {}).get('boxscore') or {}).get('teams')) or {}
        for side in ['away', 'home']:
            team_block = box.get(side) or {}
            team_info = ((game.get('teams') or {}).get(side) or {}).get('team') or {}
            team_abbrev = str(team_info.get('abbreviation') or team_info.get('teamName') or 'UNK')
            bullpen_ids = team_block.get('bullpen') or []
            if not bullpen_ids:
                pitchers = team_block.get('pitchers') or []
                bullpen_ids = pitchers[1:] if len(pitchers) > 1 else []
            players = team_block.get('players') or {}
            reliever_scores, reliever_weights = [], []
            for pitcher_id in bullpen_ids:
                player = players.get(f'ID{pitcher_id}') or {}
                stats = ((player.get('stats') or {}).get('pitching') or {})
                workload = finite(stats.get('battersFaced'), 0.0)
                if workload <= 0:
                    workload = 3.0 * finite(stats.get('inningsPitched'), 0.0)
                if workload <= 0:
                    workload = 1.0
                name = canonical_player_name(((player.get('person') or {}).get('fullName')))
                record = current_by_id.get(pitcher_id) or current_by_name.get(name) or prior_by_id.get(pitcher_id) or prior_by_name.get(name) or {}
                reliever_scores.append(finite(record.get('pitcher_score'), 0.0))
                reliever_weights.append(workload)
            if reliever_scores:
                rows.append({'team': team_abbrev, 'quality': float(np.average(reliever_scores, weights=reliever_weights)), 'workload': float(np.sum(reliever_weights))})
    if not rows:
        return pd.DataFrame(columns=['team', 'bullpen_score'])
    out = pd.DataFrame(rows).groupby('team', as_index=False).agg(quality=('quality', 'mean'), workload=('workload', 'sum'))
    out['bullpen_score'] = (zscore(out['quality']) - (0.45 * zscore(out['workload']))).fillna(0.0)
    return out[['team', 'bullpen_score']]

def fetch_venue_metadata(client, venue_id, venue_name, cache):
    if venue_id is None:
        return {'venue': venue_name or 'Unknown venue', 'roof_type': 'open_air', 'surface_type': 'grass', 'elevation_ft': np.nan, 'latitude': np.nan, 'longitude': np.nan, 'azimuth_angle': np.nan}
    key = str(int(venue_id))
    if key in cache:
        return dict(cache[key])
    meta = {'venue': venue_name or f'id:{venue_id}', 'roof_type': 'open_air', 'surface_type': 'grass', 'elevation_ft': np.nan, 'latitude': np.nan, 'longitude': np.nan, 'azimuth_angle': np.nan}
    try:
        venue = (client.get_json(STATSAPI_VENUE_URL.format(venue_id=int(venue_id))).get('venues') or [{}])[0]
        location = venue.get('location') or {}
        field_info = venue.get('fieldInfo') or {}
        meta.update({'venue': str(venue.get('name') or meta['venue']), 'roof_type': str(field_info.get('roofType') or 'open_air').strip().lower().replace(' ', '_'), 'surface_type': str(field_info.get('turfType') or 'grass').strip().lower().replace(' ', '_'), 'elevation_ft': safe_float(location.get('elevation')), 'latitude': safe_float((location.get('defaultCoordinates') or {}).get('latitude')), 'longitude': safe_float((location.get('defaultCoordinates') or {}).get('longitude')), 'azimuth_angle': safe_float(field_info.get('leftLine'))})
    except Exception:
        pass
    cache[key] = meta
    return dict(meta)
def haversine_miles(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    lat1_rad, lon1_rad = math.radians(float(lat1)), math.radians(float(lon1))
    lat2_rad, lon2_rad = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat / 2.0) ** 2) + (math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(dlon / 2.0) ** 2))
    return 3958.8 * (2.0 * math.asin(min(1.0, math.sqrt(a))))

def most_recent_team_game(client, team_abbrev, target_date, lookback_days=TRAVEL_REST_LOOKBACK_DAYS):
    candidates = []
    for day_offset in range(1, lookback_days + 1):
        current = target_date - dt.timedelta(days=day_offset)
        payload = schedule_for_date(client, current)
        for day in payload.get('dates', []):
            for game in day.get('games', []):
                away = ((game.get('teams') or {}).get('away') or {}).get('team') or {}
                home = ((game.get('teams') or {}).get('home') or {}).get('team') or {}
                away_abbrev = str(away.get('abbreviation') or away.get('teamName') or away.get('name') or '')
                home_abbrev = str(home.get('abbreviation') or home.get('teamName') or home.get('name') or '')
                if team_abbrev not in {away_abbrev, home_abbrev}:
                    continue
                try:
                    scheduled_utc = dt.datetime.fromisoformat(str(game['gameDate']).replace('Z', '+00:00')).astimezone(UTC_TZ)
                except Exception:
                    scheduled_utc = dt.datetime.combine(current, dt.time(0, 0), tzinfo=UTC_TZ)
                candidates.append({
                    'game_date': current,
                    'scheduled_utc': scheduled_utc,
                    'venue_id': (game.get('venue') or {}).get('id'),
                    'venue_name': str((game.get('venue') or {}).get('name') or 'Unknown venue'),
                    'was_home': team_abbrev == home_abbrev,
                })
    if not candidates:
        return None
    candidates.sort(key=lambda row: row['scheduled_utc'], reverse=True)
    return candidates[0]

def team_travel_rest_context(client, team_abbrev, game, venue_cache, lookback_days=TRAVEL_REST_LOOKBACK_DAYS):
    target_date = game['scheduled_utc'].astimezone(LOCAL_TZ).date()
    previous_game = most_recent_team_game(client, team_abbrev, target_date, lookback_days=lookback_days)
    if previous_game is None:
        return {'off_days': 1, 'travel_miles': 0.0, 'score': 0.02, 'previous_game_found': False}
    current_meta = fetch_venue_metadata(client, game.get('venue_id'), game.get('venue_name'), venue_cache)
    previous_meta = fetch_venue_metadata(client, previous_game.get('venue_id'), previous_game.get('venue_name'), venue_cache)
    off_days = max(0, (target_date - previous_game['game_date']).days - 1)
    travel_miles = haversine_miles(
        safe_float(previous_meta.get('latitude')),
        safe_float(previous_meta.get('longitude')),
        safe_float(current_meta.get('latitude')),
        safe_float(current_meta.get('longitude')),
    )
    score = 0.0
    if off_days >= 1:
        score += min(off_days, 2) * 0.035
    if off_days == 0:
        if travel_miles > 1400:
            score -= 0.16
        elif travel_miles > 850:
            score -= 0.10
        elif travel_miles > 300:
            score -= 0.05
    elif off_days == 1 and travel_miles > 1200:
        score -= 0.03
    if previous_game.get('was_home') and game.get('home_team') == team_abbrev and travel_miles < 30:
        score += 0.015
    return {
        'off_days': int(off_days),
        'travel_miles': float(travel_miles),
        'score': float(np.clip(score, -0.20, 0.10)),
        'previous_game_found': True,
    }


def park_history_adjustment(park_cache, report_date, venue_id, venue_name):
    if park_cache.empty:
        return {'total_adj': 0.0, 'home_edge_adj': 0.0}
    work = park_cache.copy()
    work['date'] = pd.to_datetime(work['date'], errors='coerce').dt.date
    work = work[work['date'].notna() & (work['date'] < report_date)].copy()
    if work.empty:
        return {'total_adj': 0.0, 'home_edge_adj': 0.0}
    venue_rows = work[pd.to_numeric(work.get('venue_id'), errors='coerce') == int(venue_id)] if venue_id is not None else work[work['venue'].astype(str) == str(venue_name)]
    if venue_rows.empty:
        return {'total_adj': 0.0, 'home_edge_adj': 0.0}
    work['age'] = (report_date - work['date']).apply(lambda value: value.days)
    venue_rows = venue_rows.copy()
    venue_rows['age'] = (report_date - venue_rows['date']).apply(lambda value: value.days)
    work['weight'] = np.power(0.5, work['age'] / 120.0)
    venue_rows['weight'] = np.power(0.5, venue_rows['age'] / 120.0)
    league_total = float(np.average(pd.to_numeric(work['total_runs'], errors='coerce').fillna(0.0), weights=work['weight']))
    league_home = float(np.average(pd.to_numeric(work['home_win'], errors='coerce').fillna(0.5), weights=work['weight']))
    venue_total = float(np.average(pd.to_numeric(venue_rows['total_runs'], errors='coerce').fillna(0.0), weights=venue_rows['weight']))
    venue_home = float(np.average(pd.to_numeric(venue_rows['home_win'], errors='coerce').fillna(0.5), weights=venue_rows['weight']))
    return {'total_adj': float(np.clip(venue_total - league_total, -1.25, 1.25)), 'home_edge_adj': float(np.clip(venue_home - league_home, -0.10, 0.10))}

def park_static_adjustment(meta):
    total_adj = 0.0
    elevation = finite(meta.get('elevation_ft'), 0.0)
    if elevation > 0:
        total_adj += float(np.clip((elevation - 500.0) / 2500.0, -0.10, 0.35))
    if 'turf' in str(meta.get('surface_type') or ''):
        total_adj += 0.05
    if 'dome' in str(meta.get('roof_type') or '') or 'closed' in str(meta.get('roof_type') or ''):
        total_adj += 0.02
    return {'total_adj': total_adj, 'home_edge_adj': 0.0}

def estimate_wind_vector(meta, wind_deg):
    azimuth = safe_float(meta.get('azimuth_angle'))
    if azimuth is None or wind_deg is None:
        return 0.0
    return float(np.clip(math.cos(math.radians(float(wind_deg) - float(azimuth))), -1.0, 1.0))

def fetch_weather_context(client, game, meta):
    lat, lon = safe_float(meta.get('latitude')), safe_float(meta.get('longitude'))
    if lat is None or lon is None:
        return {'source': 'neutral', 'temp_f': np.nan, 'wind_mph': 0.0, 'wind_out_in': 0.0, 'precip_prob': np.nan}
    target_hour = game['scheduled_utc'].replace(minute=0, second=0, microsecond=0)
    cache_key = (round(lat, 4), round(lon, 4), target_hour.isoformat())
    if cache_key in client.weather_cache:
        return dict(client.weather_cache[cache_key])
    try:
        payload = client.get_json(OPEN_METEO_FORECAST_URL, params={'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m,precipitation_probability,wind_speed_10m,wind_direction_10m', 'timezone': 'UTC', 'start_date': game['scheduled_utc'].date().isoformat(), 'end_date': game['scheduled_utc'].date().isoformat()})
        hourly = payload.get('hourly') or {}
        times = hourly.get('time') or []
        target_key = target_hour.strftime('%Y-%m-%dT%H:%M')
        if target_key in times:
            idx = times.index(target_key)
            weather = {'source': 'open_meteo', 'temp_f': (finite((hourly.get('temperature_2m') or [0.0])[idx], 0.0) * 9.0 / 5.0) + 32.0, 'wind_mph': finite((hourly.get('wind_speed_10m') or [0.0])[idx], 0.0) / 1.60934, 'wind_out_in': estimate_wind_vector(meta, safe_float((hourly.get('wind_direction_10m') or [None])[idx])), 'precip_prob': finite((hourly.get('precipitation_probability') or [0.0])[idx], 0.0)}
            client.weather_cache[cache_key] = weather
            return dict(weather)
    except Exception:
        pass
    return {'source': 'neutral', 'temp_f': np.nan, 'wind_mph': 0.0, 'wind_out_in': 0.0, 'precip_prob': np.nan}

def weather_adjustments(weather):
    temp_f, wind_mph, wind_out_in, precip_prob = finite(weather.get('temp_f'), 72.0), finite(weather.get('wind_mph'), 0.0), finite(weather.get('wind_out_in'), 0.0), finite(weather.get('precip_prob'), 0.0)
    total_adj = (0.02 * ((temp_f - 72.0) / 5.0)) + (0.035 * wind_mph * wind_out_in) - (0.10 * (precip_prob / 100.0))
    return {'total_adj': float(np.clip(total_adj, -0.75, 0.75)), 'home_edge_adj': float(np.clip(0.01 * wind_out_in, -0.03, 0.03))}

def build_validation_summary(backtest_df):
    summary = {'games': 0, 'oos_games': 0, 'log_loss': None, 'accuracy': None, 'total_mae': None, 'total_rmse': None, 'margin_mae': None, 'margin_rmse': None, 'date_span': 'N/A', 'oos_span': 'N/A'}
    if backtest_df is None or backtest_df.empty:
        return summary
    work = backtest_df.copy()
    work['date'] = pd.to_datetime(work.get('date'), errors='coerce').dt.date
    work = work[work['date'].notna()].copy()
    if work.empty:
        return summary
    unique_dates = sorted(work['date'].unique())
    holdout = max(1, int(round(len(unique_dates) * 0.4)))
    oos = work[work['date'].isin(set(unique_dates[-holdout:]))].copy()
    if oos.empty:
        return summary
    y = pd.to_numeric(oos.get('y_home'), errors='coerce')
    p = pd.to_numeric(oos.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    valid = (~y.isna()) & (~p.isna())
    if valid.any():
        yv, pv = y[valid].astype(int), p[valid].astype(float)
        summary['log_loss'] = float((-(yv * np.log(pv)) - ((1 - yv) * np.log(1.0 - pv))).mean())
        summary['accuracy'] = float(np.mean((pv >= 0.5).astype(int) == yv))
    actual_total = pd.to_numeric(oos.get('home_score'), errors='coerce') + pd.to_numeric(oos.get('away_score'), errors='coerce')
    predicted_total = pd.to_numeric(oos.get('projected_total'), errors='coerce')
    total_valid = (~actual_total.isna()) & (~predicted_total.isna())
    if total_valid.any():
        total_error = actual_total[total_valid] - predicted_total[total_valid]
        summary['total_mae'] = float(np.mean(np.abs(total_error)))
        summary['total_rmse'] = float(np.sqrt(np.mean(np.square(total_error))))
    actual_margin = pd.to_numeric(oos.get('home_score'), errors='coerce') - pd.to_numeric(oos.get('away_score'), errors='coerce')
    predicted_margin = pd.to_numeric(oos.get('projected_margin'), errors='coerce')
    margin_valid = (~actual_margin.isna()) & (~predicted_margin.isna())
    if margin_valid.any():
        margin_error = actual_margin[margin_valid] - predicted_margin[margin_valid]
        summary['margin_mae'] = float(np.mean(np.abs(margin_error)))
        summary['margin_rmse'] = float(np.sqrt(np.mean(np.square(margin_error))))
    summary['games'], summary['oos_games'] = int(len(work)), int(len(oos))
    summary['date_span'], summary['oos_span'] = f"{unique_dates[0].isoformat()} to {unique_dates[-1].isoformat()}", f"{sorted(oos['date'].unique())[0].isoformat()} to {sorted(oos['date'].unique())[-1].isoformat()}"
    return summary
def estimate_margin_sigma(backtest_df):
    if backtest_df is None or backtest_df.empty:
        return 3.25
    work = backtest_df.copy()
    for column in ['projected_margin', 'home_score', 'away_score']:
        if column not in work.columns:
            return 3.25
        work[column] = pd.to_numeric(work[column], errors='coerce')
    work['actual_margin'] = work['home_score'] - work['away_score']
    work = work.dropna(subset=['projected_margin', 'actual_margin']).copy()
    if work.empty:
        return 3.25
    sigma = float(np.sqrt(np.mean(np.square(work['actual_margin'] - work['projected_margin']))))
    return float(np.clip(sigma, 1.75, 6.0)) if math.isfinite(sigma) else 3.25

def fair_run_line_prices(margin, sigma):
    sigma_value = float(np.clip(finite(sigma, 3.25), 1.75, 6.0))
    home_cover = 1.0 - normal_cdf((1.5 - finite(margin, 0.0)) / sigma_value)
    home_cover = float(np.clip(home_cover, 1e-6, 1.0 - 1e-6))
    away_cover = 1.0 - home_cover
    return american_from_probability(home_cover), american_from_probability(away_cover)

def american_from_probability(probability):
    p = min(max(finite(probability, 0.5), 1e-6), 1.0 - 1e-6)
    return int(round((-100.0 * p) / (1.0 - p))) if p >= 0.5 else int(round((100.0 * (1.0 - p)) / p))

def round_to_half(value):
    return round(finite(value, 0.0) * 2.0) / 2.0

def confidence_tier(probability, phase):
    edge = abs(finite(probability, 0.5) - 0.5)
    if phase == 'spring':
        return 'High' if edge >= 0.16 else 'Medium' if edge >= 0.10 else 'Low'
    return 'High' if edge >= 0.12 else 'Medium' if edge >= 0.07 else 'Low'

def lineup_confirmation_counts(client, game_pk):
    try:
        if game_pk not in client.feed_cache:
            client.feed_cache[game_pk] = client.get_json(STATSAPI_GAME_FEED_URL.format(game_pk=game_pk))
        box = (((client.feed_cache[game_pk].get('liveData') or {}).get('boxscore') or {}).get('teams')) or {}
        return {side: int(min(len((box.get(side) or {}).get('battingOrder') or []), 9)) for side in ['away', 'home']}
    except Exception:
        return {'away': 0, 'home': 0}

def predict_game(game, phase, weights, state, strengths, current_pitchers, prior_pitchers, bullpen_map, park_cache, client, venue_cache, margin_sigma, lineup_counts_override=None):
    strength_index = strengths.set_index('team') if strengths is not None and not strengths.empty else pd.DataFrame().set_index(pd.Index([], name='team'))
    away_row = strength_index.loc[game['away_team']].to_dict() if game['away_team'] in strength_index.index else {}
    home_row = strength_index.loc[game['home_team']].to_dict() if game['home_team'] in strength_index.index else {}
    away_off_base = finite(away_row.get('offense_score_blended'), 0.0)
    home_off_base = finite(home_row.get('offense_score_blended'), 0.0)
    away_def = finite(away_row.get('defense_score_blended'), 0.0)
    home_def = finite(home_row.get('defense_score_blended'), 0.0)
    away_starter = resolve_pitcher_projection(game.get('away_pitcher'), game.get('away_pitcher_id'), current_pitchers, prior_pitchers)
    home_starter = resolve_pitcher_projection(game.get('home_pitcher'), game.get('home_pitcher_id'), current_pitchers, prior_pitchers)
    away_matchup = matchup_adjusted_offense(away_row, home_starter.get('hand'))
    home_matchup = matchup_adjusted_offense(home_row, away_starter.get('hand'))
    away_off = finite(away_matchup.get('score'), away_off_base)
    home_off = finite(home_matchup.get('score'), home_off_base)
    away_bullpen = finite(bullpen_map.get(game['away_team']), 0.0)
    home_bullpen = finite(bullpen_map.get(game['home_team']), 0.0)
    venue_meta = fetch_venue_metadata(client, game.get('venue_id'), game.get('venue_name'), venue_cache)
    park_history = park_history_adjustment(park_cache, game['scheduled_utc'].astimezone(LOCAL_TZ).date(), game.get('venue_id'), game.get('venue_name'))
    park_static = park_static_adjustment(venue_meta)
    weather = fetch_weather_context(client, game, venue_meta)
    weather_adj = weather_adjustments(weather)
    lineups = dict(lineup_counts_override) if lineup_counts_override is not None else lineup_confirmation_counts(client, game['game_pk'])
    lineup_component = 0.04 * ((lineups.get('home', 0) - lineups.get('away', 0)) / 9.0)
    away_schedule = team_travel_rest_context(client, game['away_team'], game, venue_cache)
    home_schedule = team_travel_rest_context(client, game['home_team'], game, venue_cache)
    context_feature_mult = 0.0 if phase == 'spring' else 1.0
    schedule_component = context_feature_mult * (finite(home_schedule.get('score'), 0.0) - finite(away_schedule.get('score'), 0.0))
    defense_component = context_feature_mult * (0.09 * (home_def - away_def))
    x_context = lineup_component + schedule_component + defense_component
    x_off = home_off - away_off
    x_starter = finite(home_starter['score'], 0.0) - finite(away_starter['score'], 0.0)
    x_bullpen = home_bullpen - away_bullpen
    x_park = finite(park_history['home_edge_adj'], 0.0)
    x_weather = finite(weather_adj['home_edge_adj'], 0.0)
    logit = (weights['w_off'] * x_off) + (weights['w_starter'] * x_starter) + (weights['w_bullpen'] * x_bullpen) + (weights['w_park'] * x_park) + (weights['w_weather'] * x_weather) + (weights['w_context'] * x_context) + weights['home_field']
    p_home_raw = sigmoid(logit)
    p_home = apply_probability_calibration(p_home_raw, state, phase)
    starter_uncertainty = (1.0 - finite(away_starter['confidence'], 0.5)) + (1.0 - finite(home_starter['confidence'], 0.5))
    shared_env = max(0.0, (0.10 * (abs(away_off) + abs(home_off))) + (0.08 * max(0.0, -(finite(away_starter['score'], 0.0) + finite(home_starter['score'], 0.0)))) + (0.05 * max(0.0, -(away_bullpen + home_bullpen))) + (0.15 * starter_uncertainty))
    shared_total_adj = finite(park_history['total_adj'], 0.0) + finite(park_static['total_adj'], 0.0) + finite(weather_adj['total_adj'], 0.0) + shared_env
    base = LEAGUE_BASE_RUNS.get(phase, 4.6)
    away_runs_raw = base + (0.70 * away_off) - (0.50 * finite(home_starter['score'], 0.0)) - (0.22 * home_bullpen) + (0.50 * shared_total_adj) + (0.06 * context_feature_mult * finite(away_schedule.get('score'), 0.0)) - (0.10 * context_feature_mult * home_def) - (0.08 * lineup_component)
    home_runs_raw = base + (0.70 * home_off) - (0.50 * finite(away_starter['score'], 0.0)) - (0.22 * away_bullpen) + (0.50 * shared_total_adj) + (0.06 * context_feature_mult * finite(home_schedule.get('score'), 0.0)) - (0.10 * context_feature_mult * away_def) + (0.08 * lineup_component)
    away_runs_raw = float(np.clip(away_runs_raw, 2.0, 8.5))
    home_runs_raw = float(np.clip(home_runs_raw, 2.0, 8.5))
    total_raw = away_runs_raw + home_runs_raw
    total_calibrated = apply_total_calibration(total_raw, state, phase)
    scale = total_calibrated / total_raw if total_raw > 0 else 1.0
    away_runs = away_runs_raw * scale
    home_runs = home_runs_raw * scale
    margin_raw = home_runs - away_runs
    margin_calibrated = apply_margin_calibration(margin_raw, state, phase)
    winner = game['home_team'] if p_home >= 0.5 else game['away_team']
    winner_prob = p_home if winner == game['home_team'] else 1.0 - p_home
    return {
        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home,
        'fair_home_ml': american_from_probability(p_home), 'fair_away_ml': american_from_probability(1.0 - p_home),
        'away_runs': away_runs, 'home_runs': home_runs, 'total_calibrated': total_calibrated, 'total_bet_line': round_to_half(total_calibrated),
        'margin_calibrated': margin_calibrated, 'run_line_prices': fair_run_line_prices(margin_calibrated, margin_sigma), 'confidence_tier': confidence_tier(winner_prob, phase),
        'features': {
            'away_row': away_row, 'home_row': home_row, 'away_starter': away_starter, 'home_starter': home_starter,
            'away_bullpen': away_bullpen, 'home_bullpen': home_bullpen, 'park_history': park_history, 'park_static': park_static,
            'weather': weather, 'weather_adj': weather_adj, 'lineups': lineups, 'shared_env': shared_env,
            'away_off_base': away_off_base, 'home_off_base': home_off_base, 'away_off_matchup': away_matchup, 'home_off_matchup': home_matchup,
            'away_defense': away_def, 'home_defense': home_def, 'away_schedule': away_schedule, 'home_schedule': home_schedule,
            'context_components': {'lineup': lineup_component, 'schedule': schedule_component, 'defense': defense_component},
            'x_off': x_off, 'x_starter': x_starter, 'x_bullpen': x_bullpen, 'x_park': x_park, 'x_weather': x_weather, 'x_context': x_context, 'venue_meta': venue_meta,
        },
    }

def manual_market_warning(report_date):
    import market_helpers as market
    path = market.manual_market_template_path(report_date)
    return f"Manual market file available: {os.path.basename(path)}" if os.path.exists(path) else None


def selection_cover_probability(selection_team, line_value, home_team, margin, sigma):
    sigma_value = float(np.clip(finite(sigma, 3.25), 1.75, 6.0))
    line = finite(line_value, 0.0)
    if str(selection_team).upper() == str(home_team).upper():
        threshold = -line
        prob = 1.0 - normal_cdf((threshold - finite(margin, 0.0)) / sigma_value)
    else:
        prob = normal_cdf((line - finite(margin, 0.0)) / sigma_value)
    return float(np.clip(prob, 1e-6, 1.0 - 1e-6))


def actionability_assessment(market_type, edge_value, sportsbook_count, prediction, phase):
    import market_helpers as market

    features = prediction.get('features', {}) or {}
    thresholds = market.market_thresholds(market_type, sportsbook_count=sportsbook_count, model_phase=phase)
    magnitude = abs(finite(edge_value, 0.0))
    high = max(finite(thresholds.get('high'), 1.0), 1e-6)
    score = 22.0 + (58.0 * min(magnitude / high, 1.25))
    notes = []

    lineup_counts = features.get('lineups', {}) or {}
    lineup_min = min(int(lineup_counts.get('away', 0) or 0), int(lineup_counts.get('home', 0) or 0))
    if lineup_min >= 9:
        score += 8.0
        notes.append('confirmed lineups')
    elif lineup_min >= 7:
        score += 3.0
        notes.append('mostly confirmed lineups')
    else:
        score -= 8.0
        notes.append('lineups still thin')

    away_starter = features.get('away_starter', {}) or {}
    home_starter = features.get('home_starter', {}) or {}
    starter_conf = (finite(away_starter.get('confidence'), 0.5) + finite(home_starter.get('confidence'), 0.5)) / 2.0
    if starter_conf >= 0.75:
        score += 8.0
        notes.append('starter signal stable')
    elif starter_conf < 0.45:
        score -= 8.0
        notes.append('starter priors heavy')

    starter_sources = f"{away_starter.get('source', '')} | {home_starter.get('source', '')}".lower()
    if 'neutral prior' in starter_sources:
        score -= 6.0
        notes.append('neutral starter prior in play')
    elif 'prior' in starter_sources and 'current season' not in starter_sources:
        score -= 3.0

    weather_source = str((features.get('weather', {}) or {}).get('source', 'neutral') or 'neutral').lower()
    if weather_source != 'neutral':
        score += 4.0
    else:
        score -= 2.0
        notes.append('weather neutral')

    if int(sportsbook_count or 0) >= 2:
        score += 4.0
        notes.append('multi-book consensus')
    else:
        notes.append('single-book consensus')

    if str(phase).lower() == 'spring':
        score -= 6.0
        notes.append('spring caution')

    score = float(np.clip(score, 0.0, 100.0))
    label = market.market_signal_label(market_type, edge_value, sportsbook_count, model_phase=phase)
    if label != 'PASS':
        if score < 42.0:
            label = 'PASS'
        elif label == 'MUST TAKE' and score < 72.0:
            label = 'BET'
        elif label in ['MUST TAKE', 'BET'] and score < 56.0:
            label = 'WATCH'
    note_text = '; '.join(dict.fromkeys([note for note in notes if note][:3])) or 'edge-driven signal'
    return {'label': label, 'score': score, 'notes': note_text}


def actionability_summary_text(action):
    if not action:
        return 'PASS | no actionability signal'
    return f"{str(action.get('label', 'PASS'))} | score {float(action.get('score', 0.0)):.0f}/100 | {str(action.get('notes', '')).strip()}"


def market_comparison_for_prediction(prediction, market_map, phase, margin_sigma):
    import market_helpers as market

    game = prediction['game']
    key = market.market_game_key(game.get('game_pk'), game.get('away_team'), game.get('home_team'))
    market_row = (market_map or {}).get(key)
    if not market_row:
        return {'available': False}

    out = {
        'available': True,
        'market_label': str(market_row.get('market_label') or 'Market'),
        'sportsbook_count': int(finite(market_row.get('sportsbook_count'), 0.0)),
        'moneyline': {'available': False},
        'run_line': {'available': False},
        'total': {'available': False},
    }

    home_ml = market.parse_market_int(market_row.get('home_ml'))
    away_ml = market.parse_market_int(market_row.get('away_ml'))
    market_home_prob, market_away_prob = market.no_vig_probabilities(home_ml, away_ml)
    if home_ml is not None and away_ml is not None and market_home_prob is not None and market_away_prob is not None:
        home_edge = float(prediction['p_home'] - market_home_prob)
        away_edge = float((1.0 - prediction['p_home']) - market_away_prob)
        if home_edge >= away_edge:
            best_team = game['home_team']
            best_edge = home_edge
            best_prob = float(prediction['p_home'])
            best_market_prob = float(market_home_prob)
            best_price = home_ml
            fair_price = int(prediction['fair_home_ml'])
        else:
            best_team = game['away_team']
            best_edge = away_edge
            best_prob = float(1.0 - prediction['p_home'])
            best_market_prob = float(market_away_prob)
            best_price = away_ml
            fair_price = int(prediction['fair_away_ml'])
        action = actionability_assessment('moneyline', best_edge, out['sportsbook_count'], prediction, phase)
        out['moneyline'] = {
            'available': True,
            'home_ml': home_ml,
            'away_ml': away_ml,
            'home_market_prob': float(market_home_prob),
            'away_market_prob': float(market_away_prob),
            'best_side': best_team,
            'best_edge': float(best_edge),
            'best_price': best_price,
            'model_probability': best_prob,
            'market_probability': best_market_prob,
            'fair_price': fair_price,
            'action': action,
        }

    home_line = market.parse_market_float(market_row.get('home_run_line'))
    away_line = market.parse_market_float(market_row.get('away_run_line'))
    home_price = market.parse_market_int(market_row.get('home_run_line_price'))
    away_price = market.parse_market_int(market_row.get('away_run_line_price'))
    market_home_cover_prob, market_away_cover_prob = market.no_vig_probabilities(home_price, away_price)
    if home_line is not None and away_line is not None and home_price is not None and away_price is not None and market_home_cover_prob is not None and market_away_cover_prob is not None:
        home_cover_prob = selection_cover_probability(game['home_team'], home_line, game['home_team'], prediction['margin_calibrated'], margin_sigma)
        away_cover_prob = selection_cover_probability(game['away_team'], away_line, game['home_team'], prediction['margin_calibrated'], margin_sigma)
        home_edge = float(home_cover_prob - market_home_cover_prob)
        away_edge = float(away_cover_prob - market_away_cover_prob)
        if home_edge >= away_edge:
            best_team = game['home_team']
            best_line = float(home_line)
            best_price = home_price
            best_edge = home_edge
            model_probability = home_cover_prob
            market_probability = float(market_home_cover_prob)
        else:
            best_team = game['away_team']
            best_line = float(away_line)
            best_price = away_price
            best_edge = away_edge
            model_probability = away_cover_prob
            market_probability = float(market_away_cover_prob)
        action = actionability_assessment('run_line', best_edge, out['sportsbook_count'], prediction, phase)
        out['run_line'] = {
            'available': True,
            'home_line': float(home_line),
            'away_line': float(away_line),
            'home_price': home_price,
            'away_price': away_price,
            'best_selection': f"{best_team} {best_line:+.1f}",
            'best_team': best_team,
            'best_line': best_line,
            'best_price': best_price,
            'best_edge': float(best_edge),
            'model_probability': float(model_probability),
            'market_probability': market_probability,
            'fair_price': american_from_probability(model_probability),
            'action': action,
        }

    market_total = market.parse_market_float(market_row.get('total_line'))
    over_price = market.parse_market_int(market_row.get('over_price'))
    under_price = market.parse_market_int(market_row.get('under_price'))
    if market_total is not None:
        diff = float(prediction['total_calibrated'] - market_total)
        lean = 'OVER' if diff > 0.05 else ('UNDER' if diff < -0.05 else 'FLAT')
        best_price = over_price if lean == 'OVER' else (under_price if lean == 'UNDER' else None)
        action = actionability_assessment('total', diff, out['sportsbook_count'], prediction, phase)
        if lean == 'FLAT':
            action = {'label': 'PASS', 'score': action['score'], 'notes': 'model total is near market'}
        out['total'] = {
            'available': True,
            'market_total': float(market_total),
            'over_price': over_price,
            'under_price': under_price,
            'diff': diff,
            'lean': lean,
            'best_price': best_price,
            'action': action,
        }
    return out


def seed_open_bet_journal(report_date, predictions, market_map, phase, margin_sigma):
    import market_helpers as market

    market.ensure_market_db()
    records = []
    created_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    for prediction in predictions:
        game = prediction['game']
        market_comp = prediction.get('market_comp') or market_comparison_for_prediction(prediction, market_map, phase, margin_sigma)
        prediction['market_comp'] = market_comp
        sportsbook_label = str(market_comp.get('market_label') or 'Consensus')

        ml = market_comp.get('moneyline', {}) or {}
        ml_action = ml.get('action', {}) or {}
        if ml.get('available') and ml_action.get('label') in ['WATCH', 'BET', 'MUST TAKE'] and float(ml.get('best_edge', 0.0)) > 0:
            records.append((
                created_ts, report_date.isoformat(), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
                'moneyline', str(ml.get('best_side')), None, int(ml.get('best_price')), int(ml.get('fair_price')),
                None, float(ml.get('model_probability')), float(ml.get('market_probability')), float(ml.get('best_edge')),
                'probability', str(prediction.get('confidence_tier', 'Low')), str(ml_action.get('label')), float(ml_action.get('score')),
                str(ml_action.get('notes')), 'watch', None, 'clean_runner_market_seed', None, None, None,
            ))

        rl = market_comp.get('run_line', {}) or {}
        rl_action = rl.get('action', {}) or {}
        if rl.get('available') and rl_action.get('label') in ['WATCH', 'BET', 'MUST TAKE'] and float(rl.get('best_edge', 0.0)) > 0:
            records.append((
                created_ts, report_date.isoformat(), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
                'run_line', str(rl.get('best_selection')), float(rl.get('best_line')), int(rl.get('best_price')), int(rl.get('fair_price')),
                float(prediction.get('margin_calibrated', 0.0)), float(rl.get('model_probability')), float(rl.get('market_probability')), float(rl.get('best_edge')),
                'probability', str(prediction.get('confidence_tier', 'Low')), str(rl_action.get('label')), float(rl_action.get('score')),
                str(rl_action.get('notes')), 'watch', None, 'clean_runner_market_seed', None, None, None,
            ))

        total_comp = market_comp.get('total', {}) or {}
        total_action = total_comp.get('action', {}) or {}
        if total_comp.get('available') and total_action.get('label') in ['WATCH', 'BET', 'MUST TAKE'] and str(total_comp.get('lean')) in ['OVER', 'UNDER']:
            market_total = float(total_comp.get('market_total', 0.0))
            records.append((
                created_ts, report_date.isoformat(), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
                'total', f"{str(total_comp.get('lean'))} {market_total:.1f}", market_total, market.parse_market_int(total_comp.get('best_price')), None,
                float(prediction.get('total_calibrated', 0.0)), None, None, float(total_comp.get('diff', 0.0)),
                'runs', str(prediction.get('confidence_tier', 'Low')), str(total_action.get('label')), float(total_action.get('score')),
                str(total_action.get('notes')), 'watch', None, 'clean_runner_market_seed', None, None, None,
            ))

    with sqlite3.connect(market.SPORTSBOOK_DB_PATH) as conn:
        conn.execute('DELETE FROM bet_journal WHERE report_date = ? AND result_units IS NULL', (report_date.isoformat(),))
        if records:
            conn.executemany(
                """
                INSERT INTO bet_journal (
                    created_ts, report_date, game_pk, away_team, home_team, sportsbook,
                    market_type, selection, line_value, book_price, model_price, model_line_value,
                    model_probability, market_probability, edge_value, edge_kind, confidence_tier,
                    actionability_label, actionability_score, actionability_notes, status, stake_units,
                    notes, result_units, clv_value, settled_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
    return len(records), []


def write_report(report_date, phase, weights, state, predictions, validation, warnings):
    path = os.path.join(OUT_DIR, f"MLB_Report_{report_date.isoformat()}.txt")
    lines = [f"MLB PREDICTIVE REPORT - {report_date.isoformat()}", 'Engine: clean parallel rebuild', 'Source policy: MLB StatsAPI schedule/feed, Baseball Savant Statcast CSV, park cache, and Open-Meteo forecast fallback.', '']
    if warnings:
        lines.append('WARNINGS')
        lines.extend([f" - {warning}" for warning in warnings])
        lines.append('')
    if not predictions:
        lines.append('No games were available for this report date.')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
        return path
    strongest, highest_total, lowest_total = max(predictions, key=lambda item: abs(item['winner_prob'] - 0.5)), max(predictions, key=lambda item: item['total_calibrated']), min(predictions, key=lambda item: item['total_calibrated'])
    market_games = sum(1 for item in predictions if (item.get('market_comp') or {}).get('available'))
    lines.extend(['SUMMARY', f"Strongest edge: {strongest['game']['away_team']} @ {strongest['game']['home_team']} - {strongest['winner']} ({strongest['winner_prob'] * 100:.1f}%)", f"Highest projected total: {highest_total['game']['away_team']} @ {highest_total['game']['home_team']} - {highest_total['total_calibrated']:.1f}", f"Lowest projected total: {lowest_total['game']['away_team']} @ {lowest_total['game']['home_team']} - {lowest_total['total_calibrated']:.1f}", f"Games with sportsbook consensus: {market_games}/{len(predictions)}", '', f"Model regime: {phase.title()}", f"Active weights: off={weights['w_off']:.3f} starter={weights['w_starter']:.3f} bullpen={weights['w_bullpen']:.3f} park={weights['w_park']:.3f} weather={weights['w_weather']:.3f} context={weights['w_context']:.3f} hfa={weights['home_field']:.3f}"])
    probability_profile, total_profile = ((state.get('probability_calibration') or {}).get(phase) or {}), ((state.get('total_calibration') or {}).get(phase) or {})
    if probability_profile.get('apply'):
        lines.append(f"Probability calibration: {probability_profile.get('mode', 'unknown')} | slope {finite(probability_profile.get('slope'), 1.0):.3f} | intercept {finite(probability_profile.get('intercept'), 0.0):+.3f}")
    if total_profile.get('apply'):
        lines.append(f"Total calibration: {total_profile.get('mode', 'unknown')} | slope {finite(total_profile.get('slope'), 1.0):.3f} | intercept {finite(total_profile.get('intercept'), 0.0):+.3f}")
    lines.extend(['', 'VALIDATION SNAPSHOT', f"Backtest sample: {validation['games']} games ({validation['date_span']}) | OOS slice: {validation['oos_games']} games ({validation['oos_span']})"])
    if validation['log_loss'] is not None:
        lines.append(f"OOS moneyline: log loss {validation['log_loss']:.4f} | accuracy {validation['accuracy'] * 100:.1f}%")
    if validation['total_mae'] is not None:
        lines.append(f"OOS totals: MAE {validation['total_mae']:.3f} | RMSE {validation['total_rmse']:.3f}")
    if validation['margin_mae'] is not None:
        lines.append(f"OOS margin: MAE {validation['margin_mae']:.3f} | RMSE {validation['margin_rmse']:.3f}")
    lines.append('')
    for prediction in predictions:
        game, feature = prediction['game'], prediction['features']
        away_row, home_row = feature['away_row'], feature['home_row']
        lines.extend(['=' * 78, f"{game['away_team']} @ {game['home_team']} - {game['venue_name']} - {game['start_time_ct']}", f"Probables: {game.get('away_pitcher') or 'TBD'} vs {game.get('home_pitcher') or 'TBD'}", '', f"Predicted winner: {prediction['winner']}", f"Win probability: {prediction['winner_prob'] * 100:.1f}% | Confidence tier: {prediction['confidence_tier']}", f"Fair moneyline: {game['away_team']} {prediction['fair_away_ml']:+d} | {game['home_team']} {prediction['fair_home_ml']:+d}"])
        margin_team, margin_runs = (game['home_team'] if prediction['margin_calibrated'] >= 0 else game['away_team']), abs(prediction['margin_calibrated'])
        lines.append('Projected margin: Near even' if margin_runs < 0.15 else f"Projected margin: {margin_team} by {margin_runs:.1f} runs")
        lines.append(f"Model expected total: {prediction['total_calibrated']:.1f}")
        lines.append(f"Projected team runs: {game['away_team']} {prediction['away_runs']:.1f} | {game['home_team']} {prediction['home_runs']:.1f}")
        lines.append(f"Fair betting total: {prediction['total_bet_line']:.1f}")
        lines.append(f"Fair run line: {game['home_team']} -1.5 {prediction['run_line_prices'][0]:+d} | {game['away_team']} +1.5 {prediction['run_line_prices'][1]:+d}")
        market_comp = prediction.get('market_comp') or {}
        if market_comp.get('available'):
            lines.append('')
            lines.append(f"Sportsbook comparison: {market_comp.get('market_label', 'Market')}")
            ml_comp = market_comp.get('moneyline', {}) or {}
            if ml_comp.get('available'):
                lines.append(f" - Moneyline: {game['away_team']} {ml_comp.get('away_ml'):+d} | {game['home_team']} {ml_comp.get('home_ml'):+d} | best edge {ml_comp.get('best_side')} {float(ml_comp.get('best_edge', 0.0)) * 100:+.1f} pts")
                lines.append(f" - Moneyline actionability: {actionability_summary_text(ml_comp.get('action'))}")
            rl_comp = market_comp.get('run_line', {}) or {}
            if rl_comp.get('available'):
                lines.append(f" - Run line: {game['home_team']} {float(rl_comp.get('home_line', 0.0)):+.1f} {int(rl_comp.get('home_price', 0)):+d} | {game['away_team']} {float(rl_comp.get('away_line', 0.0)):+.1f} {int(rl_comp.get('away_price', 0)):+d} | best edge {rl_comp.get('best_selection')} {float(rl_comp.get('best_edge', 0.0)) * 100:+.1f} pts")
                lines.append(f" - Run-line actionability: {actionability_summary_text(rl_comp.get('action'))}")
            total_comp = market_comp.get('total', {}) or {}
            if total_comp.get('available'):
                total_price_text = ''
                if total_comp.get('over_price') is not None and total_comp.get('under_price') is not None:
                    total_price_text = f" (O {int(total_comp.get('over_price')):+d} / U {int(total_comp.get('under_price')):+d})"
                lines.append(f" - Total: {float(total_comp.get('market_total', 0.0)):.1f}{total_price_text} | lean {str(total_comp.get('lean', 'FLAT')).title()} by {float(total_comp.get('diff', 0.0)):+.1f} runs")
                lines.append(f" - Total actionability: {actionability_summary_text(total_comp.get('action'))}")
        lines.extend(['', 'Feature breakdown', f" - Offense edge: {feature['x_off']:+.2f} | Starter edge: {feature['x_starter']:+.2f} | Bullpen edge: {feature['x_bullpen']:+.2f}", f" - Park side adj: {feature['x_park']:+.3f} | Weather side adj: {feature['x_weather']:+.3f} | Context side adj: {feature['x_context']:+.3f}", f" - Shared run environment: {feature['shared_env']:+.2f} | Park total adj: {(finite(feature['park_history']['total_adj']) + finite(feature['park_static']['total_adj'])):+.2f} | Weather total adj: {feature['weather_adj']['total_adj']:+.2f}", f" - Away starter ({game.get('away_pitcher') or 'TBD'}): {feature['away_starter']['score']:+.2f} [{feature['away_starter']['source']}]", f" - Home starter ({game.get('home_pitcher') or 'TBD'}): {feature['home_starter']['score']:+.2f} [{feature['home_starter']['source']}]", f" - Bullpen scores: {game['away_team']} {feature['away_bullpen']:+.2f} | {game['home_team']} {feature['home_bullpen']:+.2f}", f" - Venue traits: elev {display_number(feature['venue_meta'].get('elevation_ft'), '.0f')} ft | {feature['venue_meta'].get('roof_type', 'unknown')} | {feature['venue_meta'].get('surface_type', 'unknown')}", f" - Weather: {feature['weather'].get('source', 'neutral')} | temp {display_number(feature['weather'].get('temp_f'), '.0f')}F | wind {display_number(feature['weather'].get('wind_mph'), '.0f')} mph | precip {display_number(feature['weather'].get('precip_prob'), '.0f')}%", f" - Confirmed lineups: {game['away_team']} {feature['lineups'].get('away', 0)}/9 | {game['home_team']} {feature['lineups'].get('home', 0)}/9", '', f" - {game['away_team']}: OffScore {finite(away_row.get('offense_score_blended'), 0.0):+.2f} | prior wt {finite(away_row.get('prior_weight'), 0.0) * 100:.0f}% | xwOBA/wOBA {display_number(away_row.get('xwoba'), '.3f')} | xSLG {display_number(away_row.get('xslg'), '.3f')} | xBA {display_number(away_row.get('xba'), '.3f')} | EV {display_number(away_row.get('avg_ev'), '.1f')} | HH% {finite(away_row.get('hardhit_rate'), 0.0) * 100:.1f}%", f" - {game['home_team']}: OffScore {finite(home_row.get('offense_score_blended'), 0.0):+.2f} | prior wt {finite(home_row.get('prior_weight'), 0.0) * 100:.0f}% | xwOBA/wOBA {display_number(home_row.get('xwoba'), '.3f')} | xSLG {display_number(home_row.get('xslg'), '.3f')} | xBA {display_number(home_row.get('xba'), '.3f')} | EV {display_number(home_row.get('avg_ev'), '.1f')} | HH% {finite(home_row.get('hardhit_rate'), 0.0) * 100:.1f}%", ''])
    lines.extend(['MODEL NOTES'] + [f" - {note}" for note in REPORT_NOTES])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    return path


def write_workbook(strengths, batters, pitchers, bullpen):
    path = os.path.join(OUT_DIR, 'MLB_Season_To_Date_2026.xlsx')
    team_metrics = strengths.copy().merge((bullpen if bullpen is not None else pd.DataFrame(columns=['team', 'bullpen_score'])), on='team', how='left')
    team_metrics['bullpen_score'] = pd.to_numeric(team_metrics.get('bullpen_score'), errors='coerce').fillna(0.0)
    team_metrics['power_rating'] = team_metrics['offense_score_blended'].fillna(0.0) + (0.35 * team_metrics['bullpen_score'])
    power = team_metrics.sort_values('power_rating', ascending=False).reset_index(drop=True)
    power.insert(0, 'rank', np.arange(1, len(power) + 1))
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        team_metrics.to_excel(writer, sheet_name='Team_Metrics', index=False)
        batters.sort_values('batter_score', ascending=False).head(20).to_excel(writer, sheet_name='Top_20_Batters', index=False)
        pitchers.sort_values('pitcher_score', ascending=False).head(20).to_excel(writer, sheet_name='Top_20_Pitchers', index=False)
        power.to_excel(writer, sheet_name='Power_Rankings', index=False)
    return path


def main():
    import market_helpers as market

    os.makedirs(OUT_DIR, exist_ok=True)
    report_date = today_local()
    warnings = []
    manual_warning = manual_market_warning(report_date)
    if manual_warning:
        warnings.append(manual_warning)
    client = RequestClient()
    model_state = load_model_state()
    backtest_log = load_validation_log()
    park_cache = load_park_cache()
    venue_cache = load_venue_metadata_cache()
    games, schedule_warnings = fetch_schedule_games(client, report_date)
    warnings.extend(schedule_warnings)
    phase = season_phase_for_games(games) if games else 'spring'
    weights = phase_weights_from_state(model_state, phase)
    validation = build_validation_summary(backtest_log)
    margin_sigma = estimate_margin_sigma(backtest_log)
    if not games:
        print(write_report(report_date, phase, weights, model_state, [], validation, warnings))
        return

    market.ensure_out_dir()
    market.ensure_market_db()
    market.ensure_market_csv_template()
    market_import_info, market_import_warn = market.import_all_market_lines(report_date, games)
    warnings.extend(market_import_warn)
    if int(market_import_info.get('total', 0)) > 0:
        warnings.append(f"Market rows loaded for report step: {int(market_import_info.get('total', 0))}.")

    closing_marked, closing_warn = market.mark_closing_market_snapshots(report_date, games)
    warnings.extend(closing_warn)
    if closing_marked:
        warnings.append(f"Closing market snapshots marked: {closing_marked} row(s).")

    settled_count, settle_warn = market.settle_bet_journal(report_date)
    warnings.extend(settle_warn)
    if settled_count:
        warnings.append(f"Historical market candidates settled: {settled_count}.")

    market_map, market_warn = market.load_market_consensus_for_date(report_date, games)
    warnings.extend(market_warn)
    if market_map:
        warnings.append(f"Current sportsbook consensus loaded for {len(market_map)} game(s).")

    season_start, pregame_end, game_types = dt.date(report_date.year, 1, 1), report_date, ('S|' if phase == 'spring' else 'R|')
    current_batter_raw = savant_statcast_csv(client, 'batter', report_date.year, season_start, pregame_end, game_types)
    batter_raw_30 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=30), pregame_end, game_types)
    batter_raw_14 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=14), pregame_end, game_types)
    batter_raw_7 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=7), pregame_end, game_types)
    season_offense = aggregate_team_offense(current_batter_raw)
    strengths = blend_team_offense(
        season_offense,
        aggregate_team_offense(batter_raw_30),
        aggregate_team_offense(batter_raw_14),
        aggregate_team_offense(batter_raw_7),
        previous_regular_season_team_offense(client, report_date.year - 1),
    )
    strengths = merge_handed_offense_splits(
        strengths,
        blend_team_offense(
            aggregate_team_offense(current_batter_raw, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_30, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_14, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_7, pitcher_hand='L'),
            previous_regular_season_team_offense(client, report_date.year - 1, pitcher_hand='L'),
        ),
        'vs_lhp',
    )
    strengths = merge_handed_offense_splits(
        strengths,
        blend_team_offense(
            aggregate_team_offense(current_batter_raw, pitcher_hand='R'),
            aggregate_team_offense(batter_raw_30, pitcher_hand='R'),
            aggregate_team_offense(batter_raw_14, pitcher_hand='R'),
            aggregate_team_offense(batter_raw_7, pitcher_hand='R'),
            previous_regular_season_team_offense(client, report_date.year - 1, pitcher_hand='R'),
        ),
        'vs_rhp',
    )
    defense_strengths = blend_team_defense(
        aggregate_team_defense(current_batter_raw),
        aggregate_team_defense(batter_raw_30),
        aggregate_team_defense(batter_raw_14),
        aggregate_team_defense(batter_raw_7),
        previous_regular_season_team_defense(client, report_date.year - 1),
    )
    strengths = strengths.merge(defense_strengths, on='team', how='left')
    batters = aggregate_batter_quality(current_batter_raw)
    current_pitchers = aggregate_pitcher_quality(savant_statcast_csv(client, 'pitcher', report_date.year, season_start, pregame_end, game_types))
    prior_pitchers = previous_regular_season_pitchers(client, report_date.year - 1)
    bullpen = bullpen_snapshot(client, report_date, 7, current_pitchers, prior_pitchers)
    bullpen_map = {str(row['team']): finite(row['bullpen_score'], 0.0) for _, row in bullpen.iterrows()} if not bullpen.empty else {}
    predictions = [predict_game(game, phase, weights, model_state, strengths, current_pitchers, prior_pitchers, bullpen_map, park_cache, client, venue_cache, margin_sigma) for game in games]
    for prediction in predictions:
        prediction['market_comp'] = market_comparison_for_prediction(prediction, market_map, phase, margin_sigma)

    journal_count, journal_warn = seed_open_bet_journal(report_date, predictions, market_map, phase, margin_sigma)
    warnings.extend(journal_warn)
    if journal_count:
        warnings.append(f"Bet journal refreshed with {journal_count} actionable market candidate(s).")

    save_venue_metadata_cache(venue_cache)
    report_path = write_report(report_date, phase, weights, model_state, predictions, validation, warnings)
    workbook_path = write_workbook(strengths, batters, current_pitchers, bullpen)
    print(report_path)
    print(workbook_path)


if __name__ == '__main__':
    main()



