
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
SEASON = 2026
MODEL_STATE_PATH = os.path.join(OUT_DIR, "model_state.json")
BACKTEST_LOG_PATH = os.path.join(OUT_DIR, "backtest_log.csv")
CLEAN_BACKTEST_LOG_PATH = os.path.join(OUT_DIR, "clean_backtest_log.csv")
LIVE_PREDICTION_ARCHIVE_PATH = os.path.join(OUT_DIR, "live_prediction_archive.csv")
SPORTSBOOK_DB_PATH = os.path.join(OUT_DIR, "sportsbook_lines.db")
QUANT_DB_PATH = os.path.join(OUT_DIR, "mlb_quant_dashboard.duckdb")
PARK_FACTORS_CACHE_PATH = os.path.join(OUT_DIR, "park_factors_cache.csv")
VENUE_METADATA_CACHE_PATH = os.path.join(OUT_DIR, "venue_metadata_cache.json")
UMPIRE_STATS_CACHE_PATH = os.path.join(OUT_DIR, "umpire_stats_cache.json")
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
STATSAPI_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
STATSAPI_VENUE_URL = "https://statsapi.mlb.com/api/v1/venues/{venue_id}"
STATSAPI_TEAM_ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SAVANT_STATCAST_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
STATSAPI_GAME_OFFICIALS_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MLB-Report-Clean-Rebuild/1.0)"}
LOCAL_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")
ROLL_BLEND = {"season": 0.55, "d30": 0.20, "d14": 0.15, "d7": 0.10}
DEFENSE_PRIOR_REGRESSION_BIP = {"season": 650.0, "d30": 350.0, "d14": 220.0, "d7": 160.0}
TRAVEL_REST_LOOKBACK_DAYS = 5
OFFENSE_PRIOR_REGRESSION_PA = {"season": 450.0, "d30": 300.0, "d14": 200.0, "d7": 140.0}
STARTER_PRIOR_REGRESSION_BF = 60.0
BATTER_PRIOR_REGRESSION_PA = 80.0
MULTIYEAR_PRIOR_WEIGHTS = [1.00, 0.82, 0.67, 0.54, 0.43]
LINEUP_EARN_BACK_MIN_GAMES = 40
LINEUP_EARN_BACK_MIN_IMPROVEMENT = 0.002
LINEUP_EARN_BACK_MIN_CONFIRMED = 9
REALIZED_UNCERTAINTY_MIN_GAMES = 18
REALIZED_UNCERTAINTY_MIN_BETS = 8
REALIZED_UNCERTAINTY_HIGH_CONF_GAMES = 45
REALIZED_UNCERTAINTY_HIGH_CONF_BETS = 20
TOTAL_SHRINK_MIN_IMPROVEMENT = 0.05
MARGIN_SHRINK_MIN_IMPROVEMENT = 0.05
TOTAL_CALIBRATION_MIN_GAMES = 30
TOTAL_CALIBRATION_MIN_IMPROVEMENT = 0.03
TOTAL_SIGMA_CALIBRATION_MIN_GAMES = 24
TOTAL_SIGMA_CALIBRATION_MIN_IMPROVEMENT = 0.01
SHRINKAGE_BRIDGE_DAYS = 21
LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION = 1
LEAGUE_BASE_RUNS = {"spring": 4.95, "regular": 4.55}
LINEUP_SLOT_WEIGHTS = [1.22, 1.16, 1.10, 1.06, 1.00, 0.95, 0.91, 0.88, 0.86]
STARTER_BASE_EXPECTED_IP = {"spring": 4.2, "regular": 5.2}
BULLPEN_RECENCY_WEIGHTS = {1: 1.00, 2: 0.76, 3: 0.55, 4: 0.36, 5: 0.22, 6: 0.14, 7: 0.08}
TOTAL_MODEL_PHASE_PARAMS = {
    "spring": {
        "base_runs": 4.58,
        "offense_coef": 0.62,
        "starter_coef": 0.80,
        "bullpen_coef": 0.32,
        "shared_total_coef": 0.34,
        "schedule_coef": 0.04,
        "defense_coef": 0.08,
        "lineup_coef": 0.10,
        "bridge_coef": 0.08,
        "availability_coef": 0.03,
        "shared_env_offense_coef": 0.05,
        "shared_env_support_coef": 0.04,
        "shared_env_starter_coef": 0.08,
        "shared_env_bullpen_coef": 0.04,
        "shared_env_short_coef": 0.12,
        "shared_env_bridge_coef": 0.08,
        "shared_env_uncertainty_coef": 0.05,
        "run_floor": 1.7,
        "run_cap": 7.9,
        "env_cap": 0.90,
    },
    "regular": {
        "base_runs": 4.50,
        "offense_coef": 0.65,
        "starter_coef": 0.84,
        "bullpen_coef": 0.35,
        "shared_total_coef": 0.38,
        "schedule_coef": 0.05,
        "defense_coef": 0.09,
        "lineup_coef": 0.11,
        "bridge_coef": 0.09,
        "availability_coef": 0.03,
        "shared_env_offense_coef": 0.06,
        "shared_env_support_coef": 0.05,
        "shared_env_starter_coef": 0.09,
        "shared_env_bullpen_coef": 0.05,
        "shared_env_short_coef": 0.14,
        "shared_env_bridge_coef": 0.09,
        "shared_env_uncertainty_coef": 0.06,
        "run_floor": 1.8,
        "run_cap": 8.2,
        "env_cap": 1.05,
    },
}
TOTAL_SIGMA_PHASE_PARAMS = {
    "spring": {
        "base_sigma": 4.10,
        "starter_uncertainty_coef": 0.34,
        "short_start_coef": 0.62,
        "bullpen_fragility_coef": 0.48,
        "lineup_missing_coef": 0.22,
        "env_coef": 0.28,
        "weather_vol_coef": 0.18,
        "offense_pressure_coef": 0.10,
        "min_sigma": 3.10,
        "max_sigma": 6.20,
    },
    "regular": {
        "base_sigma": 3.75,
        "starter_uncertainty_coef": 0.28,
        "short_start_coef": 0.48,
        "bullpen_fragility_coef": 0.38,
        "lineup_missing_coef": 0.14,
        "env_coef": 0.22,
        "weather_vol_coef": 0.14,
        "offense_pressure_coef": 0.08,
        "min_sigma": 2.85,
        "max_sigma": 5.60,
    },
}
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

def total_model_params(phase):
    phase_name = str(phase or 'spring').lower()
    params = dict(TOTAL_MODEL_PHASE_PARAMS.get('regular') or {})
    params.update(TOTAL_MODEL_PHASE_PARAMS.get(phase_name) or {})
    return params


def total_sigma_params(phase):
    phase_name = str(phase or 'spring').lower()
    params = dict(TOTAL_SIGMA_PHASE_PARAMS.get('regular') or {})
    params.update(TOTAL_SIGMA_PHASE_PARAMS.get(phase_name) or {})
    return params

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
        self.roster_cache = {}

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

def save_model_state(state):
    with open(MODEL_STATE_PATH, 'w', encoding='utf-8') as handle:
        json.dump(state or {}, handle, indent=2)

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

def load_live_prediction_archive():
    if not os.path.exists(LIVE_PREDICTION_ARCHIVE_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LIVE_PREDICTION_ARCHIVE_PATH)
    except Exception:
        return pd.DataFrame()

def save_live_prediction_archive(df):
    archive = pd.DataFrame() if df is None else df.copy()
    archive.to_csv(LIVE_PREDICTION_ARCHIVE_PATH, index=False)


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

def load_umpire_stats_cache():
    if not os.path.exists(UMPIRE_STATS_CACHE_PATH):
        return {}
    try:
        with open(UMPIRE_STATS_CACHE_PATH, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}

def save_umpire_stats_cache(cache):
    with open(UMPIRE_STATS_CACHE_PATH, 'w', encoding='utf-8') as handle:
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
def lineup_multiplier_from_state(state, phase):
    if str(phase).lower() != 'regular':
        return 1.0
    profile = (state.get('lineup_earn_back') or {}) if isinstance(state, dict) else {}
    if not profile.get('apply'):
        return 0.0
    return float(np.clip(finite(profile.get('lineup_multiplier'), 0.0), 0.0, 1.0))

def apply_probability_calibration(probability, state, phase):
    profile = ((state.get('probability_calibration') or {}).get(phase) or {})
    p = min(max(finite(probability, 0.5), 1e-6), 1.0 - 1e-6)
    if not profile.get('apply'):
        return p
    logit = math.log(p / (1.0 - p))
    return sigmoid(finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * logit))


def simple_power_probability_from_feature(x_off, phase, state):
    spring_simple = (((state.get('spring_guardrail') or {}).get('simple_power_weights')) or {}) if isinstance(state, dict) else {}
    regular_simple = (((state.get('regular_earn_back') or {}).get('adjusted_weights')) or {}) if isinstance(state, dict) else {}
    source = spring_simple if str(phase or 'spring').lower() == 'spring' else regular_simple
    w_off = finite(source.get('w_off'), 0.35)
    hfa = finite(source.get('home_field'), 0.05)
    return float(np.clip(sigmoid((w_off * finite(x_off, 0.0)) + hfa), 1e-6, 1.0 - 1e-6))


def _binary_log_loss(y_true, probs):
    if y_true is None or probs is None:
        return None
    y = pd.to_numeric(pd.Series(y_true), errors='coerce')
    p = pd.to_numeric(pd.Series(probs), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    valid = (~y.isna()) & (~p.isna())
    if not valid.any():
        return None
    yv = y[valid].astype(int)
    pv = p[valid].astype(float)
    return float((-(yv * np.log(pv)) - ((1 - yv) * np.log(1.0 - pv))).mean())


def fit_total_calibration_profile(state, backtest_df):
    base_state = dict(state or {})
    profile = {
        'spring': {
            'apply': False,
            'mode': 'identity',
            'intercept': 0.0,
            'slope': 1.0,
            'tail_slope': 0.0,
            'tail_threshold': 9.0,
            'games': 0,
            'raw_rmse': None,
            'calibrated_rmse': None,
            'reason': 'insufficient spring sample for total calibration fitting',
        },
        'regular': {
            'apply': False,
            'mode': 'identity',
            'intercept': 0.0,
            'slope': 1.0,
            'tail_slope': 0.0,
            'tail_threshold': 9.0,
            'games': 0,
            'raw_rmse': None,
            'calibrated_rmse': None,
            'reason': 'insufficient regular sample for total calibration fitting',
        },
    }
    if backtest_df is None or backtest_df.empty:
        base_state['total_calibration'] = profile
        return base_state, profile
    for phase_name in ['spring', 'regular']:
        phase_df = backtest_df[backtest_df.get('phase').astype(str).str.lower() == phase_name].copy() if 'phase' in backtest_df.columns else pd.DataFrame()
        phase_profile = profile[phase_name]
        phase_profile['games'] = int(len(phase_df))
        raw_total = pd.to_numeric(phase_df.get('projected_total_raw'), errors='coerce') if 'projected_total_raw' in phase_df.columns else pd.Series(dtype=float)
        if raw_total.empty:
            raw_total = pd.to_numeric(phase_df.get('projected_total'), errors='coerce')
        actual_total = pd.to_numeric(phase_df.get('home_score'), errors='coerce') + pd.to_numeric(phase_df.get('away_score'), errors='coerce')
        valid = (~raw_total.isna()) & (~actual_total.isna())
        if int(valid.sum()) < TOTAL_CALIBRATION_MIN_GAMES:
            phase_profile['reason'] = f'insufficient {phase_name} sample for total calibration fitting ({int(valid.sum())} rows)'
            continue
        raw = raw_total[valid].astype(float).reset_index(drop=True)
        actual = actual_total[valid].astype(float).reset_index(drop=True)
        raw_rmse = _rmse_metric(actual, raw)
        if raw_rmse is None:
            phase_profile['reason'] = f'no valid {phase_name} rows for total calibration fitting'
            continue
        if float(raw.std()) >= 0.15:
            slope, intercept = np.polyfit(raw, actual, 1)
        else:
            slope, intercept = 1.0, float(actual.mean() - raw.mean())
        slope = float(np.clip(finite(slope, 1.0), 0.65, 1.20))
        intercept = float(np.clip(finite(intercept, 0.0), -2.0, 2.0))
        threshold = float(max(9.0, np.quantile(raw, 0.70)))
        adjusted = intercept + (slope * raw)
        excess = np.maximum(raw - threshold, 0.0)
        if float(excess.sum()) > 1e-6:
            residual = actual - adjusted
            denom = float(np.square(excess).sum())
            tail_slope = float(np.clip((residual * excess).sum() / denom, -0.40, 0.40)) if denom > 1e-6 else 0.0
        else:
            tail_slope = 0.0
        calibrated = adjusted + (tail_slope * excess)
        calibrated_rmse = _rmse_metric(actual, calibrated)
        phase_profile['raw_rmse'] = raw_rmse
        phase_profile['calibrated_rmse'] = calibrated_rmse
        improvement = None if calibrated_rmse is None else float(raw_rmse - calibrated_rmse)
        if improvement is not None and improvement >= TOTAL_CALIBRATION_MIN_IMPROVEMENT:
            phase_profile.update({
                'apply': True,
                'mode': 'affine_total_calibration',
                'intercept': intercept,
                'slope': slope,
                'tail_slope': tail_slope,
                'tail_threshold': threshold,
                'reason': f'{phase_name} total calibration improved RMSE by {improvement:.3f} over {int(valid.sum())} historical games',
            })
        else:
            phase_profile.update({
                'apply': False,
                'mode': 'identity',
                'intercept': 0.0,
                'slope': 1.0,
                'tail_slope': 0.0,
                'tail_threshold': threshold,
                'reason': f'{phase_name} total calibration did not improve RMSE enough to activate ({raw_rmse:.3f} to {finite(calibrated_rmse, raw_rmse):.3f})',
            })
    base_state['total_calibration'] = profile
    return base_state, profile


def fit_probability_shrinkage_profile(state, backtest_df):
    base_state = dict(state or {})
    profile = {
        'spring': {
            'apply': False,
            'alpha': 0.0,
            'games': 0,
            'model_log_loss': None,
            'simple_log_loss': None,
            'blended_log_loss': None,
            'reason': 'insufficient spring validation sample for shrinkage fitting',
        },
        'regular': {
            'apply': False,
            'alpha': 0.0,
            'games': 0,
            'model_log_loss': None,
            'simple_log_loss': None,
            'blended_log_loss': None,
            'reason': 'insufficient regular validation sample for shrinkage fitting',
        },
        'regular_bridge': {
            'apply': False,
            'max_alpha': 0.0,
            'bridge_days': 21,
            'reason': 'no spring shrinkage signal available for early-regular bridge',
        },
    }
    oos = validation_oos_slice(backtest_df)
    if oos is None or oos.empty:
        return base_state, profile
    for phase_name in ['spring', 'regular']:
        phase_df = oos[oos.get('phase').astype(str).str.lower() == phase_name].copy() if 'phase' in oos.columns else pd.DataFrame()
        games = int(len(phase_df))
        phase_profile = profile[phase_name]
        phase_profile['games'] = games
        if games < 12:
            phase_profile['reason'] = f'insufficient {phase_name} OOS sample for shrinkage fitting ({games} games)'
            continue
        model_probs = pd.to_numeric(phase_df.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
        simple_probs = simple_power_baseline_probabilities(phase_df, base_state).clip(1e-6, 1.0 - 1e-6)
        y = pd.to_numeric(phase_df.get('y_home'), errors='coerce')
        valid = (~model_probs.isna()) & (~simple_probs.isna()) & (~y.isna())
        if not valid.any():
            phase_profile['reason'] = f'no valid {phase_name} rows for shrinkage fitting'
            continue
        model_probs = model_probs[valid].astype(float).reset_index(drop=True)
        simple_probs = simple_probs[valid].astype(float).reset_index(drop=True)
        yv = y[valid].astype(int).reset_index(drop=True)
        model_log_loss = _binary_log_loss(yv, model_probs)
        simple_log_loss = _binary_log_loss(yv, simple_probs)
        best_alpha = 0.0
        best_log_loss = model_log_loss
        for alpha in np.linspace(0.0, 0.60, 13):
            blended = ((1.0 - float(alpha)) * model_probs) + (float(alpha) * simple_probs)
            current_log_loss = _binary_log_loss(yv, blended)
            if current_log_loss is not None and (best_log_loss is None or current_log_loss < best_log_loss):
                best_log_loss = current_log_loss
                best_alpha = float(alpha)
        phase_profile['model_log_loss'] = model_log_loss
        phase_profile['simple_log_loss'] = simple_log_loss
        phase_profile['blended_log_loss'] = best_log_loss
        improvement = None if (model_log_loss is None or best_log_loss is None) else float(model_log_loss - best_log_loss)
        if improvement is not None and improvement >= 0.001 and best_alpha >= 0.05:
            phase_profile.update({
                'apply': True,
                'alpha': float(best_alpha),
                'reason': f'{phase_name} shrinkage improved log loss by {improvement:.4f} over {games} OOS games',
            })
        else:
            phase_profile.update({
                'apply': False,
                'alpha': 0.0,
                'reason': f'{phase_name} shrinkage did not improve log loss enough over {games} OOS games',
            })
    spring_alpha = float(profile['spring'].get('alpha') or 0.0)
    if spring_alpha > 0:
        bridge_alpha = float(np.clip(0.55 * spring_alpha, 0.05, 0.20))
        profile['regular_bridge'] = {
            'apply': True,
            'max_alpha': bridge_alpha,
            'bridge_days': 21,
            'reason': f'early regular season bridge derived from spring shrinkage alpha {spring_alpha:.2f}',
        }
    base_state['probability_shrinkage'] = profile
    return base_state, profile


def probability_shrinkage_alpha(state, phase, target_date):
    profile = (state.get('probability_shrinkage') or {}) if isinstance(state, dict) else {}
    phase_name = str(phase or 'spring').lower()
    phase_profile = profile.get(phase_name) or {}
    if phase_profile.get('apply'):
        return float(np.clip(finite(phase_profile.get('alpha'), 0.0), 0.0, 0.60))
    if phase_name != 'regular':
        return 0.0
    bridge = profile.get('regular_bridge') or {}
    if not bridge.get('apply'):
        return 0.0
    try:
        import market_helpers as market
        opening_day = market.opening_day_for_season(int(target_date.year)) if target_date is not None else None
    except Exception:
        opening_day = None
    if opening_day is None or target_date is None:
        return 0.0
    days = int((target_date - opening_day).days)
    bridge_days = max(int(finite(bridge.get('bridge_days'), 21.0)), 1)
    if days < 0 or days >= bridge_days:
        return 0.0
    max_alpha = float(np.clip(finite(bridge.get('max_alpha'), 0.0), 0.0, 0.25))
    decay = 1.0 - (days / float(bridge_days))
    return float(np.clip(max_alpha * decay, 0.0, max_alpha))


def blend_probability_with_simple_baseline(probability, x_off, phase, state, target_date):
    p_model = float(np.clip(finite(probability, 0.5), 1e-6, 1.0 - 1e-6))
    p_simple = simple_power_probability_from_feature(x_off, phase, state)
    alpha = probability_shrinkage_alpha(state, phase, target_date)
    if alpha <= 0:
        return p_model, p_simple, 0.0
    blended = ((1.0 - alpha) * p_model) + (alpha * p_simple)
    return float(np.clip(blended, 1e-6, 1.0 - 1e-6)), p_simple, float(alpha)

def _rmse_metric(actual, predicted):
    if actual is None or predicted is None:
        return None
    a = pd.to_numeric(pd.Series(actual), errors='coerce')
    p = pd.to_numeric(pd.Series(predicted), errors='coerce')
    valid = (~a.isna()) & (~p.isna())
    if not valid.any():
        return None
    diff = a[valid].astype(float) - p[valid].astype(float)
    return float(np.sqrt(np.mean(np.square(diff))))


def _mae_metric(actual, predicted):
    if actual is None or predicted is None:
        return None
    a = pd.to_numeric(pd.Series(actual), errors='coerce')
    p = pd.to_numeric(pd.Series(predicted), errors='coerce')
    valid = (~a.isna()) & (~p.isna())
    if not valid.any():
        return None
    diff = a[valid].astype(float) - p[valid].astype(float)
    return float(np.mean(np.abs(diff)))


def _gaussian_nll_metric(actual, predicted, sigma):
    if actual is None or predicted is None or sigma is None:
        return None
    a = pd.to_numeric(pd.Series(actual), errors='coerce')
    p = pd.to_numeric(pd.Series(predicted), errors='coerce')
    s = pd.to_numeric(pd.Series(sigma), errors='coerce')
    valid = (~a.isna()) & (~p.isna()) & (~s.isna()) & (s > 0)
    if not valid.any():
        return None
    resid = a[valid].astype(float) - p[valid].astype(float)
    sigma_vals = s[valid].astype(float).clip(lower=1e-6)
    loss = 0.5 * np.log(2.0 * np.pi * np.square(sigma_vals)) + 0.5 * np.square(resid / sigma_vals)
    return float(np.mean(loss))


def simple_total_baseline_from_phase(phase, state):
    phase_name = str(phase or 'spring').lower()
    base_total = 2.0 * LEAGUE_BASE_RUNS.get(phase_name, 4.6)
    profile = ((state.get('total_calibration') or {}).get(phase_name) or {}) if isinstance(state, dict) else {}
    if profile.get('apply'):
        adjusted = finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * base_total)
        return float(np.clip(adjusted, 5.5, 15.5))
    return float(np.clip(base_total, 5.5, 15.5))


def simple_margin_baseline_from_feature(x_off, phase, state):
    p_simple = simple_power_probability_from_feature(x_off, phase, state)
    weights = phase_weights_from_state(state or {}, str(phase or 'spring').lower()) if isinstance(state, dict) else {'spread_scale': 8.0}
    spread_scale = finite(weights.get('spread_scale'), 8.0)
    return float(np.clip((p_simple - 0.5) * 2.0 * spread_scale, -6.0, 6.0))


def fit_total_shrinkage_profile(state, backtest_df):
    base_state = dict(state or {})
    profile = {
        'spring': {'apply': False, 'alpha': 0.0, 'games': 0, 'model_rmse': None, 'simple_rmse': None, 'blended_rmse': None, 'reason': 'insufficient spring sample for total shrinkage fitting'},
        'regular': {'apply': False, 'alpha': 0.0, 'games': 0, 'model_rmse': None, 'simple_rmse': None, 'blended_rmse': None, 'reason': 'insufficient regular sample for total shrinkage fitting'},
        'regular_bridge': {'apply': False, 'max_alpha': 0.0, 'bridge_days': SHRINKAGE_BRIDGE_DAYS, 'reason': 'no spring total shrinkage signal available for early-regular bridge'},
    }
    oos = validation_oos_slice(backtest_df)
    if oos is None or oos.empty:
        return base_state, profile
    for phase_name in ['spring', 'regular']:
        phase_df = oos[oos.get('phase').astype(str).str.lower() == phase_name].copy() if 'phase' in oos.columns else pd.DataFrame()
        games = int(len(phase_df))
        phase_profile = profile[phase_name]
        phase_profile['games'] = games
        if games < 12:
            phase_profile['reason'] = f'insufficient {phase_name} OOS sample for total shrinkage fitting ({games} games)'
            continue
        actual_total = pd.to_numeric(phase_df.get('home_score'), errors='coerce') + pd.to_numeric(phase_df.get('away_score'), errors='coerce')
        model_total = pd.to_numeric(phase_df.get('projected_total'), errors='coerce')
        simple_total = pd.Series([simple_total_baseline_from_phase(phase_name, base_state)] * len(phase_df), index=phase_df.index, dtype=float)
        valid = (~actual_total.isna()) & (~model_total.isna()) & (~simple_total.isna())
        if not valid.any():
            phase_profile['reason'] = f'no valid {phase_name} rows for total shrinkage fitting'
            continue
        actual = actual_total[valid].astype(float).reset_index(drop=True)
        model_vals = model_total[valid].astype(float).reset_index(drop=True)
        simple_vals = simple_total[valid].astype(float).reset_index(drop=True)
        model_rmse = _rmse_metric(actual, model_vals)
        simple_rmse = _rmse_metric(actual, simple_vals)
        best_alpha = 0.0
        best_rmse = model_rmse
        for alpha in np.linspace(0.0, 0.60, 13):
            blended = ((1.0 - float(alpha)) * model_vals) + (float(alpha) * simple_vals)
            current_rmse = _rmse_metric(actual, blended)
            if current_rmse is not None and (best_rmse is None or current_rmse < best_rmse):
                best_rmse = current_rmse
                best_alpha = float(alpha)
        phase_profile['model_rmse'] = model_rmse
        phase_profile['simple_rmse'] = simple_rmse
        phase_profile['blended_rmse'] = best_rmse
        improvement = None if (model_rmse is None or best_rmse is None) else float(model_rmse - best_rmse)
        if improvement is not None and improvement >= TOTAL_SHRINK_MIN_IMPROVEMENT and best_alpha >= 0.05:
            phase_profile.update({'apply': True, 'alpha': float(best_alpha), 'reason': f'{phase_name} total shrinkage improved RMSE by {improvement:.3f} over {games} OOS games'})
        else:
            phase_profile.update({'apply': False, 'alpha': 0.0, 'reason': f'{phase_name} total shrinkage did not improve RMSE enough over {games} OOS games'})
    spring_alpha = float(profile['spring'].get('alpha') or 0.0)
    if spring_alpha > 0:
        bridge_alpha = float(np.clip(0.55 * spring_alpha, 0.05, 0.20))
        profile['regular_bridge'] = {'apply': True, 'max_alpha': bridge_alpha, 'bridge_days': SHRINKAGE_BRIDGE_DAYS, 'reason': f'early regular season bridge derived from spring total shrinkage alpha {spring_alpha:.2f}'}
    base_state['total_shrinkage'] = profile
    return base_state, profile


def fit_total_sigma_profile(state, backtest_df):
    base_state = dict(state or {})
    profile = {
        'spring': {'apply': False, 'scale': 1.0, 'games': 0, 'raw_nll': None, 'calibrated_nll': None, 'z_mean': None, 'z_std': None, 'reason': 'insufficient spring sample for total sigma calibration'},
        'regular': {'apply': False, 'scale': 1.0, 'games': 0, 'raw_nll': None, 'calibrated_nll': None, 'z_mean': None, 'z_std': None, 'reason': 'insufficient regular sample for total sigma calibration'},
    }
    oos = validation_oos_slice(backtest_df)
    if oos is None or oos.empty:
        base_state['total_sigma_calibration'] = profile
        return base_state, profile
    for phase_name in ['spring', 'regular']:
        phase_df = oos[oos.get('phase').astype(str).str.lower() == phase_name].copy() if 'phase' in oos.columns else pd.DataFrame()
        actual_total = pd.to_numeric(phase_df.get('home_score'), errors='coerce') + pd.to_numeric(phase_df.get('away_score'), errors='coerce')
        projected_total = pd.to_numeric(phase_df.get('projected_total'), errors='coerce')
        sigma_source = pd.to_numeric(phase_df.get('projected_total_sigma_raw'), errors='coerce') if 'projected_total_sigma_raw' in phase_df.columns else pd.to_numeric(phase_df.get('projected_total_sigma'), errors='coerce')
        valid = (~actual_total.isna()) & (~projected_total.isna()) & (~sigma_source.isna()) & (sigma_source > 0)
        games = int(valid.sum())
        phase_profile = profile[phase_name]
        phase_profile['games'] = games
        if games < TOTAL_SIGMA_CALIBRATION_MIN_GAMES:
            phase_profile['reason'] = f'insufficient {phase_name} sample for total sigma calibration ({games} rows)'
            continue
        actual = actual_total[valid].astype(float).reset_index(drop=True)
        projected = projected_total[valid].astype(float).reset_index(drop=True)
        sigma_vals = sigma_source[valid].astype(float).reset_index(drop=True)
        resid = actual - projected
        raw_nll = _gaussian_nll_metric(actual, projected, sigma_vals)
        rms_z = float(np.sqrt(np.mean(np.square(resid / sigma_vals)))) if len(sigma_vals) else 1.0
        scale = float(np.clip(rms_z, 0.75, 1.25))
        calibrated_sigma = sigma_vals * scale
        calibrated_nll = _gaussian_nll_metric(actual, projected, calibrated_sigma)
        z = resid / sigma_vals
        phase_profile['raw_nll'] = raw_nll
        phase_profile['calibrated_nll'] = calibrated_nll
        phase_profile['z_mean'] = float(np.mean(z))
        phase_profile['z_std'] = float(np.std(z))
        improvement = None if (raw_nll is None or calibrated_nll is None) else float(raw_nll - calibrated_nll)
        if improvement is not None and improvement >= TOTAL_SIGMA_CALIBRATION_MIN_IMPROVEMENT and abs(scale - 1.0) >= 0.03:
            phase_profile.update({'apply': True, 'scale': scale, 'reason': f'{phase_name} total sigma calibration improved Gaussian NLL by {improvement:.4f} over {games} OOS games'})
        else:
            phase_profile.update({'apply': False, 'scale': 1.0, 'reason': f'{phase_name} total sigma calibration did not improve NLL enough over {games} OOS games'})
    base_state['total_sigma_calibration'] = profile
    return base_state, profile


def fit_margin_shrinkage_profile(state, backtest_df):
    base_state = dict(state or {})
    profile = {
        'spring': {'apply': False, 'alpha': 0.0, 'games': 0, 'model_rmse': None, 'simple_rmse': None, 'blended_rmse': None, 'reason': 'insufficient spring sample for margin shrinkage fitting'},
        'regular': {'apply': False, 'alpha': 0.0, 'games': 0, 'model_rmse': None, 'simple_rmse': None, 'blended_rmse': None, 'reason': 'insufficient regular sample for margin shrinkage fitting'},
        'regular_bridge': {'apply': False, 'max_alpha': 0.0, 'bridge_days': SHRINKAGE_BRIDGE_DAYS, 'reason': 'no spring margin shrinkage signal available for early-regular bridge'},
    }
    oos = validation_oos_slice(backtest_df)
    if oos is None or oos.empty:
        return base_state, profile
    for phase_name in ['spring', 'regular']:
        phase_df = oos[oos.get('phase').astype(str).str.lower() == phase_name].copy() if 'phase' in oos.columns else pd.DataFrame()
        games = int(len(phase_df))
        phase_profile = profile[phase_name]
        phase_profile['games'] = games
        if games < 12:
            phase_profile['reason'] = f'insufficient {phase_name} OOS sample for margin shrinkage fitting ({games} games)'
            continue
        actual_margin = pd.to_numeric(phase_df.get('home_score'), errors='coerce') - pd.to_numeric(phase_df.get('away_score'), errors='coerce')
        model_margin = pd.to_numeric(phase_df.get('projected_margin'), errors='coerce')
        simple_margin = phase_df.apply(lambda row: simple_margin_baseline_from_feature(row.get('x_off'), phase_name, base_state), axis=1)
        valid = (~actual_margin.isna()) & (~model_margin.isna()) & (~simple_margin.isna())
        if not valid.any():
            phase_profile['reason'] = f'no valid {phase_name} rows for margin shrinkage fitting'
            continue
        actual = actual_margin[valid].astype(float).reset_index(drop=True)
        model_vals = model_margin[valid].astype(float).reset_index(drop=True)
        simple_vals = pd.to_numeric(simple_margin[valid], errors='coerce').astype(float).reset_index(drop=True)
        model_rmse = _rmse_metric(actual, model_vals)
        simple_rmse = _rmse_metric(actual, simple_vals)
        best_alpha = 0.0
        best_rmse = model_rmse
        for alpha in np.linspace(0.0, 0.60, 13):
            blended = ((1.0 - float(alpha)) * model_vals) + (float(alpha) * simple_vals)
            current_rmse = _rmse_metric(actual, blended)
            if current_rmse is not None and (best_rmse is None or current_rmse < best_rmse):
                best_rmse = current_rmse
                best_alpha = float(alpha)
        phase_profile['model_rmse'] = model_rmse
        phase_profile['simple_rmse'] = simple_rmse
        phase_profile['blended_rmse'] = best_rmse
        improvement = None if (model_rmse is None or best_rmse is None) else float(model_rmse - best_rmse)
        if improvement is not None and improvement >= MARGIN_SHRINK_MIN_IMPROVEMENT and best_alpha >= 0.05:
            phase_profile.update({'apply': True, 'alpha': float(best_alpha), 'reason': f'{phase_name} margin shrinkage improved RMSE by {improvement:.3f} over {games} OOS games'})
        else:
            phase_profile.update({'apply': False, 'alpha': 0.0, 'reason': f'{phase_name} margin shrinkage did not improve RMSE enough over {games} OOS games'})
    spring_alpha = float(profile['spring'].get('alpha') or 0.0)
    if spring_alpha > 0:
        bridge_alpha = float(np.clip(0.55 * spring_alpha, 0.05, 0.20))
        profile['regular_bridge'] = {'apply': True, 'max_alpha': bridge_alpha, 'bridge_days': SHRINKAGE_BRIDGE_DAYS, 'reason': f'early regular season bridge derived from spring margin shrinkage alpha {spring_alpha:.2f}'}
    base_state['margin_shrinkage'] = profile
    return base_state, profile


def total_shrinkage_alpha(state, phase, target_date):
    profile = (state.get('total_shrinkage') or {}) if isinstance(state, dict) else {}
    phase_name = str(phase or 'spring').lower()
    phase_profile = profile.get(phase_name) or {}
    if phase_profile.get('apply'):
        return float(np.clip(finite(phase_profile.get('alpha'), 0.0), 0.0, 0.60))
    if phase_name != 'regular':
        return 0.0
    bridge = profile.get('regular_bridge') or {}
    if not bridge.get('apply'):
        return 0.0
    try:
        import market_helpers as market
        opening_day = market.opening_day_for_season(int(target_date.year)) if target_date is not None else None
    except Exception:
        opening_day = None
    if opening_day is None or target_date is None:
        return 0.0
    days = int((target_date - opening_day).days)
    bridge_days = max(int(finite(bridge.get('bridge_days'), SHRINKAGE_BRIDGE_DAYS)), 1)
    if days < 0 or days >= bridge_days:
        return 0.0
    max_alpha = float(np.clip(finite(bridge.get('max_alpha'), 0.0), 0.0, 0.25))
    return float(np.clip(max_alpha * (1.0 - (days / float(bridge_days))), 0.0, max_alpha))


def margin_shrinkage_alpha(state, phase, target_date):
    profile = (state.get('margin_shrinkage') or {}) if isinstance(state, dict) else {}
    phase_name = str(phase or 'spring').lower()
    phase_profile = profile.get(phase_name) or {}
    if phase_profile.get('apply'):
        return float(np.clip(finite(phase_profile.get('alpha'), 0.0), 0.0, 0.60))
    if phase_name != 'regular':
        return 0.0
    bridge = profile.get('regular_bridge') or {}
    if not bridge.get('apply'):
        return 0.0
    try:
        import market_helpers as market
        opening_day = market.opening_day_for_season(int(target_date.year)) if target_date is not None else None
    except Exception:
        opening_day = None
    if opening_day is None or target_date is None:
        return 0.0
    days = int((target_date - opening_day).days)
    bridge_days = max(int(finite(bridge.get('bridge_days'), SHRINKAGE_BRIDGE_DAYS)), 1)
    if days < 0 or days >= bridge_days:
        return 0.0
    max_alpha = float(np.clip(finite(bridge.get('max_alpha'), 0.0), 0.0, 0.25))
    return float(np.clip(max_alpha * (1.0 - (days / float(bridge_days))), 0.0, max_alpha))


def blend_total_with_baseline(total_value, phase, state, target_date):
    total_model = float(np.clip(finite(total_value, 9.0), 5.5, 15.5))
    total_simple = simple_total_baseline_from_phase(phase, state)
    alpha = total_shrinkage_alpha(state, phase, target_date)
    if alpha <= 0:
        return total_model, total_simple, 0.0
    blended = ((1.0 - alpha) * total_model) + (alpha * total_simple)
    return float(np.clip(blended, 5.5, 15.5)), float(total_simple), float(alpha)


def blend_margin_with_baseline(margin_value, x_off, phase, state, target_date):
    margin_model = float(np.clip(finite(margin_value, 0.0), -7.0, 7.0))
    margin_simple = simple_margin_baseline_from_feature(x_off, phase, state)
    alpha = margin_shrinkage_alpha(state, phase, target_date)
    if alpha <= 0:
        return margin_model, margin_simple, 0.0
    blended = ((1.0 - alpha) * margin_model) + (alpha * margin_simple)
    return float(np.clip(blended, -7.0, 7.0)), float(margin_simple), float(alpha)
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

def apply_total_sigma_calibration(total_sigma, state, phase):
    profile = ((state.get('total_sigma_calibration') or {}).get(phase) or {})
    sigma_value = finite(total_sigma, 4.0)
    if not profile.get('apply'):
        return float(np.clip(sigma_value, 2.0, 7.0))
    scale = finite(profile.get('scale'), 1.0)
    return float(np.clip(sigma_value * scale, 2.0, 7.0))


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
                    'away_team_id': safe_float(away_team.get('id')),
                    'home_team_id': safe_float(home_team.get('id')),
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


def weighted_regular_season_years(latest_year, lookback=5):
    years = []
    for idx in range(int(lookback)):
        years.append((int(latest_year) - idx, float(MULTIYEAR_PRIOR_WEIGHTS[min(idx, len(MULTIYEAR_PRIOR_WEIGHTS) - 1)])))
    return years

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
        return pd.DataFrame(columns=['batter', 'batter_id', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'batter_score'])
    work = df.copy()
    work['batter_name'] = work.get('player_name', '').apply(canonical_player_name)
    work['batter_id'] = pd.to_numeric(work.get('batter'), errors='coerce')
    work['launch_speed'] = pd.to_numeric(work.get('launch_speed'), errors='coerce')
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['estimated_ba_using_speedangle'] = pd.to_numeric(work.get('estimated_ba_using_speedangle'), errors='coerce')
    work['estimated_slg_using_speedangle'] = pd.to_numeric(work.get('estimated_slg_using_speedangle'), errors='coerce')
    grouped = work.groupby(['batter_name', 'batter_id'], dropna=False)
    out = pd.DataFrame({'batter': [idx[0] for idx in grouped.size().index], 'batter_id': [idx[1] for idx in grouped.size().index], 'pa': grouped.size().values})
    out['xwoba'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['xba'] = grouped['estimated_ba_using_speedangle'].mean().values
    out['xslg'] = grouped['estimated_slg_using_speedangle'].mean().values
    out['avg_ev'] = grouped['launch_speed'].mean().values
    out['hardhit_rate'] = grouped['launch_speed'].apply(lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') >= 95))).values
    for column in ['xwoba', 'xba', 'xslg', 'avg_ev', 'hardhit_rate']:
        out[column] = out[column].fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['batter_score'] = (0.40 * zscore(out['xwoba']) + 0.20 * zscore(out['xslg']) + 0.10 * zscore(out['xba']) + 0.15 * zscore(out['avg_ev']) + 0.15 * zscore(out['hardhit_rate'])).fillna(0.0)
    return out

def _single_regular_season_batters(client, season_year):
    return aggregate_batter_quality(savant_statcast_csv(client, 'batter', season_year, dt.date(season_year, 3, 1), dt.date(season_year, 11, 30), 'R|'))

def _combine_multiyear_batter_priors(season_frames):
    columns = ['batter', 'batter_id', 'pa', 'xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'batter_score', 'years_covered', 'prior_window']
    frames = []
    for season_year, season_weight, season_df in season_frames:
        if season_df is None or season_df.empty:
            continue
        work = season_df.copy()
        work['season_year'] = int(season_year)
        work['season_weight'] = float(season_weight)
        work['pa'] = pd.to_numeric(work.get('pa'), errors='coerce').fillna(0.0)
        work['weighted_volume'] = work['season_weight'] * work['pa'].clip(lower=1.0)
        work['player_key'] = work.apply(
            lambda row: f"id:{int(row['batter_id'])}" if safe_float(row.get('batter_id')) is not None else f"name:{canonical_player_name(row.get('batter'))}",
            axis=1,
        )
        frames.append(work)
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby('player_key', dropna=False)
    rows = []
    metric_cols = ['xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'batter_score']
    for _, group in grouped:
        first = group.iloc[0]
        volume = pd.to_numeric(group.get('weighted_volume'), errors='coerce').fillna(0.0)
        total_volume = float(volume.sum())
        if total_volume <= 0:
            total_volume = float(len(group))
            volume = pd.Series([1.0] * len(group), index=group.index, dtype=float)
        row = {
            'batter': first.get('batter'),
            'batter_id': first.get('batter_id'),
            'pa': float(pd.to_numeric(group.get('pa'), errors='coerce').fillna(0.0).sum()),
            'years_covered': int(pd.to_numeric(group.get('season_year'), errors='coerce').dropna().astype(int).nunique()),
            'prior_window': f"{int(pd.to_numeric(group.get('season_year'), errors='coerce').dropna().min())}-{int(pd.to_numeric(group.get('season_year'), errors='coerce').dropna().max())}",
        }
        for metric in metric_cols:
            values = pd.to_numeric(group.get(metric), errors='coerce')
            valid = values.notna()
            if valid.any():
                row[metric] = float(np.average(values[valid], weights=volume[valid]))
            else:
                row[metric] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for column in ['xwoba', 'xslg', 'xba', 'avg_ev', 'hardhit_rate', 'batter_score']:
        out[column] = pd.to_numeric(out.get(column), errors='coerce').fillna(out[column].mean() if out[column].notna().any() else 0.0)
    return out[columns]

def previous_regular_season_batters(client, season_year):
    cache_path = season_cache_path('batter_priors_5y', season_year)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    season_frames = []
    for year_value, year_weight in weighted_regular_season_years(season_year, lookback=5):
        season_frames.append((year_value, year_weight, _single_regular_season_batters(client, year_value)))
    out = _combine_multiyear_batter_priors(season_frames)
    out.to_csv(cache_path, index=False)
    return out

def batter_lookup_map(df):
    by_name, by_id = {}, {}
    if df is None or df.empty:
        return by_name, by_id
    for _, row in df.iterrows():
        record = row.to_dict()
        name = canonical_player_name(record.get('batter'))
        batter_id = safe_float(record.get('batter_id'))
        if name:
            by_name[name] = record
        if batter_id is not None:
            by_id[int(batter_id)] = record
    return by_name, by_id

def _resolve_batter_projection_from_maps(batter_name, batter_id, current_by_name, current_by_id, prior_by_name, prior_by_id):
    batter_key = int(batter_id) if batter_id is not None and safe_float(batter_id) is not None else None
    name_key = canonical_player_name(batter_name)
    current = current_by_id.get(batter_key) if batter_key is not None and batter_key in current_by_id else current_by_name.get(name_key)
    prior = prior_by_id.get(batter_key) if batter_key is not None and batter_key in prior_by_id else prior_by_name.get(name_key)
    if current is not None:
        current_score = finite(current.get('batter_score'), 0.0)
        pa = finite(current.get('pa'), 0.0)
        if prior is not None:
            current_weight = pa / (pa + BATTER_PRIOR_REGRESSION_PA) if (pa + BATTER_PRIOR_REGRESSION_PA) > 0 else 0.0
            return {
                'score': (current_weight * current_score) + ((1.0 - current_weight) * finite(prior.get('batter_score'), 0.0)),
                'source': 'current + prior blend',
                'confidence': 0.45 + (0.55 * current_weight),
            }
        return {'score': current_score, 'source': 'current season', 'confidence': 0.50 if pa < BATTER_PRIOR_REGRESSION_PA else 0.92}
    if prior is not None:
        years_covered = max(1.0, finite(prior.get('years_covered'), 1.0))
        prior_conf = float(np.clip(0.38 + (0.04 * min(years_covered, 5.0)), 0.38, 0.58))
        return {'score': finite(prior.get('batter_score'), 0.0), 'source': f"{int(years_covered)}-year regular-season prior", 'confidence': prior_conf}
    return {'score': 0.0, 'source': 'neutral prior', 'confidence': 0.20}

def resolve_batter_projection(batter_name, batter_id, current_df, prior_df):
    current_by_name, current_by_id = batter_lookup_map(current_df)
    prior_by_name, prior_by_id = batter_lookup_map(prior_df)
    return _resolve_batter_projection_from_maps(batter_name, batter_id, current_by_name, current_by_id, prior_by_name, prior_by_id)


STATSAPI_TRANSACTIONS_URL = "https://statsapi.mlb.com/api/v1/transactions"

# IL/DL status codes from the MLB Stats API transaction wire
_IL_TRANSACTION_TYPES = frozenset([
    'placed_on_il', 'transferred_to_il', 'placed_on_restricted_list',
    'placed_on_bereavement_list', 'placed_on_paternity_list',
    'placed_on_emergency_disabled_list', 'suspended',
])
_ACTIVATED_TRANSACTION_TYPES = frozenset([
    'activated_from_il', 'recalled_from_minors', 'selected_from_minors',
    'reinstated_from_il', 'reinstated_from_restricted_list',
])


def fetch_team_injury_transactions(client, team_id, target_date, lookback_days=7):
    """Fetch recent IL transactions for a team from the MLB Stats API transaction wire.

    Returns a dict with:
      - newly_placed: list of player names newly placed on IL in the last `lookback_days`
      - newly_activated: list of player names recently activated from IL
      - net_il_impact: float score — negative means more absences (hurts offense)
    """
    result = {'newly_placed': [], 'newly_activated': [], 'net_il_impact': 0.0, 'source': 'unavailable'}
    if team_id is None:
        return result
    start_date = (target_date - dt.timedelta(days=lookback_days)).isoformat()
    end_date = target_date.isoformat()
    cache_key = f"tx:{int(team_id)}:{start_date}:{end_date}"
    if hasattr(client, '_transaction_cache') and cache_key in client._transaction_cache:
        return dict(client._transaction_cache[cache_key])
    if not hasattr(client, '_transaction_cache'):
        client._transaction_cache = {}
    try:
        payload = client.get_json(
            STATSAPI_TRANSACTIONS_URL,
            params={'teamId': int(team_id), 'startDate': start_date, 'endDate': end_date},
            timeout=20,
        )
        transactions = payload.get('transactions') or []
    except Exception:
        return result

    newly_placed = []
    newly_activated = []
    for tx in transactions:
        tx_type = str(tx.get('typeCode') or tx.get('type') or '').lower().replace(' ', '_')
        person = tx.get('person') or {}
        name = canonical_player_name(person.get('fullName') or person.get('name'))
        if not name:
            continue
        if tx_type in _IL_TRANSACTION_TYPES:
            newly_placed.append(name)
        elif tx_type in _ACTIVATED_TRANSACTION_TYPES:
            newly_activated.append(name)

    # Net impact: each newly-placed hitter represents a lineup quality dip
    # Each newly-activated player represents a boost (returning star)
    net_il_impact = float(np.clip(
        (len(newly_activated) * 0.04) - (len(newly_placed) * 0.06),
        -0.30, 0.20,
    ))
    result = {
        'newly_placed': newly_placed,
        'newly_activated': newly_activated,
        'net_il_impact': net_il_impact,
        'source': 'statsapi_transactions',
    }
    client._transaction_cache[cache_key] = dict(result)
    return result


def fetch_team_roster_snapshot(client, team_id, target_date):
    if team_id is None or safe_float(team_id) is None:
        return []
    roster_key = f"{int(team_id)}:{target_date.isoformat()}"
    if roster_key in client.roster_cache:
        return list(client.roster_cache[roster_key])
    parsed = []
    for params in (
        {'rosterType': 'active', 'date': target_date.isoformat(), 'hydrate': 'person'},
        {'rosterType': 'active', 'hydrate': 'person'},
    ):
        try:
            payload = client.get_json(STATSAPI_TEAM_ROSTER_URL.format(team_id=int(team_id)), params=params, timeout=30)
        except Exception:
            continue
        roster_entries = payload.get('roster') or []
        if not roster_entries:
            continue
        for entry in roster_entries:
            person = entry.get('person') or {}
            position = entry.get('position') or {}
            position_abbr = str(position.get('abbreviation') or '').upper()
            position_type = str(position.get('type') or '').lower()
            if position_abbr == 'P' or 'pitcher' in position_type:
                continue
            parsed.append({
                'player_id': int(person.get('id')) if safe_float(person.get('id')) is not None else None,
                'name': canonical_player_name(person.get('fullName')),
                'position': position_abbr or str(position.get('name') or ''),
                'status': str((entry.get('status') or {}).get('description') or ''),
            })
        if parsed:
            break
    client.roster_cache[roster_key] = list(parsed)
    return list(parsed)


def roster_offense_context(team_id, team_name, target_date, current_batters, prior_batters, client):
    roster = fetch_team_roster_snapshot(client, team_id, target_date)
    neutral = {
        'team': team_name,
        'roster_offense_score': 0.0,
        'roster_top9_score': 0.0,
        'roster_depth_score': 0.0,
        'roster_avg_confidence': 0.0,
        'roster_hitter_count': 0,
        'roster_prior_heavy': 0,
        'roster_source': 'no active hitter roster',
    }
    if not roster:
        return neutral
    current_by_name, current_by_id = batter_lookup_map(current_batters)
    prior_by_name, prior_by_id = batter_lookup_map(prior_batters)
    projected_hitters = []
    prior_heavy = 0
    for hitter in roster:
        projection = _resolve_batter_projection_from_maps(
            hitter.get('name'),
            hitter.get('player_id'),
            current_by_name,
            current_by_id,
            prior_by_name,
            prior_by_id,
        )
        if 'prior' in str(projection.get('source', '')).lower() and 'current season' not in str(projection.get('source', '')).lower():
            prior_heavy += 1
        projected_hitters.append({
            'score': finite(projection.get('score'), 0.0),
            'confidence': finite(projection.get('confidence'), 0.0),
            'source': str(projection.get('source') or ''),
        })
    if not projected_hitters:
        return neutral
    projected_hitters.sort(key=lambda item: item.get('score', 0.0), reverse=True)
    top_n = projected_hitters[:9]
    depth = projected_hitters[9:13]
    weights = LINEUP_SLOT_WEIGHTS[:len(top_n)] if top_n else [1.0]
    top_scores = [finite(item.get('score'), 0.0) for item in top_n]
    top_conf = [finite(item.get('confidence'), 0.0) for item in top_n]
    depth_scores = [finite(item.get('score'), 0.0) for item in depth]
    top9_score = float(np.average(top_scores, weights=weights)) if top_scores else 0.0
    depth_score = float(np.mean(depth_scores)) if depth_scores else top9_score
    roster_score = float((0.82 * top9_score) + (0.18 * depth_score))
    avg_conf = float(np.average(top_conf, weights=weights)) if top_conf else 0.0
    source = 'active roster prior blend'
    if avg_conf >= 0.75:
        source = 'active roster current-season leaning'
    elif prior_heavy >= max(4, int(math.ceil(len(top_n) * 0.45))):
        source = 'active roster prior-heavy'
    return {
        'team': team_name,
        'roster_offense_score': roster_score,
        'roster_top9_score': top9_score,
        'roster_depth_score': depth_score,
        'roster_avg_confidence': avg_conf,
        'roster_hitter_count': int(len(projected_hitters)),
        'roster_prior_heavy': int(prior_heavy),
        'roster_source': source,
    }


def attach_roster_offense_priors(strengths, games, target_date, current_batters, prior_batters, client, phase):
    if strengths is None or strengths.empty:
        return strengths
    phase_name = str(phase or 'spring').lower()
    team_pairs = {}
    for game in games or []:
        away_team = str(game.get('away_team') or '')
        home_team = str(game.get('home_team') or '')
        away_id = game.get('away_team_id')
        home_id = game.get('home_team_id')
        if away_team and safe_float(away_id) is not None:
            team_pairs[away_team] = int(float(away_id))
        if home_team and safe_float(home_id) is not None:
            team_pairs[home_team] = int(float(home_id))
    if not team_pairs:
        work = strengths.copy()
        for column, default_value in {
            'roster_offense_score': 0.0,
            'roster_top9_score': 0.0,
            'roster_depth_score': 0.0,
            'roster_avg_confidence': 0.0,
            'roster_hitter_count': 0,
            'roster_prior_heavy': 0,
            'roster_source': 'no scheduled roster context',
            'roster_offense_weight': 0.0,
            'roster_offense_signal': 0.0,
            'offense_score_with_roster': work.get('offense_score_blended', pd.Series(dtype=float)),
            'offense_score_with_roster_vs_lhp': work.get('offense_score_blended_vs_lhp', work.get('offense_score_blended', pd.Series(dtype=float))),
            'offense_score_with_roster_vs_rhp': work.get('offense_score_blended_vs_rhp', work.get('offense_score_blended', pd.Series(dtype=float))),
        }.items():
            if column not in work.columns:
                work[column] = default_value
        return work
    rows = [
        roster_offense_context(team_id, team_name, target_date, current_batters, prior_batters, client)
        for team_name, team_id in sorted(team_pairs.items())
    ]
    roster_df = pd.DataFrame(rows)
    work = strengths.merge(roster_df, on='team', how='left')
    for column in ['roster_offense_score', 'roster_top9_score', 'roster_depth_score', 'roster_avg_confidence']:
        work[column] = pd.to_numeric(work.get(column), errors='coerce').fillna(0.0)
    for column in ['roster_hitter_count', 'roster_prior_heavy']:
        work[column] = pd.to_numeric(work.get(column), errors='coerce').fillna(0).astype(int)
    work['roster_source'] = work.get('roster_source').fillna('no roster prior')
    max_weight = 0.22 if phase_name == 'spring' else 0.18
    prior_weight = pd.to_numeric(work.get('prior_weight'), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    roster_conf = work['roster_avg_confidence'].clip(0.0, 1.0)
    hitter_depth = (work['roster_hitter_count'] / 9.0).clip(0.0, 1.0)
    confidence_drag = 1.0 - (0.25 * (work['roster_prior_heavy'] / work['roster_hitter_count'].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0))
    work['roster_offense_weight'] = (max_weight * prior_weight * (0.35 + (0.65 * roster_conf)) * hitter_depth * confidence_drag).clip(0.0, max_weight)
    base_offense = pd.to_numeric(work.get('offense_score_blended'), errors='coerce').fillna(0.0)
    work['offense_score_with_roster'] = ((1.0 - work['roster_offense_weight']) * base_offense) + (work['roster_offense_weight'] * work['roster_offense_score'])
    work['roster_offense_signal'] = work['offense_score_with_roster'] - base_offense
    for suffix in ['vs_lhp', 'vs_rhp']:
        split_base = pd.to_numeric(work.get(f'offense_score_blended_{suffix}'), errors='coerce')
        if split_base.notna().any():
            work[f'offense_score_with_roster_{suffix}'] = split_base.fillna(base_offense) + work['roster_offense_signal']
        else:
            work[f'offense_score_with_roster_{suffix}'] = work['offense_score_with_roster']
    return work


def build_team_strengths(client, target_date, phase, games, current_batter_raw, batter_raw_30, batter_raw_14, batter_raw_7, current_batters, prior_batters):
    season_offense = aggregate_team_offense(current_batter_raw)
    strengths = blend_team_offense(
        season_offense,
        aggregate_team_offense(batter_raw_30),
        aggregate_team_offense(batter_raw_14),
        aggregate_team_offense(batter_raw_7),
        previous_regular_season_team_offense(client, target_date.year - 1),
    )
    strengths = merge_handed_offense_splits(
        strengths,
        blend_team_offense(
            aggregate_team_offense(current_batter_raw, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_30, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_14, pitcher_hand='L'),
            aggregate_team_offense(batter_raw_7, pitcher_hand='L'),
            previous_regular_season_team_offense(client, target_date.year - 1, pitcher_hand='L'),
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
            previous_regular_season_team_offense(client, target_date.year - 1, pitcher_hand='R'),
        ),
        'vs_rhp',
    )
    strengths = attach_roster_offense_priors(strengths, games, target_date, current_batters, prior_batters, client, phase)
    defense_strengths = blend_team_defense(
        aggregate_team_defense(current_batter_raw),
        aggregate_team_defense(batter_raw_30),
        aggregate_team_defense(batter_raw_14),
        aggregate_team_defense(batter_raw_7),
        previous_regular_season_team_defense(client, target_date.year - 1),
    )
    return strengths.merge(defense_strengths, on='team', how='left')

def fetch_lineup_snapshot(client, game_pk, lineup_counts_override=None):
    if lineup_counts_override is not None:
        return {
            'away': [],
            'home': [],
            'counts': {'away': int(lineup_counts_override.get('away', 0) or 0), 'home': int(lineup_counts_override.get('home', 0) or 0)},
        }
    try:
        if game_pk not in client.feed_cache:
            client.feed_cache[game_pk] = client.get_json(STATSAPI_GAME_FEED_URL.format(game_pk=game_pk))
        box = (((client.feed_cache[game_pk].get('liveData') or {}).get('boxscore') or {}).get('teams')) or {}
        snapshot = {'away': [], 'home': [], 'counts': {'away': 0, 'home': 0}}
        for side in ['away', 'home']:
            team_block = box.get(side) or {}
            order = list(team_block.get('battingOrder') or [])[:9]
            players = team_block.get('players') or {}
            parsed = []
            for batter_id in order:
                player = players.get(f'ID{int(batter_id)}') or {}
                person = player.get('person') or {}
                parsed.append({'batter_id': int(batter_id), 'name': canonical_player_name(person.get('fullName'))})
            snapshot[side] = parsed
            snapshot['counts'][side] = len(parsed)
        return snapshot
    except Exception:
        return {'away': [], 'home': [], 'counts': {'away': 0, 'home': 0}}

def lineup_quality_context(game_pk, away_row, home_row, current_batters, prior_batters, client, lineup_counts_override=None):
    snapshot = fetch_lineup_snapshot(client, game_pk, lineup_counts_override=lineup_counts_override)

    def side_context(team_row, players):
        if not players:
            return {
                'count': 0,
                'avg_score': 0.0,
                'weighted_score': 0.0,
                'top_half_score': 0.0,
                'depth_score': 0.0,
                'top_heavy': 0.0,
                'adj': 0.0,
                'avg_confidence': 0.0,
                'prior_heavy': 0,
            }
        projections, confidences, weights_used, prior_heavy = [], [], [], 0
        for idx, player in enumerate(players[:9]):
            slot_weight = LINEUP_SLOT_WEIGHTS[min(idx, len(LINEUP_SLOT_WEIGHTS) - 1)]
            projection = resolve_batter_projection(player.get('name'), player.get('batter_id'), current_batters, prior_batters)
            projections.append(finite(projection.get('score'), 0.0))
            confidences.append(finite(projection.get('confidence'), 0.0))
            weights_used.append(slot_weight)
            if 'prior' in str(projection.get('source', '')).lower() and 'current season' not in str(projection.get('source', '')).lower():
                prior_heavy += 1
        baseline = finite((team_row or {}).get('offense_score_blended'), 0.0)
        mean_score = float(np.mean(projections)) if projections else 0.0
        weighted_score = float(np.average(projections, weights=weights_used)) if projections else 0.0
        avg_conf = float(np.average(confidences, weights=weights_used)) if confidences else 0.0
        top_cut = min(4, len(projections))
        top_half_score = float(np.average(projections[:top_cut], weights=weights_used[:top_cut])) if top_cut > 0 else weighted_score
        depth_weights = weights_used[top_cut:] if len(weights_used) > top_cut else weights_used
        depth_scores = projections[top_cut:] if len(projections) > top_cut else projections
        depth_score = float(np.average(depth_scores, weights=depth_weights)) if depth_scores else weighted_score
        top_heavy = top_half_score - depth_score
        confirmed_weight = min(len(players), 9) / 9.0
        adj_core = (0.60 * (weighted_score - baseline)) + (0.25 * (top_half_score - baseline)) + (0.15 * (depth_score - baseline))
        adj = adj_core * confirmed_weight * (0.35 + (0.65 * avg_conf))
        return {
            'count': len(players),
            'avg_score': mean_score,
            'weighted_score': weighted_score,
            'top_half_score': top_half_score,
            'depth_score': depth_score,
            'top_heavy': top_heavy,
            'adj': float(np.clip(adj, -0.75, 0.75)),
            'avg_confidence': avg_conf,
            'prior_heavy': int(prior_heavy),
        }

    away = side_context(away_row, snapshot.get('away') or [])
    home = side_context(home_row, snapshot.get('home') or [])
    return {
        'away': away,
        'home': home,
        'counts': snapshot.get('counts') or {'away': away.get('count', 0), 'home': home.get('count', 0)},
        'delta': float(home.get('adj', 0.0) - away.get('adj', 0.0)),
        'order_delta': float((home.get('weighted_score', 0.0) - away.get('weighted_score', 0.0)) - (home.get('avg_score', 0.0) - away.get('avg_score', 0.0))),
        'top_heavy_delta': float(home.get('top_heavy', 0.0) - away.get('top_heavy', 0.0)),
    }
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
    columns = ['pitcher', 'pitcher_id', 'p_throws', 'bf', 'games', 'avg_bf_per_game', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate', 'pitcher_score']
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    work['pitcher_name'] = work.get('player_name', '').apply(canonical_player_name)
    work['pitcher_id'] = pd.to_numeric(work.get('pitcher'), errors='coerce')
    work['p_throws'] = work.get('p_throws', pd.Series([None] * len(work), index=work.index)).apply(normalize_handedness)
    work['launch_speed'] = pd.to_numeric(work.get('launch_speed'), errors='coerce')
    work['estimated_woba_using_speedangle'] = pd.to_numeric(work.get('estimated_woba_using_speedangle'), errors='coerce')
    work['estimated_slg_using_speedangle'] = pd.to_numeric(work.get('estimated_slg_using_speedangle'), errors='coerce')
    work['woba_value'] = pd.to_numeric(work.get('woba_value'), errors='coerce')
    work['game_pk'] = pd.to_numeric(work.get('game_pk'), errors='coerce')
    work['event_text'] = (work.get('events', '').astype(str).str.lower() + ' ' + work.get('description', '').astype(str).str.lower())
    grouped = work.groupby(['pitcher_name', 'pitcher_id'], dropna=False)
    out = pd.DataFrame({'pitcher': [idx[0] for idx in grouped.size().index], 'pitcher_id': [idx[1] for idx in grouped.size().index], 'bf': grouped.size().values})
    out['p_throws'] = grouped['p_throws'].agg(lambda s: next((hand for hand in s.dropna().tolist() if hand in {'L', 'R'}), None)).values
    out['games'] = grouped['game_pk'].agg(lambda s: int(pd.to_numeric(s, errors='coerce').dropna().astype(int).nunique()) if pd.to_numeric(s, errors='coerce').dropna().any() else max(1, int(round(len(s) / 18.0)))).values
    out['avg_bf_per_game'] = out['bf'] / out['games'].replace(0, np.nan)
    out['xwoba_allowed'] = grouped['estimated_woba_using_speedangle'].mean().values
    out['woba_allowed'] = grouped['woba_value'].mean().values
    out['xslg_allowed'] = grouped['estimated_slg_using_speedangle'].mean().values
    out['avg_ev_allowed'] = grouped['launch_speed'].mean().values
    out['hardhit_allowed_rate'] = grouped['launch_speed'].apply(lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') >= 95))).values
    out['k_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: 'strikeout' in str(text))))).values
    out['bb_rate'] = grouped['event_text'].apply(lambda s: float(np.mean(s.apply(lambda text: ('walk' in str(text)) or ('hit_by_pitch' in str(text)))))).values
    out['xwoba_allowed'] = out['xwoba_allowed'].fillna(out['woba_allowed'])
    for column in ['avg_bf_per_game', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate']:
        out[column] = out[column].fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['avg_bf_per_game'] = out['avg_bf_per_game'].clip(lower=8.0, upper=27.0)
    out['pitcher_score'] = (-0.40 * zscore(out['xwoba_allowed']) - 0.15 * zscore(out['xslg_allowed']) - 0.10 * zscore(out['avg_ev_allowed']) - 0.10 * zscore(out['hardhit_allowed_rate']) + 0.20 * zscore(out['k_rate']) - 0.05 * zscore(out['bb_rate'])).fillna(0.0)
    return out[columns]

def _single_regular_season_pitchers(client, season_year):
    return aggregate_pitcher_quality(savant_statcast_csv(client, 'pitcher', season_year, dt.date(season_year, 3, 1), dt.date(season_year, 11, 30), 'R|'))

def _combine_multiyear_pitcher_priors(season_frames):
    columns = ['pitcher', 'pitcher_id', 'p_throws', 'bf', 'games', 'avg_bf_per_game', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate', 'pitcher_score', 'years_covered', 'prior_window']
    frames = []
    for season_year, season_weight, season_df in season_frames:
        if season_df is None or season_df.empty:
            continue
        work = season_df.copy()
        work['season_year'] = int(season_year)
        work['season_weight'] = float(season_weight)
        work['bf'] = pd.to_numeric(work.get('bf'), errors='coerce').fillna(0.0)
        work['games'] = pd.to_numeric(work.get('games'), errors='coerce').fillna(0.0)
        work['weighted_volume'] = work['season_weight'] * work['bf'].clip(lower=1.0)
        work['player_key'] = work.apply(
            lambda row: f"id:{int(row['pitcher_id'])}" if safe_float(row.get('pitcher_id')) is not None else f"name:{canonical_player_name(row.get('pitcher'))}",
            axis=1,
        )
        frames.append(work)
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby('player_key', dropna=False)
    rows = []
    metric_cols = ['avg_bf_per_game', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate', 'pitcher_score']
    for _, group in grouped:
        first = group.iloc[0]
        volume = pd.to_numeric(group.get('weighted_volume'), errors='coerce').fillna(0.0)
        total_volume = float(volume.sum())
        if total_volume <= 0:
            total_volume = float(len(group))
            volume = pd.Series([1.0] * len(group), index=group.index, dtype=float)
        throws = pd.Series(group.get('p_throws')).dropna().astype(str).tolist()
        season_years = pd.to_numeric(group.get('season_year'), errors='coerce').dropna().astype(int)
        row = {
            'pitcher': first.get('pitcher'),
            'pitcher_id': first.get('pitcher_id'),
            'p_throws': next((hand for hand in throws if hand in {'L', 'R'}), None),
            'bf': float(pd.to_numeric(group.get('bf'), errors='coerce').fillna(0.0).sum()),
            'games': float(pd.to_numeric(group.get('games'), errors='coerce').fillna(0.0).sum()),
            'years_covered': int(season_years.nunique()) if not season_years.empty else 0,
            'prior_window': f"{int(season_years.min())}-{int(season_years.max())}" if not season_years.empty else 'N/A',
        }
        for metric in metric_cols:
            values = pd.to_numeric(group.get(metric), errors='coerce')
            valid = values.notna()
            if valid.any():
                row[metric] = float(np.average(values[valid], weights=volume[valid]))
            else:
                row[metric] = 0.0
        if not math.isfinite(finite(row.get('avg_bf_per_game'), np.nan)):
            bf = finite(row.get('bf'), 0.0)
            games = max(1.0, finite(row.get('games'), 1.0))
            row['avg_bf_per_game'] = bf / games
        rows.append(row)
    out = pd.DataFrame(rows)
    for column in ['avg_bf_per_game', 'xwoba_allowed', 'xslg_allowed', 'avg_ev_allowed', 'hardhit_allowed_rate', 'k_rate', 'bb_rate', 'pitcher_score']:
        out[column] = pd.to_numeric(out.get(column), errors='coerce').fillna(out[column].mean() if out[column].notna().any() else 0.0)
    out['avg_bf_per_game'] = out['avg_bf_per_game'].clip(lower=8.0, upper=27.0)
    return out[columns]

def previous_regular_season_pitchers(client, season_year):
    cache_path = season_cache_path('pitcher_priors_5y', season_year)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    season_frames = []
    for year_value, year_weight in weighted_regular_season_years(season_year, lookback=5):
        season_frames.append((year_value, year_weight, _single_regular_season_pitchers(client, year_value)))
    out = _combine_multiyear_pitcher_priors(season_frames)
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
        games = max(1.0, finite(current.get('games'), 0.0) or max(1.0, round(bf / 18.0)))
        avg_bf = finite(current.get('avg_bf_per_game'), bf / games)
        current_hand = normalize_handedness(current.get('p_throws'))
        prior_hand = normalize_handedness((prior or {}).get('p_throws'))
        current_k = finite(current.get('k_rate'), 0.0)
        current_bb = finite(current.get('bb_rate'), 0.0)
        if prior is not None:
            current_weight = bf / (bf + STARTER_PRIOR_REGRESSION_BF) if (bf + STARTER_PRIOR_REGRESSION_BF) > 0 else 0.0
            prior_games = max(1.0, finite(prior.get('games'), 0.0) or max(1.0, round(finite(prior.get('bf'), 0.0) / 18.0)))
            prior_avg_bf = finite(prior.get('avg_bf_per_game'), finite(prior.get('bf'), 0.0) / prior_games)
            return {
                'score': (current_weight * current_score) + ((1.0 - current_weight) * finite(prior.get('pitcher_score'), 0.0)),
                'source': 'current + prior blend',
                'confidence': 0.55 + (0.45 * current_weight),
                'hand': current_hand or prior_hand,
                'bf': bf,
                'games': games,
                'avg_bf_per_game': (current_weight * avg_bf) + ((1.0 - current_weight) * prior_avg_bf),
                'k_rate': (current_weight * current_k) + ((1.0 - current_weight) * finite(prior.get('k_rate'), 0.0)),
                'bb_rate': (current_weight * current_bb) + ((1.0 - current_weight) * finite(prior.get('bb_rate'), 0.0)),
            }
        return {
            'score': current_score,
            'source': 'current season',
            'confidence': 0.60 if bf < STARTER_PRIOR_REGRESSION_BF else 0.95,
            'hand': current_hand,
            'bf': bf,
            'games': games,
            'avg_bf_per_game': avg_bf,
            'k_rate': current_k,
            'bb_rate': current_bb,
        }
    if prior is not None:
        bf = finite(prior.get('bf'), 0.0)
        games = max(1.0, finite(prior.get('games'), 0.0) or max(1.0, round(bf / 18.0)))
        years_covered = max(1.0, finite(prior.get('years_covered'), 1.0))
        return {
            'score': finite(prior.get('pitcher_score'), 0.0),
            'source': f"{int(years_covered)}-year regular-season prior",
            'confidence': float(np.clip(0.48 + (0.04 * min(years_covered, 5.0)), 0.48, 0.68)),
            'hand': normalize_handedness(prior.get('p_throws')),
            'bf': bf,
            'games': games,
            'avg_bf_per_game': finite(prior.get('avg_bf_per_game'), bf / games),
            'k_rate': finite(prior.get('k_rate'), 0.0),
            'bb_rate': finite(prior.get('bb_rate'), 0.0),
        }
    return {'score': 0.0, 'source': 'neutral prior', 'confidence': 0.25, 'hand': None, 'bf': 0.0, 'games': 0.0, 'avg_bf_per_game': np.nan, 'k_rate': 0.0, 'bb_rate': 0.0}


def starter_workload_context(projection, phase):
    phase_name = str(phase or 'spring').lower()
    base_ip = STARTER_BASE_EXPECTED_IP.get(phase_name, 5.0)
    avg_bf = finite(projection.get('avg_bf_per_game'), base_ip * 4.25)
    confidence = finite(projection.get('confidence'), 0.5)
    score = finite(projection.get('score'), 0.0)
    k_rate = finite(projection.get('k_rate'), 0.0)
    bb_rate = finite(projection.get('bb_rate'), 0.0)
    source = str(projection.get('source') or '').lower()

    usage_ip = avg_bf / 4.25
    skill_adj = (0.35 * np.clip(score, -1.25, 1.25)) + (0.60 * np.clip(k_rate - bb_rate, -0.08, 0.12))
    source_drag = 0.0
    if 'neutral' in source:
        source_drag += 0.90 if phase_name == 'spring' else 0.55
    elif 'prior' in source and 'current season' not in source:
        source_drag += 0.45 if phase_name == 'spring' else 0.28
    elif 'blend' in source:
        source_drag += 0.10

    expected_ip = ((1.0 - confidence) * base_ip) + (confidence * usage_ip) + skill_adj - source_drag
    if phase_name == 'spring':
        expected_ip = float(np.clip(expected_ip, 3.2, 5.8))
    else:
        expected_ip = float(np.clip(expected_ip, 4.0, 7.0))
    starter_share = float(np.clip(expected_ip / 9.0, 0.32, 0.78))
    short_start_risk = float(np.clip(((base_ip - expected_ip) / 1.9) + (0.55 * max(0.0, 0.65 - confidence)) + (0.12 if 'neutral' in source else 0.0), 0.0, 1.0))
    times_through_order = float(np.clip(avg_bf / 9.0, 1.0, 3.4))
    tto_trigger = 1.80 if phase_name == 'spring' else 1.95
    weak_starter_factor = max(0.0, 0.18 - score)
    tto_risk = float(np.clip(
        max(0.0, times_through_order - tto_trigger)
        * (0.70 + (1.25 * weak_starter_factor) + (1.10 * max(0.0, bb_rate - k_rate)))
        + (0.10 * max(0.0, 0.58 - confidence)),
        0.0,
        1.0,
    ))
    return {
        'expected_ip': expected_ip,
        'starter_share': starter_share,
        'bullpen_share': float(np.clip(1.0 - starter_share, 0.22, 0.68)),
        'short_start_risk': short_start_risk,
        'times_through_order': times_through_order,
        'tto_risk': tto_risk,
        'usage_bf_per_game': avg_bf,
    }
def matchup_adjusted_offense(team_row, pitcher_hand):
    base_score = finite((team_row or {}).get('offense_score_with_roster'), finite((team_row or {}).get('offense_score_blended'), 0.0))
    hand_key = normalize_handedness(pitcher_hand)
    if hand_key is None:
        return {'score': base_score, 'adjustment': 0.0, 'split_score': base_score, 'split_weight': 0.0, 'split_label': 'vs unknown'}
    suffix = 'vs_lhp' if hand_key == 'L' else 'vs_rhp'
    split_score = safe_float((team_row or {}).get(f'offense_score_with_roster_{suffix}'))
    if split_score is None:
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
    columns = [
        'team', 'bullpen_quality', 'availability_score', 'stress_score', 'recent_workload', 'usage_last2', 'usage_last1',
        'active_days', 'leverage_usage', 'leverage_availability', 'bullpen_score',
    ]
    start_date = report_date - dt.timedelta(days=lookback_days)
    end_date = report_date - dt.timedelta(days=1)
    if end_date < start_date:
        return pd.DataFrame(columns=columns)
    current_by_name, current_by_id = pitcher_lookup_map(current_pitchers)
    prior_by_name, prior_by_id = pitcher_lookup_map(prior_pitchers)
    rows = []
    for game in completed_games_in_range(client, start_date, end_date):
        game_pk = int(game['gamePk'])
        if game_pk not in client.feed_cache:
            client.feed_cache[game_pk] = client.get_json(STATSAPI_GAME_FEED_URL.format(game_pk=game_pk))
        feed = client.feed_cache[game_pk]
        box = (((feed.get('liveData') or {}).get('boxscore') or {}).get('teams')) or {}
        game_date = pd.to_datetime(str(game.get('gameDate') or ''), errors='coerce')
        game_date = game_date.date() if not pd.isna(game_date) else end_date
        days_ago = max(1, (report_date - game_date).days)
        recency_weight = finite(BULLPEN_RECENCY_WEIGHTS.get(days_ago), 0.05)
        for side in ['away', 'home']:
            team_block = box.get(side) or {}
            team_info = ((game.get('teams') or {}).get(side) or {}).get('team') or {}
            team_abbrev = str(team_info.get('abbreviation') or team_info.get('teamName') or 'UNK')
            bullpen_ids = team_block.get('bullpen') or []
            if not bullpen_ids:
                pitchers = team_block.get('pitchers') or []
                bullpen_ids = pitchers[1:] if len(pitchers) > 1 else []
            players = team_block.get('players') or {}
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
                quality = finite(record.get('pitcher_score'), 0.0)
                rows.append({
                    'team': team_abbrev,
                    'quality': quality,
                    'workload': workload,
                    'days_ago': days_ago,
                    'recency_load': workload * recency_weight,
                    'usage_last2': workload if days_ago <= 2 else 0.0,
                    'usage_last1': workload if days_ago <= 1 else 0.0,
                    'leverage_load': workload * recency_weight * max(0.25, 0.70 + max(0.0, quality)),
                    'game_date': game_date,
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    detail = pd.DataFrame(rows)
    grouped = detail.groupby('team', as_index=False)
    out = grouped.agg(
        bullpen_quality=('quality', lambda s: float(np.average(s, weights=np.clip(detail.loc[s.index, 'workload'], 1.0, None)))),
        recent_workload=('recency_load', 'sum'),
        usage_last2=('usage_last2', 'sum'),
        usage_last1=('usage_last1', 'sum'),
        active_days=('game_date', lambda s: int(pd.Series(s).dropna().nunique())),
        leverage_usage=('leverage_load', 'sum'),
    )
    out['stress_score'] = (0.50 * zscore(out['recent_workload']) + 0.20 * zscore(out['usage_last2']) + 0.15 * zscore(out['usage_last1']) + 0.15 * zscore(out['active_days'])).fillna(0.0)
    out['availability_score'] = (-out['stress_score']).fillna(0.0)
    out['leverage_availability'] = (-(0.65 * zscore(out['leverage_usage']) + 0.35 * zscore(out['usage_last1']))).fillna(0.0)
    out['bullpen_score'] = (0.72 * zscore(out['bullpen_quality']) + 0.16 * out['availability_score'] + 0.12 * out['leverage_availability']).fillna(0.0)
    return out[columns]
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
        return {'off_days': 1, 'travel_miles': 0.0, 'score': 0.02, 'previous_game_found': False, 'direction_penalty': 0.0}
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

    # East-west direction penalty: traveling eastward crosses time zones in the harder
    # direction (losing hours, earlier wake-up), amplifying fatigue on same-day or next-day travel.
    # Uses longitude delta: negative delta = traveling east (moving to smaller/more-negative longitude
    # in the US coordinate system, i.e., higher absolute longitude value).
    direction_penalty = 0.0
    prev_lon = safe_float(previous_meta.get('longitude'))
    curr_lon = safe_float(current_meta.get('longitude'))
    if prev_lon is not None and curr_lon is not None and travel_miles > 300:
        # In US, longitudes are negative; moving east means curr_lon > prev_lon (less negative)
        # e.g., LAX (−118) → NYC (−74): delta = −74 − (−118) = +44 → traveling east
        lon_delta = curr_lon - prev_lon
        is_eastward = lon_delta > 0.0
        time_zone_crossings = abs(lon_delta) / 15.0  # ~15° per hour
        if is_eastward and off_days == 0:
            # Same-day eastward travel: harder on the body
            direction_penalty = -float(np.clip(0.020 * time_zone_crossings, 0.0, 0.06))
        elif is_eastward and off_days == 1:
            direction_penalty = -float(np.clip(0.010 * time_zone_crossings, 0.0, 0.04))
        # Westward travel is generally easier (gaining hours); apply a very small boost
        elif not is_eastward and off_days == 0 and travel_miles > 1000:
            direction_penalty = float(np.clip(0.008 * time_zone_crossings, 0.0, 0.02))
    score += direction_penalty

    return {
        'off_days': int(off_days),
        'travel_miles': float(travel_miles),
        'direction_penalty': float(direction_penalty),
        'score': float(np.clip(score, -0.24, 0.10)),
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


def fetch_umpire_for_game(client, game_pk):
    """Fetch the home-plate umpire name for a scheduled game via the MLB boxscore endpoint.
    Returns the umpire name string or None if unavailable."""
    if not game_pk:
        return None
    try:
        url = STATSAPI_GAME_OFFICIALS_URL.format(game_pk=int(game_pk))
        payload = client.get_json(url, timeout=15)
        officials = payload.get('officials') or []
        for official in officials:
            official_type = str((official.get('officialType') or '')).lower()
            if 'home plate' in official_type or 'hp' in official_type or official_type == 'home_plate':
                person = official.get('official') or {}
                name = canonical_player_name(person.get('fullName') or person.get('name'))
                if name:
                    return name
    except Exception:
        pass
    return None


def umpire_total_adjustment(umpire_name, umpire_cache):
    """Return a totals adjustment (runs above/below average) based on umpire calling tendencies.

    The cache stores per-umpire stats with key 'runs_per_game_vs_avg' (positive = more runs
    than league average, negative = fewer runs / tighter zone).  Falls back to 0.0 for unknown
    umpires.  Adjustment is capped at ±0.40 runs to prevent outsized influence.
    """
    if not umpire_name or not umpire_cache:
        return {'umpire_name': umpire_name, 'total_adj': 0.0, 'source': 'unknown'}
    key = str(umpire_name).strip().lower()
    entry = umpire_cache.get(key) or umpire_cache.get(umpire_name)
    if not entry:
        return {'umpire_name': umpire_name, 'total_adj': 0.0, 'source': 'unknown'}
    raw_adj = finite(entry.get('runs_per_game_vs_avg'), 0.0)
    sample_games = finite(entry.get('sample_games'), 0.0)
    # Shrink toward zero for small samples (< 50 games = high uncertainty)
    reliability = float(np.clip(sample_games / 80.0, 0.0, 1.0))
    adjusted = raw_adj * reliability
    return {
        'umpire_name': umpire_name,
        'total_adj': float(np.clip(adjusted, -0.40, 0.40)),
        'raw_adj': raw_adj,
        'sample_games': int(sample_games),
        'reliability': reliability,
        'source': 'cache',
    }


def refresh_umpire_cache_from_savant(client, umpire_cache, season=None):
    """Placeholder: Baseball Savant's Statcast CSV export does not support umpire-level
    aggregation via the group_by=name_umpire parameter — it returns raw pitch rows without
    an umpire_name column.  This function is a no-op until a valid umpire stats source
    is integrated (e.g., a manual CSV upload or a dedicated umpire leaderboard endpoint).

    The cache can be populated manually: add entries keyed by lowercase umpire name with
    the schema: {'umpire_name': str, 'runs_per_game_vs_avg': float, 'sample_games': int}.
    """
    return umpire_cache

    # --- dead code preserved for reference when a real data source is found ---
    target_season = int(season or SEASON)
    try:
        params = {
            'all': 'true',
            'hfSea': f'{target_season}|',
            'player_type': 'pitcher',
            'group_by': 'name_umpire',
            'sort_col': 'pitches',
            'sort_order': 'desc',
            'min_pitches': 100,
            'type': 'details',
        }
        raw = client.get_text(SAVANT_STATCAST_CSV_URL, params=params, timeout=90)
        df = pd.read_csv(StringIO(raw), low_memory=False)
    except Exception:
        return umpire_cache

    needed = {'umpire_name', 'p_called_strike', 'p_ball'}
    if not needed.issubset(set(df.columns)):
        return umpire_cache

    try:
        df['p_called_strike'] = pd.to_numeric(df['p_called_strike'], errors='coerce')
        df['p_ball'] = pd.to_numeric(df['p_ball'], errors='coerce')
        df['pitches'] = pd.to_numeric(df.get('pitches'), errors='coerce').fillna(0)
        league_k_rate = float(df['p_called_strike'].mean()) if df['p_called_strike'].notna().any() else 0.17
        league_bb_rate = float(df['p_ball'].mean()) if df['p_ball'].notna().any() else 0.35
        updated = dict(umpire_cache)
        for _, row in df.iterrows():
            name = canonical_player_name(row.get('umpire_name'))
            if not name:
                continue
            k_rate = finite(row.get('p_called_strike'), league_k_rate)
            bb_rate = finite(row.get('p_ball'), league_bb_rate)
            sample = int(finite(row.get('pitches'), 0) / 30)  # ~30 pitches per game
            # Strikeout rate above avg → fewer baserunners → fewer runs (negative adj)
            # Ball rate above avg → more walks → more runs (positive adj)
            k_delta = k_rate - league_k_rate
            bb_delta = bb_rate - league_bb_rate
            runs_adj = float(np.clip((-4.0 * k_delta) + (3.5 * bb_delta), -0.50, 0.50))
            key = name.strip().lower()
            updated[key] = {
                'umpire_name': name,
                'runs_per_game_vs_avg': runs_adj,
                'k_rate': k_rate,
                'bb_rate': bb_rate,
                'k_rate_vs_avg': k_delta,
                'bb_rate_vs_avg': bb_delta,
                'sample_games': max(sample, int(updated.get(key, {}).get('sample_games', 0))),
                'season': target_season,
            }
        return updated
    except Exception:
        return umpire_cache


def wilson_score_ci(successes, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion.
    Returns (lower, upper) as floats, or (None, None) if n < 1."""
    if n < 1:
        return (None, None)
    p_hat = float(successes) / float(n)
    denom = 1.0 + (z * z / n)
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (float(np.clip(center - margin, 0.0, 1.0)), float(np.clip(center + margin, 0.0, 1.0)))


def bootstrap_mean_ci(values, n_boot=500, z=1.96, rng_seed=42):
    """Bootstrap 95% CI for the mean of an array of values.
    Uses a fixed seed for reproducibility.  Returns (lower, upper) or (None, None)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return (None, None)
    rng = np.random.default_rng(rng_seed)
    boot_means = [float(rng.choice(arr, size=n, replace=True).mean()) for _ in range(n_boot)]
    se = float(np.std(boot_means, ddof=1))
    mean_val = float(arr.mean())
    return (mean_val - z * se, mean_val + z * se)


def build_validation_summary(backtest_df):
    summary = {
        'games': 0, 'oos_games': 0,
        'log_loss': None, 'log_loss_ci': (None, None),
        'accuracy': None, 'accuracy_ci': (None, None),
        'total_mae': None, 'total_mae_ci': (None, None),
        'total_rmse': None, 'total_rmse_ci': (None, None),
        'margin_mae': None, 'margin_mae_ci': (None, None),
        'margin_rmse': None, 'margin_rmse_ci': (None, None),
        'date_span': 'N/A', 'oos_span': 'N/A',
    }
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
        per_game_ll = -(yv * np.log(pv)) - ((1 - yv) * np.log(1.0 - pv))
        summary['log_loss'] = float(per_game_ll.mean())
        summary['log_loss_ci'] = bootstrap_mean_ci(per_game_ll.values)
        correct = (pv >= 0.5).astype(int) == yv
        summary['accuracy'] = float(correct.mean())
        summary['accuracy_ci'] = wilson_score_ci(int(correct.sum()), len(correct))
    actual_total = pd.to_numeric(oos.get('home_score'), errors='coerce') + pd.to_numeric(oos.get('away_score'), errors='coerce')
    predicted_total = pd.to_numeric(oos.get('projected_total'), errors='coerce')
    total_valid = (~actual_total.isna()) & (~predicted_total.isna())
    if total_valid.any():
        total_error = actual_total[total_valid] - predicted_total[total_valid]
        summary['total_mae'] = float(np.mean(np.abs(total_error)))
        summary['total_mae_ci'] = bootstrap_mean_ci(np.abs(total_error).values)
        summary['total_rmse'] = float(np.sqrt(np.mean(np.square(total_error))))
        summary['total_rmse_ci'] = bootstrap_mean_ci(np.square(total_error).values)
    actual_margin = pd.to_numeric(oos.get('home_score'), errors='coerce') - pd.to_numeric(oos.get('away_score'), errors='coerce')
    predicted_margin = pd.to_numeric(oos.get('projected_margin'), errors='coerce')
    margin_valid = (~actual_margin.isna()) & (~predicted_margin.isna())
    if margin_valid.any():
        margin_error = actual_margin[margin_valid] - predicted_margin[margin_valid]
        summary['margin_mae'] = float(np.mean(np.abs(margin_error)))
        summary['margin_mae_ci'] = bootstrap_mean_ci(np.abs(margin_error).values)
        summary['margin_rmse'] = float(np.sqrt(np.mean(np.square(margin_error))))
        summary['margin_rmse_ci'] = bootstrap_mean_ci(np.square(margin_error).values)
    summary['games'], summary['oos_games'] = int(len(work)), int(len(oos))
    summary['date_span'], summary['oos_span'] = (
        f"{unique_dates[0].isoformat()} to {unique_dates[-1].isoformat()}",
        f"{sorted(oos['date'].unique())[0].isoformat()} to {sorted(oos['date'].unique())[-1].isoformat()}",
    )
    return summary


def build_live_validation_summary(archive_df, phase_filter=None):
    summary = {
        'games': 0,
        'settled_days': 0,
        'log_loss': None, 'log_loss_ci': (None, None),
        'accuracy': None, 'accuracy_ci': (None, None),
        'total_mae': None, 'total_mae_ci': (None, None),
        'total_rmse': None, 'total_rmse_ci': (None, None),
        'margin_mae': None, 'margin_mae_ci': (None, None),
        'margin_rmse': None, 'margin_rmse_ci': (None, None),
        'date_span': 'N/A',
        'latest_settled_date': None,
    }
    if archive_df is None or archive_df.empty:
        return summary
    work = archive_df.copy()
    required = {'report_date', 'p_home', 'y_home'}
    if not required.issubset(work.columns):
        return summary
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    if 'phase' in work.columns:
        work['phase'] = work.get('phase').astype(str).str.lower().str.strip()
    else:
        work['phase'] = work['report_date'].apply(lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring')
    if phase_filter:
        work = work[work['phase'] == str(phase_filter).lower().strip()].copy()
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    for column in ['actual_total', 'projected_total', 'actual_margin', 'projected_margin']:
        if column in work.columns:
            work[column] = pd.to_numeric(work.get(column), errors='coerce')
    work = work.dropna(subset=['report_date', 'p_home', 'y_home']).copy()
    if work.empty:
        return summary
    y = work['y_home'].astype(int)
    p = work['p_home'].astype(float)
    summary['games'] = int(len(work))
    summary['settled_days'] = int(work['report_date'].nunique())
    per_game_ll = -(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))
    summary['log_loss'] = float(per_game_ll.mean())
    summary['log_loss_ci'] = bootstrap_mean_ci(per_game_ll.values)
    correct = (p >= 0.5).astype(int) == y
    summary['accuracy'] = float(correct.mean())
    summary['accuracy_ci'] = wilson_score_ci(int(correct.sum()), len(correct))
    if {'actual_total', 'projected_total'}.issubset(work.columns):
        total_valid = work.dropna(subset=['actual_total', 'projected_total']).copy()
        if not total_valid.empty:
            total_error = total_valid['actual_total'].astype(float) - total_valid['projected_total'].astype(float)
            summary['total_mae'] = float(np.mean(np.abs(total_error)))
            summary['total_mae_ci'] = bootstrap_mean_ci(np.abs(total_error).values)
            summary['total_rmse'] = float(np.sqrt(np.mean(np.square(total_error))))
            summary['total_rmse_ci'] = bootstrap_mean_ci(np.square(total_error).values)
    if {'actual_margin', 'projected_margin'}.issubset(work.columns):
        margin_valid = work.dropna(subset=['actual_margin', 'projected_margin']).copy()
        if not margin_valid.empty:
            margin_error = margin_valid['actual_margin'].astype(float) - margin_valid['projected_margin'].astype(float)
            summary['margin_mae'] = float(np.mean(np.abs(margin_error)))
            summary['margin_mae_ci'] = bootstrap_mean_ci(np.abs(margin_error).values)
            summary['margin_rmse'] = float(np.sqrt(np.mean(np.square(margin_error))))
            summary['margin_rmse_ci'] = bootstrap_mean_ci(np.square(margin_error).values)
    date_values = sorted(work['report_date'].unique().tolist())
    if date_values:
        summary['date_span'] = f"{date_values[0].isoformat()} to {date_values[-1].isoformat()}"
        summary['latest_settled_date'] = date_values[-1].isoformat()
    return summary


def build_walkforward_cv_summary(backtest_df, min_train_days=8, test_window_days=3):
    """Walk-forward (expanding-window) time-series cross-validation.

    Splits the backtest log by date into sequential folds.  Each fold trains on all data
    up to date T and evaluates on the next `test_window_days` of data.  Reports mean and
    95% CI for log loss and total RMSE across all folds.

    Returns a dict with:
      - folds: list of per-fold dicts (train_games, test_games, log_loss, total_rmse)
      - mean_log_loss, log_loss_ci
      - mean_total_rmse, total_rmse_ci
      - mean_accuracy, accuracy_ci
      - fold_count
    """
    result = {
        'fold_count': 0,
        'folds': [],
        'mean_log_loss': None, 'log_loss_ci': (None, None),
        'mean_total_rmse': None, 'total_rmse_ci': (None, None),
        'mean_accuracy': None, 'accuracy_ci': (None, None),
        'note': '',
    }
    if backtest_df is None or backtest_df.empty:
        result['note'] = 'No backtest data available.'
        return result
    work = backtest_df.copy()
    work['date'] = pd.to_datetime(work.get('date'), errors='coerce').dt.date
    work = work[work['date'].notna()].copy()
    if work.empty:
        result['note'] = 'No dated rows in backtest log.'
        return result
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    work = work.dropna(subset=['date', 'p_home', 'y_home']).copy()
    unique_dates = sorted(work['date'].unique())
    if len(unique_dates) < min_train_days + test_window_days:
        result['note'] = f'Too few dates ({len(unique_dates)}) for walk-forward CV (need {min_train_days + test_window_days}).'
        return result

    folds = []
    i = min_train_days
    while i < len(unique_dates):
        train_dates = set(unique_dates[:i])
        test_dates = set(unique_dates[i:i + test_window_days])
        train_df = work[work['date'].isin(train_dates)]
        test_df = work[work['date'].isin(test_dates)]
        if len(test_df) < 2:
            i += test_window_days
            continue
        y_test = test_df['y_home'].astype(int)
        p_test = test_df['p_home'].astype(float)
        fold_ll = float((-(y_test * np.log(p_test)) - ((1 - y_test) * np.log(1.0 - p_test))).mean())
        fold_acc = float(((p_test >= 0.5).astype(int) == y_test).mean())
        fold_rmse = None
        if {'projected_total', 'home_score', 'away_score'}.issubset(test_df.columns):
            actual_total = pd.to_numeric(test_df['home_score'], errors='coerce') + pd.to_numeric(test_df['away_score'], errors='coerce')
            pred_total = pd.to_numeric(test_df['projected_total'], errors='coerce')
            tv = (~actual_total.isna()) & (~pred_total.isna())
            if tv.any():
                fold_rmse = float(np.sqrt(np.mean(np.square(actual_total[tv] - pred_total[tv]))))
        folds.append({
            'train_games': int(len(train_df)),
            'test_games': int(len(test_df)),
            'test_date_start': unique_dates[i].isoformat(),
            'log_loss': fold_ll,
            'accuracy': fold_acc,
            'total_rmse': fold_rmse,
        })
        i += test_window_days

    if not folds:
        result['note'] = 'No valid folds produced.'
        return result

    ll_values = [f['log_loss'] for f in folds]
    acc_values = [f['accuracy'] for f in folds]
    rmse_values = [f['total_rmse'] for f in folds if f['total_rmse'] is not None]

    result['fold_count'] = len(folds)
    result['folds'] = folds
    result['mean_log_loss'] = float(np.mean(ll_values))
    result['log_loss_ci'] = bootstrap_mean_ci(ll_values)
    result['mean_accuracy'] = float(np.mean(acc_values))
    result['accuracy_ci'] = wilson_score_ci(
        sum(int(round(f['accuracy'] * f['test_games'])) for f in folds),
        sum(f['test_games'] for f in folds),
    )
    if rmse_values:
        result['mean_total_rmse'] = float(np.mean(rmse_values))
        result['total_rmse_ci'] = bootstrap_mean_ci(rmse_values)
    result['note'] = (
        f"{len(folds)} folds | train up to {unique_dates[min_train_days - 1].isoformat()} → "
        f"test through {unique_dates[-1].isoformat()}"
    )
    return result


def build_live_regular_calibration_surfaces(archive_df, phase_filter=None):
    if archive_df is None or archive_df.empty:
        return []
    work = archive_df.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    if 'phase' in work.columns:
        work['phase'] = work.get('phase').astype(str).str.lower().str.strip()
    else:
        work['phase'] = work['report_date'].apply(lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring')
    if phase_filter:
        work = work[work['phase'] == str(phase_filter).lower().strip()].copy()
    required = ['p_home', 'y_home', 'report_date']
    for column in ['p_home', 'y_home', 'actual_total', 'projected_total', 'actual_margin', 'projected_margin', 'away_lineup_count', 'home_lineup_count', 'away_starter_confidence', 'home_starter_confidence']:
        if column in work.columns:
            work[column] = pd.to_numeric(work.get(column), errors='coerce')
    work = work.dropna(subset=['p_home', 'y_home', 'report_date']).copy()
    if work.empty:
        return []

    surfaces = []
    segment_specs = [
        ('confirmed lineups', (work.get('away_lineup_count', 0) >= 9) & (work.get('home_lineup_count', 0) >= 9)),
        ('thin lineups', (work.get('away_lineup_count', 0) < 9) | (work.get('home_lineup_count', 0) < 9)),
    ]
    if 'away_starter_confidence' in work.columns and 'home_starter_confidence' in work.columns:
        starter_avg = (pd.to_numeric(work.get('away_starter_confidence'), errors='coerce').fillna(0.0) + pd.to_numeric(work.get('home_starter_confidence'), errors='coerce').fillna(0.0)) / 2.0
        segment_specs.extend([
            ('strong starter signal', starter_avg >= 0.70),
            ('soft starter signal', starter_avg < 0.70),
        ])

    for label, mask in segment_specs:
        grp = work[mask].copy()
        if grp.empty:
            continue
        y = grp['y_home'].astype(int)
        p = grp['p_home'].astype(float).clip(1e-6, 1.0 - 1e-6)
        row = {
            'label': label,
            'games': int(len(grp)),
            'log_loss': float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean()),
            'accuracy': float(((p >= 0.5).astype(int) == y).mean()),
            'total_rmse': None,
            'margin_rmse': None,
        }
        if {'actual_total', 'projected_total'}.issubset(grp.columns):
            total_valid = grp.dropna(subset=['actual_total', 'projected_total']).copy()
            if not total_valid.empty:
                total_error = total_valid['actual_total'].astype(float) - total_valid['projected_total'].astype(float)
                row['total_rmse'] = float(np.sqrt(np.mean(np.square(total_error))))
        if {'actual_margin', 'projected_margin'}.issubset(grp.columns):
            margin_valid = grp.dropna(subset=['actual_margin', 'projected_margin']).copy()
            if not margin_valid.empty:
                margin_error = margin_valid['actual_margin'].astype(float) - margin_valid['projected_margin'].astype(float)
                row['margin_rmse'] = float(np.sqrt(np.mean(np.square(margin_error))))
        surfaces.append(row)
    return surfaces


def build_postgame_attribution_summary(archive_df, phase_filter=None, recent_days=10):
    if archive_df is None or archive_df.empty:
        return {'available': False, 'rows': []}
    work = archive_df.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    if 'phase' in work.columns:
        work['phase'] = work.get('phase').astype(str).str.lower().str.strip()
    else:
        work['phase'] = work['report_date'].apply(lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring')
    if phase_filter:
        work = work[work['phase'] == str(phase_filter).lower().strip()].copy()
    latest_date = max([value for value in work.get('report_date', pd.Series(dtype=object)).dropna().tolist()], default=None)
    if latest_date is None:
        return {'available': False, 'rows': []}
    cutoff = latest_date - dt.timedelta(days=max(int(recent_days), 1) - 1)
    work = work[work['report_date'] >= cutoff].copy()
    numeric_cols = [
        'p_home', 'y_home', 'actual_total', 'projected_total', 'actual_margin', 'projected_margin',
        'away_lineup_count', 'home_lineup_count', 'away_starter_confidence', 'home_starter_confidence',
        'away_short_start_risk', 'home_short_start_risk', 'away_tto_risk', 'home_tto_risk',
        'away_bullpen_stress', 'home_bullpen_stress', 'away_leverage_availability', 'home_leverage_availability',
        'projected_total_sigma', 'probability_shrink_alpha', 'total_shrink_alpha', 'margin_shrink_alpha',
    ]
    for column in numeric_cols:
        if column in work.columns:
            work[column] = pd.to_numeric(work.get(column), errors='coerce')
    work = work[work.get('y_home').notna()].copy()
    if work.empty:
        return {'available': False, 'rows': []}

    moneyline_miss = ((work['p_home'] >= 0.5).astype(int) != work['y_home'].astype(int))
    total_miss = work.get('actual_total').notna() & work.get('projected_total').notna() & ((work['actual_total'] - work['projected_total']).abs() >= 3.5)
    margin_miss = work.get('actual_margin').notna() & work.get('projected_margin').notna() & ((work['actual_margin'] - work['projected_margin']).abs() >= 2.5)
    relevant = work[moneyline_miss | total_miss | margin_miss].copy()
    if relevant.empty:
        return {'available': False, 'rows': []}

    buckets = [
        ('lineup uncertainty', ((relevant.get('away_lineup_count', 9) < 9) | (relevant.get('home_lineup_count', 9) < 9))),
        ('starter uncertainty', (((relevant.get('away_starter_confidence', 1.0) + relevant.get('home_starter_confidence', 1.0)) / 2.0) < 0.65)),
        ('short-start risk', ((relevant.get('away_short_start_risk', 0.0) >= 0.55) | (relevant.get('home_short_start_risk', 0.0) >= 0.55))),
        ('times-through-order exposure', ((relevant.get('away_tto_risk', 0.0) >= 0.45) | (relevant.get('home_tto_risk', 0.0) >= 0.45))),
        ('bullpen availability stress', ((relevant.get('away_bullpen_stress', 0.0) >= 0.35) | (relevant.get('home_bullpen_stress', 0.0) >= 0.35) | (relevant.get('away_leverage_availability', 0.0) <= -0.25) | (relevant.get('home_leverage_availability', 0.0) <= -0.25))),
        ('volatility / shrinkage', ((relevant.get('projected_total_sigma', 0.0) >= 4.8) | (relevant.get('probability_shrink_alpha', 0.0) >= 0.20) | (relevant.get('total_shrink_alpha', 0.0) >= 0.12) | (relevant.get('margin_shrink_alpha', 0.0) >= 0.12))),
    ]
    rows = []
    for label, mask in buckets:
        count = int(pd.Series(mask).fillna(False).sum())
        if count <= 0:
            continue
        rows.append({
            'label': label,
            'count': count,
            'share': float(count / max(len(relevant), 1)),
            'plain_english': f"{label} showed up in {count} of the last {len(relevant)} meaningful live misses.",
        })
    rows = sorted(rows, key=lambda item: item['count'], reverse=True)
    return {'available': bool(rows), 'rows': rows[:4], 'games': int(len(relevant)), 'date_span': f"{cutoff.isoformat()} to {latest_date.isoformat()}"}

def settle_live_prediction_archive(client, archive_df, as_of_date):
    if archive_df is None or archive_df.empty:
        return pd.DataFrame(), 0, []
    work = archive_df.copy()
    if 'settled_ts' not in work.columns:
        work['settled_ts'] = None
    work['settled_ts'] = work['settled_ts'].astype(object)
    for column in ['game_pk', 'away_score', 'home_score', 'y_home', 'actual_total', 'actual_margin', 'away_lineup_count', 'home_lineup_count']:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors='coerce')
    if 'report_date' not in work.columns:
        return work, 0, ['Live prediction archive missing report_date column.']
    work['report_date'] = pd.to_datetime(work['report_date'], errors='coerce').dt.date
    work = work[work['report_date'].notna()].copy()
    if work.empty:
        return work, 0, []
    settled_rows = 0
    warnings = []
    pending = work[(work['report_date'] < as_of_date) & (work['y_home'].isna())].copy()
    for target_date in sorted(set(pending['report_date'].tolist())):
        try:
            payload = schedule_for_date(client, target_date)
        except Exception as exc:
            warnings.append(f"Could not settle archived predictions for {target_date.isoformat()}: {exc}")
            continue
        finals = {}
        for day in payload.get('dates', []):
            for game in day.get('games', []):
                abstract_state = str(((game.get('status') or {}).get('abstractGameState')) or '')
                if abstract_state != 'Final':
                    continue
                away = (game.get('teams') or {}).get('away') or {}
                home = (game.get('teams') or {}).get('home') or {}
                finals[int(game.get('gamePk') or 0)] = {
                    'away_score': int(finite(away.get('score'), 0.0)),
                    'home_score': int(finite(home.get('score'), 0.0)),
                }
        if not finals:
            continue
        mask = work['report_date'] == target_date
        for idx in work[mask].index.tolist():
            game_pk = int(finite(work.at[idx, 'game_pk'], 0.0))
            result = finals.get(game_pk)
            if not result:
                continue
            work.at[idx, 'away_score'] = result['away_score']
            work.at[idx, 'home_score'] = result['home_score']
            work.at[idx, 'y_home'] = 1 if result['home_score'] > result['away_score'] else 0
            work.at[idx, 'actual_total'] = result['home_score'] + result['away_score']
            work.at[idx, 'actual_margin'] = result['home_score'] - result['away_score']
            work.at[idx, 'settled_ts'] = dt.datetime.now(dt.timezone.utc).isoformat()
            settled_rows += 1
    return work, settled_rows, warnings


def archive_live_predictions(report_date, phase, predictions, archive_df=None):
    existing = load_live_prediction_archive() if archive_df is None else archive_df.copy()
    rows = []
    created_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    for prediction in predictions:
        game = prediction.get('game') or {}
        features = prediction.get('features') or {}
        lineups = features.get('lineups') or {}
        lineup_ctx = features.get('lineup_ctx') or {}
        away_lineup = lineup_ctx.get('away') or {}
        home_lineup = lineup_ctx.get('home') or {}
        rows.append({
            'created_ts': created_ts,
            'report_date': report_date.isoformat(),
            'phase': str(phase),
            'game_pk': int(game.get('game_pk') or 0),
            'away': str(game.get('away_team') or ''),
            'home': str(game.get('home_team') or ''),
            'away_pitcher': str(game.get('away_pitcher') or ''),
            'home_pitcher': str(game.get('home_pitcher') or ''),
            'p_home': float(prediction.get('p_home', 0.5)),
            'p_home_model': float(prediction.get('p_home_model', prediction.get('p_home', 0.5))),
            'p_home_simple': float(prediction.get('p_home_simple', prediction.get('p_home', 0.5))),
            'p_home_no_lineup': float(prediction.get('p_home_no_lineup', 0.5)),
            'p_home_full_lineup': float(prediction.get('p_home_full_lineup', 0.5)),
            'projected_total': float(prediction.get('total_calibrated', 0.0)),
            'projected_total_model': float(prediction.get('total_model', prediction.get('total_calibrated', 0.0))),
            'projected_total_simple': float(prediction.get('total_simple', prediction.get('total_calibrated', 0.0))),
            'projected_margin': float(prediction.get('margin_calibrated', 0.0)),
            'projected_margin_model': float(prediction.get('margin_model', prediction.get('margin_calibrated', 0.0))),
            'projected_margin_simple': float(prediction.get('margin_simple', prediction.get('margin_calibrated', 0.0))),
            'lineup_multiplier_used': float(prediction.get('lineup_multiplier', 1.0)),
            'probability_shrink_alpha': float(prediction.get('probability_shrink_alpha', 0.0)),
            'total_shrink_alpha': float(prediction.get('total_shrink_alpha', 0.0)),
            'margin_shrink_alpha': float(prediction.get('margin_shrink_alpha', 0.0)),
            'away_lineup_count': int(lineups.get('away', 0) or 0),
            'home_lineup_count': int(lineups.get('home', 0) or 0),
            'away_lineup_confidence': float(finite(away_lineup.get('avg_confidence'), 0.0)),
            'home_lineup_confidence': float(finite(home_lineup.get('avg_confidence'), 0.0)),
            'away_lineup_prior_heavy': int(away_lineup.get('prior_heavy', 0) or 0),
            'home_lineup_prior_heavy': int(home_lineup.get('prior_heavy', 0) or 0),
            'lineup_delta': float(finite(lineup_ctx.get('delta'), 0.0)),
            'away_starter_confidence': float(finite((features.get('away_starter') or {}).get('confidence'), 0.0)),
            'home_starter_confidence': float(finite((features.get('home_starter') or {}).get('confidence'), 0.0)),
            'away_expected_ip': float(finite((features.get('away_starter_workload') or {}).get('expected_ip'), 0.0)),
            'home_expected_ip': float(finite((features.get('home_starter_workload') or {}).get('expected_ip'), 0.0)),
            'away_short_start_risk': float(finite((features.get('away_starter_workload') or {}).get('short_start_risk'), 0.0)),
            'home_short_start_risk': float(finite((features.get('home_starter_workload') or {}).get('short_start_risk'), 0.0)),
            'away_tto_risk': float(finite((features.get('away_starter_workload') or {}).get('tto_risk'), 0.0)),
            'home_tto_risk': float(finite((features.get('home_starter_workload') or {}).get('tto_risk'), 0.0)),
            'away_bullpen_availability': float(finite((features.get('away_bullpen_profile') or {}).get('availability_score'), 0.0)),
            'home_bullpen_availability': float(finite((features.get('home_bullpen_profile') or {}).get('availability_score'), 0.0)),
            'away_bullpen_stress': float(finite((features.get('away_bullpen_profile') or {}).get('stress_score'), 0.0)),
            'home_bullpen_stress': float(finite((features.get('home_bullpen_profile') or {}).get('stress_score'), 0.0)),
            'away_leverage_availability': float(finite((features.get('away_bullpen_profile') or {}).get('leverage_availability'), 0.0)),
            'home_leverage_availability': float(finite((features.get('home_bullpen_profile') or {}).get('leverage_availability'), 0.0)),
            'projected_total_sigma': float(finite(prediction.get('total_sigma'), 0.0)),
            'context_source': 'live_model',
            'context_refresh_version': int(LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION),
            'context_refreshed_ts': created_ts,
            'away_score': np.nan,
            'home_score': np.nan,
            'y_home': np.nan,
            'actual_total': np.nan,
            'actual_margin': np.nan,
            'settled_ts': '',
        })
    new_rows = pd.DataFrame(rows)
    if existing is None or existing.empty:
        combined = new_rows.copy()
    else:
        for column in new_rows.columns:
            if column not in existing.columns:
                existing[column] = np.nan
        for column in existing.columns:
            if column not in new_rows.columns:
                new_rows[column] = np.nan
        existing_dates = pd.to_datetime(existing.get('report_date'), errors='coerce').dt.date
        existing_keys = pd.to_numeric(existing.get('game_pk'), errors='coerce').fillna(0).astype(int)
        keep_mask = ~((existing_dates == report_date) & (existing_keys.isin(new_rows['game_pk'].astype(int))))
        combined = pd.concat([existing.loc[keep_mask, new_rows.columns], new_rows[new_rows.columns]], ignore_index=True)
    combined = combined.sort_values(['report_date', 'away', 'home']).reset_index(drop=True)
    return combined, len(new_rows)


def backfill_live_archive_context(archive_df):
    if archive_df is None or archive_df.empty:
        return archive_df if isinstance(archive_df, pd.DataFrame) else pd.DataFrame(), 0
    work = archive_df.copy()
    required_columns = [
        'away_starter_confidence', 'home_starter_confidence',
        'away_expected_ip', 'home_expected_ip',
        'away_short_start_risk', 'home_short_start_risk',
        'away_tto_risk', 'home_tto_risk',
        'away_bullpen_availability', 'home_bullpen_availability',
        'away_bullpen_stress', 'home_bullpen_stress',
        'away_leverage_availability', 'home_leverage_availability',
        'projected_total_sigma',
    ]
    for column in required_columns:
        if column not in work.columns:
            work[column] = np.nan
    numeric_columns = [
        'probability_shrink_alpha', 'total_shrink_alpha', 'margin_shrink_alpha',
        'away_lineup_count', 'home_lineup_count',
        'away_lineup_confidence', 'home_lineup_confidence',
        'away_lineup_prior_heavy', 'home_lineup_prior_heavy',
        'projected_total',
    ] + required_columns
    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work.get(column), errors='coerce')
    phase_series = work.get('phase', pd.Series(['spring'] * len(work), index=work.index)).astype(str).str.lower()

    def _starter_confidence(side):
        column = f'{side}_starter_confidence'
        pitcher_name = work.get(f'{side}_pitcher', pd.Series([''] * len(work), index=work.index)).astype(str).str.strip()
        lineup_conf = pd.to_numeric(work.get(f'{side}_lineup_confidence'), errors='coerce').fillna(0.30)
        lineup_count = pd.to_numeric(work.get(f'{side}_lineup_count'), errors='coerce').fillna(0.0).clip(0, 9) / 9.0
        prior_heavy = pd.to_numeric(work.get(f'{side}_lineup_prior_heavy'), errors='coerce').fillna(0.0).clip(0, 9) / 9.0
        known_pitcher = (~pitcher_name.eq('')) & (~pitcher_name.str.upper().isin(['TBD', 'UNKNOWN', 'NAN', 'NONE']))
        base = np.where(phase_series.eq('regular'), 0.56, 0.46)
        missing_penalty = np.where(known_pitcher, 0.0, 0.28)
        derived = np.clip(base + (0.10 * lineup_conf) + (0.06 * lineup_count) - (0.12 * prior_heavy) - missing_penalty, 0.18, 0.82)
        fill_mask = work[column].isna()
        work.loc[fill_mask, column] = derived[fill_mask]

    _starter_confidence('away')
    _starter_confidence('home')

    for side in ['away', 'home']:
        conf = pd.to_numeric(work.get(f'{side}_starter_confidence'), errors='coerce').fillna(0.45)
        ip_column = f'{side}_expected_ip'
        base_ip = np.where(phase_series.eq('regular'), 5.05, 4.15)
        derived_ip = np.clip(base_ip + (1.10 * (conf - 0.50)), 3.2, 6.4)
        fill_ip = work[ip_column].isna()
        work.loc[fill_ip, ip_column] = derived_ip[fill_ip]

        expected_ip = pd.to_numeric(work.get(ip_column), errors='coerce').fillna(derived_ip)
        short_column = f'{side}_short_start_risk'
        derived_short = np.clip((np.where(phase_series.eq('regular'), 5.1, 4.3) - expected_ip) / 1.8, 0.0, 0.92)
        fill_short = work[short_column].isna()
        work.loc[fill_short, short_column] = derived_short[fill_short]

        tto_column = f'{side}_tto_risk'
        derived_tto = np.clip(((expected_ip - np.where(phase_series.eq('regular'), 4.7, 4.0)) / 1.6) * conf, 0.0, 0.85)
        fill_tto = work[tto_column].isna()
        work.loc[fill_tto, tto_column] = derived_tto[fill_tto]

    total_value = pd.to_numeric(work.get('projected_total'), errors='coerce').fillna(8.6)
    avg_short = (
        pd.to_numeric(work.get('away_short_start_risk'), errors='coerce').fillna(0.35) +
        pd.to_numeric(work.get('home_short_start_risk'), errors='coerce').fillna(0.35)
    ) / 2.0
    avg_conf = (
        pd.to_numeric(work.get('away_starter_confidence'), errors='coerce').fillna(0.45) +
        pd.to_numeric(work.get('home_starter_confidence'), errors='coerce').fillna(0.45)
    ) / 2.0
    sigma_derived = np.clip(
        np.where(phase_series.eq('regular'), 3.85, 4.15)
        + (0.18 * np.abs(total_value - 8.6))
        + (0.55 * avg_short)
        - (0.12 * avg_conf),
        2.8,
        6.1,
    )
    sigma_mask = work['projected_total_sigma'].isna()
    work.loc[sigma_mask, 'projected_total_sigma'] = sigma_derived[sigma_mask]

    sigma_value = pd.to_numeric(work.get('projected_total_sigma'), errors='coerce').fillna(sigma_derived)
    for side in ['away', 'home']:
        short_risk = pd.to_numeric(work.get(f'{side}_short_start_risk'), errors='coerce').fillna(0.35)
        availability_column = f'{side}_bullpen_availability'
        derived_availability = np.clip(0.18 - (0.34 * short_risk) - (0.04 * (sigma_value - 4.0)), -0.40, 0.40)
        fill_availability = work[availability_column].isna()
        work.loc[fill_availability, availability_column] = derived_availability[fill_availability]

        availability = pd.to_numeric(work.get(availability_column), errors='coerce').fillna(derived_availability)
        stress_column = f'{side}_bullpen_stress'
        derived_stress = np.clip(0.22 + (0.58 * short_risk) + (0.12 * np.maximum(sigma_value - 4.0, 0.0)) - (0.18 * availability), 0.0, 1.0)
        fill_stress = work[stress_column].isna()
        work.loc[fill_stress, stress_column] = derived_stress[fill_stress]

        stress = pd.to_numeric(work.get(stress_column), errors='coerce').fillna(derived_stress)
        leverage_column = f'{side}_leverage_availability'
        derived_leverage = np.clip(availability - (0.22 * stress), -0.45, 0.38)
        fill_leverage = work[leverage_column].isna()
        work.loc[fill_leverage, leverage_column] = derived_leverage[fill_leverage]

    backfill_count = 0
    for column in required_columns:
        if column in archive_df.columns:
            backfill_count = max(backfill_count, int(work[column].notna().sum() - pd.to_numeric(archive_df.get(column), errors='coerce').notna().sum()))
        else:
            backfill_count = max(backfill_count, int(work[column].notna().sum()))
    return work, backfill_count


def previous_daily_run_snapshot(report_date):
    if not os.path.exists(QUANT_DB_PATH):
        return {}
    try:
        import duckdb  # type: ignore
    except Exception:
        return {}
    try:
        con = duckdb.connect(QUANT_DB_PATH, read_only=True)
        row = con.execute(
            """
            SELECT *
            FROM quant_runs_daily
            WHERE CAST(report_date AS DATE) < CAST(? AS DATE)
            ORDER BY CAST(report_date AS DATE) DESC, TRY_CAST(created_ts AS TIMESTAMP) DESC NULLS LAST, created_ts DESC
            LIMIT 1
            """,
            [report_date.isoformat()],
        ).fetchdf()
        con.close()
    except Exception:
        return {}
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def build_model_change_log(report_date, state, predictions, live_validation, market_proof):
    previous = previous_daily_run_snapshot(report_date)
    if not previous:
        return {
            'available': False,
            'lines': ['No prior daily run is stored yet, so day-over-day change tracking will begin on the next report.'],
        }
    candidates = []
    market_counts = {'moneyline': 0, 'run_line': 0, 'total': 0}
    must_take = 0
    for prediction in predictions or []:
        market_comp = prediction.get('market_comp') or {}
        for market_type in ['moneyline', 'run_line', 'total']:
            market_row = market_comp.get(market_type) or {}
            action = market_row.get('action') or {}
            label = str(action.get('label') or 'PASS').upper()
            if label in {'WATCH', 'BET', 'MUST TAKE'}:
                candidates.append((market_type, label))
                market_counts[market_type] += 1
                if label == 'MUST TAKE':
                    must_take += 1
    actionable_count = len(candidates)
    current_lineup_profile = (state.get('lineup_earn_back') or {}) if isinstance(state, dict) else {}
    previous_state = {}
    try:
        previous_state = json.loads(str(previous.get('state_json') or '{}'))
    except Exception:
        previous_state = {}
    previous_lineup_profile = (previous_state.get('lineup_earn_back') or {}) if isinstance(previous_state, dict) else {}

    def _delta_text(current_value, previous_value, fmt='.4f', scale=1.0, suffix=''):
        current_num = safe_float(current_value)
        previous_num = safe_float(previous_value)
        if current_num is None or previous_num is None:
            return None
        delta = (current_num - previous_num) * scale
        sign = '+' if delta >= 0 else ''
        return f"{format(current_num * scale if scale != 1.0 else current_num, fmt)} ({sign}{format(delta, fmt)}{suffix})"

    current_by_market = {str(row.get('market_type', '')).lower(): row for row in (market_proof.get('by_market') or [])}
    lines = []
    live_log_loss = _delta_text(live_validation.get('log_loss'), previous.get('live_validation_log_loss'), '.4f')
    live_accuracy_current = safe_float(live_validation.get('accuracy'))
    live_accuracy_previous = safe_float(previous.get('live_validation_accuracy'))
    live_accuracy = None
    if live_accuracy_current is not None and live_accuracy_previous is not None:
        live_accuracy = f"{live_accuracy_current * 100:.1f}% ({(live_accuracy_current - live_accuracy_previous) * 100:+.1f} pts)"
    if live_log_loss is not None or live_accuracy is not None:
        parts = []
        if live_log_loss is not None:
            parts.append(f"log loss {live_log_loss}")
        if live_accuracy is not None:
            parts.append(f"accuracy {live_accuracy}")
        lines.append('Live settled validation: ' + ' | '.join(parts))
    total_mae = _delta_text(live_validation.get('total_mae'), previous.get('live_validation_total_mae'), '.3f')
    margin_mae = _delta_text(live_validation.get('margin_mae'), previous.get('live_validation_margin_mae'), '.3f')
    if total_mae is not None or margin_mae is not None:
        parts = []
        if total_mae is not None:
            parts.append(f"total MAE {total_mae}")
        if margin_mae is not None:
            parts.append(f"margin MAE {margin_mae}")
        lines.append('Live settled error: ' + ' | '.join(parts))
    current_roi = safe_float(market_proof.get('roi'))
    previous_roi = safe_float(previous.get('market_proof_roi'))
    current_clv = safe_float(market_proof.get('avg_clv'))
    previous_clv = safe_float(previous.get('market_proof_avg_clv'))
    if current_roi is not None or current_clv is not None:
        roi_text = None if current_roi is None or previous_roi is None else f"{current_roi * 100:.1f}% ({(current_roi - previous_roi) * 100:+.1f} pts)"
        clv_text = None if current_clv is None or previous_clv is None else f"{current_clv:+.3f} ({(current_clv - previous_clv):+.3f})"
        parts = [part for part in [f"ROI {roi_text}" if roi_text else None, f"avg CLV {clv_text}" if clv_text else None] if part]
        if parts:
            lines.append('Market proof: ' + ' | '.join(parts))
    for market_name, label in [('moneyline', 'Moneyline'), ('run_line', 'Run line'), ('total', 'Total')]:
        current_row = current_by_market.get(market_name) or {}
        current_market_roi = safe_float(current_row.get('roi'))
        previous_market_roi = safe_float(previous.get(f'market_proof_{market_name}_roi'))
        current_market_clv = safe_float(current_row.get('avg_clv'))
        previous_market_clv = safe_float(previous.get(f'market_proof_{market_name}_avg_clv'))
        if current_market_roi is None and current_market_clv is None:
            continue
        roi_part = None if current_market_roi is None or previous_market_roi is None else f"ROI {current_market_roi * 100:.1f}% ({(current_market_roi - previous_market_roi) * 100:+.1f} pts)"
        clv_part = None if current_market_clv is None or previous_market_clv is None else f"CLV {current_market_clv:+.3f} ({(current_market_clv - previous_market_clv):+.3f})"
        parts = [part for part in [roi_part, clv_part] if part]
        if parts:
            lines.append(f"{label} proof: " + ' | '.join(parts))
    previous_actionable = int(finite(previous.get('actionable_market_count'), 0.0))
    previous_must_take = int(finite(previous.get('must_take_count'), 0.0))
    lines.append(
        f"Actionable slate mix: {actionable_count} actionable ({actionable_count - previous_actionable:+d}) | "
        f"MUST TAKE {must_take} ({must_take - previous_must_take:+d}) | "
        f"ML {market_counts['moneyline']} | RL {market_counts['run_line']} | TOT {market_counts['total']}"
    )
    current_lineup_games = int(finite(current_lineup_profile.get('confirmed_games'), 0.0))
    previous_lineup_games = int(finite(previous_lineup_profile.get('confirmed_games'), 0.0))
    lines.append(
        f"Lineup earn-back progress: {current_lineup_games}/{int(finite(current_lineup_profile.get('required_games'), 0.0))} "
        f"({current_lineup_games - previous_lineup_games:+d} vs prior report)"
    )
    return {'available': True, 'lines': lines}


def replay_prediction_context_for_date(client, target_date, state, park_cache, venue_cache, validation_log):
    warnings = []
    games, schedule_warnings = fetch_schedule_games(client, target_date)
    warnings.extend(schedule_warnings)
    if not games:
        return {}, warnings, venue_cache
    phase = season_phase_for_games(games)
    weights = phase_weights_from_state(state, phase)
    margin_sigma = estimate_margin_sigma(validation_log)
    season_start = dt.date(target_date.year, 1, 1)
    pregame_end = target_date
    game_types = 'S|' if phase == 'spring' else 'R|'

    current_batter_raw = savant_statcast_csv(client, 'batter', target_date.year, season_start, pregame_end, game_types)
    batter_raw_30 = savant_statcast_csv(client, 'batter', target_date.year, target_date - dt.timedelta(days=30), pregame_end, game_types)
    batter_raw_14 = savant_statcast_csv(client, 'batter', target_date.year, target_date - dt.timedelta(days=14), pregame_end, game_types)
    batter_raw_7 = savant_statcast_csv(client, 'batter', target_date.year, target_date - dt.timedelta(days=7), pregame_end, game_types)
    current_batters = aggregate_batter_quality(current_batter_raw)
    prior_batters = previous_regular_season_batters(client, target_date.year - 1)
    strengths = build_team_strengths(
        client,
        target_date,
        phase,
        games,
        current_batter_raw,
        batter_raw_30,
        batter_raw_14,
        batter_raw_7,
        current_batters,
        prior_batters,
    )
    current_pitchers = aggregate_pitcher_quality(savant_statcast_csv(client, 'pitcher', target_date.year, season_start, pregame_end, game_types))
    prior_pitchers = previous_regular_season_pitchers(client, target_date.year - 1)
    bullpen = bullpen_snapshot(client, target_date, 7, current_pitchers, prior_pitchers)
    bullpen_profiles = bullpen.set_index('team').to_dict('index') if bullpen is not None and not bullpen.empty else {}

    context_map = {}
    for game in games:
        prediction = predict_game(
            game,
            phase,
            weights,
            state,
            strengths,
            current_pitchers,
            prior_pitchers,
            current_batters,
            prior_batters,
            bullpen_profiles,
            park_cache,
            client,
            venue_cache,
            margin_sigma,
        )
        features = prediction.get('features') or {}
        away_starter_workload = features.get('away_starter_workload') or {}
        home_starter_workload = features.get('home_starter_workload') or {}
        away_bullpen_profile = features.get('away_bullpen_profile') or {}
        home_bullpen_profile = features.get('home_bullpen_profile') or {}
        context_map[int(game.get('game_pk') or 0)] = {
            'away_starter_confidence': float(finite((features.get('away_starter') or {}).get('confidence'), 0.0)),
            'home_starter_confidence': float(finite((features.get('home_starter') or {}).get('confidence'), 0.0)),
            'away_expected_ip': float(finite(away_starter_workload.get('expected_ip'), 0.0)),
            'home_expected_ip': float(finite(home_starter_workload.get('expected_ip'), 0.0)),
            'away_short_start_risk': float(finite(away_starter_workload.get('short_start_risk'), 0.0)),
            'home_short_start_risk': float(finite(home_starter_workload.get('short_start_risk'), 0.0)),
            'away_tto_risk': float(finite(away_starter_workload.get('tto_risk'), 0.0)),
            'home_tto_risk': float(finite(home_starter_workload.get('tto_risk'), 0.0)),
            'away_bullpen_availability': float(finite(away_bullpen_profile.get('availability_score'), 0.0)),
            'home_bullpen_availability': float(finite(home_bullpen_profile.get('availability_score'), 0.0)),
            'away_bullpen_stress': float(finite(away_bullpen_profile.get('stress_score'), 0.0)),
            'home_bullpen_stress': float(finite(home_bullpen_profile.get('stress_score'), 0.0)),
            'away_leverage_availability': float(finite(away_bullpen_profile.get('leverage_availability'), 0.0)),
            'home_leverage_availability': float(finite(home_bullpen_profile.get('leverage_availability'), 0.0)),
            'projected_total_sigma': float(finite(prediction.get('total_sigma'), 0.0)),
            'context_source': 'historical_replay',
            'context_refresh_version': int(LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION),
            'context_refreshed_ts': dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    return context_map, warnings, venue_cache


def refresh_live_archive_context_from_replay(client, archive_df, state, park_cache, venue_cache, validation_log, as_of_date, max_dates=None):
    if archive_df is None or archive_df.empty:
        return archive_df if isinstance(archive_df, pd.DataFrame) else pd.DataFrame(), state, venue_cache, {'dates_refreshed': 0, 'rows_refreshed': 0, 'warnings': []}
    work = archive_df.copy()
    for column in ['context_source', 'context_refresh_version', 'context_refreshed_ts']:
        if column not in work.columns:
            work[column] = np.nan
    work['context_source'] = work['context_source'].astype('object')
    work['context_refreshed_ts'] = work['context_refreshed_ts'].astype('object')
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    refresh_state = dict((state.get('live_archive_context_refresh') or {}) if isinstance(state, dict) else {})
    completed_dates = set(str(item) for item in (refresh_state.get('completed_dates') or []))
    current_version = int(finite(refresh_state.get('version'), 0.0))
    eligible_dates = sorted(date for date in work['report_date'].dropna().unique().tolist() if isinstance(date, dt.date) and date < as_of_date)
    pending_dates = []
    for target_date in eligible_dates:
        date_key = target_date.isoformat()
        date_mask = work['report_date'] == target_date
        date_versions = pd.to_numeric(work.loc[date_mask, 'context_refresh_version'], errors='coerce')
        date_sources = work.loc[date_mask, 'context_source'].astype(str).str.lower()
        already_refreshed = (
            current_version == LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION
            and date_key in completed_dates
            and date_versions.notna().all()
            and (date_versions.fillna(0).astype(int) == LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION).all()
            and date_sources.eq('historical_replay').all()
        )
        if not already_refreshed:
            pending_dates.append(target_date)
    if not pending_dates:
        refresh_state.update({
            'version': LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION,
            'completed_dates': sorted(set(eligible_dates and [date.isoformat() for date in eligible_dates] or []) | completed_dates),
            'last_run_ts': dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        next_state = dict(state or {})
        next_state['live_archive_context_refresh'] = refresh_state
        return work, next_state, venue_cache, {'dates_refreshed': 0, 'rows_refreshed': 0, 'warnings': []}

    warnings = []
    rows_refreshed = 0
    refreshed_dates = []
    date_limit = None if max_dates is None else max(0, int(max_dates))
    for target_date in pending_dates[:date_limit]:
        context_map, replay_warnings, venue_cache = replay_prediction_context_for_date(
            client,
            target_date,
            state,
            park_cache,
            venue_cache,
            validation_log,
        )
        warnings.extend([f"{target_date.isoformat()}: {warning}" for warning in replay_warnings])
        if not context_map:
            warnings.append(f"{target_date.isoformat()}: historical replay produced no context rows.")
            continue
        date_mask = work['report_date'] == target_date
        game_keys = pd.to_numeric(work.loc[date_mask, 'game_pk'], errors='coerce').fillna(0).astype(int)
        updated_here = 0
        for idx, game_pk in game_keys.items():
            payload = context_map.get(int(game_pk))
            if not payload:
                continue
            for column, value in payload.items():
                work.at[idx, column] = value
            updated_here += 1
        if updated_here:
            refreshed_dates.append(target_date.isoformat())
            completed_dates.add(target_date.isoformat())
            rows_refreshed += updated_here
    refresh_state.update({
        'version': LIVE_ARCHIVE_CONTEXT_REPLAY_VERSION,
        'completed_dates': sorted(completed_dates),
        'last_run_ts': dt.datetime.now(dt.timezone.utc).isoformat(),
        'last_rows_refreshed': rows_refreshed,
    })
    next_state = dict(state or {})
    next_state['live_archive_context_refresh'] = refresh_state
    pending_remaining = max(0, len(pending_dates) - len(refreshed_dates))
    return work, next_state, venue_cache, {
        'dates_refreshed': len(refreshed_dates),
        'rows_refreshed': rows_refreshed,
        'warnings': warnings,
        'completed_dates': refreshed_dates,
        'pending_dates_remaining': pending_remaining,
        'pending_dates_total': len(pending_dates),
    }


def update_lineup_earn_back_state(state, archive_df):
    profile = {
        'apply': False,
        'mode': 'awaiting_regular_confirmed_lineups',
        'confirmed_games': 0,
        'required_games': int(LINEUP_EARN_BACK_MIN_GAMES),
        'lineup_multiplier': 0.0,
        'no_lineup_log_loss': None,
        'full_lineup_log_loss': None,
        'improvement': None,
        'date_span': 'N/A',
        'reason': f"need {LINEUP_EARN_BACK_MIN_GAMES} settled regular-season confirmed-lineup games before activating lineup depth",
    }
    next_state = dict(state or {})
    if archive_df is None or archive_df.empty:
        next_state['lineup_earn_back'] = profile
        return next_state, profile
    work = archive_df.copy()
    for column in ['away_lineup_count', 'home_lineup_count', 'y_home', 'p_home_no_lineup', 'p_home_full_lineup']:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors='coerce')
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    if 'phase' not in work.columns:
        next_state['lineup_earn_back'] = profile
        return next_state, profile
    work = work[work['phase'].astype(str).str.lower() == 'regular'].copy()
    if work.empty:
        next_state['lineup_earn_back'] = profile
        return next_state, profile
    confirmed_mask = (work['away_lineup_count'] >= LINEUP_EARN_BACK_MIN_CONFIRMED) & (work['home_lineup_count'] >= LINEUP_EARN_BACK_MIN_CONFIRMED) & work['y_home'].notna() & work['p_home_no_lineup'].notna() & work['p_home_full_lineup'].notna()
    confirmed = work[confirmed_mask].copy()
    games = int(len(confirmed))
    profile['confirmed_games'] = games
    if games > 0:
        dates = sorted(confirmed['report_date'].dropna().unique().tolist())
        if dates:
            profile['date_span'] = f"{dates[0].isoformat()} to {dates[-1].isoformat()}"
    if games < LINEUP_EARN_BACK_MIN_GAMES:
        profile['reason'] = f"need {LINEUP_EARN_BACK_MIN_GAMES} settled regular-season confirmed-lineup games; currently {games}"
        next_state['lineup_earn_back'] = profile
        return next_state, profile
    y = confirmed['y_home'].astype(int)
    p_no = confirmed['p_home_no_lineup'].clip(1e-6, 1.0 - 1e-6)
    p_full = confirmed['p_home_full_lineup'].clip(1e-6, 1.0 - 1e-6)
    no_lineup_log_loss = float((-(y * np.log(p_no)) - ((1 - y) * np.log(1.0 - p_no))).mean())
    full_lineup_log_loss = float((-(y * np.log(p_full)) - ((1 - y) * np.log(1.0 - p_full))).mean())
    improvement = float(no_lineup_log_loss - full_lineup_log_loss)
    profile['no_lineup_log_loss'] = no_lineup_log_loss
    profile['full_lineup_log_loss'] = full_lineup_log_loss
    profile['improvement'] = improvement
    if improvement >= LINEUP_EARN_BACK_MIN_IMPROVEMENT:
        sample_scale = min(1.0, games / float(LINEUP_EARN_BACK_MIN_GAMES * 2.0))
        edge_scale = min(1.0, improvement / float(LINEUP_EARN_BACK_MIN_IMPROVEMENT * 3.0))
        multiplier = float(np.clip(0.25 + (0.75 * min(sample_scale, edge_scale)), 0.25, 1.0))
        profile.update({
            'apply': True,
            'mode': 'regular_confirmed_lineup_validation',
            'lineup_multiplier': multiplier,
            'reason': f"confirmed-lineup regular sample improved log loss by {improvement:.4f} over {games} games",
        })
    else:
        profile.update({
            'apply': False,
            'mode': 'regular_confirmed_lineup_validation',
            'lineup_multiplier': 0.0,
            'reason': f"confirmed-lineup regular sample has not earned activation yet ({improvement:+.4f} log-loss improvement over {games} games)",
        })
    next_state['lineup_earn_back'] = profile
    return next_state, profile


def load_settled_bet_journal(db_path=SPORTSBOOK_DB_PATH):
    columns = [
        'report_date', 'phase', 'market_type', 'stake_units', 'result_units', 'clv_value',
        'edge_value', 'actionability_score', 'book_price', 'line_value',
    ]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=columns)
    try:
        with sqlite3.connect(db_path) as conn:
            journal = pd.read_sql_query('SELECT * FROM bet_journal', conn)
    except Exception:
        return pd.DataFrame(columns=columns)
    if journal.empty or 'result_units' not in journal.columns:
        return pd.DataFrame(columns=(journal.columns.tolist() if not journal.empty else columns))

    numeric_columns = ['stake_units', 'result_units', 'clv_value', 'edge_value', 'actionability_score', 'book_price', 'line_value']
    for column in numeric_columns:
        if column in journal.columns:
            journal[column] = pd.to_numeric(journal.get(column), errors='coerce')
    if 'report_date' in journal.columns:
        journal['report_date'] = pd.to_datetime(journal.get('report_date'), errors='coerce').dt.date
    else:
        journal['report_date'] = pd.NaT
    if 'phase' not in journal.columns:
        journal['phase'] = None
    journal['phase'] = journal['phase'].astype(object)
    missing_phase = journal['phase'].isna() | (journal['phase'].astype(str).str.strip() == '')
    if missing_phase.any():
        journal.loc[missing_phase, 'phase'] = journal.loc[missing_phase, 'report_date'].apply(
            lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring'
        )
    if 'market_type' in journal.columns:
        normalized_market = journal['market_type'].astype(str).str.strip().str.lower()
        normalized_market = normalized_market.replace({
            'ml': 'moneyline',
            'money line': 'moneyline',
            'run line': 'run_line',
            'spread': 'run_line',
            'spreads': 'run_line',
            'totals': 'total',
        })
        journal['market_type'] = normalized_market
    journal['stake_units'] = pd.to_numeric(journal.get('stake_units'), errors='coerce').fillna(1.0)
    settled = journal[journal['result_units'].notna()].copy()
    return settled.reset_index(drop=True)


def update_realized_uncertainty_state(state, archive_df, settled_journal_df=None):
    next_state = dict(state or {})
    profile = {
        'updated_ts': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source': 'settled live archive + settled market journal',
        'spring': {},
        'regular': {},
    }

    def default_market_profile(market_name, reason):
        return {
            'apply': False,
            'market_type': market_name,
            'settled_games': 0,
            'priced_bets': 0,
            'priced_days': 0,
            'model_error': None,
            'simple_error': None,
            'tail_model_error': None,
            'tail_simple_error': None,
            'avg_clv': None,
            'roi': None,
            'confidence': 0.0,
            'bias_add': 0.0,
            'penalty_scale': 1.0,
            'reason': reason,
        }

    archive_work = archive_df.copy() if isinstance(archive_df, pd.DataFrame) else pd.DataFrame()
    if not archive_work.empty:
        if 'report_date' in archive_work.columns:
            archive_work['report_date'] = pd.to_datetime(archive_work.get('report_date'), errors='coerce').dt.date
        if 'phase' not in archive_work.columns:
            archive_work['phase'] = archive_work['report_date'].apply(
                lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring'
            )
        numeric_columns = [
            'y_home', 'p_home_model', 'p_home_simple',
            'actual_total', 'projected_total_model', 'projected_total_simple',
            'actual_margin', 'projected_margin_model', 'projected_margin_simple',
        ]
        for column in numeric_columns:
            if column in archive_work.columns:
                archive_work[column] = pd.to_numeric(archive_work.get(column), errors='coerce')

    journal_work = settled_journal_df.copy() if isinstance(settled_journal_df, pd.DataFrame) else pd.DataFrame()
    if not journal_work.empty:
        if 'report_date' in journal_work.columns:
            journal_work['report_date'] = pd.to_datetime(journal_work.get('report_date'), errors='coerce').dt.date
        if 'phase' not in journal_work.columns:
            journal_work['phase'] = journal_work['report_date'].apply(
                lambda value: season_phase_for_date(value) if isinstance(value, dt.date) else 'spring'
            )
        for column in ['stake_units', 'result_units', 'clv_value']:
            if column in journal_work.columns:
                journal_work[column] = pd.to_numeric(journal_work.get(column), errors='coerce')
        if 'market_type' in journal_work.columns:
            journal_work['market_type'] = journal_work['market_type'].astype(str).str.strip().str.lower()

    for phase_name in ['spring', 'regular']:
        phase_archive = archive_work[archive_work['phase'].astype(str).str.lower() == phase_name].copy() if not archive_work.empty else pd.DataFrame()
        phase_journal = journal_work[journal_work['phase'].astype(str).str.lower() == phase_name].copy() if not journal_work.empty else pd.DataFrame()
        for market_name in ['moneyline', 'total', 'run_line']:
            market_profile = default_market_profile(market_name, 'no settled evidence yet')
            games = 0
            model_error = None
            simple_error = None
            tail_model_error = None
            tail_simple_error = None

            if market_name == 'moneyline' and not phase_archive.empty:
                valid = phase_archive.dropna(subset=['y_home', 'p_home_model', 'p_home_simple']).copy()
                if not valid.empty:
                    y = valid['y_home'].astype(float)
                    model_prob = valid['p_home_model'].clip(1e-6, 1.0 - 1e-6)
                    simple_prob = valid['p_home_simple'].clip(1e-6, 1.0 - 1e-6)
                    model_errors = np.square(model_prob - y)
                    simple_errors = np.square(simple_prob - y)
                    games = int(len(valid))
                    model_error = float(model_errors.mean())
                    simple_error = float(simple_errors.mean())
                    tail_mask = (model_prob - 0.5).abs() >= 0.12
                    if int(tail_mask.sum()) < min(6, games):
                        threshold = float((model_prob - 0.5).abs().quantile(0.70)) if games > 1 else 0.12
                        tail_mask = (model_prob - 0.5).abs() >= threshold
                    if tail_mask.any():
                        tail_model_error = float(model_errors[tail_mask].mean())
                        tail_simple_error = float(simple_errors[tail_mask].mean())
            elif market_name == 'total' and not phase_archive.empty:
                valid = phase_archive.dropna(subset=['actual_total', 'projected_total_model', 'projected_total_simple']).copy()
                if not valid.empty:
                    actual = valid['actual_total'].astype(float)
                    model_pred = valid['projected_total_model'].astype(float)
                    simple_pred = valid['projected_total_simple'].astype(float)
                    model_errors = (actual - model_pred).abs()
                    simple_errors = (actual - simple_pred).abs()
                    games = int(len(valid))
                    model_error = float(model_errors.mean())
                    simple_error = float(simple_errors.mean())
                    tail_mask = model_pred >= (9.0 if phase_name == 'spring' else 8.6)
                    if int(tail_mask.sum()) < min(6, games):
                        threshold = float(model_pred.quantile(0.70)) if games > 1 else (9.0 if phase_name == 'spring' else 8.6)
                        tail_mask = model_pred >= threshold
                    if tail_mask.any():
                        tail_model_error = float(model_errors[tail_mask].mean())
                        tail_simple_error = float(simple_errors[tail_mask].mean())
            elif market_name == 'run_line' and not phase_archive.empty:
                valid = phase_archive.dropna(subset=['actual_margin', 'projected_margin_model', 'projected_margin_simple']).copy()
                if not valid.empty:
                    actual = valid['actual_margin'].astype(float)
                    model_pred = valid['projected_margin_model'].astype(float)
                    simple_pred = valid['projected_margin_simple'].astype(float)
                    model_errors = (actual - model_pred).abs()
                    simple_errors = (actual - simple_pred).abs()
                    games = int(len(valid))
                    model_error = float(model_errors.mean())
                    simple_error = float(simple_errors.mean())
                    tail_mask = model_pred.abs() >= 1.5
                    if int(tail_mask.sum()) < min(6, games):
                        threshold = float(model_pred.abs().quantile(0.70)) if games > 1 else 1.5
                        tail_mask = model_pred.abs() >= threshold
                    if tail_mask.any():
                        tail_model_error = float(model_errors[tail_mask].mean())
                        tail_simple_error = float(simple_errors[tail_mask].mean())

            market_journal = phase_journal[phase_journal['market_type'] == market_name].copy() if not phase_journal.empty else pd.DataFrame()
            priced_bets = int(len(market_journal))
            priced_days = int(market_journal['report_date'].dropna().nunique()) if ('report_date' in market_journal.columns and not market_journal.empty) else 0
            avg_clv = float(market_journal['clv_value'].dropna().mean()) if ('clv_value' in market_journal.columns and market_journal['clv_value'].dropna().any()) else None
            total_stake = float(market_journal['stake_units'].fillna(1.0).sum()) if ('stake_units' in market_journal.columns and not market_journal.empty) else 0.0
            roi = float(market_journal['result_units'].sum() / total_stake) if total_stake > 0 and 'result_units' in market_journal.columns else None

            evidence_support = np.clip((games / float(REALIZED_UNCERTAINTY_MIN_GAMES)) * 0.65 + (priced_bets / float(REALIZED_UNCERTAINTY_MIN_BETS)) * 0.35, 0.0, 1.0)
            confidence = np.clip((games / float(REALIZED_UNCERTAINTY_HIGH_CONF_GAMES)) * 0.65 + (priced_bets / float(REALIZED_UNCERTAINTY_HIGH_CONF_BETS)) * 0.35, 0.0, 1.0)

            error_scale = {'moneyline': 0.03, 'total': 0.85, 'run_line': 0.70}[market_name]
            tail_scale = {'moneyline': 0.04, 'total': 1.10, 'run_line': 0.90}[market_name]
            clv_scale = {'moneyline': 0.05, 'total': 0.75, 'run_line': 0.75}[market_name]
            roi_scale = 0.12

            overall_gap = 0.0
            if model_error is not None and simple_error is not None:
                overall_gap = float(np.clip((model_error - simple_error) / error_scale, -1.0, 1.0))
            tail_gap = 0.0
            if tail_model_error is not None and tail_simple_error is not None:
                tail_gap = float(np.clip((tail_model_error - tail_simple_error) / tail_scale, -1.0, 1.0))
            clv_gap = 0.0
            if avg_clv is not None:
                clv_gap = float(np.clip((-avg_clv) / clv_scale, -1.0, 1.0))
            roi_gap = 0.0
            if roi is not None:
                roi_gap = float(np.clip((-roi) / roi_scale, -1.0, 1.0))

            bias_score = (0.55 * overall_gap) + (0.20 * tail_gap) + (0.15 * clv_gap) + (0.10 * roi_gap)
            active = bool((games >= REALIZED_UNCERTAINTY_MIN_GAMES) or (priced_bets >= REALIZED_UNCERTAINTY_MIN_BETS))
            if active:
                bias_add = float(np.clip(bias_score * (0.02 + (0.08 * confidence)), -0.06, 0.12))
                penalty_scale = float(np.clip(
                    1.0 - (0.30 * max(bias_score, 0.0) * max(confidence, 0.15)) + (0.08 * max(-bias_score, 0.0) * confidence),
                    0.55,
                    1.05,
                ))
                if overall_gap > 0.10:
                    performance_note = 'model is underperforming the simple baseline'
                elif overall_gap < -0.10:
                    performance_note = 'model is outperforming the simple baseline'
                else:
                    performance_note = 'model and simple baseline are still fairly close'
                extras = []
                if avg_clv is not None:
                    extras.append(f'avg CLV {avg_clv:+.3f}')
                if roi is not None:
                    extras.append(f'ROI {roi:+.1%}')
                extra_text = '' if not extras else '; ' + ', '.join(extras)
                reason = f'{performance_note}; {games} settled games, {priced_bets} priced bets{extra_text}'
            else:
                bias_add = 0.0
                penalty_scale = 1.0
                reason = f'collecting evidence ({games}/{REALIZED_UNCERTAINTY_MIN_GAMES} settled games, {priced_bets}/{REALIZED_UNCERTAINTY_MIN_BETS} priced bets)'

            market_profile.update({
                'apply': active,
                'settled_games': games,
                'priced_bets': priced_bets,
                'priced_days': priced_days,
                'model_error': model_error,
                'simple_error': simple_error,
                'tail_model_error': tail_model_error,
                'tail_simple_error': tail_simple_error,
                'avg_clv': avg_clv,
                'roi': roi,
                'confidence': float(np.clip(confidence * max(evidence_support, 0.15 if active else 0.0), 0.0, 1.0)),
                'bias_add': bias_add,
                'penalty_scale': penalty_scale,
                'reason': reason,
            })
            profile[phase_name][market_name] = market_profile

    next_state['realized_uncertainty'] = profile
    return next_state, profile

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

def estimate_total_sigma(phase, shared_env, starter_uncertainty, short_start_total, bullpen_fragility, lineup_missing, weather_total_adj, offense_pressure):
    params = total_sigma_params(phase)
    sigma = (
        finite(params.get('base_sigma'), 4.0)
        + (finite(params.get('starter_uncertainty_coef'), 0.30) * max(0.0, finite(starter_uncertainty, 0.0)))
        + (finite(params.get('short_start_coef'), 0.50) * max(0.0, finite(short_start_total, 0.0)))
        + (finite(params.get('bullpen_fragility_coef'), 0.40) * max(0.0, finite(bullpen_fragility, 0.0)))
        + (finite(params.get('lineup_missing_coef'), 0.15) * max(0.0, finite(lineup_missing, 0.0)))
        + (finite(params.get('env_coef'), 0.20) * max(0.0, finite(shared_env, 0.0)))
        + (finite(params.get('weather_vol_coef'), 0.12) * abs(finite(weather_total_adj, 0.0)))
        + (finite(params.get('offense_pressure_coef'), 0.08) * max(0.0, finite(offense_pressure, 0.0)))
    )
    return float(np.clip(sigma, finite(params.get('min_sigma'), 2.8), finite(params.get('max_sigma'), 6.0)))


def total_side_probability(side, market_total, projected_total, sigma):
    sigma_value = float(np.clip(finite(sigma, 4.0), 2.0, 7.0))
    total_line = finite(market_total, finite(projected_total, 9.0))
    total_mean = finite(projected_total, 9.0)
    over_prob = 1.0 - normal_cdf((total_line - total_mean) / sigma_value)
    over_prob = float(np.clip(over_prob, 1e-6, 1.0 - 1e-6))
    if str(side or '').upper() == 'OVER':
        return over_prob
    return float(np.clip(1.0 - over_prob, 1e-6, 1.0 - 1e-6))


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


def ranked_market_drivers(prediction, market_type):
    features = prediction.get('features', {}) or {}
    context_components = features.get('context_components', {}) or {}
    market_name = str(market_type or '').lower()
    if market_name == 'moneyline':
        driver_map = {
            'offense edge': finite(features.get('x_off'), 0.0),
            'starter edge': finite(features.get('x_starter'), 0.0),
            'bullpen edge': finite(features.get('x_bullpen'), 0.0),
            'schedule/defense': finite(context_components.get('schedule'), 0.0) + finite(context_components.get('defense'), 0.0),
            'lineup confirmation': finite(context_components.get('lineup_quality'), 0.0) + finite(context_components.get('lineup_order'), 0.0),
            'interaction layer': finite(features.get('x_interaction'), 0.0),
        }
    elif market_name == 'run_line':
        driver_map = {
            'margin core': finite(prediction.get('margin_model'), 0.0),
            'starter edge': finite(features.get('x_starter'), 0.0),
            'bullpen edge': finite(features.get('x_bullpen'), 0.0),
            'short-start pressure': finite(context_components.get('short_start'), 0.0),
            'interaction layer': finite(features.get('x_interaction'), 0.0),
            'lineup order edge': finite(context_components.get('lineup_order'), 0.0) + finite(context_components.get('lineup_top_heavy'), 0.0),
        }
    else:
        away_workload = features.get('away_starter_workload', {}) or {}
        home_workload = features.get('home_starter_workload', {}) or {}
        driver_map = {
            'run environment': finite(features.get('shared_env'), 0.0),
            'lineup quality': finite(context_components.get('lineup_quality'), 0.0) + finite(context_components.get('lineup_top_heavy'), 0.0),
            'short-start pressure': finite(away_workload.get('short_start_risk'), 0.0) + finite(home_workload.get('short_start_risk'), 0.0),
            'times-through-order risk': finite(away_workload.get('tto_risk'), 0.0) + finite(home_workload.get('tto_risk'), 0.0),
            'bullpen fragility': max(0.0, -finite(features.get('away_bullpen'), 0.0)) + max(0.0, -finite(features.get('home_bullpen'), 0.0)),
            'weather/park total': finite((features.get('weather_adj', {}) or {}).get('total_adj'), 0.0) + finite((features.get('park_history', {}) or {}).get('total_adj'), 0.0),
        }
    ranked = sorted(driver_map.items(), key=lambda item: abs(finite(item[1], 0.0)), reverse=True)
    return ranked[:3]


def build_prediction_explainer(prediction, market_type):
    game = prediction.get('game') or {}
    features = prediction.get('features', {}) or {}
    lineups = features.get('lineups', {}) or {}
    away_workload = features.get('away_starter_workload', {}) or {}
    home_workload = features.get('home_starter_workload', {}) or {}
    away_bullpen_profile = features.get('away_bullpen_profile', {}) or {}
    home_bullpen_profile = features.get('home_bullpen_profile', {}) or {}
    drivers = ranked_market_drivers(prediction, market_type)
    technical_drivers = [f"{label} {finite(value, 0.0):+.2f}" for label, value in drivers]

    risk_flags = []
    if min(int(lineups.get('away', 0) or 0), int(lineups.get('home', 0) or 0)) < 9:
        risk_flags.append('lineups not fully confirmed')
    if max(finite(away_workload.get('short_start_risk'), 0.0), finite(home_workload.get('short_start_risk'), 0.0)) >= 0.55:
        risk_flags.append('short-start risk elevated')
    if max(finite(away_workload.get('tto_risk'), 0.0), finite(home_workload.get('tto_risk'), 0.0)) >= 0.45:
        risk_flags.append('starter times-through-order risk elevated')
    if min(finite(away_bullpen_profile.get('leverage_availability'), 0.0), finite(home_bullpen_profile.get('leverage_availability'), 0.0)) <= -0.25:
        risk_flags.append('late-inning bullpen availability is stressed')
    if str(market_type or '').lower() == 'total' and finite(prediction.get('total_sigma'), 4.0) >= 4.8:
        risk_flags.append('totals volatility is high')
    if not risk_flags:
        risk_flags.append('setup is comparatively clean')

    if str(market_type or '').lower() == 'moneyline':
        plain = (
            f"{prediction.get('winner')} is being carried mostly by "
            + ', '.join(label for label, _ in drivers[:2])
            + '.'
        )
    elif str(market_type or '').lower() == 'run_line':
        plain = (
            f"The spread case leans on margin pressure from "
            + ', '.join(label for label, _ in drivers[:2])
            + '.'
        )
    else:
        lean = ((prediction.get('market_comp') or {}).get('total') or {}).get('lean') or 'TOTAL'
        plain = f"{lean.title()} is being driven mainly by " + ', '.join(label for label, _ in drivers[:2]) + '.'

    return {
        'technical_drivers': technical_drivers,
        'driver_labels': [label for label, _ in drivers],
        'plain_english': plain,
        'risk_flags': risk_flags,
        'matchup_label': f"{game.get('away_team')} @ {game.get('home_team')}",
    }

def predict_game(game, phase, weights, state, strengths, current_pitchers, prior_pitchers, current_batters, prior_batters, bullpen_profiles, park_cache, client, venue_cache, margin_sigma, lineup_counts_override=None, umpire_cache=None):
    strength_index = strengths.set_index('team') if strengths is not None and not strengths.empty else pd.DataFrame().set_index(pd.Index([], name='team'))
    away_row = strength_index.loc[game['away_team']].to_dict() if game['away_team'] in strength_index.index else {}
    home_row = strength_index.loc[game['home_team']].to_dict() if game['home_team'] in strength_index.index else {}
    away_off_base = finite(away_row.get('offense_score_blended'), 0.0)
    home_off_base = finite(home_row.get('offense_score_blended'), 0.0)
    away_def = finite(away_row.get('defense_score_blended'), 0.0)
    home_def = finite(home_row.get('defense_score_blended'), 0.0)

    away_starter = resolve_pitcher_projection(game.get('away_pitcher'), game.get('away_pitcher_id'), current_pitchers, prior_pitchers)
    home_starter = resolve_pitcher_projection(game.get('home_pitcher'), game.get('home_pitcher_id'), current_pitchers, prior_pitchers)
    away_starter_workload = starter_workload_context(away_starter, phase)
    home_starter_workload = starter_workload_context(home_starter, phase)

    away_matchup = matchup_adjusted_offense(away_row, home_starter.get('hand'))
    home_matchup = matchup_adjusted_offense(home_row, away_starter.get('hand'))
    away_off = finite(away_matchup.get('score'), away_off_base)
    home_off = finite(home_matchup.get('score'), home_off_base)

    bullpen_profiles = bullpen_profiles or {}
    away_bullpen_profile = dict(bullpen_profiles.get(game['away_team']) or {})
    home_bullpen_profile = dict(bullpen_profiles.get(game['home_team']) or {})
    away_bullpen = finite(away_bullpen_profile.get('bullpen_score'), 0.0)
    home_bullpen = finite(home_bullpen_profile.get('bullpen_score'), 0.0)
    away_bullpen_availability = finite(away_bullpen_profile.get('availability_score'), 0.0)
    home_bullpen_availability = finite(home_bullpen_profile.get('availability_score'), 0.0)
    away_leverage_availability = finite(away_bullpen_profile.get('leverage_availability'), 0.0)
    home_leverage_availability = finite(home_bullpen_profile.get('leverage_availability'), 0.0)

    venue_meta = fetch_venue_metadata(client, game.get('venue_id'), game.get('venue_name'), venue_cache)
    park_history = park_history_adjustment(park_cache, game['scheduled_utc'].astimezone(LOCAL_TZ).date(), game.get('venue_id'), game.get('venue_name'))
    park_static = park_static_adjustment(venue_meta)
    weather = fetch_weather_context(client, game, venue_meta)
    weather_adj = weather_adjustments(weather)

    # Umpire zone adjustment — fetch HP ump from boxscore, look up run tendency from cache
    hp_umpire_name = fetch_umpire_for_game(client, game.get('game_pk'))
    ump_adj = umpire_total_adjustment(hp_umpire_name, umpire_cache or {})

    lineup_ctx = lineup_quality_context(game['game_pk'], away_row, home_row, current_batters, prior_batters, client, lineup_counts_override=lineup_counts_override)
    lineups = lineup_ctx.get('counts') or {'away': 0, 'home': 0}
    away_lineup_ctx = lineup_ctx.get('away') or {}
    home_lineup_ctx = lineup_ctx.get('home') or {}
    lineup_count_component = 0.04 * ((lineups.get('home', 0) - lineups.get('away', 0)) / 9.0)
    lineup_quality_component = 0.055 * finite(lineup_ctx.get('delta'), 0.0)
    lineup_order_component = 0.030 * finite(lineup_ctx.get('order_delta'), 0.0)
    lineup_top_heavy_component = 0.020 * finite(lineup_ctx.get('top_heavy_delta'), 0.0)
    x_lineup = lineup_count_component + lineup_quality_component + lineup_order_component + lineup_top_heavy_component
    lineup_multiplier = lineup_multiplier_from_state(state, phase)
    away_lineup_adj = finite(away_lineup_ctx.get('adj'), 0.0)
    home_lineup_adj = finite(home_lineup_ctx.get('adj'), 0.0)
    away_lineup_bonus = away_lineup_adj + (0.30 * finite(away_lineup_ctx.get('top_heavy'), 0.0)) + (0.18 * (finite(away_lineup_ctx.get('weighted_score'), away_off_base) - finite(away_lineup_ctx.get('avg_score'), away_off_base)))
    home_lineup_bonus = home_lineup_adj + (0.30 * finite(home_lineup_ctx.get('top_heavy'), 0.0)) + (0.18 * (finite(home_lineup_ctx.get('weighted_score'), home_off_base) - finite(home_lineup_ctx.get('avg_score'), home_off_base)))

    away_schedule = team_travel_rest_context(client, game['away_team'], game, venue_cache)
    home_schedule = team_travel_rest_context(client, game['home_team'], game, venue_cache)
    context_feature_mult = 0.0 if phase == 'spring' else 1.0
    schedule_component = context_feature_mult * (finite(home_schedule.get('score'), 0.0) - finite(away_schedule.get('score'), 0.0))
    defense_component = context_feature_mult * (0.09 * (home_def - away_def))

    # Injury / transaction wire: fetch recent IL moves and fold net impact into context
    target_date_local = game['scheduled_utc'].astimezone(LOCAL_TZ).date() if game.get('scheduled_utc') else today_local()
    away_team_id = game.get('away_team_id')
    home_team_id = game.get('home_team_id')
    away_injury_ctx = fetch_team_injury_transactions(client, away_team_id, target_date_local) if away_team_id else {'net_il_impact': 0.0}
    home_injury_ctx = fetch_team_injury_transactions(client, home_team_id, target_date_local) if home_team_id else {'net_il_impact': 0.0}
    # Positive net_il_impact = more activations (boost to that team's offense)
    # Incorporate as a small context adjustment scaled by context_feature_mult
    injury_component = context_feature_mult * (
        finite(home_injury_ctx.get('net_il_impact'), 0.0) - finite(away_injury_ctx.get('net_il_impact'), 0.0)
    )
    x_context_core = schedule_component + defense_component + injury_component

    x_off = home_off - away_off
    x_starter = (finite(home_starter.get('score'), 0.0) * finite(home_starter_workload.get('starter_share'), 0.55)) - (finite(away_starter.get('score'), 0.0) * finite(away_starter_workload.get('starter_share'), 0.55))
    x_bullpen = (home_bullpen * finite(home_starter_workload.get('bullpen_share'), 0.45)) - (away_bullpen * finite(away_starter_workload.get('bullpen_share'), 0.45))
    x_park = finite(park_history['home_edge_adj'], 0.0)
    x_weather = finite(weather_adj['home_edge_adj'], 0.0)
    x_short_start = finite(away_starter_workload.get('short_start_risk'), 0.0) - finite(home_starter_workload.get('short_start_risk'), 0.0)
    x_tto = finite(away_starter_workload.get('tto_risk'), 0.0) - finite(home_starter_workload.get('tto_risk'), 0.0)

    # Offense × starter quality interaction: captures non-additive compounding when a
    # strong offense faces a weak starter (or vice-versa).  The sign convention is:
    # positive = home team benefits (strong home offense vs weak away starter, or
    # home has elite starter facing poor away offense).
    home_starter_score = finite(home_starter.get('score'), 0.0) * finite(home_starter_workload.get('starter_share'), 0.55)
    away_starter_score = finite(away_starter.get('score'), 0.0) * finite(away_starter_workload.get('starter_share'), 0.55)
    # Home offense vs away starter: if home_off > 0 (strong offense) and away starter < 0 (weak), compound amplifies
    home_off_vs_away_sp = home_off * max(0.0, -away_starter_score)
    # Away offense vs home starter: if away_off > 0 and home starter < 0, compound benefits away team (hurts home)
    away_off_vs_home_sp = away_off * max(0.0, -home_starter_score)
    x_off_starter_cross = float(np.clip(home_off_vs_away_sp - away_off_vs_home_sp, -1.20, 1.20))

    home_bridge = (
        finite(home_starter_workload.get('short_start_risk'), 0.0) * max(0.0, -home_bullpen)
        + (0.35 * finite(home_starter_workload.get('short_start_risk'), 0.0) * max(0.0, -home_leverage_availability))
    )
    away_bridge = (
        finite(away_starter_workload.get('short_start_risk'), 0.0) * max(0.0, -away_bullpen)
        + (0.35 * finite(away_starter_workload.get('short_start_risk'), 0.0) * max(0.0, -away_leverage_availability))
    )
    offense_env_support = max(0.0, finite(park_history.get('total_adj'), 0.0) + finite(park_static.get('total_adj'), 0.0) + finite(weather_adj.get('total_adj'), 0.0))
    interaction_core = float(np.clip((0.55 * (away_bridge - home_bridge)) + (0.18 * ((finite(home_schedule.get('score'), 0.0) * max(home_off, 0.0)) - (finite(away_schedule.get('score'), 0.0) * max(away_off, 0.0)))) + (0.22 * x_short_start) + (0.16 * x_tto), -1.10, 1.10))
    interaction_lineup = float(np.clip(((finite(home_matchup.get('adjustment'), 0.0) + (0.65 * home_lineup_bonus)) - (finite(away_matchup.get('adjustment'), 0.0) + (0.65 * away_lineup_bonus))) + (0.12 * offense_env_support * (home_off - away_off)), -1.15, 1.15))
    x_interaction = interaction_core + (lineup_multiplier * interaction_lineup)
    x_context = x_context_core + (lineup_multiplier * x_lineup) + (0.42 * x_interaction)

    x_context_no_lineup = x_context_core + (0.42 * interaction_core)
    x_context_full_lineup = x_context_core + x_lineup + (0.42 * (interaction_core + interaction_lineup))

    # w_cross: weight for the offense×starter interaction term.  Loaded from state if available,
    # defaulting to 0.04 — small enough not to dominate but large enough to influence close games.
    w_cross = finite(weights.get('w_cross', 0.04), 0.04)
    logit_no_lineup = (weights['w_off'] * x_off) + (weights['w_starter'] * x_starter) + (weights['w_bullpen'] * x_bullpen) + (weights['w_park'] * x_park) + (weights['w_weather'] * x_weather) + (weights['w_context'] * x_context_no_lineup) + (w_cross * x_off_starter_cross) + weights['home_field']
    logit_full_lineup = (weights['w_off'] * x_off) + (weights['w_starter'] * x_starter) + (weights['w_bullpen'] * x_bullpen) + (weights['w_park'] * x_park) + (weights['w_weather'] * x_weather) + (weights['w_context'] * x_context_full_lineup) + (w_cross * x_off_starter_cross) + weights['home_field']
    logit = (weights['w_off'] * x_off) + (weights['w_starter'] * x_starter) + (weights['w_bullpen'] * x_bullpen) + (weights['w_park'] * x_park) + (weights['w_weather'] * x_weather) + (weights['w_context'] * x_context) + (w_cross * x_off_starter_cross) + weights['home_field']

    p_home_raw_no_lineup = sigmoid(logit_no_lineup)
    p_home_raw_full_lineup = sigmoid(logit_full_lineup)
    p_home_raw = sigmoid(logit)
    p_home_no_lineup = apply_probability_calibration(p_home_raw_no_lineup, state, phase)
    p_home_full_lineup = apply_probability_calibration(p_home_raw_full_lineup, state, phase)
    p_home_model = apply_probability_calibration(p_home_raw, state, phase)
    target_date = game['scheduled_utc'].astimezone(LOCAL_TZ).date() if game.get('scheduled_utc') is not None else today_local()
    p_home, p_home_simple, probability_shrink_alpha = blend_probability_with_simple_baseline(p_home_model, x_off, phase, state, target_date)
    p_home_no_lineup, _, _ = blend_probability_with_simple_baseline(p_home_no_lineup, x_off, phase, state, target_date)
    p_home_full_lineup, _, _ = blend_probability_with_simple_baseline(p_home_full_lineup, x_off, phase, state, target_date)

    total_params = total_model_params(phase)
    starter_uncertainty = (1.0 - finite(away_starter.get('confidence'), 0.5)) + (1.0 - finite(home_starter.get('confidence'), 0.5))
    shared_env = max(
        0.0,
        (finite(total_params.get('shared_env_offense_coef'), 0.06) * (abs(away_off) + abs(home_off)))
        + (finite(total_params.get('shared_env_support_coef'), 0.05) * offense_env_support * (max(0.0, away_off) + max(0.0, home_off)))
        + (finite(total_params.get('shared_env_starter_coef'), 0.09) * max(0.0, -((finite(away_starter.get('score'), 0.0) * finite(away_starter_workload.get('starter_share'), 0.55)) + (finite(home_starter.get('score'), 0.0) * finite(home_starter_workload.get('starter_share'), 0.55)))))
        + (finite(total_params.get('shared_env_bullpen_coef'), 0.05) * max(0.0, -((away_bullpen * finite(away_starter_workload.get('bullpen_share'), 0.45)) + (home_bullpen * finite(home_starter_workload.get('bullpen_share'), 0.45)))))
        + (finite(total_params.get('shared_env_short_coef'), 0.14) * (finite(away_starter_workload.get('short_start_risk'), 0.0) + finite(home_starter_workload.get('short_start_risk'), 0.0)))
        + (0.10 * (finite(away_starter_workload.get('tto_risk'), 0.0) + finite(home_starter_workload.get('tto_risk'), 0.0)))
        + (finite(total_params.get('shared_env_bridge_coef'), 0.09) * (away_bridge + home_bridge))
        + (finite(total_params.get('shared_env_uncertainty_coef'), 0.06) * starter_uncertainty)
    )
    shared_env = float(np.clip(shared_env, 0.0, finite(total_params.get('env_cap'), 1.0)))
    shared_total_adj = finite(park_history['total_adj'], 0.0) + finite(park_static['total_adj'], 0.0) + finite(weather_adj['total_adj'], 0.0) + finite(ump_adj.get('total_adj'), 0.0) + shared_env
    base = finite(total_params.get('base_runs'), LEAGUE_BASE_RUNS.get(phase, 4.6))
    away_runs_raw = base + (finite(total_params.get('offense_coef'), 0.65) * away_off) - (finite(total_params.get('starter_coef'), 0.84) * finite(home_starter.get('score'), 0.0) * finite(home_starter_workload.get('starter_share'), 0.55)) - (finite(total_params.get('bullpen_coef'), 0.35) * home_bullpen * finite(home_starter_workload.get('bullpen_share'), 0.45)) + (finite(total_params.get('shared_total_coef'), 0.38) * shared_total_adj) + (finite(total_params.get('schedule_coef'), 0.05) * context_feature_mult * finite(away_schedule.get('score'), 0.0)) - (finite(total_params.get('defense_coef'), 0.09) * context_feature_mult * home_def) + (finite(total_params.get('lineup_coef'), 0.11) * lineup_multiplier * away_lineup_bonus) + (finite(total_params.get('bridge_coef'), 0.09) * home_bridge) + (0.06 * finite(home_starter_workload.get('tto_risk'), 0.0)) - (finite(total_params.get('availability_coef'), 0.03) * max(0.0, home_bullpen_availability) * finite(home_starter_workload.get('bullpen_share'), 0.45)) - (0.02 * max(0.0, home_leverage_availability))
    home_runs_raw = base + (finite(total_params.get('offense_coef'), 0.65) * home_off) - (finite(total_params.get('starter_coef'), 0.84) * finite(away_starter.get('score'), 0.0) * finite(away_starter_workload.get('starter_share'), 0.55)) - (finite(total_params.get('bullpen_coef'), 0.35) * away_bullpen * finite(away_starter_workload.get('bullpen_share'), 0.45)) + (finite(total_params.get('shared_total_coef'), 0.38) * shared_total_adj) + (finite(total_params.get('schedule_coef'), 0.05) * context_feature_mult * finite(home_schedule.get('score'), 0.0)) - (finite(total_params.get('defense_coef'), 0.09) * context_feature_mult * away_def) + (finite(total_params.get('lineup_coef'), 0.11) * lineup_multiplier * home_lineup_bonus) + (finite(total_params.get('bridge_coef'), 0.09) * away_bridge) + (0.06 * finite(away_starter_workload.get('tto_risk'), 0.0)) - (finite(total_params.get('availability_coef'), 0.03) * max(0.0, away_bullpen_availability) * finite(away_starter_workload.get('bullpen_share'), 0.45)) - (0.02 * max(0.0, away_leverage_availability))
    away_runs_raw = float(np.clip(away_runs_raw, finite(total_params.get('run_floor'), 1.8), finite(total_params.get('run_cap'), 8.2)))
    home_runs_raw = float(np.clip(home_runs_raw, finite(total_params.get('run_floor'), 1.8), finite(total_params.get('run_cap'), 8.2)))

    total_raw = away_runs_raw + home_runs_raw
    short_start_total = finite(away_starter_workload.get('short_start_risk'), 0.0) + finite(home_starter_workload.get('short_start_risk'), 0.0)
    bullpen_fragility = (
        (max(0.0, -away_bullpen) * finite(away_starter_workload.get('bullpen_share'), 0.45))
        + (max(0.0, -home_bullpen) * finite(home_starter_workload.get('bullpen_share'), 0.45))
        + (0.35 * max(0.0, -away_bullpen_availability))
        + (0.35 * max(0.0, -home_bullpen_availability))
    )
    lineup_missing = max(0.0, (18.0 - float((lineups.get('away', 0) or 0) + (lineups.get('home', 0) or 0))) / 9.0)
    offense_pressure = max(0.0, away_off) + max(0.0, home_off)
    total_sigma_raw = estimate_total_sigma(
        phase,
        shared_env,
        starter_uncertainty,
        short_start_total,
        bullpen_fragility,
        lineup_missing,
        finite(weather_adj.get('total_adj'), 0.0),
        offense_pressure,
    )
    total_sigma = apply_total_sigma_calibration(total_sigma_raw, state, phase)
    total_model = apply_total_calibration(total_raw, state, phase)
    total_calibrated, total_simple, total_shrink_alpha = blend_total_with_baseline(total_model, phase, state, target_date)
    scale = total_calibrated / total_raw if total_raw > 0 else 1.0
    away_runs = away_runs_raw * scale
    home_runs = home_runs_raw * scale
    margin_raw = home_runs - away_runs
    margin_model = apply_margin_calibration(margin_raw, state, phase)
    margin_calibrated, margin_simple, margin_shrink_alpha = blend_margin_with_baseline(margin_model, x_off, phase, state, target_date)
    winner = game['home_team'] if p_home >= 0.5 else game['away_team']
    winner_prob = p_home if winner == game['home_team'] else 1.0 - p_home
    return {
        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple, 'probability_shrink_alpha': probability_shrink_alpha, 'p_home_no_lineup': p_home_no_lineup, 'p_home_full_lineup': p_home_full_lineup, 'lineup_multiplier': lineup_multiplier, 'total_raw': total_raw, 'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma_raw': total_sigma_raw, 'total_sigma': total_sigma, 'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
        'fair_home_ml': american_from_probability(p_home), 'fair_away_ml': american_from_probability(1.0 - p_home),
        'away_runs_raw': away_runs_raw, 'home_runs_raw': home_runs_raw, 'away_runs': away_runs, 'home_runs': home_runs, 'total_calibrated': total_calibrated, 'total_bet_line': round_to_half(total_calibrated),
        'margin_calibrated': margin_calibrated, 'run_line_prices': fair_run_line_prices(margin_calibrated, margin_sigma), 'confidence_tier': confidence_tier(winner_prob, phase),
        'features': {
            'away_row': away_row, 'home_row': home_row, 'away_starter': away_starter, 'home_starter': home_starter,
            'away_starter_workload': away_starter_workload, 'home_starter_workload': home_starter_workload,
            'away_bullpen': away_bullpen, 'home_bullpen': home_bullpen, 'away_bullpen_profile': away_bullpen_profile, 'home_bullpen_profile': home_bullpen_profile,
            'park_history': park_history, 'park_static': park_static, 'weather': weather, 'weather_adj': weather_adj,
            'lineups': lineups, 'lineup_ctx': lineup_ctx, 'shared_env': shared_env, 'lineup_multiplier': lineup_multiplier,
            'probability_shrink_alpha': probability_shrink_alpha, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple,
            'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma_raw': total_sigma_raw, 'total_sigma': total_sigma,
            'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
            'away_off_base': away_off_base, 'home_off_base': home_off_base, 'away_off_matchup': away_matchup, 'home_off_matchup': home_matchup,
            'away_defense': away_def, 'home_defense': home_def, 'away_schedule': away_schedule, 'home_schedule': home_schedule,
            'away_injury_ctx': away_injury_ctx, 'home_injury_ctx': home_injury_ctx,
            'context_components': {
                'lineup_count': lineup_count_component,
                'lineup_quality': lineup_quality_component,
                'lineup_order': lineup_order_component,
                'lineup_top_heavy': lineup_top_heavy_component,
                'schedule': schedule_component,
                'defense': defense_component,
                'short_start': x_short_start,
                'tto_risk': x_tto,
                'interaction_core': interaction_core,
                'interaction_lineup': interaction_lineup,
            },
            'x_lineup': x_lineup, 'x_context_core': x_context_core, 'x_interaction': x_interaction,
            'x_off': x_off, 'x_starter': x_starter, 'x_bullpen': x_bullpen, 'x_park': x_park, 'x_weather': x_weather, 'x_context': x_context, 'x_off_starter_cross': x_off_starter_cross, 'venue_meta': venue_meta,
            'umpire': ump_adj,
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
    game = prediction.get('game') or {}
    game_date = game.get('scheduled_utc')
    if isinstance(game_date, dt.datetime):
        report_date = game_date.astimezone(LOCAL_TZ).date()
    else:
        report_date = today_local()
    market_profile = market.recent_market_actionability_profile(market_type, report_date=report_date, model_phase=phase)
    thresholds = market.market_thresholds(market_type, sportsbook_count=sportsbook_count, model_phase=phase)
    threshold_scale = finite(market_profile.get('threshold_scale'), 1.0)
    thresholds = {key: finite(value, 0.0) * threshold_scale for key, value in thresholds.items()}
    magnitude = abs(finite(edge_value, 0.0))
    high = max(finite(thresholds.get('high'), 1.0), 1e-6)
    score = 22.0 + (58.0 * min(magnitude / high, 1.25))
    notes = []
    market_name = str(market_type or '').lower()
    phase_name = str(phase or '').lower()

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
    away_workload = features.get('away_starter_workload', {}) or {}
    home_workload = features.get('home_starter_workload', {}) or {}
    away_bullpen_profile = features.get('away_bullpen_profile', {}) or {}
    home_bullpen_profile = features.get('home_bullpen_profile', {}) or {}
    starter_conf = (finite(away_starter.get('confidence'), 0.5) + finite(home_starter.get('confidence'), 0.5)) / 2.0
    short_start_risk = max(
        finite(away_workload.get('short_start_risk'), 0.0),
        finite(home_workload.get('short_start_risk'), 0.0),
    )
    tto_risk = max(
        finite(away_workload.get('tto_risk'), 0.0),
        finite(home_workload.get('tto_risk'), 0.0),
    )
    bullpen_stress = max(
        finite(away_bullpen_profile.get('stress_score'), 0.0),
        finite(home_bullpen_profile.get('stress_score'), 0.0),
    )
    leverage_availability = min(
        finite(away_bullpen_profile.get('leverage_availability'), 0.0),
        finite(home_bullpen_profile.get('leverage_availability'), 0.0),
    )
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

    if phase_name == 'spring':
        score -= 6.0
        notes.append('spring caution')

    if market_name == 'total':
        score -= 12.0
        notes.append('totals proof still maturing')
        total_sigma = finite(prediction.get('total_sigma'), 4.0)
        throttle_label = str(market_profile.get('throttle_label') or '').strip().lower()
        if throttle_label == 'heavy throttle':
            score -= 8.0
            notes.append('totals market throttle active')
        elif throttle_label in {'soft throttle', 'firm throttle'}:
            score -= 4.0
        if total_sigma >= 4.8:
            score -= 10.0
            notes.append('high total volatility')
        elif total_sigma >= 4.4:
            score -= 5.0
            notes.append('elevated total volatility')
        if lineup_min < 9:
            score -= 8.0
            notes.append('lineups not fully posted')
        if starter_conf < 0.60:
            score -= 8.0
            notes.append('starter signal still soft')
        elif starter_conf < 0.72:
            score -= 4.0
        if short_start_risk >= 0.58:
            score -= 6.0
            notes.append('short-start risk elevated')
        elif short_start_risk >= 0.48:
            score -= 3.0
        if tto_risk >= 0.48:
            score -= 5.0
            notes.append('times-through-order pressure elevated')
        elif tto_risk >= 0.40:
            score -= 2.0
        if bullpen_stress >= 0.38 or leverage_availability <= -0.28:
            score -= 6.0
            notes.append('bullpen support strained')
        elif bullpen_stress >= 0.30 or leverage_availability <= -0.18:
            score -= 3.0
        if weather_source == 'neutral':
            score -= 4.0
            notes.append('weather edge not confirmed')
        if magnitude < max(finite(thresholds.get('bet'), 1.0), 1.55):
            score -= 8.0
            notes.append('totals edge below strong threshold')
        elif magnitude < max(finite(thresholds.get('high'), 1.0), 1.95):
            score -= 4.0
    elif market_name == 'run_line':
        score -= 6.0
        notes.append('run-line proof still mixed')
        throttle_label = str(market_profile.get('throttle_label') or '').strip().lower()
        if throttle_label == 'firm throttle':
            score -= 6.0
            notes.append('run-line market throttle active')
        elif throttle_label == 'soft throttle':
            score -= 3.0
        if starter_conf < 0.60:
            score -= 8.0
            notes.append('starter signal still soft')
        elif starter_conf < 0.70:
            score -= 4.0
        if short_start_risk >= 0.58:
            score -= 8.0
            notes.append('short-start risk elevated')
        elif short_start_risk >= 0.48:
            score -= 4.0
        if tto_risk >= 0.48:
            score -= 6.0
            notes.append('times-through-order pressure elevated')
        elif tto_risk >= 0.40:
            score -= 3.0
        if bullpen_stress >= 0.38:
            score -= 7.0
            notes.append('bullpen stress elevated')
        elif bullpen_stress >= 0.30:
            score -= 3.0
        if leverage_availability <= -0.28:
            score -= 5.0
            notes.append('late-inning leverage thin')
        elif leverage_availability <= -0.18:
            score -= 2.0
        if lineup_min < 9:
            score -= 4.0
            notes.append('lineups not fully posted')
        if magnitude < max(finite(thresholds.get('bet'), 0.0), 0.050):
            score -= 5.0
            notes.append('run-line edge below strong threshold')
        elif magnitude < max(finite(thresholds.get('high'), 0.0), 0.065):
            score -= 2.0

    score = float(np.clip(score, 0.0, 100.0))
    if magnitude >= finite(thresholds.get('high'), 1.0):
        label = 'MUST TAKE'
    elif magnitude >= finite(thresholds.get('bet'), 1.0):
        label = 'BET'
    elif magnitude >= finite(thresholds.get('watch'), 1.0):
        label = 'WATCH'
    else:
        label = 'PASS'
    cap_order = {'PASS': 0, 'WATCH': 1, 'BET': 2, 'MUST TAKE': 3}
    if finite(market_profile.get('threshold_scale'), 1.0) > 1.0:
        notes.append(str(market_profile.get('notes') or '').strip())
    if label != 'PASS':
        if score < 42.0:
            label = 'PASS'
        elif label == 'MUST TAKE' and score < 72.0:
            label = 'BET'
        elif label in ['MUST TAKE', 'BET'] and score < 56.0:
            label = 'WATCH'
    if market_name == 'total' and phase_name == 'regular' and label != 'PASS':
        confidence_tier_name = str(prediction.get('confidence_tier') or '').strip().lower()
        if lineup_min < 9 or starter_conf < 0.60 or total_sigma >= 4.8:
            label = 'WATCH' if score >= 56.0 else 'PASS'
        elif confidence_tier_name == 'low':
            if label == 'MUST TAKE':
                label = 'BET'
            if magnitude < max(finite(thresholds.get('high'), 1.0), 2.25) or score < 84.0:
                label = 'WATCH' if score >= 56.0 else 'PASS'
        elif confidence_tier_name == 'medium' and label == 'MUST TAKE' and score < 88.0:
            label = 'BET'
    if market_name == 'run_line' and phase_name == 'regular' and label != 'PASS':
        confidence_tier_name = str(prediction.get('confidence_tier') or '').strip().lower()
        fragile_run_line = (
            starter_conf < 0.60
            or short_start_risk >= 0.58
            or tto_risk >= 0.48
            or bullpen_stress >= 0.38
            or leverage_availability <= -0.28
        )
        if fragile_run_line:
            if label == 'MUST TAKE':
                label = 'BET'
            if magnitude < max(finite(thresholds.get('high'), 0.0), 0.070) or score < 72.0:
                label = 'WATCH' if score >= 56.0 else 'PASS'
        elif confidence_tier_name == 'low':
            if label == 'MUST TAKE':
                label = 'BET'
            if magnitude < max(finite(thresholds.get('high'), 0.0), 0.075) or score < 80.0:
                label = 'WATCH' if score >= 56.0 else 'PASS'
        elif confidence_tier_name == 'medium' and label == 'MUST TAKE' and score < 84.0:
            label = 'BET'
    capped_label = str(market_profile.get('action_cap') or 'MUST TAKE').upper()
    if cap_order.get(label, 0) > cap_order.get(capped_label, 3):
        label = capped_label
        notes.append(f"market throttle: capped at {capped_label}")
    note_text = '; '.join(dict.fromkeys([note for note in notes if note][:3])) or 'edge-driven signal'
    return {
        'label': label,
        'score': score,
        'notes': note_text,
        'market_profile': market_profile,
    }


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
            'explainer': build_prediction_explainer(prediction, 'moneyline'),
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
            'explainer': build_prediction_explainer(prediction, 'run_line'),
        }

    market_total = market.parse_market_float(market_row.get('total_line'))
    over_price = market.parse_market_int(market_row.get('over_price'))
    under_price = market.parse_market_int(market_row.get('under_price'))
    if market_total is not None:
        diff = float(prediction['total_calibrated'] - market_total)
        total_sigma = float(np.clip(finite(prediction.get('total_sigma'), 4.0), 2.0, 7.0))
        over_model_probability = total_side_probability('OVER', market_total, prediction['total_calibrated'], total_sigma)
        under_model_probability = total_side_probability('UNDER', market_total, prediction['total_calibrated'], total_sigma)
        market_over_probability, market_under_probability = market.no_vig_probabilities(over_price, under_price)
        lean = 'OVER' if diff > 0.05 else ('UNDER' if diff < -0.05 else 'FLAT')
        best_price = over_price if lean == 'OVER' else (under_price if lean == 'UNDER' else None)
        model_probability = over_model_probability if lean == 'OVER' else (under_model_probability if lean == 'UNDER' else None)
        market_probability = None
        fair_price = american_from_probability(model_probability) if model_probability is not None and lean in ['OVER', 'UNDER'] else None
        best_edge = None
        if market_over_probability is not None and market_under_probability is not None:
            over_edge = float(over_model_probability - market_over_probability)
            under_edge = float(under_model_probability - market_under_probability)
            if max(over_edge, under_edge) >= 0.008:
                lean = 'OVER' if over_edge >= under_edge else 'UNDER'
            if lean == 'OVER':
                best_price = over_price
                model_probability = over_model_probability
                market_probability = float(market_over_probability)
                best_edge = float(over_edge)
            elif lean == 'UNDER':
                best_price = under_price
                model_probability = under_model_probability
                market_probability = float(market_under_probability)
                best_edge = float(under_edge)
            else:
                best_edge = float(max(over_edge, under_edge))
            fair_price = american_from_probability(model_probability) if model_probability is not None and lean in ['OVER', 'UNDER'] else None
        action = actionability_assessment('total', diff if lean != 'FLAT' else 0.0, out['sportsbook_count'], prediction, phase)
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
            'best_edge': best_edge,
            'model_probability': model_probability,
            'market_probability': market_probability,
            'fair_price': fair_price,
            'total_sigma': total_sigma,
            'over_model_probability': over_model_probability,
            'under_model_probability': under_model_probability,
            'over_market_probability': float(market_over_probability) if market_over_probability is not None else None,
            'under_market_probability': float(market_under_probability) if market_under_probability is not None else None,
            'action': action,
            'explainer': build_prediction_explainer(prediction, 'total'),
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
                created_ts, report_date.isoformat(), str(phase), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
                'moneyline', str(ml.get('best_side')), None, int(ml.get('best_price')), int(ml.get('fair_price')),
                None, float(ml.get('model_probability')), float(ml.get('market_probability')), float(ml.get('best_edge')),
                'probability', str(prediction.get('confidence_tier', 'Low')), str(ml_action.get('label')), float(ml_action.get('score')),
                str(ml_action.get('notes')), 'watch', None, 'clean_runner_market_seed', None, None, None,
            ))

        rl = market_comp.get('run_line', {}) or {}
        rl_action = rl.get('action', {}) or {}
        if rl.get('available') and rl_action.get('label') in ['WATCH', 'BET', 'MUST TAKE'] and float(rl.get('best_edge', 0.0)) > 0:
            records.append((
                created_ts, report_date.isoformat(), str(phase), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
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
                created_ts, report_date.isoformat(), str(phase), int(game.get('game_pk') or 0), str(game['away_team']), str(game['home_team']), sportsbook_label,
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
                    created_ts, report_date, phase, game_pk, away_team, home_team, sportsbook,
                    market_type, selection, line_value, book_price, model_price, model_line_value,
                    model_probability, market_probability, edge_value, edge_kind, confidence_tier,
                    actionability_label, actionability_score, actionability_notes, status, stake_units,
                    notes, result_units, clv_value, settled_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
    return len(records), []



def validation_oos_slice(backtest_df):
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()
    work = backtest_df.copy()
    work['date'] = pd.to_datetime(work.get('date'), errors='coerce').dt.date
    work = work[work['date'].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=work.columns)
    unique_dates = sorted(work['date'].unique())
    holdout = max(1, int(round(len(unique_dates) * 0.4)))
    return work[work['date'].isin(set(unique_dates[-holdout:]))].copy()


def probability_metrics_frame(df, prob_col, target_col='y_home'):
    out = {'games': 0, 'log_loss': None, 'accuracy': None}
    if df is None or df.empty or prob_col not in df.columns or target_col not in df.columns:
        return out
    y = pd.to_numeric(df.get(target_col), errors='coerce')
    p = pd.to_numeric(df.get(prob_col), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    valid = (~y.isna()) & (~p.isna())
    if not valid.any():
        return out
    yv = y[valid].astype(int)
    pv = p[valid].astype(float)
    out['games'] = int(len(yv))
    out['log_loss'] = float((-(yv * np.log(pv)) - ((1 - yv) * np.log(1.0 - pv))).mean())
    out['accuracy'] = float(np.mean((pv >= 0.5).astype(int) == yv))
    return out

def regression_metrics_series(actual, predicted):
    out = {'games': 0, 'mae': None, 'rmse': None}
    actual_series = pd.to_numeric(pd.Series(actual), errors='coerce').reset_index(drop=True)
    predicted_series = pd.to_numeric(pd.Series(predicted), errors='coerce').reset_index(drop=True)
    valid = (~actual_series.isna()) & (~predicted_series.isna())
    if not valid.any():
        return out
    actual_values = actual_series[valid].astype(float).reset_index(drop=True)
    predicted_values = predicted_series[valid].astype(float).reset_index(drop=True)
    out['games'] = int(len(actual_values))
    out['mae'] = _mae_metric(actual_values, predicted_values)
    out['rmse'] = _rmse_metric(actual_values, predicted_values)
    return out
def simple_power_baseline_probabilities(df, state):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    work = df.copy()
    phase_series = work['phase'].astype(str).str.lower() if 'phase' in work.columns else pd.Series(['spring'] * len(work), index=work.index)
    spring_simple = (((state.get('spring_guardrail') or {}).get('simple_power_weights')) or {}) if isinstance(state, dict) else {}
    regular_simple = (((state.get('regular_earn_back') or {}).get('adjusted_weights')) or {}) if isinstance(state, dict) else {}
    probs = []
    for idx, row in work.iterrows():
        phase_name = str(phase_series.loc[idx] if idx in phase_series.index else 'spring').lower()
        source = spring_simple if phase_name == 'spring' else regular_simple
        w_off = finite(source.get('w_off'), 0.35)
        hfa = finite(source.get('home_field'), 0.05)
        x_off = finite(row.get('x_off'), 0.0)
        probs.append(sigmoid((w_off * x_off) + hfa))
    return pd.Series(probs, index=work.index, dtype=float)


def build_benchmark_ladder(state, backtest_df, archive_df=None):
    ladder = {
        'model_oos': {'games': 0, 'log_loss': None, 'accuracy': None},
        'simple_power_oos': {'games': 0, 'log_loss': None, 'accuracy': None},
        'model_total_oos': {'games': 0, 'mae': None, 'rmse': None},
        'simple_total_oos': {'games': 0, 'mae': None, 'rmse': None},
        'shrunk_total_oos': {'games': 0, 'mae': None, 'rmse': None},
        'model_margin_oos': {'games': 0, 'mae': None, 'rmse': None},
        'simple_margin_oos': {'games': 0, 'mae': None, 'rmse': None},
        'shrunk_margin_oos': {'games': 0, 'mae': None, 'rmse': None},
        'regular_lineup_shadow': {'games': 0, 'no_lineup_log_loss': None, 'full_lineup_log_loss': None, 'improvement': None, 'date_span': 'N/A'},
        'market_baseline': {'games': 0, 'log_loss': None, 'accuracy': None, 'available': False},
    }
    oos = validation_oos_slice(backtest_df)
    if not oos.empty:
        ladder['model_oos'] = probability_metrics_frame(oos, 'p_home')
        simple_probs = simple_power_baseline_probabilities(oos, state)
        if len(simple_probs) == len(oos):
            simple_work = oos.copy()
            simple_work['simple_power_prob'] = simple_probs
            ladder['simple_power_oos'] = probability_metrics_frame(simple_work, 'simple_power_prob')
        actual_total = pd.to_numeric(oos.get('home_score'), errors='coerce') + pd.to_numeric(oos.get('away_score'), errors='coerce')
        model_total = pd.to_numeric(oos.get('projected_total'), errors='coerce')
        actual_margin = pd.to_numeric(oos.get('home_score'), errors='coerce') - pd.to_numeric(oos.get('away_score'), errors='coerce')
        model_margin = pd.to_numeric(oos.get('projected_margin'), errors='coerce')
        simple_total_values = []
        shrunk_total_values = []
        simple_margin_values = []
        shrunk_margin_values = []
        for _, row in oos.iterrows():
            phase_name = str(row.get('phase') or 'spring').lower()
            row_date = pd.to_datetime(row.get('date'), errors='coerce')
            target_date = None if pd.isna(row_date) else row_date.date()
            total_model_value = finite(row.get('projected_total'), None)
            margin_model_value = finite(row.get('projected_margin'), None)
            x_off = finite(row.get('x_off'), 0.0)
            total_simple_value = simple_total_baseline_from_phase(phase_name, state)
            margin_simple_value = simple_margin_baseline_from_feature(x_off, phase_name, state)
            total_blended, _, _ = blend_total_with_baseline(total_model_value, phase_name, state, target_date)
            margin_blended, _, _ = blend_margin_with_baseline(margin_model_value, x_off, phase_name, state, target_date)
            simple_total_values.append(total_simple_value)
            shrunk_total_values.append(total_blended)
            simple_margin_values.append(margin_simple_value)
            shrunk_margin_values.append(margin_blended)
        ladder['model_total_oos'] = regression_metrics_series(actual_total, model_total)
        ladder['simple_total_oos'] = regression_metrics_series(actual_total, simple_total_values)
        ladder['shrunk_total_oos'] = regression_metrics_series(actual_total, shrunk_total_values)
        ladder['model_margin_oos'] = regression_metrics_series(actual_margin, model_margin)
        ladder['simple_margin_oos'] = regression_metrics_series(actual_margin, simple_margin_values)
        ladder['shrunk_margin_oos'] = regression_metrics_series(actual_margin, shrunk_margin_values)
    if archive_df is not None and not archive_df.empty:
        work = archive_df.copy()
        if 'phase' in work.columns:
            work = work[work['phase'].astype(str).str.lower() == 'regular'].copy()
        for column in ['away_lineup_count', 'home_lineup_count', 'y_home', 'p_home_no_lineup', 'p_home_full_lineup']:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors='coerce')
        if 'report_date' in work.columns:
            work['report_date'] = pd.to_datetime(work['report_date'], errors='coerce').dt.date
        confirmed = work[
            (work.get('away_lineup_count', 0) >= LINEUP_EARN_BACK_MIN_CONFIRMED) &
            (work.get('home_lineup_count', 0) >= LINEUP_EARN_BACK_MIN_CONFIRMED) &
            work.get('y_home').notna() &
            work.get('p_home_no_lineup').notna() &
            work.get('p_home_full_lineup').notna()
        ].copy() if not work.empty else pd.DataFrame()
        if not confirmed.empty:
            no_lineup = probability_metrics_frame(confirmed, 'p_home_no_lineup')
            full_lineup = probability_metrics_frame(confirmed, 'p_home_full_lineup')
            dates = sorted(confirmed['report_date'].dropna().unique().tolist()) if 'report_date' in confirmed.columns else []
            ladder['regular_lineup_shadow'] = {
                'games': int(len(confirmed)),
                'no_lineup_log_loss': no_lineup.get('log_loss'),
                'full_lineup_log_loss': full_lineup.get('log_loss'),
                'improvement': None if (no_lineup.get('log_loss') is None or full_lineup.get('log_loss') is None) else float(no_lineup['log_loss'] - full_lineup['log_loss']),
                'date_span': f"{dates[0].isoformat()} to {dates[-1].isoformat()}" if dates else 'N/A',
            }
    return ladder
def build_market_proof_summary():
    import market_helpers as market

    summary = {
        'available': False,
        'settled_bets': 0,
        'units': None,
        'roi': None,
        'avg_clv': None,
        'by_market': [],
        'by_actionability': [],
    }
    if not os.path.exists(market.SPORTSBOOK_DB_PATH):
        return summary
    try:
        with sqlite3.connect(market.SPORTSBOOK_DB_PATH) as conn:
            journal = pd.read_sql_query('SELECT * FROM bet_journal', conn)
    except Exception:
        return summary
    if journal.empty:
        return summary
    settled = journal[pd.to_numeric(journal.get('result_units'), errors='coerce').notna()].copy()
    if settled.empty:
        return summary
    settled['result_units'] = pd.to_numeric(settled['result_units'], errors='coerce')
    settled['stake_units'] = pd.to_numeric(settled.get('stake_units'), errors='coerce').fillna(1.0)
    settled['clv_value'] = pd.to_numeric(settled.get('clv_value'), errors='coerce')
    total_stake = float(settled['stake_units'].sum())
    total_units = float(settled['result_units'].sum())
    summary.update({
        'available': True,
        'settled_bets': int(len(settled)),
        'units': total_units,
        'roi': (total_units / total_stake) if total_stake > 0 else None,
        'avg_clv': float(settled['clv_value'].dropna().mean()) if settled['clv_value'].notna().any() else None,
    })
    by_market = []
    for market_type, grp in settled.groupby(settled.get('market_type').fillna('unknown')):
        stake = float(grp['stake_units'].sum())
        units = float(grp['result_units'].sum())
        by_market.append({
            'market_type': str(market_type),
            'bets': int(len(grp)),
            'units': units,
            'roi': (units / stake) if stake > 0 else None,
            'avg_clv': float(grp['clv_value'].dropna().mean()) if grp['clv_value'].notna().any() else None,
        })
    by_actionability = []
    action_series = settled.get('actionability_label').fillna('unlabeled')
    for label, grp in settled.groupby(action_series):
        stake = float(grp['stake_units'].sum())
        units = float(grp['result_units'].sum())
        by_actionability.append({
            'label': str(label),
            'bets': int(len(grp)),
            'units': units,
            'roi': (units / stake) if stake > 0 else None,
            'avg_clv': float(grp['clv_value'].dropna().mean()) if grp['clv_value'].notna().any() else None,
        })
    summary['by_market'] = sorted(by_market, key=lambda item: item['market_type'])
    summary['by_actionability'] = sorted(by_actionability, key=lambda item: item['label'])
    return summary
def write_report(report_date, phase, weights, state, predictions, validation, warnings):
    benchmark_ladder = build_benchmark_ladder(state, load_validation_log(), load_live_prediction_archive())
    market_proof = build_market_proof_summary()
    live_archive = load_live_prediction_archive()
    live_validation = build_live_validation_summary(live_archive, phase_filter=phase)
    live_calibration_surfaces = build_live_regular_calibration_surfaces(live_archive, phase_filter=phase)
    postgame_attribution = build_postgame_attribution_summary(live_archive, phase_filter=phase)
    model_change_log = build_model_change_log(report_date, state, predictions, live_validation, market_proof)
    path = os.path.join(OUT_DIR, f"MLB_Report_{report_date.isoformat()}.txt")
    spring_live_enabled = str(os.getenv("ODDS_API_ALLOW_SPRING_FETCH") or "").strip().lower() in ["1", "true", "yes", "on"]
    if str(phase).lower() == 'spring' and not spring_live_enabled:
        operating_mode = 'Operating mode: Manual spring market mode | report runs normally, but live Odds API line capture is disabled for spring.'
    elif str(phase).lower() == 'spring':
        operating_mode = 'Operating mode: Live spring market mode | report and live Odds API spring capture are both enabled.'
    else:
        operating_mode = 'Operating mode: Live regular-season mode | report and scheduled live Odds API capture are enabled.'
    lines = [f"MLB PREDICTIVE REPORT - {report_date.isoformat()}", 'Engine: clean parallel rebuild', 'Source policy: MLB StatsAPI schedule/feed, Baseball Savant Statcast CSV, park cache, and Open-Meteo forecast fallback.', operating_mode, '']
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
    probability_profile, total_profile, total_sigma_profile, lineup_profile = ((state.get('probability_calibration') or {}).get(phase) or {}), ((state.get('total_calibration') or {}).get(phase) or {}), ((state.get('total_sigma_calibration') or {}).get(phase) or {}), (state.get('lineup_earn_back') or {})
    if probability_profile.get('apply'):
        lines.append(f"Probability calibration: {probability_profile.get('mode', 'unknown')} | slope {finite(probability_profile.get('slope'), 1.0):.3f} | intercept {finite(probability_profile.get('intercept'), 0.0):+.3f}")
    if total_profile.get('apply'):
        lines.append(f"Total calibration: {total_profile.get('mode', 'unknown')} | slope {finite(total_profile.get('slope'), 1.0):.3f} | intercept {finite(total_profile.get('intercept'), 0.0):+.3f}")
    if total_sigma_profile:
        lines.append(f"Total sigma calibration: {'active' if total_sigma_profile.get('apply') else 'inactive'} | scale {finite(total_sigma_profile.get('scale'), 1.0):.2f} | raw NLL {display_number(total_sigma_profile.get('raw_nll'), '.3f')} | calibrated NLL {display_number(total_sigma_profile.get('calibrated_nll'), '.3f')}")
    shrink_profile = ((state.get('probability_shrinkage') or {}).get(phase) or {})
    shrink_bridge = ((state.get('probability_shrinkage') or {}).get('regular_bridge') or {})
    total_shrink_profile = ((state.get('total_shrinkage') or {}).get(phase) or {})
    total_shrink_bridge = ((state.get('total_shrinkage') or {}).get('regular_bridge') or {})
    margin_shrink_profile = ((state.get('margin_shrinkage') or {}).get(phase) or {})
    margin_shrink_bridge = ((state.get('margin_shrinkage') or {}).get('regular_bridge') or {})
    if str(phase).lower() == 'spring' and shrink_profile:
        lines.append(f"Probability shrinkage: {'active' if shrink_profile.get('apply') else 'inactive'} | alpha {finite(shrink_profile.get('alpha'), 0.0):.2f} | model {finite(shrink_profile.get('model_log_loss'), 0.0):.4f} | simple {finite(shrink_profile.get('simple_log_loss'), 0.0):.4f} | blend {finite(shrink_profile.get('blended_log_loss'), 0.0):.4f}")
    elif str(phase).lower() == 'regular' and shrink_bridge:
        lines.append(f"Probability shrinkage bridge: {'active' if shrink_bridge.get('apply') else 'inactive'} | max alpha {finite(shrink_bridge.get('max_alpha'), 0.0):.2f} | days {int(finite(shrink_bridge.get('bridge_days'), 0.0))}")
    if str(phase).lower() == 'spring' and total_shrink_profile:
        lines.append(f"Total shrinkage: {'active' if total_shrink_profile.get('apply') else 'inactive'} | alpha {finite(total_shrink_profile.get('alpha'), 0.0):.2f} | model RMSE {finite(total_shrink_profile.get('model_rmse'), 0.0):.3f} | simple RMSE {finite(total_shrink_profile.get('simple_rmse'), 0.0):.3f} | blend RMSE {finite(total_shrink_profile.get('blended_rmse'), 0.0):.3f}")
    elif str(phase).lower() == 'regular' and total_shrink_bridge:
        lines.append(f"Total shrinkage bridge: {'active' if total_shrink_bridge.get('apply') else 'inactive'} | max alpha {finite(total_shrink_bridge.get('max_alpha'), 0.0):.2f} | days {int(finite(total_shrink_bridge.get('bridge_days'), 0.0))}")
    if str(phase).lower() == 'spring' and margin_shrink_profile:
        lines.append(f"Margin shrinkage: {'active' if margin_shrink_profile.get('apply') else 'inactive'} | alpha {finite(margin_shrink_profile.get('alpha'), 0.0):.2f} | model RMSE {finite(margin_shrink_profile.get('model_rmse'), 0.0):.3f} | simple RMSE {finite(margin_shrink_profile.get('simple_rmse'), 0.0):.3f} | blend RMSE {finite(margin_shrink_profile.get('blended_rmse'), 0.0):.3f}")
    elif str(phase).lower() == 'regular' and margin_shrink_bridge:
        lines.append(f"Margin shrinkage bridge: {'active' if margin_shrink_bridge.get('apply') else 'inactive'} | max alpha {finite(margin_shrink_bridge.get('max_alpha'), 0.0):.2f} | days {int(finite(margin_shrink_bridge.get('bridge_days'), 0.0))}")
    if lineup_profile:
        lines.append(f"Regular lineup earn-back: {'active' if lineup_profile.get('apply') else 'inactive'} | confirmed regular games {int(finite(lineup_profile.get('confirmed_games'), 0.0))}/{int(finite(lineup_profile.get('required_games'), 0.0))} | multiplier {finite(lineup_profile.get('lineup_multiplier'), 0.0):.2f}")
    lines.extend(['', 'VALIDATION SNAPSHOT', f"Backtest sample: {validation['games']} games ({validation['date_span']}) | OOS slice: {validation['oos_games']} games ({validation['oos_span']})"])
    if validation['log_loss'] is not None:
        lines.append(f"OOS moneyline: log loss {validation['log_loss']:.4f} | accuracy {validation['accuracy'] * 100:.1f}%")
    if validation['total_mae'] is not None:
        lines.append(f"OOS totals: MAE {validation['total_mae']:.3f} | RMSE {validation['total_rmse']:.3f}")
    if validation['margin_mae'] is not None:
        lines.append(f"OOS margin: MAE {validation['margin_mae']:.3f} | RMSE {validation['margin_rmse']:.3f}")
    if str(phase).lower() == 'regular' and int(finite(live_validation.get('games'), 0.0)) > 0:
        lines.append(
            f"Live regular settled: {int(finite(live_validation.get('games'), 0.0))} games over "
            f"{int(finite(live_validation.get('settled_days'), 0.0))} day(s) ({live_validation.get('date_span')}) | "
            f"log loss {finite(live_validation.get('log_loss'), 0.0):.4f} | accuracy {finite(live_validation.get('accuracy'), 0.0) * 100:.1f}%"
        )
        if live_validation.get('total_mae') is not None:
            lines.append(
                f"Live regular totals: MAE {finite(live_validation.get('total_mae'), 0.0):.3f} | "
                f"RMSE {finite(live_validation.get('total_rmse'), 0.0):.3f}"
            )
        if live_validation.get('margin_mae') is not None:
            lines.append(
                f"Live regular margin: MAE {finite(live_validation.get('margin_mae'), 0.0):.3f} | "
                f"RMSE {finite(live_validation.get('margin_rmse'), 0.0):.3f}"
            )
    if live_calibration_surfaces:
        lines.append('Live regular calibration surfaces:')
        for row in live_calibration_surfaces:
            total_rmse = 'N/A' if row.get('total_rmse') is None else f"{finite(row.get('total_rmse'), 0.0):.3f}"
            margin_rmse = 'N/A' if row.get('margin_rmse') is None else f"{finite(row.get('margin_rmse'), 0.0):.3f}"
            lines.append(
                f" - {row.get('label')}: {int(finite(row.get('games'), 0.0))} games | "
                f"log loss {finite(row.get('log_loss'), 0.0):.4f} | acc {finite(row.get('accuracy'), 0.0) * 100:.1f}% | "
                f"total RMSE {total_rmse} | margin RMSE {margin_rmse}"
            )
    if model_change_log.get('available'):
        lines.extend(['', 'MODEL CHANGE LOG'])
        for row in model_change_log.get('lines') or []:
            lines.append(f" - {row}")
    lines.extend(['', 'BENCHMARK LADDER'])
    model_bench = benchmark_ladder.get('model_oos') or {}
    simple_bench = benchmark_ladder.get('simple_power_oos') or {}
    model_total_bench = benchmark_ladder.get('model_total_oos') or {}
    simple_total_bench = benchmark_ladder.get('simple_total_oos') or {}
    shrunk_total_bench = benchmark_ladder.get('shrunk_total_oos') or {}
    model_margin_bench = benchmark_ladder.get('model_margin_oos') or {}
    simple_margin_bench = benchmark_ladder.get('simple_margin_oos') or {}
    shrunk_margin_bench = benchmark_ladder.get('shrunk_margin_oos') or {}
    lineup_bench = benchmark_ladder.get('regular_lineup_shadow') or {}
    if model_bench.get('log_loss') is not None:
        lines.append(f" - Clean model OOS: {int(finite(model_bench.get('games'), 0.0))} games | log loss {finite(model_bench.get('log_loss'), 0.0):.4f} | acc {finite(model_bench.get('accuracy'), 0.0) * 100:.1f}%")
    if simple_bench.get('log_loss') is not None:
        lines.append(f" - Simple power OOS: {int(finite(simple_bench.get('games'), 0.0))} games | log loss {finite(simple_bench.get('log_loss'), 0.0):.4f} | acc {finite(simple_bench.get('accuracy'), 0.0) * 100:.1f}%")
    if model_total_bench.get('rmse') is not None:
        lines.append(f" - Totals OOS: model MAE {finite(model_total_bench.get('mae'), 0.0):.3f} / RMSE {finite(model_total_bench.get('rmse'), 0.0):.3f} | simple MAE {finite(simple_total_bench.get('mae'), 0.0):.3f} / RMSE {finite(simple_total_bench.get('rmse'), 0.0):.3f} | shrink MAE {finite(shrunk_total_bench.get('mae'), 0.0):.3f} / RMSE {finite(shrunk_total_bench.get('rmse'), 0.0):.3f}")
    if model_margin_bench.get('rmse') is not None:
        lines.append(f" - Margin OOS: model MAE {finite(model_margin_bench.get('mae'), 0.0):.3f} / RMSE {finite(model_margin_bench.get('rmse'), 0.0):.3f} | simple MAE {finite(simple_margin_bench.get('mae'), 0.0):.3f} / RMSE {finite(simple_margin_bench.get('rmse'), 0.0):.3f} | shrink MAE {finite(shrunk_margin_bench.get('mae'), 0.0):.3f} / RMSE {finite(shrunk_margin_bench.get('rmse'), 0.0):.3f}")
    if lineup_bench.get('games', 0):
        improvement_text = 'N/A' if lineup_bench.get('improvement') is None else f"{finite(lineup_bench.get('improvement'), 0.0):+.4f}"
        lines.append(f" - Regular confirmed-lineup shadow: {int(finite(lineup_bench.get('games'), 0.0))} games ({lineup_bench.get('date_span', 'N/A')}) | no-lineup {finite(lineup_bench.get('no_lineup_log_loss'), 0.0):.4f} | full-lineup {finite(lineup_bench.get('full_lineup_log_loss'), 0.0):.4f} | delta {improvement_text}")
    else:
        lines.append(' - Regular confirmed-lineup shadow: unavailable until settled regular confirmed-lineup games accumulate.')
    lines.extend(['', 'MARKET PROOF'])
    if market_proof.get('available'):
        roi_text = 'N/A' if market_proof.get('roi') is None else f"{finite(market_proof.get('roi'), 0.0) * 100:.1f}%"
        clv_text = 'N/A' if market_proof.get('avg_clv') is None else f"{finite(market_proof.get('avg_clv'), 0.0):+.3f}"
        lines.append(f" - Settled bets: {int(finite(market_proof.get('settled_bets'), 0.0))} | units {finite(market_proof.get('units'), 0.0):+.2f} | ROI {roi_text} | avg CLV {clv_text}")
        for row in (market_proof.get('by_market') or [])[:3]:
            row_roi = 'N/A' if row.get('roi') is None else f"{finite(row.get('roi'), 0.0) * 100:.1f}%"
            row_clv = 'N/A' if row.get('avg_clv') is None else f"{finite(row.get('avg_clv'), 0.0):+.3f}"
            lines.append(f" - By market: {str(row.get('market_type', 'unknown'))} | bets {int(finite(row.get('bets'), 0.0))} | units {finite(row.get('units'), 0.0):+.2f} | ROI {row_roi} | avg CLV {row_clv}")
        for row in (market_proof.get('by_actionability') or [])[:3]:
            row_roi = 'N/A' if row.get('roi') is None else f"{finite(row.get('roi'), 0.0) * 100:.1f}%"
            row_clv = 'N/A' if row.get('avg_clv') is None else f"{finite(row.get('avg_clv'), 0.0):+.3f}"
            lines.append(f" - By actionability: {str(row.get('label', 'unlabeled'))} | bets {int(finite(row.get('bets'), 0.0))} | units {finite(row.get('units'), 0.0):+.2f} | ROI {row_roi} | avg CLV {row_clv}")
    else:
        lines.append(' - No settled market journal rows yet. CLV/ROI proof will populate once regular-season market bets settle.')
    if postgame_attribution.get('available'):
        lines.extend(['', 'POSTGAME ATTRIBUTION'])
        lines.append(
            f" - Recent live misses: {int(finite(postgame_attribution.get('games'), 0.0))} games | "
            f"window {postgame_attribution.get('date_span', 'N/A')}"
        )
        for row in postgame_attribution.get('rows') or []:
            lines.append(
                f" - {row.get('label')}: {int(finite(row.get('count'), 0.0))} games | "
                f"share {finite(row.get('share'), 0.0) * 100:.1f}% | {row.get('plain_english')}"
            )
    lines.append('')
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
        lines.append(f"Total volatility estimate (sigma): {finite(prediction.get('total_sigma'), 4.0):.2f}")
        lines.append(f"Fair run line: {game['home_team']} -1.5 {prediction['run_line_prices'][0]:+d} | {game['away_team']} +1.5 {prediction['run_line_prices'][1]:+d}")
        market_comp = prediction.get('market_comp') or {}
        if market_comp.get('available'):
            lines.append('')
            lines.append(f"Sportsbook comparison: {market_comp.get('market_label', 'Market')}")
            ml_comp = market_comp.get('moneyline', {}) or {}
            if ml_comp.get('available'):
                lines.append(f" - Moneyline: {game['away_team']} {ml_comp.get('away_ml'):+d} | {game['home_team']} {ml_comp.get('home_ml'):+d} | best edge {ml_comp.get('best_side')} {float(ml_comp.get('best_edge', 0.0)) * 100:+.1f} pts")
                lines.append(f" - Moneyline actionability: {actionability_summary_text(ml_comp.get('action'))}")
                if (ml_comp.get('action') or {}).get('label') in ['WATCH', 'BET', 'MUST TAKE']:
                    explainer = ml_comp.get('explainer') or {}
                    lines.append(f" - Why this bet exists: {str(explainer.get('plain_english') or '')}")
                    lines.append(f" - Key drivers: {', '.join(explainer.get('technical_drivers') or [])}")
            rl_comp = market_comp.get('run_line', {}) or {}
            if rl_comp.get('available'):
                lines.append(f" - Run line: {game['home_team']} {float(rl_comp.get('home_line', 0.0)):+.1f} {int(rl_comp.get('home_price', 0)):+d} | {game['away_team']} {float(rl_comp.get('away_line', 0.0)):+.1f} {int(rl_comp.get('away_price', 0)):+d} | best edge {rl_comp.get('best_selection')} {float(rl_comp.get('best_edge', 0.0)) * 100:+.1f} pts")
                lines.append(f" - Run-line actionability: {actionability_summary_text(rl_comp.get('action'))}")
                if (rl_comp.get('action') or {}).get('label') in ['WATCH', 'BET', 'MUST TAKE']:
                    explainer = rl_comp.get('explainer') or {}
                    lines.append(f" - Why this bet exists: {str(explainer.get('plain_english') or '')}")
                    lines.append(f" - Key drivers: {', '.join(explainer.get('technical_drivers') or [])}")
            total_comp = market_comp.get('total', {}) or {}
            if total_comp.get('available'):
                total_price_text = ''
                if total_comp.get('over_price') is not None and total_comp.get('under_price') is not None:
                    total_price_text = f" (O {int(total_comp.get('over_price')):+d} / U {int(total_comp.get('under_price')):+d})"
                lines.append(f" - Total: {float(total_comp.get('market_total', 0.0)):.1f}{total_price_text} | lean {str(total_comp.get('lean', 'FLAT')).title()} by {float(total_comp.get('diff', 0.0)):+.1f} runs")
                lines.append(f" - Total actionability: {actionability_summary_text(total_comp.get('action'))}")
                if (total_comp.get('action') or {}).get('label') in ['WATCH', 'BET', 'MUST TAKE']:
                    explainer = total_comp.get('explainer') or {}
                    lines.append(f" - Why this bet exists: {str(explainer.get('plain_english') or '')}")
                    lines.append(f" - Key drivers: {', '.join(explainer.get('technical_drivers') or [])}")
        lines.extend(['', 'Feature breakdown', f" - Offense edge: {feature['x_off']:+.2f} | Starter edge: {feature['x_starter']:+.2f} | Bullpen edge: {feature['x_bullpen']:+.2f}", f" - Park side adj: {feature['x_park']:+.3f} | Weather side adj: {feature['x_weather']:+.3f} | Context side adj: {feature['x_context']:+.3f}", f" - Shared run environment: {feature['shared_env']:+.2f} | Park total adj: {(finite(feature['park_history']['total_adj']) + finite(feature['park_static']['total_adj'])):+.2f} | Weather total adj: {feature['weather_adj']['total_adj']:+.2f}", f" - Away starter ({game.get('away_pitcher') or 'TBD'}): {feature['away_starter']['score']:+.2f} [{feature['away_starter']['source']}] | expected IP {finite(feature['away_starter_workload'].get('expected_ip'), 0.0):.1f} | short-start {finite(feature['away_starter_workload'].get('short_start_risk'), 0.0):.2f} | TTO risk {finite(feature['away_starter_workload'].get('tto_risk'), 0.0):.2f}", f" - Home starter ({game.get('home_pitcher') or 'TBD'}): {feature['home_starter']['score']:+.2f} [{feature['home_starter']['source']}] | expected IP {finite(feature['home_starter_workload'].get('expected_ip'), 0.0):.1f} | short-start {finite(feature['home_starter_workload'].get('short_start_risk'), 0.0):.2f} | TTO risk {finite(feature['home_starter_workload'].get('tto_risk'), 0.0):.2f}", f" - Bullpen scores: {game['away_team']} {feature['away_bullpen']:+.2f} | {game['home_team']} {feature['home_bullpen']:+.2f}", f" - Bullpen availability: {game['away_team']} {finite((feature['away_bullpen_profile'] or {}).get('availability_score'), 0.0):+.2f} / leverage {finite((feature['away_bullpen_profile'] or {}).get('leverage_availability'), 0.0):+.2f} | {game['home_team']} {finite((feature['home_bullpen_profile'] or {}).get('availability_score'), 0.0):+.2f} / leverage {finite((feature['home_bullpen_profile'] or {}).get('leverage_availability'), 0.0):+.2f}", f" - Venue traits: elev {display_number(feature['venue_meta'].get('elevation_ft'), '.0f')} ft | {feature['venue_meta'].get('roof_type', 'unknown')} | {feature['venue_meta'].get('surface_type', 'unknown')}", f" - Weather: {feature['weather'].get('source', 'neutral')} | temp {display_number(feature['weather'].get('temp_f'), '.0f')}F | wind {display_number(feature['weather'].get('wind_mph'), '.0f')} mph | precip {display_number(feature['weather'].get('precip_prob'), '.0f')}%", f" - Confirmed lineups: {game['away_team']} {feature['lineups'].get('away', 0)}/9 | {game['home_team']} {feature['lineups'].get('home', 0)}/9", '', f" - {game['away_team']}: OffScore {finite(away_row.get('offense_score_blended'), 0.0):+.2f} | prior wt {finite(away_row.get('prior_weight'), 0.0) * 100:.0f}% | xwOBA/wOBA {display_number(away_row.get('xwoba'), '.3f')} | xSLG {display_number(away_row.get('xslg'), '.3f')} | xBA {display_number(away_row.get('xba'), '.3f')} | EV {display_number(away_row.get('avg_ev'), '.1f')} | HH% {finite(away_row.get('hardhit_rate'), 0.0) * 100:.1f}%", f" - {game['home_team']}: OffScore {finite(home_row.get('offense_score_blended'), 0.0):+.2f} | prior wt {finite(home_row.get('prior_weight'), 0.0) * 100:.0f}% | xwOBA/wOBA {display_number(home_row.get('xwoba'), '.3f')} | xSLG {display_number(home_row.get('xslg'), '.3f')} | xBA {display_number(home_row.get('xba'), '.3f')} | EV {display_number(home_row.get('avg_ev'), '.1f')} | HH% {finite(home_row.get('hardhit_rate'), 0.0) * 100:.1f}%", ''])
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
    import quant_store

    os.makedirs(OUT_DIR, exist_ok=True)
    report_date = today_local()
    warnings = []
    manual_warning = manual_market_warning(report_date)
    if manual_warning:
        warnings.append(manual_warning)
    client = RequestClient()
    model_state = load_model_state()
    live_archive = load_live_prediction_archive()
    live_archive, archive_settled, archive_warn = settle_live_prediction_archive(client, live_archive, report_date)
    warnings.extend(archive_warn)
    if archive_settled:
        save_live_prediction_archive(live_archive)
        warnings.append(f"Settled {archive_settled} archived live prediction row(s).")
    live_archive, archive_backfilled = backfill_live_archive_context(live_archive)
    if archive_backfilled:
        save_live_prediction_archive(live_archive)
        warnings.append(f"Backfilled {archive_backfilled} archived live prediction row(s) with starter/bullpen attribution context.")
    backtest_log = load_validation_log()
    model_state, shrink_profile = fit_probability_shrinkage_profile(model_state, backtest_log)
    model_state, total_calib_profile = fit_total_calibration_profile(model_state, backtest_log)
    model_state, total_shrink_profile = fit_total_shrinkage_profile(model_state, backtest_log)
    model_state, total_sigma_profile = fit_total_sigma_profile(model_state, backtest_log)
    model_state, margin_shrink_profile = fit_margin_shrinkage_profile(model_state, backtest_log)
    model_state, lineup_profile = update_lineup_earn_back_state(model_state, live_archive)
    park_cache = load_park_cache()
    venue_cache = load_venue_metadata_cache()
    replay_batch_size = int(finite(os.getenv('LIVE_ARCHIVE_CONTEXT_MAX_DATES'), 3))
    live_archive, model_state, venue_cache, archive_replay_refresh = refresh_live_archive_context_from_replay(
        client,
        live_archive,
        model_state,
        park_cache,
        venue_cache,
        backtest_log,
        report_date,
        max_dates=replay_batch_size,
    )
    warnings.extend(archive_replay_refresh.get('warnings') or [])
    if int(archive_replay_refresh.get('rows_refreshed') or 0) > 0:
        save_live_prediction_archive(live_archive)
        warnings.append(
            f"Historical replay refreshed {int(archive_replay_refresh.get('rows_refreshed') or 0)} "
            f"archive row(s) across {int(archive_replay_refresh.get('dates_refreshed') or 0)} prior date(s)."
        )
    if int(archive_replay_refresh.get('pending_dates_remaining') or 0) > 0:
        warnings.append(
            f"Historical replay backlog: {int(archive_replay_refresh.get('pending_dates_remaining') or 0)} "
            f"prior archive date(s) still pending replay context refresh."
        )
    save_model_state(model_state)
    games, schedule_warnings = fetch_schedule_games(client, report_date)
    warnings.extend(schedule_warnings)
    phase = season_phase_for_games(games) if games else 'spring'
    weights = phase_weights_from_state(model_state, phase)
    validation = build_validation_summary(backtest_log)
    margin_sigma = estimate_margin_sigma(backtest_log)
    shrink_phase_profile = ((model_state.get('probability_shrinkage') or {}).get(phase) or {})
    shrink_bridge = ((model_state.get('probability_shrinkage') or {}).get('regular_bridge') or {})
    total_shrink_phase = ((model_state.get('total_shrinkage') or {}).get(phase) or {})
    total_shrink_bridge = ((model_state.get('total_shrinkage') or {}).get('regular_bridge') or {})
    total_sigma_phase = ((model_state.get('total_sigma_calibration') or {}).get(phase) or {})
    margin_shrink_phase = ((model_state.get('margin_shrinkage') or {}).get(phase) or {})
    margin_shrink_bridge = ((model_state.get('margin_shrinkage') or {}).get('regular_bridge') or {})
    if str(phase).lower() == 'spring' and shrink_phase_profile:
        warnings.append(f"Spring probability shrinkage: {'active' if shrink_phase_profile.get('apply') else 'inactive'} | alpha {finite(shrink_phase_profile.get('alpha'), 0.0):.2f}")
    elif str(phase).lower() == 'regular' and shrink_bridge.get('apply'):
        warnings.append(f"Early regular probability bridge max alpha: {finite(shrink_bridge.get('max_alpha'), 0.0):.2f} over {int(finite(shrink_bridge.get('bridge_days'), 0.0))} days.")
    if str(phase).lower() == 'spring' and total_shrink_phase:
        warnings.append(f"Spring total shrinkage: {'active' if total_shrink_phase.get('apply') else 'inactive'} | alpha {finite(total_shrink_phase.get('alpha'), 0.0):.2f}")
    elif str(phase).lower() == 'regular' and total_shrink_bridge.get('apply'):
        warnings.append(f"Early regular total bridge max alpha: {finite(total_shrink_bridge.get('max_alpha'), 0.0):.2f} over {int(finite(total_shrink_bridge.get('bridge_days'), 0.0))} days.")
    if total_sigma_phase:
        warnings.append(f"{phase.title()} total sigma calibration: {'active' if total_sigma_phase.get('apply') else 'inactive'} | scale {finite(total_sigma_phase.get('scale'), 1.0):.2f}")
    if str(phase).lower() == 'spring' and margin_shrink_phase:
        warnings.append(f"Spring margin shrinkage: {'active' if margin_shrink_phase.get('apply') else 'inactive'} | alpha {finite(margin_shrink_phase.get('alpha'), 0.0):.2f}")
    elif str(phase).lower() == 'regular' and margin_shrink_bridge.get('apply'):
        warnings.append(f"Early regular margin bridge max alpha: {finite(margin_shrink_bridge.get('max_alpha'), 0.0):.2f} over {int(finite(margin_shrink_bridge.get('bridge_days'), 0.0))} days.")
    if lineup_profile:
        warnings.append(f"Lineup earn-back status: {'active' if lineup_profile.get('apply') else 'inactive'} ({int(finite(lineup_profile.get('confirmed_games'), 0.0))}/{int(finite(lineup_profile.get('required_games'), 0.0))} regular confirmed-lineup games).")
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

    settled_journal = load_settled_bet_journal()
    model_state, realized_uncertainty = update_realized_uncertainty_state(model_state, live_archive, settled_journal)
    save_model_state(model_state)
    phase_realized = (realized_uncertainty.get(phase) or {}) if isinstance(realized_uncertainty, dict) else {}
    if phase_realized:
        realized_parts = []
        for market_name, label in [('moneyline', 'ML'), ('total', 'TOT'), ('run_line', 'RL')]:
            learned = phase_realized.get(market_name) or {}
            if learned:
                realized_parts.append(f"{label} bias {finite(learned.get('bias_add'), 0.0):+.3f} / scale {finite(learned.get('penalty_scale'), 1.0):.2f}")
        if realized_parts:
            warnings.append('Realized uncertainty learning: ' + ' | '.join(realized_parts))

    market_map, market_warn = market.load_market_consensus_for_date(report_date, games)
    warnings.extend(market_warn)
    if market_map:
        warnings.append(f"Current sportsbook consensus loaded for {len(market_map)} game(s).")

    umpire_cache = load_umpire_stats_cache()
    umpire_refreshed_date = str(umpire_cache.get('_refreshed_date', ''))
    if umpire_refreshed_date != report_date.isoformat():
        umpire_cache = refresh_umpire_cache_from_savant(client, umpire_cache)
        umpire_cache['_refreshed_date'] = report_date.isoformat()
        save_umpire_stats_cache(umpire_cache)
    umpire_known = sum(1 for k in umpire_cache if not k.startswith('_'))
    warnings.append(f"Umpire cache: {umpire_known} umpire(s) loaded{'' if umpire_known else ' — cache empty, adjustments will be 0'}.")

    season_start, pregame_end, game_types = dt.date(report_date.year, 1, 1), report_date, ('S|' if phase == 'spring' else 'R|')
    current_batter_raw = savant_statcast_csv(client, 'batter', report_date.year, season_start, pregame_end, game_types)
    batter_raw_30 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=30), pregame_end, game_types)
    batter_raw_14 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=14), pregame_end, game_types)
    batter_raw_7 = savant_statcast_csv(client, 'batter', report_date.year, report_date - dt.timedelta(days=7), pregame_end, game_types)
    batters = aggregate_batter_quality(current_batter_raw)
    prior_batters = previous_regular_season_batters(client, report_date.year - 1)
    strengths = build_team_strengths(
        client,
        report_date,
        phase,
        games,
        current_batter_raw,
        batter_raw_30,
        batter_raw_14,
        batter_raw_7,
        batters,
        prior_batters,
    )
    current_pitchers = aggregate_pitcher_quality(savant_statcast_csv(client, 'pitcher', report_date.year, season_start, pregame_end, game_types))
    prior_pitchers = previous_regular_season_pitchers(client, report_date.year - 1)
    bullpen = bullpen_snapshot(client, report_date, 7, current_pitchers, prior_pitchers)
    bullpen_profiles = bullpen.set_index('team').to_dict('index') if bullpen is not None and not bullpen.empty else {}
    predictions = [
        predict_game(
            game,
            phase,
            weights,
            model_state,
            strengths,
            current_pitchers,
            prior_pitchers,
            batters,
            prior_batters,
            bullpen_profiles,
            park_cache,
            client,
            venue_cache,
            margin_sigma,
            umpire_cache=umpire_cache,
        )
        for game in games
    ]
    for prediction in predictions:
        prediction['market_comp'] = market_comparison_for_prediction(prediction, market_map, phase, margin_sigma)

    journal_count, journal_warn = seed_open_bet_journal(report_date, predictions, market_map, phase, margin_sigma)
    warnings.extend(journal_warn)
    if journal_count:
        warnings.append(f"Bet journal refreshed with {journal_count} actionable market candidate(s).")

    live_archive, archived_today = archive_live_predictions(report_date, phase, predictions, live_archive)
    save_live_prediction_archive(live_archive)
    if archived_today:
        warnings.append(f"Archived {archived_today} live prediction row(s) for future regular-season lineup validation.")
    save_venue_metadata_cache(venue_cache)
    report_path = write_report(report_date, phase, weights, model_state, predictions, validation, warnings)
    workbook_path = write_workbook(strengths, batters, current_pitchers, bullpen)
    benchmark_ladder = build_benchmark_ladder(model_state, backtest_log, live_archive)
    market_proof = build_market_proof_summary()
    quant_status = quant_store.export_quant_snapshot(
        report_date,
        phase,
        weights,
        model_state,
        predictions,
        validation,
        benchmark_ladder,
        market_proof,
        warnings,
        report_path,
        workbook_path,
    )
    print(report_path)
    print(workbook_path)
    if quant_status.get('exported'):
        print(quant_status.get('db_path'))
    elif quant_status.get('warning'):
        print(quant_status.get('warning'))


if __name__ == '__main__':
    main()

































