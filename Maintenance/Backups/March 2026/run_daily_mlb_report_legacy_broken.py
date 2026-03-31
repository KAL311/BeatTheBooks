from __future__ import annotations
import os
import re
import math
import time
import json
import copy
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------
OUT_DIR = r"C:\Users\fraun\OneDrive\Documents\2026 MLB Report"
SEASON = 2026

SAVANT_PROBABLE_PITCHERS_URL = "https://baseballsavant.mlb.com/probable-pitchers"
SAVANT_STATCAST_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SAVANT_SCOREBOARD_DATA_URL = "https://baseballsavant.mlb.com/scoreboard-data"
SAVANT_GF_URL = "https://baseballsavant.mlb.com/gf"
STATSAPI_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
STATSAPI_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}"
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
STATSAPI_VENUE_URL = "https://statsapi.mlb.com/api/v1/venues/{venue_id}"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# Rolling windows (days)
WINDOWS = [7, 14, 30]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MLB-Predictive-Report-Engine/1.0)"
}
MODEL_STATE_PATH = os.path.join(OUT_DIR, "model_state.json")
BACKTEST_LOG_PATH = os.path.join(OUT_DIR, "backtest_log.csv")
PARK_FACTORS_CACHE_PATH = os.path.join(OUT_DIR, "park_factors_cache.csv")
VENUE_METADATA_CACHE_PATH = os.path.join(OUT_DIR, "venue_metadata_cache.json")
SPORTSBOOK_DB_PATH = os.path.join(OUT_DIR, "sportsbook_lines.db")
SPORTSBOOK_CSV_PATH = os.path.join(OUT_DIR, "sportsbook_lines.csv")

# Rolling blend (fixed; stable & reproducible)
ROLL_BLEND = {"season": 0.55, "d30": 0.20, "d14": 0.15, "d7": 0.10}
OFFENSE_PRIOR_REGRESSION_PA = {"season": 500.0, "d30": 360.0, "d14": 240.0, "d7": 160.0}

# Backtest window used for diagnostics (not required for updating, but stored)
BACKTEST_LOOKBACK_DAYS = 30
BACKTEST_FEATURE_VERSION = 5
TOTALS_FEATURE_VERSION = 2
BACKTEST_HISTORY_REFRESH_BATCH_DATES = 1

# Moderate adaptation speed + regularization
WEIGHT_LR = 0.15
L2_LAMBDA = 0.02
DEFAULT_GAME_TYPES = "R|S|"
BASE_WEIGHT_DEFAULTS = {"w_off": 0.65, "w_starter": 0.55, "w_bullpen": 0.30, "w_park": 0.03, "w_weather": 0.02, "w_context": 0.12, "home_field": 0.12, "spread_scale": 3.6}
REGULAR_TRANSITION_BRIDGE_DAYS = 14
REGULAR_TRANSITION_SPRING_CARRYOVER_MAX = 0.35
PROBABILITY_CALIBRATION_MIN_GAMES = 24
PROBABILITY_CALIBRATION_LR = 0.08
PROBABILITY_CALIBRATION_L2_INTERCEPT = 0.30
PROBABILITY_CALIBRATION_L2_SLOPE = 0.80
PROBABILITY_CALIBRATION_MIN_IMPROVEMENT = 0.0015
PROBABILITY_CALIBRATION_HALF_LIFE_DAYS = 45.0
TOTAL_CALIBRATION_MIN_GAMES = 24
TOTAL_CALIBRATION_MIN_IMPROVEMENT = 0.05
TOTAL_CALIBRATION_HALF_LIFE_DAYS = 45.0
MARGIN_CALIBRATION_MIN_GAMES = 24
MARGIN_CALIBRATION_MIN_IMPROVEMENT = 0.05
MARGIN_CALIBRATION_HALF_LIFE_DAYS = 45.0
PARK_CACHE_START = dt.date(SEASON - 1, 3, 1)
PARK_RECENCY_HALF_LIFE_DAYS = 120.0
PARK_TOTAL_SHRINK = 18.0
PARK_HOME_EDGE_SHRINK = 45.0
PARK_STATIC_TOTAL_PRIOR_WEIGHT = 14.0
PARK_STATIC_HOME_PRIOR_WEIGHT = 60.0

# Static park priors give new or low-sample venues a physical baseline.
PARK_STATIC_PROFILES = [
    {"aliases": ["Oriole Park at Camden Yards", "Camden Yards"], "elevation_ft": 20, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.02, "home_edge_bias": 0.004},
    {"aliases": ["Fenway Park"], "elevation_ft": 20, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.18, "home_edge_bias": 0.018},
    {"aliases": ["Yankee Stadium"], "elevation_ft": 55, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.12, "home_edge_bias": 0.010},
    {"aliases": ["Rogers Centre", "SkyDome"], "elevation_ft": 250, "roof_type": "retractable", "surface_type": "turf", "run_bias": 0.06, "home_edge_bias": 0.010},
    {"aliases": ["Tropicana Field", "Tropicana"], "elevation_ft": 15, "roof_type": "dome", "surface_type": "turf", "run_bias": -0.09, "home_edge_bias": 0.012},
    {"aliases": ["George M. Steinbrenner Field", "Steinbrenner Field"], "elevation_ft": 45, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.09, "home_edge_bias": 0.010},
    {"aliases": ["Progressive Field"], "elevation_ft": 650, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.01, "home_edge_bias": 0.004},
    {"aliases": ["Comerica Park"], "elevation_ft": 640, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.10, "home_edge_bias": 0.004},
    {"aliases": ["Kauffman Stadium"], "elevation_ft": 910, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.09, "home_edge_bias": 0.004},
    {"aliases": ["Target Field"], "elevation_ft": 835, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.03, "home_edge_bias": 0.004},
    {"aliases": ["Rate Field", "Guaranteed Rate Field", "U.S. Cellular Field", "US Cellular Field"], "elevation_ft": 595, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.07, "home_edge_bias": 0.005},
    {"aliases": ["Angel Stadium", "Angel Stadium of Anaheim"], "elevation_ft": 160, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.03, "home_edge_bias": 0.003},
    {"aliases": ["Globe Life Field"], "elevation_ft": 550, "roof_type": "retractable", "surface_type": "turf", "run_bias": 0.02, "home_edge_bias": 0.006},
    {"aliases": ["Sutter Health Park", "Raley Field"], "elevation_ft": 20, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.14, "home_edge_bias": 0.010},
    {"aliases": ["T-Mobile Park", "Safeco Field"], "elevation_ft": 20, "roof_type": "retractable", "surface_type": "grass", "run_bias": -0.11, "home_edge_bias": 0.006},
    {"aliases": ["Daikin Park", "Minute Maid Park"], "elevation_ft": 50, "roof_type": "retractable", "surface_type": "grass", "run_bias": 0.10, "home_edge_bias": 0.008},
    {"aliases": ["Chase Field"], "elevation_ft": 1086, "roof_type": "retractable", "surface_type": "turf", "run_bias": 0.08, "home_edge_bias": 0.008},
    {"aliases": ["Truist Park", "SunTrust Park"], "elevation_ft": 1050, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.05, "home_edge_bias": 0.005},
    {"aliases": ["loanDepot park", "LoanDepot Park", "Marlins Park"], "elevation_ft": 10, "roof_type": "retractable", "surface_type": "turf", "run_bias": -0.18, "home_edge_bias": 0.008},
    {"aliases": ["Citi Field"], "elevation_ft": 15, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.06, "home_edge_bias": 0.004},
    {"aliases": ["Citizens Bank Park"], "elevation_ft": 40, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.11, "home_edge_bias": 0.006},
    {"aliases": ["Nationals Park"], "elevation_ft": 25, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.02, "home_edge_bias": 0.004},
    {"aliases": ["Wrigley Field"], "elevation_ft": 600, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.01, "home_edge_bias": 0.006},
    {"aliases": ["Great American Ball Park"], "elevation_ft": 480, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.17, "home_edge_bias": 0.006},
    {"aliases": ["American Family Field", "Miller Park"], "elevation_ft": 620, "roof_type": "retractable", "surface_type": "grass", "run_bias": 0.06, "home_edge_bias": 0.006},
    {"aliases": ["PNC Park"], "elevation_ft": 730, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.08, "home_edge_bias": 0.005},
    {"aliases": ["Busch Stadium"], "elevation_ft": 465, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.05, "home_edge_bias": 0.004},
    {"aliases": ["Oracle Park", "AT&T Park", "SBC Park"], "elevation_ft": 60, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.20, "home_edge_bias": 0.006},
    {"aliases": ["Petco Park"], "elevation_ft": 60, "roof_type": "open_air", "surface_type": "grass", "run_bias": -0.13, "home_edge_bias": 0.005},
    {"aliases": ["Dodger Stadium"], "elevation_ft": 340, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.02, "home_edge_bias": 0.004},
    {"aliases": ["Coors Field"], "elevation_ft": 5180, "roof_type": "open_air", "surface_type": "grass", "run_bias": 0.34, "home_edge_bias": 0.014},
]
# -----------------------------
# UTIL
# -----------------------------
def today_local() -> dt.date:
    # Uses local machine date (your Windows box); schedule depends on that.
    return dt.date.today()

def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def fetch_html(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text
def fetch_json(url: str, params: Dict[str, str], timeout: int = 30) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def savant_scoreboard(date_: dt.date) -> dict:
    # returns JSON from /scoreboard-data
    return fetch_json(SAVANT_SCOREBOARD_DATA_URL, {"date": date_.isoformat()})

def statsapi_schedule(date_: dt.date, hydrate: str = "probablePitcher,team") -> dict:
    cache = getattr(statsapi_schedule, "_cache", None)
    if cache is None:
        cache = {}
        statsapi_schedule._cache = cache

    key = (date_.isoformat(), hydrate)
    if key not in cache:
        cache[key] = fetch_json(
            STATSAPI_SCHEDULE_URL,
            {
                "sportId": "1",
                "date": date_.isoformat(),
                "hydrate": hydrate,
            },
        )
    return cache[key]

def statsapi_schedule_range(start_date: dt.date, end_date: dt.date, hydrate: str = "team") -> dict:
    cache = getattr(statsapi_schedule_range, "_cache", None)
    if cache is None:
        cache = {}
        statsapi_schedule_range._cache = cache

    key = (start_date.isoformat(), end_date.isoformat(), hydrate)
    if key not in cache:
        cache[key] = fetch_json(
            STATSAPI_SCHEDULE_URL,
            {
                "sportId": "1",
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "hydrate": hydrate,
            },
        )
    return cache[key]


def normalize_model_phase(raw: Optional[object]) -> Optional[str]:
    txt = str(raw or "").strip().lower()
    if txt in ["spring", "preseason", "exhibition"]:
        return "spring"
    if txt in ["regular", "regular_season", "season"]:
        return "regular"
    return None


def normalize_game_type(raw: Optional[object]) -> Optional[str]:
    txt = str(raw or "").strip().upper()
    return txt[:1] if txt else None


def phase_from_game_type(raw: Optional[object]) -> Optional[str]:
    game_type = normalize_game_type(raw)
    if game_type == "S":
        return "spring"
    if game_type == "R":
        return "regular"
    return None


def game_types_for_phase(phase: Optional[str]) -> str:
    phase_name = normalize_model_phase(phase)
    if phase_name == "spring":
        return "S|"
    if phase_name == "regular":
        return "R|"
    return DEFAULT_GAME_TYPES


def opening_day_for_season(season: int) -> Optional[dt.date]:
    cache = getattr(opening_day_for_season, "_cache", None)
    if cache is None:
        cache = {}
        opening_day_for_season._cache = cache
    season = int(season)
    if season not in cache:
        opening_day: Optional[dt.date] = None
        try:
            payload = statsapi_schedule_range(dt.date(season, 3, 1), dt.date(season, 4, 30), hydrate="team")
            for day in payload.get("dates") or []:
                if not isinstance(day, dict):
                    continue
                try:
                    day_date = dt.date.fromisoformat(str(day.get("date")))
                except Exception:
                    continue
                game_types = [normalize_game_type((g or {}).get("gameType")) for g in (day.get("games") or []) if isinstance(g, dict)]
                if any(gt == "R" for gt in game_types):
                    opening_day = day_date
                    break
        except Exception:
            opening_day = None
        cache[season] = opening_day
    return cache[season]


def season_phase_for_date(date_: dt.date) -> str:
    opening_day = opening_day_for_season(date_.year)
    if opening_day is not None:
        return "spring" if date_ < opening_day else "regular"
    if date_.month <= 3:
        return "spring"
    return "regular"


def confidence_thresholds(model_phase: Optional[str] = None) -> Tuple[float, float]:
    phase_name = normalize_model_phase(model_phase)
    if phase_name == "spring":
        return 0.16, 0.10
    return 0.12, 0.07


def market_thresholds(market_type: str, sportsbook_count: int = 1, model_phase: Optional[str] = None) -> Dict[str, float]:
    phase_name = normalize_model_phase(model_phase)
    books = max(int(_finite(sportsbook_count, 1)), 1)
    kind = str(market_type or "").strip().lower()
    if kind == "total":
        if phase_name == "spring":
            return {"watch": 1.00, "high": 1.50 if books >= 2 else 1.65}
        return {"watch": 0.75, "high": 1.25 if books >= 2 else 1.40}
    if phase_name == "spring":
        return {"watch": 0.040, "high": 0.075 if books >= 2 else 0.085}
    return {"watch": 0.025, "high": 0.055 if books >= 2 else 0.065}


def pregame_statcast_end_date(model_date: dt.date) -> dt.date:
    """
    Savant Statcast Search uses a strict game_date_lt upper bound.
    Returning the game date itself keeps same-day results out of pregame features.
    """
    return model_date

def blend_weight_sets(primary: Dict[str, object], secondary: Dict[str, object], secondary_share: float) -> Dict[str, float]:
    share = float(np.clip(_finite(secondary_share, 0.0), 0.0, 1.0))
    out: Dict[str, float] = {}
    for key, default in BASE_WEIGHT_DEFAULTS.items():
        primary_value = _finite((primary or {}).get(key, default), default)
        secondary_value = _finite((secondary or {}).get(key, default), default)
        out[key] = float(((1.0 - share) * primary_value) + (share * secondary_value))
    return out


def ensure_phase_weight_state(state: Dict) -> Dict:
    top_weights = state.get("weights") if isinstance(state.get("weights"), dict) else {}
    raw_profiles = state.get("weights_by_phase") if isinstance(state.get("weights_by_phase"), dict) else {}
    profiles: Dict[str, Dict[str, float]] = {}
    for phase in ["spring", "regular"]:
        profile = dict(BASE_WEIGHT_DEFAULTS)
        if phase == "spring" and not raw_profiles and isinstance(top_weights, dict):
            profile.update(top_weights)
        raw_profile = raw_profiles.get(phase) if isinstance(raw_profiles.get(phase), dict) else {}
        profile.update(raw_profile)
        profiles[phase] = {key: float(_finite(profile.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    if not raw_profiles and isinstance(top_weights, dict):
        profiles["spring"] = {key: float(_finite(top_weights.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
        profiles["regular"] = {key: float(_finite(top_weights.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}

    transition = {
        "bridge_days": REGULAR_TRANSITION_BRIDGE_DAYS,
        "spring_carryover_max": REGULAR_TRANSITION_SPRING_CARRYOVER_MAX,
    }
    if isinstance(state.get("regular_transition"), dict):
        transition.update(state["regular_transition"])
    bridge_days = safe_float(transition.get("bridge_days"))
    carryover = safe_float(transition.get("spring_carryover_max"))
    transition["bridge_days"] = max(int(bridge_days), 0) if bridge_days is not None else REGULAR_TRANSITION_BRIDGE_DAYS
    transition["spring_carryover_max"] = float(np.clip(_finite(carryover, REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), 0.0, 1.0))

    state["weights_by_phase"] = profiles
    state["regular_transition"] = transition
    state["weights"] = dict(profiles["regular"])
    return state


def identity_probability_calibration(reason: str = "identity probability mapping") -> Dict[str, object]:
    return {
        "apply": False,
        "mode": "identity",
        "intercept": 0.0,
        "slope": 1.0,
        "games": 0,
        "raw_log_loss": None,
        "calibrated_log_loss": None,
        "reason": reason,
    }


def ensure_probability_calibration_state(state: Dict) -> Dict:
    raw_profiles = state.get("probability_calibration") if isinstance(state.get("probability_calibration"), dict) else {}
    profiles: Dict[str, Dict[str, object]] = {}
    for phase in ["spring", "regular"]:
        profile = identity_probability_calibration()
        raw_profile = raw_profiles.get(phase) if isinstance(raw_profiles.get(phase), dict) else {}
        profile.update(raw_profile)
        profile["apply"] = bool(profile.get("apply"))
        profile["mode"] = str(profile.get("mode") or "identity")
        profile["intercept"] = float(np.clip(_finite(profile.get("intercept"), 0.0), -1.5, 1.5))
        profile["slope"] = float(np.clip(_finite(profile.get("slope"), 1.0), 0.25, 1.5))
        profile["games"] = int(max(_finite(profile.get("games"), 0), 0))
        profile["raw_log_loss"] = safe_float(profile.get("raw_log_loss"))
        profile["calibrated_log_loss"] = safe_float(profile.get("calibrated_log_loss"))
        profile["reason"] = str(profile.get("reason") or "identity probability mapping")
        profiles[phase] = profile
    state["probability_calibration"] = profiles
    return state


def identity_total_calibration(reason: str = "identity total mapping") -> Dict[str, object]:
    return {
        "apply": False,
        "mode": "identity",
        "intercept": 0.0,
        "slope": 1.0,
        "tail_slope": 0.0,
        "tail_threshold": 9.0,
        "games": 0,
        "raw_rmse": None,
        "calibrated_rmse": None,
        "reason": reason,
    }


def ensure_total_calibration_state(state: Dict) -> Dict:
    raw_profiles = state.get("total_calibration") if isinstance(state.get("total_calibration"), dict) else {}
    profiles: Dict[str, Dict[str, object]] = {}
    for phase in ["spring", "regular"]:
        profile = identity_total_calibration()
        raw_profile = raw_profiles.get(phase) if isinstance(raw_profiles.get(phase), dict) else {}
        profile.update(raw_profile)
        profile["apply"] = bool(profile.get("apply"))
        profile["mode"] = str(profile.get("mode") or "identity")
        profile["intercept"] = float(np.clip(_finite(profile.get("intercept"), 0.0), -3.0, 3.0))
        profile["slope"] = float(np.clip(_finite(profile.get("slope"), 1.0), 0.65, 1.35))
        profile["tail_slope"] = float(np.clip(_finite(profile.get("tail_slope"), 0.0), -1.0, 1.25))
        profile["tail_threshold"] = float(np.clip(_finite(profile.get("tail_threshold"), 9.0), 8.0, 10.5))
        profile["games"] = int(max(_finite(profile.get("games"), 0), 0))
        profile["raw_rmse"] = safe_float(profile.get("raw_rmse"))
        profile["calibrated_rmse"] = safe_float(profile.get("calibrated_rmse"))
        profile["reason"] = str(profile.get("reason") or "identity total mapping")
        profiles[phase] = profile
    state["total_calibration"] = profiles
    return state


def identity_margin_calibration(reason: str = "identity margin mapping") -> Dict[str, object]:
    return {
        "apply": False,
        "mode": "identity",
        "intercept": 0.0,
        "slope": 1.0,
        "games": 0,
        "raw_rmse": None,
        "calibrated_rmse": None,
        "reason": reason,
    }


def ensure_margin_calibration_state(state: Dict) -> Dict:
    raw_profiles = state.get("margin_calibration") if isinstance(state.get("margin_calibration"), dict) else {}
    profiles: Dict[str, Dict[str, object]] = {}
    for phase in ["spring", "regular"]:
        profile = identity_margin_calibration()
        raw_profile = raw_profiles.get(phase) if isinstance(raw_profiles.get(phase), dict) else {}
        profile.update(raw_profile)
        profile["apply"] = bool(profile.get("apply"))
        profile["mode"] = str(profile.get("mode") or "identity")
        profile["intercept"] = float(np.clip(_finite(profile.get("intercept"), 0.0), -1.5, 1.5))
        profile["slope"] = float(np.clip(_finite(profile.get("slope"), 1.0), 0.50, 1.50))
        profile["games"] = int(max(_finite(profile.get("games"), 0), 0))
        profile["raw_rmse"] = safe_float(profile.get("raw_rmse"))
        profile["calibrated_rmse"] = safe_float(profile.get("calibrated_rmse"))
        profile["reason"] = str(profile.get("reason") or "identity margin mapping")
        profiles[phase] = profile
    state["margin_calibration"] = profiles
    return state


def blend_margin_calibration_profiles(primary: Dict[str, object], secondary: Dict[str, object], secondary_share: float, label: str) -> Dict[str, object]:
    share = float(np.clip(_finite(secondary_share, 0.0), 0.0, 1.0))
    primary_profile = identity_margin_calibration()
    if isinstance(primary, dict):
        primary_profile.update(primary)
    secondary_profile = identity_margin_calibration()
    if isinstance(secondary, dict):
        secondary_profile.update(secondary)

    primary_apply = bool(primary_profile.get("apply"))
    secondary_apply = bool(secondary_profile.get("apply"))
    primary_intercept = float(_finite(primary_profile.get("intercept"), 0.0)) if primary_apply else 0.0
    secondary_intercept = float(_finite(secondary_profile.get("intercept"), 0.0)) if secondary_apply else 0.0
    primary_slope = float(_finite(primary_profile.get("slope"), 1.0)) if primary_apply else 1.0
    secondary_slope = float(_finite(secondary_profile.get("slope"), 1.0)) if secondary_apply else 1.0

    blended = identity_margin_calibration()
    blended.update({
        "apply": bool(primary_apply or secondary_apply),
        "mode": label if (primary_apply or secondary_apply) else "identity",
        "intercept": float(((1.0 - share) * primary_intercept) + (share * secondary_intercept)),
        "slope": float(np.clip(((1.0 - share) * primary_slope) + (share * secondary_slope), 0.50, 1.50)),
        "games": int(round(((1.0 - share) * float(_finite(primary_profile.get("games"), 0.0))) + (share * float(_finite(secondary_profile.get("games"), 0.0))))),
        "raw_rmse": safe_float(primary_profile.get("raw_rmse")) if primary_apply else safe_float(secondary_profile.get("raw_rmse")),
        "calibrated_rmse": safe_float(primary_profile.get("calibrated_rmse")) if primary_apply else safe_float(secondary_profile.get("calibrated_rmse")),
        "reason": f"blended margin calibration ({label}) with spring share {share:.0%}",
    })
    if not blended["apply"]:
        blended["reason"] = "identity margin mapping"
    return blended


def active_margin_calibration(state: Dict, model_date: Optional[dt.date] = None, phase: Optional[str] = None) -> Dict[str, object]:
    state = ensure_margin_calibration_state(state)
    model_date = model_date or today_local()
    phase_name = normalize_model_phase(phase) or season_phase_for_date(model_date)
    spring_profile = dict(state["margin_calibration"].get("spring", identity_margin_calibration()))
    regular_profile = dict(state["margin_calibration"].get("regular", identity_margin_calibration()))
    if phase_name != "regular":
        return spring_profile if phase_name == "spring" else regular_profile

    opening_day = opening_day_for_season(model_date.year)
    transition = state.get("regular_transition") if isinstance(state.get("regular_transition"), dict) else {}
    bridge_days = max(int(_finite(transition.get("bridge_days"), REGULAR_TRANSITION_BRIDGE_DAYS)), 0)
    max_share = float(np.clip(_finite(transition.get("spring_carryover_max"), REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), 0.0, 1.0))
    if opening_day is None or bridge_days <= 0 or model_date < opening_day:
        return regular_profile

    days_since_open = max((model_date - opening_day).days, 0)
    if days_since_open >= bridge_days:
        return regular_profile

    spring_share = max_share * (1.0 - (days_since_open / max(float(bridge_days), 1.0)))
    return blend_margin_calibration_profiles(regular_profile, spring_profile, spring_share, "regular_margin_bridge")


def apply_margin_calibration(projected_margin: float, calibration: Optional[Dict[str, object]]) -> float:
    raw_margin = float(_finite(projected_margin, 0.0))
    if not isinstance(calibration, dict) or not bool(calibration.get("apply")):
        return float(np.clip(raw_margin, -6.5, 6.5))
    intercept = float(_finite(calibration.get("intercept"), 0.0))
    slope = float(_finite(calibration.get("slope"), 1.0))
    calibrated = intercept + (slope * raw_margin)
    return float(np.clip(calibrated, -6.5, 6.5))


def blend_total_calibration_profiles(primary: Dict[str, object], secondary: Dict[str, object], secondary_share: float, label: str) -> Dict[str, object]:
    share = float(np.clip(_finite(secondary_share, 0.0), 0.0, 1.0))
    primary_profile = identity_total_calibration()
    if isinstance(primary, dict):
        primary_profile.update(primary)
    secondary_profile = identity_total_calibration()
    if isinstance(secondary, dict):
        secondary_profile.update(secondary)

    primary_apply = bool(primary_profile.get("apply"))
    secondary_apply = bool(secondary_profile.get("apply"))
    primary_intercept = float(_finite(primary_profile.get("intercept"), 0.0)) if primary_apply else 0.0
    secondary_intercept = float(_finite(secondary_profile.get("intercept"), 0.0)) if secondary_apply else 0.0
    primary_slope = float(_finite(primary_profile.get("slope"), 1.0)) if primary_apply else 1.0
    secondary_slope = float(_finite(secondary_profile.get("slope"), 1.0)) if secondary_apply else 1.0
    primary_tail_slope = float(_finite(primary_profile.get("tail_slope"), 0.0)) if primary_apply else 0.0
    secondary_tail_slope = float(_finite(secondary_profile.get("tail_slope"), 0.0)) if secondary_apply else 0.0
    primary_tail_threshold = float(_finite(primary_profile.get("tail_threshold"), 9.0))
    secondary_tail_threshold = float(_finite(secondary_profile.get("tail_threshold"), 9.0))

    blended = identity_total_calibration()
    blended.update({
        "apply": bool(primary_apply or secondary_apply),
        "mode": label if (primary_apply or secondary_apply) else "identity",
        "intercept": float(((1.0 - share) * primary_intercept) + (share * secondary_intercept)),
        "slope": float(((1.0 - share) * primary_slope) + (share * secondary_slope)),
        "tail_slope": float(((1.0 - share) * primary_tail_slope) + (share * secondary_tail_slope)),
        "tail_threshold": float(((1.0 - share) * primary_tail_threshold) + (share * secondary_tail_threshold)),
        "games": int(round(((1.0 - share) * float(_finite(primary_profile.get("games"), 0.0))) + (share * float(_finite(secondary_profile.get("games"), 0.0))))),
        "raw_rmse": safe_float(primary_profile.get("raw_rmse")) if primary_apply else safe_float(secondary_profile.get("raw_rmse")),
        "calibrated_rmse": safe_float(primary_profile.get("calibrated_rmse")) if primary_apply else safe_float(secondary_profile.get("calibrated_rmse")),
        "reason": f"blended total calibration ({label}) with spring share {share:.0%}",
    })
    if not blended["apply"]:
        blended["reason"] = "identity total mapping"
    return blended


def active_total_calibration(state: Dict, model_date: Optional[dt.date] = None, phase: Optional[str] = None) -> Dict[str, object]:
    state = ensure_total_calibration_state(state)
    model_date = model_date or today_local()
    phase_name = normalize_model_phase(phase) or season_phase_for_date(model_date)
    spring_profile = dict(state["total_calibration"].get("spring", identity_total_calibration()))
    regular_profile = dict(state["total_calibration"].get("regular", identity_total_calibration()))
    if phase_name != "regular":
        return spring_profile if phase_name == "spring" else regular_profile

    opening_day = opening_day_for_season(model_date.year)
    transition = state.get("regular_transition") if isinstance(state.get("regular_transition"), dict) else {}
    bridge_days = max(int(_finite(transition.get("bridge_days"), REGULAR_TRANSITION_BRIDGE_DAYS)), 0)
    max_share = float(np.clip(_finite(transition.get("spring_carryover_max"), REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), 0.0, 1.0))
    if opening_day is None or bridge_days <= 0 or model_date < opening_day:
        return regular_profile

    days_since_open = max((model_date - opening_day).days, 0)
    if days_since_open >= bridge_days:
        return regular_profile

    spring_share = max_share * (1.0 - (days_since_open / max(float(bridge_days), 1.0)))
    return blend_total_calibration_profiles(regular_profile, spring_profile, spring_share, "regular_total_bridge")


def apply_total_calibration(total_runs: float, calibration: Optional[Dict[str, object]]) -> float:
    raw_total = float(_finite(total_runs, 8.5))
    if not isinstance(calibration, dict) or not bool(calibration.get("apply")):
        return float(np.clip(raw_total, 5.5, 14.5))
    intercept = float(_finite(calibration.get("intercept"), 0.0))
    slope = float(_finite(calibration.get("slope"), 1.0))
    tail_slope = float(_finite(calibration.get("tail_slope"), 0.0))
    tail_threshold = float(_finite(calibration.get("tail_threshold"), 9.0))
    excess = max(raw_total - tail_threshold, 0.0)
    calibrated = intercept + (slope * raw_total) + (tail_slope * excess)
    return float(np.clip(calibrated, 5.5, 14.5))


def blend_probability_calibration_profiles(primary: Dict[str, object], secondary: Dict[str, object], secondary_share: float, label: str) -> Dict[str, object]:
    share = float(np.clip(_finite(secondary_share, 0.0), 0.0, 1.0))
    primary_profile = identity_probability_calibration()
    if isinstance(primary, dict):
        primary_profile.update(primary)
    secondary_profile = identity_probability_calibration()
    if isinstance(secondary, dict):
        secondary_profile.update(secondary)

    primary_apply = bool(primary_profile.get("apply"))
    secondary_apply = bool(secondary_profile.get("apply"))
    primary_intercept = float(_finite(primary_profile.get("intercept"), 0.0)) if primary_apply else 0.0
    secondary_intercept = float(_finite(secondary_profile.get("intercept"), 0.0)) if secondary_apply else 0.0
    primary_slope = float(_finite(primary_profile.get("slope"), 1.0)) if primary_apply else 1.0
    secondary_slope = float(_finite(secondary_profile.get("slope"), 1.0)) if secondary_apply else 1.0

    blended = identity_probability_calibration()
    blended.update({
        "apply": bool(primary_apply or secondary_apply),
        "mode": str(label or "identity"),
        "intercept": float(((1.0 - share) * primary_intercept) + (share * secondary_intercept)),
        "slope": float(np.clip(((1.0 - share) * primary_slope) + (share * secondary_slope), 0.25, 1.5)),
        "games": int(max(_finite(primary_profile.get("games"), 0), _finite(secondary_profile.get("games"), 0))),
        "raw_log_loss": safe_float(primary_profile.get("raw_log_loss")),
        "calibrated_log_loss": safe_float(primary_profile.get("calibrated_log_loss")),
        "reason": str(primary_profile.get("reason") or secondary_profile.get("reason") or "identity probability mapping"),
    })
    return blended


def active_probability_calibration(state: Dict, model_date: Optional[dt.date] = None, phase: Optional[str] = None) -> Dict[str, object]:
    state = ensure_probability_calibration_state(state)
    model_date = model_date or today_local()
    phase_name = normalize_model_phase(phase) or season_phase_for_date(model_date)
    spring_profile = dict(state["probability_calibration"].get("spring", identity_probability_calibration()))
    regular_profile = dict(state["probability_calibration"].get("regular", identity_probability_calibration()))

    if phase_name == "spring":
        spring_profile["phase"] = "spring"
        spring_profile["label"] = "spring"
        return spring_profile

    spring_share = regular_transition_spring_share(state, model_date, phase_name)
    if spring_share > 0.0:
        blended = blend_probability_calibration_profiles(regular_profile, spring_profile, spring_share, "regular_bridge")
        blended["phase"] = "regular"
        blended["label"] = "regular bridge"
        return blended

    regular_profile["phase"] = "regular"
    regular_profile["label"] = "regular"
    return regular_profile


def apply_probability_calibration(probability: float, calibration: Optional[Dict[str, object]]) -> float:
    prob = float(np.clip(_finite(probability, 0.5), 1e-6, 1.0 - 1e-6))
    if not isinstance(calibration, dict) or not calibration.get("apply"):
        return prob
    intercept = float(np.clip(_finite(calibration.get("intercept"), 0.0), -1.5, 1.5))
    slope = float(np.clip(_finite(calibration.get("slope"), 1.0), 0.25, 1.5))
    raw_logit = math.log(prob / (1.0 - prob))
    calibrated_logit = np.clip(intercept + (slope * raw_logit), -20.0, 20.0)
    return float(1.0 / (1.0 + np.exp(-calibrated_logit)))
def regular_transition_spring_share(state: Dict, model_date: Optional[dt.date], phase: Optional[str] = None) -> float:
    if model_date is None:
        return 0.0
    phase_name = normalize_model_phase(phase) or season_phase_for_date(model_date)
    if phase_name != "regular":
        return 0.0
    state = ensure_phase_weight_state(state)
    opening_day = opening_day_for_season(model_date.year)
    if opening_day is None or model_date < opening_day:
        return 0.0
    days_since = (model_date - opening_day).days
    bridge_days = int(state.get("regular_transition", {}).get("bridge_days", REGULAR_TRANSITION_BRIDGE_DAYS) or 0)
    if bridge_days <= 0 or days_since >= bridge_days:
        return 0.0
    carryover = float(np.clip(_finite(state.get("regular_transition", {}).get("spring_carryover_max", REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), 0.0, 1.0))
    return float(carryover * max(0.0, 1.0 - (days_since / float(bridge_days))))


def active_weight_profile(state: Dict, model_date: Optional[dt.date] = None, phase: Optional[str] = None) -> Dict[str, object]:
    state = ensure_phase_weight_state(state)
    model_date = model_date or today_local()
    phase_name = normalize_model_phase(phase) or season_phase_for_date(model_date)
    spring_weights = spring_guardrail_active_weights(state)
    regular_weights = regular_earn_back_active_weights(state)
    spring_guardrail = state.get("spring_guardrail") if isinstance(state.get("spring_guardrail"), dict) else {}
    regular_earn_back = state.get("regular_earn_back") if isinstance(state.get("regular_earn_back"), dict) else {}
    if phase_name == "spring":
        return {
            "phase": "spring",
            "label": "spring guarded" if spring_guardrail.get("apply") else "spring",
            "spring_share": 1.0,
            "regular_share": 0.0,
            "weights": dict(spring_weights),
            "guardrail": spring_guardrail if spring_guardrail.get("apply") else None,
            "earn_back": None,
        }
    spring_share = regular_transition_spring_share(state, model_date, phase_name)
    active_weights = blend_weight_sets(regular_weights, spring_weights, spring_share) if spring_share > 0 else dict(regular_weights)
    regular_label = "regular earned back" if regular_earn_back.get("apply") else "regular"
    return {
        "phase": "regular",
        "label": f"{regular_label} bridge" if spring_share > 0 else regular_label,
        "spring_share": spring_share,
        "regular_share": 1.0 - spring_share,
        "weights": active_weights,
        "guardrail": spring_guardrail if (spring_share > 0 and spring_guardrail.get("apply")) else None,
        "earn_back": regular_earn_back if regular_earn_back.get("apply") else None,
    }

def statsapi_person(person_id: Optional[int], timeout: int = 30) -> dict:
    if person_id is None:
        return {}
    try:
        person_id = int(person_id)
    except Exception:
        return {}

    cache = getattr(statsapi_person, "_cache", None)
    if cache is None:
        cache = {}
        statsapi_person._cache = cache
    if person_id in cache:
        return cache[person_id]

    r = requests.get(STATSAPI_PEOPLE_URL.format(person_id=person_id), headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    cache[person_id] = payload
    return payload


def person_handedness(person_id: Optional[int]) -> Dict[str, Optional[str]]:
    payload = statsapi_person(person_id)
    people = payload.get("people") or []
    person = people[0] if isinstance(people, list) and people else {}
    bat_side = str(((person.get("batSide") or {}).get("code") or "")).strip().upper() or None
    pitch_hand = str(((person.get("pitchHand") or {}).get("code") or "")).strip().upper() or None
    return {
        "bat_side": bat_side,
        "pitch_hand": pitch_hand,
    }


def haversine_miles(lat1: Optional[float], lon1: Optional[float], lat2: Optional[float], lon2: Optional[float]) -> float:
    vals = [safe_float(lat1), safe_float(lon1), safe_float(lat2), safe_float(lon2)]
    if any(v is None for v in vals):
        return 0.0
    lat1f, lon1f, lat2f, lon2f = [float(v) for v in vals]
    radius_miles = 3958.7613
    dlat = math.radians(lat2f - lat1f)
    dlon = math.radians(lon2f - lon1f)
    a = (
        math.sin(dlat / 2.0) ** 2 +
        math.cos(math.radians(lat1f)) * math.cos(math.radians(lat2f)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return float(radius_miles * c)

def _collect_game_dicts(obj) -> List[dict]:
    """Recursively collect dicts that look like 'a game' by containing a game_pk."""
    found = []
    if isinstance(obj, dict):
        if ("game_pk" in obj) or ("gamePk" in obj) or ("gamepk" in obj):
            found.append(obj)
        for v in obj.values():
            found.extend(_collect_game_dicts(v))
    elif isinstance(obj, list):
        for it in obj:
            found.extend(_collect_game_dicts(it))
    return found

def canonical_player_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    # Convert "Last, First" -> "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_venue_name(name: Optional[str]) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def venue_key(venue_id: Optional[int], venue_name: Optional[str]) -> str:
    if venue_id is not None and str(venue_id).strip() != "":
        try:
            return f"id:{int(venue_id)}"
        except Exception:
            pass
    return f"name:{normalize_venue_name(venue_name)}"

def probable_pitcher_name_and_id(obj) -> Tuple[Optional[str], Optional[int]]:
    if not isinstance(obj, dict):
        return None, None

    name = obj.get("fullName") or obj.get("name")
    if not name:
        first = obj.get("firstName") or obj.get("first_name")
        last = obj.get("lastName") or obj.get("last_name")
        if isinstance(first, str) and isinstance(last, str) and first.strip() and last.strip():
            name = f"{first.strip()} {last.strip()}"

    return canonical_player_name(name), extract_pitcher_id(obj)


def scoreboard_probables_from_game(g: dict) -> Dict[str, Optional[object]]:
    probs = g.get("probablePitchers") if isinstance(g, dict) else None
    away_name, away_id = probable_pitcher_name_and_id(probs.get("away")) if isinstance(probs, dict) else (None, None)
    home_name, home_id = probable_pitcher_name_and_id(probs.get("home")) if isinstance(probs, dict) else (None, None)
    return {
        "away_pitcher": away_name,
        "away_pitcher_id": away_id,
        "home_pitcher": home_name,
        "home_pitcher_id": home_id,
    }


def probables_from_statsapi_schedule(date_: dt.date) -> Dict[int, Dict[str, object]]:
    payload = statsapi_schedule(date_)
    out: Dict[int, Dict[str, object]] = {}

    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for g in day.get("games") or []:
            if not isinstance(g, dict):
                continue
            game_pk = g.get("gamePk")
            try:
                game_pk = int(game_pk)
            except Exception:
                continue

            teams = g.get("teams") or {}
            away_team = teams.get("away") if isinstance(teams, dict) else {}
            home_team = teams.get("home") if isinstance(teams, dict) else {}
            away_name, away_id = probable_pitcher_name_and_id((away_team or {}).get("probablePitcher"))
            home_name, home_id = probable_pitcher_name_and_id((home_team or {}).get("probablePitcher"))

            out[game_pk] = {
                "away_pitcher": away_name,
                "away_pitcher_id": away_id,
                "home_pitcher": home_name,
                "home_pitcher_id": home_id,
                "scheduled_dt_utc": g.get("gameDate"),
                "venue_id": ((g.get("venue") or {}).get("id") if isinstance(g.get("venue"), dict) else None),
            }

    return out


def starter_candidates_from_probables_map(probables_map: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for payload in probables_map.values():
        if not isinstance(payload, dict):
            continue
        for side in ["away", "home"]:
            name = canonical_player_name(payload.get(f"{side}_pitcher"))
            pid = payload.get(f"{side}_pitcher_id")
            if name is None:
                continue
            try:
                pid = int(pid) if pid is not None and str(pid).strip() != "" else None
            except Exception:
                pid = None
            candidates.append({"pitcher": name, "pitcher_id": pid})
    return candidates


def fill_missing_probables_from_map(
    games: List["GameInfo"],
    probables_map: Dict[int, Dict[str, object]],
    source_label: str,
) -> Tuple[List["GameInfo"], List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    ensure_market_csv_template()
    out: List["GameInfo"] = []
    filled_games = 0

    for gi in games:
        payload = probables_map.get(gi.game_pk, {})
        away_pitcher = gi.away_pitcher or canonical_player_name(payload.get("away_pitcher"))
        home_pitcher = gi.home_pitcher or canonical_player_name(payload.get("home_pitcher"))

        if away_pitcher != gi.away_pitcher or home_pitcher != gi.home_pitcher:
            filled_games += 1

        out.append(GameInfo(
            game_pk=gi.game_pk,
            away=gi.away,
            home=gi.home,
            venue=gi.venue,
            start_time_et=gi.start_time_et,
            away_pitcher=away_pitcher,
            home_pitcher=home_pitcher,
            venue_id=gi.venue_id,
            away_pitcher_id=gi.away_pitcher_id or payload.get("away_pitcher_id"),
            home_pitcher_id=gi.home_pitcher_id or payload.get("home_pitcher_id"),
            scheduled_dt_utc=gi.scheduled_dt_utc or payload.get("scheduled_dt_utc"),
        ))

    if filled_games:
        warnings.append(f"Probable starters supplemented for {filled_games}/{len(games)} games via {source_label}.")

    return out, warnings


def extract_venue_info_from_scoreboard_game(g: dict) -> Tuple[Optional[int], str]:
    teams = g.get("teams", {}) if isinstance(g, dict) else {}
    away_obj = teams.get("away", {}) if isinstance(teams, dict) else {}
    home_obj = teams.get("home", {}) if isinstance(teams, dict) else {}

    venue_obj = None
    if isinstance(home_obj.get("venue"), dict):
        venue_obj = home_obj.get("venue")
    elif isinstance(away_obj.get("venue"), dict):
        venue_obj = away_obj.get("venue")
    elif isinstance(g.get("venue"), dict):
        venue_obj = g.get("venue")

    venue_id = None
    venue_name = None
    if isinstance(venue_obj, dict):
        raw_id = venue_obj.get("id")
        if isinstance(raw_id, int):
            venue_id = raw_id
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            venue_id = int(raw_id.strip())
        if venue_obj.get("name"):
            venue_name = str(venue_obj.get("name"))

    if not venue_name:
        venue_name = "Unknown venue"
    return venue_id, venue_name


def games_from_scoreboard(sb: dict) -> Tuple[List[GameInfo], List[str]]:
    warnings: List[str] = []
    games: List[GameInfo] = []
    
    if not isinstance(sb, dict) or "games" not in sb or not isinstance(sb["games"], list):
        return [], ["Scoreboard-data: missing 'games' list."]

    for g in sb["games"]:
        try:
            game_pk = g.get("gamePk") or g.get("game_pk")
            if not game_pk:
                continue
            game_pk = int(game_pk)

            teams = g.get("teams", {})
            away_obj = teams.get("away", {}) if isinstance(teams, dict) else {}
            home_obj = teams.get("home", {}) if isinstance(teams, dict) else {}

            away = away_obj.get("abbreviation") or away_obj.get("teamName") or away_obj.get("name")
            home = home_obj.get("abbreviation") or home_obj.get("teamName") or home_obj.get("name")

            if not away or not home:
                continue

            venue_id, venue = extract_venue_info_from_scoreboard_game(g)
            probable_info = scoreboard_probables_from_game(g)

            # Start time is nested under the datetime object.
            start_et = "TBD"
            dt_obj = g.get("datetime")
            if isinstance(dt_obj, dict):
                time_s = dt_obj.get("time")
                ampm = dt_obj.get("ampm")
                if isinstance(time_s, str) and time_s.strip():
                    start_et = time_s.strip()
                    if isinstance(ampm, str) and ampm.strip():
                        start_et = f"{start_et} {ampm.strip()}"
                elif dt_obj.get("dateTime"):
                    start_et = str(dt_obj["dateTime"])

            games.append(GameInfo(
                game_pk=game_pk,
                away=str(away),
                home=str(home),
                venue=str(venue),
                start_time_et=str(start_et),
                away_pitcher=probable_info["away_pitcher"],
                home_pitcher=probable_info["home_pitcher"],
                venue_id=venue_id,
                away_pitcher_id=probable_info["away_pitcher_id"],
                home_pitcher_id=probable_info["home_pitcher_id"],
                scheduled_dt_utc=(dt_obj.get("dateTime") if isinstance(dt_obj, dict) else None),
            ))
        except Exception as e:
            warnings.append(f"Scoreboard-data: failed parsing one game: {e}")

    if not games:
        warnings.append("Scoreboard-data: parsed games list but produced 0 usable games (team fields missing).")

    return games, warnings


def probable_starter_candidates_from_scoreboard(sb: dict) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    if not isinstance(sb, dict) or "games" not in sb or not isinstance(sb["games"], list):
        return candidates

    for g in sb["games"]:
        probable_info = scoreboard_probables_from_game(g)
        for side in ["away", "home"]:
            name = probable_info.get(f"{side}_pitcher")
            pid = probable_info.get(f"{side}_pitcher_id")
            if isinstance(name, str) and name.strip():
                candidates.append({
                    "pitcher": canonical_player_name(name),
                    "pitcher_id": pid,
                })

    return candidates


def savant_gf(game_pk: int) -> dict:
    # returns JSON from /gf?game_pk=...
    return fetch_json(SAVANT_GF_URL, {"game_pk": str(int(game_pk))})

def _dig(d: dict, path: List[str]):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def find_name_by_id_in_gf(gf: dict, pid: int) -> Optional[str]:
    """Best-effort BFS search inside gf payload to resolve MLBAM id -> canonical player name."""
    if gf is None:
        return None
    stack = [gf]
    seen = set()
    name_keys = [
        "fullName", "full_name", "name", "player_name",
        "name_display_first_last", "name_first_last", "first_last"
    ]

    while stack:
        cur = stack.pop()
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(cur, dict):
            # MLB API style: {"person": {"id": 123, "fullName": "..."}}
            person = cur.get("person")
            if isinstance(person, dict) and person.get("id") == pid:
                for k in name_keys:
                    v = person.get(k)
                    if isinstance(v, str) and v.strip():
                        return canonical_player_name(v.strip())

            # Direct dict id
            for k in ["id", "mlbam_id", "player_id", "pitcher_id"]:
                v = cur.get(k)
                if v == pid or (isinstance(v, str) and v.strip().isdigit() and int(v.strip()) == pid):
                    for nk in name_keys:
                        nm = cur.get(nk)
                        if isinstance(nm, str) and nm.strip():
                            return canonical_player_name(nm.strip())

            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)

        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    return None

def get_boxscore_pitcher_ids(gf: dict, side: str) -> List[int]:
    """
    side: 'home' or 'away'
    Returns pitcher MLBAM ids used in the game from gf boxscore when possible.
    """
    box = gf.get("boxscore")
    if not isinstance(box, dict):
        return []

    # MLB Stats API-like:
    # box["teams"]["home"]["pitchers"] = [id1, id2, ...]
    ids = _dig(box, ["teams", side, "pitchers"])
    if isinstance(ids, list) and ids:
        out = []
        for x in ids:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                out.append(int(x.strip()))
        return out

    # Fallback (rare): search direct keys
    for k in [f"{side}_pitchers", f"{side}Pitchers"]:
        ids = box.get(k)
        if isinstance(ids, list):
            out = []
            for x in ids:
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.strip().isdigit():
                    out.append(int(x.strip()))
            return out

    return []

def get_pitcher_workload_from_boxscore(gf: dict, pid: int) -> float:
    """
    Returns a workload weight for a pitcher from boxscore:
    prefers battersFaced, else numberOfPitches, else outsRecorded, else 1.0.
    """
    box = gf.get("boxscore")
    if not isinstance(box, dict):
        return 1.0

    players = box.get("players")
    if isinstance(players, dict):
        key = f"ID{pid}"
        p = players.get(key)
        if isinstance(p, dict):
            pitching = _dig(p, ["stats", "pitching"])
            if isinstance(pitching, dict):
                for k in ["battersFaced", "numberOfPitches", "outs", "outsRecorded", "pitchesThrown"]:
                    v = pitching.get(k)
                    if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                        return float(v)

    # If structure differs, just return neutral weight
    return 1.0

def zscore_series(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


MARKET_CSV_COLUMNS = [
    "report_date",
    "game_pk",
    "away_team",
    "home_team",
    "sportsbook",
    "home_ml",
    "away_ml",
    "home_run_line",
    "home_run_line_price",
    "away_run_line",
    "away_run_line_price",
    "total_line",
    "over_price",
    "under_price",
    "is_closing",
    "source",
    "snapshot_ts",
]

MANUAL_MARKET_TEMPLATE_COLUMNS = [
    "report_date",
    "game_pk",
    "away_team",
    "home_team",
    "venue",
    "start_time_et",
    "sportsbook",
    "home_ml",
    "away_ml",
    "home_run_line",
    "home_run_line_price",
    "away_run_line",
    "away_run_line_price",
    "total_line",
    "over_price",
    "under_price",
    "snapshot_ts",
    "source",
    "notes",
]


def market_game_key(game_pk: Optional[int], away_team: Optional[str], home_team: Optional[str]) -> str:
    try:
        if game_pk is not None and str(game_pk).strip() != "":
            return f"pk:{int(game_pk)}"
    except Exception:
        pass
    away = str(away_team or "").strip().upper()
    home = str(home_team or "").strip().upper()
    return f"teams:{away}@{home}"


def manual_market_template_path(report_date: dt.date) -> str:
    return os.path.join(OUT_DIR, f"Manual_Sportsbook_Lines_{report_date.isoformat()}.csv")


def ensure_market_csv_template() -> None:
    if os.path.exists(SPORTSBOOK_CSV_PATH) and os.path.getsize(SPORTSBOOK_CSV_PATH) > 0:
        return
    pd.DataFrame(columns=MARKET_CSV_COLUMNS).to_csv(SPORTSBOOK_CSV_PATH, index=False)


def build_manual_market_template_rows(report_date: dt.date, games: List["GameInfo"]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for g in games:
        rows.append({
            "report_date": report_date.isoformat(),
            "game_pk": int(g.game_pk),
            "away_team": str(g.away),
            "home_team": str(g.home),
            "venue": str(g.venue or ""),
            "start_time_et": str(g.start_time_et or ""),
            "sportsbook": "ManualConsensus",
            "home_ml": "",
            "away_ml": "",
            "home_run_line": "",
            "home_run_line_price": "",
            "away_run_line": "",
            "away_run_line_price": "",
            "total_line": "",
            "over_price": "",
            "under_price": "",
            "snapshot_ts": "",
            "source": "manual_template",
            "notes": "",
        })
    return rows


def ensure_manual_market_template(report_date: dt.date, games: List["GameInfo"]) -> Tuple[str, bool]:
    path = manual_market_template_path(report_date)
    template_df = pd.DataFrame(build_manual_market_template_rows(report_date, games), columns=MANUAL_MARKET_TEMPLATE_COLUMNS)
    created_or_updated = False

    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=MANUAL_MARKET_TEMPLATE_COLUMNS)
        except Exception:
            existing = pd.DataFrame(columns=MANUAL_MARKET_TEMPLATE_COLUMNS)

        for col in MANUAL_MARKET_TEMPLATE_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[MANUAL_MARKET_TEMPLATE_COLUMNS].copy()
        existing["report_date"] = report_date.isoformat()
        existing["sportsbook"] = existing["sportsbook"].fillna("ManualConsensus").astype(str).str.strip()
        existing.loc[existing["sportsbook"] == "", "sportsbook"] = "ManualConsensus"
        existing["source"] = existing["source"].fillna("manual_template").astype(str).str.strip()
        existing.loc[existing["source"] == "", "source"] = "manual_template"
        existing["away_team"] = existing["away_team"].fillna("").astype(str).str.upper().str.strip()
        existing["home_team"] = existing["home_team"].fillna("").astype(str).str.upper().str.strip()
        existing["game_pk"] = pd.to_numeric(existing["game_pk"], errors="coerce").fillna(0).astype(int)

        existing_keys = {
            (int(row["game_pk"]), str(row["away_team"]), str(row["home_team"]), str(row["sportsbook"]))
            for _, row in existing.iterrows()
        }
        additions = []
        for _, row in template_df.iterrows():
            key = (int(row["game_pk"]), str(row["away_team"]), str(row["home_team"]), str(row["sportsbook"]))
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
        final_df["game_pk_sort"] = pd.to_numeric(final_df["game_pk"], errors="coerce").fillna(0).astype(int)
        final_df["start_sort"] = final_df["start_time_et"].fillna("").astype(str)
        final_df = final_df.sort_values(["start_sort", "away_team", "home_team", "sportsbook", "game_pk_sort"]).drop(columns=["game_pk_sort", "start_sort"])
    final_df.to_csv(path, index=False)
    return path, created_or_updated


def ensure_market_db() -> None:
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_ts TEXT NOT NULL,
                report_date TEXT NOT NULL,
                game_pk INTEGER NOT NULL DEFAULT 0,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
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
                    report_date,
                    game_pk,
                    away_team,
                    home_team,
                    sportsbook,
                    market_type,
                    selection,
                    IFNULL(line_value, 0),
                    IFNULL(book_price, 0)
                );
            """
        )

        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(bet_journal)").fetchall()}
        if "model_line_value" not in cols:
            conn.execute("ALTER TABLE bet_journal ADD COLUMN model_line_value REAL")
        if "actionability_label" not in cols:
            conn.execute("ALTER TABLE bet_journal ADD COLUMN actionability_label TEXT")
        if "actionability_score" not in cols:
            conn.execute("ALTER TABLE bet_journal ADD COLUMN actionability_score REAL")
        if "actionability_notes" not in cols:
            conn.execute("ALTER TABLE bet_journal ADD COLUMN actionability_notes TEXT")


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


def import_market_lines_csv(
    report_date: dt.date,
    csv_path: str = SPORTSBOOK_CSV_PATH,
    default_source: str = "manual_csv",
) -> Tuple[int, List[str]]:
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
    except Exception as e:
        return 0, [f"Sportsbook CSV import failed: {e}"]

    if df.empty:
        return 0, warnings

    report_iso = report_date.isoformat()
    if "report_date" not in df.columns:
        df["report_date"] = report_iso
    else:
        df["report_date"] = df["report_date"].fillna(report_iso).astype(str).str.strip()
        df.loc[df["report_date"] == "", "report_date"] = report_iso

    if "snapshot_ts" not in df.columns:
        df["snapshot_ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
    else:
        df["snapshot_ts"] = df["snapshot_ts"].fillna(dt.datetime.now(dt.timezone.utc).isoformat()).astype(str).str.strip()
        df.loc[df["snapshot_ts"] == "", "snapshot_ts"] = dt.datetime.now(dt.timezone.utc).isoformat()

    if "source" not in df.columns:
        df["source"] = default_source
    else:
        df["source"] = df["source"].fillna(default_source).astype(str).str.strip()
        df.loc[df["source"] == "", "source"] = default_source

    if "is_closing" not in df.columns:
        df["is_closing"] = 0

    target = df[df["report_date"].astype(str) == report_iso].copy()
    if target.empty:
        return 0, warnings

    required = ["away_team", "home_team", "sportsbook"]
    missing = [col for col in required if col not in target.columns]
    if missing:
        return 0, [f"Sportsbook CSV missing required columns: {', '.join(missing)}"]

    records: List[Tuple] = []
    source_values: set[str] = set()
    skipped = 0
    for _, row in target.iterrows():
        away_team = str(row.get("away_team", "") or "").strip().upper()
        home_team = str(row.get("home_team", "") or "").strip().upper()
        sportsbook = str(row.get("sportsbook", "") or "").strip()
        if not away_team or not home_team or not sportsbook:
            skipped += 1
            continue
        home_ml = parse_market_int(row.get("home_ml"))
        away_ml = parse_market_int(row.get("away_ml"))
        home_run_line = parse_market_float(row.get("home_run_line"))
        home_run_line_price = parse_market_int(row.get("home_run_line_price"))
        away_run_line = parse_market_float(row.get("away_run_line"))
        away_run_line_price = parse_market_int(row.get("away_run_line_price"))
        total_line = parse_market_float(row.get("total_line"))
        over_price = parse_market_int(row.get("over_price"))
        under_price = parse_market_int(row.get("under_price"))
        if all(v is None for v in [home_ml, away_ml, home_run_line, home_run_line_price, away_run_line, away_run_line_price, total_line, over_price, under_price]):
            skipped += 1
            continue
        game_pk = parse_market_int(row.get("game_pk")) or 0
        source_label = str(row.get("source", default_source) or default_source).strip() or default_source
        source_values.add(source_label)
        records.append((
            str(row.get("snapshot_ts")),
            report_iso,
            game_pk,
            away_team,
            home_team,
            sportsbook,
            home_ml,
            away_ml,
            home_run_line,
            home_run_line_price,
            away_run_line,
            away_run_line_price,
            total_line,
            over_price,
            under_price,
            int(bool(parse_market_int(row.get("is_closing")) or 0)),
            source_label,
        ))

    if not records:
        if skipped:
            warnings.append(f"{os.path.basename(csv_path)} skipped {skipped} incomplete or blank row(s) for {report_iso}.")
        return 0, warnings

    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        for source_label in sorted(source_values):
            conn.execute("DELETE FROM odds_snapshots WHERE report_date = ? AND source = ?", (report_iso, source_label))
        conn.executemany(
            """
            INSERT INTO odds_snapshots (
                snapshot_ts, report_date, game_pk, away_team, home_team, sportsbook,
                home_ml, away_ml, home_run_line, home_run_line_price,
                away_run_line, away_run_line_price, total_line, over_price, under_price,
                is_closing, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    if skipped:
        warnings.append(f"{os.path.basename(csv_path)} skipped {skipped} incomplete or blank row(s) for {report_iso}.")
    return len(records), warnings


def import_all_market_lines(report_date: dt.date, games: List["GameInfo"]) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    ensure_market_csv_template()
    template_path, template_updated = ensure_manual_market_template(report_date, games)
    imported_csv, csv_warn = import_market_lines_csv(report_date, SPORTSBOOK_CSV_PATH, default_source="manual_csv")
    warnings.extend(csv_warn)
    imported_template, template_warn = import_market_lines_csv(report_date, template_path, default_source="manual_template")
    if int(imported_template) == 0:
        template_warn = [w for w in template_warn if not (os.path.basename(template_path) in w and "skipped" in w.lower())]
    warnings.extend(template_warn)
    warnings.append(f"Manual market template ready: {os.path.basename(template_path)}")
    return {
        "manual_csv": int(imported_csv),
        "manual_template": int(imported_template),
        "total": int(imported_csv + imported_template),
        "template_path": template_path,
        "template_updated": bool(template_updated),
    }, warnings


def load_market_consensus_for_date(report_date: dt.date, games: Optional[List["GameInfo"]] = None) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    warnings: List[str] = []
    ensure_market_db()
    report_iso = report_date.isoformat()
    desired_keys = set()
    if games:
        desired_keys = {market_game_key(g.game_pk, g.away, g.home) for g in games}

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
        row_game_pk = int(row["game_pk"]) if row["game_pk"] is not None else 0
        key = market_game_key(row_game_pk if row_game_pk > 0 else None, row["away_team"], row["home_team"])
        if desired_keys and key not in desired_keys:
            continue
        sportsbook = str(row["sportsbook"] or "").strip()
        if not sportsbook:
            continue
        grouped.setdefault(key, {})
        if sportsbook not in grouped[key]:
            grouped[key][sportsbook] = row

    def median_value(items: List[Optional[float]], integer: bool = False):
        vals = [float(v) for v in items if v is not None and np.isfinite(v)]
        if not vals:
            return None
        med = float(np.median(np.array(vals, dtype=float)))
        return int(round(med)) if integer else med

    out: Dict[str, Dict[str, object]] = {}
    for key, book_rows in grouped.items():
        selected = list(book_rows.values())
        if not selected:
            continue
        out[key] = {
            "market_label": f"Consensus ({len(selected)} book{'s' if len(selected) != 1 else ''})",
            "sportsbook_count": len(selected),
            "sportsbooks": sorted(str(r["sportsbook"]) for r in selected),
            "away_team": str(selected[0]["away_team"]),
            "home_team": str(selected[0]["home_team"]),
            "game_pk": int(selected[0]["game_pk"] or 0),
            "home_ml": median_value([parse_market_int(r["home_ml"]) for r in selected], integer=True),
            "away_ml": median_value([parse_market_int(r["away_ml"]) for r in selected], integer=True),
            "home_run_line": median_value([parse_market_float(r["home_run_line"]) for r in selected]),
            "home_run_line_price": median_value([parse_market_int(r["home_run_line_price"]) for r in selected], integer=True),
            "away_run_line": median_value([parse_market_float(r["away_run_line"]) for r in selected]),
            "away_run_line_price": median_value([parse_market_int(r["away_run_line_price"]) for r in selected], integer=True),
            "total_line": median_value([parse_market_float(r["total_line"]) for r in selected]),
            "over_price": median_value([parse_market_int(r["over_price"]) for r in selected], integer=True),
            "under_price": median_value([parse_market_int(r["under_price"]) for r in selected], integer=True),
            "snapshot_ts": max(str(r["snapshot_ts"]) for r in selected),
        }

    return out, warnings


ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_DEFAULT_SPORT = "baseball_mlb"

MLB_TEAM_NAME_ALIASES = {
    "ARIZONA DIAMONDBACKS": "AZ",
    "DIAMONDBACKS": "AZ",
    "ATLANTA BRAVES": "ATL",
    "BRAVES": "ATL",
    "BALTIMORE ORIOLES": "BAL",
    "ORIOLES": "BAL",
    "BOSTON RED SOX": "BOS",
    "RED SOX": "BOS",
    "CHICAGO CUBS": "CHC",
    "CUBS": "CHC",
    "CHICAGO WHITE SOX": "CWS",
    "WHITE SOX": "CWS",
    "CINCINNATI REDS": "CIN",
    "REDS": "CIN",
    "CLEVELAND GUARDIANS": "CLE",
    "GUARDIANS": "CLE",
    "COLORADO ROCKIES": "COL",
    "ROCKIES": "COL",
    "DETROIT TIGERS": "DET",
    "TIGERS": "DET",
    "HOUSTON ASTROS": "HOU",
    "ASTROS": "HOU",
    "KANSAS CITY ROYALS": "KC",
    "ROYALS": "KC",
    "LOS ANGELES ANGELS": "LAA",
    "LA ANGELS": "LAA",
    "ANGELS": "LAA",
    "LOS ANGELES DODGERS": "LAD",
    "LA DODGERS": "LAD",
    "DODGERS": "LAD",
    "MIAMI MARLINS": "MIA",
    "MARLINS": "MIA",
    "MILWAUKEE BREWERS": "MIL",
    "BREWERS": "MIL",
    "MINNESOTA TWINS": "MIN",
    "TWINS": "MIN",
    "NEW YORK METS": "NYM",
    "METS": "NYM",
    "NEW YORK YANKEES": "NYY",
    "YANKEES": "NYY",
    "ATHLETICS": "ATH",
    "OAKLAND ATHLETICS": "ATH",
    "SACRAMENTO ATHLETICS": "ATH",
    "AS": "ATH",
    "PHILADELPHIA PHILLIES": "PHI",
    "PHILLIES": "PHI",
    "PITTSBURGH PIRATES": "PIT",
    "PIRATES": "PIT",
    "SAN DIEGO PADRES": "SD",
    "PADRES": "SD",
    "SAN FRANCISCO GIANTS": "SF",
    "GIANTS": "SF",
    "SEATTLE MARINERS": "SEA",
    "MARINERS": "SEA",
    "ST LOUIS CARDINALS": "STL",
    "SAINT LOUIS CARDINALS": "STL",
    "CARDINALS": "STL",
    "TAMPA BAY RAYS": "TB",
    "RAYS": "TB",
    "TEXAS RANGERS": "TEX",
    "RANGERS": "TEX",
    "TORONTO BLUE JAYS": "TOR",
    "BLUE JAYS": "TOR",
    "WASHINGTON NATIONALS": "WSH",
    "NATIONALS": "WSH",
}
MLB_TEAM_ABBREVIATIONS = sorted(set(MLB_TEAM_NAME_ALIASES.values()))


def normalize_team_name_key(value: Optional[object]) -> str:
    txt = str(value or "").strip().upper()
    txt = txt.replace("&", "AND")
    txt = re.sub(r"[^A-Z0-9 ]+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def mlb_team_abbreviation(name: Optional[object]) -> Optional[str]:
    key = normalize_team_name_key(name)
    if not key:
        return None
    if key in MLB_TEAM_ABBREVIATIONS:
        return key
    return MLB_TEAM_NAME_ALIASES.get(key)


def median_market_value(items: List[Optional[float]], integer: bool = False):
    vals = [float(v) for v in items if v is not None and np.isfinite(v)]
    if not vals:
        return None
    med = float(np.median(np.array(vals, dtype=float)))
    return int(round(med)) if integer else med


def select_latest_market_rows(rows: List[sqlite3.Row]) -> List[sqlite3.Row]:
    grouped: Dict[str, sqlite3.Row] = {}
    for row in rows:
        sportsbook = str(row["sportsbook"] or "").strip()
        if not sportsbook:
            continue
        if sportsbook not in grouped:
            grouped[sportsbook] = row
    return list(grouped.values())


def load_market_consensus_row_from_conn(
    conn: sqlite3.Connection,
    report_iso: str,
    away_team: str,
    home_team: str,
    game_pk: int = 0,
    closing_only: bool = False,
) -> Optional[Dict[str, object]]:
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
        closing_rows = [row for row in rows if int(row["is_closing"] or 0) == 1]
        if closing_rows:
            selected_rows = closing_rows
    selected = select_latest_market_rows(selected_rows)
    if not selected:
        return None

    return {
        "market_label": f"Consensus ({len(selected)} book{'s' if len(selected) != 1 else ''})",
        "sportsbook_count": len(selected),
        "sportsbooks": sorted(str(r["sportsbook"]) for r in selected),
        "away_team": str(selected[0]["away_team"]),
        "home_team": str(selected[0]["home_team"]),
        "game_pk": int(selected[0]["game_pk"] or 0),
        "home_ml": median_market_value([parse_market_int(r["home_ml"]) for r in selected], integer=True),
        "away_ml": median_market_value([parse_market_int(r["away_ml"]) for r in selected], integer=True),
        "home_run_line": median_market_value([parse_market_float(r["home_run_line"]) for r in selected]),
        "home_run_line_price": median_market_value([parse_market_int(r["home_run_line_price"]) for r in selected], integer=True),
        "away_run_line": median_market_value([parse_market_float(r["away_run_line"]) for r in selected]),
        "away_run_line_price": median_market_value([parse_market_int(r["away_run_line_price"]) for r in selected], integer=True),
        "total_line": median_market_value([parse_market_float(r["total_line"]) for r in selected]),
        "over_price": median_market_value([parse_market_int(r["over_price"]) for r in selected], integer=True),
        "under_price": median_market_value([parse_market_int(r["under_price"]) for r in selected], integer=True),
        "snapshot_ts": max(str(r["snapshot_ts"]) for r in selected),
    }


def match_market_event_to_game(event: Dict[str, object], games: List["GameInfo"]) -> Dict[str, object]:
    away_abbr = mlb_team_abbreviation(event.get("away_team"))
    home_abbr = mlb_team_abbreviation(event.get("home_team"))
    if away_abbr is None:
        away_abbr = normalize_team_name_key(event.get("away_team"))[:8]
    if home_abbr is None:
        home_abbr = normalize_team_name_key(event.get("home_team"))[:8]

    candidates = [g for g in games if g.away == away_abbr and g.home == home_abbr]
    if not candidates:
        return {"game_pk": 0, "away": away_abbr, "home": home_abbr}
    if len(candidates) == 1:
        g = candidates[0]
        return {"game_pk": g.game_pk, "away": g.away, "home": g.home}

    event_dt = parse_utc_datetime(event.get("commence_time"))
    if event_dt is not None:
        def game_time_distance(game: "GameInfo") -> float:
            game_dt = parse_utc_datetime(game.scheduled_dt_utc)
            if game_dt is None:
                return 1e18
            return abs((game_dt - event_dt).total_seconds())
        g = sorted(candidates, key=game_time_distance)[0]
    else:
        g = candidates[0]
    return {"game_pk": g.game_pk, "away": g.away, "home": g.home}


def parse_odds_api_book_row(event: Dict[str, object], bookmaker: Dict[str, object], games: List["GameInfo"], report_date: dt.date) -> Optional[Tuple]:
    matched = match_market_event_to_game(event, games)
    away_team = str(matched.get("away") or "").strip().upper()
    home_team = str(matched.get("home") or "").strip().upper()
    if not away_team or not home_team:
        return None

    game_pk = int(matched.get("game_pk") or 0)
    if game_pk <= 0:
        return None

    snapshot_ts = str(bookmaker.get("last_update") or event.get("commence_time") or dt.datetime.now(dt.timezone.utc).isoformat())
    sportsbook = str(bookmaker.get("title") or bookmaker.get("key") or "").strip()
    if not sportsbook:
        return None

    row = {
        "home_ml": None,
        "away_ml": None,
        "home_run_line": None,
        "home_run_line_price": None,
        "away_run_line": None,
        "away_run_line_price": None,
        "total_line": None,
        "over_price": None,
        "under_price": None,
    }

    for market in bookmaker.get("markets") or []:
        market_key = str((market or {}).get("key") or "").strip().lower()
        outcomes = market.get("outcomes") or []
        if market_key == "h2h":
            for outcome in outcomes:
                team_abbr = mlb_team_abbreviation(outcome.get("name"))
                price = parse_market_int(outcome.get("price"))
                if team_abbr == home_team:
                    row["home_ml"] = price
                elif team_abbr == away_team:
                    row["away_ml"] = price
        elif market_key == "spreads":
            for outcome in outcomes:
                team_abbr = mlb_team_abbreviation(outcome.get("name"))
                point = parse_market_float(outcome.get("point"))
                price = parse_market_int(outcome.get("price"))
                if team_abbr == home_team:
                    row["home_run_line"] = point
                    row["home_run_line_price"] = price
                elif team_abbr == away_team:
                    row["away_run_line"] = point
                    row["away_run_line_price"] = price
        elif market_key == "totals":
            for outcome in outcomes:
                label = str(outcome.get("name") or "").strip().lower()
                point = parse_market_float(outcome.get("point"))
                price = parse_market_int(outcome.get("price"))
                if label == "over":
                    row["total_line"] = point
                    row["over_price"] = price
                elif label == "under":
                    if row["total_line"] is None:
                        row["total_line"] = point
                    row["under_price"] = price

    if all(row[key] is None for key in row.keys()):
        return None

    return (
        snapshot_ts,
        report_date.isoformat(),
        int(matched.get("game_pk") or 0),
        away_team,
        home_team,
        sportsbook,
        row["home_ml"],
        row["away_ml"],
        row["home_run_line"],
        row["home_run_line_price"],
        row["away_run_line"],
        row["away_run_line_price"],
        row["total_line"],
        row["over_price"],
        row["under_price"],
        0,
        "odds_api",
    )


def fetch_market_lines_from_odds_api(report_date: dt.date, games: List["GameInfo"]) -> Tuple[int, List[str]]:
    api_key = str(os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or "").strip()
    if not api_key:
        return 0, []

    sport = str(os.getenv("ODDS_API_SPORT") or ODDS_API_DEFAULT_SPORT).strip() or ODDS_API_DEFAULT_SPORT
    regions = str(os.getenv("ODDS_API_REGIONS") or "us").strip() or "us"
    markets = str(os.getenv("ODDS_API_MARKETS") or "h2h,spreads,totals").strip() or "h2h,spreads,totals"
    bookmakers = str(os.getenv("ODDS_API_BOOKMAKERS") or "").strip()
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    warnings: List[str] = []
    try:
        resp = requests.get(
            f"{ODDS_API_BASE_URL}/sports/{sport}/odds",
            params=params,
            headers=HEADERS,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return 0, [f"Live sportsbook fetch failed: {e}"]

    if not isinstance(payload, list):
        return 0, ["Live sportsbook fetch returned an unexpected payload."]

    records: List[Tuple] = []
    unmatched_books = 0
    for event in payload:
        if not isinstance(event, dict):
            continue
        for bookmaker in event.get("bookmakers") or []:
            if not isinstance(bookmaker, dict):
                continue
            record = parse_odds_api_book_row(event, bookmaker, games, report_date)
            if record is None:
                unmatched_books += 1
                continue
            records.append(record)

    if not records:
        if unmatched_books:
            warnings.append(f"Live sportsbook fetch returned {unmatched_books} bookmaker row(s), but none matched today's scheduled games.")
        return 0, warnings

    ensure_market_db()
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO odds_snapshots (
                snapshot_ts, report_date, game_pk, away_team, home_team, sportsbook,
                home_ml, away_ml, home_run_line, home_run_line_price,
                away_run_line, away_run_line_price, total_line, over_price, under_price,
                is_closing, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    if unmatched_books:
        warnings.append(f"Live sportsbook fetch skipped {unmatched_books} unmatched bookmaker row(s).")
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        warnings.append(f"Live sportsbook fetch credits remaining: {remaining}.")
    return len(records), warnings


def mark_closing_market_snapshots(report_date: dt.date, games: List["GameInfo"]) -> Tuple[int, List[str]]:
    ensure_market_db()
    now_utc = dt.datetime.now(dt.timezone.utc)
    report_iso = report_date.isoformat()
    marked = 0

    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for g in games:
            scheduled = parse_utc_datetime(g.scheduled_dt_utc or start_time_et_to_utc_iso(report_date, g.start_time_et))
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


def payout_units_from_american(price: Optional[int], stake_units: float, result: str) -> Optional[float]:
    american = parse_market_int(price)
    stake = float(_finite(stake_units, 1.0))
    if american is None:
        return None
    if result == "push":
        return 0.0
    if result == "lose":
        return -stake
    if result != "win":
        return None
    if american > 0:
        return stake * (american / 100.0)
    return stake * (100.0 / abs(american))


def compute_bet_clv(conn: sqlite3.Connection, row: sqlite3.Row) -> Optional[float]:
    report_iso = str(row["report_date"])
    away_team = str(row["away_team"])
    home_team = str(row["home_team"])
    game_pk = int(row["game_pk"] or 0)
    market_type = str(row["market_type"] or "")
    selection = str(row["selection"] or "")
    closing = load_market_consensus_row_from_conn(conn, report_iso, away_team, home_team, game_pk=game_pk, closing_only=True)
    if closing is None:
        closing = load_market_consensus_row_from_conn(conn, report_iso, away_team, home_team, game_pk=game_pk, closing_only=False)
    if closing is None:
        return None

    market_probability = safe_float(row["market_probability"])
    line_value = safe_float(row["line_value"])
    if market_type == "moneyline":
        selected_team = selection.strip().upper()
        closing_home_prob, closing_away_prob = no_vig_probabilities(closing.get("home_ml"), closing.get("away_ml"))
        closing_prob = closing_home_prob if selected_team == home_team else closing_away_prob
        if closing_prob is None:
            return None
        if market_probability is not None and np.isfinite(market_probability):
            return float(closing_prob - market_probability)
        open_prob = american_implied_probability(parse_market_int(row["book_price"]))
        return None if open_prob is None else float(closing_prob - open_prob)

    if market_type == "run_line":
        m = re.match(r"^([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?)$", selection.strip().upper())
        if not m:
            return None
        team_code = m.group(1)
        bet_line = float(m.group(2))
        closing_home_prob, closing_away_prob = no_vig_probabilities(closing.get("home_run_line_price"), closing.get("away_run_line_price"))
        closing_line = closing.get("home_run_line") if team_code == home_team else closing.get("away_run_line")
        closing_prob = closing_home_prob if team_code == home_team else closing_away_prob
        if closing_line is not None and abs(float(closing_line) - bet_line) > 0.01:
            return float((float(closing_line) - bet_line) * (1.0 if bet_line > 0 else -1.0))
        if closing_prob is None:
            return None
        if market_probability is not None and np.isfinite(market_probability):
            return float(closing_prob - market_probability)
        open_prob = american_implied_probability(parse_market_int(row["book_price"]))
        return None if open_prob is None else float(closing_prob - open_prob)

    if market_type == "total":
        m = re.match(r"^(OVER|UNDER)\s+(\d+(?:\.\d+)?)$", selection.strip().upper())
        if not m:
            return None
        side = m.group(1)
        bet_total = float(m.group(2))
        closing_total = safe_float(closing.get("total_line"))
        if closing_total is None:
            return None
        if side == "OVER":
            return float(closing_total - bet_total)
        return float(bet_total - closing_total)

    return None


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
        for iso in sorted({str(row["report_date"]) for row in rows}):
            try:
                target_date = dt.date.fromisoformat(iso)
            except Exception:
                continue
            finals_df, final_warn = backtest_finals_from_scoreboard(target_date)
            warnings.extend(final_warn)
            keyed: Dict[Tuple[str, str], Dict[str, object]] = {}
            for _, final_row in finals_df.iterrows():
                keyed[(str(final_row["away_team"]), str(final_row["home_team"]))] = {
                    "away_score": int(final_row["away_score"]),
                    "home_score": int(final_row["home_score"]),
                }
            finals_by_date[iso] = keyed

        for row in rows:
            report_iso = str(row["report_date"])
            away_team = str(row["away_team"])
            home_team = str(row["home_team"])
            final = finals_by_date.get(report_iso, {}).get((away_team, home_team))
            if final is None:
                continue

            away_score = int(final["away_score"])
            home_score = int(final["home_score"])
            market_type = str(row["market_type"] or "")
            selection = str(row["selection"] or "")
            result = None

            if market_type == "moneyline":
                winner = home_team if home_score > away_score else away_team
                result = "win" if selection.strip().upper() == winner else "lose"
            elif market_type == "run_line":
                m = re.match(r"^([A-Z]{2,3})\s+([+-]?\d+(?:\.\d+)?)$", selection.strip().upper())
                if m:
                    team_code = m.group(1)
                    line_value = float(m.group(2))
                    selected_score = home_score if team_code == home_team else away_score
                    opp_score = away_score if team_code == home_team else home_score
                    adjusted = selected_score + line_value
                    if abs(adjusted - opp_score) < 1e-9:
                        result = "push"
                    elif adjusted > opp_score:
                        result = "win"
                    else:
                        result = "lose"
            elif market_type == "total":
                m = re.match(r"^(OVER|UNDER)\s+(\d+(?:\.\d+)?)$", selection.strip().upper())
                if m:
                    side = m.group(1)
                    total_line = float(m.group(2))
                    total_runs = away_score + home_score
                    if abs(total_runs - total_line) < 1e-9:
                        result = "push"
                    elif side == "OVER":
                        result = "win" if total_runs > total_line else "lose"
                    else:
                        result = "win" if total_runs < total_line else "lose"
            if result is None:
                continue

            stake_units = safe_float(row["stake_units"])
            if stake_units is None or not np.isfinite(stake_units):
                stake_units = 1.0
            result_units = payout_units_from_american(parse_market_int(row["book_price"]), stake_units, result)
            status = "graded" if str(row["status"] or "watch") == "watch" else "settled"
            clv_value = compute_bet_clv(conn, row)
            conn.execute(
                """
                UPDATE bet_journal
                SET result_units = ?, clv_value = ?, settled_ts = ?, status = ?
                WHERE id = ?
                """,
                (
                    result_units,
                    clv_value,
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                    status,
                    int(row["id"]),
                ),
            )
            settled += 1

    return settled, warnings
def pick_pitcher_name(obj) -> Optional[str]:
    """Best-effort: handle dicts or strings and normalize names consistently."""
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return canonical_player_name(s) if s else None
    if isinstance(obj, dict):
        for k in [
            "player_name", "name", "fullName", "full_name",
            "name_display_first_last", "name_display_roster",
            "name_first_last", "first_last", "last_first",
        ]:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return canonical_player_name(v.strip())
        first = obj.get("first_name") or obj.get("firstName")
        last = obj.get("last_name") or obj.get("lastName")
        if isinstance(first, str) and isinstance(last, str) and first.strip() and last.strip():
            return canonical_player_name(f"{first.strip()} {last.strip()}")
    return None


def starters_from_gf(game_pk: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (away_starter_name, home_starter_name) using Savant gf payload.
    Handles starter ids as well as direct name objects.
    """
    gf = savant_gf(game_pk)

    def resolve_pitcher(obj) -> Optional[str]:
        if obj is None:
            return None
        pid = None
        if isinstance(obj, int):
            pid = obj
        elif isinstance(obj, str) and obj.strip().isdigit():
            pid = int(obj.strip())
        elif isinstance(obj, dict):
            for k in ["id", "mlbam_id", "player_id", "pitcher_id"]:
                v = obj.get(k)
                if isinstance(v, int):
                    pid = v
                    break
                if isinstance(v, str) and v.strip().isdigit():
                    pid = int(v.strip())
                    break
        if pid is not None:
            resolved = find_name_by_id_in_gf(gf, pid)
            if resolved:
                return resolved
        return pick_pitcher_name(obj)

    away_p = None
    home_p = None

    hpl = gf.get("home_pitcher_lineup")
    apl = gf.get("away_pitcher_lineup")
    if isinstance(hpl, list) and hpl:
        home_p = resolve_pitcher(hpl[0])
    if isinstance(apl, list) and apl:
        away_p = resolve_pitcher(apl[0])

    if home_p is None:
        hp = gf.get("home_pitchers")
        if isinstance(hp, list) and hp:
            home_p = resolve_pitcher(hp[0])
    if away_p is None:
        ap = gf.get("away_pitchers")
        if isinstance(ap, list) and ap:
            away_p = resolve_pitcher(ap[0])

    return canonical_player_name(away_p), canonical_player_name(home_p)


def enrich_probables_from_gf(games: List[GameInfo]) -> Tuple[List[GameInfo], List[str]]:
    warnings: List[str] = []
    out: List[GameInfo] = []
    enriched_games = 0

    for gi in games:
        away_p = gi.away_pitcher
        home_p = gi.home_pitcher
        try:
            gf_away, gf_home = starters_from_gf(gi.game_pk)
            gf_away = canonical_player_name(gf_away)
            gf_home = canonical_player_name(gf_home)
            if gf_away and gf_away != away_p:
                away_p = gf_away
            if gf_home and gf_home != home_p:
                home_p = gf_home
            if (gf_away and gf_away != gi.away_pitcher) or (gf_home and gf_home != gi.home_pitcher):
                enriched_games += 1
        except Exception as e:
            if not gi.away_pitcher or not gi.home_pitcher:
                warnings.append(f"GF probable starter enrichment failed for game_pk={gi.game_pk}: {e}")

        out.append(GameInfo(
            game_pk=gi.game_pk,
            away=gi.away,
            home=gi.home,
            venue=gi.venue,
            start_time_et=gi.start_time_et,
            away_pitcher=away_p,
            home_pitcher=home_p,
            venue_id=gi.venue_id,
            away_pitcher_id=gi.away_pitcher_id,
            home_pitcher_id=gi.home_pitcher_id,
            scheduled_dt_utc=gi.scheduled_dt_utc,
        ))

    if enriched_games:
        warnings.append(f"Probable starters enriched for {enriched_games}/{len(games)} games via Savant gf.")
    return out, warnings


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def normal_cdf(x: float) -> float:
    value = float(_finite(x, 0.0))
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
def log_loss(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
# -----------------------------
# MODEL STATE + BACKTESTING (SAVANT-ONLY)
# -----------------------------

def default_model_state() -> Dict:
    return ensure_margin_calibration_state(ensure_total_calibration_state(ensure_probability_calibration_state(ensure_phase_weight_state({
        "version": 2,
        "weights": dict(BASE_WEIGHT_DEFAULTS),
        "weights_by_phase": {
            "spring": dict(BASE_WEIGHT_DEFAULTS),
            "regular": dict(BASE_WEIGHT_DEFAULTS),
        },
        "regular_transition": {
            "bridge_days": REGULAR_TRANSITION_BRIDGE_DAYS,
            "spring_carryover_max": REGULAR_TRANSITION_SPRING_CARRYOVER_MAX,
        },
        "last_update_date": None,
        "backtest_refresh": {
            "version": 0,
            "completed_dates": []
        },
        "totals_refresh": {
            "version": 0,
            "completed_dates": []
        },
    }))))


def normalize_backtest_date_str(value) -> Optional[str]:
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _normalize_refresh_state(state: Dict, key: str) -> Dict:
    refresh = state.get(key)
    if not isinstance(refresh, dict):
        refresh = {}
    raw_version = safe_float(refresh.get("version"))
    refresh_version = int(raw_version) if raw_version is not None else 0
    completed_raw = refresh.get("completed_dates", [])
    completed_dates: List[str] = []
    if isinstance(completed_raw, list):
        seen = set()
        for value in completed_raw:
            iso = normalize_backtest_date_str(value)
            if iso and iso not in seen:
                seen.add(iso)
                completed_dates.append(iso)
    refresh["version"] = refresh_version
    refresh["completed_dates"] = sorted(completed_dates)
    state[key] = refresh
    return state


def ensure_backtest_refresh_state(state: Dict) -> Dict:
    return _normalize_refresh_state(state, "backtest_refresh")


def ensure_totals_refresh_state(state: Dict) -> Dict:
    return _normalize_refresh_state(state, "totals_refresh")

def backtest_refresh_requires_full_rebuild(state: Dict, version: int = BACKTEST_FEATURE_VERSION) -> bool:
    state = ensure_backtest_refresh_state(state)
    refresh = state.get("backtest_refresh", {}) if isinstance(state, dict) else {}
    return int(refresh.get("version", 0) or 0) < int(version)

def mark_backtest_refresh_dates(state: Dict, dates: List[object], version: int = BACKTEST_FEATURE_VERSION) -> Dict:
    state = ensure_backtest_refresh_state(state)
    refresh = state["backtest_refresh"]
    if int(refresh.get("version", 0)) != int(version):
        refresh["completed_dates"] = []
    completed = set(refresh.get("completed_dates", []))
    for value in dates:
        iso = normalize_backtest_date_str(value)
        if iso:
            completed.add(iso)
    refresh["version"] = int(version)
    refresh["completed_dates"] = sorted(completed)
    return state

def mark_totals_refresh_dates(state: Dict, dates: List[object], version: int = TOTALS_FEATURE_VERSION) -> Dict:
    state = ensure_totals_refresh_state(state)
    refresh = state["totals_refresh"]
    if int(refresh.get("version", 0)) != int(version):
        refresh["completed_dates"] = []
    completed = set(refresh.get("completed_dates", []))
    for value in dates:
        iso = normalize_backtest_date_str(value)
        if iso:
            completed.add(iso)
    refresh["version"] = int(version)
    refresh["completed_dates"] = sorted(completed)
    return state

def load_model_state() -> Dict:
    st = default_model_state()
    if os.path.exists(MODEL_STATE_PATH):
        with open(MODEL_STATE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            st.update({
                k: v for k, v in loaded.items()
                if k not in [
                    "weights",
                    "weights_by_phase",
                    "regular_transition",
                    "backtest_refresh",
                    "totals_refresh",
                    "probability_calibration",
                    "total_calibration",
                    "margin_calibration",
                ]
            })
            if isinstance(loaded.get("weights"), dict):
                st["weights"].update(loaded["weights"])
            if isinstance(loaded.get("weights_by_phase"), dict):
                st["weights_by_phase"].update(loaded["weights_by_phase"])
            if isinstance(loaded.get("regular_transition"), dict):
                st["regular_transition"].update(loaded["regular_transition"])
            if isinstance(loaded.get("backtest_refresh"), dict):
                st["backtest_refresh"].update(loaded["backtest_refresh"])
            if isinstance(loaded.get("totals_refresh"), dict):
                st["totals_refresh"].update(loaded["totals_refresh"])
            if isinstance(loaded.get("probability_calibration"), dict):
                st["probability_calibration"] = copy.deepcopy(loaded["probability_calibration"])
            if isinstance(loaded.get("total_calibration"), dict):
                st["total_calibration"] = copy.deepcopy(loaded["total_calibration"])
            if isinstance(loaded.get("margin_calibration"), dict):
                st["margin_calibration"] = copy.deepcopy(loaded["margin_calibration"])

        st["version"] = 2
        st = update_spring_guardrail(
            ensure_totals_refresh_state(
                ensure_margin_calibration_state(
                    ensure_total_calibration_state(
                        ensure_probability_calibration_state(
                            ensure_backtest_refresh_state(ensure_phase_weight_state(st))
                        )
                    )
                )
            )
        )
        st = update_regular_feature_earn_back(st)
        st = update_probability_calibration(st)
        st = update_total_calibration(st)
        return update_margin_calibration(st)
    save_model_state(st)
    st["version"] = 2
    st = update_spring_guardrail(
        ensure_totals_refresh_state(
            ensure_margin_calibration_state(
                ensure_total_calibration_state(
                    ensure_probability_calibration_state(
                        ensure_backtest_refresh_state(ensure_phase_weight_state(st))
                    )
                )
            )
        )
    )
    st = update_regular_feature_earn_back(st)
    st = update_probability_calibration(st)
    st = update_total_calibration(st)
    return update_margin_calibration(st)


def save_model_state(state: Dict) -> None:
    payload = ensure_totals_refresh_state(
        ensure_margin_calibration_state(
            ensure_total_calibration_state(
                ensure_probability_calibration_state(
                    ensure_backtest_refresh_state(ensure_phase_weight_state(state))
                )
            )
        )
    )
    if isinstance(state.get("probability_calibration"), dict):
        payload["probability_calibration"] = copy.deepcopy(state["probability_calibration"])
    if isinstance(state.get("total_calibration"), dict):
        payload["total_calibration"] = copy.deepcopy(state["total_calibration"])
    if isinstance(state.get("margin_calibration"), dict):
        payload["margin_calibration"] = copy.deepcopy(state["margin_calibration"])
    if isinstance(state.get("totals_refresh"), dict):
        payload["totals_refresh"] = copy.deepcopy(state["totals_refresh"])
    payload["version"] = 2
    payload["weights"] = dict(payload["weights_by_phase"]["regular"])
    with open(MODEL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finals_from_gf(gf: dict) -> Optional[Dict[str, object]]:
    if not isinstance(gf, dict):
        return None

    def dig(d, path):
        cur = d
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur

    # Try team identifiers from multiple likely locations
    away = (
        dig(gf, ["away_team_data", "abbrev"]) or dig(gf, ["away_team_data", "name"]) or
        dig(gf, ["game_data", "teams", "away", "abbreviation"]) or dig(gf, ["game_data", "teams", "away", "name"])
    )
    home = (
        dig(gf, ["home_team_data", "abbrev"]) or dig(gf, ["home_team_data", "name"]) or
        dig(gf, ["game_data", "teams", "home", "abbreviation"]) or dig(gf, ["game_data", "teams", "home", "name"])
    )

    # Scores: probe common scoreboard/linescore locations
    home_score = (
        dig(gf, ["scoreboard", "home_score"]) or dig(gf, ["scoreboard", "homeScore"]) or dig(gf, ["scoreboard", "home"]) or
        dig(gf, ["linescore", "home_team_runs"]) or dig(gf, ["linescore", "home_runs"]) or
        dig(gf, ["liveData", "linescore", "teams", "home", "runs"])
    )
    away_score = (
        dig(gf, ["scoreboard", "away_score"]) or dig(gf, ["scoreboard", "awayScore"]) or dig(gf, ["scoreboard", "away"]) or
        dig(gf, ["linescore", "away_team_runs"]) or dig(gf, ["linescore", "away_runs"]) or
        dig(gf, ["liveData", "linescore", "teams", "away", "runs"])
    )

    status = (
        dig(gf, ["scoreboard", "status"]) or dig(gf, ["scoreboard", "game_status"]) or
        dig(gf, ["game_data", "status", "abstractGameState"]) or dig(gf, ["game_data", "status", "detailedState"])
    )

    hs = safe_float(home_score)
    aas = safe_float(away_score)
    if hs is None or aas is None:
        return None

    return {
        "home": str(home) if home is not None else "HOME",
        "away": str(away) if away is not None else "AWAY",
        "home_score": int(round(hs)),
        "away_score": int(round(aas)),
        "status": str(status) if status is not None else ""
    }


def backtest_finals_from_scoreboard(date_: dt.date) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    sb = savant_scoreboard(date_)

    if not isinstance(sb, dict) or "games" not in sb or not isinstance(sb["games"], list):
        return pd.DataFrame(columns=["away_team", "home_team", "away_score", "home_score", "status", "game_pk", "venue_id", "venue"]), \
               ["Backtest: scoreboard-data missing games list."]

    rows = []
    for g in sb["games"]:
        try:
            game_pk = g.get("gamePk") or g.get("game_pk")
            if not game_pk:
                continue
            game_pk = int(game_pk)

            status = ""
            if isinstance(g.get("status"), dict):
                status = g["status"].get("abstractGameState") or g["status"].get("detailedState") or ""

            teams = g.get("teams", {})
            away = teams.get("away", {}).get("abbreviation") if isinstance(teams, dict) else None
            home = teams.get("home", {}).get("abbreviation") if isinstance(teams, dict) else None

            linescore = g.get("linescore", {})
            # Finals live here:
            away_runs = None
            home_runs = None
            if isinstance(linescore, dict) and isinstance(linescore.get("teams"), dict):
                away_runs = linescore["teams"].get("away", {}).get("runs")
                home_runs = linescore["teams"].get("home", {}).get("runs")

            if away is None or home is None or away_runs is None or home_runs is None:
                continue

            venue_id, venue = extract_venue_info_from_scoreboard_game(g)
            rows.append({
                "game_pk": game_pk,
                "away_team": str(away),
                "home_team": str(home),
                "away_score": int(away_runs),
                "home_score": int(home_runs),
                "status": str(status),
                "venue_id": venue_id,
                "venue": str(venue),
            })
        except Exception as e:
            warnings.append(f"Backtest: failed parsing one game for finals: {e}")

    if not rows:
        warnings.append("Backtest: scoreboard-data produced 0 finals rows (no runs found in linescore).")

    return pd.DataFrame(rows), warnings

def append_backtest_log(rows: List[Dict]) -> None:
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    if os.path.exists(BACKTEST_LOG_PATH):
        try:
            existing_df = pd.read_csv(BACKTEST_LOG_PATH)
        except Exception:
            existing_df = pd.DataFrame(columns=new_df.columns)
        df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df

    key_cols = [c for c in ["date", "away", "home"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last")
        df = df.sort_values(key_cols).reset_index(drop=True)

    df.to_csv(BACKTEST_LOG_PATH, mode="w", header=True, index=False)


def load_backtest_log() -> pd.DataFrame:
    cols = [
        "date", "phase", "away", "home", "venue_id", "venue", "home_score", "away_score", "y_home", "p_home", "pred_home",
        "projected_total", "away_expected_runs", "home_expected_runs", "projected_margin_raw", "projected_margin",
        "logit", "x_off", "x_starter", "x_bullpen", "x_park", "x_weather", "x_context", "W_OFF", "W_ST", "W_BP", "W_PK", "W_WX", "W_CTX", "HFA"
    ]
    if not os.path.exists(BACKTEST_LOG_PATH):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(BACKTEST_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    phase_map: Dict[str, str] = {}
    if "date" in df.columns:
        for value in df["date"].dropna().astype(str).unique().tolist():
            iso = normalize_backtest_date_str(value)
            if iso:
                try:
                    phase_map[iso] = season_phase_for_date(dt.date.fromisoformat(iso))
                except Exception:
                    continue
    normalized_phase = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df["date"].tolist()):
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None:
            iso = normalize_backtest_date_str(raw_date)
            phase_name = phase_map.get(iso, "regular") if iso else "regular"
        normalized_phase.append(phase_name)
    df["phase"] = normalized_phase
    for col in ["projected_total", "away_expected_runs", "home_expected_runs", "projected_margin_raw", "projected_margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "projected_margin_raw" in df.columns:
        missing_raw_margin = df["projected_margin_raw"].isna()
        if missing_raw_margin.any() and "projected_margin" in df.columns:
            df.loc[missing_raw_margin, "projected_margin_raw"] = df.loc[missing_raw_margin, "projected_margin"]
    if "projected_margin" in df.columns:
        missing_margin = df["projected_margin"].isna()
        if missing_margin.any() and "projected_margin_raw" in df.columns:
            df.loc[missing_margin, "projected_margin"] = df.loc[missing_margin, "projected_margin_raw"]
        if missing_margin.any() and {"home_expected_runs", "away_expected_runs"}.issubset(df.columns):
            df.loc[missing_margin, "projected_margin"] = df.loc[missing_margin, "home_expected_runs"] - df.loc[missing_margin, "away_expected_runs"]
    return df[cols].copy()


def model_state_schema_missing_context() -> bool:
    if not os.path.exists(MODEL_STATE_PATH):
        return False
    try:
        with open(MODEL_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        weights = payload.get("weights") if isinstance(payload, dict) else {}
        phase_weights = payload.get("weights_by_phase") if isinstance(payload, dict) else {}
        regular_transition = payload.get("regular_transition") if isinstance(payload, dict) else {}
        version = safe_float(payload.get("version")) if isinstance(payload, dict) else None
        return (
            (not isinstance(weights, dict)) or
            ("w_context" not in weights) or
            (not isinstance(phase_weights, dict)) or
            (not isinstance(phase_weights.get("spring"), dict)) or
            (not isinstance(phase_weights.get("regular"), dict)) or
            (not isinstance(regular_transition, dict)) or
            (version is None) or
            (int(version) < 2)
        )
    except Exception:
        return False


def backtest_log_schema_missing_context() -> bool:
    if not os.path.exists(BACKTEST_LOG_PATH):
        return False
    try:
        with open(BACKTEST_LOG_PATH, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        return ("phase" not in header) or ("x_context" not in header) or ("W_CTX" not in header) or ("projected_total" not in header) or ("away_expected_runs" not in header) or ("home_expected_runs" not in header)
    except Exception:
        return False


def pending_backtest_refresh_dates(existing: pd.DataFrame, model_state: Dict, version: int = BACKTEST_FEATURE_VERSION) -> List[dt.date]:
    if existing.empty or "date" not in existing.columns:
        return []

    normalized_dates: List[str] = []
    seen = set()
    for value in existing["date"].tolist():
        iso = normalize_backtest_date_str(value)
        if iso and iso not in seen:
            seen.add(iso)
            normalized_dates.append(iso)
    if not normalized_dates:
        return []

    state = ensure_backtest_refresh_state(model_state)
    refresh = state.get("backtest_refresh", {})
    completed = set()
    if int(refresh.get("version", 0)) == int(version) and not backtest_log_schema_missing_context():
        completed = set(refresh.get("completed_dates", []))

    pending = [iso for iso in normalized_dates if iso not in completed]
    return sorted((dt.date.fromisoformat(iso) for iso in pending), reverse=True)


def pending_totals_refresh_dates(existing: pd.DataFrame, model_state: Dict, version: int = TOTALS_FEATURE_VERSION) -> List[dt.date]:
    if existing.empty or "date" not in existing.columns:
        return []

    normalized_dates: List[str] = []
    seen = set()
    for value in existing["date"].tolist():
        iso = normalize_backtest_date_str(value)
        if iso and iso not in seen:
            seen.add(iso)
            normalized_dates.append(iso)
    if not normalized_dates:
        return []

    state = ensure_totals_refresh_state(model_state)
    refresh = state.get("totals_refresh", {})
    completed = set(refresh.get("completed_dates", [])) if int(refresh.get("version", 0)) == int(version) else set()

    required_cols = ["projected_total", "away_expected_runs", "home_expected_runs"]
    incomplete_dates: set[str] = set()
    if not set(required_cols).issubset(existing.columns):
        incomplete_dates = set(normalized_dates)
    else:
        work = existing.copy()
        work["date"] = work["date"].apply(normalize_backtest_date_str)
        missing_mask = work[required_cols].isna().any(axis=1)
        incomplete_dates = {str(v) for v in work.loc[missing_mask, "date"].dropna().astype(str).tolist()}

    pending = [iso for iso in normalized_dates if (iso not in completed) or (iso in incomplete_dates)]
    return sorted((dt.date.fromisoformat(iso) for iso in pending), reverse=True)


def refresh_backtest_history_incremental(
    model_state: Dict,
    max_dates: int = BACKTEST_HISTORY_REFRESH_BATCH_DATES,
    recalibrate: bool = True,
) -> Tuple[Dict, pd.DataFrame, List[str], List[str], List[str]]:
    existing = load_backtest_log()
    warnings: List[str] = []
    processed_dates: List[str] = []
    pending_dates = pending_backtest_refresh_dates(existing, model_state)
    full_rebuild_required = backtest_refresh_requires_full_rebuild(model_state)
    if existing.empty or not pending_dates or (max_dates <= 0 and not full_rebuild_required):
        return model_state, existing, warnings, processed_dates, [d.isoformat() for d in pending_dates]

    target_dates = pending_dates if full_rebuild_required else pending_dates[:max_dates]
    if full_rebuild_required:
        warnings.append(
            f"Backtest history refresh: feature version {BACKTEST_FEATURE_VERSION} requires a one-time full row rebuild "
            f"for {len(target_dates)} stored date(s) to remove same-day leakage."
        )

    for target_date in target_dates:
        if full_rebuild_required:
            rows, row_warnings = build_backtest_rows_for_date(target_date, model_state)
        else:
            rows, row_warnings = build_backtest_context_refresh_rows_for_date(target_date)
        warnings.extend(row_warnings)
        if not rows:
            if full_rebuild_required:
                warnings.append(f"Backtest full rebuild produced no rows for {target_date.isoformat()}; existing log entries were kept.")
            else:
                warnings.append(f"Backtest history refresh produced no rows for {target_date.isoformat()}; existing log entries were kept.")
            continue
        append_backtest_log(rows)
        processed_dates.append(target_date.isoformat())
        if not full_rebuild_required:
            model_state = mark_backtest_refresh_dates(model_state, [target_date])
            save_model_state(model_state)

    if processed_dates and full_rebuild_required:
        model_state = mark_backtest_refresh_dates(model_state, processed_dates)
        model_state = mark_totals_refresh_dates(model_state, processed_dates)
        save_model_state(model_state)

    refreshed_df = load_backtest_log()
    remaining_dates = pending_backtest_refresh_dates(refreshed_df, model_state)
    if processed_dates and recalibrate:
        model_state = recalibrate_model_weights_from_log(model_state, refreshed_df)
        refreshed_df = rewrite_backtest_predictions_with_model_state(model_state)
        save_model_state(model_state)

    return model_state, refreshed_df, warnings, processed_dates, [d.isoformat() for d in remaining_dates]

def rebuild_backtest_log_from_existing_dates(model_state: Dict) -> Tuple[pd.DataFrame, List[str]]:
    existing = load_backtest_log()
    warnings: List[str] = []
    if existing.empty or "date" not in existing.columns:
        return existing, warnings

    pending_dates = pending_backtest_refresh_dates(existing, model_state)
    if not pending_dates:
        return existing, warnings

    _, rebuilt_df, refresh_warnings, _, _ = refresh_backtest_history_incremental(
        model_state,
        max_dates=len(pending_dates),
        recalibrate=False,
    )
    warnings.extend(refresh_warnings)
    return rebuilt_df, warnings


def build_backtest_totals_refresh_rows_for_date(target_date: dt.date, model_state: Dict) -> Tuple[List[Dict], List[str]]:
    warnings: List[str] = []
    existing = load_backtest_log()
    day_iso = target_date.isoformat()
    day_rows = existing[existing["date"].astype(str) == day_iso].copy()
    if day_rows.empty:
        return [], warnings

    rebuilt_rows, rebuilt_warnings = build_backtest_rows_for_date(target_date, model_state)
    warnings.extend(rebuilt_warnings)
    if not rebuilt_rows:
        warnings.append(f"Backtest totals refresh: rebuilt totals were unavailable for {day_iso}; existing totals were kept.")
        return [], warnings

    rebuilt_by_key = {
        (str(row.get("away", "")), str(row.get("home", ""))): row
        for row in rebuilt_rows
        if str(row.get("away", "")).strip() and str(row.get("home", "")).strip()
    }

    updated_rows: List[Dict] = []
    missing_matches = 0
    for _, row in day_rows.iterrows():
        key = (str(row.get("away", "")), str(row.get("home", "")))
        rebuilt = rebuilt_by_key.get(key)
        if not rebuilt:
            missing_matches += 1
            continue
        merged = row.to_dict()
        merged["phase"] = rebuilt.get("phase", merged.get("phase"))
        for field in ["projected_total", "away_expected_runs", "home_expected_runs"]:
            merged[field] = rebuilt.get(field, merged.get(field))
        updated_rows.append(merged)

    if missing_matches:
        warnings.append(f"Backtest totals refresh: {missing_matches} stored row(s) for {day_iso} could not be matched to rebuilt totals.")
    if not updated_rows:
        warnings.append(f"Backtest totals refresh: no stored rows matched rebuilt totals for {day_iso}.")
    return updated_rows, warnings


def refresh_backtest_totals_incremental(
    model_state: Dict,
    max_dates: int = BACKTEST_HISTORY_REFRESH_BATCH_DATES,
) -> Tuple[Dict, pd.DataFrame, List[str], List[str], List[str]]:
    existing = load_backtest_log()
    warnings: List[str] = []
    processed_dates: List[str] = []
    pending_dates = pending_totals_refresh_dates(existing, model_state)
    if existing.empty or not pending_dates or max_dates <= 0:
        return model_state, existing, warnings, processed_dates, [d.isoformat() for d in pending_dates]

    target_dates = pending_dates[:max_dates]
    for target_date in target_dates:
        rows, row_warnings = build_backtest_totals_refresh_rows_for_date(target_date, model_state)
        warnings.extend(row_warnings)
        if not rows:
            continue
        append_backtest_log(rows)
        processed_dates.append(target_date.isoformat())
        model_state = mark_totals_refresh_dates(model_state, [target_date])
        save_model_state(model_state)

    refreshed_df = load_backtest_log()
    remaining_dates = pending_totals_refresh_dates(refreshed_df, model_state)
    if processed_dates:
        model_state = update_total_calibration(model_state, refreshed_df)
        save_model_state(model_state)

    return model_state, refreshed_df, warnings, processed_dates, [d.isoformat() for d in remaining_dates]


def extract_pitcher_id(obj) -> Optional[int]:
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str) and obj.strip().isdigit():
        return int(obj.strip())
    if isinstance(obj, dict):
        for k in ["id", "mlbam_id", "player_id", "pitcher_id"]:
            v = obj.get(k)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
    return None


def starter_source_confidence(source: Optional[str]) -> float:
    src = str(source or "").strip().lower()
    if not src:
        return 1.0
    if "neutral prior" in src:
        return 0.0
    if f"{SEASON} current-season".lower() in src:
        return 1.0
    if f"{SEASON - 1} regular-season".lower() in src:
        return 0.90
    if f"{SEASON - 1} spring+regular".lower() in src:
        return 0.82
    if f"{SEASON - 2} regular-season".lower() in src:
        return 0.72
    if f"{SEASON - 2} spring+regular".lower() in src:
        return 0.60
    if f"{SEASON - 3} regular-season".lower() in src:
        return 0.50
    if "fallback" in src:
        return 0.75
    return 1.0


def starter_candidates_from_gf_game(game_pk: int) -> List[Dict[str, object]]:
    gf = savant_gf(game_pk)
    candidates: List[Dict[str, object]] = []
    for lineup_key, pitchers_key in [
        ("away_pitcher_lineup", "away_pitchers"),
        ("home_pitcher_lineup", "home_pitchers"),
    ]:
        obj = None
        lineup = gf.get(lineup_key)
        if isinstance(lineup, list) and lineup:
            obj = lineup[0]
        if obj is None:
            pitchers = gf.get(pitchers_key)
            if isinstance(pitchers, list) and pitchers:
                obj = pitchers[0]
        if obj is None:
            continue
        pid = extract_pitcher_id(obj)
        name = find_name_by_id_in_gf(gf, pid) if pid is not None else pick_pitcher_name(obj)
        if not name:
            name = pick_pitcher_name(obj)
        if not name:
            continue
        candidates.append({
            "pitcher": canonical_player_name(name),
            "pitcher_id": pid,
        })
    return candidates


def park_static_profile(venue_name: Optional[str]) -> Dict[str, object]:
    lookup = getattr(park_static_profile, "_lookup", None)
    if lookup is None:
        lookup = {}
        for profile in PARK_STATIC_PROFILES:
            base = {k: v for k, v in profile.items() if k != "aliases"}
            for alias in profile.get("aliases", []):
                alias_key = normalize_venue_name(alias)
                if alias_key:
                    lookup[alias_key] = base.copy()
        park_static_profile._lookup = lookup
    key = normalize_venue_name(venue_name)
    if not key:
        return {}
    return dict(lookup.get(key, {}))


def normalize_roof_type(raw: Optional[str]) -> str:
    txt = str(raw or "").strip().lower()
    if not txt:
        return "unknown"
    if "dome" in txt or "fixed" in txt:
        return "dome"
    if "retract" in txt:
        return "retractable"
    if "open" in txt:
        return "open_air"
    return txt.replace(" ", "_")


def normalize_surface_type(raw: Optional[str]) -> str:
    txt = str(raw or "").strip().lower()
    if not txt:
        return "unknown"
    if "grass" in txt:
        return "grass"
    if any(token in txt for token in ["turf", "synthetic", "artificial"]):
        return "turf"
    return txt


def load_venue_metadata_cache() -> Dict[str, Dict[str, object]]:
    if not os.path.exists(VENUE_METADATA_CACHE_PATH):
        return {}
    try:
        with open(VENUE_METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_venue_metadata_cache(cache: Dict[str, Dict[str, object]]) -> None:
    with open(VENUE_METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def hydrate_venue_metadata(venue_ids: List[Optional[int]]) -> Dict[int, Dict[str, object]]:
    cache = load_venue_metadata_cache()
    updated = False
    out: Dict[int, Dict[str, object]] = {}

    wanted = []
    for vid in venue_ids:
        if vid is None:
            continue
        try:
            wanted.append(int(vid))
        except Exception:
            continue

    for venue_id in sorted(set(wanted)):
        key = str(venue_id)
        cached_meta = cache.get(key) if isinstance(cache.get(key), dict) else {}
        needs_refresh = (
            not cached_meta or
            cached_meta.get("latitude") is None or
            cached_meta.get("longitude") is None or
            not str(cached_meta.get("venue_timezone") or "").strip()
        )
        if needs_refresh:
            try:
                r = requests.get(
                    STATSAPI_VENUE_URL.format(venue_id=venue_id),
                    params={"hydrate": "location,timeZone,fieldInfo"},
                    headers=HEADERS,
                    timeout=30,
                )
                r.raise_for_status()
                payload = r.json()
                venues = payload.get("venues") or []
                venue = venues[0] if venues else {}
                location = venue.get("location") or {}
                field_info = venue.get("fieldInfo") or {}
                coords = location.get("defaultCoordinates") or {}
                timezone_info = venue.get("timeZone") or location.get("timeZone") or {}
                venue_timezone = (
                    timezone_info.get("id") or
                    timezone_info.get("tz") or
                    timezone_info.get("name") or
                    venue.get("timeZoneId") or
                    location.get("timeZoneId")
                )
                cache[key] = {
                    "venue_id": venue_id,
                    "venue": venue.get("name"),
                    "roof_type": normalize_roof_type(field_info.get("roofType")),
                    "surface_type": normalize_surface_type(field_info.get("turfType")),
                    "elevation_ft": safe_float(location.get("elevation")),
                    "latitude": safe_float(coords.get("latitude")),
                    "longitude": safe_float(coords.get("longitude")),
                    "azimuth_angle": safe_float(location.get("azimuthAngle")),
                    "venue_timezone": venue_timezone,
                }
                updated = True
            except Exception:
                continue
        meta = cache.get(key)
        if isinstance(meta, dict):
            out[venue_id] = dict(meta)

    if updated:
        save_venue_metadata_cache(cache)
    return out


def official_venue_context(venue_id: Optional[int]) -> Dict[str, object]:
    try:
        if venue_id is None:
            return {}
        meta = hydrate_venue_metadata([int(venue_id)]).get(int(venue_id), {})
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def static_park_context(venue_id: Optional[int], venue_name: Optional[str]) -> Dict[str, object]:
    profile = park_static_profile(venue_name)
    official = official_venue_context(venue_id)

    roof_type = str(official.get("roof_type") or profile.get("roof_type") or "unknown")
    surface_type = str(official.get("surface_type") or profile.get("surface_type") or "unknown")
    elevation_raw = official.get("elevation_ft") if official.get("elevation_ft") is not None else profile.get("elevation_ft")
    elevation_ft = safe_float(elevation_raw)
    elevation_ft = float(elevation_ft) if elevation_ft is not None else 0.0
    run_bias = safe_float(profile.get("run_bias"))
    run_bias = float(run_bias) if run_bias is not None else 0.0
    home_edge_bias = safe_float(profile.get("home_edge_bias"))
    home_edge_bias = float(home_edge_bias) if home_edge_bias is not None else 0.0

    default = {
        "venue_key": venue_key(venue_id, venue_name),
        "venue": str(venue_name or official.get("venue") or "Unknown venue"),
        "park_total_adj": 0.0,
        "park_home_edge": 0.0,
        "park_hist_total_adj": 0.0,
        "park_hist_home_edge": 0.0,
        "park_static_total_adj": 0.0,
        "park_static_home_edge": 0.0,
        "park_games": 0.0,
        "park_weighted_games": 0.0,
        "roof_type": roof_type,
        "surface_type": surface_type,
        "elevation_ft": elevation_ft if np.isfinite(elevation_ft) else np.nan,
        "latitude": official.get("latitude"),
        "longitude": official.get("longitude"),
        "azimuth_angle": official.get("azimuth_angle"),
        "venue_timezone": official.get("venue_timezone"),
        "park_profile_found": 1.0 if profile or official else 0.0,
    }
    if not profile and not official:
        return default

    elev_total = float(np.clip(((elevation_ft - 500.0) / 1000.0) * 0.07, -0.05, 0.33))
    roof_total = {"open_air": 0.0, "retractable": 0.02, "dome": 0.04}.get(roof_type, 0.0)
    surface_total = {"grass": 0.0, "turf": 0.05}.get(surface_type, 0.0)
    park_static_total_adj = float(np.clip(elev_total + roof_total + surface_total + run_bias, -0.75, 0.95))

    elev_home = float(np.clip(max(elevation_ft - 1500.0, 0.0) / 4000.0 * 0.03, 0.0, 0.03))
    roof_home = {"open_air": 0.0, "retractable": 0.004, "dome": 0.007}.get(roof_type, 0.0)
    surface_home = {"grass": 0.0, "turf": 0.006}.get(surface_type, 0.0)
    park_static_home_edge = float(np.clip(elev_home + roof_home + surface_home + home_edge_bias, 0.0, 0.06))

    default.update({
        "park_total_adj": park_static_total_adj,
        "park_home_edge": park_static_home_edge,
        "park_static_total_adj": park_static_total_adj,
        "park_static_home_edge": park_static_home_edge,
    })
    return default

def load_park_factor_cache() -> pd.DataFrame:
    cols = ["date", "game_pk", "venue_id", "venue", "venue_key", "total_runs", "home_win"]
    if not os.path.exists(PARK_FACTORS_CACHE_PATH):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(PARK_FACTORS_CACHE_PATH)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].copy()


def update_park_factor_cache(as_of: dt.date) -> pd.DataFrame:
    cache = load_park_factor_cache()
    end_date = as_of - dt.timedelta(days=1)
    if end_date < PARK_CACHE_START:
        return cache

    if cache.empty:
        start_date = PARK_CACHE_START
    else:
        cached_dates = pd.to_datetime(cache["date"], errors="coerce").dropna()
        if cached_dates.empty:
            start_date = PARK_CACHE_START
        else:
            start_date = max(PARK_CACHE_START, (cached_dates.max().date() + dt.timedelta(days=1)))

    rows: List[Dict[str, object]] = []
    d = start_date
    while d <= end_date:
        try:
            sb = savant_scoreboard(d)
        except Exception:
            d += dt.timedelta(days=1)
            continue

        games = sb.get("games", []) if isinstance(sb, dict) else []
        if not isinstance(games, list):
            d += dt.timedelta(days=1)
            continue

        for g in games:
            try:
                game_pk = g.get("gamePk") or g.get("game_pk")
                if not game_pk:
                    continue
                game_pk = int(game_pk)
                linescore = g.get("linescore", {})
                if not (isinstance(linescore, dict) and isinstance(linescore.get("teams"), dict)):
                    continue
                away_runs = linescore["teams"].get("away", {}).get("runs")
                home_runs = linescore["teams"].get("home", {}).get("runs")
                if away_runs is None or home_runs is None:
                    continue
                venue_id, venue = extract_venue_info_from_scoreboard_game(g)
                total_runs = int(away_runs) + int(home_runs)
                home_win = 1.0 if int(home_runs) > int(away_runs) else (0.0 if int(home_runs) < int(away_runs) else 0.5)
                rows.append({
                    "date": d.isoformat(),
                    "game_pk": game_pk,
                    "venue_id": venue_id,
                    "venue": str(venue),
                    "venue_key": venue_key(venue_id, venue),
                    "total_runs": total_runs,
                    "home_win": home_win,
                })
            except Exception:
                continue
        d += dt.timedelta(days=1)

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True, sort=False)
        cache = cache.drop_duplicates(subset=["game_pk"], keep="last").sort_values(["date", "game_pk"]).reset_index(drop=True)
        cache.to_csv(PARK_FACTORS_CACHE_PATH, index=False)
    return cache


def build_park_factor_table(as_of: dt.date) -> pd.DataFrame:
    cols = [
        "venue_key", "venue_id", "venue", "park_total_adj", "park_home_edge", "park_games", "park_weighted_games",
        "park_hist_total_adj", "park_hist_home_edge", "park_static_total_adj", "park_static_home_edge",
        "roof_type", "surface_type", "elevation_ft", "park_profile_found"
    ]
    cache = update_park_factor_cache(as_of)
    if cache.empty:
        return pd.DataFrame(columns=cols)

    df = cache.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "total_runs", "home_win", "venue_key"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df[df["date"] < pd.Timestamp(as_of)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    age_days = (pd.Timestamp(as_of) - df["date"]).dt.days.clip(lower=0)
    df["_w"] = np.power(0.5, age_days / PARK_RECENCY_HALF_LIFE_DAYS)

    league_total = float(np.average(pd.to_numeric(df["total_runs"], errors="coerce"), weights=df["_w"]))
    league_home = float(np.average(pd.to_numeric(df["home_win"], errors="coerce"), weights=df["_w"]))
    league_home = float(np.clip(league_home, 0.05, 0.95))
    league_logit = math.log(league_home / (1.0 - league_home))

    rows = []
    for key, grp in df.groupby("venue_key"):
        w = pd.to_numeric(grp["_w"], errors="coerce").fillna(0.0)
        total_runs = pd.to_numeric(grp["total_runs"], errors="coerce").fillna(0.0)
        home_win = pd.to_numeric(grp["home_win"], errors="coerce").fillna(0.5)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue

        avg_total = float(np.average(total_runs, weights=w))
        avg_home = float(np.average(home_win, weights=w))
        total_shrink = w_sum / (w_sum + PARK_TOTAL_SHRINK)
        home_shrink = w_sum / (w_sum + PARK_HOME_EDGE_SHRINK)
        avg_home = float(np.clip(avg_home, 0.05, 0.95))
        home_logit = math.log(avg_home / (1.0 - avg_home))
        hist_total_adj = float(np.clip(total_shrink * (avg_total - league_total), -1.25, 1.25))
        hist_home_edge = float(np.clip(home_shrink * (home_logit - league_logit), -0.12, 0.12))

        venue_ids = pd.to_numeric(grp["venue_id"], errors="coerce").dropna()
        venue_id = int(venue_ids.iloc[0]) if not venue_ids.empty else None
        venue_series = grp["venue"].dropna().astype(str)
        venue_name = str(venue_series.iloc[-1]) if not venue_series.empty else "Unknown venue"

        static_ctx = static_park_context(venue_id, venue_name)
        static_total_adj_raw = safe_float(static_ctx.get("park_static_total_adj"))
        static_home_edge_raw = safe_float(static_ctx.get("park_static_home_edge"))
        static_total_adj = float(static_total_adj_raw) if static_total_adj_raw is not None else 0.0
        static_home_edge = float(static_home_edge_raw) if static_home_edge_raw is not None else 0.0

        park_total_adj = float(np.clip(
            ((w_sum * hist_total_adj) + (PARK_STATIC_TOTAL_PRIOR_WEIGHT * static_total_adj)) /
            (w_sum + PARK_STATIC_TOTAL_PRIOR_WEIGHT),
            -1.25,
            1.25,
        ))
        park_home_edge = float(np.clip(
            ((w_sum * hist_home_edge) + (PARK_STATIC_HOME_PRIOR_WEIGHT * static_home_edge)) /
            (w_sum + PARK_STATIC_HOME_PRIOR_WEIGHT),
            -0.12,
            0.12,
        ))

        rows.append({
            "venue_key": key,
            "venue_id": venue_id,
            "venue": venue_name,
            "park_total_adj": park_total_adj,
            "park_home_edge": park_home_edge,
            "park_games": int(len(grp)),
            "park_weighted_games": w_sum,
            "park_hist_total_adj": hist_total_adj,
            "park_hist_home_edge": hist_home_edge,
            "park_static_total_adj": static_total_adj,
            "park_static_home_edge": static_home_edge,
            "roof_type": str(static_ctx.get("roof_type", "unknown")),
            "surface_type": str(static_ctx.get("surface_type", "unknown")),
            "elevation_ft": static_ctx.get("elevation_ft", np.nan),
            "park_profile_found": float(static_ctx.get("park_profile_found", 0.0) or 0.0),
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)

def lookup_park_context(park_factors: Optional[pd.DataFrame], venue_id: Optional[int], venue_name: Optional[str]) -> Dict[str, float]:
    default = static_park_context(venue_id, venue_name)
    if park_factors is None or park_factors.empty:
        return default

    key = venue_key(venue_id, venue_name)
    match = park_factors[park_factors["venue_key"] == key]
    if match.empty and venue_name:
        match = park_factors[park_factors["venue_key"] == venue_key(None, venue_name)]
    if match.empty:
        return default

    row = match.iloc[0]
    out = default.copy()
    for c in [
        "venue", "park_total_adj", "park_home_edge", "park_games", "park_weighted_games",
        "park_hist_total_adj", "park_hist_home_edge", "park_static_total_adj", "park_static_home_edge",
        "roof_type", "surface_type", "elevation_ft", "park_profile_found"
    ]:
        if c in row:
            out[c] = row[c]
    return out

def _finite(x: float, fallback: float = 0.0) -> float:
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return float(fallback)
        return xf
    except Exception:
        return float(fallback)


def statsapi_game_feed(game_pk: int, timeout: int = 30) -> dict:
    cache = getattr(statsapi_game_feed, "_cache", None)
    if cache is None:
        cache = {}
        statsapi_game_feed._cache = cache
    if game_pk in cache:
        return cache[game_pk]
    r = requests.get(STATSAPI_GAME_FEED_URL.format(game_pk=game_pk), headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    cache[game_pk] = payload
    return payload


def weather_code_label(code: Optional[int]) -> str:
    mapping = {
        0: "Clear",
        1: "Mostly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Fog",
        51: "Drizzle",
        53: "Drizzle",
        55: "Drizzle",
        61: "Rain",
        63: "Rain",
        65: "Rain",
        71: "Snow",
        73: "Snow",
        75: "Snow",
        80: "Showers",
        81: "Showers",
        82: "Showers",
        95: "Thunderstorms",
        96: "Thunderstorms",
        99: "Thunderstorms",
    }
    try:
        return mapping.get(int(code), "Unknown")
    except Exception:
        return "Unknown"


def extract_boxscore_info_map(info_rows: object) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(info_rows, list):
        return out
    for row in info_rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        value = str(row.get("value") or "").strip()
        if label:
            out[label.lower()] = value
    return out


def parse_wind_details(wind_text: Optional[str], azimuth_angle: Optional[float] = None, direction_degrees: Optional[float] = None) -> Dict[str, object]:
    raw = str(wind_text or "").strip()
    mph_match = re.search(r'(-?\d+(?:\.\d+)?)\s*mph', raw, flags=re.I)
    mph = float(mph_match.group(1)) if mph_match else 0.0
    direction_label = ""
    out_in_component = 0.0

    if "," in raw:
        direction_label = raw.split(",", 1)[1].strip().rstrip('.')
    elif raw:
        direction_label = raw

    direction_upper = direction_label.upper()
    if any(token in direction_upper for token in ["OUT TO CF", "OUT TO LF", "OUT TO RF", "OUT TO CENTER"]):
        out_in_component = 1.0
    elif any(token in direction_upper for token in ["IN FROM CF", "IN FROM LF", "IN FROM RF", "IN FROM CENTER"]):
        out_in_component = -1.0
    elif "L TO R" in direction_upper or "R TO L" in direction_upper:
        out_in_component = 0.0
    elif direction_degrees is not None and azimuth_angle is not None:
        try:
            wind_to_bearing = (float(direction_degrees) + 180.0) % 360.0
            diff = math.radians(((wind_to_bearing - float(azimuth_angle) + 180.0) % 360.0) - 180.0)
            out_in_component = float(math.cos(diff))
            cross_component = float(math.sin(diff))
            if abs(out_in_component) >= 0.45:
                direction_label = "Out to CF" if out_in_component > 0 else "In from CF"
            elif abs(cross_component) >= 0.35:
                direction_label = "L to R" if cross_component > 0 else "R to L"
            else:
                direction_label = "Light/variable"
        except Exception:
            out_in_component = 0.0

    return {
        "wind_text": raw,
        "wind_mph": mph,
        "wind_direction": direction_label,
        "wind_out_in": float(np.clip(out_in_component, -1.0, 1.0)),
    }


def condition_has_precip(condition: Optional[str]) -> bool:
    txt = str(condition or "").strip().lower()
    return any(token in txt for token in ["rain", "drizzle", "storm", "shower", "snow", "sleet"])


def estimate_roof_status(roof_type: str, condition: Optional[str], temp_f: Optional[float], wind_mph: float, precip_prob: Optional[float], precip_in: Optional[float]) -> Tuple[str, str]:
    cond = str(condition or "").strip().lower()
    if "roof closed" in cond:
        return "closed", "statsapi"
    if "roof open" in cond:
        return "open", "statsapi"
    if roof_type == "dome":
        return "closed", "venue"
    if roof_type == "open_air":
        return "open", "venue"
    if roof_type != "retractable":
        return "unknown", "none"

    precip_prob = _finite(precip_prob, 0.0)
    precip_in = _finite(precip_in, 0.0)
    temp_val = safe_float(temp_f)
    if condition_has_precip(cond) or precip_prob >= 45 or precip_in >= 0.03 or wind_mph >= 16:
        return "likely_closed", "estimated"
    if temp_val is not None and (temp_val < 58 or temp_val > 92):
        return "likely_closed", "estimated"
    if temp_val is not None and 64 <= temp_val <= 86 and precip_prob <= 20 and wind_mph <= 12:
        return "likely_open", "estimated"
    return "unknown", "estimated"


def weather_adjustments(temp_f: Optional[float], wind_mph: float, wind_out_in: float, condition: Optional[str], roof_status: str, precip_prob: Optional[float], precip_in: Optional[float]) -> Tuple[float, float]:
    exposure = {
        "closed": 0.0,
        "likely_closed": 0.25,
        "unknown": 0.60,
        "likely_open": 0.85,
        "open": 1.0,
    }.get(str(roof_status or "unknown"), 0.60)

    temp_val = safe_float(temp_f)
    precip_prob = _finite(precip_prob, 0.0)
    precip_in = _finite(precip_in, 0.0)

    if temp_val is None:
        temp_adj = 0.0
    else:
        temp_scale = 0.012 if exposure >= 0.5 else 0.004
        temp_adj = float(np.clip((temp_val - 70.0) * temp_scale, -0.35, 0.30))

    wind_adj = float(np.clip(wind_mph * wind_out_in * 0.04 * exposure, -0.55, 0.55))

    precip_adj = 0.0
    if precip_prob > 0:
        precip_adj -= 0.0025 * precip_prob * exposure
    if precip_in > 0:
        precip_adj -= min(precip_in, 0.25) * 0.80 * exposure
    if condition_has_precip(condition):
        precip_adj -= 0.12 * max(exposure, 0.50)

    total_adj = float(np.clip(temp_adj + wind_adj + precip_adj, -0.90, 0.90))
    volatility = abs(wind_adj) + abs(precip_adj) + max(abs(temp_adj) - 0.10, 0.0)
    home_edge = float(np.clip(0.04 * volatility * exposure, 0.0, 0.03))
    return total_adj, home_edge


def forecast_weather_for_game(venue_ctx: Dict[str, object], scheduled_dt_utc: Optional[str]) -> Dict[str, object]:
    lat = safe_float(venue_ctx.get("latitude"))
    lon = safe_float(venue_ctx.get("longitude"))
    azimuth_angle = safe_float(venue_ctx.get("azimuth_angle"))
    if lat is None or lon is None or not scheduled_dt_utc:
        return {}

    try:
        scheduled_utc = dt.datetime.fromisoformat(str(scheduled_dt_utc).replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        target_utc = scheduled_utc.replace(minute=0, second=0, microsecond=0)
        if scheduled_utc.minute >= 30:
            target_utc += dt.timedelta(hours=1)
    except Exception:
        return {}

    try:
        r = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "UTC",
                "start_date": target_utc.date().isoformat(),
                "end_date": target_utc.date().isoformat(),
            },
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        target_key = target_utc.strftime("%Y-%m-%dT%H:%M")
        if target_key not in times:
            return {}
        idx = times.index(target_key)
        temp_f = safe_float((hourly.get("temperature_2m") or [None])[idx])
        precip_prob = safe_float((hourly.get("precipitation_probability") or [None])[idx])
        precip_in = safe_float((hourly.get("precipitation") or [None])[idx])
        wind_mph = safe_float((hourly.get("wind_speed_10m") or [0.0])[idx]) or 0.0
        wind_deg = safe_float((hourly.get("wind_direction_10m") or [None])[idx])
        weather_code = (hourly.get("weather_code") or [None])[idx]
        wind = parse_wind_details(None, azimuth_angle=azimuth_angle, direction_degrees=wind_deg)
        return {
            "weather_source": "open_meteo_forecast",
            "condition": weather_code_label(weather_code),
            "temp_f": temp_f,
            "wind_text": f"{wind_mph:.0f} mph est",
            "wind_mph": wind_mph,
            "wind_direction": wind.get("wind_direction", ""),
            "wind_out_in": wind.get("wind_out_in", 0.0),
            "precip_prob": precip_prob,
            "precip_in": precip_in,
        }
    except Exception:
        return {}
def game_weather_context(game_pk: int, venue_id: Optional[int], venue_name: Optional[str], scheduled_dt_utc_hint: Optional[str] = None) -> Dict[str, object]:
    cache = getattr(game_weather_context, "_cache", None)
    if cache is None:
        cache = {}
        game_weather_context._cache = cache
    if game_pk in cache:
        return dict(cache[game_pk])

    venue_ctx = static_park_context(venue_id, venue_name)
    out = {
        "game_pk": game_pk,
        "weather_source": "neutral",
        "condition": "Unavailable",
        "temp_f": np.nan,
        "wind_text": "Unavailable",
        "wind_mph": 0.0,
        "wind_direction": "",
        "wind_out_in": 0.0,
        "precip_prob": np.nan,
        "precip_in": np.nan,
        "roof_status": "unknown",
        "roof_status_source": "none",
        "weather_total_adj": 0.0,
        "weather_home_edge": 0.0,
        "roof_type": venue_ctx.get("roof_type", "unknown"),
        "surface_type": venue_ctx.get("surface_type", "unknown"),
        "elevation_ft": venue_ctx.get("elevation_ft", np.nan),
    }

    try:
        feed = statsapi_game_feed(game_pk)
        game_data = feed.get("gameData") or {}
        weather = game_data.get("weather") or {}
        info_map = extract_boxscore_info_map(((feed.get("liveData") or {}).get("boxscore") or {}).get("info"))
        scheduled_dt_utc = ((game_data.get("datetime") or {}).get("dateTime")) or scheduled_dt_utc_hint
        azimuth_angle = safe_float(venue_ctx.get("azimuth_angle"))

        condition = str(weather.get("condition") or "").strip()
        temp_f = safe_float(weather.get("temp"))
        wind_text = str(weather.get("wind") or "").strip()

        if not condition and "weather" in info_map:
            weather_line = info_map.get("weather", "")
            temp_match = re.search(r'(\d+(?:\.\d+)?)\s*degrees', weather_line, flags=re.I)
            if temp_match:
                temp_f = safe_float(temp_match.group(1))
            condition = re.sub(r'^\s*\d+(?:\.\d+)?\s*degrees\s*,?\s*', '', weather_line, flags=re.I).strip().strip('.')
        if not wind_text and "wind" in info_map:
            wind_text = info_map.get("wind", "")

        weather_ctx = {}
        if condition or temp_f is not None or wind_text:
            wind = parse_wind_details(wind_text, azimuth_angle=azimuth_angle)
            weather_ctx = {
                "weather_source": "statsapi_actual",
                "condition": condition or "Reported",
                "temp_f": temp_f,
                "wind_text": wind.get("wind_text") or wind_text or "0 mph, None",
                "wind_mph": wind.get("wind_mph", 0.0),
                "wind_direction": wind.get("wind_direction", ""),
                "wind_out_in": wind.get("wind_out_in", 0.0),
                "precip_prob": np.nan,
                "precip_in": np.nan,
            }
            if str(condition or "").strip().lower() != "roof closed" and str(venue_ctx.get("roof_type")) == "retractable" and (wind.get("wind_mph", 0.0) > 0 or condition):
                weather_ctx["roof_status"] = "open"
                weather_ctx["roof_status_source"] = "statsapi_weather"
        else:
            weather_ctx = forecast_weather_for_game(venue_ctx, scheduled_dt_utc)

        out.update(weather_ctx)
        roof_status, roof_source = estimate_roof_status(
            str(venue_ctx.get("roof_type", "unknown")),
            out.get("condition"),
            safe_float(out.get("temp_f")),
            _finite(out.get("wind_mph", 0.0), 0.0),
            safe_float(out.get("precip_prob")),
            safe_float(out.get("precip_in")),
        )
        if out.get("roof_status") in {None, "unknown", ""}:
            out["roof_status"] = roof_status
            out["roof_status_source"] = roof_source
        total_adj, home_edge = weather_adjustments(
            safe_float(out.get("temp_f")),
            _finite(out.get("wind_mph", 0.0), 0.0),
            _finite(out.get("wind_out_in", 0.0), 0.0),
            out.get("condition"),
            str(out.get("roof_status", "unknown")),
            safe_float(out.get("precip_prob")),
            safe_float(out.get("precip_in")),
        )
        out["weather_total_adj"] = total_adj
        out["weather_home_edge"] = home_edge
    except Exception:
        pass

    cache[game_pk] = dict(out)
    return dict(out)


def build_game_weather_map(game_refs: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for ref in game_refs:
        try:
            game_pk = int(ref.get("game_pk"))
        except Exception:
            continue
        venue_id = ref.get("venue_id")
        try:
            venue_id = int(venue_id) if venue_id is not None and str(venue_id) != "" and not pd.isna(venue_id) else None
        except Exception:
            venue_id = None
        venue_name = ref.get("venue")
        if venue_name is not None and pd.isna(venue_name):
            venue_name = None
        scheduled_dt_utc_hint = ref.get("scheduled_dt_utc")
        out[game_pk] = game_weather_context(game_pk, venue_id, venue_name, scheduled_dt_utc_hint=scheduled_dt_utc_hint)
    return out


def lookup_game_weather_context(weather_map: Optional[Dict[int, Dict[str, object]]], game_pk: Optional[int]) -> Dict[str, object]:
    default = {
        "weather_source": "neutral",
        "condition": "Unavailable",
        "temp_f": np.nan,
        "wind_text": "Unavailable",
        "wind_mph": 0.0,
        "wind_direction": "",
        "wind_out_in": 0.0,
        "precip_prob": np.nan,
        "precip_in": np.nan,
        "roof_status": "unknown",
        "roof_status_source": "none",
        "weather_total_adj": 0.0,
        "weather_home_edge": 0.0,
    }
    if weather_map is None or game_pk is None:
        return default
    ctx = weather_map.get(int(game_pk))
    if not isinstance(ctx, dict):
        return default
    out = default.copy()
    out.update(ctx)
    return out


def build_backtest_context_refresh_rows_for_date(target_date: dt.date) -> Tuple[List[Dict], List[str]]:
    warnings: List[str] = []
    existing = load_backtest_log()
    day_iso = target_date.isoformat()
    day_rows = existing[existing["date"].astype(str) == day_iso].copy()
    if day_rows.empty:
        return [], warnings

    finals_df, finals_warn = backtest_finals_from_scoreboard(target_date)
    warnings.extend(finals_warn)
    if finals_df.empty:
        return [], warnings

    phase = season_phase_for_date(target_date)
    game_types = game_types_for_phase(phase)
    season_start = dt.date(target_date.year, 1, 1)

    prior_offense_y = pd.DataFrame()
    try:
        prior_offense_y = previous_regular_season_team_offense(target_date.year - 1)
    except Exception as e:
        warnings.append(f"Prior-season team offense unavailable for {target_date.isoformat()}: {e}")

    season_raw_y = statcast_team_battedball_snapshot(season_start, pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d30_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=30), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d14_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=14), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d7_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=7), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    strengths_y = build_strengths_bundle_from_snapshots(
        season_raw_y,
        d30_raw_y,
        d14_raw_y,
        d7_raw_y,
        prior_agg=prior_offense_y,
    )

    b_df_y = statcast_batter_snapshot(season_start, pregame_statcast_end_date(target_date), game_types=game_types)
    bat_agg_y = aggregate_batter_quality(b_df_y)
    bat_comp_y = build_batter_composite(bat_agg_y)

    game_refs_y: List[Dict[str, object]] = []
    for _, r in finals_df.iterrows():
        venue_id_raw = r.get("venue_id") if isinstance(r, pd.Series) else None
        venue_id = int(venue_id_raw) if pd.notna(venue_id_raw) else None
        venue_name = str(r.get("venue")) if isinstance(r, pd.Series) and pd.notna(r.get("venue")) else None
        game_refs_y.append({
            "game_pk": int(r["game_pk"]),
            "away": str(r["away_team"]),
            "home": str(r["home_team"]),
            "venue_id": venue_id,
            "venue": venue_name,
        })

    game_context_y = build_game_context_map(
        game_refs_y,
        strengths_y,
        season_raw_y,
        bat_comp_y,
        prior_batter_season=target_date.year - 1,
    )

    existing_by_key = {
        (str(row["away"]), str(row["home"])): row.to_dict()
        for _, row in day_rows.iterrows()
    }
    updated_rows: List[Dict] = []
    for ref in game_refs_y:
        key = (str(ref["away"]), str(ref["home"]))
        if key not in existing_by_key:
            continue
        ctx_comp = context_edge_from_game(
            str(ref["away"]),
            str(ref["home"]),
            strengths_y,
            game_context_y,
            int(ref["game_pk"]),
        )
        row = existing_by_key[key].copy()
        row["phase"] = phase
        row["x_context"] = float(_finite(ctx_comp.get("x_context", row.get("x_context", 0.0)), 0.0))
        updated_rows.append(row)

    if not updated_rows:
        warnings.append(f"Backtest history refresh: no stored rows matched refreshed context for {day_iso}.")
    return updated_rows, warnings


def build_backtest_rows_for_date(target_date: dt.date, model_state: Dict) -> Tuple[List[Dict], List[str]]:
    warnings: List[str] = []
    finals_df, finals_warn = backtest_finals_from_scoreboard(target_date)
    warnings.extend(finals_warn)
    if finals_df.empty:
        return [], warnings

    phase = season_phase_for_date(target_date)
    game_types = game_types_for_phase(phase)
    season_start = dt.date(target_date.year, 1, 1)

    prior_offense_y = pd.DataFrame()
    try:
        prior_offense_y = previous_regular_season_team_offense(target_date.year - 1)
    except Exception as e:
        warnings.append(f"Prior-season team offense unavailable for {target_date.isoformat()}: {e}")

    season_raw_y = statcast_team_battedball_snapshot(season_start, pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d30_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=30), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d14_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=14), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    d7_raw_y = statcast_team_battedball_snapshot(target_date - dt.timedelta(days=7), pregame_statcast_end_date(target_date), game_types, season_year=target_date.year)
    strengths_y = build_strengths_bundle_from_snapshots(
        season_raw_y,
        d30_raw_y,
        d14_raw_y,
        d7_raw_y,
        prior_agg=prior_offense_y,
    )

    p_df_y = statcast_pitching_snapshot(season_start, pregame_statcast_end_date(target_date), game_types=game_types)
    pitch_comp_y = build_pitcher_matchup_table(p_df_y)
    park_factors_y = build_park_factor_table(target_date)
    schedule_probables_y: Dict[int, Dict[str, object]] = {}
    try:
        schedule_probables_y = probables_from_statsapi_schedule(target_date)
    except Exception as e:
        warnings.append(f"Backtest schedule probable starters unavailable for {target_date.isoformat()}: {e}")

    starter_meta: Dict[int, Dict[str, object]] = {}
    starter_candidates: List[Dict[str, object]] = []
    starter_candidates.extend(starter_candidates_from_probables_map(schedule_probables_y))
    for _, r in finals_df.iterrows():
        game_pk = int(r["game_pk"])
        schedule_probables = schedule_probables_y.get(game_pk, {})
        try:
            away_st, home_st = starters_from_gf(game_pk)
        except Exception as e:
            warnings.append(f"Backtest: starter fetch failed for game_pk={game_pk}: {e}")
            away_st, home_st = (None, None)
        away_st = away_st or canonical_player_name(schedule_probables.get("away_pitcher"))
        home_st = home_st or canonical_player_name(schedule_probables.get("home_pitcher"))
        starter_meta[game_pk] = {
            "away_pitcher": away_st,
            "home_pitcher": home_st,
            "away_pitcher_id": schedule_probables.get("away_pitcher_id"),
            "home_pitcher_id": schedule_probables.get("home_pitcher_id"),
            "scheduled_dt_utc": schedule_probables.get("scheduled_dt_utc"),
        }
        try:
            starter_candidates.extend(starter_candidates_from_gf_game(game_pk))
        except Exception:
            pass

    if starter_candidates:
        pitch_comp_y, _ = augment_pitcher_scores_with_previous_season(
            pitch_comp_y,
            starter_candidates,
            fallback_season=target_date.year - 1,
        )

    pen3_y = bullpen_snapshot(target_date, lookback_days=3, pitch_comp=pitch_comp_y)
    pen7_y = bullpen_snapshot(target_date, lookback_days=7, pitch_comp=pitch_comp_y)
    if pitch_comp_y.empty:
        warnings.append(f"Backtest pitcher composite empty for {target_date.isoformat()}; starters are neutral.")

    b_df_y = statcast_batter_snapshot(season_start, pregame_statcast_end_date(target_date), game_types=game_types)
    bat_agg_y = aggregate_batter_quality(b_df_y)
    bat_comp_y = build_batter_composite(bat_agg_y)

    game_refs_y: List[Dict[str, object]] = []
    weather_refs_y: List[Dict[str, object]] = []
    for _, r in finals_df.iterrows():
        game_pk = int(r["game_pk"])
        meta = starter_meta.get(game_pk, {})
        venue_id_raw = r.get("venue_id") if isinstance(r, pd.Series) else None
        venue_id = int(venue_id_raw) if pd.notna(venue_id_raw) else None
        venue_name = str(r.get("venue")) if isinstance(r, pd.Series) and pd.notna(r.get("venue")) else None
        ref = {
            "game_pk": game_pk,
            "away": str(r["away_team"]),
            "home": str(r["home_team"]),
            "venue_id": venue_id,
            "venue": venue_name,
            "scheduled_dt_utc": meta.get("scheduled_dt_utc"),
        }
        game_refs_y.append(ref)
        weather_refs_y.append({
            "game_pk": game_pk,
            "venue_id": venue_id,
            "venue": venue_name,
            "scheduled_dt_utc": meta.get("scheduled_dt_utc"),
        })

    game_context_y = build_game_context_map(
        game_refs_y,
        strengths_y,
        season_raw_y,
        bat_comp_y,
        prior_batter_season=target_date.year - 1,
    )
    weather_map_y = build_game_weather_map(weather_refs_y)
    active_weights = active_weight_profile(model_state, model_date=target_date, phase=phase)["weights"]

    rows: List[Dict] = []
    for _, r in finals_df.iterrows():
        home = str(r["home_team"])
        away = str(r["away_team"])
        hs = int(r["home_score"])
        aas = int(r["away_score"])
        y_home = 1 if hs > aas else 0
        game_pk = int(r["game_pk"])
        meta = starter_meta.get(game_pk, {})

        venue_id_raw = r.get("venue_id") if isinstance(r, pd.Series) else None
        venue_id = int(venue_id_raw) if pd.notna(venue_id_raw) else None
        venue_name = str(r.get("venue")) if isinstance(r, pd.Series) and pd.notna(r.get("venue")) else None

        _, p_home, _, comp = predict_game(
            away,
            home,
            strengths_y,
            meta.get("away_pitcher"),
            meta.get("home_pitcher"),
            pitch_comp_y,
            pen3_y,
            pen7_y,
            model_state,
            park_factors=park_factors_y,
            venue_id=venue_id,
            venue_name=venue_name,
            weather_map=weather_map_y,
            game_pk=game_pk,
            away_pitcher_id=meta.get("away_pitcher_id"),
            home_pitcher_id=meta.get("home_pitcher_id"),
            game_context_map=game_context_y,
            model_date=target_date,
            model_phase=phase,
        )

        p_home = _finite(p_home, 0.5)
        for key in ["logit", "x_off", "x_starter", "x_bullpen", "x_park", "x_weather", "x_context"]:
            comp[key] = _finite(comp.get(key, 0.0), 0.0)
        total = project_total_runs(
            away,
            home,
            strengths_y,
            meta.get("away_pitcher"),
            meta.get("home_pitcher"),
            pitch_comp_y,
            pen3_y,
            pen7_y,
            model_state,
            park_factors_y,
            venue_id,
            venue_name,
            weather_map=weather_map_y,
            game_pk=game_pk,
            away_pitcher_id=meta.get("away_pitcher_id"),
            home_pitcher_id=meta.get("home_pitcher_id"),
            game_context_map=game_context_y,
            model_date=target_date,
            model_phase=phase,
            game_components=comp,
        )

        rows.append({
            "date": target_date.isoformat(),
            "phase": phase,
            "away": away,
            "home": home,
            "venue_id": venue_id,
            "venue": venue_name,
            "home_score": hs,
            "away_score": aas,
            "y_home": y_home,
            "p_home": p_home,
            "pred_home": 1 if p_home >= 0.5 else 0,
            "projected_total": float(_finite(total, np.nan)),
            "away_expected_runs": float(_finite(comp.get("away_expected_runs", np.nan), np.nan)),
            "home_expected_runs": float(_finite(comp.get("home_expected_runs", np.nan), np.nan)),
            "projected_margin_raw": float(_finite(comp.get("projected_margin_raw", comp.get("projected_margin", np.nan)), np.nan)),
            "projected_margin": float(_finite(comp.get("projected_margin", np.nan), np.nan)),
            "logit": comp["logit"],
            "x_off": comp["x_off"],
            "x_starter": comp["x_starter"],
            "x_bullpen": comp["x_bullpen"],
            "x_park": comp["x_park"],
            "x_weather": comp["x_weather"],
            "x_context": comp["x_context"],
            "W_OFF": comp.get("W_OFF", active_weights["w_off"]),
            "W_ST": comp.get("W_ST", active_weights["w_starter"]),
            "W_BP": comp.get("W_BP", active_weights["w_bullpen"]),
            "W_PK": comp.get("W_PK", active_weights.get("w_park", 0.03)),
            "W_WX": comp.get("W_WX", active_weights.get("w_weather", 0.02)),
            "W_CTX": comp.get("W_CTX", active_weights.get("w_context", 0.12)),
            "HFA": comp.get("HFA", active_weights["home_field"]),
        })

    return rows, warnings


def recalibrate_model_weights_from_log(
    state: Dict,
    backtest_df: Optional[pd.DataFrame] = None,
    refresh_meta: bool = True,
) -> Dict:
    state = ensure_phase_weight_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    if df is None or df.empty:
        return state

    work = df.copy()
    for c in ["x_off", "x_starter", "x_bullpen", "x_park", "x_weather", "x_context", "y_home", "home_score", "away_score"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        else:
            work[c] = 0.0
    work = work.dropna(subset=["y_home"]).copy()
    if work.empty:
        return state

    if "phase" not in work.columns:
        work["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(work["phase"].tolist(), work.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    work["phase"] = normalized_phase
    work["date"] = normalized_date
    work = work[work["date"].notna()].copy()
    if work.empty:
        return state

    def fit_phase_weights(rows: pd.DataFrame, current: Dict[str, float], phase_name: str, spring_anchor: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        rows = rows.copy()
        if rows.empty:
            return dict(current)

        x_off = pd.to_numeric(rows.get("x_off"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_st = pd.to_numeric(rows.get("x_starter"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_bp = pd.to_numeric(rows.get("x_bullpen"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_pk = pd.to_numeric(rows.get("x_park"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_wx = pd.to_numeric(rows.get("x_weather"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_ctx = pd.to_numeric(rows.get("x_context"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = pd.to_numeric(rows.get("y_home"), errors="coerce").fillna(0.0).astype(int).to_numpy(dtype=float)

        dts = pd.to_datetime(rows.get("date"), errors="coerce")
        latest = dts.max()
        if pd.isna(latest):
            age_days = np.zeros(len(rows), dtype=float)
        else:
            age_days = (latest - dts).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
        half_life = 18.0 if phase_name == "spring" else 35.0
        sample_w = np.power(0.5, age_days / half_life)
        sample_w = np.where(np.isfinite(sample_w), sample_w, 1.0)
        weight_sum = float(sample_w.sum()) or float(len(rows)) or 1.0

        prior = {
            "w_off": float(_finite(current.get("w_off"), BASE_WEIGHT_DEFAULTS["w_off"])),
            "w_starter": float(_finite(current.get("w_starter"), BASE_WEIGHT_DEFAULTS["w_starter"])),
            "w_bullpen": float(_finite(current.get("w_bullpen"), BASE_WEIGHT_DEFAULTS["w_bullpen"])),
            "w_park": float(_finite(current.get("w_park"), BASE_WEIGHT_DEFAULTS["w_park"])),
            "w_weather": float(_finite(current.get("w_weather"), BASE_WEIGHT_DEFAULTS["w_weather"])),
            "w_context": float(_finite(current.get("w_context"), BASE_WEIGHT_DEFAULTS["w_context"])),
            "home_field": float(_finite(current.get("home_field"), BASE_WEIGHT_DEFAULTS["home_field"])),
            "spread_scale": float(_finite(current.get("spread_scale"), BASE_WEIGHT_DEFAULTS["spread_scale"])),
        }
        if phase_name == "regular" and isinstance(spring_anchor, dict):
            prior["w_off"] = float(_finite(spring_anchor.get("w_off"), prior["w_off"]))
            prior["home_field"] = float(_finite(spring_anchor.get("home_field"), prior["home_field"]))

        w_off = prior["w_off"]
        w_st = prior["w_starter"]
        w_bp = prior["w_bullpen"]
        w_pk = prior["w_park"]
        w_wx = prior["w_weather"]
        w_ctx = prior["w_context"]
        hfa = prior["home_field"]
        spread_scale = prior["spread_scale"]

        lr = 0.09 if phase_name == "spring" else 0.07
        prior_strength = 0.20 if phase_name == "spring" else 0.14
        park_prior_strength = 0.55
        weather_prior_strength = 0.75
        context_prior_strength = 0.35
        hfa_prior_strength = 0.45

        for _ in range(350):
            z = np.clip((w_off * x_off) + (w_st * x_st) + (w_bp * x_bp) + (w_pk * x_pk) + (w_wx * x_wx) + (w_ctx * x_ctx) + hfa, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y) * sample_w

            g_off = float((err * x_off).sum() / weight_sum) + prior_strength * (w_off - prior["w_off"])
            g_st = float((err * x_st).sum() / weight_sum) + prior_strength * (w_st - prior["w_starter"])
            g_bp = float((err * x_bp).sum() / weight_sum) + prior_strength * (w_bp - prior["w_bullpen"])
            g_pk = float((err * x_pk).sum() / weight_sum) + park_prior_strength * (w_pk - prior["w_park"])
            g_wx = float((err * x_wx).sum() / weight_sum) + weather_prior_strength * (w_wx - prior["w_weather"])
            g_ctx = float((err * x_ctx).sum() / weight_sum) + context_prior_strength * (w_ctx - prior["w_context"])
            g_hf = float(err.sum() / weight_sum) + hfa_prior_strength * (hfa - prior["home_field"])

            w_off = float(np.clip(w_off - (lr * g_off), 0.05, 2.0))
            w_st = float(np.clip(w_st - (lr * g_st), 0.05, 2.0))
            w_bp = float(np.clip(w_bp - (lr * g_bp), 0.00, 1.5))
            w_pk = float(np.clip(w_pk - (lr * g_pk), 0.00, 0.25))
            w_wx = float(np.clip(w_wx - (lr * g_wx), 0.00, 1.0))
            w_ctx = float(np.clip(w_ctx - (lr * g_ctx), 0.00, 1.0))
            hfa = float(np.clip(hfa - (lr * g_hf), -0.08, 0.25))

        z = np.clip((w_off * x_off) + (w_st * x_st) + (w_bp * x_bp) + (w_pk * x_pk) + (w_wx * x_wx) + (w_ctx * x_ctx) + hfa, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        if "home_score" in rows.columns and "away_score" in rows.columns:
            home_score = pd.to_numeric(rows["home_score"], errors="coerce").to_numpy(dtype=float)
            away_score = pd.to_numeric(rows["away_score"], errors="coerce").to_numpy(dtype=float)
            margin = home_score - away_score
            valid_margin = np.isfinite(margin)
            if valid_margin.sum() >= 8:
                margin_x = p[valid_margin] - 0.5
                denom = float((sample_w[valid_margin] * np.square(margin_x)).sum())
                if denom > 1e-8:
                    fitted_scale = float((sample_w[valid_margin] * margin_x * margin[valid_margin]).sum() / denom)
                    spread_scale = float(np.clip(abs(fitted_scale), 1.0, 8.0))

        return {
            "w_off": float(_finite(w_off, prior["w_off"])),
            "w_starter": float(_finite(w_st, prior["w_starter"])),
            "w_bullpen": float(_finite(w_bp, prior["w_bullpen"])),
            "w_park": float(_finite(w_pk, prior["w_park"])),
            "w_weather": float(_finite(w_wx, prior["w_weather"])),
            "w_context": float(_finite(w_ctx, prior["w_context"])),
            "home_field": float(_finite(hfa, prior["home_field"])),
            "spread_scale": float(_finite(spread_scale, prior["spread_scale"])),
        }

    spring_rows = work[work["phase"] == "spring"].copy()
    regular_rows = work[work["phase"] == "regular"].copy()

    spring_weights = dict(state["weights_by_phase"].get("spring", BASE_WEIGHT_DEFAULTS))
    regular_weights = dict(state["weights_by_phase"].get("regular", BASE_WEIGHT_DEFAULTS))

    if not spring_rows.empty:
        spring_weights = fit_phase_weights(spring_rows, spring_weights, "spring")
    if regular_rows.empty and not spring_rows.empty:
        regular_weights = dict(spring_weights)
    elif not regular_rows.empty:
        regular_weights = fit_phase_weights(regular_rows, regular_weights, "regular", spring_anchor=spring_weights)
    state["weights_by_phase"]["spring"] = spring_weights
    state["weights_by_phase"]["regular"] = regular_weights
    state["weights"] = dict(regular_weights)
    state["last_update_date"] = today_local().isoformat()
    if refresh_meta:
        state = update_spring_guardrail(state, work)
        state = update_regular_feature_earn_back(state, work)
        state = update_probability_calibration(state, work)
        state = update_total_calibration(state, work)
        state = update_margin_calibration(state, work)
    return state

def weight_update_step(state: Dict, examples: List[Dict]) -> Dict:
    w = state["weights"]

    # Pull current weights, but force finite
    w_off = _finite(w.get("w_off", 0.65), 0.65)
    w_st  = _finite(w.get("w_starter", 0.55), 0.55)
    w_bp  = _finite(w.get("w_bullpen", 0.30), 0.30)
    hfa   = _finite(w.get("home_field", 0.12), 0.12)

    # If no examples, keep weights unchanged
    if not examples:
        state["weights"].update({"w_off": w_off, "w_starter": w_st, "w_bullpen": w_bp, "home_field": hfa})
        state["last_update_date"] = today_local().isoformat()
        return state

    g_off = g_st = g_bp = g_hf = 0.0
    n_used = 0

    for ex in examples:
        y = int(ex.get("y_home", 0))
        x_off = _finite(ex.get("x_off", 0.0), 0.0)
        x_st  = _finite(ex.get("x_starter", 0.0), 0.0)
        x_bp  = _finite(ex.get("x_bullpen", 0.0), 0.0)

        z = (w_off * x_off) + (w_st * x_st) + (w_bp * x_bp) + hfa
        z = _finite(z, 0.0)

        p = sigmoid(z)
        p = _finite(p, 0.5)
        p = min(max(p, 1e-6), 1 - 1e-6)

        dz = (p - y)
        g_off += dz * x_off
        g_st  += dz * x_st
        g_bp  += dz * x_bp
        g_hf  += dz
        n_used += 1

    if n_used == 0:
        state["weights"].update({"w_off": w_off, "w_starter": w_st, "w_bullpen": w_bp, "home_field": hfa})
        state["last_update_date"] = today_local().isoformat()
        return state

    # Average gradients
    g_off /= n_used
    g_st  /= n_used
    g_bp  /= n_used
    g_hf  /= n_used

    # L2 regularization
    g_off += L2_LAMBDA * w_off
    g_st  += L2_LAMBDA * w_st
    g_bp  += L2_LAMBDA * w_bp
    g_hf  += L2_LAMBDA * hfa

    # Gradient step
    w_off = _finite(w_off - WEIGHT_LR * g_off, 0.65)
    w_st  = _finite(w_st  - WEIGHT_LR * g_st, 0.55)
    w_bp  = _finite(w_bp  - WEIGHT_LR * g_bp, 0.30)
    hfa   = _finite(hfa   - WEIGHT_LR * g_hf, 0.12)

    # Clip
    w_off = float(np.clip(w_off, 0.05, 2.0))
    w_st  = float(np.clip(w_st, 0.05, 2.0))
    w_bp  = float(np.clip(w_bp, 0.00, 1.5))
    hfa   = float(np.clip(hfa, -0.25, 0.40))

    state["weights"].update({"w_off": w_off, "w_starter": w_st, "w_bullpen": w_bp, "home_field": hfa})
    state["last_update_date"] = today_local().isoformat()
    return state


def backtest_rows_with_model_state(backtest_df: Optional[pd.DataFrame], state: Dict, apply_calibration: bool = True) -> pd.DataFrame:
    state = ensure_margin_calibration_state(ensure_phase_weight_state(state))
    if backtest_df is None:
        return pd.DataFrame()

    df = backtest_df.copy()
    if df.empty:
        return df

    for c in ["x_off", "x_starter", "x_bullpen", "x_park", "x_weather", "x_context"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["projected_total", "away_expected_runs", "home_expected_runs", "projected_margin_raw", "projected_margin", "home_score", "away_score"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    missing_total = df["projected_total"].isna()
    if missing_total.any():
        df.loc[missing_total, "projected_total"] = df.loc[missing_total, "away_expected_runs"] + df.loc[missing_total, "home_expected_runs"]
    missing_margin_raw = df["projected_margin_raw"].isna()
    if missing_margin_raw.any() and "projected_margin" in df.columns:
        df.loc[missing_margin_raw, "projected_margin_raw"] = df.loc[missing_margin_raw, "projected_margin"]
    missing_margin_raw = df["projected_margin_raw"].isna()
    if missing_margin_raw.any():
        df.loc[missing_margin_raw, "projected_margin_raw"] = df.loc[missing_margin_raw, "home_expected_runs"] - df.loc[missing_margin_raw, "away_expected_runs"]
    missing_margin = df["projected_margin"].isna()
    if missing_margin.any():
        df.loc[missing_margin, "projected_margin"] = df.loc[missing_margin, "projected_margin_raw"]

    logits: List[float] = []
    raw_probs: List[float] = []
    probs: List[float] = []
    pred_home: List[int] = []
    w_off_vals: List[float] = []
    w_st_vals: List[float] = []
    w_bp_vals: List[float] = []
    w_pk_vals: List[float] = []
    w_wx_vals: List[float] = []
    w_ctx_vals: List[float] = []
    hfa_vals: List[float] = []
    phases: List[str] = []
    margin_raw_vals: List[float] = []
    margin_vals: List[float] = []
    margin_modes: List[str] = []
    away_run_vals: List[float] = []
    home_run_vals: List[float] = []

    for _, row in df.iterrows():
        iso = normalize_backtest_date_str(row.get("date"))
        model_date = dt.date.fromisoformat(iso) if iso else today_local()
        phase_name = normalize_model_phase(row.get("phase")) or season_phase_for_date(model_date)
        profile = active_weight_profile(state, model_date=model_date, phase=phase_name)
        weights = profile["weights"]
        z = (
            float(weights["w_off"]) * float(row["x_off"]) +
            float(weights["w_starter"]) * float(row["x_starter"]) +
            float(weights["w_bullpen"]) * float(row["x_bullpen"]) +
            float(weights["w_park"]) * float(row["x_park"]) +
            float(weights["w_weather"]) * float(row["x_weather"]) +
            float(weights["w_context"]) * float(row["x_context"]) +
            float(weights["home_field"])
        )
        z = float(_finite(z, 0.0))
        raw_p_home = float(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))
        probability_calibration = active_probability_calibration(state, model_date=model_date, phase=phase_name)
        p_home = apply_probability_calibration(raw_p_home, probability_calibration) if apply_calibration else raw_p_home

        raw_margin = safe_float(row.get("projected_margin_raw"))
        if raw_margin is None:
            raw_margin = safe_float(row.get("projected_margin"))
        if raw_margin is None:
            raw_margin = float(_finite(row.get("home_expected_runs"), 0.0) - _finite(row.get("away_expected_runs"), 0.0))
        raw_margin = float(np.clip(_finite(raw_margin, 0.0), -6.5, 6.5))
        margin_calibration = active_margin_calibration(state, model_date=model_date, phase=phase_name)
        calibrated_margin = apply_margin_calibration(raw_margin, margin_calibration) if apply_calibration else raw_margin

        total_value = safe_float(row.get("projected_total"))
        away_runs = safe_float(row.get("away_expected_runs"))
        home_runs = safe_float(row.get("home_expected_runs"))
        if total_value is None:
            total_value = _finite(away_runs, 0.0) + _finite(home_runs, 0.0)
        total_value = float(np.clip(_finite(total_value, 8.5), 5.5, 14.5))
        if np.isfinite(calibrated_margin):
            home_runs = float((total_value + calibrated_margin) / 2.0)
            away_runs = float(total_value - home_runs)
            if home_runs < 1.5:
                home_runs = 1.5
                away_runs = total_value - home_runs
            if away_runs < 1.5:
                away_runs = 1.5
                home_runs = total_value - away_runs
            home_runs = float(np.clip(home_runs, 1.5, 10.5))
            away_runs = float(np.clip(total_value - home_runs, 1.5, 10.5))
            total_value = float(np.clip(home_runs + away_runs, 5.5, 14.5))
            calibrated_margin = float(np.clip(home_runs - away_runs, -6.5, 6.5))

        logits.append(z)
        raw_probs.append(raw_p_home)
        probs.append(p_home)
        pred_home.append(1 if p_home >= 0.5 else 0)
        w_off_vals.append(float(weights["w_off"]))
        w_st_vals.append(float(weights["w_starter"]))
        w_bp_vals.append(float(weights["w_bullpen"]))
        w_pk_vals.append(float(weights["w_park"]))
        w_wx_vals.append(float(weights["w_weather"]))
        w_ctx_vals.append(float(weights["w_context"]))
        hfa_vals.append(float(weights["home_field"]))
        phases.append(phase_name)
        margin_raw_vals.append(raw_margin)
        margin_vals.append(calibrated_margin)
        margin_modes.append(str(margin_calibration.get("mode", "identity")))
        away_run_vals.append(float(away_runs))
        home_run_vals.append(float(home_runs))

    df["phase"] = phases
    df["logit"] = logits
    df["p_home_raw"] = raw_probs
    df["p_home"] = probs
    df["pred_home"] = pred_home
    df["projected_margin_raw"] = margin_raw_vals
    df["projected_margin"] = margin_vals
    df["margin_calibration_mode"] = margin_modes
    df["away_expected_runs"] = away_run_vals
    df["home_expected_runs"] = home_run_vals
    df["projected_total"] = pd.to_numeric(df["away_expected_runs"], errors="coerce") + pd.to_numeric(df["home_expected_runs"], errors="coerce")
    df["W_OFF"] = w_off_vals
    df["W_ST"] = w_st_vals
    df["W_BP"] = w_bp_vals
    df["W_PK"] = w_pk_vals
    df["W_WX"] = w_wx_vals
    df["W_CTX"] = w_ctx_vals
    df["HFA"] = hfa_vals
    return df


def rewrite_backtest_predictions_with_model_state(state: Dict) -> pd.DataFrame:
    df = backtest_rows_with_model_state(load_backtest_log(), state)
    if not df.empty:
        df.to_csv(BACKTEST_LOG_PATH, index=False)
    return df


def backtest_prediction_metrics(backtest_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    metrics = {
        "games": 0,
        "start_date": None,
        "end_date": None,
        "log_loss": None,
        "brier": None,
        "accuracy": None,
    }
    if backtest_df is None or backtest_df.empty:
        return metrics

    work = backtest_df.copy()
    if "date" in work.columns:
        normalized_dates = [normalize_backtest_date_str(v) for v in work["date"].tolist()]
        valid_dates = [iso for iso in normalized_dates if iso]
        if valid_dates:
            metrics["start_date"] = min(valid_dates)
            metrics["end_date"] = max(valid_dates)

    if "p_home" not in work.columns or "y_home" not in work.columns:
        return metrics

    work["p_home"] = pd.to_numeric(work["p_home"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    work["y_home"] = pd.to_numeric(work["y_home"], errors="coerce")
    work = work.dropna(subset=["p_home", "y_home"]).copy()
    if work.empty:
        return metrics

    y = work["y_home"].astype(int)
    p = work["p_home"].astype(float)
    metrics["games"] = int(len(work))
    metrics["log_loss"] = float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean())
    metrics["brier"] = float(np.mean(np.square(p - y)))
    metrics["accuracy"] = float(np.mean((p >= 0.5).astype(int) == y))
    return metrics



def estimate_margin_distribution_sigma(state: Optional[Dict] = None, backtest_df: Optional[pd.DataFrame] = None) -> float:
    fallback = 3.25
    df = backtest_df.copy() if isinstance(backtest_df, pd.DataFrame) else load_backtest_log()
    if df is None or df.empty:
        return fallback

    work = df.copy()
    for col in ["projected_margin_raw", "projected_margin", "home_score", "away_score"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    margin_pred = work["projected_margin_raw"] if "projected_margin_raw" in work.columns else pd.Series(np.nan, index=work.index, dtype=float)
    if "projected_margin" in work.columns:
        margin_pred = margin_pred.fillna(work["projected_margin"])
    work["margin_pred"] = pd.to_numeric(margin_pred, errors="coerce")
    work["actual_margin"] = pd.to_numeric(work["home_score"], errors="coerce") - pd.to_numeric(work["away_score"], errors="coerce")
    work = work.dropna(subset=["margin_pred", "actual_margin"]).copy()
    if work.empty:
        return fallback

    residual = work["actual_margin"].astype(float) - work["margin_pred"].astype(float)
    sigma = float(np.sqrt(np.mean(np.square(residual))))
    if not np.isfinite(sigma):
        return fallback
    return float(np.clip(sigma, 1.75, 6.0))


def walk_forward_date_split(unique_dates: List[dt.date]) -> Optional[Dict[str, List[dt.date]]]:
    dates = sorted({d for d in unique_dates if isinstance(d, dt.date)})
    n_dates = len(dates)
    if n_dates < 5:
        return None

    validation_dates = max(1, int(round(n_dates * 0.2)))
    test_dates = max(1, int(round(n_dates * 0.2)))
    max_holdout = n_dates - 3
    while (validation_dates + test_dates) > max_holdout and (validation_dates > 1 or test_dates > 1):
        if test_dates >= validation_dates and test_dates > 1:
            test_dates -= 1
        elif validation_dates > 1:
            validation_dates -= 1
        else:
            break

    train_dates = n_dates - validation_dates - test_dates
    if train_dates < 3:
        return None

    train = dates[:train_dates]
    validation = dates[train_dates:train_dates + validation_dates]
    test = dates[train_dates + validation_dates:]
    if not train or not validation or not test:
        return None
    return {"train": train, "validation": validation, "test": test}


def format_backtest_date_span(dates: List[dt.date]) -> str:
    if not dates:
        return "N/A"
    if len(dates) == 1 or dates[0] == dates[-1]:
        return dates[0].isoformat()
    return f"{dates[0].isoformat()} to {dates[-1].isoformat()}"


def evaluation_base_model_state(source_state: Optional[Dict] = None) -> Dict:
    base = default_model_state()
    if isinstance(source_state, dict) and isinstance(source_state.get("regular_transition"), dict):
        base["regular_transition"].update(source_state["regular_transition"])
    return ensure_phase_weight_state(base)


def walk_forward_evaluation(
    backtest_df: Optional[pd.DataFrame] = None,
    phase: Optional[str] = None,
    base_state: Optional[Dict] = None,
) -> Dict[str, object]:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    phase_name = normalize_model_phase(phase)
    result: Dict[str, object] = {
        "available": False,
        "phase": phase_name or "overall",
        "reason": "no backtest rows",
    }
    if df is None or df.empty:
        return result

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        current_phase = normalize_model_phase(raw_phase)
        if current_phase is None and iso:
            current_phase = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(current_phase or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
        result["phase"] = phase_name
    if df.empty:
        result["reason"] = "no backtest rows for this phase"
        return result

    unique_dates = [dt.date.fromisoformat(iso) for iso in sorted(df["date"].astype(str).unique())]
    splits = walk_forward_date_split(unique_dates)
    if splits is None:
        result["reason"] = "need at least 5 unique backtest dates for a clean train/validation/test split"
        result["games"] = int(len(df))
        result["dates"] = [d.isoformat() for d in unique_dates]
        return result

    def subset_for_dates(date_values: List[dt.date]) -> pd.DataFrame:
        wanted = {d.isoformat() for d in date_values}
        return df[df["date"].astype(str).isin(wanted)].copy().reset_index(drop=True)

    train_rows = subset_for_dates(splits["train"])
    validation_rows = subset_for_dates(splits["validation"])
    test_rows = subset_for_dates(splits["test"])

    train_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_rows, refresh_meta=False)
    train_pred = backtest_rows_with_model_state(train_rows, train_state)
    validation_pred = backtest_rows_with_model_state(validation_rows, train_state)

    train_validation_rows = pd.concat([train_rows, validation_rows], ignore_index=True)
    test_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_validation_rows, refresh_meta=False)
    test_pred = backtest_rows_with_model_state(test_rows, test_state)
    out_of_sample_pred = pd.concat([validation_pred, test_pred], ignore_index=True)
    train_sigma = estimate_margin_distribution_sigma(train_state, train_pred)
    train_validation_pred = pd.concat([train_pred, validation_pred], ignore_index=True)
    test_sigma = estimate_margin_distribution_sigma(test_state, train_validation_pred if not train_validation_pred.empty else test_pred)

    result.update({
        "available": True,
        "phase": result["phase"],
        "games": int(len(df)),
        "split_dates": {key: [d.isoformat() for d in value] for key, value in splits.items()},
        "train_span": format_backtest_date_span(splits["train"]),
        "validation_span": format_backtest_date_span(splits["validation"]),
        "test_span": format_backtest_date_span(splits["test"]),
        "train_metrics": backtest_prediction_metrics(train_pred),
        "validation_metrics": backtest_prediction_metrics(validation_pred),
        "test_metrics": backtest_prediction_metrics(test_pred),
        "out_of_sample_metrics": backtest_prediction_metrics(out_of_sample_pred),
        "train_total_metrics": total_prediction_metrics(train_pred, calibration_state=train_state),
        "validation_total_metrics": total_prediction_metrics(validation_pred, calibration_state=train_state),
        "test_total_metrics": total_prediction_metrics(test_pred, calibration_state=test_state),
        "out_of_sample_total_metrics": total_prediction_metrics(out_of_sample_pred, calibration_state=test_state),
        "train_margin_metrics": margin_prediction_metrics(train_pred, sigma=train_sigma, model_state=train_state),
        "validation_margin_metrics": margin_prediction_metrics(validation_pred, sigma=train_sigma, model_state=train_state),
        "test_margin_metrics": margin_prediction_metrics(test_pred, sigma=test_sigma, model_state=test_state),
        "out_of_sample_margin_metrics": margin_prediction_metrics(out_of_sample_pred, sigma=test_sigma, model_state=test_state),
        "train_run_line_metrics": run_line_prediction_metrics(train_pred, sigma=train_sigma, model_state=train_state),
        "validation_run_line_metrics": run_line_prediction_metrics(validation_pred, sigma=train_sigma, model_state=train_state),
        "test_run_line_metrics": run_line_prediction_metrics(test_pred, sigma=test_sigma, model_state=test_state),
        "out_of_sample_run_line_metrics": run_line_prediction_metrics(out_of_sample_pred, sigma=test_sigma, model_state=test_state),
    })
    return result
def fit_simple_power_baseline(rows: Optional[pd.DataFrame]) -> Dict[str, float]:
    prior = {"w_off": 0.55, "home_field": 0.08}
    if rows is None or rows.empty:
        return dict(prior)

    work = rows.copy()
    x_off = pd.to_numeric(work.get("x_off"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(work.get("y_home"), errors="coerce").fillna(0.0).astype(int).to_numpy(dtype=float)

    if "date" in work.columns:
        dts = pd.to_datetime(work["date"], errors="coerce")
        latest = dts.max()
        if pd.isna(latest):
            age_days = np.zeros(len(work), dtype=float)
        else:
            age_days = (latest - dts).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
    else:
        age_days = np.zeros(len(work), dtype=float)

    recency = np.power(0.5, age_days / 21.0)
    sample_w = np.where(np.isfinite(recency), recency, 1.0)
    weight_sum = float(sample_w.sum()) or float(len(sample_w)) or 1.0

    w_off = float(prior["w_off"])
    hfa = float(prior["home_field"])
    lr = 0.08
    prior_strength = 0.18
    hfa_prior_strength = 0.40

    for _ in range(250):
        z = np.clip((w_off * x_off) + hfa, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sample_w
        g_off = float((err * x_off).sum() / weight_sum) + prior_strength * (w_off - prior["w_off"])
        g_hfa = float(err.sum() / weight_sum) + hfa_prior_strength * (hfa - prior["home_field"])
        w_off = float(np.clip(w_off - (lr * g_off), 0.05, 2.0))
        hfa = float(np.clip(hfa - (lr * g_hfa), -0.08, 0.25))

    return {"w_off": w_off, "home_field": hfa}


def simple_power_baseline_rows(backtest_df: Optional[pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    if backtest_df is None:
        return pd.DataFrame()
    df = backtest_df.copy()
    if df.empty:
        return df
    x_off = pd.to_numeric(df.get("x_off"), errors="coerce").fillna(0.0)
    z = np.clip((float(_finite(weights.get("w_off"), 0.55)) * x_off) + float(_finite(weights.get("home_field"), 0.08)), -20.0, 20.0)
    p_home = 1.0 / (1.0 + np.exp(-z))
    df["logit"] = z
    df["p_home"] = p_home
    df["pred_home"] = (p_home >= 0.5).astype(int)
    return df


def walk_forward_simple_power_benchmark(backtest_df: Optional[pd.DataFrame] = None, phase: Optional[str] = None) -> Dict[str, object]:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    phase_name = normalize_model_phase(phase)
    result: Dict[str, object] = {
        "available": False,
        "phase": phase_name or "overall",
        "label": "simple power baseline (x_off + home field)",
        "reason": "no backtest rows",
    }
    if df is None or df.empty:
        return result

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        current_phase = normalize_model_phase(raw_phase)
        if current_phase is None and iso:
            current_phase = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(current_phase or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
        result["phase"] = phase_name
    if df.empty:
        result["reason"] = "no backtest rows for this phase"
        return result

    unique_dates = [dt.date.fromisoformat(iso) for iso in sorted(df["date"].astype(str).unique())]
    splits = walk_forward_date_split(unique_dates)
    if splits is None:
        result["reason"] = "need at least 5 unique backtest dates for a clean benchmark split"
        result["games"] = int(len(df))
        return result

    def subset_for_dates(date_values: List[dt.date]) -> pd.DataFrame:
        wanted = {d.isoformat() for d in date_values}
        return df[df["date"].astype(str).isin(wanted)].copy().reset_index(drop=True)

    train_rows = subset_for_dates(splits["train"])
    validation_rows = subset_for_dates(splits["validation"])
    test_rows = subset_for_dates(splits["test"])

    train_weights = fit_simple_power_baseline(train_rows)
    train_pred = simple_power_baseline_rows(train_rows, train_weights)
    validation_pred = simple_power_baseline_rows(validation_rows, train_weights)

    train_validation_rows = pd.concat([train_rows, validation_rows], ignore_index=True)
    test_weights = fit_simple_power_baseline(train_validation_rows)
    test_pred = simple_power_baseline_rows(test_rows, test_weights)
    out_of_sample_pred = pd.concat([validation_pred, test_pred], ignore_index=True)

    result.update({
        "available": True,
        "phase": result["phase"],
        "games": int(len(df)),
        "train_span": format_backtest_date_span(splits["train"]),
        "validation_span": format_backtest_date_span(splits["validation"]),
        "test_span": format_backtest_date_span(splits["test"]),
        "train_metrics": backtest_prediction_metrics(train_pred),
        "validation_metrics": backtest_prediction_metrics(validation_pred),
        "test_metrics": backtest_prediction_metrics(test_pred),
        "out_of_sample_metrics": backtest_prediction_metrics(out_of_sample_pred),
        "train_total_metrics": total_prediction_metrics(train_pred, apply_calibration=False),
        "validation_total_metrics": total_prediction_metrics(validation_pred, apply_calibration=False),
        "test_total_metrics": total_prediction_metrics(test_pred, apply_calibration=False),
        "out_of_sample_total_metrics": total_prediction_metrics(out_of_sample_pred, apply_calibration=False),
        "train_weights": train_weights,
        "test_weights": test_weights,
    })
    return result


def market_probability_backtest_rows(backtest_df: Optional[pd.DataFrame], closing_only: bool = True) -> pd.DataFrame:
    if backtest_df is None or backtest_df.empty or not os.path.exists(SPORTSBOOK_DB_PATH):
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for _, row in backtest_df.iterrows():
            iso = normalize_backtest_date_str(row.get("date"))
            away = str(row.get("away") or "").strip()
            home = str(row.get("home") or "").strip()
            if not iso or not away or not home:
                continue
            market_row = load_market_consensus_row_from_conn(conn, iso, away, home, game_pk=None, closing_only=closing_only)
            if market_row is None:
                continue
            home_prob, away_prob = no_vig_probabilities(market_row.get("home_ml"), market_row.get("away_ml"))
            if home_prob is None or away_prob is None:
                continue
            rows.append({
                "date": iso,
                "phase": row.get("phase"),
                "away": away,
                "home": home,
                "y_home": row.get("y_home"),
                "p_home": float(home_prob),
                "pred_home": 1 if float(home_prob) >= 0.5 else 0,
                "sportsbook": market_row.get("market_label") or market_row.get("sportsbook") or "consensus",
            })
    return pd.DataFrame(rows)


def market_benchmark_summary(backtest_df: Optional[pd.DataFrame] = None, phase: Optional[str] = None) -> Dict[str, object]:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    phase_name = normalize_model_phase(phase)
    result: Dict[str, object] = {
        "available": False,
        "phase": phase_name or "overall",
        "reason": "no archived sportsbook snapshots overlap the backtest log",
    }
    if df is None or df.empty:
        return result

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        current_phase = normalize_model_phase(raw_phase)
        if current_phase is None and iso:
            current_phase = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(current_phase or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
        result["phase"] = phase_name
    if df.empty:
        result["reason"] = "no backtest rows for this phase"
        return result

    unique_dates = [dt.date.fromisoformat(iso) for iso in sorted(df["date"].astype(str).unique())]
    splits = walk_forward_date_split(unique_dates)
    if splits is None:
        result["reason"] = "need at least 5 unique backtest dates for a clean market benchmark split"
        result["games"] = int(len(df))
        return result

    market_df = market_probability_backtest_rows(df, closing_only=True)
    if market_df.empty:
        return result

    oos_dates = {d.isoformat() for d in (splits["validation"] + splits["test"])}
    market_oos = market_df[market_df["date"].astype(str).isin(oos_dates)].copy()
    if market_oos.empty:
        result["reason"] = "no archived sportsbook snapshots overlap the walk-forward out-of-sample window"
        return result

    metrics = backtest_prediction_metrics(market_oos)
    favorite_accuracy = None
    if "pred_home" in market_oos.columns and "y_home" in market_oos.columns:
        pred = pd.to_numeric(market_oos["pred_home"], errors="coerce")
        y = pd.to_numeric(market_oos["y_home"], errors="coerce")
        valid = pred.notna() & y.notna()
        if valid.any():
            favorite_accuracy = float((pred[valid].astype(int) == y[valid].astype(int)).mean())

    result.update({
        "available": True,
        "phase": result["phase"],
        "games": int(len(market_oos)),
        "closing_implied_metrics": metrics,
        "market_favorite_accuracy": favorite_accuracy,
        "out_of_sample_span": format_backtest_date_span(splits["validation"] + splits["test"]),
        "total_market_games": int(len(market_df)),
    })
    return result
def simple_power_profile_from_weights(weights: Dict[str, float], spread_scale: Optional[float] = None) -> Dict[str, float]:
    profile = {key: 0.0 for key in BASE_WEIGHT_DEFAULTS.keys()}
    profile["w_off"] = float(_finite((weights or {}).get("w_off"), 0.55))
    profile["home_field"] = float(_finite((weights or {}).get("home_field"), 0.08))
    profile["spread_scale"] = float(_finite(spread_scale, BASE_WEIGHT_DEFAULTS["spread_scale"]))
    return profile


def spring_guardrail_active_weights(state: Dict) -> Dict[str, float]:
    state = ensure_phase_weight_state(state)
    raw_spring = {key: float(_finite(state["weights_by_phase"]["spring"].get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    guardrail = state.get("spring_guardrail") if isinstance(state.get("spring_guardrail"), dict) else {}
    adjusted = guardrail.get("adjusted_weights") if isinstance(guardrail.get("adjusted_weights"), dict) else None
    if not guardrail.get("apply") or adjusted is None:
        return raw_spring
    merged = dict(raw_spring)
    for key, default in BASE_WEIGHT_DEFAULTS.items():
        merged[key] = float(_finite(adjusted.get(key, merged.get(key, default)), merged.get(key, default)))
    return merged


def update_spring_guardrail(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_phase_weight_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    raw_spring = {key: float(_finite(state["weights_by_phase"]["spring"].get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    default_guardrail = {
        "apply": False,
        "mode": "learned",
        "blend": 0.0,
        "reason": "insufficient spring backtest data",
        "full_oos_log_loss": None,
        "simple_oos_log_loss": None,
        "oos_gap": None,
        "out_of_sample_games": 0,
        "adjusted_weights": dict(raw_spring),
        "simple_power_weights": simple_power_profile_from_weights({}, spread_scale=raw_spring.get("spread_scale")),
    }

    if df is None or df.empty:
        state["spring_guardrail"] = default_guardrail
        return state

    work = df.copy()
    if "phase" not in work.columns:
        work["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(work["phase"].tolist(), work.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    work["phase"] = normalized_phase
    work["date"] = normalized_date
    work = work[(work["phase"] == "spring") & work["date"].notna()].copy()
    if work.empty:
        default_guardrail["reason"] = "no spring backtest rows available"
        state["spring_guardrail"] = default_guardrail
        return state

    unique_dates = sorted(work["date"].astype(str).unique())
    if len(unique_dates) < 5:
        default_guardrail["reason"] = "need at least 5 unique spring backtest dates"
        state["spring_guardrail"] = default_guardrail
        return state

    raw_state = copy.deepcopy(state)
    if isinstance(raw_state, dict):
        raw_state.pop("spring_guardrail", None)
    full_eval = walk_forward_evaluation(work, phase="spring", base_state=raw_state)
    simple_eval = walk_forward_simple_power_benchmark(work, phase="spring")

    full_oos = full_eval.get("out_of_sample_metrics", {}) if isinstance(full_eval, dict) else {}
    simple_oos = simple_eval.get("out_of_sample_metrics", {}) if isinstance(simple_eval, dict) else {}
    full_ll = safe_float(full_oos.get("log_loss"))
    simple_ll = safe_float(simple_oos.get("log_loss"))
    simple_fit = fit_simple_power_baseline(work)
    simple_profile = simple_power_profile_from_weights(simple_fit, spread_scale=raw_spring.get("spread_scale"))

    guardrail = dict(default_guardrail)
    guardrail["simple_power_weights"] = dict(simple_profile)
    guardrail["full_oos_log_loss"] = full_ll
    guardrail["simple_oos_log_loss"] = simple_ll
    guardrail["out_of_sample_games"] = int(simple_oos.get("games", full_oos.get("games", 0)) or 0)

    if full_ll is None or simple_ll is None:
        guardrail["reason"] = "spring walk-forward benchmark metrics unavailable"
        state["spring_guardrail"] = guardrail
        return state

    gap = float(full_ll - simple_ll)
    guardrail["oos_gap"] = gap
    if gap > 0.005:
        blend = float(np.clip((gap - 0.005) / 0.02, 0.35, 1.0))
        adjusted = blend_weight_sets(raw_spring, simple_profile, blend)
        guardrail["apply"] = True
        guardrail["blend"] = blend
        guardrail["mode"] = "simple_power_override" if blend >= 0.999 else "simple_power_blend"
        guardrail["reason"] = (
            f"spring walk-forward log loss {full_ll:.4f} trails simple power baseline {simple_ll:.4f}; "
            f"blending {blend*100:.0f}% toward the simpler profile until extra features earn their keep"
        )
        guardrail["adjusted_weights"] = {key: float(_finite(adjusted.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    else:
        guardrail["reason"] = "learned spring profile matches or beats the simple power baseline out of sample"
        guardrail["adjusted_weights"] = dict(raw_spring)

    state["spring_guardrail"] = guardrail
    return state
FEATURE_WEIGHT_KEYS = {
    "starter": "w_starter",
    "bullpen": "w_bullpen",
    "park": "w_park",
    "weather": "w_weather",
    "context": "w_context",
}


def apply_feature_subset_to_phase_weights(state: Dict, phase_name: str, enabled_features: List[str]) -> Dict:
    tuned = copy.deepcopy(ensure_phase_weight_state(state))
    if phase_name not in ["spring", "regular"]:
        return tuned

    raw_weights = tuned["weights_by_phase"][phase_name]
    base_profile = simple_power_profile_from_weights(raw_weights, spread_scale=raw_weights.get("spread_scale"))
    base_profile["w_off"] = float(_finite(raw_weights.get("w_off"), base_profile["w_off"]))
    base_profile["home_field"] = float(_finite(raw_weights.get("home_field"), base_profile["home_field"]))
    for feature_name, weight_key in FEATURE_WEIGHT_KEYS.items():
        if feature_name in enabled_features:
            base_profile[weight_key] = float(_finite(raw_weights.get(weight_key), 0.0))
    tuned["weights_by_phase"][phase_name] = {key: float(_finite(base_profile.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    if phase_name == "regular":
        tuned["weights"] = dict(tuned["weights_by_phase"][phase_name])
    return tuned


def walk_forward_feature_subset_benchmark(
    backtest_df: Optional[pd.DataFrame] = None,
    phase: Optional[str] = None,
    enabled_features: Optional[List[str]] = None,
    base_state: Optional[Dict] = None,
    label: Optional[str] = None,
) -> Dict[str, object]:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    phase_name = normalize_model_phase(phase)
    enabled = [str(f) for f in (enabled_features or []) if str(f) in FEATURE_WEIGHT_KEYS]
    result: Dict[str, object] = {
        "available": False,
        "phase": phase_name or "overall",
        "label": str(label or ", ".join(enabled) or "simple power"),
        "enabled_features": enabled,
        "reason": "no backtest rows",
    }
    if df is None or df.empty:
        return result

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        current_phase = normalize_model_phase(raw_phase)
        if current_phase is None and iso:
            current_phase = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(current_phase or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
        result["phase"] = phase_name
    if df.empty:
        result["reason"] = "no backtest rows for this phase"
        return result

    unique_dates = [dt.date.fromisoformat(iso) for iso in sorted(df["date"].astype(str).unique())]
    splits = walk_forward_date_split(unique_dates)
    if splits is None:
        result["reason"] = "need at least 5 unique backtest dates for a clean benchmark split"
        result["games"] = int(len(df))
        return result

    def subset_for_dates(date_values: List[dt.date]) -> pd.DataFrame:
        wanted = {d.isoformat() for d in date_values}
        return df[df["date"].astype(str).isin(wanted)].copy().reset_index(drop=True)

    train_rows = subset_for_dates(splits["train"])
    validation_rows = subset_for_dates(splits["validation"])
    test_rows = subset_for_dates(splits["test"])

    train_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_rows, refresh_meta=False)
    train_state = apply_feature_subset_to_phase_weights(train_state, result["phase"], enabled)
    train_pred = backtest_rows_with_model_state(train_rows, train_state)
    validation_pred = backtest_rows_with_model_state(validation_rows, train_state)

    train_validation_rows = pd.concat([train_rows, validation_rows], ignore_index=True)
    test_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_validation_rows, refresh_meta=False)
    test_state = apply_feature_subset_to_phase_weights(test_state, result["phase"], enabled)
    test_pred = backtest_rows_with_model_state(test_rows, test_state)
    out_of_sample_pred = pd.concat([validation_pred, test_pred], ignore_index=True)

    result.update({
        "available": True,
        "games": int(len(df)),
        "train_span": format_backtest_date_span(splits["train"]),
        "validation_span": format_backtest_date_span(splits["validation"]),
        "test_span": format_backtest_date_span(splits["test"]),
        "train_metrics": backtest_prediction_metrics(train_pred),
        "validation_metrics": backtest_prediction_metrics(validation_pred),
        "test_metrics": backtest_prediction_metrics(test_pred),
        "out_of_sample_metrics": backtest_prediction_metrics(out_of_sample_pred),
        "train_total_metrics": total_prediction_metrics(train_pred, calibration_state=train_state),
        "validation_total_metrics": total_prediction_metrics(validation_pred, calibration_state=train_state),
        "test_total_metrics": total_prediction_metrics(test_pred, calibration_state=test_state),
        "out_of_sample_total_metrics": total_prediction_metrics(out_of_sample_pred, calibration_state=test_state),
    })
    return result


def regular_earn_back_active_weights(state: Dict) -> Dict[str, float]:
    state = ensure_phase_weight_state(state)
    raw_regular = {key: float(_finite(state["weights_by_phase"]["regular"].get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    earn_back = state.get("regular_earn_back") if isinstance(state.get("regular_earn_back"), dict) else {}
    adjusted = earn_back.get("adjusted_weights") if isinstance(earn_back.get("adjusted_weights"), dict) else None
    if not earn_back.get("apply") or adjusted is None:
        return raw_regular
    merged = dict(raw_regular)
    for key, default in BASE_WEIGHT_DEFAULTS.items():
        merged[key] = float(_finite(adjusted.get(key, merged.get(key, default)), merged.get(key, default)))
    return merged


def update_regular_feature_earn_back(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_phase_weight_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    raw_regular = {key: float(_finite(state["weights_by_phase"]["regular"].get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    default_earn_back = {
        "apply": False,
        "mode": "awaiting_regular_sample",
        "enabled_features": [],
        "reason": "regular-season backtest sample not available yet",
        "base_oos_log_loss": None,
        "selected_oos_log_loss": None,
        "feature_results": {},
        "adjusted_weights": simple_power_profile_from_weights(raw_regular, spread_scale=raw_regular.get("spread_scale")),
    }
    default_earn_back["adjusted_weights"]["w_off"] = float(_finite(raw_regular.get("w_off"), default_earn_back["adjusted_weights"]["w_off"]))
    default_earn_back["adjusted_weights"]["home_field"] = float(_finite(raw_regular.get("home_field"), default_earn_back["adjusted_weights"]["home_field"]))

    if df is None or df.empty:
        state["regular_earn_back"] = default_earn_back
        return state

    work = df.copy()
    if "phase" not in work.columns:
        work["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(work["phase"].tolist(), work.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    work["phase"] = normalized_phase
    work["date"] = normalized_date
    work = work[(work["phase"] == "regular") & work["date"].notna()].copy()
    if work.empty:
        state["regular_earn_back"] = default_earn_back
        return state

    unique_dates = sorted(work["date"].astype(str).unique())
    if len(unique_dates) < 5:
        default_earn_back["reason"] = "need at least 5 unique regular-season backtest dates"
        state["regular_earn_back"] = default_earn_back
        return state

    raw_state = copy.deepcopy(state)
    if isinstance(raw_state, dict):
        raw_state.pop("regular_earn_back", None)

    base_result = walk_forward_feature_subset_benchmark(work, phase="regular", enabled_features=[], base_state=raw_state, label="simple power")
    base_oos = base_result.get("out_of_sample_metrics", {}) if isinstance(base_result, dict) else {}
    base_ll = safe_float(base_oos.get("log_loss"))
    if base_ll is None:
        default_earn_back["reason"] = "regular earn-back baseline metrics unavailable"
        state["regular_earn_back"] = default_earn_back
        return state

    selected: List[str] = []
    remaining = list(FEATURE_WEIGHT_KEYS.keys())
    current_ll = float(base_ll)
    feature_results: Dict[str, Dict[str, object]] = {}

    while remaining:
        best_feature: Optional[str] = None
        best_result: Optional[Dict[str, object]] = None
        best_improvement = 0.0
        for feature in remaining:
            candidate = walk_forward_feature_subset_benchmark(
                work,
                phase="regular",
                enabled_features=selected + [feature],
                base_state=raw_state,
                label=f"simple power + {feature}",
            )
            candidate_ll = safe_float((candidate.get("out_of_sample_metrics") or {}).get("log_loss"))
            improvement = float(current_ll - candidate_ll) if candidate_ll is not None else None
            feature_results[feature] = {
                "available": bool(candidate.get("available")),
                "oos_log_loss": candidate_ll,
                "improvement": improvement,
            }
            if candidate_ll is None:
                continue
            if improvement is not None and improvement > best_improvement:
                best_improvement = float(improvement)
                best_feature = feature
                best_result = candidate
        if best_feature is None or best_result is None or best_improvement <= 0.005:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        current_ll = float((best_result.get("out_of_sample_metrics") or {}).get("log_loss"))

    adjusted_state = apply_feature_subset_to_phase_weights(state, "regular", selected)
    adjusted_weights = adjusted_state["weights_by_phase"]["regular"]
    earn_back = dict(default_earn_back)
    earn_back["apply"] = True
    earn_back["mode"] = "greedy_feature_earn_back"
    earn_back["enabled_features"] = selected
    earn_back["reason"] = (
        "regular-season features must beat the simple power baseline one by one" if selected else
        "regular-season simple power baseline still beats every tested add-on feature"
    )
    earn_back["base_oos_log_loss"] = float(base_ll)
    earn_back["selected_oos_log_loss"] = float(current_ll)
    earn_back["feature_results"] = feature_results
    earn_back["adjusted_weights"] = {key: float(_finite(adjusted_weights.get(key, default), default)) for key, default in BASE_WEIGHT_DEFAULTS.items()}
    state["regular_earn_back"] = earn_back
    return state
def update_probability_calibration(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_probability_calibration_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    default_profiles = {
        "spring": identity_probability_calibration("no spring backtest rows available for probability calibration"),
        "regular": identity_probability_calibration("no regular-season backtest rows available for probability calibration"),
    }
    if df is None or df.empty:
        state["probability_calibration"] = default_profiles
        return state

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if df.empty:
        state["probability_calibration"] = default_profiles
        return state

    calibration_state = copy.deepcopy(ensure_probability_calibration_state(state))
    calibration_state["probability_calibration"] = {
        "spring": identity_probability_calibration(),
        "regular": identity_probability_calibration(),
    }

    fitted_profiles: Dict[str, Dict[str, object]] = {}
    for phase_name in ["spring", "regular"]:
        phase_rows = df[df["phase"] == phase_name].copy()
        profile = identity_probability_calibration(f"no {phase_name} backtest rows available for probability calibration")
        if phase_rows.empty:
            fitted_profiles[phase_name] = profile
            continue

        raw_pred = backtest_rows_with_model_state(phase_rows, calibration_state, apply_calibration=False)
        if raw_pred.empty:
            profile["reason"] = f"unable to build {phase_name} raw prediction rows for calibration"
            fitted_profiles[phase_name] = profile
            continue

        raw_pred["p_home"] = pd.to_numeric(raw_pred.get("p_home"), errors="coerce")
        raw_pred["y_home"] = pd.to_numeric(raw_pred.get("y_home"), errors="coerce")
        raw_pred["logit"] = pd.to_numeric(raw_pred.get("logit"), errors="coerce")
        raw_pred = raw_pred.dropna(subset=["p_home", "y_home", "logit"]).copy()
        if raw_pred.empty:
            profile["reason"] = f"no valid {phase_name} prediction rows available for calibration"
            fitted_profiles[phase_name] = profile
            continue

        p_raw = raw_pred["p_home"].astype(float).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
        y = raw_pred["y_home"].astype(int).to_numpy(dtype=float)
        x = raw_pred["logit"].astype(float).to_numpy(dtype=float)
        n_games = int(len(raw_pred))
        raw_log_loss = float((-(y * np.log(p_raw)) - ((1.0 - y) * np.log(1.0 - p_raw))).mean())

        profile["games"] = n_games
        profile["raw_log_loss"] = raw_log_loss
        if n_games < PROBABILITY_CALIBRATION_MIN_GAMES:
            profile["reason"] = f"need at least {PROBABILITY_CALIBRATION_MIN_GAMES} {phase_name} games before enabling probability calibration"
            fitted_profiles[phase_name] = profile
            continue
        if int(np.nansum(y)) == 0 or int(np.nansum(1.0 - y)) == 0:
            profile["reason"] = f"need both wins and losses in {phase_name} calibration sample"
            fitted_profiles[phase_name] = profile
            continue

        if "date" in raw_pred.columns:
            dts = pd.to_datetime(raw_pred["date"], errors="coerce")
            latest = dts.max()
            if pd.isna(latest):
                age_days = np.zeros(len(raw_pred), dtype=float)
            else:
                age_days = (latest - dts).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
        else:
            age_days = np.zeros(len(raw_pred), dtype=float)
        sample_w = np.power(0.5, age_days / PROBABILITY_CALIBRATION_HALF_LIFE_DAYS)
        sample_w = np.where(np.isfinite(sample_w), sample_w, 1.0)
        weight_sum = float(sample_w.sum()) or float(len(sample_w)) or 1.0

        intercept = 0.0
        slope = 1.0
        for _ in range(300):
            z = np.clip(intercept + (slope * x), -20.0, 20.0)
            p_cal = 1.0 / (1.0 + np.exp(-z))
            err = (p_cal - y) * sample_w
            g_intercept = float(err.sum() / weight_sum) + (PROBABILITY_CALIBRATION_L2_INTERCEPT * intercept)
            g_slope = float((err * x).sum() / weight_sum) + (PROBABILITY_CALIBRATION_L2_SLOPE * (slope - 1.0))
            intercept = float(np.clip(intercept - (PROBABILITY_CALIBRATION_LR * g_intercept), -1.25, 1.25))
            slope = float(np.clip(slope - (PROBABILITY_CALIBRATION_LR * g_slope), 0.35, 1.35))

        shrink = float(n_games / (n_games + 45.0))
        intercept *= shrink
        slope = 1.0 + ((slope - 1.0) * shrink)
        calibrated_prob = 1.0 / (1.0 + np.exp(-np.clip(intercept + (slope * x), -20.0, 20.0)))
        calibrated_log_loss = float((-(y * np.log(calibrated_prob)) - ((1.0 - y) * np.log(1.0 - calibrated_prob))).mean())
        improvement = float(raw_log_loss - calibrated_log_loss)

        profile.update({
            "apply": bool(improvement > PROBABILITY_CALIBRATION_MIN_IMPROVEMENT),
            "mode": "platt_scaling" if improvement > PROBABILITY_CALIBRATION_MIN_IMPROVEMENT else "identity",
            "intercept": float(intercept),
            "slope": float(slope),
            "games": n_games,
            "raw_log_loss": raw_log_loss,
            "calibrated_log_loss": calibrated_log_loss,
            "reason": (
                f"{phase_name} probability calibration improved log loss by {improvement:.4f} over {n_games} historical games"
                if improvement > PROBABILITY_CALIBRATION_MIN_IMPROVEMENT else
                f"{phase_name} probability calibration did not improve log loss enough to activate ({raw_log_loss:.4f} to {calibrated_log_loss:.4f})"
            ),
        })
        fitted_profiles[phase_name] = profile

    state["probability_calibration"] = fitted_profiles
    return state


def update_total_calibration(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_total_calibration_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    default_profiles = {
        "spring": identity_total_calibration("no spring backtest rows available for total calibration"),
        "regular": identity_total_calibration("no regular-season backtest rows available for total calibration"),
    }
    if df is None or df.empty:
        state["total_calibration"] = default_profiles
        return state

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if df.empty:
        state["total_calibration"] = default_profiles
        return state

    fitted_profiles: Dict[str, Dict[str, object]] = {}
    for phase_name in ["spring", "regular"]:
        phase_rows = df[df["phase"] == phase_name].copy()
        profile = identity_total_calibration(f"no {phase_name} backtest rows available for total calibration")
        raw_rows = total_prediction_rows(phase_rows, apply_calibration=False)
        if raw_rows.empty:
            fitted_profiles[phase_name] = profile
            continue

        x = pd.to_numeric(raw_rows["projected_total"], errors="coerce").astype(float).to_numpy(dtype=float)
        y = pd.to_numeric(raw_rows["actual_total"], errors="coerce").astype(float).to_numpy(dtype=float)
        n_games = int(len(raw_rows))
        raw_rmse = float(np.sqrt(np.mean(np.square(y - x))))
        profile["games"] = n_games
        profile["raw_rmse"] = raw_rmse
        if n_games < TOTAL_CALIBRATION_MIN_GAMES:
            profile["reason"] = f"need at least {TOTAL_CALIBRATION_MIN_GAMES} {phase_name} games before enabling total calibration"
            fitted_profiles[phase_name] = profile
            continue

        dts = pd.to_datetime(raw_rows.get("date"), errors="coerce")
        latest = dts.max()
        if pd.isna(latest):
            age_days = np.zeros(len(raw_rows), dtype=float)
        else:
            age_days = (latest - dts).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
        sample_w = np.power(0.5, age_days / TOTAL_CALIBRATION_HALF_LIFE_DAYS)
        sample_w = np.where(np.isfinite(sample_w), sample_w, 1.0)
        weight_sum = float(sample_w.sum()) or float(len(sample_w)) or 1.0

        x_bar = float(np.sum(sample_w * x) / weight_sum)
        y_bar = float(np.sum(sample_w * y) / weight_sum)
        var_x = float(np.sum(sample_w * np.square(x - x_bar)) / weight_sum)
        cov_xy = float(np.sum(sample_w * (x - x_bar) * (y - y_bar)) / weight_sum)
        slope = (cov_xy / var_x) if var_x > 1e-6 else 1.0
        intercept = y_bar - (slope * x_bar)

        shrink = float(n_games / (n_games + 40.0))
        slope = 1.0 + ((slope - 1.0) * shrink)
        intercept *= shrink
        slope = float(np.clip(slope, 0.65, 1.35))
        intercept = float(np.clip(intercept, -2.5, 2.5))

        affine_pred = np.clip(intercept + (slope * x), 5.5, 14.5)
        affine_rmse = float(np.sqrt(np.mean(np.square(y - affine_pred))))

        tail_threshold = 9.0
        excess = np.clip(x - tail_threshold, 0.0, None)
        var_excess = float(np.sum(sample_w * np.square(excess)) / weight_sum)
        tail_slope = 0.0
        n_high = int(np.sum(excess > 0.0))
        if var_excess > 1e-6 and n_high > 0:
            residual = y - affine_pred
            cov_excess = float(np.sum(sample_w * excess * residual) / weight_sum)
            tail_slope = cov_excess / var_excess
            tail_shrink = float((n_high / (n_high + 18.0)) * shrink)
            tail_slope = float(np.clip(tail_slope * tail_shrink, -1.0, 1.25))

        nonlinear_pred = np.clip(intercept + (slope * x) + (tail_slope * excess), 5.5, 14.5)
        nonlinear_rmse = float(np.sqrt(np.mean(np.square(y - nonlinear_pred))))
        use_tail = bool((n_high >= 6) and ((affine_rmse - nonlinear_rmse) > 0.01) and (abs(tail_slope) >= 0.03))
        final_tail_slope = tail_slope if use_tail else 0.0
        calibrated = nonlinear_pred if use_tail else affine_pred
        calibrated_rmse = nonlinear_rmse if use_tail else affine_rmse
        improvement = float(raw_rmse - calibrated_rmse)
        mode = "tail_total_calibration" if (improvement > TOTAL_CALIBRATION_MIN_IMPROVEMENT and use_tail) else ("affine_total_calibration" if improvement > TOTAL_CALIBRATION_MIN_IMPROVEMENT else "identity")
        if improvement > TOTAL_CALIBRATION_MIN_IMPROVEMENT:
            if use_tail:
                reason = f"{phase_name} total calibration improved RMSE by {improvement:.3f} over {n_games} historical games with tail slope {final_tail_slope:+.3f} above {tail_threshold:.1f}"
            else:
                reason = f"{phase_name} total calibration improved RMSE by {improvement:.3f} over {n_games} historical games"
        else:
            reason = f"{phase_name} total calibration did not improve RMSE enough to activate ({raw_rmse:.3f} to {calibrated_rmse:.3f})"

        profile.update({
            "apply": bool(improvement > TOTAL_CALIBRATION_MIN_IMPROVEMENT),
            "mode": mode,
            "intercept": intercept,
            "slope": slope,
            "tail_slope": float(final_tail_slope),
            "tail_threshold": float(tail_threshold),
            "games": n_games,
            "raw_rmse": raw_rmse,
            "calibrated_rmse": calibrated_rmse,
            "reason": reason,
        })
        fitted_profiles[phase_name] = profile

    state["total_calibration"] = fitted_profiles
    return state


def update_margin_calibration(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_margin_calibration_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    default_profiles = {
        "spring": identity_margin_calibration("no spring backtest rows available for margin calibration"),
        "regular": identity_margin_calibration("no regular-season backtest rows available for margin calibration"),
    }
    if df is None or df.empty:
        state["margin_calibration"] = default_profiles
        return state

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if df.empty:
        state["margin_calibration"] = default_profiles
        return state

    fitted_profiles: Dict[str, Dict[str, object]] = {}
    for phase_name in ["spring", "regular"]:
        phase_rows = df[df["phase"] == phase_name].copy()
        profile = identity_margin_calibration(f"no {phase_name} backtest rows available for margin calibration")
        if phase_rows.empty:
            fitted_profiles[phase_name] = profile
            continue

        for col in ["projected_margin_raw", "projected_margin", "away_expected_runs", "home_expected_runs", "home_score", "away_score"]:
            if col not in phase_rows.columns:
                phase_rows[col] = np.nan
            phase_rows[col] = pd.to_numeric(phase_rows[col], errors="coerce")
        missing_raw = phase_rows["projected_margin_raw"].isna()
        if missing_raw.any() and "projected_margin" in phase_rows.columns:
            phase_rows.loc[missing_raw, "projected_margin_raw"] = phase_rows.loc[missing_raw, "projected_margin"]
        missing_raw = phase_rows["projected_margin_raw"].isna()
        if missing_raw.any() and {"home_expected_runs", "away_expected_runs"}.issubset(phase_rows.columns):
            phase_rows.loc[missing_raw, "projected_margin_raw"] = phase_rows.loc[missing_raw, "home_expected_runs"] - phase_rows.loc[missing_raw, "away_expected_runs"]
        phase_rows["actual_margin"] = phase_rows["home_score"] - phase_rows["away_score"]
        phase_rows = phase_rows.dropna(subset=["projected_margin_raw", "actual_margin"]).copy()
        if phase_rows.empty:
            profile["reason"] = f"no valid {phase_name} margin rows available for calibration"
            fitted_profiles[phase_name] = profile
            continue

        x = pd.to_numeric(phase_rows["projected_margin_raw"], errors="coerce").astype(float).to_numpy(dtype=float)
        y = pd.to_numeric(phase_rows["actual_margin"], errors="coerce").astype(float).to_numpy(dtype=float)
        n_games = int(len(phase_rows))
        raw_rmse = float(np.sqrt(np.mean(np.square(y - x))))
        profile["games"] = n_games
        profile["raw_rmse"] = raw_rmse
        if n_games < MARGIN_CALIBRATION_MIN_GAMES:
            profile["reason"] = f"need at least {MARGIN_CALIBRATION_MIN_GAMES} {phase_name} games before enabling margin calibration"
            fitted_profiles[phase_name] = profile
            continue

        dts = pd.to_datetime(phase_rows.get("date"), errors="coerce")
        latest = dts.max()
        if pd.isna(latest):
            age_days = np.zeros(len(phase_rows), dtype=float)
        else:
            age_days = (latest - dts).dt.days.fillna(0).clip(lower=0).to_numpy(dtype=float)
        sample_w = np.power(0.5, age_days / MARGIN_CALIBRATION_HALF_LIFE_DAYS)
        sample_w = np.where(np.isfinite(sample_w), sample_w, 1.0)
        weight_sum = float(sample_w.sum()) or float(len(sample_w)) or 1.0

        x_bar = float(np.sum(sample_w * x) / weight_sum)
        y_bar = float(np.sum(sample_w * y) / weight_sum)
        var_x = float(np.sum(sample_w * np.square(x - x_bar)) / weight_sum)
        cov_xy = float(np.sum(sample_w * (x - x_bar) * (y - y_bar)) / weight_sum)
        slope = (cov_xy / var_x) if var_x > 1e-6 else 1.0
        intercept = y_bar - (slope * x_bar)

        shrink = float(n_games / (n_games + 40.0))
        slope = 1.0 + ((slope - 1.0) * shrink)
        intercept *= shrink
        slope = float(np.clip(slope, 0.50, 1.50))
        intercept = float(np.clip(intercept, -1.5, 1.5))

        calibrated = np.clip(intercept + (slope * x), -6.5, 6.5)
        calibrated_rmse = float(np.sqrt(np.mean(np.square(y - calibrated))))
        improvement = float(raw_rmse - calibrated_rmse)
        profile.update({
            "apply": bool(improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT),
            "mode": "affine_margin_calibration" if improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT else "identity",
            "intercept": intercept,
            "slope": slope,
            "games": n_games,
            "raw_rmse": raw_rmse,
            "calibrated_rmse": calibrated_rmse,
            "reason": (
                f"{phase_name} margin calibration improved RMSE by {improvement:.3f} over {n_games} historical games"
                if improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT else
                f"{phase_name} margin calibration did not improve RMSE enough to activate ({raw_rmse:.3f} to {calibrated_rmse:.3f})"
            ),
        })
        fitted_profiles[phase_name] = profile

    state["margin_calibration"] = fitted_profiles
    return state


def walk_forward_prediction_frames(
    backtest_df: Optional[pd.DataFrame] = None,
    phase: Optional[str] = None,
    base_state: Optional[Dict] = None,
) -> Dict[str, object]:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    phase_name = normalize_model_phase(phase)
    result: Dict[str, object] = {
        "available": False,
        "phase": phase_name or "overall",
        "reason": "no backtest rows",
    }
    if df is None or df.empty:
        return result

    if "phase" not in df.columns:
        df["phase"] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df["phase"].tolist(), df.get("date", pd.Series(dtype=str)).tolist()):
        iso = normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        current_phase = normalize_model_phase(raw_phase)
        if current_phase is None and iso:
            current_phase = season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(current_phase or "regular")
    df["phase"] = normalized_phase
    df["date"] = normalized_date
    df = df[df["date"].notna()].copy()
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
        result["phase"] = phase_name
    if df.empty:
        result["reason"] = "no backtest rows for this phase"
        return result

    unique_dates = [dt.date.fromisoformat(iso) for iso in sorted(df["date"].astype(str).unique())]
    splits = walk_forward_date_split(unique_dates)
    if splits is None:
        result["reason"] = "need at least 5 unique backtest dates for a clean train/validation/test split"
        result["games"] = int(len(df))
        result["dates"] = [d.isoformat() for d in unique_dates]
        return result

    def subset_for_dates(date_values: List[dt.date]) -> pd.DataFrame:
        wanted = {d.isoformat() for d in date_values}
        return df[df["date"].astype(str).isin(wanted)].copy().reset_index(drop=True)

    train_rows = subset_for_dates(splits["train"])
    validation_rows = subset_for_dates(splits["validation"])
    test_rows = subset_for_dates(splits["test"])

    train_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_rows, refresh_meta=False)
    train_pred = backtest_rows_with_model_state(train_rows, train_state)
    validation_pred = backtest_rows_with_model_state(validation_rows, train_state)

    train_validation_rows = pd.concat([train_rows, validation_rows], ignore_index=True)
    test_state = recalibrate_model_weights_from_log(evaluation_base_model_state(base_state), train_validation_rows, refresh_meta=False)
    test_pred = backtest_rows_with_model_state(test_rows, test_state)
    out_of_sample_pred = pd.concat([validation_pred, test_pred], ignore_index=True)

    result.update({
        "available": True,
        "phase": result["phase"],
        "games": int(len(df)),
        "split_dates": {key: [d.isoformat() for d in value] for key, value in splits.items()},
        "train_span": format_backtest_date_span(splits["train"]),
        "validation_span": format_backtest_date_span(splits["validation"]),
        "test_span": format_backtest_date_span(splits["test"]),
        "train_pred": train_pred,
        "validation_pred": validation_pred,
        "test_pred": test_pred,
        "out_of_sample_pred": out_of_sample_pred,
        "train_metrics": backtest_prediction_metrics(train_pred),
        "validation_metrics": backtest_prediction_metrics(validation_pred),
        "test_metrics": backtest_prediction_metrics(test_pred),
        "out_of_sample_metrics": backtest_prediction_metrics(out_of_sample_pred),
        "train_total_metrics": total_prediction_metrics(train_pred, calibration_state=train_state),
        "validation_total_metrics": total_prediction_metrics(validation_pred, calibration_state=train_state),
        "test_total_metrics": total_prediction_metrics(test_pred, calibration_state=test_state),
        "out_of_sample_total_metrics": total_prediction_metrics(out_of_sample_pred, calibration_state=test_state),
    })
    return result


def prediction_pick_rows(pred_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    work = pred_df.copy()
    work["p_home"] = pd.to_numeric(work.get("p_home"), errors="coerce")
    work["y_home"] = pd.to_numeric(work.get("y_home"), errors="coerce")
    if "pred_home" in work.columns:
        work["pred_home"] = pd.to_numeric(work.get("pred_home"), errors="coerce")
    else:
        work["pred_home"] = np.where(work["p_home"] >= 0.5, 1, 0)
    work = work.dropna(subset=["p_home", "y_home", "pred_home"]).copy()
    if work.empty:
        return work

    work["pred_home"] = work["pred_home"].astype(int)
    work["y_home"] = work["y_home"].astype(int)
    work["pick_prob"] = np.where(work["pred_home"] == 1, work["p_home"], 1.0 - work["p_home"])
    work["pick_prob"] = pd.to_numeric(work["pick_prob"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    work["pick_result"] = (work["pred_home"] == work["y_home"]).astype(int)
    work["pick_side"] = np.where(work["pred_home"] == 1, "Home", "Away")
    work["confidence_tier"] = [confidence_tier(float(prob), phase) for prob, phase in zip(work["pick_prob"].tolist(), work["phase"].tolist())]
    return work.reset_index(drop=True)


def predicted_side_metrics(pred_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    rows = prediction_pick_rows(pred_df)
    metrics = {
        "games": 0,
        "win_rate": None,
        "avg_confidence": None,
        "log_loss": None,
    }
    if rows.empty:
        return metrics

    p = rows["pick_prob"].astype(float)
    y = rows["pick_result"].astype(int)
    metrics["games"] = int(len(rows))
    metrics["win_rate"] = float(y.mean())
    metrics["avg_confidence"] = float(p.mean())
    metrics["log_loss"] = float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean())
    return metrics


def prediction_calibration_table(pred_df: Optional[pd.DataFrame], bucket_size: float = 0.05) -> List[Dict[str, object]]:
    rows = prediction_pick_rows(pred_df)
    if rows.empty:
        return []

    p = rows["pick_prob"].astype(float)
    y = rows["pick_result"].astype(int)
    out: List[Dict[str, object]] = []
    start = 0.50
    while start < 1.0:
        end = min(start + bucket_size, 1.000001)
        if end >= 1.0:
            mask = (p >= start) & (p <= 1.0)
        else:
            mask = (p >= start) & (p < end)
        if mask.any():
            avg_conf = float(p[mask].mean())
            actual = float(y[mask].mean())
            out.append({
                "label": f"{int(round(start * 100))}-{int(round(min(end, 1.0) * 100))}%",
                "games": int(mask.sum()),
                "avg_confidence": avg_conf,
                "actual_rate": actual,
                "gap": float(actual - avg_conf),
            })
        start += bucket_size
    return out


def grouped_prediction_segments(pred_df: Optional[pd.DataFrame], group_col: str, order: Optional[List[str]] = None) -> List[Dict[str, object]]:
    rows = prediction_pick_rows(pred_df)
    if rows.empty or group_col not in rows.columns:
        return []

    labels = [str(v) for v in (order or []) if str(v) in set(rows[group_col].astype(str).tolist())]
    if not labels:
        labels = sorted({str(v) for v in rows[group_col].astype(str).tolist() if str(v).strip()})

    out: List[Dict[str, object]] = []
    for label in labels:
        group = rows[rows[group_col].astype(str) == label].copy()
        if group.empty:
            continue
        metrics = predicted_side_metrics(group)
        out.append({
            "label": label,
            "games": int(metrics["games"]),
            "win_rate": metrics["win_rate"],
            "avg_confidence": metrics["avg_confidence"],
            "log_loss": metrics["log_loss"],
        })
    return out


def total_prediction_rows(
    pred_df: Optional[pd.DataFrame],
    calibration_state: Optional[Dict] = None,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    work = pred_df.copy()
    for col in ["projected_total", "away_expected_runs", "home_expected_runs", "home_score", "away_score"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    missing_total = work["projected_total"].isna()
    if missing_total.any():
        work.loc[missing_total, "projected_total"] = work.loc[missing_total, "away_expected_runs"] + work.loc[missing_total, "home_expected_runs"]
    work["actual_total"] = work["home_score"] + work["away_score"]
    work = work.dropna(subset=["projected_total", "actual_total"]).copy()
    if work.empty:
        return work

    if apply_calibration:
        state = calibration_state if isinstance(calibration_state, dict) else load_model_state()
        if "phase" not in work.columns:
            work["phase"] = np.nan
        calibrated_totals: List[float] = []
        for _, row in work.iterrows():
            iso = normalize_backtest_date_str(row.get("date"))
            model_date = dt.date.fromisoformat(iso) if iso else today_local()
            phase_name = normalize_model_phase(row.get("phase")) or season_phase_for_date(model_date)
            profile = active_total_calibration(state, model_date=model_date, phase=phase_name)
            calibrated_totals.append(apply_total_calibration(float(_finite(row.get("projected_total"), 8.5)), profile))
        work["projected_total"] = calibrated_totals

    work["total_error"] = work["actual_total"] - work["projected_total"]
    proj = work["projected_total"].astype(float)
    work["total_band"] = np.where(
        proj < 8.0,
        "Low (<8.0)",
        np.where(proj < 9.0, "Mid (8.0-8.9)", "High (9.0+)"),
    )
    return work.reset_index(drop=True)


def total_prediction_metrics(pred_df: Optional[pd.DataFrame], calibration_state: Optional[Dict] = None, apply_calibration: bool = True) -> Dict[str, object]:
    rows = total_prediction_rows(pred_df, calibration_state=calibration_state, apply_calibration=apply_calibration)
    metrics = {
        "games": 0,
        "avg_projected_total": None,
        "avg_actual_total": None,
        "bias": None,
        "mae": None,
        "rmse": None,
        "within_one_run": None,
    }
    if rows.empty:
        return metrics

    err = pd.to_numeric(rows["total_error"], errors="coerce").dropna().astype(float)
    if err.empty:
        return metrics

    metrics["games"] = int(len(err))
    metrics["avg_projected_total"] = float(pd.to_numeric(rows["projected_total"], errors="coerce").mean())
    metrics["avg_actual_total"] = float(pd.to_numeric(rows["actual_total"], errors="coerce").mean())
    metrics["bias"] = float(err.mean())
    metrics["mae"] = float(np.mean(np.abs(err)))
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(err))))
    metrics["within_one_run"] = float(np.mean(np.abs(err) <= 1.0))
    return metrics


def total_calibration_table(pred_df: Optional[pd.DataFrame], bucket_size: float = 1.0, calibration_state: Optional[Dict] = None, apply_calibration: bool = True) -> List[Dict[str, object]]:
    rows = total_prediction_rows(pred_df, calibration_state=calibration_state, apply_calibration=apply_calibration)
    if rows.empty:
        return []

    projected = pd.to_numeric(rows["projected_total"], errors="coerce").astype(float)
    actual = pd.to_numeric(rows["actual_total"], errors="coerce").astype(float)
    start = float(math.floor(float(projected.min())))
    end = float(math.ceil(float(projected.max())))
    out: List[Dict[str, object]] = []
    current = start
    while current < (end + 1e-9):
        upper = current + bucket_size
        mask = (projected >= current) & (projected < upper)
        if mask.any():
            model_avg = float(projected[mask].mean())
            actual_avg = float(actual[mask].mean())
            out.append({
                "label": f"{current:.1f}-{upper - 0.1:.1f}",
                "games": int(mask.sum()),
                "avg_projected_total": model_avg,
                "avg_actual_total": actual_avg,
                "gap": float(actual_avg - model_avg),
            })
        current += bucket_size
    return out


def grouped_total_segments(pred_df: Optional[pd.DataFrame], order: Optional[List[str]] = None, calibration_state: Optional[Dict] = None, apply_calibration: bool = True) -> List[Dict[str, object]]:
    rows = total_prediction_rows(pred_df, calibration_state=calibration_state, apply_calibration=apply_calibration)
    if rows.empty or "total_band" not in rows.columns:
        return []

    labels = [str(v) for v in (order or []) if str(v) in set(rows["total_band"].astype(str).tolist())]
    if not labels:
        labels = sorted({str(v) for v in rows["total_band"].astype(str).tolist() if str(v).strip()})

    out: List[Dict[str, object]] = []
    for label in labels:
        group = rows[rows["total_band"].astype(str) == label].copy()
        if group.empty:
            continue
        metrics = total_prediction_metrics(group, calibration_state=calibration_state, apply_calibration=False)
        out.append({
            "label": label,
            "games": int(metrics["games"]),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "bias": metrics["bias"],
        })
    return out


def margin_prediction_rows(
    pred_df: Optional[pd.DataFrame],
    sigma: Optional[float] = None,
    model_state: Optional[Dict] = None,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    work = pred_df.copy()
    for col in ["projected_margin", "away_expected_runs", "home_expected_runs", "home_score", "away_score"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    missing_margin = work["projected_margin"].isna()
    if missing_margin.any():
        work.loc[missing_margin, "projected_margin"] = work.loc[missing_margin, "home_expected_runs"] - work.loc[missing_margin, "away_expected_runs"]
    work["actual_margin"] = work["home_score"] - work["away_score"]
    work = work.dropna(subset=["projected_margin", "actual_margin"]).copy()
    if work.empty:
        return work

    sigma_value = float(np.clip(_finite(sigma, estimate_margin_distribution_sigma(model_state, work)), 1.75, 6.0))
    work["margin_error"] = work["actual_margin"] - work["projected_margin"]
    work["margin_abs"] = work["projected_margin"].abs()
    work["margin_band"] = np.where(
        work["margin_abs"] < 0.75,
        "Tight (<0.8)",
        np.where(work["margin_abs"] < 1.5, "Lean (0.8-1.4)", "Strong (1.5+)"),
    )

    projected_margin = work["projected_margin"].astype(float).to_numpy(dtype=float)
    actual_margin = work["actual_margin"].astype(float).to_numpy(dtype=float)
    favorite_home = projected_margin >= 0.0
    favorite_cover_prob = np.where(
        favorite_home,
        1.0 - np.vectorize(normal_cdf)((1.5 - projected_margin) / sigma_value),
        np.vectorize(normal_cdf)((-1.5 - projected_margin) / sigma_value),
    )
    favorite_cover_prob = np.clip(favorite_cover_prob.astype(float), 1e-6, 1.0 - 1e-6)
    favorite_covered = np.where(favorite_home, actual_margin > 1.5, actual_margin < -1.5).astype(float)
    selected_prob = np.where(favorite_cover_prob >= 0.5, favorite_cover_prob, 1.0 - favorite_cover_prob)
    selected_covered = np.where(favorite_cover_prob >= 0.5, favorite_covered, 1.0 - favorite_covered)

    work["favorite_side"] = np.where(favorite_home, "Home favorite", "Away favorite")
    work["favorite_cover_prob"] = favorite_cover_prob
    work["favorite_covered"] = favorite_covered
    work["selected_cover_prob"] = selected_prob
    work["selected_covered"] = selected_covered
    work["run_line_pick"] = np.where(favorite_cover_prob >= 0.5, "Favorite -1.5", "Underdog +1.5")
    work["run_line_correct"] = (selected_covered >= 0.5).astype(float)
    work["margin_sigma"] = sigma_value
    return work.reset_index(drop=True)


def margin_prediction_metrics(
    pred_df: Optional[pd.DataFrame],
    sigma: Optional[float] = None,
    model_state: Optional[Dict] = None,
) -> Dict[str, object]:
    rows = margin_prediction_rows(pred_df, sigma=sigma, model_state=model_state)
    metrics = {
        "games": 0,
        "avg_projected_margin": None,
        "avg_actual_margin": None,
        "bias": None,
        "mae": None,
        "rmse": None,
        "within_one_run": None,
    }
    if rows.empty:
        return metrics

    err = pd.to_numeric(rows["margin_error"], errors="coerce").dropna().astype(float)
    if err.empty:
        return metrics

    metrics["games"] = int(len(err))
    metrics["avg_projected_margin"] = float(pd.to_numeric(rows["projected_margin"], errors="coerce").mean())
    metrics["avg_actual_margin"] = float(pd.to_numeric(rows["actual_margin"], errors="coerce").mean())
    metrics["bias"] = float(err.mean())
    metrics["mae"] = float(np.mean(np.abs(err)))
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(err))))
    metrics["within_one_run"] = float(np.mean(np.abs(err) <= 1.0))
    return metrics


def run_line_prediction_metrics(
    pred_df: Optional[pd.DataFrame],
    sigma: Optional[float] = None,
    model_state: Optional[Dict] = None,
) -> Dict[str, object]:
    rows = margin_prediction_rows(pred_df, sigma=sigma, model_state=model_state)
    metrics = {
        "games": 0,
        "brier": None,
        "accuracy": None,
        "avg_selected_cover_prob": None,
        "selected_cover_rate": None,
        "favorite_cover_rate": None,
    }
    if rows.empty:
        return metrics

    prob = pd.to_numeric(rows["selected_cover_prob"], errors="coerce").dropna().astype(float)
    outcome = pd.to_numeric(rows["selected_covered"], errors="coerce").dropna().astype(float)
    if prob.empty or outcome.empty:
        return metrics

    metrics["games"] = int(len(rows))
    metrics["brier"] = float(np.mean(np.square(prob - outcome)))
    metrics["accuracy"] = float(np.mean((prob >= 0.5).astype(float) == outcome))
    metrics["avg_selected_cover_prob"] = float(prob.mean())
    metrics["selected_cover_rate"] = float(outcome.mean())
    favorite_cov = pd.to_numeric(rows["favorite_covered"], errors="coerce").dropna().astype(float)
    metrics["favorite_cover_rate"] = float(favorite_cov.mean()) if not favorite_cov.empty else None
    return metrics


def run_line_calibration_table(
    pred_df: Optional[pd.DataFrame],
    bucket_size: float = 0.10,
    sigma: Optional[float] = None,
    model_state: Optional[Dict] = None,
) -> List[Dict[str, object]]:
    rows = margin_prediction_rows(pred_df, sigma=sigma, model_state=model_state)
    if rows.empty:
        return []

    projected = pd.to_numeric(rows["selected_cover_prob"], errors="coerce").astype(float)
    actual = pd.to_numeric(rows["selected_covered"], errors="coerce").astype(float)
    start = 0.50
    end = float(min(1.0, math.ceil(float(projected.max()) * 10.0) / 10.0))
    out: List[Dict[str, object]] = []
    current = start
    while current < (end + 1e-9):
        upper = current + bucket_size
        mask = (projected >= current) & (projected < upper)
        if mask.any():
            model_avg = float(projected[mask].mean())
            actual_avg = float(actual[mask].mean())
            out.append({
                "label": f"{current:.2f}-{upper - 0.01:.2f}",
                "games": int(mask.sum()),
                "avg_selected_prob": model_avg,
                "actual_cover_rate": actual_avg,
                "gap": float(actual_avg - model_avg),
            })
        current += bucket_size
    return out


def grouped_margin_segments(
    pred_df: Optional[pd.DataFrame],
    group_col: str,
    order: Optional[List[str]] = None,
    sigma: Optional[float] = None,
    model_state: Optional[Dict] = None,
) -> List[Dict[str, object]]:
    rows = margin_prediction_rows(pred_df, sigma=sigma, model_state=model_state)
    if rows.empty or group_col not in rows.columns:
        return []

    labels = [str(v) for v in (order or []) if str(v) in set(rows[group_col].astype(str).tolist())]
    if not labels:
        labels = sorted({str(v) for v in rows[group_col].astype(str).tolist() if str(v).strip()})

    out: List[Dict[str, object]] = []
    for label in labels:
        group = rows[rows[group_col].astype(str) == label].copy()
        if group.empty:
            continue
        margin_metrics = margin_prediction_metrics(group, sigma=sigma, model_state=model_state)
        run_line_metrics = run_line_prediction_metrics(group, sigma=sigma, model_state=model_state)
        out.append({
            "label": label,
            "games": int(margin_metrics.get("games", 0) or 0),
            "mae": margin_metrics.get("mae"),
            "rmse": margin_metrics.get("rmse"),
            "bias": margin_metrics.get("bias"),
            "run_line_accuracy": run_line_metrics.get("accuracy"),
            "run_line_brier": run_line_metrics.get("brier"),
        })
    return out


def prediction_scoreboard_summary(
    backtest_df: Optional[pd.DataFrame] = None,
    phase: Optional[str] = None,
    base_state: Optional[Dict] = None,
) -> Dict[str, object]:
    frames = walk_forward_prediction_frames(backtest_df=backtest_df, phase=phase, base_state=base_state)
    result: Dict[str, object] = {
        "available": bool(frames.get("available")),
        "phase": frames.get("phase", normalize_model_phase(phase) or "overall"),
        "reason": frames.get("reason"),
        "oos_metrics": {},
        "pick_metrics": {},
        "total_metrics": {},
        "margin_metrics": {},
        "run_line_metrics": {},
        "calibration": [],
        "total_calibration": [],
        "run_line_calibration": [],
        "confidence_segments": [],
        "actionability_segments": [],
        "side_segments": [],
        "total_segments": [],
        "margin_segments": [],
        "favorite_side_segments": [],
    }
    if not frames.get("available"):
        return result

    oos_pred = frames.get("out_of_sample_pred")
    margin_sigma = estimate_margin_distribution_sigma(base_state, oos_pred)
    result.update({
        "oos_metrics": frames.get("out_of_sample_metrics", {}),
        "pick_metrics": predicted_side_metrics(oos_pred),
        "total_metrics": total_prediction_metrics(oos_pred, calibration_state=base_state),
        "margin_metrics": margin_prediction_metrics(oos_pred, sigma=margin_sigma, model_state=base_state),
        "run_line_metrics": run_line_prediction_metrics(oos_pred, sigma=margin_sigma, model_state=base_state),
        "calibration": prediction_calibration_table(oos_pred),
        "total_calibration": total_calibration_table(oos_pred, calibration_state=base_state),
        "run_line_calibration": run_line_calibration_table(oos_pred, sigma=margin_sigma, model_state=base_state),
        "confidence_segments": grouped_prediction_segments(oos_pred, "confidence_tier", order=["High", "Medium", "Low"]),
        "side_segments": grouped_prediction_segments(oos_pred, "pick_side", order=["Home", "Away"]),
        "total_segments": grouped_total_segments(oos_pred, order=["Low (<8.0)", "Mid (8.0-8.9)", "High (9.0+)"], calibration_state=base_state),
        "margin_segments": grouped_margin_segments(oos_pred, "margin_band", order=["Tight (<0.8)", "Lean (0.8-1.4)", "Strong (1.5+)"], sigma=margin_sigma, model_state=base_state),
        "favorite_side_segments": grouped_margin_segments(oos_pred, "favorite_side", order=["Home favorite", "Away favorite"], sigma=margin_sigma, model_state=base_state),
    })
    return result


def load_bet_journal_df(phase: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(SPORTSBOOK_DB_PATH):
        return pd.DataFrame()
    ensure_market_db()
    with sqlite3.connect(SPORTSBOOK_DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM bet_journal ORDER BY report_date, created_ts, id", conn)
    if df.empty:
        return df

    df["report_date"] = df["report_date"].map(normalize_backtest_date_str)
    df = df[df["report_date"].notna()].copy()
    if df.empty:
        return df

    df["phase"] = [season_phase_for_date(dt.date.fromisoformat(str(iso))) for iso in df["report_date"].astype(str)]
    phase_name = normalize_model_phase(phase)
    if phase_name is not None:
        df = df[df["phase"] == phase_name].copy()
    return df.reset_index(drop=True)


def selection_family_from_row(row: pd.Series) -> str:
    market_type = str(row.get("market_type") or "").strip().lower()
    selection = str(row.get("selection") or "").strip().upper()
    home_team = str(row.get("home_team") or "").strip().upper()
    away_team = str(row.get("away_team") or "").strip().upper()
    if market_type == "total":
        if selection.startswith("OVER"):
            return "Over"
        if selection.startswith("UNDER"):
            return "Under"
        return "Total"
    if selection.startswith(home_team):
        return "Home"
    if selection.startswith(away_team):
        return "Away"
    return "Unknown"


def side_role_from_row(row: pd.Series) -> str:
    market_type = str(row.get("market_type") or "").strip().lower()
    price = parse_market_int(row.get("book_price"))
    line_value = safe_float(row.get("line_value"))
    if market_type == "moneyline":
        if price is None:
            return "Unknown"
        return "Favorite" if price < 0 else "Underdog"
    if market_type == "run_line":
        if line_value is not None:
            if line_value < 0:
                return "Favorite"
            if line_value > 0:
                return "Underdog"
        if price is not None:
            return "Favorite" if price < 0 else "Underdog"
    return "N/A"


def max_drawdown_units(results: pd.Series) -> Optional[float]:
    if results is None:
        return None
    vals = pd.to_numeric(results, errors="coerce").dropna()
    if vals.empty:
        return None
    curve = vals.cumsum()
    drawdown = curve.cummax() - curve
    return float(drawdown.max()) if not drawdown.empty else 0.0


def capture_window_from_row(row: pd.Series) -> str:
    raw = row.get("created_ts")
    try:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return "Unknown"
    local = ts.tz_convert("America/Chicago") if getattr(ts, "tzinfo", None) is not None else ts
    hour = int(local.hour)
    if hour < 11:
        return "Early"
    if hour < 16:
        return "Midday"
    return "Late"


def edge_bucket_from_row(row: pd.Series) -> str:
    report_iso = normalize_backtest_date_str(row.get("report_date"))
    phase_name = season_phase_for_date(dt.date.fromisoformat(report_iso)) if report_iso else None
    return market_edge_tier(str(row.get("market_type") or ""), float(_finite(row.get("edge_value"), 0.0)), model_phase=phase_name)


def grouped_betting_segments(rows: pd.DataFrame, group_col: str, order: Optional[List[str]] = None) -> List[Dict[str, object]]:
    if rows is None or rows.empty or group_col not in rows.columns:
        return []

    labels = [str(v) for v in (order or []) if str(v) in set(rows[group_col].astype(str).tolist())]
    if not labels:
        labels = sorted({str(v) for v in rows[group_col].astype(str).tolist() if str(v).strip()})

    out: List[Dict[str, object]] = []
    for label in labels:
        group = rows[rows[group_col].astype(str) == label].copy()
        if group.empty:
            continue
        stake = pd.to_numeric(group["stake_units"], errors="coerce").fillna(1.0).astype(float)
        result_units = pd.to_numeric(group["result_units"], errors="coerce").fillna(0.0).astype(float)
        clv = pd.to_numeric(group.get("clv_value"), errors="coerce")
        settled = int(len(group))
        profit = float(result_units.sum())
        total_staked = float(stake.sum())
        wins = int((result_units > 0).sum())
        losses = int((result_units < 0).sum())
        pushes = int((result_units == 0).sum())
        out.append({
            "label": label,
            "bets": settled,
            "profit_units": profit,
            "roi": (profit / total_staked) if total_staked > 0 else None,
            "avg_clv": float(clv.dropna().mean()) if clv.notna().any() else None,
            "positive_clv_rate": float((clv.dropna() > 0).mean()) if clv.notna().any() else None,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
        })
    return out


def betting_scoreboard_summary(phase: Optional[str] = None) -> Dict[str, object]:
    df = load_bet_journal_df(phase=phase)
    result: Dict[str, object] = {
        "available": False,
        "phase": normalize_model_phase(phase) or "overall",
        "reason": "no market journal rows yet",
        "candidates": 0,
        "open_watch": 0,
        "settled": 0,
        "graded": 0,
        "profit_units": None,
        "roi": None,
        "yield_per_bet": None,
        "avg_clv": None,
        "positive_clv_rate": None,
        "max_drawdown": None,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "market_segments": [],
        "confidence_segments": [],
        "role_segments": [],
        "selection_segments": [],
        "sportsbook_segments": [],
        "edge_segments": [],
        "capture_segments": [],
    }
    if df.empty:
        return result

    result["available"] = True
    result["candidates"] = int(len(df))
    result["open_watch"] = int((df["result_units"].isna() & (df["status"].astype(str) == "watch")).sum())
    result["graded"] = int((df["result_units"].notna()).sum())

    settled = df[df["result_units"].notna()].copy()
    if settled.empty:
        result["reason"] = "market candidates exist, but none have settled yet"
        return result

    settled["stake_units"] = pd.to_numeric(settled["stake_units"], errors="coerce").fillna(1.0).astype(float)
    settled["result_units"] = pd.to_numeric(settled["result_units"], errors="coerce").fillna(0.0).astype(float)
    settled["clv_value"] = pd.to_numeric(settled.get("clv_value"), errors="coerce")
    settled["selection_family"] = settled.apply(selection_family_from_row, axis=1)
    settled["side_role"] = settled.apply(side_role_from_row, axis=1)
    settled["capture_window"] = settled.apply(capture_window_from_row, axis=1)
    settled["edge_tier"] = settled.apply(edge_bucket_from_row, axis=1)
    settled["sportsbook"] = settled["sportsbook"].fillna("Unknown").astype(str)

    total_staked = float(settled["stake_units"].sum())
    profit_units = float(settled["result_units"].sum())
    wins = int((settled["result_units"] > 0).sum())
    losses = int((settled["result_units"] < 0).sum())
    pushes = int((settled["result_units"] == 0).sum())
    clv = settled["clv_value"]

    result.update({
        "settled": int(len(settled)),
        "profit_units": profit_units,
        "roi": (profit_units / total_staked) if total_staked > 0 else None,
        "yield_per_bet": (profit_units / float(len(settled))) if len(settled) else None,
        "avg_clv": float(clv.dropna().mean()) if clv.notna().any() else None,
        "positive_clv_rate": float((clv.dropna() > 0).mean()) if clv.notna().any() else None,
        "max_drawdown": max_drawdown_units(settled.sort_values(["report_date", "id"])["result_units"]),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "market_segments": grouped_betting_segments(settled, "market_type", order=["moneyline", "run_line", "total"]),
        "confidence_segments": grouped_betting_segments(settled, "confidence_tier", order=["High", "Medium", "Low"]),
        "actionability_segments": grouped_betting_segments(settled[settled["actionability_label"].astype(str).isin(["MUST TAKE", "BET", "WATCH"])], "actionability_label", order=["MUST TAKE", "BET", "WATCH"]),
        "role_segments": grouped_betting_segments(settled[settled["side_role"].isin(["Favorite", "Underdog"])], "side_role", order=["Favorite", "Underdog"]),
        "selection_segments": grouped_betting_segments(settled[settled["selection_family"].isin(["Home", "Away", "Over", "Under"])], "selection_family", order=["Home", "Away", "Over", "Under"]),
        "sportsbook_segments": grouped_betting_segments(settled, "sportsbook"),
        "edge_segments": grouped_betting_segments(settled, "edge_tier", order=["High", "Medium", "Low"]),
        "capture_segments": grouped_betting_segments(settled, "capture_window", order=["Early", "Midday", "Late"]),
    })
    return result
def write_txt_report(report_date, games, strengths, pitch_comp, pen3, pen7, park_factors, weather_map, game_context_map, market_map, model_state, warnings):
    fname = os.path.join(OUT_DIR, f"MLB_Report_{report_date.isoformat()}.txt")
    report_phase = season_phase_for_date(report_date)
    active_profile = active_weight_profile(model_state, model_date=report_date, phase=report_phase)
    active_weights = active_profile["weights"]
    spring_share = float(_finite(active_profile.get("spring_share", 0.0), 0.0))
    regular_share = float(_finite(active_profile.get("regular_share", 1.0), 1.0))
    margin_sigma = estimate_margin_distribution_sigma(model_state)

    lines: List[str] = []
    lines.append(f"MLB PREDICTIVE REPORT â€” {report_date.isoformat()}")
    lines.append("Source policy: Baseball Savant core data, MLB StatsAPI venue/weather, and Open-Meteo forecast fallback.")
    lines.append("")

    def fmt(x, nd=3):
        try:
            xf = float(x)
            if np.isnan(xf) or np.isinf(xf):
                return "N/A"
            return f"{xf:.{nd}f}"
        except Exception:
            return "N/A"

    def fmt_acc(value: Optional[float]) -> str:
        val = safe_float(value)
        return "N/A" if val is None else f"{val*100:.1f}%"

    def fmt_units(value: Optional[float]) -> str:
        val = safe_float(value)
        return "N/A" if val is None else f"{val:+.2f}u"

    def fmt_pct_delta(value: Optional[float]) -> str:
        val = safe_float(value)
        return "N/A" if val is None else f"{val*100:+.1f} pts"

    def weather_summary_line(comp: Dict[str, object]) -> str:
        weather_source = str(comp.get("weather_source", "neutral") or "neutral").replace("_", " ")
        roof_status = str(comp.get("roof_status", "unknown") or "unknown").replace("_", " ")
        roof_status_source = str(comp.get("roof_status_source", "none") or "none")
        if roof_status_source == "estimated":
            roof_status = f"{roof_status} (estimated)"

        temp_value = safe_float(comp.get("temp_f"))
        temp_label = f"{temp_value:.0f}F" if temp_value is not None and np.isfinite(temp_value) else "N/A"

        wind_mph_value = safe_float(comp.get("wind_mph"))
        wind_label = f"{wind_mph_value:.0f} mph" if wind_mph_value is not None else "N/A"
        wind_direction = str(comp.get("wind_direction", "") or "").strip()
        if wind_direction and wind_direction.lower() != "light/variable":
            wind_label = f"{wind_label} {wind_direction}"

        precip_prob_value = safe_float(comp.get("precip_prob"))
        precip_label = f"{precip_prob_value:.0f}%" if precip_prob_value is not None and np.isfinite(precip_prob_value) else "N/A"

        return (
            f" - Weather/roof context ({weather_source}): {comp.get('condition', 'Unavailable')} | temp {temp_label} | "
            f"wind {wind_label} | precip {precip_label} | roof {roof_status} | "
            f"total adj {comp.get('weather_total_adj', 0.0):+.2f} | side adj {comp.get('weather_home_edge', 0.0):+.3f}"
        )

    if warnings:
        lines.append("DATA GAPS / WARNINGS:")
        for w in warnings:
            lines.append(f" - {w}")
        lines.append("")

    if not games:
        lines.append("No games scheduled per Savant scoreboard-data for this date.")
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return fname

    games_sorted = sorted(games, key=lambda g: (parse_start_time_sort_value(g.start_time_et), g.away, g.home))

    edges = []
    totals = []
    for g in games_sorted:
        winner, p_home, _, _ = predict_game(
            g.away,
            g.home,
            strengths,
            g.away_pitcher,
            g.home_pitcher,
            pitch_comp,
            pen3,
            pen7,
            model_state,
            park_factors=park_factors,
            venue_id=g.venue_id,
            venue_name=g.venue,
            weather_map=weather_map,
            game_pk=g.game_pk,
            away_pitcher_id=g.away_pitcher_id,
            home_pitcher_id=g.home_pitcher_id,
            game_context_map=game_context_map,
            model_date=report_date,
            model_phase=report_phase,
        )
        p = p_home if winner == g.home else (1.0 - p_home)
        edges.append((abs(p - 0.5), p, winner, g.away, g.home))
        totals.append((
            project_total_runs(
                g.away,
                g.home,
                strengths,
                g.away_pitcher,
                g.home_pitcher,
                pitch_comp,
                pen3,
                pen7,
                model_state,
                park_factors,
                g.venue_id,
                g.venue,
                weather_map=weather_map,
                game_pk=g.game_pk,
                away_pitcher_id=g.away_pitcher_id,
                home_pitcher_id=g.home_pitcher_id,
                game_context_map=game_context_map,
                model_date=report_date,
                model_phase=report_phase,
            ),
            g.away,
            g.home,
        ))

    strongest = sorted(edges, reverse=True, key=lambda x: x[0])[0]
    totals_sorted = sorted(totals, key=lambda x: x[0])
    highest_scoring = totals_sorted[-1]
    lowest_scoring = totals_sorted[0]

    lines.append("SUMMARY")
    lines.append(f"Strongest edge: {strongest[3]} @ {strongest[4]} â€” {strongest[2]} (Win% {strongest[1]*100:.1f}%)")
    lines.append(f"Highest projected total: {highest_scoring[1]} @ {highest_scoring[2]} â€” {highest_scoring[0]:.1f} runs")
    lines.append(f"Lowest projected total: {lowest_scoring[1]} @ {lowest_scoring[2]} â€” {lowest_scoring[0]:.1f} runs")
    lines.append("")
    active_guardrail = active_profile.get("guardrail") if isinstance(active_profile.get("guardrail"), dict) else None
    if active_guardrail and active_guardrail.get("apply"):
        lines.append(
            f"Spring guardrail: {str(active_guardrail.get('mode', 'guarded')).replace('_', ' ')} | "
            f"blend {float(_finite(active_guardrail.get('blend', 0.0), 0.0))*100:.0f}% | "
            f"full OOS log loss {fmt(active_guardrail.get('full_oos_log_loss'), 4)} vs simple baseline {fmt(active_guardrail.get('simple_oos_log_loss'), 4)}"
        )
        lines.append(f"Reason: {str(active_guardrail.get('reason', '')).strip()}")
        lines.append("")
    active_earn_back = active_profile.get("earn_back") if isinstance(active_profile.get("earn_back"), dict) else None
    if report_phase == "regular" and active_earn_back:
        enabled_features = active_earn_back.get("enabled_features") or []
        feature_label = ", ".join(str(feature) for feature in enabled_features) if enabled_features else "simple power only"
        lines.append(
            f"Regular earn-back: {str(active_earn_back.get('mode', 'earned')).replace('_', ' ')} | "
            f"enabled [{feature_label}] | "
            f"baseline OOS log loss {fmt(active_earn_back.get('base_oos_log_loss'), 4)} vs selected {fmt(active_earn_back.get('selected_oos_log_loss'), 4)}"
        )
        lines.append(f"Reason: {str(active_earn_back.get('reason', '')).strip()}")
        lines.append("")
    lines.append(
        f"Model regime: {str(active_profile.get('label', report_phase) or report_phase).title()} "
        f"(spring share {spring_share*100:.0f}% | regular share {regular_share*100:.0f}%)"
    )
    if report_phase == "spring":
        lines.append(
            "Late-spring policy: confidence tiers and market flags are intentionally stricter than regular season so the final exhibition slate stays conservative."
        )
    lines.append(
        f"Active weights: off={active_weights['w_off']:.3f} "
        f"starter={active_weights['w_starter']:.3f} "
        f"bullpen={active_weights['w_bullpen']:.3f} "
        f"park={active_weights.get('w_park', 0.03):.3f} "
        f"weather={active_weights.get('w_weather', 0.02):.3f} "
        f"context={active_weights.get('w_context', 0.12):.3f} "
        f"hfa={active_weights['home_field']:.3f}"
    )
    lines.append("")
    active_calibration = active_probability_calibration(model_state, model_date=report_date, phase=report_phase)
    if active_calibration.get("apply"):
        lines.append(
            f"Probability calibration: {str(active_calibration.get('mode', 'identity')).replace('_', ' ')} | "
            f"slope {fmt(active_calibration.get('slope'), 3)} | "
            f"intercept {fmt(active_calibration.get('intercept'), 3)} | "
            f"raw log loss {fmt(active_calibration.get('raw_log_loss'), 4)} -> calibrated {fmt(active_calibration.get('calibrated_log_loss'), 4)}"
        )
    else:
        lines.append(
            f"Probability calibration: identity | {str(active_calibration.get('reason', 'no calibration adjustment active')).strip()}"
        )
    active_margin_cal = active_margin_calibration(model_state, model_date=report_date, phase=report_phase)
    if active_margin_cal.get("apply"):
        lines.append(
            f"Margin calibration: {str(active_margin_cal.get('mode', 'identity')).replace('_', ' ')} | "
            f"slope {fmt(active_margin_cal.get('slope'), 3)} | "
            f"intercept {fmt(active_margin_cal.get('intercept'), 3)} | "
            f"raw RMSE {fmt(active_margin_cal.get('raw_rmse'), 3)} -> calibrated {fmt(active_margin_cal.get('calibrated_rmse'), 3)}"
        )
    else:
        lines.append(
            f"Margin calibration: identity | {str(active_margin_cal.get('reason', 'no margin calibration adjustment active')).strip()}"
        )
    lines.append("")
    backtest_eval = walk_forward_evaluation(load_backtest_log(), phase=report_phase, base_state=model_state)
    lines.append("WALK-FORWARD EVALUATION")
    if backtest_eval.get("available"):
        train_metrics = backtest_eval.get("train_metrics", {})
        validation_metrics = backtest_eval.get("validation_metrics", {})
        test_metrics = backtest_eval.get("test_metrics", {})
        out_of_sample_metrics = backtest_eval.get("out_of_sample_metrics", {})

        def fmt_acc(value: Optional[float]) -> str:
            val = safe_float(value)
            return "N/A" if val is None else f"{val*100:.1f}%"

        lines.append(
            f"{str(backtest_eval.get('phase', report_phase)).title()} splits: "
            f"train {backtest_eval.get('train_span', 'N/A')} ({int(train_metrics.get('games', 0) or 0)} games) | "
            f"validation {backtest_eval.get('validation_span', 'N/A')} ({int(validation_metrics.get('games', 0) or 0)} games) | "
            f"test {backtest_eval.get('test_span', 'N/A')} ({int(test_metrics.get('games', 0) or 0)} games)"
        )
        lines.append(
            f"Train fit: log loss {fmt(train_metrics.get('log_loss'), 4)} | "
            f"Brier {fmt(train_metrics.get('brier'), 4)} | Acc {fmt_acc(train_metrics.get('accuracy'))}"
        )
        lines.append(
            f"Validation (trained on train only): log loss {fmt(validation_metrics.get('log_loss'), 4)} | "
            f"Brier {fmt(validation_metrics.get('brier'), 4)} | Acc {fmt_acc(validation_metrics.get('accuracy'))}"
        )
        lines.append(
            f"Test (trained on train+validation): log loss {fmt(test_metrics.get('log_loss'), 4)} | "
            f"Brier {fmt(test_metrics.get('brier'), 4)} | Acc {fmt_acc(test_metrics.get('accuracy'))}"
        )
        lines.append(
            f"Out-of-sample only: log loss {fmt(out_of_sample_metrics.get('log_loss'), 4)} | "
            f"Brier {fmt(out_of_sample_metrics.get('brier'), 4)} | Acc {fmt_acc(out_of_sample_metrics.get('accuracy'))}"
        )
        train_total_metrics = backtest_eval.get("train_total_metrics", {})
        validation_total_metrics = backtest_eval.get("validation_total_metrics", {})
        test_total_metrics = backtest_eval.get("test_total_metrics", {})
        out_of_sample_total_metrics = backtest_eval.get("out_of_sample_total_metrics", {})
        lines.append(
            f"Train totals: MAE {fmt(train_total_metrics.get('mae'), 3)} | RMSE {fmt(train_total_metrics.get('rmse'), 3)} | "
            f"bias {fmt(train_total_metrics.get('bias'), 3)} | within 1 run {fmt_acc(train_total_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"Validation totals: MAE {fmt(validation_total_metrics.get('mae'), 3)} | RMSE {fmt(validation_total_metrics.get('rmse'), 3)} | "
            f"bias {fmt(validation_total_metrics.get('bias'), 3)} | within 1 run {fmt_acc(validation_total_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"Test totals: MAE {fmt(test_total_metrics.get('mae'), 3)} | RMSE {fmt(test_total_metrics.get('rmse'), 3)} | "
            f"bias {fmt(test_total_metrics.get('bias'), 3)} | within 1 run {fmt_acc(test_total_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"OOS totals: MAE {fmt(out_of_sample_total_metrics.get('mae'), 3)} | RMSE {fmt(out_of_sample_total_metrics.get('rmse'), 3)} | "
            f"bias {fmt(out_of_sample_total_metrics.get('bias'), 3)} | within 1 run {fmt_acc(out_of_sample_total_metrics.get('within_one_run'))}"
        )
        train_margin_metrics = backtest_eval.get("train_margin_metrics", {})
        validation_margin_metrics = backtest_eval.get("validation_margin_metrics", {})
        test_margin_metrics = backtest_eval.get("test_margin_metrics", {})
        out_of_sample_margin_metrics = backtest_eval.get("out_of_sample_margin_metrics", {})
        lines.append(
            f"Train margin: MAE {fmt(train_margin_metrics.get('mae'), 3)} | RMSE {fmt(train_margin_metrics.get('rmse'), 3)} | "
            f"bias {fmt(train_margin_metrics.get('bias'), 3)} | within 1 run {fmt_acc(train_margin_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"Validation margin: MAE {fmt(validation_margin_metrics.get('mae'), 3)} | RMSE {fmt(validation_margin_metrics.get('rmse'), 3)} | "
            f"bias {fmt(validation_margin_metrics.get('bias'), 3)} | within 1 run {fmt_acc(validation_margin_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"Test margin: MAE {fmt(test_margin_metrics.get('mae'), 3)} | RMSE {fmt(test_margin_metrics.get('rmse'), 3)} | "
            f"bias {fmt(test_margin_metrics.get('bias'), 3)} | within 1 run {fmt_acc(test_margin_metrics.get('within_one_run'))}"
        )
        lines.append(
            f"OOS margin: MAE {fmt(out_of_sample_margin_metrics.get('mae'), 3)} | RMSE {fmt(out_of_sample_margin_metrics.get('rmse'), 3)} | "
            f"bias {fmt(out_of_sample_margin_metrics.get('bias'), 3)} | within 1 run {fmt_acc(out_of_sample_margin_metrics.get('within_one_run'))}"
        )
        out_of_sample_run_line_metrics = backtest_eval.get("out_of_sample_run_line_metrics", {})
        lines.append(
            f"OOS run line (-/+1.5): Brier {fmt(out_of_sample_run_line_metrics.get('brier'), 4)} | "
            f"Acc {fmt_acc(out_of_sample_run_line_metrics.get('accuracy'))} | "
            f"model cover {fmt_acc(out_of_sample_run_line_metrics.get('avg_selected_cover_prob'))} | "
            f"actual cover {fmt_acc(out_of_sample_run_line_metrics.get('selected_cover_rate'))}"
        )
    else:
        lines.append(
            f"{report_phase.title()} walk-forward evaluation not available yet: "
            f"{str(backtest_eval.get('reason', 'insufficient backtest data'))}."
        )
    lines.append("")

    simple_power_benchmark = walk_forward_simple_power_benchmark(load_backtest_log(), phase=report_phase)
    market_benchmark = market_benchmark_summary(load_backtest_log(), phase=report_phase)
    lines.append("BENCHMARKS")
    if simple_power_benchmark.get("available"):
        sp_oos = simple_power_benchmark.get("out_of_sample_metrics", {})
        lines.append(
            f"Simple power baseline (x_off + home field) OOS: log loss {fmt(sp_oos.get('log_loss'), 4)} | "
            f"Brier {fmt(sp_oos.get('brier'), 4)} | Acc {fmt_acc(sp_oos.get('accuracy'))}"
        )
    else:
        lines.append(
            f"Simple power baseline unavailable: {str(simple_power_benchmark.get('reason', 'insufficient backtest data'))}."
        )

    if market_benchmark.get("available"):
        market_metrics = market_benchmark.get("closing_implied_metrics", {})
        lines.append(
            f"Closing implied market baseline OOS: log loss {fmt(market_metrics.get('log_loss'), 4)} | "
            f"Brier {fmt(market_metrics.get('brier'), 4)} | Acc {fmt_acc(market_metrics.get('accuracy'))}"
        )
        lines.append(
            f"Market favorite benchmark OOS: Acc {fmt_acc(market_benchmark.get('market_favorite_accuracy'))} | "
            f"Games {int(market_benchmark.get('games', 0) or 0)} | Span {str(market_benchmark.get('out_of_sample_span', 'N/A'))}"
        )
    else:
        lines.append(
            f"Closing implied market baseline unavailable: {str(market_benchmark.get('reason', 'no archived sportsbook snapshots overlap the backtest log'))}."
        )
        lines.append(
            f"Market favorite benchmark unavailable: {str(market_benchmark.get('reason', 'no archived sportsbook snapshots overlap the backtest log'))}."
        )
    lines.append("")
    prediction_scoreboard = prediction_scoreboard_summary(load_backtest_log(), phase=report_phase, base_state=model_state)
    betting_scoreboard = betting_scoreboard_summary(phase=report_phase)
    lines.append("ENGINE SCOREBOARD")
    if prediction_scoreboard.get("available"):
        pred_pick = prediction_scoreboard.get("pick_metrics", {})
        lines.append(
            f"Prediction OOS (predicted side): {int(pred_pick.get('games', 0) or 0)} games | "
            f"hit rate {fmt_acc(pred_pick.get('win_rate'))} | "
            f"avg confidence {fmt_acc(pred_pick.get('avg_confidence'))} | "
            f"log loss {fmt(pred_pick.get('log_loss'), 4)}"
        )
        calibration_rows = prediction_scoreboard.get("calibration", []) or []
        if calibration_rows:
            lines.append("Calibration buckets (predicted side):")
            for bucket in calibration_rows[:6]:
                lines.append(
                    f" - {bucket.get('label')}: model {fmt_acc(bucket.get('avg_confidence'))} | "
                    f"actual {fmt_acc(bucket.get('actual_rate'))} | "
                    f"gap {fmt_pct_delta(bucket.get('gap'))} | "
                    f"games {int(bucket.get('games', 0) or 0)}"
                )
        total_metrics = prediction_scoreboard.get("total_metrics", {})
        if int(total_metrics.get("games", 0) or 0) > 0:
            lines.append(
                f"Totals OOS: {int(total_metrics.get('games', 0) or 0)} games | "
                f"MAE {fmt(total_metrics.get('mae'), 3)} | "
                f"RMSE {fmt(total_metrics.get('rmse'), 3)} | "
                f"model avg {fmt(total_metrics.get('avg_projected_total'), 2)} | "
                f"actual avg {fmt(total_metrics.get('avg_actual_total'), 2)} | "
                f"bias {fmt(total_metrics.get('bias'), 3)} | "
                f"within 1 run {fmt_acc(total_metrics.get('within_one_run'))}"
            )
            total_calibration_rows = prediction_scoreboard.get("total_calibration", []) or []
            if total_calibration_rows:
                lines.append("Totals calibration buckets:")
                for bucket in total_calibration_rows[:5]:
                    lines.append(
                        f" - {bucket.get('label')}: model {fmt(bucket.get('avg_projected_total'), 2)} | "
                        f"actual {fmt(bucket.get('avg_actual_total'), 2)} | "
                        f"gap {fmt(bucket.get('gap'), 3)} | "
                        f"games {int(bucket.get('games', 0) or 0)}"
                    )
        margin_metrics = prediction_scoreboard.get("margin_metrics", {})
        if int(margin_metrics.get("games", 0) or 0) > 0:
            lines.append(
                f"Margin OOS: {int(margin_metrics.get('games', 0) or 0)} games | "
                f"MAE {fmt(margin_metrics.get('mae'), 3)} | "
                f"RMSE {fmt(margin_metrics.get('rmse'), 3)} | "
                f"model avg {fmt(margin_metrics.get('avg_projected_margin'), 2)} | "
                f"actual avg {fmt(margin_metrics.get('avg_actual_margin'), 2)} | "
                f"bias {fmt(margin_metrics.get('bias'), 3)} | "
                f"within 1 run {fmt_acc(margin_metrics.get('within_one_run'))}"
            )
        run_line_metrics = prediction_scoreboard.get("run_line_metrics", {})
        if int(run_line_metrics.get("games", 0) or 0) > 0:
            lines.append(
                f"Run line OOS (-/+1.5): {int(run_line_metrics.get('games', 0) or 0)} games | "
                f"Brier {fmt(run_line_metrics.get('brier'), 4)} | "
                f"Acc {fmt_acc(run_line_metrics.get('accuracy'))} | "
                f"model cover {fmt_acc(run_line_metrics.get('avg_selected_cover_prob'))} | "
                f"actual cover {fmt_acc(run_line_metrics.get('selected_cover_rate'))}"
            )
            run_line_calibration_rows = prediction_scoreboard.get("run_line_calibration", []) or []
            if run_line_calibration_rows:
                lines.append("Run-line calibration buckets:")
                for bucket in run_line_calibration_rows[:4]:
                    lines.append(
                        f" - {bucket.get('label')}: model {fmt_acc(bucket.get('avg_selected_prob'))} | "
                        f"actual {fmt_acc(bucket.get('actual_cover_rate'))} | "
                        f"gap {fmt_pct_delta(bucket.get('gap'))} | "
                        f"games {int(bucket.get('games', 0) or 0)}"
                    )
    else:
        lines.append(
            f"Prediction scoreboard unavailable: {str(prediction_scoreboard.get('reason', 'insufficient out-of-sample data'))}."
        )

    if int(betting_scoreboard.get("candidates", 0) or 0) > 0:
        lines.append(
            f"Betting ledger: candidates {int(betting_scoreboard.get('candidates', 0) or 0)} | "
            f"open {int(betting_scoreboard.get('open_watch', 0) or 0)} | "
            f"settled {int(betting_scoreboard.get('settled', 0) or 0)}"
        )
        if int(betting_scoreboard.get("settled", 0) or 0) > 0:
            lines.append(
                f"Settled results: {int(betting_scoreboard.get('wins', 0) or 0)}-"
                f"{int(betting_scoreboard.get('losses', 0) or 0)}-"
                f"{int(betting_scoreboard.get('pushes', 0) or 0)} | "
                f"profit {fmt_units(betting_scoreboard.get('profit_units'))} | "
                f"ROI {fmt_acc(betting_scoreboard.get('roi'))} | "
                f"yield/bet {fmt_units(betting_scoreboard.get('yield_per_bet'))} | "
                f"avg CLV {fmt(betting_scoreboard.get('avg_clv'), 4)} | "
                f"positive CLV {fmt_acc(betting_scoreboard.get('positive_clv_rate'))} | "
                f"max drawdown {fmt_units(betting_scoreboard.get('max_drawdown'))}"
            )
        else:
            lines.append(
                f"Betting ROI/CLV pending: {str(betting_scoreboard.get('reason', 'market candidates exist, but none have settled yet'))}."
            )
    else:
        lines.append("Betting ledger unavailable: no market journal rows yet.")
    lines.append("")

    lines.append("SEGMENTATION")
    if prediction_scoreboard.get("available"):
        prediction_conf_segments = prediction_scoreboard.get("confidence_segments", []) or []
        prediction_side_segments = prediction_scoreboard.get("side_segments", []) or []
        if prediction_conf_segments:
            lines.append("Prediction OOS by confidence tier:")
            for seg in prediction_conf_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('games', 0) or 0)} games | "
                    f"hit {fmt_acc(seg.get('win_rate'))} | "
                    f"avg conf {fmt_acc(seg.get('avg_confidence'))} | "
                    f"log loss {fmt(seg.get('log_loss'), 4)}"
                )
        if prediction_side_segments:
            lines.append("Prediction OOS by pick side:")
            for seg in prediction_side_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('games', 0) or 0)} games | "
                    f"hit {fmt_acc(seg.get('win_rate'))} | "
                    f"avg conf {fmt_acc(seg.get('avg_confidence'))} | "
                    f"log loss {fmt(seg.get('log_loss'), 4)}"
                )
        total_segments = prediction_scoreboard.get("total_segments", []) or []
        if total_segments:
            lines.append("Totals OOS by projected-total band:")
            for seg in total_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('games', 0) or 0)} games | "
                    f"MAE {fmt(seg.get('mae'), 3)} | "
                    f"RMSE {fmt(seg.get('rmse'), 3)} | "
                    f"bias {fmt(seg.get('bias'), 3)}"
                )
        margin_segments = prediction_scoreboard.get("margin_segments", []) or []
        if margin_segments:
            lines.append("Run-line OOS by projected-margin band:")
            for seg in margin_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('games', 0) or 0)} games | "
                    f"MAE {fmt(seg.get('mae'), 3)} | "
                    f"RMSE {fmt(seg.get('rmse'), 3)} | "
                    f"run-line acc {fmt_acc(seg.get('run_line_accuracy'))}"
                )
        favorite_side_segments = prediction_scoreboard.get("favorite_side_segments", []) or []
        if favorite_side_segments:
            lines.append("Run-line OOS by projected favorite side:")
            for seg in favorite_side_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('games', 0) or 0)} games | "
                    f"MAE {fmt(seg.get('mae'), 3)} | "
                    f"RMSE {fmt(seg.get('rmse'), 3)} | "
                    f"run-line acc {fmt_acc(seg.get('run_line_accuracy'))}"
                )
    else:
        lines.append(
            f"Prediction segments unavailable: {str(prediction_scoreboard.get('reason', 'insufficient out-of-sample data'))}."
        )

    if int(betting_scoreboard.get("settled", 0) or 0) > 0:
        market_segments = betting_scoreboard.get("market_segments", []) or []
        confidence_segments = betting_scoreboard.get("confidence_segments", []) or []
        actionability_segments = betting_scoreboard.get("actionability_segments", []) or []
        role_segments = betting_scoreboard.get("role_segments", []) or []
        sportsbook_segments = betting_scoreboard.get("sportsbook_segments", []) or []
        edge_segments = betting_scoreboard.get("edge_segments", []) or []
        capture_segments = betting_scoreboard.get("capture_segments", []) or []
        if market_segments:
            lines.append("Betting by market type:")
            for seg in market_segments:
                label = str(seg.get('label', '')).replace('_', ' ').title()
                lines.append(
                    f" - {label}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if confidence_segments:
            lines.append("Betting by confidence tier:")
            for seg in confidence_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if actionability_segments:
            lines.append("Betting by actionability tier:")
            for seg in actionability_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if role_segments:
            lines.append("Betting side markets by role:")
            for seg in role_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if sportsbook_segments:
            lines.append("Betting by sportsbook:")
            for seg in sportsbook_segments[:6]:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if edge_segments:
            lines.append("Betting by model edge tier:")
            for seg in edge_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
        if capture_segments:
            lines.append("Betting by capture window:")
            for seg in capture_segments:
                lines.append(
                    f" - {str(seg.get('label'))}: {int(seg.get('bets', 0) or 0)} bets | "
                    f"profit {fmt_units(seg.get('profit_units'))} | "
                    f"ROI {fmt_acc(seg.get('roi'))} | "
                    f"avg CLV {fmt(seg.get('avg_clv'), 4)}"
                )
    elif int(betting_scoreboard.get("candidates", 0) or 0) > 0:
        lines.append(
            f"Betting segments pending: {str(betting_scoreboard.get('reason', 'market candidates exist, but none have settled yet'))}."
        )
    else:
        lines.append("Betting segments unavailable: no market journal rows yet.")
    lines.append("")
    sidx = strengths.set_index("team") if not strengths.empty and "team" in strengths.columns else None
    pitch_score_map: Dict[str, float] = {}
    pitch_source_map: Dict[str, str] = {}
    if pitch_comp is not None and not pitch_comp.empty:
        for _, row in pitch_comp.iterrows():
            key = canonical_player_name(row.get("pitcher"))
            if not key:
                continue
            pitch_score_map[key] = _finite(row.get("pitcher_score", 0.0), 0.0)
            pitch_source_map[key] = str(row.get("pitcher_source", "") or "")

    def starter_line(side: str, pitcher_name: Optional[str], score: Optional[float], source: Optional[str]) -> str:
        if not pitcher_name:
            return f" - {side} starter composite: TBD"
        display_source = starter_display_source(source)
        label = f"{pitcher_name}, {display_source}" if display_source else pitcher_name
        if score is None:
            return f" - {side} starter composite ({label}): MISSING"
        return f" - {side} starter composite ({label}): {score:+.2f}"

    for g in games_sorted:
        winner, p_home, spread, comp = predict_game(
            g.away,
            g.home,
            strengths,
            g.away_pitcher,
            g.home_pitcher,
            pitch_comp,
            pen3,
            pen7,
            model_state,
            park_factors=park_factors,
            venue_id=g.venue_id,
            venue_name=g.venue,
            weather_map=weather_map,
            game_pk=g.game_pk,
            away_pitcher_id=g.away_pitcher_id,
            home_pitcher_id=g.home_pitcher_id,
            game_context_map=game_context_map,
            model_date=report_date,
            model_phase=report_phase,
        )

        p_win = p_home if winner == g.home else (1.0 - p_home)
        tier = confidence_tier(p_win, report_phase)
        total = project_total_runs(
            g.away,
            g.home,
            strengths,
            g.away_pitcher,
            g.home_pitcher,
            pitch_comp,
            pen3,
            pen7,
            model_state,
            park_factors,
            g.venue_id,
            g.venue,
            weather_map=weather_map,
            game_pk=g.game_pk,
            away_pitcher_id=g.away_pitcher_id,
            home_pitcher_id=g.home_pitcher_id,
            game_context_map=game_context_map,
            model_date=report_date,
            model_phase=report_phase,
            game_components=comp,
        )

        away_key = canonical_player_name(g.away_pitcher or "")
        home_key = canonical_player_name(g.home_pitcher or "")
        away_source = str(comp.get("away_starter_source") or pitch_source_map.get(away_key, ""))
        home_source = str(comp.get("home_starter_source") or pitch_source_map.get(home_key, ""))
        away_score = comp.get("away_starter") if g.away_pitcher else None
        home_score = comp.get("home_starter") if g.home_pitcher else None

        lines.append("=" * 78)
        lines.append(f"{g.away} @ {g.home} â€” {g.venue} â€” {g.start_time_et}")
        lines.append(f"Probables: {g.away_pitcher or 'TBD'} (AWAY) vs {g.home_pitcher or 'TBD'} (HOME)")
        lines.append("")
        lines.append(f"Predicted winner: {winner}")
        lines.append(f"Win probability: {p_win*100:.1f}%  |  Confidence tier: {tier}")
        lines.append(f"Fair moneyline: {g.away} {fair_moneyline_from_probability(1.0 - p_home):+d} | {g.home} {fair_moneyline_from_probability(p_home):+d}")
        bettable_total = round_to_half(total)
        run_line = fair_run_line_prices(g.away, g.home, spread, margin_sigma)
        lines.append(f"Projected margin: {projected_margin_text(g.home, g.away, spread)}")
        lines.append(f"Model expected total: {total:.1f}")
        if np.isfinite(_finite(comp.get("away_expected_runs"), np.nan)) and np.isfinite(_finite(comp.get("home_expected_runs"), np.nan)):
            lines.append(
                f"Projected team runs: {g.away} {float(comp.get('away_expected_runs', 0.0)):.1f} | "
                f"{g.home} {float(comp.get('home_expected_runs', 0.0)):.1f}"
            )
        lines.append(f"Fair betting total: {bettable_total:.1f}")
        lines.append(
            f"Fair run line: {run_line['favorite_team']} -1.5 {run_line['favorite_price']:+d} | "
            f"{run_line['underdog_team']} +1.5 {run_line['underdog_price']:+d}"
        )
        market_comp = market_comparison_for_game(g, p_home, spread, total, market_map=market_map, margin_sigma=margin_sigma)
        if market_comp.get("available"):
            market_books = int(_finite(market_comp.get("sportsbook_count", 0), 0))
            lines.append(f"Sportsbook comparison: {market_comp.get('market_label', 'Market')}")
            ml_comp = market_comp.get("moneyline", {})
            if ml_comp.get("available"):
                ml_action = evaluate_market_actionability(
                    "moneyline",
                    float(_finite(ml_comp.get("best_edge", 0.0), 0.0)),
                    market_books,
                    comp,
                    model_phase=report_phase,
                    model_probability=safe_float(ml_comp.get("model_probability")),
                )
                lines.append(
                    f" - Market moneyline: {g.away} {ml_comp.get('away_ml'):+d} | {g.home} {ml_comp.get('home_ml'):+d} | "
                    f"best edge {ml_comp.get('best_side')} {ml_comp.get('best_edge', 0.0) * 100:+.1f} pts"
                )
                lines.append(f" - Moneyline actionability: {actionability_summary_text(ml_action)}")
            rl_comp = market_comp.get("run_line", {})
            if rl_comp.get("available"):
                rl_action = evaluate_market_actionability(
                    "run_line",
                    float(_finite(rl_comp.get("best_edge", 0.0), 0.0)),
                    market_books,
                    comp,
                    model_phase=report_phase,
                    model_probability=safe_float(rl_comp.get("model_probability")),
                )
                lines.append(
                    f" - Market run line: {g.home} {rl_comp.get('home_line', 0.0):+.1f} {rl_comp.get('home_price', 0):+d} | "
                    f"{g.away} {rl_comp.get('away_line', 0.0):+.1f} {rl_comp.get('away_price', 0):+d} | "
                    f"best edge {rl_comp.get('best_selection')} {rl_comp.get('best_edge', 0.0) * 100:+.1f} pts"
                )
                lines.append(f" - Run-line actionability: {actionability_summary_text(rl_action)}")
            total_comp = market_comp.get("total", {})
            if total_comp.get("available"):
                total_action = evaluate_market_actionability(
                    "total",
                    float(_finite(total_comp.get("diff", 0.0), 0.0)),
                    market_books,
                    comp,
                    model_phase=report_phase,
                    model_probability=None,
                )
                total_price_text = ""
                if total_comp.get("over_price") is not None and total_comp.get("under_price") is not None:
                    total_price_text = f" (O {int(total_comp.get('over_price')):+d} / U {int(total_comp.get('under_price')):+d})"
                lines.append(
                    f" - Market total: {float(total_comp.get('market_total', 0.0)):.1f}{total_price_text} | "
                    f"lean {total_comp.get('lean', 'Flat')} by {float(total_comp.get('diff', 0.0)):+.1f} runs"
                )
                lines.append(f" - Total actionability: {actionability_summary_text(total_action)}")
        lines.append("")
        lines.append("")

        lines.append("Savant-derived offense indicators (team-level from Statcast Search CSV; xwOBA falls back to wOBA when Savant leaves xwOBA blank):")
        roof_raw = str(comp.get("roof_type", "unknown") or "unknown")
        roof_label = {"open_air": "open-air", "retractable": "retractable roof", "dome": "dome"}.get(roof_raw, roof_raw.replace("_", " "))
        surface_label = str(comp.get("surface_type", "unknown") or "unknown")
        elevation_value = safe_float(comp.get("elevation_ft"))
        elevation_label = f"{int(round(elevation_value))} ft" if elevation_value is not None and np.isfinite(elevation_value) else "unknown elevation"

        lines.append("Pitching, bullpen, park, and weather layer:")
        lines.append(starter_line("Away", g.away_pitcher, away_score, away_source))
        lines.append(starter_line("Home", g.home_pitcher, home_score, home_source))
        lines.append(f" - Away bullpen score (reliever quality + recent workload from gf boxscores): {comp.get('away_bullpen', 0.0):+.2f}")
        lines.append(f" - Home bullpen score (reliever quality + recent workload from gf boxscores): {comp.get('home_bullpen', 0.0):+.2f}")
        lines.append(f" - Park context ({g.venue}): total adj {comp.get('park_total_adj', 0.0):+.2f} | home-edge adj {comp.get('park_home_edge', 0.0):+.3f} | sample games {int(comp.get('park_games', 0.0))}")
        lines.append(f" - Shared run-environment boost: {comp.get('shared_run_env_boost', 0.0):+.2f}")
        if _finite(comp.get("park_profile_found", 0.0), 0.0) > 0:
            lines.append(f" - Static park traits: {elevation_label} | {roof_label} | {surface_label} surface | run prior {comp.get('park_static_total_adj', 0.0):+.2f} | familiarity {comp.get('park_static_home_edge', 0.0):+.3f}")
        lines.append(f" - Historical venue signal (Savant finals): run adj {comp.get('park_hist_total_adj', 0.0):+.2f} | home-edge adj {comp.get('park_hist_home_edge', 0.0):+.3f} | weighted sample {comp.get('park_weighted_games', 0.0):.1f}")
        lines.append(weather_summary_line(comp))
        away_lineup_label = "Yes" if comp.get("away_lineup_confirmed", 0.0) >= 0.5 else "No"
        home_lineup_label = "Yes" if comp.get("home_lineup_confirmed", 0.0) >= 0.5 else "No"
        lines.append("Matchup context layer:")
        lines.append(
            f" - Confirmed lineups: {g.away} {away_lineup_label} ({int(comp.get('away_lineup_size', 0.0))} hitters) | "
            f"{g.home} {home_lineup_label} ({int(comp.get('home_lineup_size', 0.0))} hitters)"
        )
        lines.append(
            f" - Lineup delta vs team baseline: {g.away} {comp.get('away_lineup_delta', 0.0):+.2f} | "
            f"{g.home} {comp.get('home_lineup_delta', 0.0):+.2f}"
        )
        lines.append(
            f" - Prior-season lineup fills used: {g.away} {int(comp.get('away_lineup_prior_scores_used', 0.0))} | "
            f"{g.home} {int(comp.get('home_lineup_prior_scores_used', 0.0))}"
        )
        lines.append(
            f" - Baserunning / defense / framing: {g.away} {comp.get('away_baserunning', 0.0):+.2f} / {comp.get('away_defense', 0.0):+.2f} / {comp.get('away_catcher_framing', 0.0):+.2f} | "
            f"{g.home} {comp.get('home_baserunning', 0.0):+.2f} / {comp.get('home_defense', 0.0):+.2f} / {comp.get('home_catcher_framing', 0.0):+.2f}"
        )
        lines.append(
            f" - Travel/rest: {g.away} {comp.get('away_rest_days', 0.0):.0f}d rest, {comp.get('away_travel_miles', 0.0):.0f} mi" +
            (f" from {comp.get('away_previous_venue')}" if comp.get('away_previous_venue') else "") +
            f" | {g.home} {comp.get('home_rest_days', 0.0):.0f}d rest, {comp.get('home_travel_miles', 0.0):.0f} mi" +
            (f" from {comp.get('home_previous_venue')}" if comp.get('home_previous_venue') else "")
        )
        lines.append(
            f" - Handedness matchup: {g.away} offense vs {comp.get('home_starter_hand', '') or 'unknown'}HP | "
            f"{g.home} offense vs {comp.get('away_starter_hand', '') or 'unknown'}HP"
        )
        lines.append("")
        for team in [g.away, g.home]:
            if sidx is not None and team in sidx.index:
                row = sidx.loc[team]
                off = row["offense_score"] if "offense_score" in row else np.nan
                xw = row["xwoba"] if "xwoba" in row else np.nan
                xslg = row["xslg"] if "xslg" in row else np.nan
                xba = row["xba"] if "xba" in row else np.nan
                ev = row["avg_ev"] if "avg_ev" in row else np.nan
                hh = row["hardhit_rate"] if "hardhit_rate" in row else np.nan
                hh_pct = (float(hh) * 100.0) if (hh is not None and np.isfinite(hh) and float(hh) <= 1.5) else hh
                prior_weight = safe_float(row.get("offense_prior_weight")) if "offense_prior_weight" in row else None
                prior_tag = f" | prior wt {prior_weight*100:.0f}%" if prior_weight is not None and np.isfinite(prior_weight) and prior_weight >= 0.05 else ""
                lines.append(
                    f" - {team}: OffScore {float(off):+.2f}{prior_tag} | xwOBA/wOBA {fmt(xw)} | xSLG {fmt(xslg)} | "
                    f"xBA {fmt(xba)} | EV {fmt(ev, 1)} | HH% {fmt(hh_pct, 1)}%"
                )
            else:
                lines.append(f" - {team}: MISSING in aggregation (Savant statcast pull returned no rows for this team/date window).")

        lines.append("")
        lines.append("MODEL NOTES:")
        lines.append(" - Spring training and regular season now use separate weight profiles, with a short carryover bridge at the start of the regular season.")
        lines.append(" - Team offense scores are shrunk toward prior regular-season offense early in the year until current plate appearances stabilize.")
        lines.append(" - Confirmed lineups can borrow prior-season player Statcast quality when current-season batter samples are still too thin.")
        lines.append(" - Starter priors still come from prior Savant seasons when current-season MLB data is thin or missing.")
        lines.append(" - Projected margin stays continuous inside the model; moneyline, run line, and total-line outputs are reporting translations for sportsbook comparison.")
        lines.append(" - Fair run-line prices use a calibrated historical margin-variance estimate from the stored backtest log.")
        lines.append(" - Park context now blends Savant venue history with static traits like elevation, roof type, and playing surface.")
        lines.append(" - Weather context uses MLB StatsAPI actual game weather when posted and Open-Meteo forecast otherwise.")
        lines.append(" - Park and weather effects are weighted more heavily into totals than side probabilities; side impact stays small and proportional.")
        lines.append(" - Retractable-roof open or closed state is estimated only when MLB has not yet posted an explicit status.")
        lines.append("")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return fname

def write_excel_season_workbook(
    report_date: dt.date,
    strengths: pd.DataFrame,
    pitch_agg: pd.DataFrame,
    pitch_comp: pd.DataFrame,
    bat_agg: pd.DataFrame,
    bat_comp: pd.DataFrame
) -> str:
    """
    Overwrites MLB_Season_To_Date_2026.xlsx with:
      Sheet1: Team metrics table (Savant-only derived)
      Sheet2: Top 20 batters (Savant-only composite)
      Sheet3: Top 20 pitchers (Savant-only composite)
      Sheet4: Power rankings (quant from offense_score baseline)
    """
    path = os.path.join(OUT_DIR, "MLB_Season_To_Date_2026.xlsx")

    # Power rankings
    power = strengths.sort_values("offense_score", ascending=False).reset_index(drop=True)
    power["rank"] = np.arange(1, len(power) + 1)

    # Top 20 batters
    if bat_agg is None or bat_agg.empty or bat_comp is None or bat_comp.empty:
        top_batters = pd.DataFrame({"note": ["Batter composite empty (no Savant batter rows returned for filters)."]})
    else:
        top_batters = (
            bat_agg.merge(bat_comp, on="batter", how="left")
            .sort_values("batter_score", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )

    # Top 20 pitchers
    # Top 20 pitchers (robust to merge suffixes / missing score)
    if pitch_agg is None or pitch_agg.empty:
        top_pitchers = pd.DataFrame({"note": ["Pitcher composite empty (no Savant pitcher rows returned for filters)."]})
    else:
    # If pitch_agg already contains pitcher_score, prefer it directly
        if "pitcher_score" in pitch_agg.columns:
            dfp = pitch_agg.copy()
            score_col = "pitcher_score"
        else:
            dfp = pitch_agg.copy()
            if pitch_comp is not None and not pitch_comp.empty:
                dfp = dfp.merge(pitch_comp, on="pitcher", how="left")

            if "pitcher_score" in dfp.columns:
                score_col = "pitcher_score"
            elif "pitcher_score_y" in dfp.columns:
                score_col = "pitcher_score_y"
            elif "pitcher_score_x" in dfp.columns:
                score_col = "pitcher_score_x"
            else:
                score_col = None

        if score_col is None:
             top_pitchers = pd.DataFrame({"note": ["Pitcher score unavailable (no pitcher_score column produced)."]})
        else:
            top_pitchers = (
                dfp.sort_values(score_col, ascending=False)
                   .head(20)
                   .reset_index(drop=True)
            )

    # Only write note if thereâ€™s at least one row
    if not top_pitchers.empty:
        if "note" not in top_pitchers.columns:
            top_pitchers["note"] = ""
        top_pitchers.loc[0, "note"] = "Pitcher composite is Savant-only from Statcast Search CSV."

    # OneDrive/Excel can lock the workbook -> write fallback filename so the run completes
    try:
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
            strengths.to_excel(writer, sheet_name="Team_Metrics", index=False)
            top_batters.to_excel(writer, sheet_name="Top_20_Batters", index=False)
            top_pitchers.to_excel(writer, sheet_name="Top_20_Pitchers", index=False)
            power[["rank", "team", "offense_score", "xwoba", "xslg", "xba", "avg_ev", "hardhit_rate", "pa"]].to_excel(
                writer, sheet_name="Power_Rankings", index=False
            )
        return path
    except PermissionError:
        alt = os.path.join(OUT_DIR, f"MLB_Season_To_Date_2026_LOCKED_{report_date.isoformat()}.xlsx")
        with pd.ExcelWriter(alt, engine="openpyxl", mode="w") as writer:
            strengths.to_excel(writer, sheet_name="Team_Metrics", index=False)
            top_batters.to_excel(writer, sheet_name="Top_20_Batters", index=False)
            top_pitchers.to_excel(writer, sheet_name="Top_20_Pitchers", index=False)
            power[["rank", "team", "offense_score", "xwoba", "xslg", "xba", "avg_ev", "hardhit_rate", "pa"]].to_excel(
                writer, sheet_name="Power_Rankings", index=False
            )
        return alt

# -----------------------------
# MAIN
# -----------------------------
def main():
    pitch_comp = pd.DataFrame(columns=["pitcher", "pitcher_id", "pitcher_source", "pitcher_score", "pitcher_score_vs_l", "pitcher_score_vs_r", "pitcher_hand"])
    ensure_out_dir()
    report_date = today_local()
    warnings: List[str] = []

    sb_today = savant_scoreboard(report_date)
    probable_candidates_today = probable_starter_candidates_from_scoreboard(sb_today)
    games, sb_warn = games_from_scoreboard(sb_today)
    warnings.extend(sb_warn)
    schedule_probables_today: Dict[int, Dict[str, object]] = {}
    try:
        schedule_probables_today = probables_from_statsapi_schedule(report_date)
    except Exception as e:
        warnings.append(f"Official MLB schedule probable starters unavailable for {report_date.isoformat()}: {e}")
    if schedule_probables_today:
        probable_candidates_today.extend(starter_candidates_from_probables_map(schedule_probables_today))
        games, schedule_warn = fill_missing_probables_from_map(games, schedule_probables_today, "official MLB schedule")
        warnings.extend(schedule_warn)
    games, gf_warn = enrich_probables_from_gf(games)
    warnings.extend(gf_warn)
    report_phase = season_phase_for_date(report_date)
    market_import_info, market_import_warn = import_all_market_lines(report_date, games)
    warnings.extend(market_import_warn)
    fetched_market_rows = 0
    if report_phase == "spring" and str(os.getenv("ODDS_API_ALLOW_SPRING_FETCH") or "").strip() not in ["1", "true", "yes", "on"]:
        warnings.append(f"Live sportsbook fetch skipped for spring schedule; use {os.path.basename(str(market_import_info.get('template_path', '')))} for manual lines.")
    else:
        fetched_market_rows, market_fetch_warn = fetch_market_lines_from_odds_api(report_date, games)
        warnings.extend(market_fetch_warn)
        if fetched_market_rows:
            warnings.append(f"Live sportsbook fetch: loaded {fetched_market_rows} row(s) from Odds API.")
    market_map, market_warn = load_market_consensus_for_date(report_date, games)
    warnings.extend(market_warn)
    if int(market_import_info.get("manual_csv", 0)):
        warnings.append(f"Sportsbook market import: loaded {int(market_import_info.get('manual_csv', 0))} row(s) from {os.path.basename(SPORTSBOOK_CSV_PATH)}.")
    if int(market_import_info.get("manual_template", 0)):
        warnings.append(f"Manual sportsbook import: loaded {int(market_import_info.get('manual_template', 0))} row(s) from {os.path.basename(str(market_import_info.get('template_path', '')))}.")
    game_types = game_types_for_phase(report_phase)
    model_state = ensure_totals_refresh_state(ensure_backtest_refresh_state(load_model_state()))
    save_model_state(model_state)
    if backtest_log_schema_missing_context():
        warnings.append("Backtest log has legacy rows; historical dates will be refreshed incrementally over future runs.")

    skip_daily_backtest = model_state.get("last_update_date") == report_date.isoformat()
    yday = report_date - dt.timedelta(days=1)

    if skip_daily_backtest:
        warnings.append(f"Backtest skipped: model_state already updated for {report_date.isoformat()}.")
    else:
        backtest_rows, backtest_warn = build_backtest_rows_for_date(yday, model_state)
        warnings.extend(backtest_warn)
        if backtest_rows:
            append_backtest_log(backtest_rows)
            model_state = mark_backtest_refresh_dates(model_state, [yday])
            model_state = mark_totals_refresh_dates(model_state, [yday])
            model_state = recalibrate_model_weights_from_log(model_state)
            rewrite_backtest_predictions_with_model_state(model_state)
            save_model_state(model_state)
        else:
            warnings.append(f"Backtest finals unavailable for {yday.isoformat()}; model state not updated.")

    model_state, _, history_refresh_warn, history_refresh_processed, history_refresh_remaining = refresh_backtest_history_incremental(
        model_state,
        max_dates=BACKTEST_HISTORY_REFRESH_BATCH_DATES,
    )
    warnings.extend(history_refresh_warn)
    if history_refresh_processed:
        warnings.append(
            f"Backtest history refresh: rebuilt {len(history_refresh_processed)} prior date(s) "
            f"({', '.join(history_refresh_processed)}); {len(history_refresh_remaining)} remain."
        )
    elif history_refresh_remaining:
        warnings.append(
            f"Backtest history refresh pending: {len(history_refresh_remaining)} prior date(s) remain; "
            f"next runs will continue incrementally."
        )

    model_state, _, totals_refresh_warn, totals_refresh_processed, totals_refresh_remaining = refresh_backtest_totals_incremental(
        model_state,
        max_dates=BACKTEST_HISTORY_REFRESH_BATCH_DATES,
    )
    warnings.extend(totals_refresh_warn)
    if totals_refresh_processed:
        warnings.append(
            f"Backtest totals refresh: refreshed {len(totals_refresh_processed)} prior date(s) "
            f"({', '.join(totals_refresh_processed)}); {len(totals_refresh_remaining)} remain."
        )
    elif totals_refresh_remaining:
        warnings.append(
            f"Backtest totals refresh pending: {len(totals_refresh_remaining)} prior date(s) remain; "
            f"next runs will continue incrementally."
        )

    season_start = dt.date(report_date.year, 1, 1)

    prior_offense = pd.DataFrame()
    try:
        prior_offense = previous_regular_season_team_offense(report_date.year - 1)
    except Exception as e:
        warnings.append(f"Prior-season team offense unavailable for {report_date.isoformat()}: {e}")

    season_raw = statcast_team_battedball_snapshot(season_start, pregame_statcast_end_date(report_date), game_types, season_year=report_date.year)
    if season_raw.empty:
        warnings.append("Season-to-date team offense pull returned empty (filters/date/game type may exclude games).")
    d30_raw = statcast_team_battedball_snapshot(report_date - dt.timedelta(days=30), pregame_statcast_end_date(report_date), game_types, season_year=report_date.year)
    d14_raw = statcast_team_battedball_snapshot(report_date - dt.timedelta(days=14), pregame_statcast_end_date(report_date), game_types, season_year=report_date.year)
    d7_raw = statcast_team_battedball_snapshot(report_date - dt.timedelta(days=7), pregame_statcast_end_date(report_date), game_types, season_year=report_date.year)
    strengths = build_strengths_bundle_from_snapshots(
        season_raw,
        d30_raw,
        d14_raw,
        d7_raw,
        prior_agg=prior_offense,
    )

    p_df = statcast_pitching_snapshot(season_start, pregame_statcast_end_date(report_date), game_types=game_types)
    if p_df.empty:
        warnings.append("Pitcher Statcast Search CSV returned empty (filters/date/game type may exclude games).")

    pitch_agg = aggregate_pitcher_quality(p_df)
    pitch_comp = build_pitcher_matchup_table(p_df)
    pitch_comp, starter_fallback_warn = augment_pitcher_scores_with_previous_season(
        pitch_comp,
        probable_candidates_today,
        fallback_season=report_date.year - 1,
    )
    warnings.extend(starter_fallback_warn)

    pen3 = bullpen_snapshot(report_date, lookback_days=3, pitch_comp=pitch_comp)
    pen7 = bullpen_snapshot(report_date, lookback_days=7, pitch_comp=pitch_comp)
    park_factors = build_park_factor_table(report_date)
    if park_factors.empty:
        warnings.append("Park-factor snapshot empty; venue adjustments are neutral.")

    b_df = statcast_batter_snapshot(season_start, pregame_statcast_end_date(report_date), game_types=game_types)
    bat_agg = aggregate_batter_quality(b_df)
    bat_comp = build_batter_composite(bat_agg)

    game_refs = [
        {
            "game_pk": g.game_pk,
            "away": g.away,
            "home": g.home,
            "venue_id": g.venue_id,
            "venue": g.venue,
            "scheduled_dt_utc": g.scheduled_dt_utc or start_time_et_to_utc_iso(report_date, g.start_time_et),
        }
        for g in games
    ]
    game_context_map = build_game_context_map(
        game_refs,
        strengths,
        season_raw,
        bat_comp,
        prior_batter_season=report_date.year - 1,
    )
    weather_map = build_game_weather_map([
        {
            "game_pk": g.game_pk,
            "venue_id": g.venue_id,
            "venue": g.venue,
            "scheduled_dt_utc": g.scheduled_dt_utc or start_time_et_to_utc_iso(report_date, g.start_time_et),
        }
        for g in games
    ])
    if games:
        neutral_weather = sum(1 for ctx in weather_map.values() if str(ctx.get("weather_source", "neutral")) == "neutral")
        if neutral_weather:
            warnings.append(f"Weather context unavailable for {neutral_weather}/{len(games)} scheduled games; weather adjustments are neutral for those matchups.")

    closing_marked, closing_warn = mark_closing_market_snapshots(report_date, games)
    warnings.extend(closing_warn)
    if closing_marked:
        warnings.append(f"Closing market snapshots marked for {closing_marked} sportsbook game rows.")

    settled_count, settle_warn = settle_bet_journal(report_date)
    warnings.extend(settle_warn)
    if settled_count:
        warnings.append(f"Bet journal settled {settled_count} historical candidate(s).")

    market_map, market_reload_warn = load_market_consensus_for_date(report_date, games)
    warnings.extend(market_reload_warn)

    journal_count, journal_warn = refresh_bet_journal(
        report_date,
        games,
        strengths,
        pitch_comp,
        pen3,
        pen7,
        park_factors,
        weather_map,
        game_context_map,
        model_state,
        market_map=market_map,
    )
    warnings.extend(journal_warn)
    if journal_count:
        warnings.append(f"Bet journal updated with {journal_count} watch candidate(s) from current market comparisons.")

    txt_path = write_txt_report(report_date, games, strengths, pitch_comp, pen3, pen7, park_factors, weather_map, game_context_map, market_map, model_state, warnings)
    xlsx_path = write_excel_season_workbook(
        report_date,
        strengths,
        pitch_agg,
        pitch_comp,
        bat_agg,
        bat_comp,
    )

    print(f"Wrote: {txt_path}")
    print(f"Wrote: {xlsx_path}")
if __name__ == "__main__":
    main()


























































































































