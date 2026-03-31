from __future__ import annotations

import copy
import datetime as dt
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PYC_PATH = _THIS_DIR / '__pycache__' / 'run_daily_mlb_report.cpython-314.pyc'
if not _PYC_PATH.exists():
    raise FileNotFoundError(f'Missing recovery bytecode: {_PYC_PATH}')

_loader = importlib.machinery.SourcelessFileLoader('_run_daily_mlb_report_core', str(_PYC_PATH))
_spec = importlib.util.spec_from_loader('_run_daily_mlb_report_core', _loader)
if _spec is None:
    raise ImportError(f'Unable to load module spec from {_PYC_PATH}')
_core = importlib.util.module_from_spec(_spec)
_loader.exec_module(_core)

for _name in dir(_core):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_core, _name)

_orig_default_model_state = _core.default_model_state
_orig_load_model_state = _core.load_model_state
_orig_recalibrate_model_weights_from_log = _core.recalibrate_model_weights_from_log
_orig_load_backtest_log = _core.load_backtest_log
_orig_build_backtest_rows_for_date = _core.build_backtest_rows_for_date
_orig_backtest_rows_with_model_state = _core.backtest_rows_with_model_state
_orig_calibrated_run_expectation_detail = _core.calibrated_run_expectation_detail
_orig_write_txt_report = _core.write_txt_report

MARGIN_CALIBRATION_MIN_GAMES = 24
MARGIN_CALIBRATION_MIN_IMPROVEMENT = 0.05
MARGIN_CALIBRATION_HALF_LIFE_DAYS = 45.0


def identity_margin_calibration(reason: str = 'identity margin mapping') -> Dict[str, object]:
    return {
        'apply': False,
        'mode': 'identity',
        'intercept': 0.0,
        'slope': 1.0,
        'games': 0,
        'raw_rmse': None,
        'calibrated_rmse': None,
        'reason': reason,
    }


def ensure_margin_calibration_state(state: Dict) -> Dict:
    raw_profiles = state.get('margin_calibration') if isinstance(state.get('margin_calibration'), dict) else {}
    profiles: Dict[str, Dict[str, object]] = {}
    for phase in ['spring', 'regular']:
        profile = identity_margin_calibration()
        raw_profile = raw_profiles.get(phase) if isinstance(raw_profiles.get(phase), dict) else {}
        profile.update(raw_profile)
        profile['apply'] = bool(profile.get('apply'))
        profile['mode'] = str(profile.get('mode') or 'identity')
        profile['intercept'] = float(np.clip(_core._finite(profile.get('intercept'), 0.0), -1.5, 1.5))
        profile['slope'] = float(np.clip(_core._finite(profile.get('slope'), 1.0), 0.50, 1.50))
        profile['games'] = int(max(_core._finite(profile.get('games'), 0), 0))
        profile['raw_rmse'] = _core.safe_float(profile.get('raw_rmse'))
        profile['calibrated_rmse'] = _core.safe_float(profile.get('calibrated_rmse'))
        profile['reason'] = str(profile.get('reason') or 'identity margin mapping')
        profiles[phase] = profile
    state['margin_calibration'] = profiles
    return state


def blend_margin_calibration_profiles(primary: Dict[str, object], secondary: Dict[str, object], secondary_share: float, label: str) -> Dict[str, object]:
    share = float(np.clip(_core._finite(secondary_share, 0.0), 0.0, 1.0))
    primary_profile = identity_margin_calibration()
    if isinstance(primary, dict):
        primary_profile.update(primary)
    secondary_profile = identity_margin_calibration()
    if isinstance(secondary, dict):
        secondary_profile.update(secondary)

    primary_apply = bool(primary_profile.get('apply'))
    secondary_apply = bool(secondary_profile.get('apply'))
    primary_intercept = float(_core._finite(primary_profile.get('intercept'), 0.0)) if primary_apply else 0.0
    secondary_intercept = float(_core._finite(secondary_profile.get('intercept'), 0.0)) if secondary_apply else 0.0
    primary_slope = float(_core._finite(primary_profile.get('slope'), 1.0)) if primary_apply else 1.0
    secondary_slope = float(_core._finite(secondary_profile.get('slope'), 1.0)) if secondary_apply else 1.0

    blended = identity_margin_calibration()
    blended.update({
        'apply': bool(primary_apply or secondary_apply),
        'mode': label if (primary_apply or secondary_apply) else 'identity',
        'intercept': float(((1.0 - share) * primary_intercept) + (share * secondary_intercept)),
        'slope': float(np.clip(((1.0 - share) * primary_slope) + (share * secondary_slope), 0.50, 1.50)),
        'games': int(round(((1.0 - share) * float(_core._finite(primary_profile.get('games'), 0.0))) + (share * float(_core._finite(secondary_profile.get('games'), 0.0))))),
        'raw_rmse': _core.safe_float(primary_profile.get('raw_rmse')) if primary_apply else _core.safe_float(secondary_profile.get('raw_rmse')),
        'calibrated_rmse': _core.safe_float(primary_profile.get('calibrated_rmse')) if primary_apply else _core.safe_float(secondary_profile.get('calibrated_rmse')),
        'reason': f'blended margin calibration ({label}) with spring share {share:.0%}',
    })
    if not blended['apply']:
        blended['reason'] = 'identity margin mapping'
    return blended


def active_margin_calibration(state: Dict, model_date: Optional[dt.date] = None, phase: Optional[str] = None) -> Dict[str, object]:
    state = ensure_margin_calibration_state(state)
    model_date = model_date or _core.today_local()
    phase_name = _core.normalize_model_phase(phase) or _core.season_phase_for_date(model_date)
    spring_profile = dict(state['margin_calibration'].get('spring', identity_margin_calibration()))
    regular_profile = dict(state['margin_calibration'].get('regular', identity_margin_calibration()))
    if phase_name != 'regular':
        return spring_profile if phase_name == 'spring' else regular_profile

    opening_day = _core.opening_day_for_season(model_date.year)
    transition = state.get('regular_transition') if isinstance(state.get('regular_transition'), dict) else {}
    bridge_days = max(int(_core._finite(transition.get('bridge_days'), _core.REGULAR_TRANSITION_BRIDGE_DAYS)), 0)
    max_share = float(np.clip(_core._finite(transition.get('spring_carryover_max'), _core.REGULAR_TRANSITION_SPRING_CARRYOVER_MAX), 0.0, 1.0))
    if opening_day is None or bridge_days <= 0 or model_date < opening_day:
        return regular_profile

    days_since_open = max((model_date - opening_day).days, 0)
    if days_since_open >= bridge_days:
        return regular_profile

    spring_share = max_share * (1.0 - (days_since_open / max(float(bridge_days), 1.0)))
    return blend_margin_calibration_profiles(regular_profile, spring_profile, spring_share, 'regular_margin_bridge')


def apply_margin_calibration(projected_margin: float, calibration: Optional[Dict[str, object]]) -> float:
    raw_margin = float(_core._finite(projected_margin, 0.0))
    if not isinstance(calibration, dict) or not bool(calibration.get('apply')):
        return float(np.clip(raw_margin, -6.5, 6.5))
    intercept = float(_core._finite(calibration.get('intercept'), 0.0))
    slope = float(_core._finite(calibration.get('slope'), 1.0))
    calibrated = intercept + (slope * raw_margin)
    return float(np.clip(calibrated, -6.5, 6.5))


def default_model_state() -> Dict:
    return ensure_margin_calibration_state(_orig_default_model_state())


def load_backtest_log() -> pd.DataFrame:
    df = _orig_load_backtest_log()
    if df.empty:
        if 'projected_margin_raw' not in df.columns:
            df['projected_margin_raw'] = np.nan
        return df
    if 'projected_margin_raw' not in df.columns:
        df['projected_margin_raw'] = pd.to_numeric(df.get('projected_margin'), errors='coerce')
    else:
        df['projected_margin_raw'] = pd.to_numeric(df['projected_margin_raw'], errors='coerce')
    if 'projected_margin' in df.columns:
        df['projected_margin'] = pd.to_numeric(df['projected_margin'], errors='coerce')
        missing_raw = df['projected_margin_raw'].isna()
        if missing_raw.any():
            df.loc[missing_raw, 'projected_margin_raw'] = df.loc[missing_raw, 'projected_margin']
    return df


def update_margin_calibration(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    state = ensure_margin_calibration_state(state)
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    default_profiles = {
        'spring': identity_margin_calibration('no spring backtest rows available for margin calibration'),
        'regular': identity_margin_calibration('no regular-season backtest rows available for margin calibration'),
    }
    if df is None or df.empty:
        state['margin_calibration'] = default_profiles
        return state

    if 'phase' not in df.columns:
        df['phase'] = np.nan
    normalized_phase = []
    normalized_date = []
    for raw_phase, raw_date in zip(df['phase'].tolist(), df.get('date', pd.Series(dtype=str)).tolist()):
        iso = _core.normalize_backtest_date_str(raw_date)
        normalized_date.append(iso)
        phase_name = _core.normalize_model_phase(raw_phase)
        if phase_name is None and iso:
            phase_name = _core.season_phase_for_date(dt.date.fromisoformat(iso))
        normalized_phase.append(phase_name or 'regular')
    df['phase'] = normalized_phase
    df['date'] = normalized_date
    df = df[df['date'].notna()].copy()
    if df.empty:
        state['margin_calibration'] = default_profiles
        return state

    fitted_profiles: Dict[str, Dict[str, object]] = {}
    for phase_name in ['spring', 'regular']:
        phase_rows = df[df['phase'] == phase_name].copy()
        profile = identity_margin_calibration(f'no {phase_name} backtest rows available for margin calibration')
        if phase_rows.empty:
            fitted_profiles[phase_name] = profile
            continue

        for col in ['projected_margin_raw', 'projected_margin', 'away_expected_runs', 'home_expected_runs', 'home_score', 'away_score']:
            if col not in phase_rows.columns:
                phase_rows[col] = np.nan
            phase_rows[col] = pd.to_numeric(phase_rows[col], errors='coerce')
        missing_raw = phase_rows['projected_margin_raw'].isna()
        if missing_raw.any():
            phase_rows.loc[missing_raw, 'projected_margin_raw'] = phase_rows.loc[missing_raw, 'projected_margin']
        missing_raw = phase_rows['projected_margin_raw'].isna()
        if missing_raw.any():
            phase_rows.loc[missing_raw, 'projected_margin_raw'] = phase_rows.loc[missing_raw, 'home_expected_runs'] - phase_rows.loc[missing_raw, 'away_expected_runs']
        phase_rows['actual_margin'] = phase_rows['home_score'] - phase_rows['away_score']
        phase_rows = phase_rows.dropna(subset=['projected_margin_raw', 'actual_margin']).copy()
        if phase_rows.empty:
            profile['reason'] = f'no valid {phase_name} margin rows available for calibration'
            fitted_profiles[phase_name] = profile
            continue

        x = pd.to_numeric(phase_rows['projected_margin_raw'], errors='coerce').astype(float).to_numpy(dtype=float)
        y = pd.to_numeric(phase_rows['actual_margin'], errors='coerce').astype(float).to_numpy(dtype=float)
        n_games = int(len(phase_rows))
        raw_rmse = float(np.sqrt(np.mean(np.square(y - x))))
        profile['games'] = n_games
        profile['raw_rmse'] = raw_rmse
        if n_games < MARGIN_CALIBRATION_MIN_GAMES:
            profile['reason'] = f'need at least {MARGIN_CALIBRATION_MIN_GAMES} {phase_name} games before enabling margin calibration'
            fitted_profiles[phase_name] = profile
            continue

        dts = pd.to_datetime(phase_rows.get('date'), errors='coerce')
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
            'apply': bool(improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT),
            'mode': 'affine_margin_calibration' if improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT else 'identity',
            'intercept': intercept,
            'slope': slope,
            'games': n_games,
            'raw_rmse': raw_rmse,
            'calibrated_rmse': calibrated_rmse,
            'reason': (
                f'{phase_name} margin calibration improved RMSE by {improvement:.3f} over {n_games} historical games'
                if improvement > MARGIN_CALIBRATION_MIN_IMPROVEMENT else
                f'{phase_name} margin calibration did not improve RMSE enough to activate ({raw_rmse:.3f} to {calibrated_rmse:.3f})'
            ),
        })
        fitted_profiles[phase_name] = profile

    state['margin_calibration'] = fitted_profiles
    return state


def load_model_state() -> Dict:
    state = ensure_margin_calibration_state(_orig_load_model_state())
    return update_margin_calibration(state)


def save_model_state(state: Dict) -> None:
    payload = ensure_margin_calibration_state(
        _core.ensure_totals_refresh_state(
            _core.ensure_total_calibration_state(
                _core.ensure_probability_calibration_state(
                    _core.ensure_backtest_refresh_state(_core.ensure_phase_weight_state(state))
                )
            )
        )
    )
    if isinstance(state.get('probability_calibration'), dict):
        payload['probability_calibration'] = copy.deepcopy(state['probability_calibration'])
    if isinstance(state.get('total_calibration'), dict):
        payload['total_calibration'] = copy.deepcopy(state['total_calibration'])
    if isinstance(state.get('margin_calibration'), dict):
        payload['margin_calibration'] = copy.deepcopy(state['margin_calibration'])
    if isinstance(state.get('totals_refresh'), dict):
        payload['totals_refresh'] = copy.deepcopy(state['totals_refresh'])
    payload['version'] = 2
    payload['weights'] = dict(payload['weights_by_phase']['regular'])
    with open(_core.MODEL_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def recalibrate_model_weights_from_log(state: Dict, backtest_df: Optional[pd.DataFrame] = None) -> Dict:
    updated = _orig_recalibrate_model_weights_from_log(state, backtest_df)
    return update_margin_calibration(updated, backtest_df)


def build_backtest_rows_for_date(target_date: dt.date, model_state: Dict):
    rows, warnings = _orig_build_backtest_rows_for_date(target_date, model_state)
    for row in rows:
        if 'projected_margin_raw' not in row or pd.isna(row.get('projected_margin_raw')):
            row['projected_margin_raw'] = row.get('projected_margin')
    return rows, warnings


def calibrated_run_expectation_detail(components: Optional[Dict[str, object]], model_state: Optional[Dict] = None, model_date: Optional[dt.date] = None, model_phase: Optional[str] = None) -> Dict[str, float]:
    detail = _orig_calibrated_run_expectation_detail(components, model_state=model_state, model_date=model_date, model_phase=model_phase)
    state = model_state if isinstance(model_state, dict) else default_model_state()
    raw_margin = _core.safe_float(detail.get('projected_margin_raw'))
    if raw_margin is None:
        raw_margin = _core.safe_float(detail.get('projected_margin'))
    if raw_margin is None:
        raw_margin = float(_core._finite(detail.get('home_expected_runs'), 0.0) - _core._finite(detail.get('away_expected_runs'), 0.0))
    raw_margin = float(np.clip(_core._finite(raw_margin, 0.0), -6.5, 6.5))
    total_runs = float(np.clip(_core._finite(detail.get('total_expected_runs'), 8.5), 5.5, 14.5))
    margin_cal = active_margin_calibration(state, model_date=model_date, phase=model_phase)
    calibrated_margin = apply_margin_calibration(raw_margin, margin_cal)
    home_runs = float((total_runs + calibrated_margin) / 2.0)
    away_runs = float(total_runs - home_runs)
    if home_runs < 1.5:
        home_runs = 1.5
        away_runs = total_runs - home_runs
    if away_runs < 1.5:
        away_runs = 1.5
        home_runs = total_runs - away_runs
    home_runs = float(np.clip(home_runs, 1.5, 10.5))
    away_runs = float(np.clip(total_runs - home_runs, 1.5, 10.5))
    total_runs = float(np.clip(home_runs + away_runs, 5.5, 14.5))
    calibrated_margin = float(np.clip(home_runs - away_runs, -6.5, 6.5))
    detail['projected_margin_raw'] = raw_margin
    detail['projected_margin'] = calibrated_margin
    detail['away_expected_runs'] = away_runs
    detail['home_expected_runs'] = home_runs
    detail['total_expected_runs'] = total_runs
    detail['margin_calibration_mode'] = str(margin_cal.get('mode', 'identity'))
    detail['margin_calibration_reason'] = str(margin_cal.get('reason', ''))
    return detail


def backtest_rows_with_model_state(backtest_df: Optional[pd.DataFrame], state: Dict, apply_calibration: bool = True) -> pd.DataFrame:
    df = _orig_backtest_rows_with_model_state(backtest_df, state, apply_calibration=apply_calibration)
    if df is None or df.empty:
        return df
    df = df.copy()
    if 'projected_margin_raw' not in df.columns:
        source = backtest_df if isinstance(backtest_df, pd.DataFrame) else pd.DataFrame()
        if isinstance(source, pd.DataFrame) and 'projected_margin_raw' in source.columns:
            df['projected_margin_raw'] = pd.to_numeric(source['projected_margin_raw'], errors='coerce')
        else:
            df['projected_margin_raw'] = pd.to_numeric(df.get('projected_margin'), errors='coerce')
    else:
        df['projected_margin_raw'] = pd.to_numeric(df['projected_margin_raw'], errors='coerce')
    if 'projected_margin' not in df.columns:
        df['projected_margin'] = np.nan
    df['projected_margin'] = pd.to_numeric(df['projected_margin'], errors='coerce')

    calibrated_vals = []
    margin_modes = []
    away_vals = []
    home_vals = []
    for _, row in df.iterrows():
        iso = _core.normalize_backtest_date_str(row.get('date'))
        model_date = dt.date.fromisoformat(iso) if iso else _core.today_local()
        phase_name = _core.normalize_model_phase(row.get('phase')) or _core.season_phase_for_date(model_date)
        raw_margin = _core.safe_float(row.get('projected_margin_raw'))
        if raw_margin is None:
            raw_margin = _core.safe_float(row.get('projected_margin'))
        if raw_margin is None:
            raw_margin = float(_core._finite(row.get('home_expected_runs'), 0.0) - _core._finite(row.get('away_expected_runs'), 0.0))
        raw_margin = float(np.clip(_core._finite(raw_margin, 0.0), -6.5, 6.5))
        margin_cal = active_margin_calibration(state, model_date=model_date, phase=phase_name)
        calibrated_margin = apply_margin_calibration(raw_margin, margin_cal) if apply_calibration else raw_margin
        total_value = _core.safe_float(row.get('projected_total'))
        if total_value is None:
            total_value = _core._finite(row.get('away_expected_runs'), 0.0) + _core._finite(row.get('home_expected_runs'), 0.0)
        total_value = float(np.clip(_core._finite(total_value, 8.5), 5.5, 14.5))
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
        calibrated_vals.append(float(np.clip(home_runs - away_runs, -6.5, 6.5)))
        margin_modes.append(str(margin_cal.get('mode', 'identity')))
        away_vals.append(away_runs)
        home_vals.append(home_runs)
    df['projected_margin'] = calibrated_vals
    df['margin_calibration_mode'] = margin_modes
    df['away_expected_runs'] = away_vals
    df['home_expected_runs'] = home_vals
    df['projected_total'] = pd.to_numeric(df['away_expected_runs'], errors='coerce') + pd.to_numeric(df['home_expected_runs'], errors='coerce')
    return df


def rewrite_backtest_predictions_with_model_state(state: Dict) -> pd.DataFrame:
    df = backtest_rows_with_model_state(load_backtest_log(), state)
    if df is not None and not df.empty:
        df.to_csv(_core.BACKTEST_LOG_PATH, index=False)
    return df


def estimate_margin_distribution_sigma(model_state: Optional[Dict] = None, backtest_df: Optional[pd.DataFrame] = None) -> float:
    df = load_backtest_log() if backtest_df is None else backtest_df.copy()
    if df is None or df.empty:
        return 3.25
    for col in ['home_score', 'away_score', 'projected_margin_raw', 'projected_margin']:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['actual_margin'] = df['home_score'] - df['away_score']
    source_margin = df['projected_margin_raw'] if 'projected_margin_raw' in df.columns else df['projected_margin']
    df['source_margin'] = pd.to_numeric(source_margin, errors='coerce')
    df = df.dropna(subset=['actual_margin', 'source_margin']).copy()
    if df.empty:
        return 3.25
    residual = df['actual_margin'].astype(float) - df['source_margin'].astype(float)
    sigma = float(np.sqrt(np.mean(np.square(residual))))
    return float(np.clip(_core._finite(sigma, 3.25), 1.75, 6.0))


def write_txt_report(report_date, games, strengths, pitch_comp, pen3, pen7, park_factors, weather_map, game_context_map, market_map, model_state, warnings):
    path = _orig_write_txt_report(report_date, games, strengths, pitch_comp, pen3, pen7, park_factors, weather_map, game_context_map, market_map, model_state, warnings)
    try:
        p = Path(path)
        lines = p.read_text(encoding='utf-8').splitlines()
        if not any(line.startswith('Margin calibration:') for line in lines):
            report_phase = _core.season_phase_for_date(report_date)
            active_margin = active_margin_calibration(model_state, model_date=report_date, phase=report_phase)
            if active_margin.get('apply'):
                margin_line = (
                    f"Margin calibration: {str(active_margin.get('mode', 'identity')).replace('_', ' ')} | "
                    f"slope {_core.fmt(active_margin.get('slope'), 3)} | "
                    f"intercept {_core.fmt(active_margin.get('intercept'), 3)} | "
                    f"raw RMSE {_core.fmt(active_margin.get('raw_rmse'), 3)} -> calibrated {_core.fmt(active_margin.get('calibrated_rmse'), 3)}"
                )
            else:
                margin_line = f"Margin calibration: identity | {str(active_margin.get('reason', 'no margin calibration adjustment active')).strip()}"
            insert_at = next((idx + 1 for idx, line in enumerate(lines) if line.startswith('Probability calibration:')), None)
            if insert_at is not None:
                lines.insert(insert_at, margin_line)
                p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    except Exception:
        pass
    return path


for _name, _value in {
    'identity_margin_calibration': identity_margin_calibration,
    'ensure_margin_calibration_state': ensure_margin_calibration_state,
    'blend_margin_calibration_profiles': blend_margin_calibration_profiles,
    'active_margin_calibration': active_margin_calibration,
    'apply_margin_calibration': apply_margin_calibration,
    'default_model_state': default_model_state,
    'load_backtest_log': load_backtest_log,
    'update_margin_calibration': update_margin_calibration,
    'load_model_state': load_model_state,
    'save_model_state': save_model_state,
    'recalibrate_model_weights_from_log': recalibrate_model_weights_from_log,
    'build_backtest_rows_for_date': build_backtest_rows_for_date,
    'calibrated_run_expectation_detail': calibrated_run_expectation_detail,
    'backtest_rows_with_model_state': backtest_rows_with_model_state,
    'rewrite_backtest_predictions_with_model_state': rewrite_backtest_predictions_with_model_state,
    'estimate_margin_distribution_sigma': estimate_margin_distribution_sigma,
    'write_txt_report': write_txt_report,
}.items():
    globals()[_name] = _value
    setattr(_core, _name, _value)

main = _core.main

if __name__ == '__main__':
    main()
