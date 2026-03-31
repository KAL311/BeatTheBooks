from __future__ import annotations

import datetime as dt
import itertools
import json
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional runtime dependency
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        'Streamlit is required to run this dashboard. Install the packages in quant_dashboard_requirements.txt and run `streamlit run quant_dashboard.py`.'
    ) from exc

try:  # pragma: no cover - optional runtime dependency
    import plotly.express as px
except ModuleNotFoundError:  # pragma: no cover
    px = None

import policy_research
import portfolio_engine as portfolio
import quant_store
import market_helpers as market

PLOTLY_COLOR_SEQUENCE = ['#0F766E', '#2563EB', '#D97706', '#DC2626', '#7C3AED', '#4B5563']


def _frame_or_empty(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    numeric = _safe_float(value)
    return 'N/A' if numeric is None else f'{numeric * 100:.1f}%'


def _fmt_num(value: Any, digits: int = 3) -> str:
    numeric = _safe_float(value)
    return 'N/A' if numeric is None else f'{numeric:.{digits}f}'


def _coerce_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors='coerce')
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def _fmt_date(value: Any) -> str:
    parsed = _coerce_date(value)
    return 'N/A' if parsed is None else parsed.isoformat()


def _pretty_label(value: Any) -> str:
    text = str(value or '').replace('_', ' ').replace('-', ' ')
    text = ' '.join(text.split())
    if not text:
        return ''
    replacements = {
        'avg': 'Average',
        'avg.': 'Average',
        'clv': 'CLV',
        'ev': 'EV',
        'games': 'Games',
        'ip': 'IP',
        'mae': 'MAE',
        'mlb': 'MLB',
        'oos': 'OOS',
        'p&l': 'P&L',
        'pct': 'Pct',
        'rmse': 'RMSE',
        'roi': 'ROI',
        'tto': 'TTO',
        'vs': 'vs',
    }
    words: list[str] = []
    for raw_word in text.split():
        lead = ''
        trail = ''
        word = raw_word
        while word and not word[0].isalnum():
            lead += word[0]
            word = word[1:]
        while word and not word[-1].isalnum():
            trail = word[-1] + trail
            word = word[:-1]
        lowered = word.lower()
        pretty = replacements.get(lowered, word.capitalize())
        words.append(f'{lead}{pretty}{trail}')
    return ' '.join(words)


def _display_frame(frame: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    renamed = frame.copy()
    overrides = overrides or {}
    renamed.columns = [overrides.get(str(col), _pretty_label(col)) for col in renamed.columns]
    return renamed


def _phase_label_options() -> dict[str, str]:
    return {
        'All Phases': 'all',
        'Spring Training': 'spring',
        'Regular Season': 'regular',
    }


def _normalize_phase(value: Any) -> str:
    text = str(value or '').strip().lower()
    if text in {'spring', 'preseason', 'spring training'}:
        return 'spring'
    if text in {'regular', 'regular season'}:
        return 'regular'
    return text


def _infer_phase_from_date(value: Any) -> str | None:
    target = _coerce_date(value)
    if target is None:
        return None
    try:
        return _normalize_phase(market.season_phase_for_date(target))
    except Exception:
        return None


def _ensure_phase_column(frame: pd.DataFrame, date_candidates: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    if 'phase' in work.columns:
        work['phase'] = work['phase'].astype(str).map(_normalize_phase)
        return work
    date_candidates = date_candidates or ['report_date', 'date', 'as_of_report_date', 'created_ts']
    derived = None
    for col in date_candidates:
        if col in work.columns:
            derived = work[col].map(_infer_phase_from_date)
            if derived.notna().any():
                break
    if derived is not None:
        work['phase'] = derived.fillna('unknown').astype(str)
    return work


def _filter_frame_by_phase(frame: pd.DataFrame, phase_filter: str, date_candidates: list[str] | None = None) -> pd.DataFrame:
    if frame.empty or phase_filter == 'all':
        return frame
    work = _ensure_phase_column(frame, date_candidates=date_candidates)
    if 'phase' not in work.columns:
        return frame
    return work[work['phase'].astype(str).map(_normalize_phase) == phase_filter].copy()


def _phase_run_tables(status: Dict[str, Any], phase_filter: str) -> Dict[str, pd.DataFrame]:
    empty_tables = {
        'latest_run': pd.DataFrame(),
        'predictions': pd.DataFrame(),
        'candidates': pd.DataFrame(),
        'portfolio_recommendations': pd.DataFrame(columns=portfolio.PORTFOLIO_RECOMMENDATION_COLUMNS),
        'portfolio_summary': pd.DataFrame(columns=portfolio.PORTFOLIO_SUMMARY_COLUMNS),
        'portfolio_market_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'portfolio_team_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'portfolio_game_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'policy_research_rankings': pd.DataFrame(),
        'policy_research_shadow_daily': pd.DataFrame(),
        'policy_research_shadow_summary': pd.DataFrame(),
        'policy_research_market_daily': pd.DataFrame(),
        'policy_research_market_policy_daily': pd.DataFrame(),
        'policy_research_market_policy_summary': pd.DataFrame(),
        'market_proof_summary': pd.DataFrame(),
        'market_proof_by_market': pd.DataFrame(),
    }
    if phase_filter == 'all':
        return empty_tables
    if not status.get('duckdb_available') or not status.get('db_exists'):
        return empty_tables
    latest_run = quant_store.query_quant(
        'SELECT * FROM quant_runs_daily WHERE lower(coalesce(phase, \'\')) = ? ORDER BY report_date DESC, created_ts DESC LIMIT 1',
        [phase_filter],
    )
    run_id = None if latest_run.empty else str(latest_run.iloc[0].get('run_id') or '')
    if not run_id:
        empty_tables['latest_run'] = latest_run
        return empty_tables
    empty_tables.update({
        'latest_run': latest_run,
        'predictions': quant_store.query_quant('SELECT * FROM prediction_snapshots WHERE run_id = ? ORDER BY winner_prob DESC, game_pk', [run_id]),
        'candidates': quant_store.query_quant('SELECT * FROM market_candidates WHERE run_id = ? ORDER BY actionability_score DESC NULLS LAST, candidate_score DESC NULLS LAST', [run_id]),
        'portfolio_recommendations': quant_store.query_quant('SELECT * FROM portfolio_recommendations WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, priority_score DESC NULLS LAST, allocation_rank', [run_id]),
        'portfolio_summary': quant_store.query_quant('SELECT * FROM portfolio_summary WHERE run_id = ?', [run_id]),
        'portfolio_market_exposure': quant_store.query_quant('SELECT * FROM portfolio_market_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
        'portfolio_team_exposure': quant_store.query_quant('SELECT * FROM portfolio_team_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
        'portfolio_game_exposure': quant_store.query_quant('SELECT * FROM portfolio_game_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
        'policy_research_rankings': quant_store.query_quant('SELECT * FROM policy_research_window_rankings WHERE run_id = ? ORDER BY window_days ASC NULLS LAST, stability_score DESC NULLS LAST, policy_name', [run_id]),
        'policy_research_shadow_daily': quant_store.query_quant('SELECT * FROM policy_research_shadow_daily WHERE run_id = ? ORDER BY report_date, policy_name', [run_id]),
        'policy_research_shadow_summary': quant_store.query_quant('SELECT * FROM policy_research_shadow_summary WHERE run_id = ? ORDER BY result_units DESC NULLS LAST, policy_name', [run_id]),
        'policy_research_market_daily': quant_store.query_quant('SELECT * FROM policy_research_market_daily WHERE run_id = ? ORDER BY report_date', [run_id]),
        'policy_research_market_policy_daily': quant_store.query_quant('SELECT * FROM policy_research_market_policy_daily WHERE run_id = ? ORDER BY report_date, policy_name', [run_id]),
        'policy_research_market_policy_summary': quant_store.query_quant('SELECT * FROM policy_research_market_policy_summary WHERE run_id = ? ORDER BY result_units DESC NULLS LAST, policy_name', [run_id]),
        'market_proof_summary': quant_store.query_quant('SELECT * FROM market_proof_summary WHERE run_id = ?', [run_id]),
        'market_proof_by_market': quant_store.query_quant('SELECT * FROM market_proof_by_market WHERE run_id = ? ORDER BY market_type', [run_id]),
    })
    return empty_tables


def _latest_and_previous_report_run(run_history: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if run_history.empty:
        return {}, {}
    work = run_history.copy()
    if 'created_ts' in work.columns:
        work['created_ts'] = pd.to_datetime(work.get('created_ts'), errors='coerce')
    if 'report_date' in work.columns:
        work['report_date_dt'] = pd.to_datetime(work.get('report_date'), errors='coerce')
    sort_cols = [col for col in ['created_ts', 'report_date_dt'] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    latest = work.iloc[0].to_dict()
    latest_report_date = _coerce_date(latest.get('report_date'))
    previous: Dict[str, Any] = {}
    if latest_report_date is not None and 'report_date_dt' in work.columns:
        previous_rows = work[work['report_date_dt'].dt.date < latest_report_date]
        if not previous_rows.empty:
            previous = previous_rows.iloc[0].to_dict()
    if not previous and len(work) > 1:
        previous = work.iloc[1].to_dict()
    return latest, previous


def _metric_delta(current: Any, previous: Any, digits: int = 3, pct: bool = False, invert_good: bool = False) -> str | None:
    current_num = _safe_float(current)
    previous_num = _safe_float(previous)
    if current_num is None or previous_num is None:
        return None
    delta = current_num - previous_num
    if abs(delta) < (0.0005 if not pct else 0.00005):
        return 'flat'
    if pct:
        text = f'{delta * 100:+.{digits}f} pts'
    else:
        text = f'{delta:+.{digits}f}'
    if invert_good:
        text += ' (lower better)'
    return text


def _validation_freshness_frame(clean_backtest: pd.DataFrame, live_archive: pd.DataFrame, latest: Dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not clean_backtest.empty and 'date' in clean_backtest.columns:
        replay_dates = pd.to_datetime(clean_backtest.get('date'), errors='coerce').dropna().dt.date
        if not replay_dates.empty:
            rows.append({
                'Source': 'Research replay OOS',
                'Through date': replay_dates.max().isoformat(),
                'Start date': replay_dates.min().isoformat(),
                'Games': int(_safe_float(latest.get('validation_games')) or len(clean_backtest)),
                'Refresh cadence': 'Only when replay is rebuilt',
                'Role': 'Research baseline',
            })
    if not live_archive.empty and 'report_date' in live_archive.columns:
        settled = live_archive.copy()
        settled['report_date'] = pd.to_datetime(settled.get('report_date'), errors='coerce').dt.date
        if 'y_home' in settled.columns:
            settled = settled[pd.to_numeric(settled.get('y_home'), errors='coerce').notna()].copy()
        elif 'actual_total' in settled.columns:
            settled = settled[pd.to_numeric(settled.get('actual_total'), errors='coerce').notna()].copy()
        settled_dates = settled['report_date'].dropna() if 'report_date' in settled.columns else pd.Series(dtype=object)
        if not settled_dates.empty:
            rows.append({
                'Source': 'Live settled validation',
                'Through date': settled_dates.max().isoformat(),
                'Start date': settled_dates.min().isoformat(),
                'Games': int(_safe_float(latest.get('live_validation_games')) or len(settled)),
                'Refresh cadence': 'Updates as archived games settle',
                'Role': 'Current regular-season health',
            })
    return pd.DataFrame(rows)


def _validation_delta_frame(latest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    pairs = [
        ('Moneyline log loss', latest.get('validation_log_loss'), latest.get('live_validation_log_loss'), 4, False, True),
        ('Moneyline accuracy', latest.get('validation_accuracy'), latest.get('live_validation_accuracy'), 1, True, False),
        ('Totals RMSE', latest.get('validation_total_rmse'), latest.get('live_validation_total_rmse'), 3, False, True),
        ('Margin RMSE', latest.get('validation_margin_rmse'), latest.get('live_validation_margin_rmse'), 3, False, True),
    ]
    for label, research_value, live_value, digits, is_pct, invert_good in pairs:
        if _safe_float(research_value) is None and _safe_float(live_value) is None:
            continue
        rows.append({
            'Metric': label,
            'Research replay': _fmt_pct(research_value) if is_pct else _fmt_num(research_value, digits),
            'Live settled': _fmt_pct(live_value) if is_pct else _fmt_num(live_value, digits),
            'Delta': _metric_delta(live_value, research_value, digits=digits, pct=is_pct, invert_good=invert_good) or 'N/A',
        })
    return pd.DataFrame(rows)


def _yesterday_change_frame(run_history: pd.DataFrame) -> pd.DataFrame:
    latest, previous = _latest_and_previous_report_run(run_history)
    if not latest or not previous:
        return pd.DataFrame()
    prior_report_date = _fmt_date(previous.get('report_date'))
    rows = []
    comparisons = [
        ('Live moneyline log loss', latest.get('live_validation_log_loss'), previous.get('live_validation_log_loss'), 4, False, True),
        ('Live moneyline accuracy', latest.get('live_validation_accuracy'), previous.get('live_validation_accuracy'), 1, True, False),
        ('Live totals RMSE', latest.get('live_validation_total_rmse'), previous.get('live_validation_total_rmse'), 3, False, True),
        ('Live margin RMSE', latest.get('live_validation_margin_rmse'), previous.get('live_validation_margin_rmse'), 3, False, True),
        ('Settled live games', latest.get('live_validation_games'), previous.get('live_validation_games'), 0, False, False),
        ('Settled priced bets', latest.get('market_proof_settled_bets'), previous.get('market_proof_settled_bets'), 0, False, False),
        ('Totals avg CLV', latest.get('market_proof_total_avg_clv'), previous.get('market_proof_total_avg_clv'), 3, False, False),
    ]
    for label, current, previous_value, digits, is_pct, invert_good in comparisons:
        if _safe_float(current) is None and _safe_float(previous_value) is None:
            continue
        rows.append({
            'Metric': label,
            'Current': _fmt_pct(current) if is_pct else _fmt_num(current, digits),
            f'Vs {prior_report_date}': _fmt_pct(previous_value) if is_pct else _fmt_num(previous_value, digits),
            'Change': _metric_delta(current, previous_value, digits=digits, pct=is_pct, invert_good=invert_good) or 'N/A',
        })
    return pd.DataFrame(rows)


def _frictional_progress(current_value: Any, target_value: float, exponent: float = 1.35) -> float:
    current = max(_safe_float(current_value) or 0.0, 0.0)
    target = max(float(target_value or 0.0), 1.0)
    raw = max(0.0, min(current / target, 1.0))
    return float(raw ** exponent)


def _blend_target(baseline: Any, stretch: Any, progress: float) -> float | None:
    baseline_num = _safe_float(baseline)
    stretch_num = _safe_float(stretch)
    if baseline_num is None or stretch_num is None:
        return baseline_num if stretch_num is None else stretch_num
    progress = max(0.0, min(float(progress or 0.0), 1.0))
    return baseline_num + ((stretch_num - baseline_num) * progress)


def _target_gap_text(current: Any, target: Any, *, higher_is_better: bool, digits: int, is_pct: bool = False) -> tuple[str, str]:
    current_num = _safe_float(current)
    target_num = _safe_float(target)
    if current_num is None or target_num is None:
        return 'Sample building', 'Gap unavailable'
    tolerance = 0.0025 if not is_pct else 0.003
    diff = current_num - target_num
    if higher_is_better:
        if diff > tolerance:
            status = 'Ahead of target'
            amount = diff
            label = 'Ahead by'
        elif diff < -tolerance:
            status = 'Below target'
            amount = abs(diff)
            label = 'Short by'
        else:
            status = 'On pace'
            amount = abs(diff)
            label = 'Within band'
    else:
        reverse = target_num - current_num
        if reverse > tolerance:
            status = 'Ahead of target'
            amount = reverse
            label = 'Better by'
        elif reverse < -tolerance:
            status = 'Above target'
            amount = abs(reverse)
            label = 'High by'
        else:
            status = 'On pace'
            amount = abs(reverse)
            label = 'Within band'
    if is_pct:
        return status, f"{label} {amount * 100:.1f} pts"
    return status, f"{label} {amount:.{digits}f}"


def _dynamic_target_bundle(latest: Dict[str, Any], selected_phase: str) -> dict[str, Any]:
    phase_key = selected_phase if selected_phase != 'all' else _normalize_phase(latest.get('phase') or 'regular')
    if phase_key not in {'spring', 'regular'}:
        phase_key = 'regular'

    research_games = int(_safe_float(latest.get('validation_games')) or 0)
    live_games = int(_safe_float(latest.get('live_validation_games')) or 0)
    settled_days = int(_safe_float(latest.get('live_validation_settled_days')) or 0)
    priced_bets = int(_safe_float(latest.get('market_proof_settled_bets')) or 0)

    if phase_key == 'spring':
        progress = (
            0.80 * _frictional_progress(research_games, 180) +
            0.20 * _frictional_progress(priced_bets, 30)
        )
    else:
        progress = (
            0.55 * _frictional_progress(live_games, 300) +
            0.15 * _frictional_progress(settled_days, 30) +
            0.30 * _frictional_progress(priced_bets, 120)
        )
    progress = max(0.0, min(progress, 1.0))

    if phase_key == 'spring':
        current_log_loss = latest.get('validation_log_loss')
        current_accuracy = latest.get('validation_accuracy')
        current_total_rmse = latest.get('validation_total_rmse')
        current_margin_rmse = latest.get('validation_margin_rmse')
        current_avg_clv = latest.get('market_proof_avg_clv')
        current_roi = latest.get('market_proof_roi')
        sample_context = f'{research_games} replay games'
    else:
        current_log_loss = latest.get('live_validation_log_loss')
        current_accuracy = latest.get('live_validation_accuracy')
        current_total_rmse = latest.get('live_validation_total_rmse')
        current_margin_rmse = latest.get('live_validation_margin_rmse')
        current_avg_clv = latest.get('market_proof_avg_clv')
        current_roi = latest.get('market_proof_roi')
        sample_context = f'{live_games} live games | {priced_bets} priced bets'

    metric_specs = [
        {
            'metric': 'Moneyline log loss',
            'current': current_log_loss,
            'baseline': latest.get('validation_log_loss'),
            'stretch': max((_safe_float(latest.get('validation_log_loss')) or 0.72) - (0.020 if phase_key == 'spring' else 0.035), 0.680 if phase_key == 'spring' else 0.640),
            'higher_is_better': False,
            'digits': 4,
            'is_pct': False,
        },
        {
            'metric': 'Moneyline accuracy',
            'current': current_accuracy,
            'baseline': latest.get('validation_accuracy'),
            'stretch': min((_safe_float(latest.get('validation_accuracy')) or 0.52) + (0.020 if phase_key == 'spring' else 0.035), 0.560 if phase_key == 'spring' else 0.600),
            'higher_is_better': True,
            'digits': 1,
            'is_pct': True,
        },
        {
            'metric': 'Totals RMSE',
            'current': current_total_rmse,
            'baseline': latest.get('validation_total_rmse'),
            'stretch': max((_safe_float(latest.get('validation_total_rmse')) or 5.0) - (0.120 if phase_key == 'spring' else 0.250), 4.800 if phase_key == 'spring' else 4.550),
            'higher_is_better': False,
            'digits': 3,
            'is_pct': False,
        },
        {
            'metric': 'Margin RMSE',
            'current': current_margin_rmse,
            'baseline': latest.get('validation_margin_rmse'),
            'stretch': max((_safe_float(latest.get('validation_margin_rmse')) or 5.0) - (0.120 if phase_key == 'spring' else 0.250), 4.900 if phase_key == 'spring' else 4.500),
            'higher_is_better': False,
            'digits': 3,
            'is_pct': False,
        },
        {
            'metric': 'Average CLV',
            'current': current_avg_clv,
            'baseline': 0.0,
            'stretch': 0.010 if phase_key == 'spring' else 0.018,
            'higher_is_better': True,
            'digits': 3,
            'is_pct': False,
        },
        {
            'metric': 'ROI',
            'current': current_roi,
            'baseline': 0.0,
            'stretch': 0.020 if phase_key == 'spring' else 0.035,
            'higher_is_better': True,
            'digits': 1,
            'is_pct': True,
        },
    ]

    rows = []
    for spec in metric_specs:
        dynamic_target = _blend_target(spec['baseline'], spec['stretch'], progress)
        status, gap_text = _target_gap_text(
            spec['current'],
            dynamic_target,
            higher_is_better=spec['higher_is_better'],
            digits=spec['digits'],
            is_pct=spec['is_pct'],
        )
        rows.append({
            'Metric': spec['metric'],
            'Current': _fmt_pct(spec['current']) if spec['is_pct'] else _fmt_num(spec['current'], spec['digits']),
            'Dynamic target': _fmt_pct(dynamic_target) if spec['is_pct'] else _fmt_num(dynamic_target, spec['digits']),
            'Baseline': _fmt_pct(spec['baseline']) if spec['is_pct'] else _fmt_num(spec['baseline'], spec['digits']),
            'Long-run goal': _fmt_pct(spec['stretch']) if spec['is_pct'] else _fmt_num(spec['stretch'], spec['digits']),
            'Status': status,
            'Gap to target': gap_text,
            'Sample context': sample_context,
        })

    return {
        'frame': pd.DataFrame(rows),
        'phase_key': phase_key,
        'progress': progress,
        'live_games': live_games,
        'settled_days': settled_days,
        'priced_bets': priced_bets,
        'research_games': research_games,
    }


def _daily_validation(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty or 'date' not in backtest.columns:
        return pd.DataFrame()
    work = backtest.copy()
    work['date'] = pd.to_datetime(work['date'], errors='coerce').dt.date
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    work['projected_total'] = pd.to_numeric(work.get('projected_total'), errors='coerce')
    work['projected_margin'] = pd.to_numeric(work.get('projected_margin'), errors='coerce')
    work['home_score'] = pd.to_numeric(work.get('home_score'), errors='coerce')
    work['away_score'] = pd.to_numeric(work.get('away_score'), errors='coerce')
    work = work.dropna(subset=['date'])
    if work.empty:
        return pd.DataFrame()
    rows = []
    for target_date, grp in work.groupby('date'):
        prob_grp = grp.dropna(subset=['p_home', 'y_home']).copy()
        total_grp = grp.dropna(subset=['projected_total', 'home_score', 'away_score']).copy()
        margin_grp = grp.dropna(subset=['projected_margin', 'home_score', 'away_score']).copy()
        log_loss = None
        accuracy = None
        if not prob_grp.empty:
            p = prob_grp['p_home'].astype(float)
            y = prob_grp['y_home'].astype(int)
            log_loss = float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean())
            accuracy = float(((p >= 0.5).astype(int) == y).mean())
        total_mae = None
        margin_mae = None
        if not total_grp.empty:
            actual_total = total_grp['home_score'] + total_grp['away_score']
            total_mae = float((actual_total - total_grp['projected_total']).abs().mean())
        if not margin_grp.empty:
            actual_margin = margin_grp['home_score'] - margin_grp['away_score']
            margin_mae = float((actual_margin - margin_grp['projected_margin']).abs().mean())
        rows.append({
            'date': target_date,
            'games': int(len(grp)),
            'log_loss': log_loss,
            'accuracy': accuracy,
            'total_mae': total_mae,
            'margin_mae': margin_mae,
        })
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def _calibration_table(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty or 'p_home' not in backtest.columns or 'y_home' not in backtest.columns:
        return pd.DataFrame()
    work = backtest.copy()
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce')
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    work = work.dropna(subset=['p_home', 'y_home'])
    if work.empty:
        return pd.DataFrame()
    bins = pd.interval_range(start=0.35, end=0.85, periods=5)
    work['bucket'] = pd.cut(work['p_home'], bins=bins)
    rows = []
    for bucket, grp in work.groupby('bucket', dropna=True):
        rows.append({
            'bucket': str(bucket),
            'games': int(len(grp)),
            'predicted_win_rate': float(grp['p_home'].mean()),
            'actual_win_rate': float(grp['y_home'].mean()),
        })
    return pd.DataFrame(rows)


def _render_line_chart(frame: pd.DataFrame, x_col: str, y_col: str, title: str):
    if frame.empty or y_col not in frame.columns:
        st.info(f'No data available for {title.lower()}.')
        return
    if px is not None:
        fig = px.line(frame, x=x_col, y=y_col, title=title, color_discrete_sequence=PLOTLY_COLOR_SEQUENCE)
        _show_plotly(fig)
    else:
        st.line_chart(frame.set_index(x_col)[y_col], use_container_width=True)


def _render_bar_chart(frame: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str | None = None):
    if frame.empty or y_col not in frame.columns:
        st.info(f'No data available for {title.lower()}.')
        return
    if px is not None:
        kwargs: Dict[str, Any] = {'title': title}
        if color_col and color_col in frame.columns:
            kwargs['color'] = color_col
        fig = px.bar(frame, x=x_col, y=y_col, color_discrete_sequence=PLOTLY_COLOR_SEQUENCE, **kwargs)
        _show_plotly(fig)
    else:
        st.bar_chart(frame.set_index(x_col)[y_col], use_container_width=True)


def _show_plotly(fig: Any, *, height: int = 360) -> None:
    if fig is None:
        return
    try:
        fig.update_layout(
            template='plotly_white',
            height=height,
            margin=dict(l=20, r=20, t=60, b=20),
            title=dict(x=0.02, xanchor='left'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1.0,
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(gridcolor='rgba(15,23,42,0.10)', zeroline=False)
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True)


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.0rem;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.5rem 0.95rem;
    }
    .quant-banner {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(15, 23, 42, 0.10);
        margin: 0.25rem 0 1rem 0;
    }
    .quant-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.25rem 0 1rem 0;
    }
    .quant-chip {
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: rgba(248, 250, 252, 0.95);
    }
    .quant-chip-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        margin-bottom: 0.15rem;
        font-weight: 600;
    }
    .quant-chip-value {
        font-size: 1.0rem;
        font-weight: 700;
        color: #0f172a;
    }
    .quant-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 700;
        margin-top: 0.35rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _status_palette(status: str) -> dict[str, str]:
    normalized = str(status or '').strip().lower()
    if normalized in {'healthy', 'normal', 'ready'}:
        return {'bg': '#ECFDF5', 'border': '#10B981', 'text': '#065F46'}
    if normalized in {'caution', 'building'}:
        return {'bg': '#FFFBEB', 'border': '#D97706', 'text': '#92400E'}
    if normalized in {'alert', 'standby', 'waiting'}:
        return {'bg': '#FEF2F2', 'border': '#DC2626', 'text': '#991B1B'}
    return {'bg': '#F8FAFC', 'border': '#CBD5E1', 'text': '#334155'}


def _render_freshness_strip(phase_label: str, latest_report_date: str, live_archive: pd.DataFrame, odds_snapshots: pd.DataFrame) -> None:
    settled_date = 'N/A'
    if not live_archive.empty and 'report_date' in live_archive.columns:
        live_work = live_archive.copy()
        live_work['report_date'] = pd.to_datetime(live_work.get('report_date'), errors='coerce').dt.date
        if 'y_home' in live_work.columns:
            live_work = live_work[pd.to_numeric(live_work.get('y_home'), errors='coerce').notna()].copy()
        settled_series = live_work.get('report_date', pd.Series(dtype=object)).dropna()
        if not settled_series.empty:
            settled_date = settled_series.max().isoformat()

    odds_stamp = 'N/A'
    if not odds_snapshots.empty:
        for col in ['odds_api_pull_ts', 'captured_ts', 'created_ts']:
            if col in odds_snapshots.columns:
                parsed = pd.to_datetime(odds_snapshots.get(col), errors='coerce').dropna()
                if not parsed.empty:
                    odds_stamp = parsed.max().strftime('%Y-%m-%d %I:%M %p')
                    break

    chips = [
        ('Phase View', phase_label),
        ('Latest Report', latest_report_date),
        ('Latest Settled Validation', settled_date),
        ('Latest Odds Snapshot', odds_stamp),
    ]
    html = ['<div class="quant-strip">']
    for label, value in chips:
        html.append(
            f'<div class="quant-chip"><div class="quant-chip-label">{label}</div><div class="quant-chip-value">{value}</div></div>'
        )
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)


def _render_operator_banner(briefing: Dict[str, Any]) -> None:
    palette = _status_palette(str(briefing.get('status') or ''))
    bullets = ''.join(f'<li>{item}</li>' for item in briefing.get('bullets') or [])
    st.markdown(
        f"""
<div class="quant-banner" style="background:{palette['bg']}; border-left: 6px solid {palette['border']};">
  <div class="quant-kicker">Operator Briefing</div>
  <div style="font-size:1.15rem; font-weight:800; color:{palette['text']}; margin:0.2rem 0 0.45rem 0;">
    {briefing.get('status') or 'Normal'} Operating Posture
  </div>
  <div style="color:#0f172a; font-weight:600; margin-bottom:0.5rem;">Top Warning: {briefing.get('top_warning') or 'No major operating warning is active.'}</div>
  <ul style="margin:0.15rem 0 0 1.2rem; color:#1e293b;">
    {bullets}
  </ul>
</div>
        """,
        unsafe_allow_html=True,
    )


def _operator_alerts_frame(
    current_health: pd.DataFrame,
    market_proof_warnings: pd.DataFrame,
    activation_readiness: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not market_proof_warnings.empty:
        for _, row in market_proof_warnings.head(3).iterrows():
            rows.append({
                'Priority': str(row.get('Status') or 'Caution'),
                'Area': str(row.get('Scope') or 'Market'),
                'Message': str(row.get('Plain English') or ''),
            })
    if not current_health.empty:
        for _, row in current_health[current_health['Status'].astype(str).isin(['Caution', 'Building'])].head(3).iterrows():
            rows.append({
                'Priority': str(row.get('Status') or 'Caution'),
                'Area': str(row.get('Domain') or 'System'),
                'Message': str(row.get('Plain English') or ''),
            })
    if not activation_readiness.empty:
        waiting = activation_readiness[activation_readiness['Status'].astype(str).isin(['Waiting', 'Building'])].head(2)
        for _, row in waiting.iterrows():
            rows.append({
                'Priority': str(row.get('Status') or 'Building'),
                'Area': str(row.get('Activation gate') or 'Activation'),
                'Message': str(row.get('Plain English') or ''),
            })
    if not rows:
        rows.append({
            'Priority': 'Normal',
            'Area': 'Operations',
            'Message': 'No immediate dashboard alert is active.',
        })
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _yes_no(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return 'No'
    except Exception:
        pass
    return 'Yes' if bool(value) else 'No'


def _market_policy_overview_frame(latest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for market_key, market_label in [('moneyline', 'Moneyline'), ('total', 'Total'), ('run_line', 'Run line')]:
        policy_name = str(latest.get(f'market_policy_{market_key}_policy') or '').strip()
        posture_name = str(latest.get(f'market_policy_{market_key}_posture') or '').strip()
        evidence_source = str(latest.get(f'market_policy_{market_key}_evidence_source') or '').strip()
        evidence_strength = str(latest.get(f'market_policy_{market_key}_evidence_strength') or '').strip()
        if not any([policy_name, posture_name, evidence_source, evidence_strength]):
            continue
        rows.append({
            'Market': market_label,
            'Policy': policy_name or 'N/A',
            'Posture': posture_name or 'N/A',
            'Evidence source': evidence_source or 'N/A',
            'Evidence strength': evidence_strength or 'N/A',
            'Real-market active': _yes_no(latest.get(f'market_policy_{market_key}_used_real_market')),
            'Policy changed': _yes_no(latest.get(f'market_policy_{market_key}_policy_changed')),
            'Strength changed': _yes_no(latest.get(f'market_policy_{market_key}_strength_changed')),
        })
    return pd.DataFrame(rows)


def _market_readiness_ladder_frame(history: pd.DataFrame, window_label: str = 'Last 14 settled days') -> pd.DataFrame:
    if history.empty or 'market_type' not in history.columns:
        return pd.DataFrame()
    work = history.copy()
    if 'window_label' in work.columns:
        preferred = work[work['window_label'].astype(str) == str(window_label)].copy()
        if not preferred.empty:
            work = preferred
    if 'as_of_report_date' in work.columns:
        work['as_of_report_date'] = pd.to_datetime(work.get('as_of_report_date'), errors='coerce')
    sort_cols = [col for col in ['as_of_report_date', 'run_id'] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    work = work.drop_duplicates(subset=['market_type'], keep='first').reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    def _progress(current: Any, remaining: Any) -> float:
        current_value = float(pd.to_numeric(pd.Series([current]), errors='coerce').fillna(0.0).iloc[0])
        remaining_value = float(pd.to_numeric(pd.Series([remaining]), errors='coerce').fillna(0.0).iloc[0])
        target = current_value + remaining_value
        if target <= 0:
            return 0.0
        return max(0.0, min(current_value / target, 1.0))

    work['slate_progress'] = [
        _progress(cur, rem) for cur, rem in zip(work.get('current_slates', pd.Series(dtype=float)), work.get('remaining_medium_slates', pd.Series(dtype=float)))
    ]
    work['pick_progress'] = [
        _progress(cur, rem) for cur, rem in zip(work.get('current_picks', pd.Series(dtype=float)), work.get('remaining_medium_picks', pd.Series(dtype=float)))
    ]
    work['takeover_progress_pct'] = ((work[['slate_progress', 'pick_progress']].min(axis=1)) * 100.0).round(1)

    readiness_labels = []
    plain_english = []
    for _, row in work.iterrows():
        high_ready = bool(row.get('high_ready')) if pd.notna(row.get('high_ready')) else False
        primary_ready = bool(row.get('primary_ready')) if pd.notna(row.get('primary_ready')) else False
        if high_ready:
            readiness_labels.append('High-confidence ready')
            plain_english.append(f"{row.get('market_label')}: real-market evidence is fully ready.")
        elif primary_ready:
            readiness_labels.append('Primary ready')
            plain_english.append(f"{row.get('market_label')}: ready for real-market takeover, still building toward high-confidence.")
        elif float(row.get('takeover_progress_pct') or 0.0) >= 75.0:
            readiness_labels.append('Close')
            plain_english.append(f"{row.get('market_label')}: close to takeover and likely the next market to graduate.")
        elif float(row.get('takeover_progress_pct') or 0.0) >= 40.0:
            readiness_labels.append('Building')
            plain_english.append(f"{row.get('market_label')}: building evidence, but not close enough to trust on its own yet.")
        else:
            readiness_labels.append('Waiting')
            plain_english.append(f"{row.get('market_label')}: still early and needs a lot more priced-bet evidence.")
    work['readiness_label'] = readiness_labels
    work['plain_english_readiness'] = plain_english
    display = pd.DataFrame({
        'Market': work.get('market_label', pd.Series(dtype=str)),
        'Takeover progress %': work['takeover_progress_pct'],
        'Readiness': work['readiness_label'],
        'Current slates': pd.to_numeric(work.get('current_slates'), errors='coerce').fillna(0).astype(int),
        'Current picks': pd.to_numeric(work.get('current_picks'), errors='coerce').fillna(0).astype(int),
        'Primary threshold': work.get('medium_threshold_label', pd.Series(dtype=str)),
        'High threshold': work.get('high_threshold_label', pd.Series(dtype=str)),
        'To takeover': pd.to_numeric(work.get('remaining_medium_slates'), errors='coerce').fillna(0).astype(int).astype(str) + ' day(s) | ' + pd.to_numeric(work.get('remaining_medium_picks'), errors='coerce').fillna(0).astype(int).astype(str) + ' bets',
        'Plain-English read': work['plain_english_readiness'],
    })
    return display.sort_values(['Takeover progress %', 'Current picks', 'Current slates'], ascending=[False, False, False]).reset_index(drop=True)


def _market_evidence_drift_frame(history: pd.DataFrame, window_label: str = 'Last 14 settled days') -> pd.DataFrame:
    if history.empty or 'market_type' not in history.columns:
        return pd.DataFrame()
    work = history.copy()
    if 'window_label' in work.columns:
        preferred = work[work['window_label'].astype(str) == str(window_label)].copy()
        if not preferred.empty:
            work = preferred
    if 'as_of_report_date' in work.columns:
        work['as_of_report_date'] = pd.to_datetime(work.get('as_of_report_date'), errors='coerce')
    sort_cols = [col for col in ['as_of_report_date', 'run_id'] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    strength_map = {'Low': 1, 'Medium': 2, 'High': 3}
    rows = []
    for market_type, grp in work.groupby('market_type', dropna=False):
        grp = grp.reset_index(drop=True)
        latest = grp.iloc[0]
        previous = grp.iloc[1] if len(grp) > 1 else pd.Series(dtype=object)
        latest_strength = str(latest.get('evidence_strength') or 'Low')
        previous_strength = str(previous.get('evidence_strength') or latest_strength)
        latest_score = strength_map.get(latest_strength, 0)
        previous_score = strength_map.get(previous_strength, latest_score)
        latest_progress = float(_market_readiness_ladder_frame(grp, window_label).get('Takeover progress %', pd.Series([0.0])).iloc[0]) if not grp.empty else 0.0
        prev_progress = latest_progress
        if len(grp) > 1:
            prev_frame = _market_readiness_ladder_frame(grp.iloc[1:].copy(), window_label)
            if not prev_frame.empty:
                prev_progress = float(pd.to_numeric(prev_frame.get('Takeover progress %'), errors='coerce').fillna(0.0).iloc[0])
        progress_delta = round(latest_progress - prev_progress, 1)
        strength_delta = latest_score - previous_score
        if strength_delta > 0 or progress_delta >= 10.0:
            trend = 'Strengthening'
            plain = f"{latest.get('market_label')}: evidence is improving versus the prior stored snapshot."
        elif strength_delta < 0 or progress_delta <= -10.0:
            trend = 'Weakening'
            plain = f"{latest.get('market_label')}: evidence quality slipped versus the prior stored snapshot."
        else:
            trend = 'Stable'
            plain = f"{latest.get('market_label')}: evidence quality is broadly unchanged versus the prior stored snapshot."
        rows.append({
            'Market': latest.get('market_label'),
            'Evidence strength': latest_strength,
            'Real-market active': _yes_no(latest.get('used_real_market')),
            'Takeover progress %': latest_progress,
            'Progress delta': progress_delta,
            'Trend': trend,
            'Primary threshold': latest.get('medium_threshold_label', 'N/A'),
            'High threshold': latest.get('high_threshold_label', 'N/A'),
            'Plain-English drift': plain,
        })
    return pd.DataFrame(rows).sort_values(['Takeover progress %', 'Market'], ascending=[False, True]).reset_index(drop=True)


def _realized_uncertainty_frame(latest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for market_key, market_label in [('moneyline', 'Moneyline'), ('total', 'Total'), ('run_line', 'Run line')]:
        bias = _safe_float(latest.get(f'uncertainty_{market_key}_realized_bias'))
        confidence = _safe_float(latest.get(f'uncertainty_{market_key}_realized_confidence'))
        games = latest.get(f'uncertainty_{market_key}_realized_games')
        bets = latest.get(f'uncertainty_{market_key}_realized_bets')
        avg_clv = _safe_float(latest.get(f'uncertainty_{market_key}_realized_avg_clv'))
        roi = _safe_float(latest.get(f'uncertainty_{market_key}_realized_roi'))
        if bias is None and confidence is None and games in [None, ''] and bets in [None, ''] and avg_clv is None and roi is None:
            continue
        if bias is None:
            learned_view = 'Neutral'
        elif bias > 0.01:
            learned_view = 'Needs a bigger haircut'
        elif bias < -0.01:
            learned_view = 'Can trust a bit more'
        else:
            learned_view = 'Close to neutral'
        rows.append({
            'Market': market_label,
            'Learned bias': _fmt_num(bias, 3),
            'Confidence': _fmt_pct(confidence),
            'Settled games': int(games or 0),
            'Priced bets': int(bets or 0),
            'Avg CLV': _fmt_num(avg_clv, 3),
            'ROI': _fmt_pct(roi),
            'Plain-English read': learned_view,
        })
    return pd.DataFrame(rows)


def _market_proof_warning_frame(latest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    overall_bets = int(_safe_float(latest.get('market_proof_settled_bets')) or 0)
    overall_roi = _safe_float(latest.get('market_proof_roi'))
    overall_clv = _safe_float(latest.get('market_proof_avg_clv'))
    if overall_bets >= 12 and overall_roi is not None and overall_clv is not None:
        if overall_roi > 0 and overall_clv < -0.05:
            rows.append({
                'Scope': 'Overall market proof',
                'Status': 'Caution',
                'Technical': f"ROI is positive ({overall_roi * 100:.1f}%) but avg CLV is negative ({overall_clv:+.3f}).",
                'Plain English': 'The model is making money so far, but the closing market is moving against it too often to trust that profit yet.',
            })
        elif overall_roi < 0 and overall_clv < 0:
            rows.append({
                'Scope': 'Overall market proof',
                'Status': 'Alert',
                'Technical': f"Both ROI ({overall_roi * 100:.1f}%) and avg CLV ({overall_clv:+.3f}) are negative.",
                'Plain English': 'The bets are losing and the close is not validating them. That is a real warning, not just bad luck.',
            })
    for market_key, market_label in [('moneyline', 'Moneyline'), ('total', 'Total'), ('run_line', 'Run line')]:
        bets = int(_safe_float(latest.get(f'market_proof_{market_key}_bets')) or 0)
        roi = _safe_float(latest.get(f'market_proof_{market_key}_roi'))
        clv = _safe_float(latest.get(f'market_proof_{market_key}_avg_clv'))
        if bets < 6 or roi is None or clv is None:
            continue
        if roi > 0 and clv < -0.05:
            rows.append({
                'Scope': market_label,
                'Status': 'Caution',
                'Technical': f"{market_label} ROI is positive ({roi * 100:.1f}%) but avg CLV is negative ({clv:+.3f}).",
                'Plain English': f"{market_label} is profitable so far, but the closing market still disagrees often enough that caution is warranted.",
            })
        elif roi < 0 and clv < 0:
            rows.append({
                'Scope': market_label,
                'Status': 'Alert',
                'Technical': f"{market_label} is losing ({roi * 100:.1f}% ROI) with negative avg CLV ({clv:+.3f}).",
                'Plain English': f"{market_label} has not earned trust yet. It is losing and the close is not backing it up.",
            })
    return pd.DataFrame(rows)


def _current_health_frame(
    latest: Dict[str, Any],
    model_state: Dict[str, Any],
    readiness_ladder: pd.DataFrame,
    market_proof_warnings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    research_log_loss = _safe_float(latest.get('validation_log_loss'))
    live_log_loss = _safe_float(latest.get('live_validation_log_loss'))
    live_games = int(_safe_float(latest.get('live_validation_games')) or 0)
    if live_games <= 0 or live_log_loss is None:
        rows.append({
            'Domain': 'Model',
            'Status': 'Building',
            'Technical': 'No settled live validation sample is available yet.',
            'Plain English': 'The model is live, but it does not have enough finished games to grade itself yet.',
        })
    else:
        delta = None if research_log_loss is None else live_log_loss - research_log_loss
        if delta is not None and delta <= 0.02:
            rows.append({
                'Domain': 'Model',
                'Status': 'Healthy',
                'Technical': f'Live log loss ({live_log_loss:.4f}) is within 0.02 of the research replay baseline ({research_log_loss:.4f}).',
                'Plain English': 'Live moneyline performance is holding close to the research benchmark.',
            })
        else:
            rows.append({
                'Domain': 'Model',
                'Status': 'Caution',
                'Technical': f'Live log loss ({live_log_loss:.4f}) is lagging the research replay baseline ({research_log_loss:.4f}).',
                'Plain English': 'Live results are running weaker than the benchmark and should be watched closely.',
            })

    settled_bets = int(_safe_float(latest.get('market_proof_settled_bets')) or 0)
    overall_clv = _safe_float(latest.get('market_proof_avg_clv'))
    if settled_bets <= 0:
        rows.append({
            'Domain': 'Market',
            'Status': 'Building',
            'Technical': 'No settled priced bets are available yet.',
            'Plain English': 'The market-proof layer is wired, but it cannot judge itself until bets settle.',
        })
    elif not market_proof_warnings.empty:
        rows.append({
            'Domain': 'Market',
            'Status': 'Caution',
            'Technical': 'One or more market-proof warnings are active, driven by weak CLV and/or unstable ROI.',
            'Plain English': 'The market results are not trustworthy enough yet, even if short-run profit is showing up.',
        })
    elif overall_clv is not None and overall_clv >= 0:
        rows.append({
            'Domain': 'Market',
            'Status': 'Healthy',
            'Technical': f'Settled priced-bet evidence is live with non-negative avg CLV ({overall_clv:+.3f}).',
            'Plain English': 'The closing market is at least starting to validate the bets.',
        })
    else:
        rows.append({
            'Domain': 'Market',
            'Status': 'Building',
            'Technical': 'Settled priced-bet evidence exists, but CLV is still not clearly positive.',
            'Plain English': 'There is some proof now, but it is too early to call the market layer strong.',
        })

    allocated_units = _safe_float(latest.get('portfolio_allocated_units'))
    corr_penalty = _safe_float(latest.get('portfolio_avg_correlation_penalty'))
    correlated = int(_safe_float(latest.get('portfolio_correlated_candidate_count')) or 0)
    if allocated_units is None or allocated_units <= 0:
        rows.append({
            'Domain': 'Portfolio',
            'Status': 'Standby',
            'Technical': 'No portfolio has been deployed on the latest run.',
            'Plain English': 'There are no live allocations right now, so risk is parked.',
        })
    elif corr_penalty is not None and corr_penalty < 0.70 and correlated >= 4:
        rows.append({
            'Domain': 'Portfolio',
            'Status': 'Caution',
            'Technical': f'Average correlation penalty is compressed ({corr_penalty:.2f}) with {correlated} correlated candidates.',
            'Plain English': 'The slate is crowded and some ideas overlap too much, so sizing is under pressure.',
        })
    else:
        rows.append({
            'Domain': 'Portfolio',
            'Status': 'Healthy',
            'Technical': f'Allocated {allocated_units:.2f} units with average correlation penalty {corr_penalty if corr_penalty is not None else 1.0:.2f}.',
            'Plain English': 'The portfolio is active and not showing abnormal concentration stress.',
        })

    lineup_profile = (model_state.get('lineup_earn_back') or {}) if isinstance(model_state, dict) else {}
    lineup_games = int(_safe_float(lineup_profile.get('confirmed_games')) or 0)
    lineup_required = int(_safe_float(lineup_profile.get('required_games')) or 0)
    ready_markets = 0 if readiness_ladder.empty else int(readiness_ladder['Readiness'].astype(str).isin(['Primary ready', 'High-confidence ready']).sum())
    if lineup_profile.get('apply') or ready_markets > 0:
        rows.append({
            'Domain': 'Evidence',
            'Status': 'Healthy',
            'Technical': f'Lineup earn-back active={bool(lineup_profile.get("apply"))}; ready markets={ready_markets}.',
            'Plain English': 'At least one activation gate is beginning to turn on.',
        })
    elif lineup_games > 0 or live_games > 0:
        rows.append({
            'Domain': 'Evidence',
            'Status': 'Building',
            'Technical': f'Lineup earn-back progress is {lineup_games}/{lineup_required}; ready markets={ready_markets}.',
            'Plain English': 'The evidence base is growing, but the self-learning gates are still in earn-back mode.',
        })
    else:
        rows.append({
            'Domain': 'Evidence',
            'Status': 'Standby',
            'Technical': 'Activation evidence has not started accumulating yet.',
            'Plain English': 'The long-term self-learning gates are wired but still waiting on sample.',
        })
    return pd.DataFrame(rows)


def _operator_briefing(
    latest: Dict[str, Any],
    phase_label: str,
    actionable_candidates: int,
    current_health: pd.DataFrame,
    readiness_ladder: pd.DataFrame,
    market_proof_warnings: pd.DataFrame,
    evidence_drift: pd.DataFrame,
) -> Dict[str, Any]:
    status = 'Normal'
    top_warning = 'No major operating warning is active.'
    if not market_proof_warnings.empty:
        alert_rows = market_proof_warnings[market_proof_warnings.get('Status', pd.Series(dtype=str)).astype(str) == 'Alert']
        chosen_warning = alert_rows.head(1) if not alert_rows.empty else market_proof_warnings.head(1)
        if not chosen_warning.empty:
            top_warning = str(chosen_warning.iloc[0].get('Plain English') or top_warning)
            status = 'Cautious'

    model_status = 'Building'
    if not current_health.empty and 'Domain' in current_health.columns:
        model_row = current_health[current_health['Domain'].astype(str) == 'Model'].head(1)
        if not model_row.empty:
            model_status = str(model_row.iloc[0].get('Status') or model_status)

    strongest_market = 'N/A'
    if not readiness_ladder.empty and 'Market' in readiness_ladder.columns:
        strongest_market = str(readiness_ladder.iloc[0].get('Market') or strongest_market)

    weakest_market = 'N/A'
    if not evidence_drift.empty and 'Trend' in evidence_drift.columns:
        weakening = evidence_drift[evidence_drift['Trend'].astype(str) == 'Weakening'].head(1)
        if not weakening.empty:
            weakest_market = str(weakening.iloc[0].get('Market') or weakest_market)

    settled_games = int(_safe_float(latest.get('live_validation_games')) or 0)
    live_log_loss = _safe_float(latest.get('live_validation_log_loss'))
    live_accuracy = _safe_float(latest.get('live_validation_accuracy'))

    bullets: list[str] = [
        f'{phase_label} view is active with {actionable_candidates} actionable position(s) on the latest slate.',
    ]
    if settled_games > 0 and live_log_loss is not None and live_accuracy is not None:
        bullets.append(
            f'Live settled moneyline validation is {_fmt_num(live_log_loss, 4)} log loss with {_fmt_pct(live_accuracy)} accuracy across {settled_games} settled game(s).'
        )
    else:
        bullets.append('Live settled validation is still building, so the research replay baseline remains the main benchmark.')
    bullets.append(f'Strongest market by readiness is {strongest_market}.')
    if weakest_market != 'N/A':
        bullets.append(f'Weakest evidence trend right now is {weakest_market}, which deserves extra caution.')
    bullets.append(
        'Operating posture should stay cautious until the current warning clears.'
        if status == 'Cautious'
        else 'Operating posture can stay normal because no major warning is active.'
    )

    return {
        'status': status,
        'model_status': model_status,
        'strongest_market': strongest_market,
        'weakest_market': weakest_market,
        'top_warning': top_warning,
        'bullets': bullets,
    }


def _activation_readiness_frame(model_state: Dict[str, Any], readiness_ladder: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lineup_profile = (model_state.get('lineup_earn_back') or {}) if isinstance(model_state, dict) else {}
    confirmed_games = int(_safe_float(lineup_profile.get('confirmed_games')) or 0)
    required_games = max(int(_safe_float(lineup_profile.get('required_games')) or 0), 1)
    lineup_progress = round(max(0.0, min((confirmed_games / float(required_games)) * 100.0, 100.0)), 1)
    lineup_status = 'Ready' if lineup_profile.get('apply') else ('Building' if confirmed_games > 0 else 'Waiting')
    rows.append({
        'Activation gate': 'Lineup earn-back',
        'Status': lineup_status,
        'Progress %': lineup_progress,
        'Technical': f'Confirmed regular lineup games: {confirmed_games}/{required_games}.',
        'Plain English': str(lineup_profile.get('reason') or 'Lineup evidence is still building.'),
    })
    if readiness_ladder.empty:
        for market_label in ['Moneyline', 'Total', 'Run line']:
            rows.append({
                'Activation gate': f'{market_label} real-market takeover',
                'Status': 'Waiting',
                'Progress %': 0.0,
                'Technical': 'No stored market readiness ladder is available yet.',
                'Plain English': 'This market has not built enough settled priced-bet history to activate on its own yet.',
            })
    else:
        readiness_lookup = {
            str(row.get('Market') or ''): row
            for _, row in readiness_ladder.iterrows()
        }
        for market_label in ['Moneyline', 'Total', 'Run line']:
            row = readiness_lookup.get(market_label, {})
            rows.append({
                'Activation gate': f'{market_label} real-market takeover',
                'Status': str(row.get('Readiness') or 'Waiting'),
                'Progress %': _safe_float(row.get('Takeover progress %')) or 0.0,
                'Technical': f"Thresholds: primary {row.get('Primary threshold') or 'N/A'} | high {row.get('High threshold') or 'N/A'}.",
                'Plain English': str(row.get('Plain-English read') or 'This market is still waiting on enough priced-bet evidence.'),
            })
    return pd.DataFrame(rows)


def _question_examples() -> list[str]:
    return [
        'Why is LAD the strongest edge today?',
        'What day was the model most accurate?',
        'Which market is strongest right now?',
        'Which target is furthest from goal?',
        'What alerts are active right now?',
        'How many settled bets do we have?',
        'Which driver is hurting totals the most?',
        'What changed since yesterday?',
        'How are totals performing lately?',
        'What is the current policy posture?',
        'Build a 2-leg +175 parlay with the highest chance to hit.',
        'Show the 10 best 2-leg parlays over +150 using only moneylines.',
        'Show the top 5 safest parlays today.',
        'What does DraftKings show for LAD today?',
        'What is the best price for LAD moneyline today?',
        'Would LAD still be a bet at -150?',
        'Would LAD -1.5 still be a bet at +110?',
    ]


def _extract_requested_count(question: str, default: int = 10, minimum: int = 1, maximum: int = 15) -> int:
    text = str(question or '').lower()
    digit_match = re.search(r'\b(?:top|show|give me|list)\s+(\d{1,2})\b', text)
    if digit_match:
        return max(minimum, min(maximum, int(digit_match.group(1))))
    word_map = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
    }
    word_match = re.search(r'\b(?:top|show|give me|list)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b', text)
    if word_match:
        return max(minimum, min(maximum, word_map.get(word_match.group(1), default)))
    return max(minimum, min(maximum, int(default)))


def _extract_requested_market(question: str) -> dict[str, Any]:
    lowered = str(question or '').lower()
    market_type = None
    selection_hint = None
    total_side = None
    if any(keyword in lowered for keyword in ['run line', 'runline', 'spread']) or re.search(r'([A-Z]{2,4})\s*[+-]1\.5', str(question or ''), flags=re.IGNORECASE):
        market_type = 'run_line'
    elif any(keyword in lowered for keyword in ['total', 'totals']) or re.search(r'\b(over|under)\s+\d+(\.\d+)?\b', lowered):
        market_type = 'total'
        if 'over' in lowered:
            total_side = 'over'
        elif 'under' in lowered:
            total_side = 'under'
    elif 'moneyline' in lowered or 'moneylines' in lowered or 'ml ' in lowered or ' ml' in lowered:
        market_type = 'moneyline'
    line_match = re.search(r'([A-Z]{2,4})\s*([+-]\d+\.\d+)', str(question or '').upper())
    if line_match:
        selection_hint = f"{line_match.group(1)} {line_match.group(2)}"
        market_type = market_type or 'run_line'
    total_match = re.search(r'\b(over|under)\s+(\d+(\.\d+)?)\b', lowered)
    if total_match:
        selection_hint = f"{total_match.group(1).upper()} {total_match.group(2)}"
        market_type = 'total'
    return {
        'market_type': market_type,
        'selection_hint': selection_hint,
        'total_side': total_side,
    }


def _analytics_metric_specs() -> list[dict[str, Any]]:
    return [
        {'metric': 'live_validation_accuracy', 'label': 'Live Validation Accuracy', 'higher_is_better': True, 'keywords': ['most accurate', 'accuracy', 'accurate'], 'format': 'pct'},
        {'metric': 'live_validation_log_loss', 'label': 'Live Validation Log Loss', 'higher_is_better': False, 'keywords': ['log loss'], 'format': 'num'},
        {'metric': 'live_validation_total_rmse', 'label': 'Live Totals RMSE', 'higher_is_better': False, 'keywords': ['totals rmse', 'total rmse', 'totals error', 'total error'], 'format': 'num'},
        {'metric': 'live_validation_margin_rmse', 'label': 'Live Margin RMSE', 'higher_is_better': False, 'keywords': ['margin rmse', 'spread rmse', 'run line rmse', 'margin error'], 'format': 'num'},
        {'metric': 'market_proof_roi', 'label': 'Market Proof ROI', 'higher_is_better': True, 'keywords': ['roi', 'return'], 'format': 'pct'},
        {'metric': 'market_proof_avg_clv', 'label': 'Average CLV', 'higher_is_better': True, 'keywords': ['clv', 'closing line value'], 'format': 'num'},
        {'metric': 'actionable_market_count', 'label': 'Actionable Markets', 'higher_is_better': True, 'keywords': ['actionable markets', 'actionable picks', 'actionable plays'], 'format': 'int'},
        {'metric': 'must_take_count', 'label': 'Must-Take Plays', 'higher_is_better': True, 'keywords': ['must take', 'must-take'], 'format': 'int'},
        {'metric': 'market_proof_settled_bets', 'label': 'Settled Bets', 'higher_is_better': True, 'keywords': ['settled bets', 'priced bets', 'bets settled'], 'format': 'int'},
        {'metric': 'prediction_count', 'label': 'Predictions', 'higher_is_better': True, 'keywords': ['predictions', 'games today', 'slate size'], 'format': 'int'},
    ]


def _match_metric_spec(question: str) -> dict[str, Any] | None:
    lowered = str(question or '').lower()
    for spec in _analytics_metric_specs():
        if any(keyword in lowered for keyword in spec['keywords']):
            return spec
    return None


def _format_metric_value(metric_spec: dict[str, Any], value: Any) -> str:
    fmt = str(metric_spec.get('format') or 'num')
    if fmt == 'pct':
        return _fmt_pct(value)
    if fmt == 'int':
        numeric = _safe_float(value)
        return 'N/A' if numeric is None else f'{int(round(numeric))}'
    return _fmt_num(value, 4)


def _parse_comparison_direction(question: str) -> str:
    lowered = str(question or '').lower()
    if any(keyword in lowered for keyword in ['worst', 'least', 'lowest']):
        return 'worst'
    return 'best'


def _latest_snapshot_rows(frame: pd.DataFrame, date_col: str, created_col: str | None = None) -> pd.DataFrame:
    work = _frame_or_empty(frame)
    if work.empty or date_col not in work.columns:
        return pd.DataFrame()
    latest_date = pd.to_datetime(work[date_col], errors='coerce').max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    latest = work[pd.to_datetime(work[date_col], errors='coerce') == latest_date].copy()
    if created_col and created_col in latest.columns:
        latest_created = pd.to_datetime(latest[created_col], errors='coerce').max()
        if not pd.isna(latest_created):
            latest = latest[pd.to_datetime(latest[created_col], errors='coerce') == latest_created].copy()
    return latest.reset_index(drop=True)


def _latest_alert_rows(alert_history: pd.DataFrame) -> pd.DataFrame:
    return _latest_snapshot_rows(alert_history, 'as_of_report_date', 'created_ts')


def _latest_target_rows(target_history: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_snapshot_rows(target_history, 'as_of_report_date', 'created_ts')
    if latest.empty:
        return latest
    if 'metric_key' in latest.columns:
        latest = latest.drop_duplicates(subset=['metric_key'], keep='first')
    return latest.reset_index(drop=True)


def _latest_market_proof_rows(market_proof_by_market: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_snapshot_rows(market_proof_by_market, 'as_of_report_date')
    if latest.empty:
        return latest
    if 'market_type' in latest.columns:
        latest = latest.drop_duplicates(subset=['market_type'], keep='first')
    return latest.reset_index(drop=True)


def _question_parser(question: str) -> dict[str, Any]:
    text = str(question or '').strip()
    lowered = text.lower()
    parsed: dict[str, Any] = {
        'raw_question': text,
        'intent': 'unknown',
        'legs': 2,
        'max_results': _extract_requested_count(text, default=10),
        'exclude_same_game': True,
        'market_filters': [],
        'risk_style': 'balanced',
        'target_odds_min': None,
        'target_odds_max': None,
        'target_odds_exact': None,
        'bookmaker': None,
        'price_counterfactual': None,
        'exclude_market_filters': [],
        'requested_market_type': None,
        'selection_hint': None,
        'best_price': False,
        'analytics_metric': None,
        'comparison_direction': 'best',
    }
    if not text:
        return parsed
    market_request = _extract_requested_market(text)
    parsed['requested_market_type'] = market_request.get('market_type')
    parsed['selection_hint'] = market_request.get('selection_hint')
    metric_spec = _match_metric_spec(text)
    if metric_spec:
        parsed['analytics_metric'] = metric_spec.get('metric')
        parsed['comparison_direction'] = _parse_comparison_direction(text)
    bookmaker_aliases = {
        'draftkings': 'DraftKings',
        'fanduel': 'FanDuel',
        'betmgm': 'BetMGM',
        'caesars': 'Caesars',
        'espnbet': 'ESPNBet',
        'espn bet': 'ESPNBet',
        'bally': 'Bally Bet',
        'bet365': 'bet365',
    }
    for alias, label in bookmaker_aliases.items():
        if alias in lowered:
            parsed['bookmaker'] = label
            break
    if any(keyword in lowered for keyword in ['best price', 'best line', 'best number', 'shop for', 'line shop', 'which book has best', 'best available price']):
        parsed['best_price'] = True
    odds_match = re.search(r'(?<!\w)([+-]\d{3,4})(?!\w)', text)
    if odds_match and any(keyword in lowered for keyword in ['would', 'still be', 'if the price', 'at ', 'price moved']):
        parsed['price_counterfactual'] = int(odds_match.group(1))
    if any(keyword in lowered for keyword in ['parlay', 'same game parlay', 'sgp']):
        parsed['intent'] = 'parlay_builder'
        if re.search(r'\b3[\s-]*leg\b|\bthree[\s-]*leg\b', lowered):
            parsed['legs'] = 3
        elif re.search(r'\b2[\s-]*leg\b|\btwo[\s-]*leg\b', lowered):
            parsed['legs'] = 2
        if any(keyword in lowered for keyword in ['highest chance', 'most likely', 'safest', 'safe', 'best chance', 'highest odds of hitting']):
            parsed['risk_style'] = 'safest'
        elif any(keyword in lowered for keyword in ['highest ev', 'best ev', 'highest expected value']):
            parsed['risk_style'] = 'highest_ev'
        else:
            parsed['risk_style'] = 'balanced'
        if 'same game' in lowered and not any(keyword in lowered for keyword in ['no same game', 'avoid same game', 'exclude same game']):
            parsed['exclude_same_game'] = False
        explicit_moneyline = any(keyword in lowered for keyword in ['moneyline', 'moneylines', 'ml only'])
        explicit_runline = any(keyword in lowered for keyword in ['run line', 'runline', 'spread'])
        explicit_total = any(keyword in lowered for keyword in [' total', ' totals', 'total ', 'totals '])
        over_under_market = bool(re.search(r'\b(over|under)\s+\d+(\.\d+)?\b', lowered))
        if any(keyword in lowered for keyword in ['only moneyline', 'only moneylines', 'moneyline only', 'moneylines only']):
            parsed['market_filters'] = ['moneyline']
        elif any(keyword in lowered for keyword in ['only run line', 'only run lines', 'run line only', 'run lines only']):
            parsed['market_filters'] = ['run_line']
        elif any(keyword in lowered for keyword in ['only total', 'only totals', 'totals only', 'total only']):
            parsed['market_filters'] = ['total']
        else:
            if explicit_moneyline:
                parsed['market_filters'].append('moneyline')
            if explicit_runline:
                parsed['market_filters'].append('run_line')
            if explicit_total or over_under_market:
                parsed['market_filters'].append('total')
        odds_match = re.search(r'([+-]\d{3,4})', text)
        if odds_match:
            odds_value = int(odds_match.group(1))
            if any(keyword in lowered for keyword in ['over ', 'at least', 'minimum', 'min ', 'plus at least']):
                parsed['target_odds_min'] = odds_value
            elif any(keyword in lowered for keyword in ['under ', 'below', 'maximum', 'max ', 'no more than']):
                parsed['target_odds_max'] = odds_value
            else:
                parsed['target_odds_exact'] = odds_value
                parsed['target_odds_min'] = odds_value - 20
                parsed['target_odds_max'] = odds_value + 20
        over_match = re.search(r'over\s*([+-]?\d{3,4})', lowered)
        if over_match and parsed['target_odds_min'] is None and parsed['target_odds_exact'] is None:
            parsed['target_odds_min'] = int(over_match.group(1))
        exclusion_patterns = {
            'total': ['no totals', 'without totals', 'exclude totals', 'avoid totals', 'no total', 'without total'],
            'run_line': ['no run line', 'no run lines', 'without run line', 'without run lines', 'exclude run line', 'exclude run lines', 'avoid run line', 'avoid run lines'],
            'moneyline': ['no moneyline', 'no moneylines', 'without moneyline', 'without moneylines', 'exclude moneyline', 'exclude moneylines', 'avoid moneyline', 'avoid moneylines'],
        }
        for market_name, phrases in exclusion_patterns.items():
            if any(phrase in lowered for phrase in phrases):
                parsed['exclude_market_filters'].append(market_name)
        parsed['market_filters'] = sorted(set(parsed['market_filters']))
        parsed['exclude_market_filters'] = sorted(set(parsed['exclude_market_filters']))
        if parsed['exclude_market_filters']:
            parsed['market_filters'] = [market for market in parsed['market_filters'] if market not in parsed['exclude_market_filters']]
        return parsed
    if any(keyword in lowered for keyword in ['what changed', 'since yesterday', 'day over day']):
        parsed['intent'] = 'change_summary'
    elif (
        metric_spec
        and any(keyword in lowered for keyword in ['what day', 'which day', 'best day', 'worst day', 'highest', 'lowest', 'most', 'least'])
    ):
        parsed['intent'] = 'historical_metric_summary'
    elif parsed.get('price_counterfactual') is not None:
        parsed['intent'] = 'price_counterfactual'
    elif parsed.get('best_price'):
        parsed['intent'] = 'best_price_summary'
    elif parsed.get('bookmaker') is not None:
        parsed['intent'] = 'bookmaker_summary'
    elif any(keyword in lowered for keyword in ['alert', 'alerts', 'warning', 'warnings']):
        parsed['intent'] = 'alerts_summary'
    elif any(keyword in lowered for keyword in ['target', 'targets', 'goal', 'goals']):
        parsed['intent'] = 'target_summary'
    elif any(keyword in lowered for keyword in ['driver', 'drivers', 'hurting', 'causing', 'misses', 'missing']):
        parsed['intent'] = 'attribution_summary'
    elif any(keyword in lowered for keyword in ['how many', 'count', 'number of']):
        parsed['intent'] = 'count_summary'
    elif any(keyword in lowered for keyword in ['strongest market', 'weakest market', 'best market', 'worst market', 'which market']):
        parsed['intent'] = 'market_strength_summary'
    elif any(keyword in lowered for keyword in ['validation', 'log loss', 'rmse', 'accuracy', 'how are we doing']):
        parsed['intent'] = 'validation_summary'
    elif any(keyword in lowered for keyword in ['policy', 'posture', 'portfolio posture', 'risk posture']):
        parsed['intent'] = 'policy_summary'
    elif any(keyword in lowered for keyword in ['total', 'totals', 'run line', 'moneyline', 'market performing', 'market proof', 'clv']):
        parsed['intent'] = 'market_summary'
    elif any(keyword in lowered for keyword in ['why', 'explain', 'thesis', 'edge']):
        parsed['intent'] = 'explain_game'
    else:
        parsed['intent'] = 'overview_summary'
    return parsed


def _american_to_decimal(american: Any) -> float | None:
    value = _safe_float(american)
    if value is None or value == 0:
        return None
    if value > 0:
        return 1.0 + (value / 100.0)
    return 1.0 + (100.0 / abs(value))


def _decimal_to_american(decimal_value: Any) -> int | None:
    value = _safe_float(decimal_value)
    if value is None or value <= 1.0:
        return None
    if value >= 2.0:
        return int(round((value - 1.0) * 100.0))
    return int(round(-100.0 / (value - 1.0)))


def _probability_after_uncertainty(probability: Any, uncertainty_penalty: Any, market_throttle: Any = None) -> float:
    p = _safe_float(probability)
    if p is None:
        return 0.5
    penalty = _safe_float(uncertainty_penalty)
    penalty = 1.0 if penalty is None else float(np.clip(penalty, 0.70, 1.02))
    throttle = str(market_throttle or '').strip().lower()
    throttle_factor = 1.0
    if throttle:
        if 'hold' in throttle or 'cap' in throttle or 'watch' in throttle:
            throttle_factor = 0.92
        elif 'reduce' in throttle or 'trim' in throttle:
            throttle_factor = 0.96
    adjusted = 0.5 + ((p - 0.5) * penalty * throttle_factor)
    return float(np.clip(adjusted, 0.01, 0.99))


def _scheduled_ct_timestamp(report_date: Any, start_time_ct: Any) -> pd.Timestamp | None:
    date_value = _coerce_date(report_date)
    if date_value is None:
        return None
    time_text = str(start_time_ct or '').strip().upper()
    if not time_text:
        return None
    time_text = re.sub(r'\s+(CT|CST|CDT)$', '', time_text).strip()
    parsed = pd.to_datetime(f'{date_value.isoformat()} {time_text}', errors='coerce')
    if pd.isna(parsed):
        return None
    try:
        return parsed.tz_localize('America/Chicago')
    except Exception:
        return None


def _pregame_game_keys(predictions: pd.DataFrame) -> set[str] | None:
    if predictions is None or predictions.empty or 'game_pk' not in predictions.columns or 'report_date' not in predictions.columns:
        return None
    work = predictions.copy()
    now_ct = pd.Timestamp.now(tz='America/Chicago')
    work['scheduled_start_ct'] = work.apply(
        lambda row: _scheduled_ct_timestamp(row.get('report_date'), row.get('start_time_ct')),
        axis=1,
    )
    pregame_keys: set[str] = set()
    for _, row in work.iterrows():
        game_pk = row.get('game_pk')
        if pd.isna(game_pk):
            continue
        game_key = str(game_pk)
        scheduled = row.get('scheduled_start_ct')
        report_date = _coerce_date(row.get('report_date'))
        if scheduled is not None:
            if scheduled > now_ct:
                pregame_keys.add(game_key)
            continue
        if report_date is not None and report_date >= now_ct.date():
            pregame_keys.add(game_key)
    return pregame_keys


def _candidate_pool_for_qa(
    candidates: pd.DataFrame,
    portfolio_recommendations: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    exclude_started_games: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not candidates.empty:
        frames.append(candidates.copy())
    if not portfolio_recommendations.empty:
        recs = portfolio_recommendations.copy()
        if 'priority_score' in recs.columns and 'candidate_score' not in recs.columns:
            recs['candidate_score'] = pd.to_numeric(recs.get('priority_score'), errors='coerce')
        frames.append(recs)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True, sort=False)
    for column in ['book_price', 'model_probability', 'edge_value', 'expected_value_per_unit', 'raw_expected_value_per_unit', 'adjusted_expected_value_per_unit', 'uncertainty_penalty', 'actionability_score', 'candidate_score', 'allocated_units']:
        if column in pool.columns:
            pool[column] = pd.to_numeric(pool.get(column), errors='coerce')
    pool['market_type'] = pool.get('market_type', pd.Series(dtype=str)).astype(str).str.lower()
    pool['actionability_label'] = pool.get('actionability_label', pd.Series(dtype=str)).astype(str)
    if 'selection' in pool.columns:
        pool['selection'] = pool['selection'].astype(str)
    pool = pool.drop_duplicates(subset=[col for col in ['game_pk', 'market_type', 'selection', 'book_price'] if col in pool.columns], keep='first')
    if 'book_price' in pool.columns:
        pool = pool[pool['book_price'].notna()].copy()
    if 'model_probability' in pool.columns:
        pool = pool[pool['model_probability'].notna()].copy()
    if 'actionability_label' in pool.columns:
        pool = pool[pool['actionability_label'].isin(['WATCH', 'BET', 'MUST TAKE'])].copy()
    if 'expected_value_per_unit' in pool.columns:
        pool = pool[pd.to_numeric(pool['expected_value_per_unit'], errors='coerce').fillna(0.0) > 0].copy()
    if 'uncertainty_label' in pool.columns:
        pool = pool[pool['uncertainty_label'].astype(str).str.upper().ne('HIGH')].copy()
    if exclude_started_games and 'game_pk' in pool.columns:
        pregame_keys = _pregame_game_keys(_frame_or_empty(predictions))
        if pregame_keys is not None:
            pool = pool[pool['game_pk'].astype(str).isin(pregame_keys)].copy()
    pool['candidate_score'] = pd.to_numeric(pool.get('candidate_score'), errors='coerce').fillna(pd.to_numeric(pool.get('actionability_score'), errors='coerce').fillna(0.0))
    pool['adjusted_probability'] = pool.apply(
        lambda row: _probability_after_uncertainty(row.get('model_probability'), row.get('uncertainty_penalty'), row.get('market_throttle')),
        axis=1,
    )
    pool['decimal_price'] = pool.get('book_price').map(_american_to_decimal)
    pool = pool[pool['decimal_price'].notna()].copy()
    if pool.empty:
        return pool
    pool['leg_label'] = pool.apply(
        lambda row: f"{row.get('away_team')} @ {row.get('home_team')} | {row.get('selection')} ({int(row.get('book_price')):+d})",
        axis=1,
    )
    pool = pool.sort_values(['candidate_score', 'adjusted_expected_value_per_unit', 'adjusted_probability'], ascending=[False, False, False])
    return pool


def _extract_known_teams(question: str, *frames: pd.DataFrame) -> list[str]:
    text = str(question or '').upper()
    tokens = set(re.findall(r'[A-Z]{2,4}', text))
    known: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for col in ['away_team', 'home_team']:
            if col in frame.columns:
                known.update([str(value).upper() for value in frame[col].dropna().astype(str).tolist() if str(value).strip()])
    return sorted([team for team in known if team in tokens])


def _market_type_display(market_type: Any) -> str:
    mapping = {'moneyline': 'Moneyline', 'run_line': 'Run Line', 'total': 'Total'}
    return mapping.get(str(market_type or '').lower(), _pretty_label(market_type))


def _filter_candidates_by_request(pool: pd.DataFrame, question: str, requested_market_type: str | None = None, selection_hint: str | None = None) -> pd.DataFrame:
    if pool.empty:
        return pool
    work = pool.copy()
    teams = _extract_known_teams(question, work)
    if teams:
        work = work[
            work['away_team'].astype(str).str.upper().isin(teams)
            | work['home_team'].astype(str).str.upper().isin(teams)
            | work.get('selection', pd.Series(dtype=str)).astype(str).str.upper().apply(lambda x: any(team in x for team in teams))
        ].copy()
    if requested_market_type:
        work = work[work['market_type'].astype(str).str.lower() == str(requested_market_type).lower()].copy()
    if selection_hint:
        hint = str(selection_hint).upper()
        selection_series = work.get('selection', pd.Series(dtype=str)).astype(str).str.upper()
        if selection_series.str.contains(re.escape(hint), regex=True).any():
            work = work[selection_series.str.contains(re.escape(hint), regex=True)].copy()
    return work


def _parlay_correlation_penalty(legs: list[dict[str, Any]]) -> tuple[float, list[str]]:
    penalty = 0.0
    notes: list[str] = []
    for left, right in itertools.combinations(legs, 2):
        same_game = str(left.get('game_pk')) == str(right.get('game_pk'))
        same_teams = {str(left.get('away_team')), str(left.get('home_team'))} & {str(right.get('away_team')), str(right.get('home_team'))}
        left_selected_raw = left.get('selected_team')
        right_selected_raw = right.get('selected_team')
        left_selected = '' if pd.isna(left_selected_raw) else str(left_selected_raw).strip().upper()
        right_selected = '' if pd.isna(right_selected_raw) else str(right_selected_raw).strip().upper()
        same_selected = bool(left_selected and right_selected and left_selected == right_selected)
        left_market = str(left.get('market_type') or '').lower()
        right_market = str(right.get('market_type') or '').lower()
        if same_game:
            penalty += 0.28
            notes.append('same-game overlap')
            if {left_market, right_market} == {'moneyline', 'run_line'}:
                penalty += 0.10
                notes.append('side/run-line linkage')
            if 'total' in {left_market, right_market}:
                penalty += 0.06
                notes.append('total linkage')
        elif same_selected or same_teams:
            penalty += 0.08
            notes.append('same-team exposure')
    return float(np.clip(penalty, 0.0, 0.45)), sorted(set(notes))


def _build_parlay_candidates(
    parsed: dict[str, Any],
    candidates: pd.DataFrame,
    portfolio_recommendations: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = _candidate_pool_for_qa(
        candidates,
        portfolio_recommendations,
        predictions=predictions,
        exclude_started_games=True,
    )
    meta = {
        'pool_size': int(len(pool)),
        'searched_combinations': 0,
        'filters': [],
        'request': parsed,
    }
    if pool.empty:
        return pd.DataFrame(), meta
    market_filters = parsed.get('market_filters') or []
    if market_filters:
        pool = pool[pool['market_type'].isin(market_filters)].copy()
        meta['filters'].append('market filter')
    exclude_market_filters = parsed.get('exclude_market_filters') or []
    if exclude_market_filters:
        pool = pool[~pool['market_type'].isin(exclude_market_filters)].copy()
        meta['filters'].append('market exclusion filter')
    if parsed.get('exclude_same_game', True):
        meta['filters'].append('no same-game parlays')
    pool_limit = 18 if int(parsed.get('legs') or 2) <= 2 else 12
    pool = pool.head(pool_limit).copy()
    if pool.empty:
        return pd.DataFrame(), meta
    results: list[dict[str, Any]] = []
    legs_requested = int(parsed.get('legs') or 2)
    combo_iter = itertools.combinations(pool.to_dict('records'), legs_requested)
    for combo in combo_iter:
        legs = list(combo)
        meta['searched_combinations'] += 1
        if parsed.get('exclude_same_game', True) and len({str(leg.get('game_pk')) for leg in legs}) < len(legs):
            continue
        decimals = [_american_to_decimal(leg.get('book_price')) for leg in legs]
        if any(value is None for value in decimals):
            continue
        parlay_decimal = float(np.prod(decimals))
        parlay_american = _decimal_to_american(parlay_decimal)
        if parlay_american is None:
            continue
        min_target = parsed.get('target_odds_min')
        max_target = parsed.get('target_odds_max')
        if min_target is not None and parlay_american < int(min_target):
            continue
        if max_target is not None and parlay_american > int(max_target):
            continue
        raw_hit_probability = float(np.prod([float(np.clip(_safe_float(leg.get('model_probability')) or 0.5, 0.01, 0.99)) for leg in legs]))
        adjusted_hit_probability = float(np.prod([float(np.clip(_safe_float(leg.get('adjusted_probability')) or 0.5, 0.01, 0.99)) for leg in legs]))
        correlation_penalty, correlation_notes = _parlay_correlation_penalty(legs)
        adjusted_hit_probability = float(np.clip(adjusted_hit_probability * (1.0 - correlation_penalty), 0.001, 0.99))
        net_payout = parlay_decimal - 1.0
        expected_value = (adjusted_hit_probability * net_payout) - (1.0 - adjusted_hit_probability)
        avg_uncertainty = float(np.mean([_safe_float(leg.get('uncertainty_score')) or 0.0 for leg in legs]))
        avg_edge = float(np.mean([_safe_float(leg.get('edge_value')) or 0.0 for leg in legs]))
        leg_notes = [str(leg.get('leg_label') or '').strip() for leg in legs]
        driver_notes = [str(leg.get('bet_drivers') or '').strip() for leg in legs if str(leg.get('bet_drivers') or '').strip() and str(leg.get('bet_drivers')).strip().lower() != 'none']
        results.append({
            'legs': len(legs),
            'parlay_legs': ' | '.join(leg_notes),
            'book_odds': parlay_american,
            'raw_hit_probability': raw_hit_probability,
            'adjusted_hit_probability': adjusted_hit_probability,
            'expected_value_per_unit': expected_value,
            'average_edge': avg_edge,
            'average_uncertainty_score': avg_uncertainty,
            'correlation_penalty': correlation_penalty,
            'correlation_note': ', '.join(correlation_notes) if correlation_notes else 'low correlation',
            'plain_english': (
                f"Built from {len(legs)} legs with {'low' if correlation_penalty < 0.10 else 'moderate' if correlation_penalty < 0.22 else 'elevated'} overlap risk. "
                f"Adjusted hit probability is {_fmt_pct(adjusted_hit_probability)} at {parlay_american:+d}."
            ),
            'technical_summary': (
                f"Raw hit probability {_fmt_pct(raw_hit_probability)}, adjusted to {_fmt_pct(adjusted_hit_probability)} after "
                f"{correlation_penalty:.2f} correlation drag and leg-level uncertainty haircuts."
            ),
            'supporting_drivers': ' | '.join(driver_notes[:3]) if driver_notes else 'Driver summary not available.',
        })
    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        return result_frame, meta
    risk_style = str(parsed.get('risk_style') or 'balanced')
    if risk_style == 'safest':
        result_frame = result_frame.sort_values(['adjusted_hit_probability', 'expected_value_per_unit', 'correlation_penalty'], ascending=[False, False, True])
    elif risk_style == 'highest_ev':
        result_frame = result_frame.sort_values(['expected_value_per_unit', 'adjusted_hit_probability', 'correlation_penalty'], ascending=[False, False, True])
    else:
        result_frame['balanced_score'] = (
            0.55 * result_frame['adjusted_hit_probability']
            + 0.35 * result_frame['expected_value_per_unit'].clip(lower=-1.0, upper=2.0)
            - 0.10 * result_frame['correlation_penalty']
        )
        result_frame = result_frame.sort_values(['balanced_score', 'adjusted_hit_probability', 'expected_value_per_unit'], ascending=[False, False, False])
    max_results = max(1, min(int(parsed.get('max_results') or 10), 15))
    return result_frame.head(max_results).reset_index(drop=True), meta


def _find_matching_context(question: str, predictions: pd.DataFrame, candidates: pd.DataFrame, portfolio_recommendations: pd.DataFrame) -> dict[str, Any]:
    matched = _extract_known_teams(question, predictions, candidates, portfolio_recommendations)
    out: dict[str, Any] = {'teams': matched}
    if len(matched) >= 2 and not predictions.empty:
        matchup = predictions[
            ((predictions['away_team'].astype(str).str.upper() == matched[0]) & (predictions['home_team'].astype(str).str.upper() == matched[1]))
            | ((predictions['away_team'].astype(str).str.upper() == matched[1]) & (predictions['home_team'].astype(str).str.upper() == matched[0]))
        ].copy()
        if not matchup.empty:
            out['prediction'] = matchup.iloc[0].to_dict()
    if 'prediction' not in out and matched and not predictions.empty:
        team_games = predictions[
            (predictions['away_team'].astype(str).str.upper() == matched[0]) | (predictions['home_team'].astype(str).str.upper() == matched[0])
        ].copy()
        if not team_games.empty:
            out['prediction'] = team_games.sort_values('winner_prob', ascending=False).iloc[0].to_dict()
    if matched and not portfolio_recommendations.empty:
        team_recs = portfolio_recommendations[
            (portfolio_recommendations['away_team'].astype(str).str.upper().isin(matched))
            | (portfolio_recommendations['home_team'].astype(str).str.upper().isin(matched))
            | (portfolio_recommendations.get('selected_team', pd.Series(dtype=str)).astype(str).str.upper().isin(matched))
        ].copy()
        if not team_recs.empty:
            sort_cols = [col for col in ['allocated_units', 'priority_score', 'actionability_score'] if col in team_recs.columns]
            if sort_cols:
                team_recs = team_recs.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            out['recommendation'] = team_recs.iloc[0].to_dict()
    return out


def _best_candidate_match(question: str, candidates: pd.DataFrame, portfolio_recommendations: pd.DataFrame) -> dict[str, Any]:
    context = _find_matching_context(question, pd.DataFrame(), candidates, portfolio_recommendations)
    pool = _candidate_pool_for_qa(candidates, portfolio_recommendations)
    if pool.empty:
        recommendation = context.get('recommendation') or {}
        return recommendation if recommendation else {}
    request = _extract_requested_market(question)
    teams = context.get('teams') or []
    filtered = _filter_candidates_by_request(pool, question, request.get('market_type'), request.get('selection_hint'))
    if not filtered.empty:
        return filtered.sort_values(['candidate_score', 'adjusted_expected_value_per_unit'], ascending=[False, False]).iloc[0].to_dict()
    if teams or request.get('market_type') or request.get('selection_hint'):
        return {}
    recommendation = context.get('recommendation') or {}
    if recommendation:
        return recommendation
    return pool.sort_values(['candidate_score', 'adjusted_expected_value_per_unit'], ascending=[False, False]).iloc[0].to_dict()


def _best_bookmaker_view(question: str, odds_snapshots: pd.DataFrame, bookmaker: str | None) -> pd.DataFrame:
    if odds_snapshots.empty:
        return pd.DataFrame()
    work = odds_snapshots.copy()
    if 'bookmaker_last_update' in work.columns:
        work['bookmaker_last_update'] = pd.to_datetime(work.get('bookmaker_last_update'), errors='coerce')
        work = work.sort_values('bookmaker_last_update', ascending=False)
    if bookmaker:
        work = work[work['sportsbook'].astype(str).str.lower() == str(bookmaker).lower()].copy()
    teams = _extract_known_teams(question, odds_snapshots)
    if teams:
        work = work[
            work['away_team'].astype(str).str.upper().isin(teams)
            | work['home_team'].astype(str).str.upper().isin(teams)
        ].copy()
    if work.empty:
        return pd.DataFrame()
    if 'report_date' in work.columns:
        report_dates = pd.to_datetime(work.get('report_date'), errors='coerce').dropna()
        if not report_dates.empty:
            latest_report_date = report_dates.max().date().isoformat()
            work = work[pd.to_datetime(work.get('report_date'), errors='coerce').dt.date.astype(str) == latest_report_date].copy()
    dedupe_cols = [col for col in ['sportsbook', 'game_pk'] if col in work.columns]
    if dedupe_cols:
        work = work.drop_duplicates(subset=dedupe_cols, keep='first')
    return work.reset_index(drop=True)


def _best_price_view(question: str, odds_snapshots: pd.DataFrame) -> pd.DataFrame:
    if odds_snapshots.empty:
        return pd.DataFrame()
    work = odds_snapshots.copy()
    request = _extract_requested_market(question)
    teams = _extract_known_teams(question, work)
    if teams:
        work = work[
            work['away_team'].astype(str).str.upper().isin(teams)
            | work['home_team'].astype(str).str.upper().isin(teams)
        ].copy()
    if work.empty:
        return pd.DataFrame()
    if 'report_date' in work.columns:
        report_dates = pd.to_datetime(work.get('report_date'), errors='coerce').dropna()
        if not report_dates.empty:
            latest_report_date = report_dates.max().date().isoformat()
            work = work[pd.to_datetime(work.get('report_date'), errors='coerce').dt.date.astype(str) == latest_report_date].copy()
    if work.empty:
        return pd.DataFrame()
    market_type = request.get('market_type') or 'moneyline'
    rows: list[dict[str, Any]] = []
    for game_pk, grp in work.groupby('game_pk', dropna=True):
        first = grp.iloc[0]
        if market_type == 'moneyline':
            sides = []
            away_prices = pd.to_numeric(grp.get('away_ml'), errors='coerce').dropna()
            home_prices = pd.to_numeric(grp.get('home_ml'), errors='coerce').dropna()
            if not away_prices.empty:
                best_away_idx = away_prices.idxmax() if away_prices.max() > 0 else away_prices.idxmax()
                best_away_row = grp.loc[best_away_idx]
                sides.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Away Moneyline',
                    'Selection': first.get('away_team'),
                    'Best Sportsbook': best_away_row.get('sportsbook'),
                    'Best Price': best_away_row.get('away_ml'),
                    'Last Update': best_away_row.get('bookmaker_last_update'),
                })
            if not home_prices.empty:
                best_home_idx = home_prices.idxmax() if home_prices.max() > 0 else home_prices.idxmax()
                best_home_row = grp.loc[best_home_idx]
                sides.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Home Moneyline',
                    'Selection': first.get('home_team'),
                    'Best Sportsbook': best_home_row.get('sportsbook'),
                    'Best Price': best_home_row.get('home_ml'),
                    'Last Update': best_home_row.get('bookmaker_last_update'),
                })
            rows.extend(sides)
        elif market_type == 'run_line':
            away_prices = pd.to_numeric(grp.get('away_run_line_price'), errors='coerce').dropna()
            home_prices = pd.to_numeric(grp.get('home_run_line_price'), errors='coerce').dropna()
            if not away_prices.empty:
                idx = away_prices.idxmax()
                row = grp.loc[idx]
                rows.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Away Run Line',
                    'Selection': f"{first.get('away_team')} {row.get('away_run_line'):+.1f}",
                    'Best Sportsbook': row.get('sportsbook'),
                    'Best Price': row.get('away_run_line_price'),
                    'Last Update': row.get('bookmaker_last_update'),
                })
            if not home_prices.empty:
                idx = home_prices.idxmax()
                row = grp.loc[idx]
                rows.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Home Run Line',
                    'Selection': f"{first.get('home_team')} {row.get('home_run_line'):+.1f}",
                    'Best Sportsbook': row.get('sportsbook'),
                    'Best Price': row.get('home_run_line_price'),
                    'Last Update': row.get('bookmaker_last_update'),
                })
        else:
            over_prices = pd.to_numeric(grp.get('over_price'), errors='coerce').dropna()
            under_prices = pd.to_numeric(grp.get('under_price'), errors='coerce').dropna()
            total_line = _safe_float(first.get('total_line'))
            if not over_prices.empty:
                idx = over_prices.idxmax()
                row = grp.loc[idx]
                rows.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Total',
                    'Selection': f"OVER {total_line:.1f}" if total_line is not None else 'OVER',
                    'Best Sportsbook': row.get('sportsbook'),
                    'Best Price': row.get('over_price'),
                    'Last Update': row.get('bookmaker_last_update'),
                })
            if not under_prices.empty:
                idx = under_prices.idxmax()
                row = grp.loc[idx]
                rows.append({
                    'Matchup': f"{first.get('away_team')} @ {first.get('home_team')}",
                    'Market': 'Total',
                    'Selection': f"UNDER {total_line:.1f}" if total_line is not None else 'UNDER',
                    'Best Sportsbook': row.get('sportsbook'),
                    'Best Price': row.get('under_price'),
                    'Last Update': row.get('bookmaker_last_update'),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    selection_hint = request.get('selection_hint')
    if teams and len(teams) == 1:
        team = teams[0]
        if result['Selection'].astype(str).str.upper().str.contains(re.escape(team), regex=True).any():
            result = result[result['Selection'].astype(str).str.upper().str.contains(re.escape(team), regex=True)].copy()
    if selection_hint:
        hint = str(selection_hint).upper()
        if result['Selection'].astype(str).str.upper().str.contains(re.escape(hint), regex=True).any():
            result = result[result['Selection'].astype(str).str.upper().str.contains(re.escape(hint), regex=True)].copy()
    return result.sort_values(['Matchup', 'Market', 'Best Price'], ascending=[True, True, False]).reset_index(drop=True)


def _historical_metric_lookup(question: str, parsed: dict[str, Any], run_history: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Historical Metric Lookup', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    metric_spec = _match_metric_spec(question)
    if metric_spec is None:
        response['summary'] = 'The question did not map cleanly to a stored historical metric.'
        return response
    metric_key = str(metric_spec['metric'])
    label = str(metric_spec['label'])
    higher_is_better = bool(metric_spec['higher_is_better'])
    history = _frame_or_empty(run_history)
    if history.empty or metric_key not in history.columns:
        response['summary'] = f'No historical {label.lower()} series is available in the stored run history.'
        return response
    work = history[['report_date', metric_key]].copy()
    for extra_col in ['live_validation_games', 'market_proof_settled_bets', 'actionable_market_count', 'must_take_count']:
        if extra_col in history.columns:
            work[extra_col] = history[extra_col]
    work[metric_key] = pd.to_numeric(work[metric_key], errors='coerce')
    work['report_date'] = pd.to_datetime(work['report_date'], errors='coerce')
    work = work.dropna(subset=['report_date', metric_key]).sort_values('report_date').reset_index(drop=True)
    if work.empty:
        response['summary'] = f'There are no non-null stored values for {label.lower()} yet.'
        return response
    best_row = work.sort_values(metric_key, ascending=not higher_is_better).iloc[0]
    worst_row = work.sort_values(metric_key, ascending=higher_is_better).iloc[0]
    direction = str(parsed.get('comparison_direction') or 'best')
    selected_row = best_row if direction == 'best' else worst_row
    response['summary'] = f"{label} was {'best' if direction == 'best' else 'worst'} on {_fmt_date(selected_row.get('report_date'))} at {_format_metric_value(metric_spec, selected_row.get(metric_key))}."
    response['bullets'] = [
        f"Best day: {_fmt_date(best_row.get('report_date'))} at {_format_metric_value(metric_spec, best_row.get(metric_key))}.",
        f"Worst day: {_fmt_date(worst_row.get('report_date'))} at {_format_metric_value(metric_spec, worst_row.get(metric_key))}.",
        f"Sample size: {int(len(work))} daily snapshot(s).",
    ]
    display = work.sort_values(metric_key, ascending=not higher_is_better).head(5).copy()
    response['table'] = _display_frame(display, {
        'report_date': 'Report Date',
        metric_key: label,
        'live_validation_games': 'Live Validation Games',
        'market_proof_settled_bets': 'Settled Bets',
        'actionable_market_count': 'Actionable Markets',
        'must_take_count': 'Must-Take Plays',
    })
    response['table_label'] = f'Top Historical Days for {label}'
    return response


def _market_strength_answer(question: str, market_proof_by_market: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Market Strength Summary', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    proof = _latest_market_proof_rows(market_proof_by_market)
    if proof.empty:
        response['summary'] = 'No current market-proof snapshot is available yet.'
        return response
    lowered = str(question or '').lower()
    ranking_metric = 'roi'
    metric_label = 'ROI'
    metric_formatter = _fmt_pct
    if 'clv' in lowered:
        ranking_metric = 'avg_clv'
        metric_label = 'Average CLV'
        metric_formatter = lambda value: _fmt_num(value, 3)
    proof[ranking_metric] = pd.to_numeric(proof[ranking_metric], errors='coerce')
    proof = proof.dropna(subset=[ranking_metric]).copy()
    if proof.empty:
        response['summary'] = f'No stored {metric_label.lower()} values are available for the current market-proof snapshot.'
        return response
    strongest = proof.sort_values(ranking_metric, ascending=False).iloc[0]
    weakest = proof.sort_values(ranking_metric, ascending=True).iloc[0]
    response['summary'] = f"The strongest market right now is {_market_type_display(strongest.get('market_type'))} by {metric_label} at {metric_formatter(strongest.get(ranking_metric))}."
    response['bullets'] = [
        f"Weakest market: {_market_type_display(weakest.get('market_type'))} at {metric_formatter(weakest.get(ranking_metric))}.",
        f"Moneyline bets: {int(_safe_float(proof.loc[proof['market_type'].astype(str).str.lower() == 'moneyline', 'bets'].sum()) or 0)} | Run line bets: {int(_safe_float(proof.loc[proof['market_type'].astype(str).str.lower() == 'run_line', 'bets'].sum()) or 0)} | Totals bets: {int(_safe_float(proof.loc[proof['market_type'].astype(str).str.lower() == 'total', 'bets'].sum()) or 0)}.",
    ]
    response['table'] = _display_frame(proof[['market_type', 'bets', 'units', 'roi', 'avg_clv']].copy(), {
        'market_type': 'Market',
        'bets': 'Bets',
        'units': 'Units',
        'roi': 'ROI',
        'avg_clv': 'Average CLV',
    })
    response['table_label'] = 'Current Market-Proof Ranking'
    return response


def _alerts_answer(alert_history: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Active Alerts', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    alerts = _latest_alert_rows(alert_history)
    if alerts.empty:
        response['summary'] = 'No stored alerts are active for the current phase view.'
        return response
    severity_rank = {'Critical': 3, 'Warning': 2, 'Note': 1, 'Info': 0}
    alerts = alerts.copy()
    alerts['severity_rank'] = alerts['severity'].astype(str).map(severity_rank).fillna(0)
    top_alert = alerts.sort_values(['severity_rank', 'category'], ascending=[False, True]).iloc[0]
    response['summary'] = f"The highest-severity active alert is {top_alert.get('severity')} in {top_alert.get('category')}: {top_alert.get('title')}."
    response['bullets'] = [
        f"Technical: {top_alert.get('technical')}.",
        f"Plain English: {top_alert.get('plain_english')}.",
        f"Alert count: {int(len(alerts))} on the latest stored report date.",
    ]
    response['table'] = _display_frame(alerts[['severity', 'category', 'title', 'plain_english']].copy(), {
        'severity': 'Severity',
        'category': 'Category',
        'title': 'Title',
        'plain_english': 'Plain-English Summary',
    })
    response['table_label'] = 'Latest Active Alerts'
    return response


def _target_answer(question: str, target_history: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Target Status Summary', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    targets = _latest_target_rows(target_history)
    if targets.empty:
        response['summary'] = 'No dynamic target snapshot is stored yet for the current phase view.'
        return response
    targets = targets.copy()
    targets['gap_value'] = pd.to_numeric(targets['gap_value'], errors='coerce')
    lowered = str(question or '').lower()
    if any(keyword in lowered for keyword in ['furthest', 'behind', 'miss']):
        selected = targets.sort_values('gap_value', ascending=True).iloc[0]
        response['summary'] = f"The target furthest from goal is {selected.get('metric_label')} at a gap of {_fmt_num(selected.get('gap_value'), 3)}."
    elif any(keyword in lowered for keyword in ['ahead', 'beating', 'best target']):
        selected = targets.sort_values('gap_value', ascending=False).iloc[0]
        response['summary'] = f"The strongest target line right now is {selected.get('metric_label')} at a positive gap of {_fmt_num(selected.get('gap_value'), 3)}."
    else:
        ahead = int((targets['status'].astype(str) == 'Ahead of target').sum())
        behind = int((targets['status'].astype(str) == 'Behind target').sum())
        response['summary'] = f"{ahead} tracked metrics are ahead of target and {behind} are behind target in the latest stored snapshot."
        selected = targets.sort_values('gap_value', ascending=True).iloc[0]
    response['bullets'] = [
        f"Most behind: {targets.sort_values('gap_value', ascending=True).iloc[0].get('metric_label')} ({_fmt_num(targets.sort_values('gap_value', ascending=True).iloc[0].get('gap_value'), 3)}).",
        f"Most ahead: {targets.sort_values('gap_value', ascending=False).iloc[0].get('metric_label')} ({_fmt_num(targets.sort_values('gap_value', ascending=False).iloc[0].get('gap_value'), 3)}).",
        f"Evidence progress: {_fmt_pct(selected.get('progress_pct'))}.",
    ]
    response['table'] = _display_frame(targets[['metric_label', 'current_value', 'dynamic_target_value', 'baseline_value', 'long_run_goal_value', 'status', 'gap_value']].copy(), {
        'metric_label': 'Metric',
        'current_value': 'Current',
        'dynamic_target_value': 'Dynamic Target',
        'baseline_value': 'Baseline',
        'long_run_goal_value': 'Long-Run Goal',
        'status': 'Status',
        'gap_value': 'Gap',
    })
    response['table_label'] = 'Latest Dynamic Targets'
    return response


def _attribution_answer(question: str, attribution_driver_mart: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Attribution Summary', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    mart = _frame_or_empty(attribution_driver_mart)
    if mart.empty:
        response['summary'] = 'No attribution mart is stored yet for the current phase view.'
        return response
    market_request = _extract_requested_market(question)
    market_type = market_request.get('market_type')
    work = mart.copy()
    if market_type:
        work = work[work['market_type'].astype(str).str.lower() == str(market_type).lower()].copy()
    work['rmse'] = pd.to_numeric(work['rmse'], errors='coerce')
    work['avg_abs_error'] = pd.to_numeric(work['avg_abs_error'], errors='coerce')
    work['driver_state_rank'] = work.get('driver_state', pd.Series(dtype=str)).astype(str).str.lower().map({'flagged': 1, 'clean': 0}).fillna(0)
    work = work.dropna(subset=['rmse']).sort_values(['driver_state_rank', 'rmse', 'games'], ascending=[False, False, False]).reset_index(drop=True)
    if work.empty:
        response['summary'] = 'No attribution rows match that question in the current phase view.'
        return response
    top = work.iloc[0]
    market_label = _market_type_display(top.get('market_type'))
    response['summary'] = f"The biggest stored miss driver for {market_label} is {top.get('driver')} ({top.get('driver_state')}) with RMSE {_fmt_num(top.get('rmse'), 3)}."
    response['bullets'] = [
        f"Average absolute error: {_fmt_num(top.get('avg_abs_error'), 3)} across {int(_safe_float(top.get('games')) or 0)} games.",
        'This attribution view is descriptive, not causal certainty, but it is the strongest stored signal about where misses are clustering.',
    ]
    response['table'] = _display_frame(work[['market_type', 'driver', 'driver_state', 'games', 'avg_abs_error', 'rmse']].copy(), {
        'market_type': 'Market',
        'driver': 'Driver',
        'driver_state': 'Driver State',
        'games': 'Games',
        'avg_abs_error': 'Average Absolute Error',
        'rmse': 'RMSE',
    })
    response['table_label'] = 'Attribution Driver Ranking'
    return response


def _count_answer(question: str, latest: dict[str, Any], alert_history: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Count Summary', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': ''}
    lowered = str(question or '').lower()
    metric_label = 'Count'
    metric_value: int | None = None
    if 'alert' in lowered or 'warning' in lowered:
        latest_alerts = _latest_alert_rows(alert_history)
        metric_label = 'Active alerts'
        metric_value = int(len(latest_alerts))
    elif 'settled bet' in lowered or 'priced bet' in lowered or 'bets' in lowered:
        metric_label = 'Settled priced bets'
        metric_value = int(_safe_float(latest.get('market_proof_settled_bets')) or 0)
    elif 'settled day' in lowered or 'days' in lowered:
        metric_label = 'Settled live days'
        metric_value = int(_safe_float(latest.get('live_validation_settled_days')) or 0)
    elif 'live game' in lowered or 'settled game' in lowered or 'games' in lowered:
        metric_label = 'Settled live games'
        metric_value = int(_safe_float(latest.get('live_validation_games')) or 0)
    elif 'must take' in lowered:
        metric_label = 'Must-take plays'
        metric_value = int(_safe_float(latest.get('must_take_count')) or 0)
    elif 'actionable' in lowered:
        metric_label = 'Actionable markets'
        metric_value = int(_safe_float(latest.get('actionable_market_count')) or 0)
    elif 'prediction' in lowered or 'slate' in lowered:
        metric_label = 'Predictions on the slate'
        metric_value = int(len(_frame_or_empty(predictions)))
    if metric_value is None:
        response['summary'] = 'The count question did not map cleanly to a stored model count.'
        return response
    response['summary'] = f"{metric_label}: {metric_value}."
    response['bullets'] = [
        f"Phase-scoped answer: this uses the latest stored snapshot for the current phase view.",
    ]
    response['table'] = pd.DataFrame([{'Metric': metric_label, 'Value': metric_value}])
    response['table_label'] = 'Count Detail'
    return response


def _answer_model_question(
    question: str,
    parsed: dict[str, Any],
    latest: dict[str, Any],
    run_history: pd.DataFrame,
    daily_kpi_snapshot: pd.DataFrame,
    rolling_kpi_windows: pd.DataFrame,
    alert_history: pd.DataFrame,
    target_history: pd.DataFrame,
    attribution_market_mart: pd.DataFrame,
    attribution_driver_mart: pd.DataFrame,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    portfolio_recommendations: pd.DataFrame,
    market_proof_by_market_bundle: pd.DataFrame,
    odds_snapshots: pd.DataFrame,
    selected_phase_label: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {'title': 'Ask the Model', 'summary': '', 'bullets': [], 'table': pd.DataFrame(), 'table_label': '', 'intent': parsed.get('intent')}
    intent = str(parsed.get('intent') or 'unknown')
    if intent == 'parlay_builder':
        parlays, meta = _build_parlay_candidates(parsed, candidates, portfolio_recommendations, predictions)
        response['title'] = 'Parlay Builder'
        if parlays.empty:
            response['summary'] = 'No qualifying parlays were found for the requested filters.'
            response['bullets'] = [
                f"Phase view: {selected_phase_label}.",
                f"Candidate pool searched: {meta.get('pool_size', 0)} eligible legs across {meta.get('searched_combinations', 0)} combinations.",
                'Games that have already started were excluded before ranking parlays.',
                'Try widening the odds band, allowing more markets, or asking for a balanced build instead of the safest only.',
            ]
            return response
        best = parlays.iloc[0].to_dict()
        response['summary'] = (
            f"The strongest match for this request is a {int(best.get('legs') or 0)}-leg parlay at {int(best.get('book_odds') or 0):+d} "
            f"with an adjusted hit probability of {_fmt_pct(best.get('adjusted_hit_probability'))}."
        )
        response['bullets'] = [
            f"Search mode: {str(parsed.get('risk_style') or 'balanced').replace('_', ' ')}.",
            f"Phase view: {selected_phase_label}; candidate legs searched: {meta.get('pool_size', 0)}; combinations scored: {meta.get('searched_combinations', 0)}.",
            'Already-started games were excluded before scoring combinations.',
            f"Same-game parlays {'excluded' if parsed.get('exclude_same_game', True) else 'allowed'}; included markets: {', '.join([_market_type_display(x) for x in parsed.get('market_filters') or []]) or 'All supported markets'}.",
            f"Excluded markets: {', '.join([_market_type_display(x) for x in parsed.get('exclude_market_filters') or []]) or 'None'}.",
        ]
        response['table'] = _display_frame(parlays, {
            'parlay_legs': 'Parlay Legs',
            'book_odds': 'Book Odds',
            'raw_hit_probability': 'Raw Hit Probability',
            'adjusted_hit_probability': 'Adjusted Hit Probability',
            'expected_value_per_unit': 'Expected Value / Unit',
            'average_edge': 'Average Leg Edge',
            'average_uncertainty_score': 'Average Uncertainty',
            'correlation_penalty': 'Correlation Penalty',
            'correlation_note': 'Correlation Notes',
            'plain_english': 'Plain-English Summary',
            'technical_summary': 'Technical Summary',
            'supporting_drivers': 'Supporting Drivers',
        })
        response['table_label'] = 'Top Parlay Combinations'
        return response
    if intent == 'historical_metric_summary':
        return _historical_metric_lookup(question, parsed, run_history)
    if intent == 'market_strength_summary':
        return _market_strength_answer(question, market_proof_by_market_bundle)
    if intent == 'alerts_summary':
        return _alerts_answer(alert_history)
    if intent == 'target_summary':
        return _target_answer(question, target_history)
    if intent == 'attribution_summary':
        return _attribution_answer(question, attribution_driver_mart)
    if intent == 'count_summary':
        return _count_answer(question, latest, alert_history, predictions)
    if intent == 'best_price_summary':
        best_price = _best_price_view(question, odds_snapshots)
        response['title'] = 'Best Available Price'
        if best_price.empty:
            response['summary'] = 'No matching line-shopping view could be built from the current odds snapshots.'
            response['bullets'] = [
                'Try naming a team, matchup, and optionally the market, for example “What is the best price for LAD moneyline today?”',
                'Supported markets here are moneyline, run line, and totals.',
            ]
            return response
        first = best_price.iloc[0].to_dict()
        response['summary'] = (
            f"The best currently stored price is {first.get('Selection')} at {int(_safe_float(first.get('Best Price')) or 0):+d} "
            f"from {first.get('Best Sportsbook')}."
        )
        response['bullets'] = [
            f"Matchup: {first.get('Matchup')}.",
            f"Market: {first.get('Market')}.",
            'This is a line-shopping answer from the latest mirrored sportsbook snapshots, not a model recommendation by itself.',
        ]
        response['table'] = best_price
        response['table_label'] = 'Best Available Prices'
        return response
    if intent == 'bookmaker_summary':
        bookmaker = parsed.get('bookmaker')
        book_view = _best_bookmaker_view(question, odds_snapshots, bookmaker)
        response['title'] = 'Bookmaker Line Summary'
        if book_view.empty:
            response['summary'] = 'No matching bookmaker snapshot was found for that question in the current phase view.'
            response['bullets'] = [
                f"Requested bookmaker: {bookmaker or 'Any available book'}.",
                'Try naming a team or matchup alongside the bookmaker, for example “What does DraftKings show for LAD today?”',
            ]
            return response
        first = book_view.iloc[0].to_dict()
        response['summary'] = (
            f"{first.get('sportsbook')} currently shows {first.get('away_team')} @ {first.get('home_team')} with "
            f"{first.get('away_team')} {int(_safe_float(first.get('away_ml')) or 0):+d} and {first.get('home_team')} {int(_safe_float(first.get('home_ml')) or 0):+d}."
        )
        response['bullets'] = [
            f"Run line: {first.get('away_team')} {_fmt_num(first.get('away_run_line'), 1)} ({int(_safe_float(first.get('away_run_line_price')) or 0):+d}) | "
            f"{first.get('home_team')} {_fmt_num(first.get('home_run_line'), 1)} ({int(_safe_float(first.get('home_run_line_price')) or 0):+d}).",
            f"Total: {_fmt_num(first.get('total_line'), 1)} | Over {int(_safe_float(first.get('over_price')) or 0):+d} | Under {int(_safe_float(first.get('under_price')) or 0):+d}.",
            f"Last bookmaker update: {str(first.get('bookmaker_last_update') or 'N/A')}.",
        ]
        response['table'] = _display_frame(book_view[[
            col for col in ['sportsbook', 'away_team', 'home_team', 'away_ml', 'home_ml', 'away_run_line', 'away_run_line_price', 'home_run_line', 'home_run_line_price', 'total_line', 'over_price', 'under_price', 'bookmaker_last_update']
            if col in book_view.columns
        ]], {
            'sportsbook': 'Sportsbook',
            'away_team': 'Away Team',
            'home_team': 'Home Team',
            'away_ml': 'Away Moneyline',
            'home_ml': 'Home Moneyline',
            'away_run_line': 'Away Run Line',
            'away_run_line_price': 'Away Run-Line Price',
            'home_run_line': 'Home Run Line',
            'home_run_line_price': 'Home Run-Line Price',
            'total_line': 'Total',
            'over_price': 'Over Price',
            'under_price': 'Under Price',
            'bookmaker_last_update': 'Last Update',
        })
        response['table_label'] = 'Bookmaker Snapshot'
        return response
    if intent == 'price_counterfactual':
        candidate = _best_candidate_match(question, candidates, portfolio_recommendations)
        target_price = parsed.get('price_counterfactual')
        response['title'] = 'Price Counterfactual'
        if not candidate or target_price is None:
            request = _extract_requested_market(question)
            requested_market = _market_type_display(request.get('market_type')) if request.get('market_type') else 'market'
            response['summary'] = f'There is no active stored {requested_market.lower()} candidate matching that question, so the counterfactual cannot be grounded cleanly.'
            response['bullets'] = ['Try naming a team or matchup and a specific American price, for example “Would LAD still be a bet at -150?”']
            return response
        fair_price = _safe_float(candidate.get('fair_price'))
        book_price = _safe_float(candidate.get('book_price'))
        model_probability = float(np.clip(_safe_float(candidate.get('model_probability')) or 0.5, 0.01, 0.99))
        target_decimal = _american_to_decimal(target_price)
        if fair_price is None or target_decimal is None:
            response['summary'] = 'The stored candidate does not have enough pricing detail to answer this counterfactual cleanly.'
            return response
        implied_probability = 1.0 / target_decimal
        counterfactual_ev = (model_probability * (target_decimal - 1.0)) - (1.0 - model_probability)
        current_ev = _safe_float(candidate.get('expected_value_per_unit'))
        current_actionability = str(candidate.get('actionability_label') or 'N/A')
        fair_decimal = _american_to_decimal(fair_price)
        fair_probability = None if fair_decimal is None else 1.0 / fair_decimal
        edge_at_target = None if fair_probability is None else model_probability - implied_probability
        current_implied_probability = None if book_price is None else (1.0 / _american_to_decimal(book_price)) if _american_to_decimal(book_price) is not None else None
        edge_delta = None if edge_at_target is None or current_implied_probability is None else edge_at_target - (model_probability - current_implied_probability)
        actionability_at_target = 'Pass'
        if counterfactual_ev > 0.10:
            actionability_at_target = 'Must Take Range'
        elif counterfactual_ev > 0.04:
            actionability_at_target = 'Bet Range'
        elif counterfactual_ev > 0:
            actionability_at_target = 'Watch Range'
        status = 'still a bet'
        if counterfactual_ev <= 0:
            status = 'not a bet anymore'
        elif counterfactual_ev < 0.03:
            status = 'still playable, but much thinner'
        response['summary'] = (
            f"At {int(target_price):+d}, this would be {status}. "
            f"The model probability is {_fmt_pct(model_probability)}, the fair price is {int(fair_price):+d}, and the counterfactual EV is {_fmt_num(counterfactual_ev, 3)} per unit."
        )
        response['bullets'] = [
            f"Current stored price: {int(book_price or 0):+d} | current EV: {_fmt_num(current_ev, 3)} | current actionability: {current_actionability}.",
            f"Counterfactual implied probability: {_fmt_pct(implied_probability)} | fair implied probability: {_fmt_pct(fair_probability)}.",
            f"Selection: {candidate.get('selection')} in {candidate.get('away_team')} @ {candidate.get('home_team')} ({_market_type_display(candidate.get('market_type'))}) | projected actionability at target price: {actionability_at_target}.",
        ]
        response['table'] = _display_frame(pd.DataFrame([{
            'matchup': f"{candidate.get('away_team')} @ {candidate.get('home_team')}",
            'market': _market_type_display(candidate.get('market_type')),
            'selection': candidate.get('selection'),
            'current_book_price': book_price,
            'fair_price': fair_price,
            'target_price': target_price,
            'model_probability': model_probability,
            'counterfactual_edge': edge_at_target,
            'edge_delta': edge_delta,
            'counterfactual_ev': counterfactual_ev,
            'current_actionability': current_actionability,
            'counterfactual_actionability': actionability_at_target,
        }]), {
            'matchup': 'Matchup',
            'market': 'Market',
            'selection': 'Selection',
            'current_book_price': 'Current Book Price',
            'fair_price': 'Fair Price',
            'target_price': 'Counterfactual Price',
            'model_probability': 'Model Probability',
            'counterfactual_edge': 'Counterfactual Edge',
            'edge_delta': 'Edge Change vs Current',
            'counterfactual_ev': 'Counterfactual EV / Unit',
            'current_actionability': 'Current Actionability',
            'counterfactual_actionability': 'Counterfactual Actionability',
        })
        response['table_label'] = 'Price Counterfactual Detail'
        return response
    if intent == 'change_summary':
        change_frame = _yesterday_change_frame(run_history)
        response['title'] = 'Day-over-Day Change Summary'
        if change_frame.empty:
            response['summary'] = 'There is not enough run history yet to compare today against the prior report date.'
            return response
        response['summary'] = 'The latest stored run has a complete day-over-day change summary available.'
        response['bullets'] = [f"{row['Metric']}: {row['Change']}" for _, row in change_frame.head(5).iterrows()]
        response['table'] = _display_frame(change_frame)
        response['table_label'] = 'What Changed Since Yesterday'
        return response
    if intent == 'validation_summary':
        response['title'] = 'Validation Summary'
        response['summary'] = (
            f"Research replay is at log loss {_fmt_num(latest.get('validation_log_loss'), 4)} and live settled regular-season validation is at "
            f"log loss {_fmt_num(latest.get('live_validation_log_loss'), 4)} over {int(latest.get('live_validation_games') or 0)} games."
        )
        response['bullets'] = [
            f"Research replay accuracy: {_fmt_pct(latest.get('validation_accuracy'))}.",
            f"Live settled accuracy: {_fmt_pct(latest.get('live_validation_accuracy'))}.",
            f"Live totals RMSE: {_fmt_num(latest.get('live_validation_total_rmse'), 3)}; live margin RMSE: {_fmt_num(latest.get('live_validation_margin_rmse'), 3)}.",
        ]
        response['table'] = _display_frame(_validation_delta_frame(latest))
        response['table_label'] = 'Research vs Live Validation'
        return response
    if intent == 'policy_summary':
        response['title'] = 'Policy and Posture Summary'
        response['summary'] = str(latest.get('policy_research_plain_english') or latest.get('market_policy_summary_plain_english') or 'The stored run does not yet have a narrative policy summary.')
        response['bullets'] = [
            'Technical: ' + str(latest.get('policy_research_technical') or latest.get('market_policy_summary_technical') or 'No technical policy note is stored.'),
            f"Canonical policy: {latest.get('policy_research_policy_name') or 'N/A'} | posture: {latest.get('policy_research_posture_name') or 'N/A'}.",
            f"Alignment status: {latest.get('market_policy_alignment_status') or 'N/A'} | real-market active: {int(latest.get('market_policy_real_market_count') or 0)}/3 markets.",
        ]
        response['table'] = _display_frame(_market_policy_overview_frame(latest))
        response['table_label'] = 'Market Policy Snapshot'
        return response
    if intent == 'market_summary':
        frame = market_proof_by_market_bundle.copy() if isinstance(market_proof_by_market_bundle, pd.DataFrame) else pd.DataFrame()
        response['title'] = 'Market Performance Summary'
        if frame.empty:
            response['summary'] = 'No market-proof summary is stored yet for this phase view.'
            return response
        proof = frame.copy()
        proof['market_type'] = proof['market_type'].astype(str).str.lower()
        preferred_market = None
        lowered = question.lower()
        for market_type in ['moneyline', 'run_line', 'total']:
            if market_type.replace('_', ' ') in lowered or market_type in lowered or (market_type == 'total' and 'totals' in lowered):
                preferred_market = market_type
                break
        if preferred_market:
            proof = proof[proof['market_type'] == preferred_market].copy()
        response['summary'] = 'Here is the current market-proof view for the selected phase.'
        response['bullets'] = [
            f"{_market_type_display(row.get('market_type'))}: ROI {_fmt_pct(row.get('roi'))} | Avg CLV {_fmt_num(row.get('avg_clv'), 3)} | Bets {int(row.get('bets') or 0)}"
            for _, row in proof.head(3).iterrows()
        ]
        response['table'] = _display_frame(proof)
        response['table_label'] = 'Market Proof'
        return response
    if intent in {'explain_game', 'overview_summary', 'unknown'}:
        context = _find_matching_context(question, predictions, candidates, portfolio_recommendations)
        response['title'] = 'Game and Edge Explanation'
        recommendation = context.get('recommendation') or {}
        prediction = context.get('prediction') or {}
        if recommendation:
            response['summary'] = str(recommendation.get('bet_thesis') or recommendation.get('market_throttle_note') or 'This position is supported by the live model recommendation stack.')
            response['bullets'] = [
                'Technical drivers: ' + str(recommendation.get('bet_drivers') or 'No driver summary is stored.'),
                'Risk flags: ' + str(recommendation.get('risk_flags') or 'No special risk flags noted.'),
                'Market throttle: ' + str(recommendation.get('market_throttle_note') or recommendation.get('market_throttle') or 'No throttle note.'),
            ]
            response['table'] = _display_frame(pd.DataFrame([{
                'matchup': f"{recommendation.get('away_team')} @ {recommendation.get('home_team')}",
                'market_type': _market_type_display(recommendation.get('market_type')),
                'selection': recommendation.get('selection'),
                'book_price': recommendation.get('book_price'),
                'model_probability': recommendation.get('model_probability'),
                'edge_value': recommendation.get('edge_value'),
                'actionability_label': recommendation.get('actionability_label'),
            }]), {'matchup': 'Matchup', 'market_type': 'Market', 'selection': 'Selection', 'book_price': 'Book Price', 'model_probability': 'Model Probability', 'edge_value': 'Edge', 'actionability_label': 'Actionability'})
            response['table_label'] = 'Supporting Position'
            return response
        if prediction:
            response['summary'] = str(prediction.get('moneyline_thesis') or prediction.get('run_line_thesis') or prediction.get('total_thesis') or 'The model has a stored game thesis for this matchup, but no richer recommendation row was found.')
            response['bullets'] = [
                f"Winner: {prediction.get('winner')} at {_fmt_pct(prediction.get('winner_prob'))}.",
                'Run-line thesis: ' + str(prediction.get('run_line_thesis') or 'Not stored.'),
                'Totals thesis: ' + str(prediction.get('total_thesis') or 'Not stored.'),
            ]
            response['table'] = _display_frame(pd.DataFrame([prediction]), {
                'away_team': 'Away Team',
                'home_team': 'Home Team',
                'winner': 'Model Favorite',
                'winner_prob': 'Win Probability',
                'confidence_tier': 'Confidence Tier',
                'fair_away_ml': 'Fair Away Moneyline',
                'fair_home_ml': 'Fair Home Moneyline',
                'total_calibrated': 'Projected Total',
                'margin_calibrated': 'Projected Margin',
            })
            response['table_label'] = 'Supporting Game Snapshot'
            return response
        if not predictions.empty:
            top = predictions.sort_values('winner_prob', ascending=False).iloc[0].to_dict()
            response['summary'] = f"The strongest current model edge is {top.get('away_team')} @ {top.get('home_team')}, with {top.get('winner')} winning {_fmt_pct(top.get('winner_prob'))} of the time."
            response['bullets'] = [
                'Moneyline thesis: ' + str(top.get('moneyline_thesis') or 'Not stored.'),
                'Run-line thesis: ' + str(top.get('run_line_thesis') or 'Not stored.'),
                'Totals thesis: ' + str(top.get('total_thesis') or 'Not stored.'),
            ]
            return response
        response['summary'] = 'The question could not be grounded in the currently loaded dashboard data.'
        response['bullets'] = [
            'Try asking about a specific team, matchup, policy posture, validation state, or a parlay target.',
            'Example: “Why is LAD the strongest edge today?”, “What does DraftKings show for LAD today?”, or “Would LAD still be a bet at -150?”',
        ]
        return response


def _portfolio_plain_english(utilization_pct: Any, correlated_candidates: Any, game_concentration_pct: Any, team_concentration_pct: Any) -> str:
    utilization = _safe_float(utilization_pct) or 0.0
    correlated = int(correlated_candidates or 0)
    game_pct = _safe_float(game_concentration_pct) or 0.0
    team_pct = _safe_float(team_concentration_pct) or 0.0
    notes = []
    if utilization < 0.25:
        notes.append('The model is deploying lightly, which usually means it does not see many prices worth pressing.')
    elif utilization < 0.60:
        notes.append('The model is taking a measured amount of risk instead of pushing the whole slate.')
    else:
        notes.append('The model is leaning into this slate, so exposure control matters more than usual.')
    if correlated <= 0:
        notes.append('There is little overlap between positions, so one game is less likely to dominate the whole card.')
    else:
        notes.append(f'{correlated} positions share game or team exposure, so the allocator is trimming overlap before it sizes them.')
    if max(game_pct, team_pct) >= 0.35:
        notes.append('One game or team is carrying a heavy share of the slate, so concentration risk is elevated.')
    elif max(game_pct, team_pct) >= 0.20:
        notes.append('There is some concentration in the book, but it is still in a manageable range.')
    else:
        notes.append('Exposure is fairly spread out, which keeps single-game damage more contained.')
    return ' '.join(notes)


def _clip(value: Any, low: float, high: float) -> float:
    numeric = _safe_float(value)
    if numeric is None:
        numeric = low
    return max(low, min(high, float(numeric)))


def _policy_specs() -> Dict[str, Dict[str, Any]]:
    return {
        'Flat': {
            'technical': 'Uses the allocator output directly with no extra edge-based rescaling.',
            'plain_english': 'Every approved play stays close to the base size, which keeps the plan simple and predictable.',
        },
        'Fractional Kelly Proxy': {
            'technical': 'Scales positions by expected value, confidence, and correlation penalty using a Kelly-like proxy rather than a full closed-form Kelly formula.',
            'plain_english': 'Stronger model edges get sized a bit bigger and thinner edges get sized a bit smaller.',
        },
        'Risk-Capped Hybrid': {
            'technical': 'Starts from a Kelly-style edge response, then applies extra concentration drag to correlated same-game and same-team exposure.',
            'plain_english': 'This tries to press the best ideas while still protecting the slate from becoming too dependent on one game or one team.',
        },
    }


def _risk_posture_specs() -> Dict[str, Dict[str, Any]]:
    return {
        'Conservative': {
            'unit_mult': 0.75,
            'deployment_mult': 0.75,
            'plain_english': 'This keeps daily swings lower and prioritizes capital preservation.',
        },
        'Base': {
            'unit_mult': 1.00,
            'deployment_mult': 1.00,
            'plain_english': 'This is the default balance between upside and drawdown control.',
        },
        'Aggressive': {
            'unit_mult': 1.20,
            'deployment_mult': 1.20,
            'plain_english': 'This leans in harder, so both upside and downside get magnified.',
        },
    }


def _portfolio_team_keys(row: pd.Series) -> list[str]:
    market_type = str(row.get('market_type') or '').lower()
    selected_team = str(row.get('selected_team') or '').strip().upper()
    away_team = str(row.get('away_team') or '').strip().upper()
    home_team = str(row.get('home_team') or '').strip().upper()
    if market_type == 'total':
        return [team for team in [away_team, home_team] if team]
    if selected_team:
        return [selected_team]
    return [team for team in [away_team, home_team] if team][:1]


def _portfolio_game_bucket(row: pd.Series) -> str:
    game_pk = row.get('game_pk')
    if pd.notna(game_pk):
        try:
            return f"gamepk:{int(float(game_pk))}"
        except Exception:
            pass
    return f"{str(row.get('away_team') or '').upper()}@{str(row.get('home_team') or '').upper()}"


def _apply_policy_to_recommendations(summary_row: Dict[str, Any], recommendations: pd.DataFrame, policy_name: str, posture_name: str, bankroll: float, unit_risk_pct: float, deployment_scalar: float):
    policy = _policy_specs().get(policy_name, _policy_specs()['Flat'])
    posture = _risk_posture_specs().get(posture_name, _risk_posture_specs()['Base'])
    recommendation_frame = recommendations.copy() if isinstance(recommendations, pd.DataFrame) else pd.DataFrame()
    if recommendation_frame.empty:
        return recommendation_frame, {
            'policy_name': policy_name,
            'posture_name': posture_name,
            'unit_size_dollars': bankroll * (unit_risk_pct / 100.0) * posture['unit_mult'],
            'deployed_risk_dollars': 0.0,
            'expected_value_dollars': 0.0,
            'max_loss_dollars': 0.0,
            'game_risk_dollars': 0.0,
            'team_risk_dollars': 0.0,
            'daily_budget_units': ((_safe_float(summary_row.get('daily_budget_units')) if summary_row else 0.0) or 0.0) * posture['deployment_mult'] * deployment_scalar,
            'budget_scale': 1.0,
            'policy_description': policy['technical'],
            'plain_english': f"{policy['plain_english']} {posture['plain_english']}",
            'avg_correlation_penalty': _safe_float(summary_row.get('avg_correlation_penalty')) if summary_row else None,
            'avg_pairwise_correlation': _safe_float(summary_row.get('avg_pairwise_correlation')) if summary_row else None,
            'peak_pairwise_correlation': _safe_float(summary_row.get('peak_pairwise_correlation')) if summary_row else None,
            'avg_matrix_correlation_penalty': _safe_float(summary_row.get('avg_matrix_correlation_penalty')) if summary_row else None,
            'avg_uncertainty_calibration_bias': None,
            'avg_uncertainty_penalty_scale': None,
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rec = recommendation_frame.copy()
    for col in [
        'allocated_units', 'expected_value_per_unit', 'edge_value', 'actionability_score', 'final_correlation_penalty',
        'same_game_peer_count', 'same_team_peer_count', 'matrix_correlation_penalty', 'avg_active_correlation',
        'max_active_correlation', 'uncertainty_calibration_bias', 'uncertainty_penalty_scale'
    ]:
        if col in rec.columns:
            rec[col] = pd.to_numeric(rec[col], errors='coerce').fillna(0.0)
        else:
            rec[col] = 0.0
    rec['game_bucket'] = rec.apply(_portfolio_game_bucket, axis=1)
    rec['team_keys'] = rec.apply(_portfolio_team_keys, axis=1)

    def _policy_factor(row: pd.Series):
        ev = _clip(row.get('expected_value_per_unit'), 0.0, 0.20)
        edge = _clip(abs(_safe_float(row.get('edge_value')) or 0.0), 0.0, 0.15)
        action = _clip((_safe_float(row.get('actionability_score')) or 0.0) / 100.0, 0.40, 1.00)
        corr = _clip(row.get('final_correlation_penalty'), 0.45, 1.00)
        avg_corr = _clip(row.get('avg_active_correlation'), 0.0, 1.00)
        max_corr = _clip(row.get('max_active_correlation'), 0.0, 1.00)
        matrix_corr = _clip(row.get('matrix_correlation_penalty'), 0.40, 1.00)
        same_game = int(_safe_float(row.get('same_game_peer_count')) or 0)
        same_team = int(_safe_float(row.get('same_team_peer_count')) or 0)
        overlap_drag = _clip((0.82 + (0.18 * corr)) * matrix_corr * (1.0 - (0.18 * avg_corr) - (0.10 * max_corr)), 0.55, 1.05)
        if policy_name == 'Flat':
            return 1.0, 'Keeps the allocator size as-is after the base uncertainty and correlation trims.'
        if policy_name == 'Fractional Kelly Proxy':
            factor = _clip((0.55 + (2.50 * ev) + (0.75 * edge) + (0.15 * action)) * overlap_drag, 0.45, 1.35)
            return factor, 'Scales up cleaner, higher-EV edges and scales down weaker or more correlated ones.'
        concentration_drag = _clip((1.0 - (0.06 * same_game) - (0.05 * same_team)) * (1.0 - (0.12 * avg_corr) - (0.08 * max_corr)) * matrix_corr, 0.62, 1.00)
        factor = _clip(min(0.60 + (2.20 * ev) + (0.60 * edge) + (0.10 * action), 1.10) * concentration_drag * (0.90 + (0.10 * corr)), 0.50, 1.10)
        return factor, 'Uses a Kelly-like response but trims same-game, same-team, and pairwise overlap more aggressively.'

    policy_details = rec.apply(_policy_factor, axis=1, result_type='expand')
    rec['policy_factor'] = pd.to_numeric(policy_details[0], errors='coerce').fillna(1.0)
    rec['policy_note'] = policy_details[1].astype(str)

    base_units = rec['allocated_units'] * posture['deployment_mult'] * deployment_scalar
    rec['policy_units_raw'] = base_units * rec['policy_factor']
    budget_units = (((_safe_float(summary_row.get('daily_budget_units')) if summary_row else None) or float(rec['allocated_units'].sum()) or 0.0) * posture['deployment_mult'] * deployment_scalar)
    raw_total = float(rec['policy_units_raw'].sum())
    budget_scale = min(1.0, (budget_units / raw_total)) if raw_total > 1e-9 else 1.0
    rec['policy_units'] = rec['policy_units_raw'] * budget_scale

    unit_size_dollars = bankroll * ((unit_risk_pct / 100.0) * posture['unit_mult'])
    rec['policy_risk_dollars'] = rec['policy_units'] * unit_size_dollars
    rec['policy_expected_value_dollars'] = rec['policy_units'] * rec['expected_value_per_unit'] * unit_size_dollars

    market_exposure = rec.groupby('market_type', dropna=False).agg(
        policy_units=('policy_units', 'sum'),
        policy_risk_dollars=('policy_risk_dollars', 'sum'),
        candidate_count=('selection', 'size'),
    ).reset_index().rename(columns={'market_type': 'bucket'})

    game_exposure = rec.groupby('game_bucket', dropna=False).agg(
        policy_units=('policy_units', 'sum'),
        policy_risk_dollars=('policy_risk_dollars', 'sum'),
        candidate_count=('selection', 'size'),
    ).reset_index().rename(columns={'game_bucket': 'bucket'})

    team_rows = []
    for _, row in rec.iterrows():
        for team in row.get('team_keys') or []:
            team_rows.append({'bucket': team, 'policy_units': row['policy_units'], 'policy_risk_dollars': row['policy_risk_dollars']})
    team_exposure = pd.DataFrame(team_rows)
    if not team_exposure.empty:
        team_exposure = team_exposure.groupby('bucket', dropna=False).agg(
            policy_units=('policy_units', 'sum'),
            policy_risk_dollars=('policy_risk_dollars', 'sum'),
            candidate_count=('bucket', 'size'),
        ).reset_index()
    else:
        team_exposure = pd.DataFrame(columns=['bucket', 'policy_units', 'policy_risk_dollars', 'candidate_count'])

    summary = {
        'policy_name': policy_name,
        'posture_name': posture_name,
        'unit_size_dollars': unit_size_dollars,
        'deployed_risk_dollars': float(rec['policy_risk_dollars'].sum()),
        'expected_value_dollars': float(rec['policy_expected_value_dollars'].sum()),
        'max_loss_dollars': float(rec['policy_risk_dollars'].sum()),
        'game_risk_dollars': float(game_exposure['policy_risk_dollars'].max()) if not game_exposure.empty else 0.0,
        'team_risk_dollars': float(team_exposure['policy_risk_dollars'].max()) if not team_exposure.empty else 0.0,
        'daily_budget_units': budget_units,
        'budget_scale': budget_scale,
        'policy_description': policy['technical'],
        'plain_english': f"{policy['plain_english']} {posture['plain_english']}",
        'avg_correlation_penalty': float(rec['final_correlation_penalty'].mean()) if 'final_correlation_penalty' in rec.columns else None,
        'avg_pairwise_correlation': float(rec['avg_active_correlation'].mean()) if 'avg_active_correlation' in rec.columns else None,
        'peak_pairwise_correlation': float(rec['max_active_correlation'].max()) if 'max_active_correlation' in rec.columns else None,
        'avg_matrix_correlation_penalty': float(rec['matrix_correlation_penalty'].mean()) if 'matrix_correlation_penalty' in rec.columns else None,
        'avg_uncertainty_calibration_bias': float(rec['uncertainty_calibration_bias'].mean()) if 'uncertainty_calibration_bias' in rec.columns else None,
        'avg_uncertainty_penalty_scale': float(rec['uncertainty_penalty_scale'].mean()) if 'uncertainty_penalty_scale' in rec.columns else None,
    }
    return rec, summary, market_exposure, team_exposure, game_exposure


def _build_portfolio_scenarios(summary_row: Dict[str, Any], recommendations: pd.DataFrame, bankroll: float, unit_risk_pct: float, deployment_scalar: float, posture_name: str) -> pd.DataFrame:
    rows = []
    for policy_name, policy in _policy_specs().items():
        _, policy_summary, _, _, _ = _apply_policy_to_recommendations(
            summary_row,
            recommendations,
            policy_name,
            posture_name,
            bankroll,
            unit_risk_pct,
            deployment_scalar,
        )
        rows.append({
            'scenario': policy_name,
            'unit_risk_pct': round((unit_risk_pct * _risk_posture_specs().get(posture_name, _risk_posture_specs()['Base'])['unit_mult']), 2),
            'deployment_scalar': round(deployment_scalar * _risk_posture_specs().get(posture_name, _risk_posture_specs()['Base'])['deployment_mult'], 2),
            'unit_size_dollars': round(_safe_float(policy_summary.get('unit_size_dollars')) or 0.0, 2),
            'deployed_risk_dollars': round(_safe_float(policy_summary.get('deployed_risk_dollars')) or 0.0, 2),
            'expected_value_dollars': round(_safe_float(policy_summary.get('expected_value_dollars')) or 0.0, 2),
            'max_loss_dollars': round(_safe_float(policy_summary.get('max_loss_dollars')) or 0.0, 2),
            'game_risk_dollars': round(_safe_float(policy_summary.get('game_risk_dollars')) or 0.0, 2),
            'team_risk_dollars': round(_safe_float(policy_summary.get('team_risk_dollars')) or 0.0, 2),
            'avg_correlation_penalty': round(_safe_float(policy_summary.get('avg_correlation_penalty')) or 0.0, 3),
            'avg_pairwise_correlation': round(_safe_float(policy_summary.get('avg_pairwise_correlation')) or 0.0, 3),
            'avg_matrix_correlation_penalty': round(_safe_float(policy_summary.get('avg_matrix_correlation_penalty')) or 0.0, 3),
            'technical_note': policy['technical'],
            'plain_english': policy_summary.get('plain_english') or policy['plain_english'],
        })
    return pd.DataFrame(rows)


def _confidence_tier_from_probability(probability: Any) -> str:
    numeric = _safe_float(probability)
    if numeric is None:
        return 'LOW'
    confidence = max(min(max(numeric, 1.0 - numeric), 1.0), 0.5)
    if confidence >= 0.67:
        return 'HIGH'
    if confidence >= 0.60:
        return 'MEDIUM'
    return 'LOW'


def _shadow_actionability_label(confidence: float) -> str:
    if confidence >= 0.67:
        return 'MUST TAKE'
    if confidence >= 0.60:
        return 'BET'
    if confidence >= 0.55:
        return 'WATCH'
    return 'PASS'


def _shadow_candidate_frame_from_archive(live_archive: pd.DataFrame) -> pd.DataFrame:
    required_columns = {'report_date', 'away', 'home', 'p_home', 'p_home_simple', 'y_home'}
    if live_archive.empty or not required_columns.issubset(live_archive.columns):
        return pd.DataFrame()
    work = live_archive.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce')
    work['p_home_simple'] = pd.to_numeric(work.get('p_home_simple'), errors='coerce').fillna(0.5)
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    work = work.dropna(subset=['report_date', 'p_home', 'y_home', 'away', 'home']).copy()
    if work.empty:
        return pd.DataFrame()
    work['phase'] = work.get('phase', 'spring').fillna('spring').astype(str).str.lower()
    work['model_confidence'] = work['p_home'].map(lambda value: max(min(max(float(value), 1.0 - float(value)), 1.0), 0.5))
    work['predicted_home'] = work['p_home'] >= 0.5
    work['selection'] = np.where(work['predicted_home'], work['home'].astype(str), work['away'].astype(str))
    work['selected_team'] = work['selection'].astype(str).str.upper()
    work['confidence_tier'] = work['p_home'].map(_confidence_tier_from_probability)
    work['actionability_label'] = work['model_confidence'].map(_shadow_actionability_label)
    confidence_gap = (work['model_confidence'] - 0.5).clip(lower=0.0)
    simple_gap = (work['p_home_simple'] - 0.5).abs().clip(lower=0.0)
    work['actionability_score'] = (35.0 + (confidence_gap / 0.17).clip(upper=1.0) * 55.0).round(1)
    work['edge_value'] = (work['p_home'] - work['p_home_simple']).abs().clip(lower=0.0, upper=0.15)
    work['expected_value_per_unit'] = (confidence_gap * 0.80).clip(lower=0.0, upper=0.20)
    work['suggested_units_preview'] = pd.Series(
        np.select(
            [
                work['actionability_label'] == 'WATCH',
                work['actionability_label'] == 'BET',
                work['actionability_label'] == 'MUST TAKE',
            ],
            [
                0.20 + (confidence_gap * 1.20),
                0.40 + (confidence_gap * 1.50),
                0.65 + (confidence_gap * 1.75),
            ],
            default=0.0,
        ),
        index=work.index,
    ).clip(lower=0.0, upper=1.20).round(2)
    work['candidate_score'] = (
        work['expected_value_per_unit'].fillna(0.0)
        * (1.0 + work['actionability_score'].fillna(0.0) / 100.0)
        * (1.0 + (work['edge_value'].fillna(0.0) + simple_gap).clip(upper=0.25))
    )
    work['shadow_result_per_unit'] = np.where(
        (work['predicted_home'] & (work['y_home'] >= 0.5)) | ((~work['predicted_home']) & (work['y_home'] < 0.5)),
        1.0,
        -1.0,
    )
    work['market_type'] = 'moneyline'
    work['market_label'] = 'Shadow archive'
    work['sportsbook_count'] = 0
    work['book_price'] = np.nan
    work['fair_price'] = np.nan
    work['line_value'] = np.nan
    work['game_pk'] = pd.to_numeric(work.get('game_pk'), errors='coerce')
    work['away_team'] = work['away'].astype(str).str.upper()
    work['home_team'] = work['home'].astype(str).str.upper()
    work['actionability_notes'] = np.where(
        work['actionability_label'] == 'PASS',
        'Shadow archive confidence was too weak to simulate an allocated position.',
        'Shadow archive position based on the model main-side pick at even-money normalized grading.',
    )
    columns = [
        'report_date', 'phase', 'game_pk', 'away_team', 'home_team', 'market_label', 'sportsbook_count',
        'confidence_tier', 'market_type', 'selection', 'line_value', 'book_price', 'fair_price',
        'edge_value', 'actionability_label', 'actionability_score', 'actionability_notes',
        'expected_value_per_unit', 'suggested_units_preview', 'candidate_score', 'selected_team',
        'shadow_result_per_unit',
    ]
    return work[columns].reset_index(drop=True)


def _build_shadow_policy_backtest(live_archive: pd.DataFrame, bankroll: float, unit_risk_pct: float, deployment_scalar: float, posture_name: str):
    shadow_candidates = _shadow_candidate_frame_from_archive(live_archive)
    if shadow_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_rows = []
    for report_date, day_candidates in shadow_candidates.groupby('report_date', dropna=True):
        phase = str(day_candidates['phase'].iloc[0] or 'spring').lower()
        recs, summary_frame, _, _, _ = portfolio.build_portfolio_plan(
            day_candidates,
            phase,
            f"shadow_{report_date.isoformat()}",
            report_date,
        )
        summary_row = {} if summary_frame.empty else summary_frame.iloc[0].to_dict()
        if recs.empty:
            continue
        shadow_lookup = day_candidates[['away_team', 'home_team', 'market_type', 'selection', 'shadow_result_per_unit']].drop_duplicates()
        recs = recs.merge(
            shadow_lookup,
            on=['away_team', 'home_team', 'market_type', 'selection'],
            how='left',
        )
        for policy_name in _policy_specs().keys():
            policy_view, policy_summary, _, _, _ = _apply_policy_to_recommendations(
                summary_row,
                recs,
                policy_name,
                posture_name,
                bankroll,
                unit_risk_pct,
                deployment_scalar,
            )
            if policy_view.empty:
                continue
            result_units = float((policy_view['policy_units'] * policy_view['shadow_result_per_unit']).sum())
            risk_units = float(policy_view['policy_units'].sum())
            unit_size_dollars = float(_safe_float(policy_summary.get('unit_size_dollars')) or 0.0)
            picks = int(len(policy_view))
            hits = int((pd.to_numeric(policy_view['shadow_result_per_unit'], errors='coerce').fillna(-1.0) > 0).sum())
            daily_rows.append({
                'report_date': report_date,
                'phase': phase,
                'policy_name': policy_name,
                'posture_name': posture_name,
                'games': int(len(day_candidates)),
                'picks': picks,
                'hits': hits,
                'hit_rate': (hits / picks) if picks else None,
                'risk_units': risk_units,
                'result_units': result_units,
                'yield_pct': (result_units / risk_units) if risk_units > 1e-9 else None,
                'unit_size_dollars': unit_size_dollars,
                'risk_dollars': risk_units * unit_size_dollars,
                'result_dollars': result_units * unit_size_dollars,
                'avg_correlation_penalty': _safe_float(policy_summary.get('avg_correlation_penalty')),
                'plain_english': policy_summary.get('plain_english') or '',
            })
    daily_frame = pd.DataFrame(daily_rows)
    if daily_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_frame = daily_frame.sort_values(['policy_name', 'report_date']).reset_index(drop=True)
    daily_frame['cumulative_units'] = daily_frame.groupby('policy_name')['result_units'].cumsum()
    daily_frame['cumulative_dollars'] = daily_frame.groupby('policy_name')['result_dollars'].cumsum()
    daily_frame['rolling_peak_units'] = daily_frame.groupby('policy_name')['cumulative_units'].cummax()
    daily_frame['drawdown_units'] = daily_frame['cumulative_units'] - daily_frame['rolling_peak_units']
    daily_frame['rolling_peak_dollars'] = daily_frame.groupby('policy_name')['cumulative_dollars'].cummax()
    daily_frame['drawdown_dollars'] = daily_frame['cumulative_dollars'] - daily_frame['rolling_peak_dollars']
    summary_rows = []
    for policy_name, grp in daily_frame.groupby('policy_name', dropna=True):
        total_risk_units = float(grp['risk_units'].sum())
        total_result_units = float(grp['result_units'].sum())
        total_risk_dollars = float(grp['risk_dollars'].sum())
        total_result_dollars = float(grp['result_dollars'].sum())
        total_picks = int(grp['picks'].sum())
        total_hits = int(grp['hits'].sum())
        max_drawdown_units = abs(float(grp['drawdown_units'].min() or 0.0))
        max_drawdown_dollars = abs(float(grp['drawdown_dollars'].min() or 0.0))
        worst_day_units = float(grp['result_units'].min() or 0.0)
        avg_daily_risk = float(grp['risk_dollars'].mean() or 0.0)
        if max_drawdown_units <= 1.0 and total_result_units >= 0.0:
            takeaway = 'This was the smoothest path, with smaller swings and more contained drawdowns.'
        elif total_result_units > 0.0 and max_drawdown_units > 1.0:
            takeaway = 'This produced more upside, but it asked you to tolerate visibly larger swings.'
        else:
            takeaway = 'This sizing style has not earned trust yet on the archived shadow sample.'
        summary_rows.append({
            'policy_name': policy_name,
            'posture_name': posture_name,
            'settled_slates': int(len(grp)),
            'settled_games': int(grp['games'].sum()),
            'settled_picks': total_picks,
            'hit_rate': (total_hits / total_picks) if total_picks else None,
            'risk_units': total_risk_units,
            'result_units': total_result_units,
            'yield_pct': (total_result_units / total_risk_units) if total_risk_units > 1e-9 else None,
            'risk_dollars': total_risk_dollars,
            'result_dollars': total_result_dollars,
            'max_drawdown_units': max_drawdown_units,
            'max_drawdown_dollars': max_drawdown_dollars,
            'worst_day_units': worst_day_units,
            'avg_daily_risk_dollars': avg_daily_risk,
            'avg_correlation_penalty': float(grp['avg_correlation_penalty'].dropna().mean()) if 'avg_correlation_penalty' in grp.columns and not grp['avg_correlation_penalty'].dropna().empty else None,
            'technical_note': _policy_specs().get(policy_name, {}).get('technical', ''),
            'plain_english': takeaway,
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values('result_units', ascending=False).reset_index(drop=True)
    return daily_frame, summary_frame


def _build_market_realized_summary(bet_journal: pd.DataFrame):
    if bet_journal.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = {'report_date', 'result_units'}
    if not required_columns.issubset(bet_journal.columns):
        return pd.DataFrame(), pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    clv_source = 'clv_value' if 'clv_value' in work.columns else ('clv' if 'clv' in work.columns else None)
    if clv_source is not None:
        work['clv'] = pd.to_numeric(work.get(clv_source), errors='coerce')
    work = work.dropna(subset=['report_date', 'result_units'])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    if 'is_settled' in work.columns:
        settled_mask = work['is_settled'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        work = work[settled_mask | work['result_units'].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work['roi_pct'] = np.where(work['stake_units'].abs() > 1e-9, work['result_units'] / work['stake_units'], np.nan)
    daily = work.groupby('report_date', dropna=True).agg(
        bets=('result_units', 'size'),
        stake_units=('stake_units', 'sum'),
        result_units=('result_units', 'sum'),
    ).reset_index()
    if 'clv' in work.columns:
        clv_daily = work.groupby('report_date', dropna=True)['clv'].mean().reset_index(name='avg_clv')
        daily = daily.merge(clv_daily, on='report_date', how='left')
    daily['cumulative_units'] = daily['result_units'].cumsum()
    daily['rolling_peak_units'] = daily['cumulative_units'].cummax()
    daily['drawdown_units'] = daily['cumulative_units'] - daily['rolling_peak_units']
    daily['roi_pct'] = np.where(daily['stake_units'].abs() > 1e-9, daily['result_units'] / daily['stake_units'], np.nan)
    summary_rows = [{
        'settled_bets': int(len(work)),
        'stake_units': float(work['stake_units'].sum()),
        'result_units': float(work['result_units'].sum()),
        'roi_pct': float(work['result_units'].sum() / work['stake_units'].sum()) if abs(float(work['stake_units'].sum())) > 1e-9 else None,
        'avg_clv': float(work['clv'].dropna().mean()) if 'clv' in work.columns and not work['clv'].dropna().empty else None,
        'max_drawdown_units': abs(float((daily['drawdown_units'].min()) or 0.0)),
    }]
    by_market = pd.DataFrame()
    if 'market_type' in work.columns:
        market_rows = []
        for market_type, grp in work.groupby('market_type', dropna=False):
            grp = grp.sort_values('report_date').copy()
            grp['cumulative_units'] = grp['result_units'].cumsum()
            grp['rolling_peak_units'] = grp['cumulative_units'].cummax()
            grp['drawdown_units'] = grp['cumulative_units'] - grp['rolling_peak_units']
            stake_units = float(grp['stake_units'].sum())
            market_rows.append({
                'market_type': market_type,
                'bets': int(len(grp)),
                'stake_units': stake_units,
                'result_units': float(grp['result_units'].sum()),
                'roi_pct': float(grp['result_units'].sum() / stake_units) if abs(stake_units) > 1e-9 else None,
                'avg_clv': float(grp['clv'].dropna().mean()) if 'clv' in grp.columns and not grp['clv'].dropna().empty else None,
                'max_drawdown_units': abs(float((grp['drawdown_units'].min()) or 0.0)),
                'settled_days': int(grp['report_date'].nunique()),
            })
        by_market = pd.DataFrame(market_rows)
    return pd.DataFrame(summary_rows), by_market


def _build_market_realized_daily(bet_journal: pd.DataFrame) -> pd.DataFrame:
    if bet_journal.empty or 'report_date' not in bet_journal.columns or 'result_units' not in bet_journal.columns:
        return pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    clv_source = 'clv_value' if 'clv_value' in work.columns else ('clv' if 'clv' in work.columns else None)
    if clv_source is not None:
        work['clv'] = pd.to_numeric(work.get(clv_source), errors='coerce')
    work = work.dropna(subset=['report_date', 'result_units'])
    if work.empty:
        return pd.DataFrame()
    if 'is_settled' in work.columns:
        settled_mask = work['is_settled'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        work = work[settled_mask | work['result_units'].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    daily = work.groupby('report_date', dropna=True).agg(
        bets=('result_units', 'size'),
        stake_units=('stake_units', 'sum'),
        result_units=('result_units', 'sum'),
    ).reset_index()
    if 'clv' in work.columns:
        clv_daily = work.groupby('report_date', dropna=True)['clv'].mean().reset_index(name='avg_clv')
        daily = daily.merge(clv_daily, on='report_date', how='left')
    daily = daily.sort_values('report_date').reset_index(drop=True)
    daily['roi_pct'] = np.where(daily['stake_units'].abs() > 1e-9, daily['result_units'] / daily['stake_units'], np.nan)
    daily['cumulative_units'] = daily['result_units'].cumsum()
    daily['rolling_peak_units'] = daily['cumulative_units'].cummax()
    daily['drawdown_units'] = daily['cumulative_units'] - daily['rolling_peak_units']
    return daily


def _filter_to_window(frame: pd.DataFrame, date_col: str, window_days: int | None) -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce').dt.date
    work = work.dropna(subset=[date_col])
    if work.empty or window_days is None:
        return work.sort_values(date_col).reset_index(drop=True)
    max_date = work[date_col].max()
    if pd.isna(max_date):
        return work.sort_values(date_col).reset_index(drop=True)
    start_date = max_date - dt.timedelta(days=max(int(window_days) - 1, 0))
    return work[work[date_col] >= start_date].sort_values(date_col).reset_index(drop=True)


def _summarize_shadow_window(shadow_daily: pd.DataFrame, window_days: int | None) -> pd.DataFrame:
    window = _filter_to_window(shadow_daily, 'report_date', window_days)
    if window.empty:
        return pd.DataFrame()
    rows = []
    for policy_name, grp in window.groupby('policy_name', dropna=True):
        grp = grp.sort_values('report_date').reset_index(drop=True)
        cumulative = grp['result_units'].cumsum()
        drawdown = cumulative - cumulative.cummax()
        volatility_units = float(pd.to_numeric(grp['result_units'], errors='coerce').std(ddof=0) or 0.0)
        result_units = float(grp['result_units'].sum())
        risk_units = float(grp['risk_units'].sum())
        max_drawdown_units = abs(float(drawdown.min() or 0.0))
        yield_pct = (result_units / risk_units) if risk_units > 1e-9 else None
        stability_score = result_units - (0.75 * max_drawdown_units) - (0.35 * volatility_units)
        rows.append({
            'policy_name': policy_name,
            'settled_slates': int(len(grp)),
            'settled_games': int(grp['games'].sum()) if 'games' in grp.columns else int(len(grp)),
            'settled_picks': int(grp['picks'].sum()) if 'picks' in grp.columns else int(len(grp)),
            'hit_rate': float(grp['hits'].sum() / grp['picks'].sum()) if 'hits' in grp.columns and float(grp['picks'].sum()) > 0 else None,
            'risk_units': risk_units,
            'result_units': result_units,
            'yield_pct': yield_pct,
            'max_drawdown_units': max_drawdown_units,
            'worst_day_units': float(grp['result_units'].min() or 0.0),
            'volatility_units': volatility_units,
            'avg_daily_risk_dollars': float(grp['risk_dollars'].mean() or 0.0) if 'risk_dollars' in grp.columns else None,
            'avg_correlation_penalty': float(grp['avg_correlation_penalty'].dropna().mean()) if 'avg_correlation_penalty' in grp.columns and not grp['avg_correlation_penalty'].dropna().empty else None,
            'stability_score': stability_score,
            'sample_ok': bool((len(grp) >= 3) and (int(grp['picks'].sum()) >= 10 if 'picks' in grp.columns else len(grp) >= 10)),
        })
    return pd.DataFrame(rows).sort_values(['stability_score', 'result_units', 'yield_pct'], ascending=[False, False, False]).reset_index(drop=True)


def _recommend_policy_research(shadow_window: pd.DataFrame, realized_daily: pd.DataFrame, window_label: str, window_days: int | None, market_window: pd.DataFrame | None = None) -> Dict[str, Any]:
    return policy_research.recommend_policy_research(shadow_window, realized_daily, window_label, window_days, market_window)


def _latest_policy_research_recommendation(history: pd.DataFrame, window_label: str) -> Dict[str, Any]:
    if history.empty:
        return {}
    work = history.copy()
    if 'window_label' in work.columns:
        filtered = work[work['window_label'].astype(str) == str(window_label)].copy()
        if not filtered.empty:
            work = filtered
    if 'as_of_report_date' in work.columns:
        work['as_of_report_date'] = pd.to_datetime(work.get('as_of_report_date'), errors='coerce')
    sort_cols = [col for col in ['as_of_report_date', 'run_id'] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    if work.empty:
        return {}
    return work.iloc[0].to_dict()


def _policy_rankings_for_window(rankings: pd.DataFrame, window_label: str, evidence_family: str | None = None) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    work = rankings.copy()
    if 'window_label' in work.columns:
        work = work[work['window_label'].astype(str) == str(window_label)].copy()
    if evidence_family is not None and 'evidence_family' in work.columns:
        work = work[work['evidence_family'].astype(str) == str(evidence_family)].copy()
    if work.empty:
        return pd.DataFrame()
    sort_cols = [col for col in ['stability_score', 'result_units', 'yield_pct'] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return work


def _load_bundle() -> Dict[str, Any]:
    return quant_store.load_dashboard_bundle()


def main() -> None:
    st.set_page_config(page_title='MLB Quant Dashboard', layout='wide')
    _inject_dashboard_styles()
    bundle = _load_bundle()
    status = bundle.get('status') or {}
    run_history = _frame_or_empty(bundle.get('run_history'))
    latest_run = _frame_or_empty(bundle.get('latest_run'))
    predictions = _frame_or_empty(bundle.get('predictions'))
    candidates = _frame_or_empty(bundle.get('candidates'))
    portfolio_recommendations = _frame_or_empty(bundle.get('portfolio_recommendations'))
    portfolio_summary = _frame_or_empty(bundle.get('portfolio_summary'))
    portfolio_market_exposure = _frame_or_empty(bundle.get('portfolio_market_exposure'))
    portfolio_team_exposure = _frame_or_empty(bundle.get('portfolio_team_exposure'))
    portfolio_game_exposure = _frame_or_empty(bundle.get('portfolio_game_exposure'))
    clean_backtest = _frame_or_empty(bundle.get('clean_backtest'))
    live_archive = _frame_or_empty(bundle.get('live_archive'))
    bet_journal = _frame_or_empty(bundle.get('bet_journal'))
    odds_snapshots = _frame_or_empty(bundle.get('odds_snapshots'))
    policy_research_history = _frame_or_empty(bundle.get('policy_research_history'))
    policy_research_rankings = _frame_or_empty(bundle.get('policy_research_rankings'))
    policy_research_shadow_daily = _frame_or_empty(bundle.get('policy_research_shadow_daily'))
    policy_research_shadow_summary = _frame_or_empty(bundle.get('policy_research_shadow_summary'))
    policy_research_market_daily = _frame_or_empty(bundle.get('policy_research_market_daily'))
    policy_research_market_policy_daily = _frame_or_empty(bundle.get('policy_research_market_policy_daily'))
    policy_research_market_policy_summary = _frame_or_empty(bundle.get('policy_research_market_policy_summary'))
    policy_research_market_type_history = _frame_or_empty(bundle.get('policy_research_market_type_history'))
    market_proof_summary_bundle = _frame_or_empty(bundle.get('market_proof_summary'))
    market_proof_by_market_bundle = _frame_or_empty(bundle.get('market_proof_by_market'))
    market_proof_by_market_history = _frame_or_empty(bundle.get('market_proof_by_market_history'))
    daily_kpi_snapshot = _frame_or_empty(bundle.get('daily_kpi_snapshot'))
    rolling_kpi_windows = _frame_or_empty(bundle.get('rolling_kpi_windows'))
    alert_history = _frame_or_empty(bundle.get('alert_history'))
    target_history = _frame_or_empty(bundle.get('target_history'))
    attribution_market_mart = _frame_or_empty(bundle.get('attribution_market_mart'))
    attribution_driver_mart = _frame_or_empty(bundle.get('attribution_driver_mart'))
    model_state = bundle.get('model_state') or {}

    st.title('MLB Quant Console')
    st.caption('A professional research and portfolio layer built on top of the daily MLB fair-line engine. The text betting report remains the printable output; this dashboard is the monitoring and decision surface.')

    with st.sidebar:
        st.header('Platform Status')
        phase_options = _phase_label_options()
        selected_phase_label = st.selectbox('Phase View', list(phase_options.keys()), index=0)
        selected_phase = phase_options[selected_phase_label]
        if status.get('duckdb_available') and status.get('db_exists'):
            st.success('DuckDB analytics store is active.')
            st.caption(status.get('db_path'))
        elif status.get('duckdb_available'):
            st.warning('DuckDB is available, but the analytics store has not been created yet. Run the daily model once to initialize the analytics store.')
            st.caption(status.get('db_path'))
        else:
            st.info('DuckDB is not available in this environment yet. The dashboard is falling back to raw source files where possible.')
            if status.get('duckdb_error'):
                st.caption(status.get('duckdb_error'))
        st.markdown('Launch with `streamlit run quant_dashboard.py`.')
        st.markdown('Optional packages are listed in `quant_dashboard_requirements.txt`.')

    phase_tables = _phase_run_tables(status, selected_phase)
    if selected_phase != 'all':
        latest_run = _frame_or_empty(phase_tables.get('latest_run'))
        predictions = _frame_or_empty(phase_tables.get('predictions'))
        candidates = _frame_or_empty(phase_tables.get('candidates'))
        portfolio_recommendations = _frame_or_empty(phase_tables.get('portfolio_recommendations'))
        portfolio_summary = _frame_or_empty(phase_tables.get('portfolio_summary'))
        portfolio_market_exposure = _frame_or_empty(phase_tables.get('portfolio_market_exposure'))
        portfolio_team_exposure = _frame_or_empty(phase_tables.get('portfolio_team_exposure'))
        portfolio_game_exposure = _frame_or_empty(phase_tables.get('portfolio_game_exposure'))
        policy_research_rankings = _frame_or_empty(phase_tables.get('policy_research_rankings'))
        policy_research_shadow_daily = _frame_or_empty(phase_tables.get('policy_research_shadow_daily'))
        policy_research_shadow_summary = _frame_or_empty(phase_tables.get('policy_research_shadow_summary'))
        policy_research_market_daily = _frame_or_empty(phase_tables.get('policy_research_market_daily'))
        policy_research_market_policy_daily = _frame_or_empty(phase_tables.get('policy_research_market_policy_daily'))
        policy_research_market_policy_summary = _frame_or_empty(phase_tables.get('policy_research_market_policy_summary'))
        market_proof_summary_bundle = _frame_or_empty(phase_tables.get('market_proof_summary'))
        market_proof_by_market_bundle = _frame_or_empty(phase_tables.get('market_proof_by_market'))

    run_history = _filter_frame_by_phase(run_history, selected_phase, ['report_date', 'created_ts'])
    clean_backtest = _filter_frame_by_phase(clean_backtest, selected_phase, ['date'])
    live_archive = _filter_frame_by_phase(live_archive, selected_phase, ['report_date'])
    bet_journal = _filter_frame_by_phase(bet_journal, selected_phase, ['report_date'])
    odds_snapshots = _filter_frame_by_phase(odds_snapshots, selected_phase, ['commence_time', 'captured_ts', 'odds_api_pull_ts', 'report_date'])
    policy_research_history = _filter_frame_by_phase(policy_research_history, selected_phase, ['as_of_report_date'])
    policy_research_market_type_history = _filter_frame_by_phase(policy_research_market_type_history, selected_phase, ['as_of_report_date'])
    market_proof_by_market_history = _filter_frame_by_phase(market_proof_by_market_history, selected_phase, ['as_of_report_date'])
    policy_research_market_daily = _filter_frame_by_phase(policy_research_market_daily, selected_phase, ['report_date'])
    policy_research_market_policy_daily = _filter_frame_by_phase(policy_research_market_policy_daily, selected_phase, ['report_date'])
    policy_research_shadow_daily = _filter_frame_by_phase(policy_research_shadow_daily, selected_phase, ['report_date'])
    daily_kpi_snapshot = _filter_frame_by_phase(daily_kpi_snapshot, selected_phase, ['report_date'])
    rolling_kpi_windows = _filter_frame_by_phase(rolling_kpi_windows, selected_phase, ['report_date'])
    alert_history = _filter_frame_by_phase(alert_history, selected_phase, ['as_of_report_date', 'created_ts'])
    target_history = _filter_frame_by_phase(target_history, selected_phase, ['as_of_report_date'])
    attribution_market_mart = _filter_frame_by_phase(attribution_market_mart, selected_phase, ['latest_report_date'])
    attribution_driver_mart = _filter_frame_by_phase(attribution_driver_mart, selected_phase, [])

    latest = {} if latest_run.empty else latest_run.iloc[0].to_dict()
    latest_report_date = str(latest.get('report_date') or (predictions['report_date'].max() if not predictions.empty and 'report_date' in predictions.columns else 'N/A'))
    market_proof_warnings = _market_proof_warning_frame(latest)
    actionable_candidates = 0
    if not portfolio_recommendations.empty and 'allocated_units' in portfolio_recommendations.columns:
        actionable_candidates = int((pd.to_numeric(portfolio_recommendations['allocated_units'], errors='coerce').fillna(0.0) > 0).sum())
    elif not candidates.empty and 'actionability_label' in candidates.columns:
        actionable_candidates = int(candidates['actionability_label'].astype(str).isin(['WATCH', 'BET', 'MUST TAKE']).sum())

    summary_cols = st.columns(4)
    summary_cols[0].metric('Latest Report Date', latest_report_date)
    summary_cols[1].metric('Current Phase', selected_phase_label if selected_phase != 'all' else str(latest.get('phase') or model_state.get('season_phase', 'N/A')).title())
    summary_cols[2].metric('Games on Slate', int(latest.get('prediction_count') or len(predictions) or 0))
    summary_cols[3].metric('Actionable Positions', actionable_candidates)
    _render_freshness_strip(selected_phase_label, latest_report_date, live_archive, odds_snapshots)

    tabs = st.tabs(['Overview', 'Today\'s Slate', 'Portfolio', 'Validation', 'Research', 'Ask the Model'])

    with tabs[0]:
        st.subheader('Executive Overview')
        metric_cols = st.columns(4)
        metric_cols[0].metric('Research Replay Log Loss', _fmt_num(latest.get('validation_log_loss'), 4))
        metric_cols[1].metric('Research Replay Accuracy', _fmt_pct(latest.get('validation_accuracy')))
        metric_cols[2].metric('Live Settled Log Loss', _fmt_num(latest.get('live_validation_log_loss'), 4))
        metric_cols[3].metric('Live Settled Accuracy', _fmt_pct(latest.get('live_validation_accuracy')))

        metric_cols2 = st.columns(4)
        metric_cols2[0].metric('Research Totals RMSE', _fmt_num(latest.get('validation_total_rmse'), 3))
        metric_cols2[1].metric('Research Margin RMSE', _fmt_num(latest.get('validation_margin_rmse'), 3))
        metric_cols2[2].metric('Settled Live Games', int(latest.get('live_validation_games') or 0))
        metric_cols2[3].metric('Settled Live Days', int(latest.get('live_validation_settled_days') or 0))
        st.caption('Research replay is the long-memory benchmark and only moves when the replay dataset is rebuilt. Live settled validation updates as archived games finish, so it is the day-to-day operating read.')
        target_bundle = _dynamic_target_bundle(latest, selected_phase)
        target_frame = _frame_or_empty(target_bundle.get('frame'))
        if not target_frame.empty:
            st.markdown('**Target vs Actual**')
            target_cols = st.columns(4)
            target_cols[0].metric('Evidence Progress', f"{(target_bundle.get('progress') or 0.0) * 100:.0f}%")
            target_cols[1].metric('Live Games', int(target_bundle.get('live_games') or 0))
            target_cols[2].metric('Settled Days', int(target_bundle.get('settled_days') or 0))
            target_cols[3].metric('Priced Bets', int(target_bundle.get('priced_bets') or 0))
            st.caption('Technical: dynamic targets blend from the frozen research baseline toward a stricter long-run goal as live sample builds. Plain English: the bar tightens over time, but it moves with friction so one hot or cold stretch does not reset expectations overnight.')
            with st.expander('View Target Reference Detail', expanded=False):
                st.dataframe(_display_frame(target_frame), use_container_width=True, hide_index=True)
        readiness_ladder = _market_readiness_ladder_frame(policy_research_market_type_history)
        current_health = _current_health_frame(latest, model_state, readiness_ladder, market_proof_warnings)
        evidence_drift = _market_evidence_drift_frame(policy_research_market_type_history)
        activation_readiness = _activation_readiness_frame(model_state, readiness_ladder)
        briefing = _operator_briefing(
            latest,
            selected_phase_label,
            actionable_candidates,
            current_health,
            readiness_ladder,
            market_proof_warnings,
            evidence_drift,
        )
        _render_operator_banner(briefing)
        brief_cols = st.columns(4)
        brief_cols[0].metric('Operating Posture', briefing['status'])
        brief_cols[1].metric('Model Health', briefing['model_status'])
        brief_cols[2].metric('Strongest Market', briefing['strongest_market'])
        brief_cols[3].metric('Weakest Trend', briefing['weakest_market'])
        alerts_frame = _operator_alerts_frame(current_health, market_proof_warnings, activation_readiness)
        st.markdown('**Alerts**')
        st.caption('This is the condensed operational alert stack for the current phase view.')
        st.dataframe(_display_frame(alerts_frame), use_container_width=True, hide_index=True)
        st.divider()
        freshness = _validation_freshness_frame(clean_backtest, live_archive, latest)
        if not freshness.empty:
            st.markdown('**Validation Source Freshness**')
            st.caption('Technical: these are separate validation sources with different update cadences. Plain English: the replay baseline is your long-memory benchmark, while live settled is the one that should keep moving during the regular season.')
            with st.expander('View Validation Source Detail', expanded=False):
                st.dataframe(_display_frame(freshness), use_container_width=True, hide_index=True)
        validation_delta = _validation_delta_frame(latest)
        if not validation_delta.empty:
            st.markdown('**Research vs Live Validation**')
            st.caption('Technical: this compares the slower replay benchmark against the current live settled track. Plain English: it shows whether the model is performing better or worse in actual regular-season play than the frozen research baseline.')
            with st.expander('View Validation Benchmark Detail', expanded=False):
                st.dataframe(_display_frame(validation_delta), use_container_width=True, hide_index=True)
        yesterday_change = _yesterday_change_frame(run_history)
        if not yesterday_change.empty:
            st.markdown('**What Changed Since Yesterday**')
            st.caption('Technical: this compares the latest stored run against the prior report date. Plain English: it is the day-over-day pulse check for live validation, priced-bet evidence, and totals proof.')
            with st.expander('View Day-over-Day Detail', expanded=False):
                st.dataframe(_display_frame(yesterday_change), use_container_width=True, hide_index=True)
        market_policy_frame = _market_policy_overview_frame(latest)
        if not market_policy_frame.empty:
            st.caption('EXECUTION')
            st.markdown('**Market Policy Snapshot**')
            policy_cols = st.columns(4)
            policy_cols[0].metric('Policy Alignment', str(latest.get('market_policy_alignment_status') or 'N/A'))
            policy_cols[1].metric('Real-Market Active', f"{int(latest.get('market_policy_real_market_count') or 0)}/3")
            policy_cols[2].metric('Policy Changes', int(latest.get('market_policy_change_count') or 0))
            strength_mix = f"H {int(latest.get('market_policy_high_strength_count') or 0)} | M {int(latest.get('market_policy_medium_strength_count') or 0)} | L {int(latest.get('market_policy_low_strength_count') or 0)}"
            policy_cols[3].metric('Evidence Mix', strength_mix)
            technical_note = str(latest.get('market_policy_summary_technical') or '').strip()
            plain_note = str(latest.get('market_policy_summary_plain_english') or '').strip()
            snapshot_note = str(latest.get('market_policy_market_snapshot') or '').strip()
            if plain_note:
                st.write(f"Plain English: {plain_note}")
            if technical_note:
                st.caption(f"Technical: {technical_note}")
            if snapshot_note:
                st.caption(snapshot_note)
            with st.expander('View Market Policy Detail', expanded=False):
                st.dataframe(_display_frame(market_policy_frame), use_container_width=True, hide_index=True)
        else:
            st.info('Market-aware policy status will appear here once the stored run summary includes per-market recommendation fields.')

        if not current_health.empty:
            st.caption('HEALTH')
            st.markdown('**Current Health**')
            st.caption('Technical: this is a typed health panel for model quality, market proof, portfolio concentration, and evidence readiness. Plain English: it is the fast answer to whether the system looks healthy, cautious, or still just building sample.')
            st.dataframe(_display_frame(current_health), use_container_width=True, hide_index=True)
        if not readiness_ladder.empty:
            st.caption('EVIDENCE')
            st.markdown('**Market Readiness Ladder**')
            readiness_cols = st.columns(3)
            leader = readiness_ladder.iloc[0]
            ready_count = int((readiness_ladder['Readiness'].astype(str).isin(['Primary ready', 'High-confidence ready'])).sum())
            readiness_cols[0].metric('Closest to Takeover', str(leader.get('Market') or 'N/A'))
            readiness_cols[1].metric('Markets Ready', f"{ready_count}/3")
            readiness_cols[2].metric('Leader Progress', f"{_fmt_num(leader.get('Takeover progress %'), 1)}%")
            st.caption('Technical: this is the market-by-market progress ladder toward real-market takeover. Plain English: it tells you which betting market is closest to earning enough settled priced-bet history to stand on its own.')
            if px is not None:
                _show_plotly(px.bar(readiness_ladder, x='Market', y='Takeover progress %', color='Readiness', title='Market Readiness for Real-Market Takeover'))
            with st.expander('View Readiness Ladder Detail', expanded=False):
                st.dataframe(_display_frame(readiness_ladder), use_container_width=True, hide_index=True)

        if not activation_readiness.empty:
            st.markdown('**Activation Readiness**')
            st.caption('Technical: these are the key self-learning gates waiting to activate as regular-season sample accumulates. Plain English: this shows what is still waiting on proof before the model lets itself trust more advanced adjustments.')
            with st.expander('View Activation Gate Detail', expanded=False):
                st.dataframe(_display_frame(activation_readiness), use_container_width=True, hide_index=True)

        if not evidence_drift.empty:
            st.markdown('**Evidence Quality Drift**')
            drift_cols = st.columns(3)
            strengthening = int((evidence_drift['Trend'].astype(str) == 'Strengthening').sum())
            weakening = int((evidence_drift['Trend'].astype(str) == 'Weakening').sum())
            drift_cols[0].metric('Strengthening Markets', strengthening)
            drift_cols[1].metric('Weakening Markets', weakening)
            drift_cols[2].metric('Stable Markets', int((evidence_drift['Trend'].astype(str) == 'Stable').sum()))
            st.caption('Technical: this compares each market against its prior stored snapshot to see whether evidence strength and takeover progress are improving or stalling. Plain English: it tells you whether trust in each market is moving in the right direction or just standing still.')
            if px is not None:
                _show_plotly(px.bar(evidence_drift, x='Market', y='Progress delta', color='Trend', title='Evidence Quality Trend by Market'))
            with st.expander('View Evidence Drift Detail', expanded=False):
                st.dataframe(_display_frame(evidence_drift), use_container_width=True, hide_index=True)

        realized_uncertainty_frame = _realized_uncertainty_frame(latest)
        if not realized_uncertainty_frame.empty:
            st.markdown('**Realized Uncertainty Learning**')
            st.caption('Technical: this shows the market-level bias learned from settled live predictions and settled priced bets. Plain English: it tells you where the model has learned to trust itself less or a little more based on actual outcomes and closing-line behavior.')
            with st.expander('View Uncertainty Learning Detail', expanded=False):
                st.dataframe(_display_frame(realized_uncertainty_frame), use_container_width=True, hide_index=True)

        if not market_proof_warnings.empty:
            st.markdown('**Market Proof Interpretation**')
            st.caption('Technical: these warnings catch combinations like positive ROI with negative CLV, which often means short-term profit is not yet trustworthy. Plain English: this is the panel that says when the model may be winning in a way that still deserves skepticism.')
            with st.expander('View Market Proof Warning Detail', expanded=False):
                st.dataframe(_display_frame(market_proof_warnings), use_container_width=True, hide_index=True)

        st.markdown('**Totals Market Proof Track**')
        totals_cols = st.columns(4)
        totals_cols[0].metric('Settled Totals Bets', int(latest.get('market_proof_total_bets') or 0))
        totals_cols[1].metric('Totals P&L', _fmt_num(latest.get('market_proof_total_units'), 2) + 'u')
        totals_cols[2].metric('Totals ROI', _fmt_pct(latest.get('market_proof_total_roi')))
        totals_cols[3].metric('Totals Average CLV', _fmt_num(latest.get('market_proof_total_avg_clv'), 3))
        st.caption('Technical: this is the totals-only real-market proof line. Plain English: it shows whether totals themselves are earning trust once priced bets settle, instead of being buried inside the combined market scorecard.')
        if not run_history.empty and 'market_proof_total_roi' in run_history.columns and pd.to_numeric(run_history['market_proof_total_roi'], errors='coerce').notna().any():
            totals_history = run_history.copy()
            totals_history['report_date'] = pd.to_datetime(totals_history['report_date'], errors='coerce')
            totals_history = totals_history.sort_values('report_date')
            left, right = st.columns(2)
            with left:
                _render_line_chart(totals_history, 'report_date', 'market_proof_total_roi', 'Totals Market-Proof ROI')
            with right:
                _render_line_chart(totals_history, 'report_date', 'market_proof_total_avg_clv', 'Totals Market-Proof CLV')
        elif not market_proof_by_market_bundle.empty:
            totals_market_bundle = market_proof_by_market_bundle[market_proof_by_market_bundle['market_type'].astype(str).str.lower() == 'total'].copy()
            if not totals_market_bundle.empty:
                st.dataframe(_display_frame(totals_market_bundle), use_container_width=True, hide_index=True)

        if not run_history.empty:
            history = run_history.copy()
            history['report_date'] = pd.to_datetime(history['report_date'], errors='coerce')
            st.caption('The trend charts below use one canonical stored run per report date, so same-day reruns do not create artificial daily noise.')
            left, right = st.columns(2)
            with left:
                _render_line_chart(history.sort_values('report_date'), 'report_date', 'validation_log_loss', 'Research Moneyline OOS Log Loss')
            with right:
                live_history = history.sort_values('report_date')
                if 'live_validation_log_loss' in live_history.columns and pd.to_numeric(live_history['live_validation_log_loss'], errors='coerce').notna().any():
                    _render_line_chart(live_history, 'report_date', 'live_validation_log_loss', 'Live Settled Moneyline Log Loss')
                else:
                    _render_line_chart(live_history, 'report_date', 'validation_total_rmse', 'Research Totals OOS RMSE')
            left, right = st.columns(2)
            with left:
                weight_frame = pd.DataFrame([
                    {'weight': 'offense', 'value': latest.get('w_off')},
                    {'weight': 'starter', 'value': latest.get('w_starter')},
                    {'weight': 'bullpen', 'value': latest.get('w_bullpen')},
                    {'weight': 'park', 'value': latest.get('w_park')},
                    {'weight': 'weather', 'value': latest.get('w_weather')},
                    {'weight': 'context', 'value': latest.get('w_context')},
                    {'weight': 'home_field', 'value': latest.get('home_field')},
                ]).dropna()
                weight_frame = _display_frame(weight_frame, {'weight': 'Model Component', 'value': 'Weight'})
                _render_bar_chart(weight_frame, 'Model Component', 'Weight', 'Current Model Weight Profile')
            with right:
                shrink_frame = pd.DataFrame([
                    {'signal': 'probability', 'alpha': latest.get('probability_shrink_alpha')},
                    {'signal': 'total', 'alpha': latest.get('total_shrink_alpha')},
                    {'signal': 'margin', 'alpha': latest.get('margin_shrink_alpha')},
                ]).fillna(0.0)
                shrink_frame = _display_frame(shrink_frame, {'signal': 'Signal', 'alpha': 'Alpha'})
                _render_bar_chart(shrink_frame, 'Signal', 'Alpha', 'Current Shrinkage Profile')
        else:
            st.info('Run history will appear here once the DuckDB analytics store has multiple snapshots.')

    with tabs[1]:
        st.subheader("Today's Slate")
        if predictions.empty:
            st.info("No current slate snapshot is available yet.")
        else:
            slate = predictions.copy()
            confidence_options = ['All'] + sorted([str(value) for value in slate.get('confidence_tier', pd.Series(dtype=str)).dropna().unique().tolist()])
            selected_confidence = st.selectbox('Confidence Tier', confidence_options, index=0)
            if selected_confidence != 'All' and 'confidence_tier' in slate.columns:
                slate = slate[slate['confidence_tier'].astype(str) == selected_confidence].copy()
            if 'margin_calibrated' in slate.columns:
                margin_size = pd.to_numeric(slate['margin_calibrated'], errors='coerce').abs().fillna(0.0)
                slate['margin_size'] = margin_size.clip(lower=0.05)
            display_cols = [
                col for col in [
                    'away_team', 'home_team', 'winner', 'winner_prob', 'confidence_tier', 'fair_away_ml', 'fair_home_ml',
                    'total_calibrated', 'margin_calibrated', 'probability_shrink_alpha', 'total_shrink_alpha', 'margin_shrink_alpha',
                    'away_lineup_count', 'home_lineup_count', 'market_label', 'sportsbook_count',
                    'moneyline_thesis', 'run_line_thesis', 'total_thesis'
                ] if col in slate.columns
            ]
            slate_display = slate[display_cols].sort_values(['winner_prob', 'total_calibrated'], ascending=[False, False]).copy()
            slate_display = _display_frame(slate_display, {
                'away_team': 'Away Team',
                'home_team': 'Home Team',
                'winner': 'Model Favorite',
                'winner_prob': 'Win Probability',
                'confidence_tier': 'Confidence Tier',
                'fair_away_ml': 'Fair Away Moneyline',
                'fair_home_ml': 'Fair Home Moneyline',
                'total_calibrated': 'Projected Total',
                'margin_calibrated': 'Projected Margin',
                'probability_shrink_alpha': 'Probability Alpha',
                'total_shrink_alpha': 'Totals Alpha',
                'margin_shrink_alpha': 'Margin Alpha',
                'away_lineup_count': 'Away Lineup Spots',
                'home_lineup_count': 'Home Lineup Spots',
                'market_label': 'Market View',
                'sportsbook_count': 'Sportsbooks',
                'moneyline_thesis': 'Moneyline Thesis',
                'run_line_thesis': 'Run-Line Thesis',
                'total_thesis': 'Totals Thesis',
            })
            with st.expander('View Full Slate Table', expanded=False):
                st.dataframe(slate_display, use_container_width=True, hide_index=True)
            if px is not None and {'winner_prob', 'total_calibrated', 'confidence_tier'}.issubset(slate.columns):
                scatter = px.scatter(
                    slate,
                    x='winner_prob',
                    y='total_calibrated',
                    color='confidence_tier',
                    size='margin_size' if 'margin_size' in slate.columns else None,
                    hover_data=['away_team', 'home_team', 'winner'],
                    title='Slate Map: Win Probability vs. Projected Total',
                    labels={'winner_prob': 'Win Probability', 'total_calibrated': 'Projected Total', 'confidence_tier': 'Confidence Tier'},
                )
                _show_plotly(scatter, height=420)
            slate_explainer = slate.copy()
            slate_explainer['matchup'] = slate_explainer['away_team'].astype(str) + ' @ ' + slate_explainer['home_team'].astype(str)
            selected_slate_matchup = st.selectbox('Game Thesis', slate_explainer['matchup'].tolist(), key='slate_bet_explainer')
            chosen_slate = slate_explainer[slate_explainer['matchup'] == selected_slate_matchup].head(1)
            if not chosen_slate.empty:
                chosen_row = chosen_slate.iloc[0]
                st.markdown('**Game Thesis Summary**')
                st.caption('Moneyline view: ' + str(chosen_row.get('moneyline_thesis') or 'No moneyline explanation is stored for this game.'))
                st.caption('Run-line view: ' + str(chosen_row.get('run_line_thesis') or 'No run-line explanation is stored for this game.'))
                st.caption('Totals view: ' + str(chosen_row.get('total_thesis') or 'No totals explanation is stored for this game.'))

    with tabs[2]:
        st.subheader('Portfolio Construction')
        if portfolio_summary.empty and portfolio_recommendations.empty and candidates.empty and live_archive.empty and bet_journal.empty:
            st.info('No portfolio plan is available yet. This section will populate once market lines exist and the model creates actionable positions.')
        else:
            summary_row = {} if portfolio_summary.empty else portfolio_summary.iloc[0].to_dict()
            phase_name = str((summary_row.get('phase') if summary_row else None) or latest.get('phase') or model_state.get('season_phase') or 'spring').lower()
            default_unit_risk_pct = round(((_safe_float(summary_row.get('base_unit_risk_pct')) if summary_row else None) or (0.0075 if phase_name == 'spring' else 0.01)) * 100.0, 2)
            if summary_row:
                portfolio_cols = st.columns(4)
                portfolio_cols[0].metric('Daily Risk Budget', f"{_safe_float(summary_row.get('daily_budget_units')) or 0.0:.2f}u")
                portfolio_cols[1].metric('Raw Demand', f"{_safe_float(summary_row.get('raw_demand_units')) or 0.0:.2f}u")
                portfolio_cols[2].metric('Allocated Units', f"{_safe_float(summary_row.get('allocated_units')) or 0.0:.2f}u")
                portfolio_cols[3].metric('Utilization', _fmt_pct(summary_row.get('utilization_pct')))

                top_cols = st.columns(4)
                top_cols[0].metric('Allocated Positions', int(summary_row.get('allocated_candidate_count') or 0))
                top_cols[1].metric('Watch / Bet / Must Take', f"{int(summary_row.get('watch_count') or 0)} / {int(summary_row.get('bet_count') or 0)} / {int(summary_row.get('must_take_count') or 0)}")
                top_cols[2].metric('Per-Game Cap', f"{_safe_float(summary_row.get('max_game_units')) or 0.0:.2f}u")
                top_cols[3].metric('Per-Team Cap', f"{_safe_float(summary_row.get('max_team_units')) or 0.0:.2f}u")

                corr_cols = st.columns(4)
                corr_cols[0].metric('Correlated Positions', int(summary_row.get('correlated_candidate_count') or 0))
                corr_cols[1].metric('Average Final Correlation Penalty', _fmt_num(summary_row.get('avg_correlation_penalty'), 3))
                corr_cols[2].metric('Average Pairwise Correlation', _fmt_num(summary_row.get('avg_pairwise_correlation'), 3))
                corr_cols[3].metric('Peak Pairwise Correlation', _fmt_num(summary_row.get('peak_pairwise_correlation'), 3))

                concentration_cols = st.columns(4)
                concentration_cols[0].metric('Largest Game Concentration', _fmt_pct(summary_row.get('largest_game_exposure_pct')))
                concentration_cols[1].metric('Largest Team Concentration', _fmt_pct(summary_row.get('largest_team_exposure_pct')))
                concentration_cols[2].metric('Average Matrix Penalty', _fmt_num(summary_row.get('avg_matrix_correlation_penalty'), 3))
                concentration_cols[3].metric('High-Overlap Positions', int(summary_row.get('high_overlap_candidate_count') or 0))

                uncertainty_cols = st.columns(4)
                uncertainty_cols[0].metric('Average Uncertainty Score', _fmt_num(summary_row.get('avg_uncertainty_score'), 1))
                uncertainty_cols[1].metric('High-Uncertainty Positions', int(summary_row.get('high_uncertainty_candidate_count') or 0))
                uncertainty_cols[2].metric('Moneyline Cap', f"{_safe_float(summary_row.get('max_moneyline_units')) or 0.0:.2f}u")
                uncertainty_cols[3].metric('Totals / Run-Line Caps', f"{_safe_float(summary_row.get('max_total_units')) or 0.0:.2f}u / {_safe_float(summary_row.get('max_run_line_units')) or 0.0:.2f}u")
            else:
                st.info('A full portfolio summary is not available yet, so this view is temporarily falling back to raw candidate data.')

            control_cols = st.columns(4)
            bankroll = control_cols[0].number_input('Simulated Bankroll ($)', min_value=100.0, value=5000.0, step=100.0, key='portfolio_bankroll')
            unit_risk_pct = control_cols[1].number_input('Unit Risk (%)', min_value=0.10, value=float(default_unit_risk_pct), step=0.05, key='portfolio_unit_pct')
            selected_policy = control_cols[2].selectbox('Sizing Policy', list(_policy_specs().keys()), index=2, key='portfolio_policy')
            selected_posture = control_cols[3].selectbox('Risk posture', list(_risk_posture_specs().keys()), index=1, key='portfolio_posture')
            deployment_scalar = st.slider('Deployment Override', min_value=0.25, max_value=1.50, value=1.00, step=0.05, key='portfolio_deployment_scalar')

            policy_recommendation_view, policy_summary, policy_market_exposure, policy_team_exposure, policy_game_exposure = _apply_policy_to_recommendations(
                summary_row,
                portfolio_recommendations,
                selected_policy,
                selected_posture,
                bankroll,
                unit_risk_pct,
                deployment_scalar,
            )
            unit_size_dollars = float(_safe_float(policy_summary.get('unit_size_dollars')) or 0.0)
            deployed_risk_dollars = float(_safe_float(policy_summary.get('deployed_risk_dollars')) or 0.0)
            expected_value_dollars = float(_safe_float(policy_summary.get('expected_value_dollars')) or 0.0)
            max_loss_dollars = float(_safe_float(policy_summary.get('max_loss_dollars')) or 0.0)
            game_risk_dollars = float(_safe_float(policy_summary.get('game_risk_dollars')) or 0.0)
            team_risk_dollars = float(_safe_float(policy_summary.get('team_risk_dollars')) or 0.0)

            sim_metrics = st.columns(5)
            sim_metrics[0].metric('Unit Size (1u Risk)', f"${unit_size_dollars:,.2f}")
            sim_metrics[1].metric('Capital at Risk', f"${deployed_risk_dollars:,.2f}")
            sim_metrics[2].metric('Expected Value', f"${expected_value_dollars:,.2f}")
            sim_metrics[3].metric('Worst-Case Day', f"-${max_loss_dollars:,.2f}")
            sim_metrics[4].metric('Top Game / Team Risk', f"${game_risk_dollars:,.2f} / ${team_risk_dollars:,.2f}")

            st.caption(
                'Technical policy note: ' + str(policy_summary.get('policy_description') or 'No policy description is available.') +
                ' Plain English: ' + str(policy_summary.get('plain_english') or 'No plain-English explanation is available.')
            )
            plain_english_summary = _portfolio_plain_english(
                summary_row.get('utilization_pct') if summary_row else None,
                summary_row.get('correlated_candidate_count') if summary_row else 0,
                summary_row.get('largest_game_exposure_pct') if summary_row else None,
                summary_row.get('largest_team_exposure_pct') if summary_row else None,
            )
            st.caption('Slate concentration summary: ' + plain_english_summary)
            if not policy_recommendation_view.empty:
                st.caption('Technical: raw EV shows the theoretical edge before uncertainty; adjusted EV shows the edge the allocator is actually willing to trust after setup-specific drag and regime-aware calibration. Plain English: this separates a good number on paper from a good number after the model worries about shaky assumptions and overlap risk.')
                raw_ev_avg = pd.to_numeric(policy_recommendation_view.get('raw_expected_value_per_unit'), errors='coerce').fillna(0.0).mean() if 'raw_expected_value_per_unit' in policy_recommendation_view.columns else 0.0
                adjusted_ev_avg = pd.to_numeric(policy_recommendation_view.get('adjusted_expected_value_per_unit'), errors='coerce').fillna(0.0).mean() if 'adjusted_expected_value_per_unit' in policy_recommendation_view.columns else 0.0
                uncertainty_avg = pd.to_numeric(policy_recommendation_view.get('uncertainty_score'), errors='coerce').fillna(0.0).mean() if 'uncertainty_score' in policy_recommendation_view.columns else 0.0
                uncertainty_bias_avg = pd.to_numeric(policy_recommendation_view.get('uncertainty_calibration_bias'), errors='coerce').fillna(0.0).mean() if 'uncertainty_calibration_bias' in policy_recommendation_view.columns else 0.0
                penalty_scale_avg = pd.to_numeric(policy_recommendation_view.get('uncertainty_penalty_scale'), errors='coerce').fillna(0.0).mean() if 'uncertainty_penalty_scale' in policy_recommendation_view.columns else 0.0
                ev_cols = st.columns(5)
                ev_cols[0].metric('Average Raw EV / Unit', _fmt_num(raw_ev_avg, 3))
                ev_cols[1].metric('Average Adjusted EV / Unit', _fmt_num(adjusted_ev_avg, 3))
                ev_cols[2].metric('Average Position Uncertainty', _fmt_num(uncertainty_avg, 1))
                ev_cols[3].metric('Average Calibration Bias', _fmt_num(uncertainty_bias_avg, 3))
                ev_cols[4].metric('Average Penalty Scale', _fmt_num(penalty_scale_avg, 3))
                st.caption('Technical EV note: raw EV is the unadjusted price edge; adjusted EV is raw EV after the uncertainty haircut, market-specific regime drag, and overlap-aware sizing filters.')
                st.caption('Plain English: a bet can look good on paper, but adjusted EV shows what is left after the model discounts fragile assumptions and crowded positions.')

            realized_uncertainty_frame = _realized_uncertainty_frame(latest)
            if not realized_uncertainty_frame.empty:
                st.markdown('**Realized Uncertainty Learning**')
                st.caption('Technical: this is the market-level uncertainty loop learning from settled miss size, CLV, and realized returns. Plain English: it is the part of the quant model that says “this market has recently been shakier than it looked” or “this market has earned a little more trust.”')
                st.dataframe(realized_uncertainty_frame, use_container_width=True, hide_index=True)
            scenario_frame = _build_portfolio_scenarios(
                summary_row,
                portfolio_recommendations,
                bankroll,
                unit_risk_pct,
                deployment_scalar,
                selected_posture,
            )
            st.markdown('**Sizing Policy Comparison**')
            st.caption('Technical: this compares the three sizing policies at the selected risk posture. Plain English: it shows how the same slate would feel under simple, edge-sensitive, and risk-capped money management.')
            if not scenario_frame.empty:
                scenario_display = scenario_frame.copy()
                scenario_display['technical view'] = scenario_display['technical_note']
                scenario_display['plain-English view'] = scenario_display['plain_english']
                scenario_display = _display_frame(scenario_display, {
                    'scenario': 'Policy',
                    'unit_risk_pct': 'Unit Risk Pct',
                    'deployment_scalar': 'Deployment Scalar',
                    'unit_size_dollars': 'Unit Size ($)',
                    'deployed_risk_dollars': 'Risk Deployed ($)',
                    'expected_value_dollars': 'Expected Value ($)',
                    'max_loss_dollars': 'Worst-Case Loss ($)',
                    'game_risk_dollars': 'Top Game Risk ($)',
                    'team_risk_dollars': 'Top Team Risk ($)',
                    'avg_correlation_penalty': 'Average Correlation Penalty',
                    'avg_pairwise_correlation': 'Average Pairwise Correlation',
                    'avg_matrix_correlation_penalty': 'Average Matrix Penalty',
                    'technical view': 'Technical Summary',
                    'plain-English view': 'Plain-English Summary',
                })
                st.dataframe(
                    scenario_display[[
                        'Policy', 'Unit Risk Pct', 'Deployment Scalar', 'Unit Size ($)', 'Risk Deployed ($)',
                        'Expected Value ($)', 'Worst-Case Loss ($)', 'Top Game Risk ($)', 'Top Team Risk ($)',
                        'Average Correlation Penalty', 'Average Pairwise Correlation', 'Average Matrix Penalty', 'Technical Summary', 'Plain-English Summary'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                if px is not None:
                    scenario_chart = scenario_frame.melt(
                        id_vars='scenario',
                        value_vars=['deployed_risk_dollars', 'expected_value_dollars', 'max_loss_dollars'],
                        var_name='metric',
                        value_name='dollars',
                    )
                    _show_plotly(
                        px.bar(
                            scenario_chart,
                            x='scenario',
                            y='dollars',
                            color='metric',
                            barmode='group',
                            title='Policy Comparison: Risk, Expected Value, and Worst-Case Loss',
                            labels={'scenario': 'Policy', 'dollars': 'Dollars', 'metric': 'Metric'},
                        )
                    )
            else:
                st.info('Policy analysis will populate once the allocator has active positions to size.')

            with st.expander('How to read this panel', expanded=False):
                st.markdown(
                    """
- `Unit size (technical: 1u risk)`: the dollar value of one model unit. In plain English, this is how much you lose if a 1-unit bet loses.
- `Deployed risk`: total dollars currently committed across the slate. In plain English, this is your money in action today.
- `Expected value`: the model's long-run profit estimate if these prices are real edges. In plain English, this is the amount the model expects to earn on average over many slates, not a promise for today.
- `Worst-case day`: the loss if every allocated position loses. In plain English, this is the bad-day stress test.
- `Sizing policy`: the rule set that converts model conviction into stake size. In plain English, this controls whether the app sizes everything evenly, leans into stronger edges, or leans into them while capping concentration harder.
- `Risk posture`: the overall throttle on the whole slate. In plain English, this is your cautious / normal / aggressive dial.
- `Correlation penalty`: the trim applied when multiple bets depend on the same game, team, or a strongly linked market story. In plain English, the model now checks pairwise overlap instead of only crude count caps so one cluster cannot quietly dominate the slate.
                    """
                )

            exposure_left, exposure_right = st.columns(2)
            with exposure_left:
                if not policy_market_exposure.empty:
                    market_exposure_display = _display_frame(policy_market_exposure, {'bucket': 'Market', 'policy_risk_dollars': 'Risk ($)'})
                    _render_bar_chart(market_exposure_display, 'Market', 'Risk ($)', 'Allocated Risk by Market ($)')
                elif not candidates.empty:
                    by_market = candidates.groupby('market_type', dropna=False)['suggested_units_preview'].sum().reset_index()
                    by_market = _display_frame(by_market, {'market_type': 'Market', 'suggested_units_preview': 'Suggested Units'})
                    _render_bar_chart(by_market, 'Market', 'Suggested Units', 'Suggested Units by Market')
                else:
                    st.info('No market exposure data is available yet.')
            with exposure_right:
                if not policy_team_exposure.empty:
                    team_exposure_display = _display_frame(policy_team_exposure.head(10), {'bucket': 'Team', 'policy_risk_dollars': 'Risk ($)'})
                    _render_bar_chart(team_exposure_display, 'Team', 'Risk ($)', 'Top Team Exposure ($)')
                elif not candidates.empty:
                    by_label = candidates.groupby('actionability_label', dropna=False)['suggested_units_preview'].sum().reset_index()
                    by_label = _display_frame(by_label, {'actionability_label': 'Actionability Tier', 'suggested_units_preview': 'Suggested Units'})
                    _render_bar_chart(by_label, 'Actionability Tier', 'Suggested Units', 'Suggested Units by Recommendation Tier')
                else:
                    st.info('No team exposure data is available yet.')

            exposure_left, exposure_right = st.columns(2)
            with exposure_left:
                if not policy_game_exposure.empty:
                    game_exposure_display = _display_frame(policy_game_exposure.head(10), {'bucket': 'Game', 'policy_risk_dollars': 'Risk ($)'})
                    _render_bar_chart(game_exposure_display, 'Game', 'Risk ($)', 'Top Game Exposure ($)')
                else:
                    st.info('No game-level exposure data is available yet.')
            with exposure_right:
                if not policy_recommendation_view.empty and {'market_type', 'allocated_units', 'policy_units'}.issubset(policy_recommendation_view.columns):
                    demand_compare = policy_recommendation_view.groupby('market_type', dropna=False)[['allocated_units', 'policy_units']].sum().reset_index()
                    if px is not None:
                        melted = demand_compare.melt(id_vars='market_type', value_vars=['allocated_units', 'policy_units'], var_name='measure', value_name='units')
                        _show_plotly(px.bar(melted, x='market_type', y='units', color='measure', barmode='group', title='Base Allocation vs. Policy Allocation'))
                    else:
                        st.dataframe(demand_compare, use_container_width=True, hide_index=True)
                else:
                    st.info('Base-versus-policy allocation comparison will appear once a portfolio plan exists.')

            if not policy_recommendation_view.empty:
                if px is not None and {'adjusted_expected_value_per_unit', 'policy_units', 'market_type'}.issubset(policy_recommendation_view.columns):
                    bubble = px.scatter(
                        policy_recommendation_view,
                        x='adjusted_expected_value_per_unit',
                        y='policy_units',
                        color='market_type',
                        size='actionability_score' if 'actionability_score' in policy_recommendation_view.columns else None,
                        hover_data=['away_team', 'home_team', 'selection', 'raw_expected_value_per_unit', 'adjusted_expected_value_per_unit', 'uncertainty_score', 'uncertainty_label', 'uncertainty_calibration_bias', 'avg_active_correlation', 'max_active_correlation', 'policy_note', 'cap_notes'],
                        title='Risk-Adjusted Portfolio Recommendations',
                        labels={'adjusted_expected_value_per_unit': 'Adjusted EV / Unit', 'policy_units': 'Policy Units', 'market_type': 'Market'},
                    )
                    _show_plotly(bubble, height=440)
                table_cols = [
                    col for col in [
                        'away_team', 'home_team', 'market_type', 'selection', 'allocated_units', 'policy_units', 'policy_risk_dollars',
                        'policy_expected_value_dollars', 'raw_expected_value_per_unit', 'adjusted_expected_value_per_unit', 'edge_value', 'uncertainty_score', 'uncertainty_label', 'uncertainty_calibration_bias', 'uncertainty_penalty_scale', 'actionability_label', 'actionability_score',
                        'same_game_peer_count', 'same_team_peer_count', 'avg_active_correlation', 'max_active_correlation', 'matrix_correlation_penalty', 'final_correlation_penalty', 'policy_factor', 'confidence_tier', 'uncertainty_notes',
                        'policy_note', 'cap_notes', 'bet_thesis', 'bet_drivers', 'risk_flags', 'market_throttle', 'market_throttle_note'
                    ] if col in policy_recommendation_view.columns
                ]
                recommendation_display = policy_recommendation_view[table_cols].sort_values(['policy_units', 'adjusted_expected_value_per_unit'], ascending=[False, False]).copy()
                recommendation_display = _display_frame(recommendation_display, {
                    'away_team': 'Away Team',
                    'home_team': 'Home Team',
                    'market_type': 'Market',
                    'selection': 'Selection',
                    'allocated_units': 'Base Units',
                    'policy_units': 'Policy Units',
                    'policy_risk_dollars': 'Risk ($)',
                    'policy_expected_value_dollars': 'Expected Value ($)',
                    'edge_value': 'Edge',
                    'uncertainty_label': 'Uncertainty Tier',
                    'actionability_label': 'Actionability Tier',
                    'actionability_score': 'Actionability Score',
                    'same_game_peer_count': 'Same-Game Peers',
                    'same_team_peer_count': 'Same-Team Peers',
                    'policy_factor': 'Policy Factor',
                    'confidence_tier': 'Confidence Tier',
                    'uncertainty_notes': 'Uncertainty Notes',
                    'policy_note': 'Policy Notes',
                    'cap_notes': 'Cap Notes',
                    'bet_thesis': 'Plain-English Thesis',
                    'bet_drivers': 'Technical Thesis',
                    'risk_flags': 'Risk Flags',
                    'market_throttle': 'Market Throttle',
                    'market_throttle_note': 'Throttle Notes',
                })
                with st.expander('View Full Recommendation Table', expanded=False):
                    st.dataframe(recommendation_display, use_container_width=True, hide_index=True)
                explanation_source = policy_recommendation_view.copy()
                explanation_source['matchup'] = explanation_source['away_team'].astype(str) + ' @ ' + explanation_source['home_team'].astype(str) + ' | ' + explanation_source['selection'].astype(str)
                selected_matchup = st.selectbox('Bet Rationale', explanation_source['matchup'].tolist(), key='portfolio_bet_explainer')
                chosen = explanation_source[explanation_source['matchup'] == selected_matchup].head(1)
                if not chosen.empty:
                    chosen_row = chosen.iloc[0].copy()
                    needs_fallback = not any([
                        str(chosen_row.get('bet_drivers') or '').strip(),
                        str(chosen_row.get('bet_thesis') or '').strip(),
                        str(chosen_row.get('risk_flags') or '').strip(),
                        str(chosen_row.get('market_throttle_note') or '').strip(),
                    ])
                    if needs_fallback and not candidates.empty:
                        fallback = candidates[
                            (candidates.get('away_team').astype(str) == str(chosen_row.get('away_team') or ''))
                            & (candidates.get('home_team').astype(str) == str(chosen_row.get('home_team') or ''))
                            & (candidates.get('market_type').astype(str) == str(chosen_row.get('market_type') or ''))
                            & (candidates.get('selection').astype(str) == str(chosen_row.get('selection') or ''))
                        ].head(1)
                        if not fallback.empty:
                            for field in ['bet_drivers', 'bet_thesis', 'risk_flags', 'market_throttle_note']:
                                if not str(chosen_row.get(field) or '').strip():
                                    chosen_row[field] = fallback.iloc[0].get(field)
                    st.caption('Technical Summary: ' + str(chosen_row.get('bet_drivers') or 'No technical summary is available.'))
                    st.caption('Plain-English Summary: ' + str(chosen_row.get('bet_thesis') or 'No plain-English summary is available.'))
                    st.caption('Risk flags: ' + str(chosen_row.get('risk_flags') or 'No special risk flags noted.'))
                    st.caption('Throttle note: ' + str(chosen_row.get('market_throttle_note') or 'No throttle note is available.'))
            elif not candidates.empty:
                st.info('No policy-adjusted allocation was produced on this run, so the table below shows raw candidates instead.')
                st.caption('Technical: totals currently use a cautious run-gap EV proxy until a direct totals-cover probability model is added. Plain English: totals are still being sized carefully because their EV is less directly modeled than moneyline and run line.')
                table_cols = [
                    col for col in [
                        'away_team', 'home_team', 'market_type', 'selection', 'book_price', 'edge_value', 'raw_expected_value_per_unit', 'adjusted_expected_value_per_unit', 'expected_value_method',
                        'uncertainty_score', 'uncertainty_base_score', 'uncertainty_calibration_bias', 'uncertainty_penalty_scale', 'uncertainty_label', 'uncertainty_notes', 'actionability_label', 'actionability_score', 'suggested_units_preview', 'confidence_tier',
                        'bet_thesis', 'bet_drivers', 'risk_flags', 'market_throttle', 'market_throttle_note'
                    ] if col in candidates.columns
                ]
                raw_candidate_display = candidates[table_cols].sort_values(['suggested_units_preview', 'candidate_score'], ascending=[False, False]).copy()
                raw_candidate_display = _display_frame(raw_candidate_display, {
                    'away_team': 'Away Team',
                    'home_team': 'Home Team',
                    'market_type': 'Market',
                    'selection': 'Selection',
                    'book_price': 'Book Price',
                    'edge_value': 'Edge',
                    'expected_value_method': 'EV Method',
                    'uncertainty_label': 'Uncertainty Tier',
                    'actionability_label': 'Actionability Tier',
                    'suggested_units_preview': 'Suggested Units',
                    'confidence_tier': 'Confidence Tier',
                    'bet_thesis': 'Plain-English Thesis',
                    'bet_drivers': 'Technical Thesis',
                    'risk_flags': 'Risk Flags',
                    'market_throttle': 'Market Throttle',
                    'market_throttle_note': 'Throttle Notes',
                })
                with st.expander('View Raw Candidate Table', expanded=False):
                    st.dataframe(raw_candidate_display, use_container_width=True, hide_index=True)

            st.markdown('**Historical Policy Simulation**')
            st.caption('Technical: this replays archived settled main-side predictions as even-money normalized shadow positions, then applies the same sizing rules you chose above. Plain English: it shows which bankroll style would have felt smoother or more aggressive if you had followed the model through the archive.')
            shadow_daily, shadow_summary = _build_shadow_policy_backtest(
                live_archive,
                bankroll,
                unit_risk_pct,
                deployment_scalar,
                selected_posture,
            )
            if not shadow_summary.empty:
                selected_shadow = shadow_summary[shadow_summary['policy_name'] == selected_policy].head(1)
                if not selected_shadow.empty:
                    shadow_row = selected_shadow.iloc[0]
                    shadow_cols = st.columns(5)
                    shadow_cols[0].metric('Shadow P&L (Units)', _fmt_num(shadow_row.get('result_units'), 2))
                    shadow_cols[1].metric('Shadow Yield', _fmt_pct(shadow_row.get('yield_pct')))
                    shadow_cols[2].metric('Max Drawdown', _fmt_num(shadow_row.get('max_drawdown_units'), 2) + 'u')
                    shadow_cols[3].metric('Settled Picks', int(shadow_row.get('settled_picks') or 0))
                    shadow_cols[4].metric('Hit Rate', _fmt_pct(shadow_row.get('hit_rate')))
                    st.caption('Selected policy note: ' + str(shadow_row.get('technical_note') or '') + ' Plain English: ' + str(shadow_row.get('plain_english') or ''))
                shadow_display = shadow_summary.copy()
                shadow_display['technical view'] = shadow_display['technical_note']
                shadow_display['plain-English view'] = shadow_display['plain_english']
                shadow_display = _display_frame(shadow_display, {
                    'policy_name': 'Policy',
                    'posture_name': 'Posture',
                    'settled_slates': 'Settled Slates',
                    'settled_picks': 'Settled Picks',
                    'hit_rate': 'Hit Rate',
                    'risk_units': 'Risk Units',
                    'result_units': 'P&L (Units)',
                    'yield_pct': 'Yield',
                    'max_drawdown_units': 'Max Drawdown',
                    'avg_daily_risk_dollars': 'Average Daily Risk ($)',
                    'technical view': 'Technical Summary',
                    'plain-English view': 'Plain-English Summary',
                })
                with st.expander('View Shadow Policy Detail', expanded=False):
                    st.dataframe(
                        shadow_display[[
                            'Policy', 'Posture', 'Settled Slates', 'Settled Picks', 'Hit Rate', 'Risk Units',
                            'P&L (Units)', 'Yield', 'Max Drawdown', 'Average Daily Risk ($)', 'Technical Summary', 'Plain-English Summary'
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )
                shadow_left, shadow_right = st.columns(2)
                with shadow_left:
                    if px is not None:
                        _show_plotly(px.line(shadow_daily, x='report_date', y='cumulative_units', color='policy_name', title='Shadow Policy Equity Curve (Units)'))
                    else:
                        st.dataframe(shadow_daily[['report_date', 'policy_name', 'cumulative_units']], use_container_width=True, hide_index=True)
                with shadow_right:
                    if px is not None:
                        dd_chart = shadow_summary.melt(
                            id_vars='policy_name',
                            value_vars=['result_units', 'max_drawdown_units'],
                            var_name='metric',
                            value_name='units',
                        )
                        _show_plotly(px.bar(dd_chart, x='policy_name', y='units', color='metric', barmode='group', title='Shadow Return vs. Drawdown'))
                    else:
                        st.dataframe(shadow_summary[['policy_name', 'result_units', 'max_drawdown_units']], use_container_width=True, hide_index=True)
            else:
                st.info('The shadow policy simulator needs settled archived predictions. It will populate automatically as the live archive grows.')

            st.markdown('**Real-Market Policy Replay**')
            st.caption('Technical: this replays settled priced journal bets under Flat, Fractional Kelly Proxy, and Risk-Capped Hybrid using the actual market prices and graded outcomes. Plain English: it shows how each bankroll style would have performed on the real bets, not just on shadow picks.')
            market_policy_daily_live, market_policy_summary_live = policy_research.build_market_policy_backtest(
                bet_journal,
                bankroll,
                unit_risk_pct,
                deployment_scalar,
                selected_posture,
            )
            if not market_policy_summary_live.empty:
                selected_market = market_policy_summary_live[market_policy_summary_live['policy_name'] == selected_policy].head(1)
                if not selected_market.empty:
                    market_row = selected_market.iloc[0]
                    market_cols = st.columns(5)
                    market_cols[0].metric('Real-Market P&L (Units)', _fmt_num(market_row.get('result_units'), 2))
                    market_cols[1].metric('Real-Market Yield', _fmt_pct(market_row.get('yield_pct')))
                    market_cols[2].metric('Max Drawdown', _fmt_num(market_row.get('max_drawdown_units'), 2) + 'u')
                    market_cols[3].metric('Settled Priced Bets', int(market_row.get('settled_picks') or 0))
                    market_cols[4].metric('Average CLV', _fmt_num(market_row.get('avg_clv'), 3))
                    st.caption('Selected real-market read: ' + str(market_row.get('technical_note') or '') + ' Plain English: ' + str(market_row.get('plain_english') or ''))
                market_display = market_policy_summary_live.copy()
                market_display['technical view'] = market_display['technical_note']
                market_display['plain-English view'] = market_display['plain_english']
                market_display = _display_frame(market_display, {
                    'policy_name': 'Policy',
                    'posture_name': 'Posture',
                    'settled_slates': 'Settled Slates',
                    'settled_picks': 'Settled Picks',
                    'hit_rate': 'Hit Rate',
                    'risk_units': 'Risk Units',
                    'result_units': 'P&L (Units)',
                    'yield_pct': 'Yield',
                    'max_drawdown_units': 'Max Drawdown',
                    'avg_clv': 'Average CLV',
                    'avg_daily_risk_dollars': 'Average Daily Risk ($)',
                    'technical view': 'Technical Summary',
                    'plain-English view': 'Plain-English Summary',
                })
                with st.expander('View Real-Market Policy Detail', expanded=False):
                    st.dataframe(
                        market_display[[
                            'Policy', 'Posture', 'Settled Slates', 'Settled Picks', 'Hit Rate', 'Risk Units',
                            'P&L (Units)', 'Yield', 'Max Drawdown', 'Average CLV', 'Average Daily Risk ($)',
                            'Technical Summary', 'Plain-English Summary'
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )
                market_left, market_right = st.columns(2)
                with market_left:
                    if px is not None:
                        _show_plotly(px.line(market_policy_daily_live, x='report_date', y='cumulative_units', color='policy_name', title='Real-Market Policy Equity Curve (Units)'))
                    else:
                        st.dataframe(market_policy_daily_live[['report_date', 'policy_name', 'cumulative_units']], use_container_width=True, hide_index=True)
                with market_right:
                    if px is not None:
                        market_dd_chart = market_policy_summary_live.melt(
                            id_vars='policy_name',
                            value_vars=['result_units', 'max_drawdown_units'],
                            var_name='metric',
                            value_name='units',
                        )
                        _show_plotly(px.bar(market_dd_chart, x='policy_name', y='units', color='metric', barmode='group', title='Real-Market Return vs. Drawdown by Policy'))
                    else:
                        st.dataframe(market_policy_summary_live[['policy_name', 'result_units', 'max_drawdown_units']], use_container_width=True, hide_index=True)
            else:
                st.info('Real-market alternate replay will populate once settled priced bets exist in the journal.')

            st.markdown('**Policy Research Recommendation**')
            st.caption('Technical: this block ranks bankroll policies over a rolling settled window, using real-market alternate-policy replay first when enough priced bets exist. Plain English: it tells you which style currently looks most trustworthy, and whether you should stay cautious or lean in.')
            window_options = {
                'Last 7 settled days': 7,
                'Last 14 settled days': 14,
                'Last 30 settled days': 30,
                'All settled history': None,
            }
            selected_window_label = st.selectbox('Research Window', list(window_options.keys()), index=1, key='policy_research_window')
            selected_window_days = window_options[selected_window_label]
            canonical_shadow_daily = policy_research_shadow_daily if not policy_research_shadow_daily.empty else shadow_daily
            canonical_market_policy_daily = policy_research_market_policy_daily if not policy_research_market_policy_daily.empty else market_policy_daily_live
            shadow_window = _policy_rankings_for_window(policy_research_rankings, selected_window_label, 'shadow')
            if shadow_window.empty:
                shadow_window = _summarize_shadow_window(canonical_shadow_daily, selected_window_days)
            market_policy_window = _policy_rankings_for_window(policy_research_rankings, selected_window_label, 'real_market')
            if market_policy_window.empty:
                market_policy_window = _summarize_shadow_window(canonical_market_policy_daily, selected_window_days)
            realized_daily = policy_research_market_daily if not policy_research_market_daily.empty else _build_market_realized_daily(bet_journal)
            policy_research_recommendation = _latest_policy_research_recommendation(policy_research_history, selected_window_label)
            if not policy_research_recommendation:
                policy_research_recommendation = _recommend_policy_research(
                    shadow_window,
                    realized_daily,
                    selected_window_label,
                    selected_window_days,
                    market_policy_window,
                )
            st.caption('Research mode: canonical persisted policy research uses real-market replay first when enough priced bets exist, then falls back to shadow evidence. The live replay blocks above remain posture-specific what-if views.')
            rec_cols = st.columns(4)
            rec_cols[0].metric('Recommended Policy', str(policy_research_recommendation.get('policy_name') or 'N/A'))
            rec_cols[1].metric('Recommended Posture', str(policy_research_recommendation.get('posture_name') or 'N/A'))
            rec_cols[2].metric('Evidence Source', str(policy_research_recommendation.get('evidence_source') or 'N/A'))
            rec_cols[3].metric('Evidence Strength', str(policy_research_recommendation.get('evidence_strength') or 'N/A'))
            st.caption('Technical read: ' + str(policy_research_recommendation.get('technical_note') or 'No research note is available yet.'))
            st.caption('Plain English: ' + str(policy_research_recommendation.get('plain_english') or 'No plain-English recommendation is available yet.'))
            takeover_status = policy_research.market_policy_takeover_status(market_policy_window, policy_research_recommendation)
            trigger_cols = st.columns(4)
            trigger_cols[0].metric('Priced Bets in Sample', int(takeover_status.get('current_picks') or 0))
            trigger_cols[1].metric('Settled Days in Sample', int(takeover_status.get('current_slates') or 0))
            trigger_cols[2].metric('To Primary Takeover', f"{int(takeover_status.get('remaining_medium_slates') or 0)} day(s) | {int(takeover_status.get('remaining_medium_picks') or 0)} bets")
            trigger_cols[3].metric('To High Confidence', f"{int(takeover_status.get('remaining_high_slates') or 0)} day(s) | {int(takeover_status.get('remaining_high_picks') or 0)} bets")
            st.caption('Technical trigger summary: primary takeover unlocks at ' + str(takeover_status.get('medium_threshold_label') or 'N/A') + ', and high-confidence market proof unlocks at ' + str(takeover_status.get('high_threshold_label') or 'N/A') + '.')
            st.caption('Plain English takeover read: ' + str(takeover_status.get('plain_english') or 'No takeover guidance is available yet.'))
            market_type_takeover = policy_research.market_type_takeover_status(bet_journal, selected_window_days)
            st.markdown('**Market Readiness by Bet Type**')
            st.caption('Technical: this breaks real-market readiness out by moneyline, total, and run line using each market''s own takeover thresholds. Plain English: it shows which betting markets are building enough settled priced-bet history to stand on their own first.')
            if not market_type_takeover.empty:
                takeover_display = market_type_takeover.copy()
                if 'medium_threshold_label' not in takeover_display.columns:
                    takeover_display['medium_threshold_label'] = 'N/A'
                if 'high_threshold_label' not in takeover_display.columns:
                    takeover_display['high_threshold_label'] = 'N/A'
                takeover_display['Primary ready'] = takeover_display['primary_ready'].map(lambda value: 'Yes' if bool(value) else 'No')
                takeover_display['High-confidence ready'] = takeover_display['high_ready'].map(lambda value: 'Yes' if bool(value) else 'No')
                takeover_display['To takeover'] = takeover_display['remaining_medium_slates'].astype(int).astype(str) + ' day(s) | ' + takeover_display['remaining_medium_picks'].astype(int).astype(str) + ' bets'
                takeover_display['To high-confidence'] = takeover_display['remaining_high_slates'].astype(int).astype(str) + ' day(s) | ' + takeover_display['remaining_high_picks'].astype(int).astype(str) + ' bets'
                takeover_display = takeover_display[[
                    'market_label', 'current_slates', 'current_picks', 'medium_threshold_label', 'high_threshold_label',
                    'Primary ready', 'High-confidence ready', 'To takeover', 'To high-confidence', 'avg_clv', 'plain_english'
                ]].copy()
                takeover_display = _display_frame(takeover_display, {
                    'market_label': 'Market',
                    'current_slates': 'Current Slates',
                    'current_picks': 'Current Picks',
                    'medium_threshold_label': 'Primary Threshold',
                    'high_threshold_label': 'High-Confidence Threshold',
                    'Primary ready': 'Primary Ready',
                    'High-confidence ready': 'High-Confidence Ready',
                    'To takeover': 'To Primary Takeover',
                    'To high-confidence': 'To High Confidence',
                    'avg_clv': 'Average CLV',
                    'plain_english': 'Plain-English Summary',
                })
                st.dataframe(
                    takeover_display[[
                        'Market', 'Current Slates', 'Current Picks', 'Primary Threshold', 'High-Confidence Threshold',
                        'Primary Ready', 'High-Confidence Ready', 'To Primary Takeover', 'To High Confidence', 'Average CLV', 'Plain-English Summary'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                if px is not None:
                    progress_chart = takeover_display.melt(
                        id_vars='Market',
                        value_vars=['Current Picks'],
                        var_name='metric',
                        value_name='value',
                    )
                    _show_plotly(px.bar(progress_chart, x='Market', y='value', color='metric', title='Settled Priced Bets by Market'))
            market_specific_history_window = pd.DataFrame()
            if not policy_research_market_type_history.empty:
                market_specific_history_window = policy_research_market_type_history.copy()
                if 'window_label' in market_specific_history_window.columns:
                    market_specific_history_window = market_specific_history_window[market_specific_history_window['window_label'].astype(str) == str(selected_window_label)].copy()
                if 'as_of_report_date' in market_specific_history_window.columns:
                    market_specific_history_window['as_of_report_date'] = pd.to_datetime(market_specific_history_window.get('as_of_report_date'), errors='coerce')
                sort_cols = [col for col in ['as_of_report_date', 'run_id'] if col in market_specific_history_window.columns]
                if sort_cols:
                    market_specific_history_window = market_specific_history_window.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            if not market_specific_history_window.empty:
                market_specific_recs = market_specific_history_window.drop_duplicates(subset=['market_type'], keep='first').reset_index(drop=True)
            else:
                market_specific_recs = policy_research.market_specific_policy_recommendations(
                    bet_journal,
                    shadow_window,
                    selected_window_label,
                    selected_window_days,
                    phase_name,
                )
            st.markdown('**Market-Specific Policy Recommendations**')
            st.caption('Technical: this recommends a bankroll style separately for moneyline, total, and run line using that market\'s own evidence path. Plain English: each betting market can now earn a different sizing style instead of sharing one global answer.')
            if not market_specific_recs.empty:
                market_specific_display = market_specific_recs.copy()
                if 'medium_threshold_label' not in market_specific_display.columns:
                    market_specific_display['medium_threshold_label'] = 'N/A'
                if 'high_threshold_label' not in market_specific_display.columns:
                    market_specific_display['high_threshold_label'] = 'N/A'
                market_specific_display['To takeover'] = market_specific_display['remaining_medium_slates'].astype(int).astype(str) + ' day(s) | ' + market_specific_display['remaining_medium_picks'].astype(int).astype(str) + ' bets'
                market_specific_display['To high-confidence'] = market_specific_display['remaining_high_slates'].astype(int).astype(str) + ' day(s) | ' + market_specific_display['remaining_high_picks'].astype(int).astype(str) + ' bets'
                market_specific_display = market_specific_display[[
                    'market_label', 'recommended_policy', 'recommended_posture', 'evidence_source', 'evidence_strength',
                    'current_slates', 'current_picks', 'medium_threshold_label', 'high_threshold_label',
                    'To takeover', 'To high-confidence', 'plain_english'
                ]].copy()
                market_specific_display = _display_frame(market_specific_display, {
                    'market_label': 'Market',
                    'recommended_policy': 'Recommended Policy',
                    'recommended_posture': 'Recommended Posture',
                    'evidence_source': 'Evidence Source',
                    'evidence_strength': 'Evidence Strength',
                    'current_slates': 'Current Slates',
                    'current_picks': 'Current Picks',
                    'medium_threshold_label': 'Primary Threshold',
                    'high_threshold_label': 'High-Confidence Threshold',
                    'To takeover': 'To Primary Takeover',
                    'To high-confidence': 'To High Confidence',
                    'plain_english': 'Plain-English Summary',
                })
                st.dataframe(
                    market_specific_display[[
                        'Market', 'Recommended Policy', 'Recommended Posture', 'Evidence Source', 'Evidence Strength',
                        'Current Slates', 'Current Picks', 'Primary Threshold', 'High-Confidence Threshold',
                        'To Primary Takeover', 'To High Confidence', 'Plain-English Summary'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                if not market_specific_history_window.empty:
                    st.markdown('**Market-Specific Recommendation History**')
                    st.caption('Technical: this tracks how each market\'s preferred policy has changed across exports for the selected research window. Plain English: it shows whether moneyline, totals, or run line are starting to trust different bankroll styles over time.')
                    if px is not None:
                        drift_market = market_specific_history_window.dropna(subset=['as_of_report_date']).copy()
                        if not drift_market.empty:
                            _show_plotly(
                                px.scatter(
                                    drift_market,
                                    x='as_of_report_date',
                                    y='recommended_policy',
                                    color='market_label',
                                    symbol='evidence_strength',
                                    hover_data=['recommended_posture', 'evidence_source', 'plain_english'],
                                    title='Market-Aware Policy Recommendation Drift',
                                ),
                                height=420,
                            )
                    drift_cols = [
                        col for col in [
                            'as_of_report_date', 'market_label', 'recommended_policy', 'recommended_posture',
                            'evidence_source', 'evidence_strength', 'plain_english'
                        ] if col in market_specific_history_window.columns
                    ]
                    if drift_cols:
                        st.dataframe(_display_frame(market_specific_history_window[drift_cols].sort_values('as_of_report_date', ascending=False)), use_container_width=True, hide_index=True)
            if not shadow_window.empty:
                research_display = shadow_window.copy()
                research_display['sample ready'] = research_display['sample_ok'].map(lambda value: 'Yes' if bool(value) else 'No')
                research_display = _display_frame(research_display, {
                    'policy_name': 'Policy',
                    'settled_slates': 'Settled Slates',
                    'settled_picks': 'Settled Picks',
                    'hit_rate': 'Hit Rate',
                    'result_units': 'P&L (Units)',
                    'yield_pct': 'Yield',
                    'max_drawdown_units': 'Max Drawdown',
                    'volatility_units': 'Volatility (Units)',
                    'stability_score': 'Stability Score',
                    'sample ready': 'Sample Ready',
                })
                st.dataframe(
                    research_display[[
                        'Policy', 'Settled Slates', 'Settled Picks', 'Hit Rate', 'P&L (Units)', 'Yield',
                        'Max Drawdown', 'Volatility (Units)', 'Stability Score', 'Sample Ready'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                filtered_shadow = _filter_to_window(canonical_shadow_daily, 'report_date', selected_window_days)
                if not filtered_shadow.empty:
                    research_left, research_right = st.columns(2)
                    with research_left:
                        if px is not None:
                            _show_plotly(px.line(filtered_shadow, x='report_date', y='cumulative_units', color='policy_name', title='Windowed Shadow Bankroll Path (Units)'))
                        else:
                            st.dataframe(filtered_shadow[['report_date', 'policy_name', 'cumulative_units']], use_container_width=True, hide_index=True)
                    with research_right:
                        if px is not None:
                            drawdown_chart = filtered_shadow.copy()
                            _show_plotly(px.line(drawdown_chart, x='report_date', y='drawdown_units', color='policy_name', title='Windowed Shadow Drawdown (Units)'))
                        else:
                            st.dataframe(filtered_shadow[['report_date', 'policy_name', 'drawdown_units']], use_container_width=True, hide_index=True)
            else:
                st.info('The shadow policy research table needs more settled archive rows before it can rank the bankroll styles.')

            if not market_policy_window.empty:
                market_window_display = market_policy_window.copy()
                market_window_display['sample ready'] = market_window_display['sample_ok'].map(lambda value: 'Yes' if bool(value) else 'No')
                st.dataframe(
                    market_window_display[[
                        'policy_name', 'settled_slates', 'settled_picks', 'hit_rate', 'result_units', 'yield_pct',
                        'max_drawdown_units', 'avg_clv', 'volatility_units', 'stability_score', 'sample ready'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                filtered_market = _filter_to_window(canonical_market_policy_daily, 'report_date', selected_window_days)
                if not filtered_market.empty:
                    market_window_left, market_window_right = st.columns(2)
                    with market_window_left:
                        if px is not None:
                            _show_plotly(px.line(filtered_market, x='report_date', y='cumulative_units', color='policy_name', title='Windowed Real-Market Policy Bankroll Path (Units)'))
                        else:
                            st.dataframe(filtered_market[['report_date', 'policy_name', 'cumulative_units']], use_container_width=True, hide_index=True)
                    with market_window_right:
                        if px is not None:
                            _show_plotly(px.line(filtered_market, x='report_date', y='drawdown_units', color='policy_name', title='Windowed Real-Market Policy Drawdown (Units)'))
                        else:
                            st.dataframe(filtered_market[['report_date', 'policy_name', 'drawdown_units']], use_container_width=True, hide_index=True)
            else:
                st.info('Real-market rolling policy ranking will populate once enough settled priced bets exist for alternate-policy replay.')

            if not realized_daily.empty:
                realized_window = _filter_to_window(realized_daily, 'report_date', selected_window_days)
                if not realized_window.empty:
                    overlay_cols = st.columns(4)
                    overlay_cols[0].metric('Real-market settled bets', int(realized_window['bets'].sum()) if 'bets' in realized_window.columns else 0)
                    overlay_cols[1].metric('Real-market ROI', _fmt_pct((realized_window['result_units'].sum() / realized_window['stake_units'].sum()) if abs(float(realized_window['stake_units'].sum())) > 1e-9 else None))
                    overlay_cols[2].metric('Real-market drawdown', _fmt_num(abs(float((realized_window['cumulative_units'] - realized_window['cumulative_units'].cummax()).min() or 0.0)), 2) + 'u')
                    overlay_cols[3].metric('Average CLV', _fmt_num(realized_window['avg_clv'].dropna().mean() if 'avg_clv' in realized_window.columns and not realized_window['avg_clv'].dropna().empty else None, 3))
                    if px is not None:
                        _show_plotly(px.line(realized_window, x='report_date', y='cumulative_units', title='Real-Market Bankroll Path (Current Deployed Policy)'))
                    st.caption('Plain English: this is the live execution health check for the currently deployed policy, while the alternate-policy replay above compares what the other bankroll styles would have done on the same priced bets.')
            st.markdown('**Recommendation drift**')
            st.caption('Technical: this is the canonical policy research history stored at export time using Base posture, phase-default unit risk, and 1.0 deployment. Plain English: it lets you see when the model started preferring a different bankroll style instead of trusting only the current screen.')
            if not policy_research_history.empty:
                drift_history = policy_research_history.copy()
                drift_history['as_of_report_date'] = pd.to_datetime(drift_history.get('as_of_report_date'), errors='coerce')
                drift_history = drift_history.dropna(subset=['as_of_report_date'])
                drift_window = drift_history[drift_history['window_label'].astype(str) == selected_window_label].copy()
                if drift_window.empty:
                    drift_window = drift_history.copy()
                drift_window = drift_window.sort_values('as_of_report_date').reset_index(drop=True)
                policy_changes = 0
                if not drift_window.empty:
                    policy_changes = int((drift_window['recommended_policy'].astype(str) != drift_window['recommended_policy'].astype(str).shift(1)).sum() - 1)
                    policy_changes = max(policy_changes, 0)
                drift_cols = st.columns(4)
                latest_drift = drift_window.tail(1)
                drift_cols[0].metric('Latest canonical policy', str(latest_drift['recommended_policy'].iloc[0]) if not latest_drift.empty else 'N/A')
                drift_cols[1].metric('Latest canonical posture', str(latest_drift['recommended_posture'].iloc[0]) if not latest_drift.empty else 'N/A')
                drift_cols[2].metric('Recommendation changes', policy_changes)
                drift_cols[3].metric('Tracked exports', int(len(drift_window)))
                if px is not None and not drift_window.empty:
                    drift_plot = drift_window.copy()
                    drift_plot['policy_posture'] = drift_plot['recommended_policy'].astype(str) + ' | ' + drift_plot['recommended_posture'].astype(str)
                    _show_plotly(
                        px.scatter(
                            drift_plot,
                            x='as_of_report_date',
                            y='recommended_policy',
                            color='evidence_strength',
                            symbol='recommended_posture',
                            hover_data=['window_label', 'evidence_source', 'plain_english'],
                            title='Canonical Policy Recommendation Drift',
                        ),
                        height=420,
                    )
                drift_cols_to_show = [
                    col for col in [
                        'as_of_report_date', 'window_label', 'recommended_policy', 'recommended_posture',
                        'evidence_source', 'evidence_strength', 'plain_english'
                    ] if col in drift_window.columns
                ]
                st.dataframe(drift_window[drift_cols_to_show].sort_values('as_of_report_date', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info('Canonical recommendation history will appear here after the next successful export run writes the research tables.')
            st.markdown('**Real-market proof**')
            st.caption('Technical: this uses settled journal rows with real stake units and results. Plain English: once priced bets are logged, this becomes the true profitability scorecard instead of the shadow simulator.')
            realized_summary, realized_by_market = _build_market_realized_summary(bet_journal)
            if not realized_summary.empty:
                realized_row = realized_summary.iloc[0]
                realized_cols = st.columns(4)
                realized_cols[0].metric('Settled market bets', int(realized_row.get('settled_bets') or 0))
                realized_cols[1].metric('Realized P&L', _fmt_num(realized_row.get('result_units'), 2) + 'u')
                realized_cols[2].metric('Realized ROI', _fmt_pct(realized_row.get('roi_pct')))
                realized_cols[3].metric('Avg CLV', _fmt_num(realized_row.get('avg_clv'), 3))
                if not realized_by_market.empty:
                    st.dataframe(realized_by_market, use_container_width=True, hide_index=True)
                    totals_realized = realized_by_market[realized_by_market['market_type'].astype(str).str.lower() == 'total'].copy()
                    if not totals_realized.empty:
                        totals_row = totals_realized.iloc[0]
                        st.markdown('**Totals-only real-market proof**')
                        totals_realized_cols = st.columns(5)
                        totals_realized_cols[0].metric('Totals bets', int(totals_row.get('bets') or 0))
                        totals_realized_cols[1].metric('Totals P&L', _fmt_num(totals_row.get('result_units'), 2) + 'u')
                        totals_realized_cols[2].metric('Totals ROI', _fmt_pct(totals_row.get('roi_pct')))
                        totals_realized_cols[3].metric('Totals Avg CLV', _fmt_num(totals_row.get('avg_clv'), 3))
                        totals_realized_cols[4].metric('Totals max DD', _fmt_num(totals_row.get('max_drawdown_units'), 2) + 'u')
                        st.caption('Technical: this isolates settled totals performance only. Plain English: it shows whether totals themselves are actually making money and beating the close, instead of hiding inside all bets combined.')
                        if not market_proof_by_market_history.empty:
                            totals_history = market_proof_by_market_history[market_proof_by_market_history['market_type'].astype(str).str.lower() == 'total'].copy()
                            if not totals_history.empty:
                                totals_history['as_of_report_date'] = pd.to_datetime(totals_history['as_of_report_date'], errors='coerce')
                                totals_history = totals_history.sort_values('as_of_report_date')
                                totals_left, totals_right = st.columns(2)
                                with totals_left:
                                    _render_line_chart(totals_history, 'as_of_report_date', 'roi', 'Totals ROI Trend')
                                with totals_right:
                                    _render_line_chart(totals_history, 'as_of_report_date', 'avg_clv', 'Totals CLV Trend')
            else:
                st.info('No settled priced bets are available yet, so this section will turn on once regular-season market journaling starts filling in.')
            if not bet_journal.empty and 'result_units' in bet_journal.columns:

                settled = bet_journal.copy()
                settled['result_units'] = pd.to_numeric(settled['result_units'], errors='coerce')
                settled['report_date'] = pd.to_datetime(settled.get('report_date'), errors='coerce')
                settled = settled.dropna(subset=['result_units', 'report_date']).sort_values('report_date')
                if not settled.empty:
                    settled['cumulative_units'] = settled['result_units'].cumsum()
                    _render_line_chart(settled, 'report_date', 'cumulative_units', 'Settled Bet Equity Curve')

    with tabs[3]:
        st.subheader('Validation and Calibration')
        target_bundle = _dynamic_target_bundle(latest, selected_phase)
        target_frame = _frame_or_empty(target_bundle.get('frame'))
        if not target_frame.empty:
            st.markdown('**Dynamic Target Reference**')
            target_cols = st.columns(4)
            target_cols[0].metric('Evidence Progress', f"{(target_bundle.get('progress') or 0.0) * 100:.0f}%")
            target_cols[1].metric('Research Games', int(target_bundle.get('research_games') or 0))
            target_cols[2].metric('Live Games', int(target_bundle.get('live_games') or 0))
            target_cols[3].metric('Priced Bets', int(target_bundle.get('priced_bets') or 0))
            st.caption('Technical: these targets are phase-aware and sample-aware. Plain English: they give each key metric a moving reference point that becomes more demanding only after enough real evidence has accumulated.')
            st.dataframe(_display_frame(target_frame), use_container_width=True, hide_index=True)
        freshness = _validation_freshness_frame(clean_backtest, live_archive, latest)
        if not freshness.empty:
            st.caption('Research replay and live settled are intentionally separate tracks. The replay baseline is frozen until rebuilt; the live archive should keep extending as real games settle.')
            st.dataframe(_display_frame(freshness), use_container_width=True, hide_index=True)
        validation_delta = _validation_delta_frame(latest)
        if not validation_delta.empty:
            st.markdown('**Validation Snapshot Comparison**')
            st.dataframe(_display_frame(validation_delta), use_container_width=True, hide_index=True)
        if clean_backtest.empty:
            st.info('Research replay data is not available yet.')
        else:
            st.caption('RESEARCH TRACK')
            st.markdown('**Research Replay OOS**')
            validation_daily = _daily_validation(clean_backtest)
            if not validation_daily.empty:
                left, right = st.columns(2)
                with left:
                    _render_line_chart(validation_daily, 'date', 'log_loss', 'Research Replay Daily Log Loss')
                with right:
                    _render_line_chart(validation_daily, 'date', 'accuracy', 'Research Replay Daily Accuracy')
                left, right = st.columns(2)
                with left:
                    _render_line_chart(validation_daily, 'date', 'total_mae', 'Research Replay Daily Totals MAE')
                with right:
                    _render_line_chart(validation_daily, 'date', 'margin_mae', 'Research Replay Daily Margin MAE')
            calibration = _calibration_table(clean_backtest)
            if not calibration.empty:
                _render_bar_chart(calibration, 'bucket', 'actual_win_rate', 'Research Replay Calibration by Probability Bucket')
                st.dataframe(_display_frame(calibration), use_container_width=True, hide_index=True)
            clean_work = clean_backtest.copy()
            if {'projected_total', 'home_score', 'away_score'}.issubset(clean_work.columns):
                clean_work['actual_total'] = pd.to_numeric(clean_work['home_score'], errors='coerce') + pd.to_numeric(clean_work['away_score'], errors='coerce')
                total_points = clean_work.dropna(subset=['projected_total', 'actual_total']).copy()
                if not total_points.empty and px is not None:
                    _show_plotly(px.scatter(total_points, x='projected_total', y='actual_total', title='Research Replay: Projected vs. Actual Totals', labels={'projected_total': 'Projected Total', 'actual_total': 'Actual Total'}), height=400)
            if {'projected_margin', 'home_score', 'away_score'}.issubset(clean_work.columns):
                clean_work['actual_margin'] = pd.to_numeric(clean_work['home_score'], errors='coerce') - pd.to_numeric(clean_work['away_score'], errors='coerce')
                margin_points = clean_work.dropna(subset=['projected_margin', 'actual_margin']).copy()
                if not margin_points.empty and px is not None:
                    _show_plotly(px.scatter(margin_points, x='projected_margin', y='actual_margin', title='Research Replay: Projected vs. Actual Margin', labels={'projected_margin': 'Projected Margin', 'actual_margin': 'Actual Margin'}), height=400)
        if not live_archive.empty and 'report_date' in live_archive.columns:
            st.caption('LIVE TRACK')
            st.markdown('**Live Settled Validation**')
            live_work = live_archive.copy()
            live_work['report_date'] = pd.to_datetime(live_work.get('report_date'), errors='coerce').dt.date
            if 'y_home' in live_work.columns:
                live_work['y_home'] = pd.to_numeric(live_work.get('y_home'), errors='coerce')
            if 'p_home' in live_work.columns:
                live_work['p_home'] = pd.to_numeric(live_work.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
            if {'home_score', 'away_score'}.issubset(live_work.columns):
                live_work['home_score'] = pd.to_numeric(live_work.get('home_score'), errors='coerce')
                live_work['away_score'] = pd.to_numeric(live_work.get('away_score'), errors='coerce')
            live_work = live_work.dropna(subset=['report_date']).copy()
            settled_mask = pd.Series(True, index=live_work.index)
            if 'y_home' in live_work.columns:
                settled_mask &= live_work['y_home'].notna()
            elif 'actual_total' in live_work.columns:
                settled_mask &= pd.to_numeric(live_work.get('actual_total'), errors='coerce').notna()
            live_work = live_work[settled_mask].copy()
            if not live_work.empty:
                live_rows = []
                for target_date, grp in live_work.groupby('report_date'):
                    log_loss = None
                    accuracy = None
                    total_rmse = None
                    margin_rmse = None
                    prob_grp = grp.dropna(subset=['p_home', 'y_home']) if {'p_home', 'y_home'}.issubset(grp.columns) else pd.DataFrame()
                    if not prob_grp.empty:
                        p = prob_grp['p_home'].astype(float)
                        y = prob_grp['y_home'].astype(int)
                        log_loss = float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean())
                        accuracy = float(((p >= 0.5).astype(int) == y).mean())
                    if {'projected_total', 'actual_total'}.issubset(grp.columns):
                        total_grp = grp.dropna(subset=['projected_total', 'actual_total']).copy()
                        if not total_grp.empty:
                            errors = pd.to_numeric(total_grp['actual_total'], errors='coerce') - pd.to_numeric(total_grp['projected_total'], errors='coerce')
                            total_rmse = float(np.sqrt(np.mean(np.square(errors))))
                    if {'projected_margin', 'actual_margin'}.issubset(grp.columns):
                        margin_grp = grp.dropna(subset=['projected_margin', 'actual_margin']).copy()
                        if not margin_grp.empty:
                            errors = pd.to_numeric(margin_grp['actual_margin'], errors='coerce') - pd.to_numeric(margin_grp['projected_margin'], errors='coerce')
                            margin_rmse = float(np.sqrt(np.mean(np.square(errors))))
                    live_rows.append({
                        'date': target_date,
                        'games': int(len(grp)),
                        'log_loss': log_loss,
                        'accuracy': accuracy,
                        'total_rmse': total_rmse,
                        'margin_rmse': margin_rmse,
                    })
                live_daily = pd.DataFrame(live_rows).sort_values('date').reset_index(drop=True)
                if not live_daily.empty:
                    left, right = st.columns(2)
                    with left:
                        _render_line_chart(live_daily, 'date', 'log_loss', 'Live Settled Daily Log Loss')
                    with right:
                        _render_line_chart(live_daily, 'date', 'accuracy', 'Live Settled Daily Accuracy')
                    left, right = st.columns(2)
                    with left:
                        _render_line_chart(live_daily, 'date', 'total_rmse', 'Live Settled Daily Totals RMSE')
                    with right:
                        _render_line_chart(live_daily, 'date', 'margin_rmse', 'Live Settled Daily Margin RMSE')
            else:
                st.info('No settled live validation rows are available yet.')

    with tabs[4]:
        st.subheader('Research and Diagnostics')
        left, right = st.columns(2)
        with left:
            st.markdown('**Model State Snapshots**')
            summary_blocks = []
            for key in ['probability_shrinkage', 'total_shrinkage', 'margin_shrinkage', 'lineup_earn_back']:
                if key in model_state:
                    summary_blocks.append({'block': key, 'payload': json.dumps(model_state.get(key), indent=2)[:500]})
            if summary_blocks:
                st.dataframe(_display_frame(pd.DataFrame(summary_blocks), {'block': 'State Block', 'payload': 'Stored Payload'}), use_container_width=True, hide_index=True)
            else:
                st.info('No model-state blocks were available.')
        with right:
            st.markdown('**Live Archive Growth**')
            if not live_archive.empty and 'report_date' in live_archive.columns:
                archive_view = live_archive.copy()
                archive_view['report_date'] = pd.to_datetime(archive_view['report_date'], errors='coerce')
                archive_daily = archive_view.groupby('report_date', dropna=True).size().reset_index(name='games_archived')
                _render_line_chart(archive_daily, 'report_date', 'games_archived', 'Archived Live Predictions by Date')
            else:
                st.info('The live prediction archive is still empty.')
        st.divider()
        if not daily_kpi_snapshot.empty:
            st.markdown('**Daily KPI Snapshots**')
            st.caption('Technical: this is the canonical one-row-per-day operating summary persisted into DuckDB. Plain English: it is the clean daily ledger for the model, market proof, and target progress.')
            recent_kpis = daily_kpi_snapshot.copy()
            if 'report_date' in recent_kpis.columns:
                recent_kpis['report_date'] = pd.to_datetime(recent_kpis['report_date'], errors='coerce')
                recent_kpis = recent_kpis.sort_values('report_date')
            kpi_cols = st.columns(2)
            with kpi_cols[0]:
                _render_line_chart(recent_kpis, 'report_date', 'live_validation_log_loss', 'Daily Live Log Loss Snapshot')
            with kpi_cols[1]:
                _render_line_chart(recent_kpis, 'report_date', 'market_proof_avg_clv', 'Daily Average CLV Snapshot')
            with st.expander('View Daily KPI Snapshot Table', expanded=False):
                display_cols = [
                    col for col in [
                        'report_date', 'phase', 'live_validation_games', 'live_validation_log_loss',
                        'live_validation_accuracy', 'live_validation_total_rmse', 'live_validation_margin_rmse',
                        'market_proof_settled_bets', 'market_proof_roi', 'market_proof_avg_clv',
                        'policy_alignment_status', 'target_progress_pct',
                    ] if col in recent_kpis.columns
                ]
                st.dataframe(_display_frame(recent_kpis[display_cols].sort_values('report_date', ascending=False)), use_container_width=True, hide_index=True)

        if not rolling_kpi_windows.empty:
            st.markdown('**Rolling Analytics Windows**')
            st.caption('Technical: these are trailing 3-, 7-, and 14-run rolling KPI windows. Plain English: they smooth out the daily noise and show the underlying direction of live performance and market proof.')
            rolling_view = rolling_kpi_windows.copy()
            if 'report_date' in rolling_view.columns:
                rolling_view['report_date'] = pd.to_datetime(rolling_view['report_date'], errors='coerce')
                rolling_view = rolling_view.sort_values('report_date')
            latest_rolling = rolling_view.groupby('window_label', dropna=False).tail(1)
            st.dataframe(_display_frame(latest_rolling), use_container_width=True, hide_index=True)

        if not alert_history.empty:
            st.markdown('**Alert History**')
            st.caption('Technical: this is a persisted event log of model, market-proof, and workflow alerts. Plain English: it gives you a memory of what the system has been warning about, not just what is flashing today.')
            alert_view = alert_history.copy()
            if 'as_of_report_date' in alert_view.columns:
                alert_view['as_of_report_date'] = pd.to_datetime(alert_view['as_of_report_date'], errors='coerce')
            alert_view = alert_view.sort_values(['as_of_report_date', 'created_ts'], ascending=[False, False]).head(20)
            st.dataframe(_display_frame(alert_view), use_container_width=True, hide_index=True)

        if not attribution_market_mart.empty or not attribution_driver_mart.empty:
            st.markdown('**Attribution Marts**')
            st.caption('Technical: these marts roll up historical misses by market and by key risk driver. Plain English: they help show where errors are clustering so the next tuning pass hits the right part of the model.')
            mart_cols = st.columns(2)
            with mart_cols[0]:
                if not attribution_market_mart.empty:
                    _render_bar_chart(_display_frame(attribution_market_mart), 'Market Type', 'Rmse', 'Historical Miss Size by Market')
                    with st.expander('View Market Attribution Detail', expanded=False):
                        st.dataframe(_display_frame(attribution_market_mart), use_container_width=True, hide_index=True)
                else:
                    st.info('No market attribution mart is available yet.')
            with mart_cols[1]:
                if not attribution_driver_mart.empty:
                    driver_view = attribution_driver_mart.copy()
                    driver_view = driver_view[driver_view['driver_state'].astype(str) == 'Flagged'] if 'driver_state' in driver_view.columns else driver_view
                    _render_bar_chart(_display_frame(driver_view.head(15)), 'Driver', 'Rmse', 'Flagged Driver Miss Size', color_col='Market Type' if 'Market Type' in _display_frame(driver_view.head(1)).columns else None)
                    with st.expander('View Driver Attribution Detail', expanded=False):
                        st.dataframe(_display_frame(attribution_driver_mart), use_container_width=True, hide_index=True)
                else:
                    st.info('No driver attribution mart is available yet.')

        if not target_history.empty:
            st.markdown('**Target History**')
            st.caption('Technical: these are persisted target snapshots by metric. Plain English: they show how the bar has tightened over time as more evidence has arrived.')
            target_view = target_history.copy()
            target_view['as_of_report_date'] = pd.to_datetime(target_view.get('as_of_report_date'), errors='coerce')
            metric_choices = sorted([str(value) for value in target_view['metric_label'].dropna().unique().tolist()]) if 'metric_label' in target_view.columns else []
            if metric_choices:
                target_metric = st.selectbox('Target history metric', metric_choices, key='target_history_metric')
                metric_frame = target_view[target_view['metric_label'].astype(str) == target_metric].copy().sort_values('as_of_report_date')
                if not metric_frame.empty and px is not None:
                    plot_frame = metric_frame.rename(columns={'current_value': 'Current', 'dynamic_target_value': 'Dynamic Target'})
                    plot_data = plot_frame.melt(id_vars='as_of_report_date', value_vars=[col for col in ['Current', 'Dynamic Target'] if col in plot_frame.columns], var_name='Series', value_name='value')
                    _show_plotly(px.line(plot_data, x='as_of_report_date', y='value', color='Series', title=f'{target_metric}: Current vs. Dynamic Target'), height=380)
                with st.expander('View Target History Detail', expanded=False):
                    st.dataframe(_display_frame(metric_frame), use_container_width=True, hide_index=True)
        with st.expander('Raw model_state.json', expanded=False):
            st.code(json.dumps(model_state, indent=2), language='json')

    with tabs[5]:
        st.subheader('Ask the Model')
        st.caption('Ask freeform questions about the slate, model health, market proof, policy posture, or parlays. The assistant responds from the stored dashboard data and current model outputs, not from open-ended guesses.')
        st.markdown('**Supported domains**')
        st.write('Game explanations, historical best/worst days, market summaries, alerts, target status, driver attribution, validation checks, policy posture, day-over-day changes, count lookups, and correlation-aware same-day parlays.')
        example_cols = st.columns(3)
        for idx, example in enumerate(_question_examples()):
            if example_cols[idx % 3].button(example, key=f'qa_example_{idx}'):
                st.session_state['qa_question'] = example
        with st.form('ask_model_form'):
            question = st.text_area(
                'Ask a question',
                key='qa_question',
                height=120,
                placeholder='Example: Build a 2-leg +175 parlay with the highest chance to hit.',
            )
            submitted = st.form_submit_button('Run Analysis')
        if submitted and str(question or '').strip():
            parsed = _question_parser(question)
            answer = _answer_model_question(
                question,
                parsed,
                latest,
                run_history,
                daily_kpi_snapshot,
                rolling_kpi_windows,
                alert_history,
                target_history,
                attribution_market_mart,
                attribution_driver_mart,
                predictions,
                candidates,
                portfolio_recommendations,
                market_proof_by_market_bundle,
                odds_snapshots,
                selected_phase_label,
            )
            analysis_id = dt.datetime.now(dt.timezone.utc).isoformat()
            st.session_state['qa_last_analysis'] = {
                'analysis_id': analysis_id,
                'question': question,
                'parsed': parsed,
                'answer': answer,
                'phase_view': selected_phase_label,
                'run_id': str(latest.get('run_id') or ''),
            }
            st.session_state[f'qa_feedback_{analysis_id}'] = ''
        elif submitted:
            st.info('Enter a question first, then run the analysis.')
        analysis = st.session_state.get('qa_last_analysis')
        if isinstance(analysis, dict) and analysis:
            parsed = analysis.get('parsed') or {}
            answer = analysis.get('answer') or {}
            analysis_id = str(analysis.get('analysis_id') or 'current')
            st.markdown(f"**{answer.get('title') or 'Answer'}**")
            if answer.get('summary'):
                st.write(str(answer.get('summary')))
            for bullet in answer.get('bullets') or []:
                st.write(f"- {bullet}")
            answer_table = _frame_or_empty(answer.get('table'))
            if not answer_table.empty:
                with st.expander(str(answer.get('table_label') or 'Supporting Detail'), expanded=True):
                    st.dataframe(answer_table, use_container_width=True, hide_index=True)
            with st.expander('How the question was interpreted', expanded=False):
                st.dataframe(_display_frame(pd.DataFrame([parsed])), use_container_width=True, hide_index=True)
            feedback_state = str(st.session_state.get(f'qa_feedback_{analysis_id}', ''))
            if feedback_state == '':
                st.markdown('**Did this answer your question?**')
                feedback_cols = st.columns([1, 1, 4])
                if feedback_cols[0].button('Yes', key=f'qa_feedback_yes_{analysis_id}'):
                    st.session_state[f'qa_feedback_{analysis_id}'] = 'yes'
                if feedback_cols[1].button('No', key=f'qa_feedback_no_{analysis_id}'):
                    log_result = quant_store.log_failed_question(
                        str(analysis.get('question') or ''),
                        parsed=parsed,
                        answer=answer,
                        phase_view=str(analysis.get('phase_view') or ''),
                        run_id=str(analysis.get('run_id') or ''),
                    )
                    if bool(log_result.get('logged')):
                        st.session_state[f'qa_feedback_{analysis_id}'] = 'logged'
                    else:
                        st.session_state[f'qa_feedback_{analysis_id}'] = 'error'
                        st.session_state[f'qa_feedback_error_{analysis_id}'] = str(log_result.get('warning') or 'Unknown logging failure')
            elif feedback_state == 'logged':
                st.caption('Logged to Failed Question Log for future coverage.')
            elif feedback_state == 'error':
                st.caption('Failed Question Log write failed: ' + str(st.session_state.get(f'qa_feedback_error_{analysis_id}', 'Unknown error')))


if __name__ == '__main__':
    main()































