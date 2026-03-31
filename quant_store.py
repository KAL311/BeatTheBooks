from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import policy_research
import portfolio_engine as portfolio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_DB_PATH = os.path.join(BASE_DIR, 'mlb_quant_dashboard.duckdb')
CLEAN_BACKTEST_LOG_PATH = os.path.join(BASE_DIR, 'clean_backtest_log.csv')
LIVE_PREDICTION_ARCHIVE_PATH = os.path.join(BASE_DIR, 'live_prediction_archive.csv')
MODEL_STATE_PATH = os.path.join(BASE_DIR, 'model_state.json')
CLEAN_BACKTEST_SUMMARY_PATH = os.path.join(BASE_DIR, 'clean_backtest_summary.json')
SPORTSBOOK_DB_PATH = os.path.join(BASE_DIR, 'sportsbook_lines.db')
CANONICAL_POLICY_WINDOW_LABEL = 'Last 14 settled days'
MARKET_POLICY_TYPES = [
    ('moneyline', 'Moneyline'),
    ('total', 'Total'),
    ('run_line', 'Run line'),
]


def _import_duckdb():
    try:
        import duckdb  # type: ignore
        return duckdb, None
    except Exception as exc:  # pragma: no cover - dependency may not be installed
        return None, str(exc)


def quant_support_status() -> Dict[str, Any]:
    duckdb, error = _import_duckdb()
    return {
        'duckdb_available': duckdb is not None,
        'duckdb_error': error,
        'db_path': QUANT_DB_PATH,
        'db_exists': os.path.exists(QUANT_DB_PATH),
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(float(value))
    except Exception:
        return None


def _json_default(value: Any):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _safe_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}


def _load_sqlite_table(db_path: str, table_name: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
    except Exception:
        return pd.DataFrame()


def _replace_table(con, table_name: str, frame: pd.DataFrame) -> None:
    if frame is None or (frame.empty and len(frame.columns) == 0):
        return
    con.register('_quant_frame', frame)
    con.execute(f'CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _quant_frame')
    con.unregister('_quant_frame')


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _duckdb_type_for_series(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    if pd.api.types.is_integer_dtype(dtype):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(dtype):
        return 'DOUBLE'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    return 'VARCHAR'


def _normalize_duckdb_type(type_name: Any) -> str:
    upper = str(type_name or '').upper()
    if not upper:
        return 'VARCHAR'
    if any(token in upper for token in ('CHAR', 'TEXT', 'STRING', 'VARCHAR', 'UUID', 'JSON')):
        return 'VARCHAR'
    if upper in {'BOOL', 'BOOLEAN'}:
        return 'BOOLEAN'
    if upper.startswith('DECIMAL') or upper in {'DOUBLE', 'FLOAT', 'REAL'}:
        return 'DOUBLE'
    if 'INT' in upper:
        return 'BIGINT'
    if 'TIMESTAMP' in upper or upper == 'DATE' or upper.startswith('TIME'):
        return 'TIMESTAMP'
    return upper


def _compatible_duckdb_type(existing_type: Any, incoming_type: Any) -> str:
    existing = _normalize_duckdb_type(existing_type)
    incoming = _normalize_duckdb_type(incoming_type)
    if existing == incoming:
        return existing
    if existing == 'VARCHAR' and incoming in {'DOUBLE', 'BIGINT', 'BOOLEAN', 'TIMESTAMP'}:
        return incoming
    if incoming == 'VARCHAR':
        return existing
    if existing == 'VARCHAR':
        return 'VARCHAR'
    if 'TIMESTAMP' in {existing, incoming}:
        return 'TIMESTAMP' if existing == incoming else 'VARCHAR'
    if 'DOUBLE' in {existing, incoming}:
        return 'DOUBLE'
    if 'BIGINT' in {existing, incoming}:
        return 'BIGINT'
    if 'BOOLEAN' in {existing, incoming}:
        return 'BOOLEAN'
    return incoming


def _create_table_with_schema(con, table_name: str, frame: pd.DataFrame) -> None:
    column_defs = [
        f"{_quote_ident(col)} {_duckdb_type_for_series(frame[col])}"
        for col in frame.columns
    ]
    con.execute(f"CREATE TABLE IF NOT EXISTS {_quote_ident(table_name)} ({', '.join(column_defs)})")


def _coerce_series_for_duckdb(series: pd.Series, duck_type: str) -> pd.Series:
    target = _normalize_duckdb_type(duck_type)
    if target == 'DOUBLE':
        return pd.to_numeric(series, errors='coerce')
    if target == 'BIGINT':
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.isna().all():
            return numeric
        return numeric.round().astype('Int64')
    if target == 'BOOLEAN':
        mapping = {
            'true': True,
            'false': False,
            '1': True,
            '0': False,
            'yes': True,
            'no': False,
            'y': True,
            'n': False,
        }
        text = series.astype(str).str.strip().str.lower()
        coerced = text.map(mapping)
        null_mask = series.isna() | text.isin({'', 'nan', 'none', 'null', '<na>'})
        return coerced.mask(null_mask).astype('boolean')
    if target == 'TIMESTAMP':
        return pd.to_datetime(series, errors='coerce')
    return series


def _duckdb_column_promotable(con, table_name: str, column_name: str, target_type: str) -> bool:
    normalized = _normalize_duckdb_type(target_type)
    if normalized not in {'DOUBLE', 'BIGINT', 'BOOLEAN', 'TIMESTAMP'}:
        return False
    quoted_table = _quote_ident(table_name)
    quoted_col = _quote_ident(column_name)
    check_sql = (
        f"SELECT COUNT(*) FROM {quoted_table} "
        f"WHERE {quoted_col} IS NOT NULL "
        f"AND NULLIF(TRIM(CAST({quoted_col} AS VARCHAR)), '') IS NOT NULL "
        f"AND TRY_CAST({quoted_col} AS {normalized}) IS NULL"
    )
    try:
        failures = con.execute(check_sql).fetchone()[0]
    except Exception:
        return False
    return int(failures or 0) == 0


def _ensure_quant_views(con) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW latest_quant_run AS
        SELECT * FROM quant_runs
        ORDER BY TRY_CAST(created_ts AS TIMESTAMP) DESC NULLS LAST, created_ts DESC, run_id DESC
        LIMIT 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW quant_runs_daily AS
        SELECT * EXCLUDE (row_num)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY report_date
                       ORDER BY TRY_CAST(created_ts AS TIMESTAMP) DESC NULLS LAST, created_ts DESC, run_id DESC
                   ) AS row_num
            FROM quant_runs
        )
        WHERE row_num = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW daily_kpi_snapshot_daily AS
        SELECT * EXCLUDE (row_num)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY report_date
                       ORDER BY TRY_CAST(created_ts AS TIMESTAMP) DESC NULLS LAST, created_ts DESC, run_id DESC
                   ) AS row_num
            FROM daily_kpi_snapshot
        )
        WHERE row_num = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW rolling_kpi_windows AS
        SELECT report_date, phase, 'Trailing 3 runs' AS window_label,
               AVG(live_validation_log_loss) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS live_validation_log_loss,
               AVG(live_validation_accuracy) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS live_validation_accuracy,
               AVG(live_validation_total_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS live_validation_total_rmse,
               AVG(live_validation_margin_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS live_validation_margin_rmse,
               AVG(market_proof_avg_clv) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS market_proof_avg_clv,
               AVG(market_proof_roi) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS market_proof_roi,
               AVG(target_progress_pct) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS target_progress_pct
        FROM daily_kpi_snapshot_daily
        UNION ALL
        SELECT report_date, phase, 'Trailing 7 runs' AS window_label,
               AVG(live_validation_log_loss) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS live_validation_log_loss,
               AVG(live_validation_accuracy) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS live_validation_accuracy,
               AVG(live_validation_total_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS live_validation_total_rmse,
               AVG(live_validation_margin_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS live_validation_margin_rmse,
               AVG(market_proof_avg_clv) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS market_proof_avg_clv,
               AVG(market_proof_roi) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS market_proof_roi,
               AVG(target_progress_pct) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS target_progress_pct
        FROM daily_kpi_snapshot_daily
        UNION ALL
        SELECT report_date, phase, 'Trailing 14 runs' AS window_label,
               AVG(live_validation_log_loss) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS live_validation_log_loss,
               AVG(live_validation_accuracy) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS live_validation_accuracy,
               AVG(live_validation_total_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS live_validation_total_rmse,
               AVG(live_validation_margin_rmse) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS live_validation_margin_rmse,
               AVG(market_proof_avg_clv) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS market_proof_avg_clv,
               AVG(market_proof_roi) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS market_proof_roi,
               AVG(target_progress_pct) OVER (PARTITION BY phase ORDER BY report_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS target_progress_pct
        FROM daily_kpi_snapshot_daily
        """
    )


def _append_table(con, table_name: str, frame: pd.DataFrame) -> None:
    if frame is None or len(frame.columns) == 0:
        return
    frame_to_write = frame.copy()
    try:
        existing = con.execute(f"PRAGMA table_info('{table_name}')").df()
        table_exists = not existing.empty
    except Exception:
        existing = pd.DataFrame()
        table_exists = False
    if table_exists:
        existing_cols = existing['name'].astype(str).tolist()
        existing_types = {
            str(row['name']): _normalize_duckdb_type(row['type'])
            for _, row in existing.iterrows()
        }
        new_cols = [col for col in frame_to_write.columns if col not in existing_cols]
        for col in new_cols:
            duck_type = _duckdb_type_for_series(frame_to_write[col])
            con.execute(f"ALTER TABLE {_quote_ident(table_name)} ADD COLUMN {_quote_ident(col)} {duck_type}")
            existing_cols.append(col)
            existing_types[col] = duck_type
        if not frame_to_write.empty:
            for col in frame_to_write.columns:
                if col not in existing_types:
                    continue
                incoming_type = _duckdb_type_for_series(frame_to_write[col])
                target_type = _compatible_duckdb_type(existing_types[col], incoming_type)
                if target_type != existing_types[col]:
                    if existing_types[col] == 'VARCHAR' and target_type in {'DOUBLE', 'BIGINT', 'BOOLEAN', 'TIMESTAMP'}:
                        if not _duckdb_column_promotable(con, table_name, col, target_type):
                            target_type = existing_types[col]
                    con.execute(
                        f"ALTER TABLE {_quote_ident(table_name)} "
                        f"ALTER COLUMN {_quote_ident(col)} SET DATA TYPE {target_type} "
                        f"USING TRY_CAST({_quote_ident(col)} AS {target_type})"
                    )
                    existing_types[col] = target_type
        for col in existing_cols:
            if col not in frame_to_write.columns:
                frame_to_write[col] = None
        frame_to_write = frame_to_write[existing_cols]
        for col, duck_type in existing_types.items():
            if col in frame_to_write.columns:
                frame_to_write[col] = _coerce_series_for_duckdb(frame_to_write[col], duck_type)
    if not table_exists:
        _create_table_with_schema(con, table_name, frame_to_write)
    con.register('_quant_append_frame', frame_to_write)
    try:
        if not frame_to_write.empty:
            con.execute(f'INSERT INTO {_quote_ident(table_name)} SELECT * FROM _quant_append_frame')
    finally:
        con.unregister('_quant_append_frame')

def _market_expected_value(probability: Any, american_price: Any) -> Optional[float]:
    p = _safe_float(probability)
    price = _safe_int(american_price)
    if p is None or price is None:
        return None
    p = max(min(p, 1.0 - 1e-6), 1e-6)
    win_profit = (price / 100.0) if price > 0 else (100.0 / abs(price))
    return float((p * win_profit) - ((1.0 - p) * 1.0))
def _total_expected_value_proxy(diff: Any, american_price: Any) -> float:
    gap = abs(_safe_float(diff) or 0.0)
    if gap <= 0.05:
        return 0.0
    price = _safe_int(american_price)
    base_value = min(max((gap - 0.10) * 0.055, 0.0), 0.14)
    juice_drag = 0.0
    if price is not None:
        juice_drag = min(max((abs(price) - 110) / 2000.0, 0.0), 0.035)
    return round(max(0.0, base_value - juice_drag), 4)

def _suggested_units_preview(label: str, score: Any, prediction: Dict[str, Any], market_type: str) -> float:
    base_units = {'PASS': 0.0, 'WATCH': 0.25, 'BET': 0.50, 'MUST TAKE': 0.75}.get(str(label or 'PASS').upper(), 0.0)
    if base_units <= 0:
        return 0.0
    features = prediction.get('features') or {}
    lineups = features.get('lineups') or {}
    away_count = _safe_float(lineups.get('away')) or 0.0
    home_count = _safe_float(lineups.get('home')) or 0.0
    lineup_completion = max(min((away_count + home_count) / 18.0, 1.0), 0.0)
    away_conf = _safe_float(((features.get('away_starter') or {}).get('confidence'))) or 0.5
    home_conf = _safe_float(((features.get('home_starter') or {}).get('confidence'))) or 0.5
    starter_conf = max(min((away_conf + home_conf) / 2.0, 1.0), 0.0)
    score_factor = 0.50 + (max(min((_safe_float(score) or 0.0), 100.0), 0.0) / 200.0)
    shrinkages = [
        _safe_float(prediction.get('probability_shrink_alpha')) or 0.0,
        _safe_float(prediction.get('total_shrink_alpha')) or 0.0,
        _safe_float(prediction.get('margin_shrink_alpha')) or 0.0,
    ]
    shrink_penalty = max(0.60, 1.0 - (sum(shrinkages) / max(len(shrinkages), 1)))
    market_factor = {'moneyline': 1.0, 'run_line': 0.90, 'total': 0.85}.get(str(market_type or '').lower(), 0.80)
    certainty_factor = max(0.35, (0.55 * starter_conf) + (0.45 * lineup_completion))
    units = base_units * score_factor * shrink_penalty * market_factor * certainty_factor
    return round(max(min(units, 1.25), 0.0), 2)


def _clip_unit(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    numeric = _safe_float(value)
    if numeric is None:
        numeric = low
    return max(low, min(high, float(numeric)))


def _market_regime_gap(model_value: Any, baseline_value: Any, scale: float) -> float:
    model_metric = _safe_float(model_value)
    baseline_metric = _safe_float(baseline_value)
    if model_metric is None or baseline_metric is None or scale <= 0:
        return 0.0
    return _clip_unit((model_metric - baseline_metric) / scale, 0.0, 1.0)


def _build_uncertainty_calibration_context(phase: str, validation: Dict[str, Any], live_validation: Dict[str, Any], benchmark_ladder: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    phase_name = str(phase or 'spring').lower()
    oos_games = int(_safe_int((validation or {}).get('oos_games')) or _safe_int((validation or {}).get('games')) or 0)
    live_games = int(_safe_int((live_validation or {}).get('games')) or 0)
    sample_strength = _clip_unit((oos_games / 140.0) + (live_games / 80.0), 0.0, 1.0)

    research_log_loss = _safe_float((validation or {}).get('log_loss'))
    live_log_loss = _safe_float((live_validation or {}).get('log_loss'))
    live_gap = 0.0
    live_relief = 0.0
    if live_log_loss is not None and research_log_loss is not None:
        gap = (live_log_loss - research_log_loss) / 0.045
        if gap >= 0:
            live_gap = _clip_unit(gap, 0.0, 1.0)
        else:
            live_relief = _clip_unit(abs(gap), 0.0, 0.6)

    realized_phase = ((((state or {}).get('realized_uncertainty') or {}).get(phase_name)) or {}) if isinstance(state, dict) else {}

    def market_context(market_name: str, model_metric: Any, baseline_metric: Any, scale: float, base_bias: float) -> Dict[str, Any]:
        regime_gap = _market_regime_gap(model_metric, baseline_metric, scale)
        realized = (realized_phase.get(market_name) or {}) if isinstance(realized_phase, dict) else {}
        realized_bias = _safe_float((realized or {}).get('bias_add')) or 0.0
        realized_penalty_scale = _safe_float((realized or {}).get('penalty_scale')) or 1.0
        realized_confidence = _clip_unit((realized or {}).get('confidence'), 0.0, 1.0)
        realized_games = int(_safe_int((realized or {}).get('settled_games')) or 0)
        realized_bets = int(_safe_int((realized or {}).get('priced_bets')) or 0)
        realized_avg_clv = _safe_float((realized or {}).get('avg_clv'))
        realized_roi = _safe_float((realized or {}).get('roi'))
        realized_reason = str((realized or {}).get('reason') or '')

        base_regime_drag = min(base_bias + (0.14 * regime_gap) + (0.08 * live_gap), 0.30)
        base_sample_relief = min((0.02 if phase_name == 'spring' else 0.04) + (0.08 * sample_strength) + (0.03 * live_relief), 0.16)
        regime_drag = min(base_regime_drag + max(realized_bias, 0.0), 0.38)
        sample_relief = min(base_sample_relief + max(-realized_bias, 0.0) + (0.02 * realized_confidence), 0.22)

        base_penalty_scale = max(
            0.55,
            min(1.00 - (0.20 * regime_gap) - (0.12 * live_gap) + (0.10 * sample_strength) + (0.04 * live_relief), 1.05),
        )
        penalty_scale = max(0.55, min(base_penalty_scale * max(min(realized_penalty_scale, 1.05), 0.55), 1.05))

        if regime_gap >= 0.66:
            stance = 'baseline is clearly outperforming the richer model'
        elif regime_gap >= 0.33:
            stance = 'baseline still has a modest edge on this market'
        else:
            stance = 'model and baseline are close enough that only a small regime drag is needed'
        if realized_confidence > 0:
            stance += f'; realized learning confidence {realized_confidence:.2f}'
        if realized_reason:
            stance += f'; {realized_reason}'
        source = 'benchmark ladder + live settled validation'
        if realized_confidence > 0 or realized_games > 0 or realized_bets > 0:
            source += ' + realized uncertainty learning'

        return {
            'market_type': market_name,
            'regime_gap': round(regime_gap, 4),
            'regime_drag': round(regime_drag, 4),
            'sample_relief': round(sample_relief, 4),
            'penalty_scale': round(penalty_scale, 4),
            'sample_strength': round(sample_strength, 4),
            'live_gap': round(live_gap, 4),
            'live_relief': round(live_relief, 4),
            'realized_bias': round(realized_bias, 4),
            'realized_penalty_scale': round(realized_penalty_scale, 4),
            'realized_confidence': round(realized_confidence, 4),
            'realized_games': realized_games,
            'realized_bets': realized_bets,
            'realized_avg_clv': realized_avg_clv,
            'realized_roi': realized_roi,
            'source': source,
            'technical_note': stance,
        }

    return {
        'moneyline': market_context(
            'moneyline',
            (benchmark_ladder.get('model_oos') or {}).get('log_loss'),
            (benchmark_ladder.get('simple_power_oos') or {}).get('log_loss'),
            0.045,
            0.05 if phase_name == 'spring' else 0.015,
        ),
        'total': market_context(
            'total',
            (benchmark_ladder.get('model_total_oos') or {}).get('rmse'),
            (benchmark_ladder.get('simple_total_oos') or {}).get('rmse'),
            0.90,
            0.08 if phase_name == 'spring' else 0.03,
        ),
        'run_line': market_context(
            'run_line',
            (benchmark_ladder.get('model_margin_oos') or {}).get('rmse'),
            (benchmark_ladder.get('simple_margin_oos') or {}).get('rmse'),
            0.75,
            0.07 if phase_name == 'spring' else 0.025,
        ),
    }

def _candidate_uncertainty(prediction: Dict[str, Any], market_type: str, sportsbook_count: Any, phase: str, calibration_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    features = prediction.get('features') or {}
    lineup_ctx = features.get('lineup_ctx') or {}
    lineups = features.get('lineups') or {}
    away_starter = features.get('away_starter') or {}
    home_starter = features.get('home_starter') or {}
    weather = features.get('weather') or {}
    venue_meta = features.get('venue_meta') or {}
    away_line = lineup_ctx.get('away') or {}
    home_line = lineup_ctx.get('home') or {}
    phase_name = str(phase or prediction.get('phase') or 'spring').lower()
    market_name = str(market_type or '').lower()

    away_conf = _clip_unit(away_starter.get('confidence'), 0.0, 1.0)
    home_conf = _clip_unit(home_starter.get('confidence'), 0.0, 1.0)
    starter_conf = (away_conf + home_conf) / 2.0
    starter_source_unc = 0.0
    for source in [str(away_starter.get('source') or '').lower(), str(home_starter.get('source') or '').lower()]:
        if source in {'prior', 'neutral', 'name_only'}:
            starter_source_unc += 0.5
        elif source not in {'current', 'current_season'} and source:
            starter_source_unc += 0.25
    starter_unc = min(1.0, (1.0 - starter_conf) + (0.20 * starter_source_unc))

    lineup_completion = _clip_unit(((float(lineups.get('away', 0) or 0) + float(lineups.get('home', 0) or 0)) / 18.0), 0.0, 1.0)
    lineup_conf = _clip_unit(((_safe_float(away_line.get('avg_confidence')) or 0.0) + (_safe_float(home_line.get('avg_confidence')) or 0.0)) / 2.0, 0.0, 1.0)
    prior_heavy = float(away_line.get('prior_heavy', 0) or 0) + float(home_line.get('prior_heavy', 0) or 0)
    prior_heavy_ratio = _clip_unit(prior_heavy / 6.0, 0.0, 1.0)
    lineup_unc = min(1.0, 1.0 - ((0.55 * lineup_completion) + (0.45 * lineup_conf)) + (0.25 * prior_heavy_ratio))

    roof_type = str(venue_meta.get('roof_type') or '').lower()
    roof_closed = ('dome' in roof_type) or ('closed' in roof_type)
    weather_source = str(weather.get('source') or 'neutral').lower()
    precip_unc = _clip_unit((_safe_float(weather.get('precip_prob')) or 0.0) / 100.0, 0.0, 1.0)
    weather_unc = 0.0
    if market_name in {'total', 'run_line'}:
        if roof_closed:
            weather_unc = 0.02
        elif weather_source == 'neutral':
            weather_unc = 0.45
        else:
            weather_unc = 0.10 + (0.20 * precip_unc)
    else:
        if roof_closed:
            weather_unc = 0.01
        elif weather_source == 'neutral':
            weather_unc = 0.18
        else:
            weather_unc = 0.05 + (0.08 * precip_unc)

    if market_name == 'moneyline':
        shrink_alpha = _clip_unit(prediction.get('probability_shrink_alpha'), 0.0, 1.0)
        disagreement = abs((_safe_float(prediction.get('p_home_model')) or 0.5) - (_safe_float(prediction.get('p_home_simple')) or 0.5))
        disagreement_unc = _clip_unit(disagreement / 0.08, 0.0, 1.0)
    elif market_name == 'total':
        shrink_alpha = _clip_unit(prediction.get('total_shrink_alpha'), 0.0, 1.0)
        disagreement = abs((_safe_float(prediction.get('total_model')) or 0.0) - (_safe_float(prediction.get('total_simple')) or 0.0))
        disagreement_unc = _clip_unit(disagreement / 1.25, 0.0, 1.0)
    else:
        shrink_alpha = _clip_unit(prediction.get('margin_shrink_alpha'), 0.0, 1.0)
        disagreement = abs((_safe_float(prediction.get('margin_model')) or 0.0) - (_safe_float(prediction.get('margin_simple')) or 0.0))
        disagreement_unc = _clip_unit(disagreement / 1.10, 0.0, 1.0)

    market_depth_unc = 1.0 - min((_safe_float(sportsbook_count) or 0.0) / (2.0 if phase_name == 'spring' else 3.0), 1.0)
    phase_unc = 0.18 if phase_name == 'spring' else 0.04
    calibration = ((calibration_context or {}).get(market_name) or {})

    if market_name == 'moneyline':
        raw_unc = (
            (0.28 * starter_unc)
            + (0.24 * lineup_unc)
            + (0.16 * market_depth_unc)
            + (0.16 * shrink_alpha)
            + (0.10 * disagreement_unc)
            + (0.06 * phase_unc)
        )
    elif market_name == 'total':
        raw_unc = (
            (0.22 * starter_unc)
            + (0.20 * lineup_unc)
            + (0.16 * weather_unc)
            + (0.14 * market_depth_unc)
            + (0.16 * shrink_alpha)
            + (0.08 * disagreement_unc)
            + (0.04 * phase_unc)
        )
    else:
        raw_unc = (
            (0.25 * starter_unc)
            + (0.18 * lineup_unc)
            + (0.14 * weather_unc)
            + (0.14 * market_depth_unc)
            + (0.17 * shrink_alpha)
            + (0.08 * disagreement_unc)
            + (0.04 * phase_unc)
        )

    raw_uncertainty = _clip_unit(raw_unc, 0.0, 1.0)
    regime_drag = _clip_unit((calibration or {}).get('regime_drag'), 0.0, 0.35)
    sample_relief = _clip_unit((calibration or {}).get('sample_relief'), 0.0, 0.18)
    calibration_bias = regime_drag - sample_relief
    uncertainty_raw = _clip_unit(raw_uncertainty + calibration_bias, 0.0, 1.0)
    uncertainty_score = round(uncertainty_raw * 100.0, 1)
    raw_score = round(raw_uncertainty * 100.0, 1)
    if uncertainty_score >= 67.0:
        uncertainty_label = 'HIGH'
    elif uncertainty_score >= 45.0:
        uncertainty_label = 'MEDIUM'
    else:
        uncertainty_label = 'LOW'
    base_penalty = max(0.40, min(1.00 - (0.55 * uncertainty_raw), 1.00))
    penalty_scale = max(0.55, min((_safe_float((calibration or {}).get('penalty_scale')) or 1.0), 1.05))
    penalty = round(max(0.30, min(base_penalty * penalty_scale, 1.00)), 4)

    components = {
        'starter certainty': starter_unc,
        'lineup certainty': lineup_unc,
        'weather certainty': weather_unc,
        'market depth': market_depth_unc,
        'model shrinkage': shrink_alpha,
        'baseline disagreement': disagreement_unc,
    }
    ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
    top_notes = [f"{label}={value:.2f}" for label, value in ranked if value >= 0.18][:3]
    if regime_drag >= 0.04:
        top_notes.append(f'regime drag={regime_drag:.2f}')
    if sample_relief >= 0.05:
        top_notes.append(f'sample relief={sample_relief:.2f}')
    if not top_notes:
        top_notes = ['setup looks relatively stable']

    if uncertainty_label == 'HIGH':
        plain = 'The edge may be real, but the setup is fragile enough that sizing should stay conservative.'
    elif uncertainty_label == 'MEDIUM':
        plain = 'The edge is playable, but there are still a few assumptions that deserve caution.'
    else:
        plain = 'The setup is comparatively clean, so the model can trust this edge more than usual.'

    return {
        'score': uncertainty_score,
        'raw_score': raw_score,
        'calibration_bias': round(calibration_bias * 100.0, 1),
        'label': uncertainty_label,
        'penalty': penalty,
        'penalty_scale': round(penalty_scale, 4),
        'notes': '; '.join(top_notes),
        'plain_english': plain,
        'source': str((calibration or {}).get('source') or 'heuristic only'),
        'technical_note': str((calibration or {}).get('technical_note') or ''),
    }


def _adjusted_expected_value(raw_ev: Any, uncertainty: Dict[str, Any], market_type: str) -> float | None:
    base_ev = _safe_float(raw_ev)
    if base_ev is None:
        return None
    market_drag = {'moneyline': 0.010, 'run_line': 0.012, 'total': 0.013}.get(str(market_type or '').lower(), 0.010)
    uncertainty_raw = (_safe_float((uncertainty or {}).get('score')) or 0.0) / 100.0
    penalty = _safe_float((uncertainty or {}).get('penalty')) or 1.0
    adjusted = max(0.0, (base_ev * penalty) - (market_drag * uncertainty_raw))
    return round(float(adjusted), 4)

def _prediction_rows(report_date: dt.date, phase: str, run_id: str, predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for prediction in predictions or []:
        game = prediction.get('game') or {}
        features = prediction.get('features') or {}
        lineups = features.get('lineups') or {}
        market_comp = prediction.get('market_comp') or {}
        rows.append({
            'run_id': run_id,
            'report_date': report_date.isoformat(),
            'phase': phase,
            'game_pk': _safe_int(game.get('game_pk')),
            'away_team': str(game.get('away_team') or ''),
            'home_team': str(game.get('home_team') or ''),
            'venue_name': str(game.get('venue_name') or ''),
            'start_time_ct': str(game.get('start_time_ct') or ''),
            'away_pitcher': str(game.get('away_pitcher') or ''),
            'home_pitcher': str(game.get('home_pitcher') or ''),
            'winner': str(prediction.get('winner') or ''),
            'winner_prob': _safe_float(prediction.get('winner_prob')),
            'confidence_tier': str(prediction.get('confidence_tier') or ''),
            'p_home': _safe_float(prediction.get('p_home')),
            'p_home_model': _safe_float(prediction.get('p_home_model')),
            'p_home_simple': _safe_float(prediction.get('p_home_simple')),
            'probability_shrink_alpha': _safe_float(prediction.get('probability_shrink_alpha')),
            'fair_home_ml': _safe_int(prediction.get('fair_home_ml')),
            'fair_away_ml': _safe_int(prediction.get('fair_away_ml')),
            'away_runs': _safe_float(prediction.get('away_runs')),
            'home_runs': _safe_float(prediction.get('home_runs')),
            'total_calibrated': _safe_float(prediction.get('total_calibrated')),
            'total_model': _safe_float(prediction.get('total_model')),
            'total_simple': _safe_float(prediction.get('total_simple')),
            'total_shrink_alpha': _safe_float(prediction.get('total_shrink_alpha')),
            'total_sigma_raw': _safe_float(prediction.get('total_sigma_raw')),
            'total_sigma': _safe_float(prediction.get('total_sigma')),
            'margin_calibrated': _safe_float(prediction.get('margin_calibrated')),
            'margin_model': _safe_float(prediction.get('margin_model')),
            'margin_simple': _safe_float(prediction.get('margin_simple')),
            'margin_shrink_alpha': _safe_float(prediction.get('margin_shrink_alpha')),
            'x_off': _safe_float(features.get('x_off')),
            'x_starter': _safe_float(features.get('x_starter')),
            'x_bullpen': _safe_float(features.get('x_bullpen')),
            'x_park': _safe_float(features.get('x_park')),
            'x_weather': _safe_float(features.get('x_weather')),
            'x_context': _safe_float(features.get('x_context')),
            'shared_env': _safe_float(features.get('shared_env')),
            'away_lineup_count': _safe_int(lineups.get('away')),
            'home_lineup_count': _safe_int(lineups.get('home')),
            'market_available': bool(market_comp.get('available')),
            'market_label': str(market_comp.get('market_label') or ''),
            'sportsbook_count': _safe_int(market_comp.get('sportsbook_count')),
            'moneyline_thesis': str((((market_comp.get('moneyline') or {}).get('explainer') or {}).get('plain_english')) or ''),
            'run_line_thesis': str((((market_comp.get('run_line') or {}).get('explainer') or {}).get('plain_english')) or ''),
            'total_thesis': str((((market_comp.get('total') or {}).get('explainer') or {}).get('plain_english')) or ''),
        })
    return pd.DataFrame(rows)


def _candidate_rows(report_date: dt.date, phase: str, run_id: str, predictions: List[Dict[str, Any]], uncertainty_context: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for prediction in predictions or []:
        game = prediction.get('game') or {}
        market_comp = prediction.get('market_comp') or {}
        if not market_comp.get('available'):
            continue
        common = {
            'run_id': run_id,
            'report_date': report_date.isoformat(),
            'phase': phase,
            'game_pk': _safe_int(game.get('game_pk')),
            'away_team': str(game.get('away_team') or ''),
            'home_team': str(game.get('home_team') or ''),
            'market_label': str(market_comp.get('market_label') or ''),
            'sportsbook_count': _safe_int(market_comp.get('sportsbook_count')),
            'confidence_tier': str(prediction.get('confidence_tier') or ''),
        }
        sportsbook_count = common['sportsbook_count']
        moneyline = market_comp.get('moneyline') or {}
        if moneyline.get('available'):
            action = moneyline.get('action') or {}
            explainer = moneyline.get('explainer') or {}
            raw_ev = _market_expected_value(moneyline.get('model_probability'), moneyline.get('best_price'))
            uncertainty = _candidate_uncertainty(prediction, 'moneyline', sportsbook_count, phase, uncertainty_context)
            adjusted_ev = _adjusted_expected_value(raw_ev, uncertainty, 'moneyline')
            action_score = _safe_float(action.get('score')) or 0.0
            edge_value = _safe_float(moneyline.get('best_edge')) or 0.0
            candidate_score = round(
                (edge_value * max(action_score, 0.0) * max((_safe_float(uncertainty.get('penalty')) or 0.0), 0.40))
                + ((adjusted_ev or 0.0) * 100.0),
                4,
            )
            rows.append({
                **common,
                'market_type': 'moneyline',
                'selection': str(moneyline.get('best_side') or ''),
                'line_value': None,
                'book_price': _safe_int(moneyline.get('best_price')),
                'fair_price': _safe_int(moneyline.get('fair_price')),
                'model_probability': _safe_float(moneyline.get('model_probability')),
                'market_probability': _safe_float(moneyline.get('market_probability')),
                'edge_value': edge_value,
                'actionability_label': str(action.get('label') or 'PASS'),
                'actionability_score': _safe_float(action.get('score')),
                'actionability_notes': str(action.get('notes') or ''),
                'expected_value_per_unit': raw_ev,
                'raw_expected_value_per_unit': raw_ev,
                'adjusted_expected_value_per_unit': adjusted_ev,
                'expected_value_method': 'priced probability EV',
                'uncertainty_score': _safe_float(uncertainty.get('score')),
                'uncertainty_base_score': _safe_float(uncertainty.get('raw_score')),
                'uncertainty_calibration_bias': _safe_float(uncertainty.get('calibration_bias')),
                'uncertainty_label': str(uncertainty.get('label') or ''),
                'uncertainty_penalty': _safe_float(uncertainty.get('penalty')),
                'uncertainty_penalty_scale': _safe_float(uncertainty.get('penalty_scale')),
                'uncertainty_notes': str(uncertainty.get('notes') or ''),
                'uncertainty_plain_english': str(uncertainty.get('plain_english') or ''),
                'uncertainty_source': str(uncertainty.get('source') or ''),
                'uncertainty_technical_note': str(uncertainty.get('technical_note') or ''),
                'bet_thesis': str(explainer.get('plain_english') or ''),
                'bet_drivers': ' | '.join(explainer.get('technical_drivers') or []),
                'risk_flags': ' | '.join(explainer.get('risk_flags') or []),
                'market_throttle': str(((action.get('market_profile') or {}).get('throttle_label')) or 'neutral'),
                'market_throttle_note': str(((action.get('market_profile') or {}).get('notes')) or ''),
                'suggested_units_preview': _suggested_units_preview(action.get('label'), action.get('score'), prediction, 'moneyline'),
                'candidate_score': candidate_score,
            })
        run_line = market_comp.get('run_line') or {}
        if run_line.get('available'):
            action = run_line.get('action') or {}
            explainer = run_line.get('explainer') or {}
            raw_ev = _market_expected_value(run_line.get('model_probability'), run_line.get('best_price'))
            uncertainty = _candidate_uncertainty(prediction, 'run_line', sportsbook_count, phase, uncertainty_context)
            adjusted_ev = _adjusted_expected_value(raw_ev, uncertainty, 'run_line')
            action_score = _safe_float(action.get('score')) or 0.0
            edge_value = _safe_float(run_line.get('best_edge')) or 0.0
            candidate_score = round(
                (edge_value * max(action_score, 0.0) * max((_safe_float(uncertainty.get('penalty')) or 0.0), 0.40))
                + ((adjusted_ev or 0.0) * 100.0),
                4,
            )
            rows.append({
                **common,
                'market_type': 'run_line',
                'selection': str(run_line.get('best_selection') or ''),
                'line_value': _safe_float(run_line.get('best_line')),
                'book_price': _safe_int(run_line.get('best_price')),
                'fair_price': _safe_int(run_line.get('fair_price')),
                'model_probability': _safe_float(run_line.get('model_probability')),
                'market_probability': _safe_float(run_line.get('market_probability')),
                'edge_value': edge_value,
                'actionability_label': str(action.get('label') or 'PASS'),
                'actionability_score': _safe_float(action.get('score')),
                'actionability_notes': str(action.get('notes') or ''),
                'expected_value_per_unit': raw_ev,
                'raw_expected_value_per_unit': raw_ev,
                'adjusted_expected_value_per_unit': adjusted_ev,
                'expected_value_method': 'priced probability EV',
                'uncertainty_score': _safe_float(uncertainty.get('score')),
                'uncertainty_base_score': _safe_float(uncertainty.get('raw_score')),
                'uncertainty_calibration_bias': _safe_float(uncertainty.get('calibration_bias')),
                'uncertainty_label': str(uncertainty.get('label') or ''),
                'uncertainty_penalty': _safe_float(uncertainty.get('penalty')),
                'uncertainty_penalty_scale': _safe_float(uncertainty.get('penalty_scale')),
                'uncertainty_notes': str(uncertainty.get('notes') or ''),
                'uncertainty_plain_english': str(uncertainty.get('plain_english') or ''),
                'uncertainty_source': str(uncertainty.get('source') or ''),
                'uncertainty_technical_note': str(uncertainty.get('technical_note') or ''),
                'bet_thesis': str(explainer.get('plain_english') or ''),
                'bet_drivers': ' | '.join(explainer.get('technical_drivers') or []),
                'risk_flags': ' | '.join(explainer.get('risk_flags') or []),
                'market_throttle': str(((action.get('market_profile') or {}).get('throttle_label')) or 'neutral'),
                'market_throttle_note': str(((action.get('market_profile') or {}).get('notes')) or ''),
                'suggested_units_preview': _suggested_units_preview(action.get('label'), action.get('score'), prediction, 'run_line'),
                'candidate_score': candidate_score,
            })
        total_market = market_comp.get('total') or {}
        if total_market.get('available'):
            action = total_market.get('action') or {}
            explainer = total_market.get('explainer') or {}
            lean = str(total_market.get('lean') or 'FLAT').upper()
            best_price = _safe_int(total_market.get('best_price'))
            model_probability = _safe_float(total_market.get('model_probability'))
            market_probability = _safe_float(total_market.get('market_probability'))
            fair_price = _safe_int(total_market.get('fair_price'))
            if model_probability is not None and best_price is not None:
                raw_ev = _market_expected_value(model_probability, best_price)
                expected_value_method = 'priced probability EV'
            else:
                raw_ev = _total_expected_value_proxy(total_market.get('diff'), best_price)
                expected_value_method = 'run-gap proxy EV'
            uncertainty = _candidate_uncertainty(prediction, 'total', sportsbook_count, phase, uncertainty_context)
            adjusted_ev = _adjusted_expected_value(raw_ev, uncertainty, 'total')
            action_score = _safe_float(action.get('score')) or 0.0
            diff_value = abs(_safe_float(total_market.get('diff')) or 0.0)
            prob_edge = abs(_safe_float(total_market.get('best_edge')) or 0.0)
            signal_value = max(diff_value, prob_edge * 10.0)
            candidate_score = round(
                (signal_value * max(action_score, 0.0) * max((_safe_float(uncertainty.get('penalty')) or 0.0), 0.40))
                + ((adjusted_ev or 0.0) * 100.0),
                4,
            )
            rows.append({
                **common,
                'market_type': 'total',
                'selection': f"{lean} {(_safe_float(total_market.get('market_total')) or 0.0):.1f}",
                'line_value': _safe_float(total_market.get('market_total')),
                'book_price': best_price,
                'fair_price': fair_price,
                'model_probability': model_probability,
                'market_probability': market_probability,
                'edge_value': _safe_float(total_market.get('best_edge')) if total_market.get('best_edge') is not None else _safe_float(total_market.get('diff')),
                'actionability_label': str(action.get('label') or 'PASS'),
                'actionability_score': _safe_float(action.get('score')),
                'actionability_notes': str(action.get('notes') or ''),
                'expected_value_per_unit': raw_ev,
                'raw_expected_value_per_unit': raw_ev,
                'adjusted_expected_value_per_unit': adjusted_ev,
                'expected_value_method': expected_value_method,
                'uncertainty_score': _safe_float(uncertainty.get('score')),
                'uncertainty_base_score': _safe_float(uncertainty.get('raw_score')),
                'uncertainty_calibration_bias': _safe_float(uncertainty.get('calibration_bias')),
                'uncertainty_label': str(uncertainty.get('label') or ''),
                'uncertainty_penalty': _safe_float(uncertainty.get('penalty')),
                'uncertainty_penalty_scale': _safe_float(uncertainty.get('penalty_scale')),
                'uncertainty_notes': str(uncertainty.get('notes') or ''),
                'uncertainty_plain_english': str(uncertainty.get('plain_english') or ''),
                'uncertainty_source': str(uncertainty.get('source') or ''),
                'uncertainty_technical_note': str(uncertainty.get('technical_note') or ''),
                'bet_thesis': str(explainer.get('plain_english') or ''),
                'bet_drivers': ' | '.join(explainer.get('technical_drivers') or []),
                'risk_flags': ' | '.join(explainer.get('risk_flags') or []),
                'market_throttle': str(((action.get('market_profile') or {}).get('throttle_label')) or 'neutral'),
                'market_throttle_note': str(((action.get('market_profile') or {}).get('notes')) or ''),
                'suggested_units_preview': _suggested_units_preview(action.get('label'), action.get('score'), prediction, 'total'),
                'candidate_score': candidate_score,
            })
    return pd.DataFrame(rows)


def _live_archive_validation_summary(live_archive: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'games': 0,
        'settled_days': 0,
        'log_loss': None,
        'accuracy': None,
        'total_mae': None,
        'total_rmse': None,
        'margin_mae': None,
        'margin_rmse': None,
        'date_span': 'N/A',
        'latest_settled_date': None,
    }
    if not isinstance(live_archive, pd.DataFrame) or live_archive.empty:
        return summary
    required = {'report_date', 'p_home', 'y_home'}
    if not required.issubset(live_archive.columns):
        return summary
    work = live_archive.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['p_home'] = pd.to_numeric(work.get('p_home'), errors='coerce').clip(1e-6, 1.0 - 1e-6)
    work['y_home'] = pd.to_numeric(work.get('y_home'), errors='coerce')
    work = work.dropna(subset=['report_date', 'p_home', 'y_home']).copy()
    if work.empty:
        return summary
    y = work['y_home'].astype(int)
    p = work['p_home'].astype(float)
    summary['games'] = int(len(work))
    summary['settled_days'] = int(work['report_date'].nunique())
    summary['log_loss'] = float((-(y * np.log(p)) - ((1 - y) * np.log(1.0 - p))).mean())
    summary['accuracy'] = float(((p >= 0.5).astype(int) == y).mean())
    if {'projected_total', 'actual_total'}.issubset(work.columns):
        total_work = work.dropna(subset=['projected_total', 'actual_total']).copy()
        if not total_work.empty:
            total_errors = pd.to_numeric(total_work['actual_total'], errors='coerce') - pd.to_numeric(total_work['projected_total'], errors='coerce')
            total_errors = total_errors.dropna()
            if not total_errors.empty:
                summary['total_mae'] = float(total_errors.abs().mean())
                summary['total_rmse'] = float(np.sqrt(np.mean(np.square(total_errors))))
    if {'projected_margin', 'actual_margin'}.issubset(work.columns):
        margin_work = work.dropna(subset=['projected_margin', 'actual_margin']).copy()
        if not margin_work.empty:
            margin_errors = pd.to_numeric(margin_work['actual_margin'], errors='coerce') - pd.to_numeric(margin_work['projected_margin'], errors='coerce')
            margin_errors = margin_errors.dropna()
            if not margin_errors.empty:
                summary['margin_mae'] = float(margin_errors.abs().mean())
                summary['margin_rmse'] = float(np.sqrt(np.mean(np.square(margin_errors))))
    date_values = sorted(work['report_date'].unique().tolist())
    if date_values:
        summary['date_span'] = f"{date_values[0].isoformat()} to {date_values[-1].isoformat()}"
        summary['latest_settled_date'] = date_values[-1].isoformat()
    return summary


def _market_proof_lookup(market_proof: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in ((market_proof or {}).get('by_market') or []):
        key = str((row or {}).get('market_type') or '').lower().strip()
        if key:
            lookup[key] = dict(row or {})
    return lookup


def _market_proof_frames(report_date: dt.date, run_id: str, market_proof: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_iso = report_date.isoformat()
    summary_frame = pd.DataFrame([{
        'run_id': run_id,
        'as_of_report_date': report_iso,
        'available': bool((market_proof or {}).get('available')),
        'settled_bets': _safe_int((market_proof or {}).get('settled_bets')),
        'units': _safe_float((market_proof or {}).get('units')),
        'roi': _safe_float((market_proof or {}).get('roi')),
        'avg_clv': _safe_float((market_proof or {}).get('avg_clv')),
    }])
    by_market_rows: list[dict[str, Any]] = []
    for row in ((market_proof or {}).get('by_market') or []):
        by_market_rows.append({
            'run_id': run_id,
            'as_of_report_date': report_iso,
            'market_type': str((row or {}).get('market_type') or ''),
            'bets': _safe_int((row or {}).get('bets')),
            'units': _safe_float((row or {}).get('units')),
            'roi': _safe_float((row or {}).get('roi')),
            'avg_clv': _safe_float((row or {}).get('avg_clv')),
        })
    by_market_frame = pd.DataFrame(by_market_rows, columns=[
        'run_id',
        'as_of_report_date',
        'market_type',
        'bets',
        'units',
        'roi',
        'avg_clv',
    ])
    return summary_frame, by_market_frame


def _run_snapshot(report_date: dt.date, phase: str, weights: Dict[str, Any], state: Dict[str, Any], predictions: List[Dict[str, Any]], validation: Dict[str, Any], live_validation: Dict[str, Any], benchmark_ladder: Dict[str, Any], market_proof: Dict[str, Any], warnings: List[str], report_path: str, workbook_path: str, run_id: str) -> pd.DataFrame:
    strongest = max(predictions, key=lambda item: abs((_safe_float(item.get('winner_prob')) or 0.5) - 0.5)) if predictions else None
    market_games = sum(1 for item in predictions if (item.get('market_comp') or {}).get('available'))
    uncertainty_context = _build_uncertainty_calibration_context(phase, validation, live_validation, benchmark_ladder, state)
    candidates = _candidate_rows(report_date, phase, run_id, predictions, uncertainty_context)
    probability_shrink = ((state.get('probability_shrinkage') or {}).get(phase) or {})
    total_shrink = ((state.get('total_shrinkage') or {}).get(phase) or {})
    margin_shrink = ((state.get('margin_shrinkage') or {}).get(phase) or {})
    market_proof_lookup = _market_proof_lookup(market_proof)
    moneyline_proof = market_proof_lookup.get('moneyline') or {}
    total_proof = market_proof_lookup.get('total') or {}
    run_line_proof = market_proof_lookup.get('run_line') or {}
    row = {
        'uncertainty_moneyline_regime_drag': _safe_float((uncertainty_context.get('moneyline') or {}).get('regime_drag')),
        'uncertainty_total_regime_drag': _safe_float((uncertainty_context.get('total') or {}).get('regime_drag')),
        'uncertainty_run_line_regime_drag': _safe_float((uncertainty_context.get('run_line') or {}).get('regime_drag')),
        'uncertainty_moneyline_penalty_scale': _safe_float((uncertainty_context.get('moneyline') or {}).get('penalty_scale')),
        'uncertainty_total_penalty_scale': _safe_float((uncertainty_context.get('total') or {}).get('penalty_scale')),
        'uncertainty_run_line_penalty_scale': _safe_float((uncertainty_context.get('run_line') or {}).get('penalty_scale')),
        'uncertainty_moneyline_realized_bias': _safe_float((uncertainty_context.get('moneyline') or {}).get('realized_bias')),
        'uncertainty_total_realized_bias': _safe_float((uncertainty_context.get('total') or {}).get('realized_bias')),
        'uncertainty_run_line_realized_bias': _safe_float((uncertainty_context.get('run_line') or {}).get('realized_bias')),
        'uncertainty_moneyline_realized_confidence': _safe_float((uncertainty_context.get('moneyline') or {}).get('realized_confidence')),
        'uncertainty_total_realized_confidence': _safe_float((uncertainty_context.get('total') or {}).get('realized_confidence')),
        'uncertainty_run_line_realized_confidence': _safe_float((uncertainty_context.get('run_line') or {}).get('realized_confidence')),
        'uncertainty_moneyline_realized_games': _safe_int((uncertainty_context.get('moneyline') or {}).get('realized_games')),
        'uncertainty_total_realized_games': _safe_int((uncertainty_context.get('total') or {}).get('realized_games')),
        'uncertainty_run_line_realized_games': _safe_int((uncertainty_context.get('run_line') or {}).get('realized_games')),
        'uncertainty_moneyline_realized_bets': _safe_int((uncertainty_context.get('moneyline') or {}).get('realized_bets')),
        'uncertainty_total_realized_bets': _safe_int((uncertainty_context.get('total') or {}).get('realized_bets')),
        'uncertainty_run_line_realized_bets': _safe_int((uncertainty_context.get('run_line') or {}).get('realized_bets')),
        'uncertainty_moneyline_realized_avg_clv': _safe_float((uncertainty_context.get('moneyline') or {}).get('realized_avg_clv')),
        'uncertainty_total_realized_avg_clv': _safe_float((uncertainty_context.get('total') or {}).get('realized_avg_clv')),
        'uncertainty_run_line_realized_avg_clv': _safe_float((uncertainty_context.get('run_line') or {}).get('realized_avg_clv')),
        'uncertainty_moneyline_realized_roi': _safe_float((uncertainty_context.get('moneyline') or {}).get('realized_roi')),
        'uncertainty_total_realized_roi': _safe_float((uncertainty_context.get('total') or {}).get('realized_roi')),
        'uncertainty_run_line_realized_roi': _safe_float((uncertainty_context.get('run_line') or {}).get('realized_roi')),
        'run_id': run_id,
        'created_ts': dt.datetime.now(dt.timezone.utc).isoformat(),
        'report_date': report_date.isoformat(),
        'phase': phase,
        'prediction_count': int(len(predictions or [])),
        'market_game_count': int(market_games),
        'actionable_market_count': int(len(candidates[candidates.get('actionability_label').astype(str).isin(['WATCH', 'BET', 'MUST TAKE'])])) if not candidates.empty else 0,
        'must_take_count': int(len(candidates[candidates.get('actionability_label').astype(str) == 'MUST TAKE'])) if not candidates.empty else 0,
        'report_path': report_path,
        'workbook_path': workbook_path,
        'validation_games': _safe_int(validation.get('games')),
        'validation_oos_games': _safe_int(validation.get('oos_games')),
        'validation_log_loss': _safe_float(validation.get('log_loss')),
        'validation_accuracy': _safe_float(validation.get('accuracy')),
        'validation_total_mae': _safe_float(validation.get('total_mae')),
        'validation_total_rmse': _safe_float(validation.get('total_rmse')),
        'validation_margin_mae': _safe_float(validation.get('margin_mae')),
        'validation_margin_rmse': _safe_float(validation.get('margin_rmse')),
        'live_validation_games': _safe_int(live_validation.get('games')),
        'live_validation_settled_days': _safe_int(live_validation.get('settled_days')),
        'live_validation_log_loss': _safe_float(live_validation.get('log_loss')),
        'live_validation_accuracy': _safe_float(live_validation.get('accuracy')),
        'live_validation_total_mae': _safe_float(live_validation.get('total_mae')),
        'live_validation_total_rmse': _safe_float(live_validation.get('total_rmse')),
        'live_validation_margin_mae': _safe_float(live_validation.get('margin_mae')),
        'live_validation_margin_rmse': _safe_float(live_validation.get('margin_rmse')),
        'live_validation_date_span': str(live_validation.get('date_span') or 'N/A'),
        'live_validation_latest_settled_date': str(live_validation.get('latest_settled_date') or ''),
        'benchmark_model_log_loss': _safe_float((benchmark_ladder.get('model_oos') or {}).get('log_loss')),
        'benchmark_simple_log_loss': _safe_float((benchmark_ladder.get('simple_power_oos') or {}).get('log_loss')),
        'benchmark_model_total_rmse': _safe_float((benchmark_ladder.get('model_total_oos') or {}).get('rmse')),
        'benchmark_simple_total_rmse': _safe_float((benchmark_ladder.get('simple_total_oos') or {}).get('rmse')),
        'benchmark_shrunk_total_rmse': _safe_float((benchmark_ladder.get('shrunk_total_oos') or {}).get('rmse')),
        'benchmark_model_margin_rmse': _safe_float((benchmark_ladder.get('model_margin_oos') or {}).get('rmse')),
        'benchmark_simple_margin_rmse': _safe_float((benchmark_ladder.get('simple_margin_oos') or {}).get('rmse')),
        'benchmark_shrunk_margin_rmse': _safe_float((benchmark_ladder.get('shrunk_margin_oos') or {}).get('rmse')),
        'w_off': _safe_float(weights.get('w_off')),
        'w_starter': _safe_float(weights.get('w_starter')),
        'w_bullpen': _safe_float(weights.get('w_bullpen')),
        'w_park': _safe_float(weights.get('w_park')),
        'w_weather': _safe_float(weights.get('w_weather')),
        'w_context': _safe_float(weights.get('w_context')),
        'home_field': _safe_float(weights.get('home_field')),
        'probability_shrink_alpha': _safe_float(probability_shrink.get('alpha')),
        'total_shrink_alpha': _safe_float(total_shrink.get('alpha')),
        'margin_shrink_alpha': _safe_float(margin_shrink.get('alpha')),
        'strongest_game': '' if strongest is None else f"{strongest['game']['away_team']} @ {strongest['game']['home_team']}",
        'strongest_pick': '' if strongest is None else str(strongest.get('winner') or ''),
        'strongest_prob': None if strongest is None else _safe_float(strongest.get('winner_prob')),
        'market_proof_available': bool(market_proof.get('available')),
        'market_proof_settled_bets': _safe_int(market_proof.get('settled_bets')),
        'market_proof_roi': _safe_float(market_proof.get('roi')),
        'market_proof_avg_clv': _safe_float(market_proof.get('avg_clv')),
        'market_proof_moneyline_bets': _safe_int(moneyline_proof.get('bets')),
        'market_proof_moneyline_units': _safe_float(moneyline_proof.get('units')),
        'market_proof_moneyline_roi': _safe_float(moneyline_proof.get('roi')),
        'market_proof_moneyline_avg_clv': _safe_float(moneyline_proof.get('avg_clv')),
        'market_proof_total_bets': _safe_int(total_proof.get('bets')),
        'market_proof_total_units': _safe_float(total_proof.get('units')),
        'market_proof_total_roi': _safe_float(total_proof.get('roi')),
        'market_proof_total_avg_clv': _safe_float(total_proof.get('avg_clv')),
        'market_proof_run_line_bets': _safe_int(run_line_proof.get('bets')),
        'market_proof_run_line_units': _safe_float(run_line_proof.get('units')),
        'market_proof_run_line_roi': _safe_float(run_line_proof.get('roi')),
        'market_proof_run_line_avg_clv': _safe_float(run_line_proof.get('avg_clv')),
        'warnings_json': _safe_json(warnings or []),
        'state_json': _safe_json(state or {}),
    }
    return pd.DataFrame([row])


def _sync_reference_tables(con) -> None:
    _replace_table(con, 'clean_backtest_log_mirror', _load_csv(CLEAN_BACKTEST_LOG_PATH))
    _replace_table(con, 'live_prediction_archive_mirror', _load_csv(LIVE_PREDICTION_ARCHIVE_PATH))
    _replace_table(con, 'bet_journal_mirror', _load_sqlite_table(SPORTSBOOK_DB_PATH, 'bet_journal'))
    _replace_table(con, 'odds_snapshots_mirror', _load_sqlite_table(SPORTSBOOK_DB_PATH, 'odds_snapshots'))
    summary = _load_json(CLEAN_BACKTEST_SUMMARY_PATH)
    if summary:
        _replace_table(con, 'clean_backtest_summary_mirror', pd.DataFrame([summary]))
    model_state = _load_json(MODEL_STATE_PATH)
    if model_state:
        _replace_table(con, 'model_state_current', pd.DataFrame([{'state_json': _safe_json(model_state)}]))


def _canonical_market_type_rows(frame: pd.DataFrame, window_label: str = CANONICAL_POLICY_WINDOW_LABEL) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or 'market_type' not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    if 'window_label' in work.columns:
        preferred = work[work['window_label'].astype(str) == str(window_label)].copy()
        if not preferred.empty:
            work = preferred
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if 'as_of_report_date' in work.columns:
        sort_cols.append('as_of_report_date')
        ascending.append(False)
    if 'run_id' in work.columns:
        sort_cols.append('run_id')
        ascending.append(False)
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=ascending)
    work = work.drop_duplicates(subset=['market_type'], keep='first').reset_index(drop=True)
    return work


def _load_previous_market_type_rows(con, window_label: str = CANONICAL_POLICY_WINDOW_LABEL) -> pd.DataFrame:
    try:
        existing = con.execute(
            'SELECT * FROM policy_research_market_type_recommendations WHERE window_label = ? ORDER BY as_of_report_date DESC, run_id DESC',
            [window_label],
        ).df()
    except Exception:
        return pd.DataFrame()
    return _canonical_market_type_rows(existing, window_label)


def _market_policy_summary_fields(current_rows: pd.DataFrame, previous_rows: pd.DataFrame, window_label: str = CANONICAL_POLICY_WINDOW_LABEL) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'market_policy_canonical_window': window_label,
        'market_policy_alignment_status': 'Unavailable',
        'market_policy_distinct_active_policy_count': 0,
        'market_policy_real_market_count': 0,
        'market_policy_no_call_count': 0,
        'market_policy_change_count': 0,
        'market_policy_strength_change_count': 0,
        'market_policy_high_strength_count': 0,
        'market_policy_medium_strength_count': 0,
        'market_policy_low_strength_count': 0,
        'market_policy_summary_technical': 'No market-aware policy snapshot is available yet.',
        'market_policy_summary_plain_english': 'The dashboard has not stored a market-specific bankroll view yet.',
    }
    current = _canonical_market_type_rows(current_rows, window_label)
    previous = _canonical_market_type_rows(previous_rows, window_label)
    if current.empty:
        return summary

    active_policies: list[str] = []
    real_market_count = 0
    no_call_count = 0
    change_count = 0
    strength_change_count = 0
    strength_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    per_market_descriptions: list[str] = []

    for market_key, market_label in MARKET_POLICY_TYPES:
        prefix = f'market_policy_{market_key}'
        current_row = current[current['market_type'].astype(str).str.lower() == market_key]
        previous_row = previous[previous['market_type'].astype(str).str.lower() == market_key]
        row = current_row.iloc[0] if not current_row.empty else pd.Series(dtype=object)
        prev = previous_row.iloc[0] if not previous_row.empty else pd.Series(dtype=object)

        policy_name = str(row.get('recommended_policy') or 'No call yet')
        posture_name = str(row.get('recommended_posture') or '')
        evidence_source = str(row.get('evidence_source') or '')
        evidence_strength = str(row.get('evidence_strength') or 'Low')
        used_real_market = bool(row.get('used_real_market')) if not row.empty else False
        previous_policy = str(prev.get('recommended_policy') or '') if not prev.empty else ''
        previous_posture = str(prev.get('recommended_posture') or '') if not prev.empty else ''
        previous_strength = str(prev.get('evidence_strength') or '') if not prev.empty else ''

        policy_changed = bool((not prev.empty) and ((policy_name != previous_policy) or (posture_name != previous_posture)))
        strength_changed = bool((not prev.empty) and (evidence_strength != previous_strength))

        summary[f'{prefix}_policy'] = policy_name
        summary[f'{prefix}_posture'] = posture_name
        summary[f'{prefix}_evidence_source'] = evidence_source
        summary[f'{prefix}_evidence_strength'] = evidence_strength
        summary[f'{prefix}_used_real_market'] = used_real_market
        summary[f'{prefix}_policy_changed'] = policy_changed
        summary[f'{prefix}_strength_changed'] = strength_changed

        if used_real_market:
            real_market_count += 1
        if policy_name.lower().startswith('no call'):
            no_call_count += 1
            per_market_descriptions.append(f'{market_label} is still waiting on enough priced-bet evidence.')
        else:
            active_policies.append(policy_name)
            per_market_descriptions.append(f'{market_label} currently leans {policy_name} with {evidence_strength.lower()}-strength evidence.')
        if policy_changed:
            change_count += 1
        if strength_changed:
            strength_change_count += 1
        if evidence_strength in strength_counts:
            strength_counts[evidence_strength] += 1

    active_policy_count = len(set(active_policies))
    if no_call_count == len(MARKET_POLICY_TYPES):
        alignment_status = 'Awaiting evidence'
        technical = 'All tracked markets are still below the market-specific takeover threshold, so no market-level bankroll recommendation should be trusted yet.'
        plain = 'The model still needs settled priced bets before it can trust separate bankroll rules by market.'
    elif active_policy_count <= 1 and no_call_count == 0:
        alignment_status = 'Aligned'
        aligned_policy = active_policies[0] if active_policies else 'No call yet'
        technical = f'All tracked markets currently converge on {aligned_policy}, so the market-aware bankroll view is internally aligned.'
        plain = f'Moneyline, totals, and run line are all pointing toward the same bankroll style right now: {aligned_policy}.'
    elif active_policy_count <= 1:
        alignment_status = 'Partially aligned'
        aligned_policy = active_policies[0] if active_policies else 'No call yet'
        technical = f'At least one market has a usable policy signal ({aligned_policy}), but other markets are still below the takeover threshold.'
        plain = 'One part of the board has a usable bankroll lean, but the rest still need more priced-bet evidence before they should be trusted.'
    else:
        alignment_status = 'Split'
        technical = 'Different market types are converging toward different bankroll policies, so a single slate-wide staking rule would be too blunt.'
        plain = 'Moneyline, totals, and run line are not behaving the same way, so the bankroll plan should stay market-specific.'

    if change_count > 0 or strength_change_count > 0:
        technical += f' Compared with the prior stored snapshot, {change_count} market policy setting(s) and {strength_change_count} evidence-strength setting(s) changed.'
        plain += f' Since the last stored run, {change_count} market recommendation(s) changed and {strength_change_count} evidence rating(s) moved.'
    else:
        technical += ' No market-level recommendation changes were detected versus the prior stored snapshot.'
        plain += ' Nothing materially changed versus the previous stored recommendation snapshot.'

    summary.update({
        'market_policy_alignment_status': alignment_status,
        'market_policy_distinct_active_policy_count': active_policy_count,
        'market_policy_real_market_count': real_market_count,
        'market_policy_no_call_count': no_call_count,
        'market_policy_change_count': change_count,
        'market_policy_strength_change_count': strength_change_count,
        'market_policy_high_strength_count': strength_counts['High'],
        'market_policy_medium_strength_count': strength_counts['Medium'],
        'market_policy_low_strength_count': strength_counts['Low'],
        'market_policy_summary_technical': technical,
        'market_policy_summary_plain_english': plain,
        'market_policy_market_snapshot': ' '.join(per_market_descriptions),
    })
    return summary


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


def _dynamic_target_snapshot(run_values: Dict[str, Any]) -> tuple[float, pd.DataFrame]:
    phase_key = str(run_values.get('phase') or 'regular').lower()
    if phase_key not in {'spring', 'regular'}:
        phase_key = 'regular'

    research_games = int(_safe_int(run_values.get('validation_games')) or 0)
    live_games = int(_safe_int(run_values.get('live_validation_games')) or 0)
    settled_days = int(_safe_int(run_values.get('live_validation_settled_days')) or 0)
    priced_bets = int(_safe_int(run_values.get('market_proof_settled_bets')) or 0)

    if phase_key == 'spring':
        progress = (
            0.80 * _frictional_progress(research_games, 180) +
            0.20 * _frictional_progress(priced_bets, 30)
        )
        current_log_loss = run_values.get('validation_log_loss')
        current_accuracy = run_values.get('validation_accuracy')
        current_total_rmse = run_values.get('validation_total_rmse')
        current_margin_rmse = run_values.get('validation_margin_rmse')
    else:
        progress = (
            0.55 * _frictional_progress(live_games, 300) +
            0.15 * _frictional_progress(settled_days, 30) +
            0.30 * _frictional_progress(priced_bets, 120)
        )
        current_log_loss = run_values.get('live_validation_log_loss')
        current_accuracy = run_values.get('live_validation_accuracy')
        current_total_rmse = run_values.get('live_validation_total_rmse')
        current_margin_rmse = run_values.get('live_validation_margin_rmse')
    progress = max(0.0, min(progress, 1.0))

    metric_specs = [
        {
            'metric_key': 'moneyline_log_loss',
            'metric_label': 'Moneyline log loss',
            'current': current_log_loss,
            'baseline': run_values.get('validation_log_loss'),
            'stretch': max((_safe_float(run_values.get('validation_log_loss')) or 0.72) - (0.020 if phase_key == 'spring' else 0.035), 0.680 if phase_key == 'spring' else 0.640),
            'higher_is_better': False,
        },
        {
            'metric_key': 'moneyline_accuracy',
            'metric_label': 'Moneyline accuracy',
            'current': current_accuracy,
            'baseline': run_values.get('validation_accuracy'),
            'stretch': min((_safe_float(run_values.get('validation_accuracy')) or 0.52) + (0.020 if phase_key == 'spring' else 0.035), 0.560 if phase_key == 'spring' else 0.600),
            'higher_is_better': True,
        },
        {
            'metric_key': 'totals_rmse',
            'metric_label': 'Totals RMSE',
            'current': current_total_rmse,
            'baseline': run_values.get('validation_total_rmse'),
            'stretch': max((_safe_float(run_values.get('validation_total_rmse')) or 5.0) - (0.120 if phase_key == 'spring' else 0.250), 4.800 if phase_key == 'spring' else 4.550),
            'higher_is_better': False,
        },
        {
            'metric_key': 'margin_rmse',
            'metric_label': 'Margin RMSE',
            'current': current_margin_rmse,
            'baseline': run_values.get('validation_margin_rmse'),
            'stretch': max((_safe_float(run_values.get('validation_margin_rmse')) or 5.0) - (0.120 if phase_key == 'spring' else 0.250), 4.900 if phase_key == 'spring' else 4.500),
            'higher_is_better': False,
        },
        {
            'metric_key': 'avg_clv',
            'metric_label': 'Average CLV',
            'current': run_values.get('market_proof_avg_clv'),
            'baseline': 0.0,
            'stretch': 0.010 if phase_key == 'spring' else 0.018,
            'higher_is_better': True,
        },
        {
            'metric_key': 'roi',
            'metric_label': 'ROI',
            'current': run_values.get('market_proof_roi'),
            'baseline': 0.0,
            'stretch': 0.020 if phase_key == 'spring' else 0.035,
            'higher_is_better': True,
        },
    ]

    rows = []
    report_date = str(run_values.get('report_date') or '')
    created_ts = str(run_values.get('created_ts') or '')
    run_id = str(run_values.get('run_id') or '')
    for spec in metric_specs:
        target_value = _blend_target(spec['baseline'], spec['stretch'], progress)
        current_value = _safe_float(spec['current'])
        gap_value = None if current_value is None or target_value is None else (current_value - target_value if spec['higher_is_better'] else target_value - current_value)
        tolerance = 0.003
        status = 'Sample building'
        if gap_value is not None:
            if gap_value > tolerance:
                status = 'Ahead of target'
            elif gap_value < -tolerance:
                status = 'Behind target'
            else:
                status = 'On pace'
        rows.append({
            'run_id': run_id,
            'as_of_report_date': report_date,
            'created_ts': created_ts,
            'phase': phase_key,
            'metric_key': spec['metric_key'],
            'metric_label': spec['metric_label'],
            'current_value': current_value,
            'dynamic_target_value': _safe_float(target_value),
            'baseline_value': _safe_float(spec['baseline']),
            'long_run_goal_value': _safe_float(spec['stretch']),
            'progress_pct': round(progress, 4),
            'live_games': live_games,
            'settled_days': settled_days,
            'priced_bets': priced_bets,
            'research_games': research_games,
            'status': status,
            'gap_value': _safe_float(gap_value),
        })
    return progress, pd.DataFrame(rows)


def _daily_kpi_snapshot_frame(run_values: Dict[str, Any], target_progress: float) -> pd.DataFrame:
    return pd.DataFrame([{
        'run_id': str(run_values.get('run_id') or ''),
        'report_date': str(run_values.get('report_date') or ''),
        'created_ts': str(run_values.get('created_ts') or ''),
        'phase': str(run_values.get('phase') or ''),
        'prediction_count': _safe_int(run_values.get('prediction_count')),
        'candidate_count': _safe_int(run_values.get('candidate_count')),
        'portfolio_recommendation_count': _safe_int(run_values.get('portfolio_recommendation_count')),
        'validation_log_loss': _safe_float(run_values.get('validation_log_loss')),
        'validation_accuracy': _safe_float(run_values.get('validation_accuracy')),
        'validation_total_rmse': _safe_float(run_values.get('validation_total_rmse')),
        'validation_margin_rmse': _safe_float(run_values.get('validation_margin_rmse')),
        'live_validation_games': _safe_int(run_values.get('live_validation_games')),
        'live_validation_settled_days': _safe_int(run_values.get('live_validation_settled_days')),
        'live_validation_log_loss': _safe_float(run_values.get('live_validation_log_loss')),
        'live_validation_accuracy': _safe_float(run_values.get('live_validation_accuracy')),
        'live_validation_total_rmse': _safe_float(run_values.get('live_validation_total_rmse')),
        'live_validation_margin_rmse': _safe_float(run_values.get('live_validation_margin_rmse')),
        'market_proof_settled_bets': _safe_int(run_values.get('market_proof_settled_bets')),
        'market_proof_roi': _safe_float(run_values.get('market_proof_roi')),
        'market_proof_avg_clv': _safe_float(run_values.get('market_proof_avg_clv')),
        'market_proof_total_roi': _safe_float(run_values.get('market_proof_total_roi')),
        'market_proof_total_avg_clv': _safe_float(run_values.get('market_proof_total_avg_clv')),
        'market_proof_run_line_roi': _safe_float(run_values.get('market_proof_run_line_roi')),
        'market_proof_run_line_avg_clv': _safe_float(run_values.get('market_proof_run_line_avg_clv')),
        'policy_alignment_status': str(run_values.get('market_policy_alignment_status') or ''),
        'market_policy_real_market_count': _safe_int(run_values.get('market_policy_real_market_count')),
        'market_policy_change_count': _safe_int(run_values.get('market_policy_change_count')),
        'target_progress_pct': round(target_progress, 4),
    }])


def _alert_history_frame(run_values: Dict[str, Any], warnings: List[str]) -> pd.DataFrame:
    run_id = str(run_values.get('run_id') or '')
    report_date = str(run_values.get('report_date') or '')
    created_ts = str(run_values.get('created_ts') or '')
    phase = str(run_values.get('phase') or '')
    rows: list[dict[str, Any]] = []

    def add_alert(severity: str, category: str, title: str, technical: str, plain_english: str) -> None:
        rows.append({
            'run_id': run_id,
            'as_of_report_date': report_date,
            'created_ts': created_ts,
            'phase': phase,
            'severity': severity,
            'category': category,
            'title': title,
            'technical': technical,
            'plain_english': plain_english,
        })

    live_log_loss = _safe_float(run_values.get('live_validation_log_loss'))
    research_log_loss = _safe_float(run_values.get('validation_log_loss'))
    if live_log_loss is not None and research_log_loss is not None and live_log_loss > research_log_loss + 0.02:
        add_alert(
            'Warning',
            'Model health',
            'Live moneyline performance is lagging replay',
            f'Live settled log loss ({live_log_loss:.4f}) is more than 0.02 worse than the research replay baseline ({research_log_loss:.4f}).',
            'Recent live moneyline performance is weaker than the benchmark and deserves a closer look.',
        )

    total_clv = _safe_float(run_values.get('market_proof_total_avg_clv'))
    total_roi = _safe_float(run_values.get('market_proof_total_roi'))
    if total_clv is not None and total_clv < 0:
        add_alert(
            'Warning',
            'Totals proof',
            'Totals CLV is still negative',
            f'Totals average CLV is {total_clv:.3f} and totals ROI is {_safe_float(total_roi) if total_roi is not None else float("nan"):.3f}.',
            'Totals are still not consistently beating the close, so they should stay on a shorter leash.',
        )

    run_line_clv = _safe_float(run_values.get('market_proof_run_line_avg_clv'))
    run_line_roi = _safe_float(run_values.get('market_proof_run_line_roi'))
    if run_line_clv is not None and run_line_clv < -0.10:
        add_alert(
            'Warning',
            'Run-line proof',
            'Run-line CLV is still negative',
            f'Run-line average CLV is {run_line_clv:.3f} and run-line ROI is {_safe_float(run_line_roi) if run_line_roi is not None else float("nan"):.3f}.',
            'Run lines may still be making money short term, but the market is not validating those prices yet, so they should be treated more cautiously.',
        )

    overall_clv = _safe_float(run_values.get('market_proof_avg_clv'))
    overall_roi = _safe_float(run_values.get('market_proof_roi'))
    if overall_clv is not None and overall_roi is not None and overall_clv < 0 and overall_roi > 0:
        add_alert(
            'Warning',
            'Market proof',
            'Profit is positive but CLV is negative',
            f'Overall ROI is {overall_roi:.3f} while average CLV is {overall_clv:.3f}.',
            'The short-term profit line looks good, but the market is not yet validating that edge quality.',
        )

    if int(_safe_int(run_values.get('market_policy_change_count')) or 0) > 0:
        add_alert(
            'Info',
            'Policy',
            'Market policy recommendation changed',
            f"{int(_safe_int(run_values.get('market_policy_change_count')) or 0)} market policy setting(s) changed versus the prior stored run.",
            'At least one bankroll recommendation changed today, so position sizing should be read carefully.',
        )

    for warning in warnings or []:
        text = str(warning or '').strip()
        if text:
            add_alert('Note', 'Workflow', 'Run warning', text, text)

    return pd.DataFrame(rows)


def _archive_attribution_marts(live_archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if live_archive.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = live_archive.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    if 'phase' not in work.columns:
        work['phase'] = 'unknown'
    numeric_cols = [
        'p_home', 'y_home', 'projected_total', 'actual_total', 'projected_margin', 'actual_margin',
        'away_lineup_count', 'home_lineup_count', 'away_lineup_confidence', 'home_lineup_confidence',
        'away_starter_confidence', 'home_starter_confidence', 'away_short_start_risk', 'home_short_start_risk',
        'away_tto_risk', 'home_tto_risk', 'away_bullpen_stress', 'home_bullpen_stress',
        'away_leverage_availability', 'home_leverage_availability',
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work.get(col), errors='coerce')
    work = work.dropna(subset=['report_date']).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    work['lineup_flag'] = ((work.get('away_lineup_count', 9).fillna(9) + work.get('home_lineup_count', 9).fillna(9)) < 18) | (
        ((work.get('away_lineup_confidence', 1.0).fillna(1.0) + work.get('home_lineup_confidence', 1.0).fillna(1.0)) / 2.0) < 0.75
    )
    work['starter_flag'] = (((work.get('away_starter_confidence', 1.0).fillna(1.0) + work.get('home_starter_confidence', 1.0).fillna(1.0)) / 2.0) < 0.65)
    work['short_start_flag'] = (work.get('away_short_start_risk', 0.0).fillna(0.0).combine(work.get('home_short_start_risk', 0.0).fillna(0.0), max) >= 0.55)
    work['tto_flag'] = (work.get('away_tto_risk', 0.0).fillna(0.0).combine(work.get('home_tto_risk', 0.0).fillna(0.0), max) >= 0.45)
    work['bullpen_flag'] = (
        (work.get('away_bullpen_stress', 0.0).fillna(0.0).combine(work.get('home_bullpen_stress', 0.0).fillna(0.0), max) >= 0.35) |
        (work.get('away_leverage_availability', 0.0).fillna(0.0).combine(work.get('home_leverage_availability', 0.0).fillna(0.0), min) <= -0.25)
    )

    market_rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        shared = {
            'report_date': row.get('report_date'),
            'phase': str(row.get('phase') or ''),
            'lineup_flag': bool(row.get('lineup_flag')),
            'starter_flag': bool(row.get('starter_flag')),
            'short_start_flag': bool(row.get('short_start_flag')),
            'tto_flag': bool(row.get('tto_flag')),
            'bullpen_flag': bool(row.get('bullpen_flag')),
        }
        p_home = _safe_float(row.get('p_home'))
        y_home = _safe_float(row.get('y_home'))
        if p_home is not None and y_home is not None:
            err = abs(y_home - p_home)
            market_rows.append({**shared, 'market_type': 'moneyline', 'abs_error': err, 'sq_error': err ** 2})
        proj_total = _safe_float(row.get('projected_total'))
        actual_total = _safe_float(row.get('actual_total'))
        if proj_total is not None and actual_total is not None:
            err = abs(actual_total - proj_total)
            market_rows.append({**shared, 'market_type': 'total', 'abs_error': err, 'sq_error': err ** 2})
        proj_margin = _safe_float(row.get('projected_margin'))
        actual_margin = _safe_float(row.get('actual_margin'))
        if proj_margin is not None and actual_margin is not None:
            err = abs(actual_margin - proj_margin)
            market_rows.append({**shared, 'market_type': 'run_line', 'abs_error': err, 'sq_error': err ** 2})
    market_work = pd.DataFrame(market_rows)
    if market_work.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_market = market_work.groupby(['phase', 'market_type'], dropna=False).agg(
        games=('abs_error', 'size'),
        avg_abs_error=('abs_error', 'mean'),
        rmse=('sq_error', lambda s: float(np.sqrt(np.mean(s))) if len(s) else np.nan),
        latest_report_date=('report_date', 'max'),
    ).reset_index()

    driver_rows: list[dict[str, Any]] = []
    for driver_col, driver_label in [
        ('lineup_flag', 'Lineup uncertainty'),
        ('starter_flag', 'Starter uncertainty'),
        ('short_start_flag', 'Short-start risk'),
        ('tto_flag', 'Times-through-order risk'),
        ('bullpen_flag', 'Bullpen stress'),
    ]:
        grouped = market_work.groupby(['phase', 'market_type', driver_col], dropna=False).agg(
            games=('abs_error', 'size'),
            avg_abs_error=('abs_error', 'mean'),
            rmse=('sq_error', lambda s: float(np.sqrt(np.mean(s))) if len(s) else np.nan),
        ).reset_index()
        grouped['driver'] = driver_label
        grouped['driver_state'] = grouped[driver_col].map({True: 'Flagged', False: 'Clean'})
        driver_rows.extend(grouped.drop(columns=[driver_col]).to_dict('records'))

    return by_market, pd.DataFrame(driver_rows)

def export_quant_snapshot(report_date: dt.date, phase: str, weights: Dict[str, Any], state: Dict[str, Any], predictions: List[Dict[str, Any]], validation: Dict[str, Any], benchmark_ladder: Dict[str, Any], market_proof: Dict[str, Any], warnings: List[str], report_path: str, workbook_path: str) -> Dict[str, Any]:
    duckdb, error = _import_duckdb()
    if duckdb is None:
        return {'exported': False, 'db_path': QUANT_DB_PATH, 'warning': f'DuckDB export skipped: {error}'}
    run_id = f"{report_date.isoformat()}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    live_archive = _load_csv(LIVE_PREDICTION_ARCHIVE_PATH)
    live_validation = _live_archive_validation_summary(live_archive)
    run_frame = _run_snapshot(report_date, phase, weights, state, predictions, validation, live_validation, benchmark_ladder, market_proof, warnings, report_path, workbook_path, run_id)
    prediction_frame = _prediction_rows(report_date, phase, run_id, predictions)
    uncertainty_context = _build_uncertainty_calibration_context(phase, validation, live_validation, benchmark_ladder, state)
    candidate_frame = _candidate_rows(report_date, phase, run_id, predictions, uncertainty_context)
    bet_journal = _load_sqlite_table(SPORTSBOOK_DB_PATH, 'bet_journal')
    portfolio_recommendations, portfolio_summary, market_exposure, team_exposure, game_exposure = portfolio.build_portfolio_plan(
        candidate_frame,
        phase,
        run_id,
        report_date,
    )
    policy_snapshot = policy_research.build_policy_research_snapshot(
        live_archive,
        bet_journal,
        phase,
        run_id,
        report_date,
    )
    market_proof_summary_frame, market_proof_by_market_frame = _market_proof_frames(report_date, run_id, market_proof)
    policy_recommendations = policy_snapshot.get('recommendations') if isinstance(policy_snapshot, dict) else pd.DataFrame()
    policy_market_daily = policy_snapshot.get('realized_daily') if isinstance(policy_snapshot, dict) else pd.DataFrame()
    if not isinstance(policy_market_daily, pd.DataFrame) or len(policy_market_daily.columns) == 0:
        policy_market_daily = pd.DataFrame(columns=[
            'report_date',
            'bets',
            'stake_units',
            'result_units',
            'avg_clv',
            'roi_pct',
            'cumulative_units',
            'rolling_peak_units',
            'drawdown_units',
            'run_id',
            'as_of_report_date',
        ])
    if not portfolio_summary.empty:
        summary = portfolio_summary.iloc[0]
        run_frame['portfolio_allocated_units'] = _safe_float(summary.get('allocated_units'))
        run_frame['portfolio_utilization_pct'] = _safe_float(summary.get('utilization_pct'))
        run_frame['portfolio_correlated_candidate_count'] = _safe_int(summary.get('correlated_candidate_count'))
        run_frame['portfolio_avg_correlation_penalty'] = _safe_float(summary.get('avg_correlation_penalty'))
    canonical_policy_row = pd.Series(dtype=object)
    if isinstance(policy_recommendations, pd.DataFrame) and not policy_recommendations.empty:
        preferred = policy_recommendations[policy_recommendations['window_label'].astype(str) == 'Last 14 settled days'].head(1)
        canonical_policy_row = preferred.iloc[0] if not preferred.empty else policy_recommendations.iloc[0]
    if not canonical_policy_row.empty:
        run_frame['policy_research_policy'] = str(canonical_policy_row.get('recommended_policy') or '')
        run_frame['policy_research_posture'] = str(canonical_policy_row.get('recommended_posture') or '')
        run_frame['policy_research_window_label'] = str(canonical_policy_row.get('window_label') or '')
        run_frame['policy_research_evidence_source'] = str(canonical_policy_row.get('evidence_source') or '')
        run_frame['policy_research_evidence_family'] = str(canonical_policy_row.get('evidence_family') or '')
        run_frame['policy_research_evidence_strength'] = str(canonical_policy_row.get('evidence_strength') or '')
    with duckdb.connect(QUANT_DB_PATH) as con:
        con.execute('BEGIN TRANSACTION')
        try:
            market_policy_summary = _market_policy_summary_fields(
                policy_snapshot.get('market_type_recommendations'),
                _load_previous_market_type_rows(con, CANONICAL_POLICY_WINDOW_LABEL),
                CANONICAL_POLICY_WINDOW_LABEL,
            )
            for key, value in market_policy_summary.items():
                run_frame[key] = value
            run_values = run_frame.iloc[0].to_dict()
            target_progress, target_history_frame = _dynamic_target_snapshot(run_values)
            daily_kpi_snapshot_frame = _daily_kpi_snapshot_frame(run_values, target_progress)
            alert_history_frame = _alert_history_frame(run_values, warnings)
            attribution_market_frame, attribution_driver_frame = _archive_attribution_marts(live_archive)
            _append_table(con, 'quant_runs', run_frame)
            _append_table(con, 'daily_kpi_snapshot', daily_kpi_snapshot_frame)
            _append_table(con, 'prediction_snapshots', prediction_frame)
            _append_table(con, 'market_candidates', candidate_frame)
            _append_table(con, 'portfolio_recommendations', portfolio_recommendations)
            _append_table(con, 'portfolio_summary', portfolio_summary)
            _append_table(con, 'portfolio_market_exposure', market_exposure)
            _append_table(con, 'portfolio_team_exposure', team_exposure)
            _append_table(con, 'portfolio_game_exposure', game_exposure)
            _append_table(con, 'alert_history', alert_history_frame)
            _append_table(con, 'target_history', target_history_frame)
            _append_table(con, 'policy_research_shadow_daily', policy_snapshot.get('shadow_daily'))
            _append_table(con, 'policy_research_shadow_summary', policy_snapshot.get('shadow_summary'))
            _append_table(con, 'policy_research_market_daily', policy_market_daily)
            _append_table(con, 'policy_research_market_policy_daily', policy_snapshot.get('market_policy_daily'))
            _append_table(con, 'policy_research_market_policy_summary', policy_snapshot.get('market_policy_summary'))
            _append_table(con, 'policy_research_market_type_recommendations', policy_snapshot.get('market_type_recommendations'))
            _append_table(con, 'policy_research_window_rankings', policy_snapshot.get('window_rankings'))
            _append_table(con, 'policy_research_recommendations', policy_recommendations)
            _append_table(con, 'market_proof_summary', market_proof_summary_frame)
            _append_table(con, 'market_proof_by_market', market_proof_by_market_frame)
            _sync_reference_tables(con)
            _replace_table(con, 'attribution_market_mart', attribution_market_frame)
            _replace_table(con, 'attribution_driver_mart', attribution_driver_frame)
            _ensure_quant_views(con)
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise
    return {
        'exported': True,
        'db_path': QUANT_DB_PATH,
        'run_id': run_id,
        'prediction_rows': int(len(prediction_frame)),
        'candidate_rows': int(len(candidate_frame)),
        'portfolio_rows': int(len(portfolio_recommendations)),
    }


def query_quant(sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
    duckdb, _ = _import_duckdb()
    if duckdb is None or not os.path.exists(QUANT_DB_PATH):
        return pd.DataFrame()
    try:
        with duckdb.connect(QUANT_DB_PATH, read_only=True) as con:
            return con.execute(sql, params or []).df()
    except Exception:
        return pd.DataFrame()


def log_failed_question(
    question: str,
    parsed: Dict[str, Any] | None = None,
    answer: Dict[str, Any] | None = None,
    phase_view: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    duckdb, error = _import_duckdb()
    if duckdb is None:
        return {'logged': False, 'warning': f'DuckDB unavailable: {error}'}
    try:
        logged_at = dt.datetime.now(dt.timezone.utc).isoformat()
        row = pd.DataFrame([{
            'entry_id': uuid.uuid4().hex,
            'logged_at': logged_at,
            'question': str(question or '').strip(),
            'intent': str((parsed or {}).get('intent') or ''),
            'parsed_json': _safe_json(parsed or {}),
            'answer_title': str((answer or {}).get('title') or ''),
            'answer_summary': str((answer or {}).get('summary') or ''),
            'phase_view': str(phase_view or ''),
            'run_id': str(run_id or ''),
        }])
        with duckdb.connect(QUANT_DB_PATH) as con:
            _create_table_with_schema(con, 'failed_question_log', row)
            _append_table(con, 'failed_question_log', row)
        return {'logged': True, 'logged_at': logged_at}
    except Exception as exc:
        return {'logged': False, 'warning': str(exc)}


def load_dashboard_bundle() -> Dict[str, Any]:
    status = quant_support_status()
    bundle: Dict[str, Any] = {
        'status': status,
        'run_history': pd.DataFrame(),
        'latest_run': pd.DataFrame(),
        'predictions': pd.DataFrame(),
        'candidates': pd.DataFrame(),
        'portfolio_recommendations': pd.DataFrame(columns=portfolio.PORTFOLIO_RECOMMENDATION_COLUMNS),
        'portfolio_summary': pd.DataFrame(columns=portfolio.PORTFOLIO_SUMMARY_COLUMNS),
        'portfolio_market_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'portfolio_team_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'portfolio_game_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS),
        'clean_backtest': pd.DataFrame(),
        'live_archive': pd.DataFrame(),
        'bet_journal': pd.DataFrame(),
        'odds_snapshots': pd.DataFrame(),
        'policy_research_history': pd.DataFrame(),
        'policy_research_rankings': pd.DataFrame(),
        'policy_research_shadow_daily': pd.DataFrame(),
        'policy_research_shadow_summary': pd.DataFrame(),
        'policy_research_market_daily': pd.DataFrame(),
        'policy_research_market_policy_daily': pd.DataFrame(),
        'policy_research_market_policy_summary': pd.DataFrame(),
        'policy_research_market_type_history': pd.DataFrame(),
        'market_proof_summary': pd.DataFrame(),
        'market_proof_by_market': pd.DataFrame(),
        'market_proof_by_market_history': pd.DataFrame(),
        'daily_kpi_snapshot': pd.DataFrame(),
        'rolling_kpi_windows': pd.DataFrame(),
        'alert_history': pd.DataFrame(),
        'target_history': pd.DataFrame(),
        'attribution_market_mart': pd.DataFrame(),
        'attribution_driver_mart': pd.DataFrame(),
        'model_state': _load_json(MODEL_STATE_PATH),
    }
    if status['duckdb_available'] and status['db_exists']:
        run_history = query_quant('SELECT * FROM quant_runs_daily ORDER BY report_date DESC, created_ts DESC')
        latest_run = query_quant('SELECT * FROM latest_quant_run')
        run_id = None if latest_run.empty else str(latest_run.iloc[0].get('run_id'))
        bundle.update({
            'run_history': run_history,
            'latest_run': latest_run,
            'predictions': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM prediction_snapshots WHERE run_id = ? ORDER BY winner_prob DESC, game_pk', [run_id]),
            'candidates': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM market_candidates WHERE run_id = ? ORDER BY actionability_score DESC NULLS LAST, candidate_score DESC NULLS LAST', [run_id]),
            'portfolio_recommendations': pd.DataFrame(columns=portfolio.PORTFOLIO_RECOMMENDATION_COLUMNS) if run_id is None else query_quant('SELECT * FROM portfolio_recommendations WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, priority_score DESC NULLS LAST, allocation_rank', [run_id]),
            'portfolio_summary': pd.DataFrame(columns=portfolio.PORTFOLIO_SUMMARY_COLUMNS) if run_id is None else query_quant('SELECT * FROM portfolio_summary WHERE run_id = ?', [run_id]),
            'portfolio_market_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS) if run_id is None else query_quant('SELECT * FROM portfolio_market_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
            'portfolio_team_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS) if run_id is None else query_quant('SELECT * FROM portfolio_team_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
            'portfolio_game_exposure': pd.DataFrame(columns=portfolio.PORTFOLIO_EXPOSURE_COLUMNS) if run_id is None else query_quant('SELECT * FROM portfolio_game_exposure WHERE run_id = ? ORDER BY allocated_units DESC NULLS LAST, bucket', [run_id]),
            'clean_backtest': query_quant('SELECT * FROM clean_backtest_log_mirror'),
            'live_archive': query_quant('SELECT * FROM live_prediction_archive_mirror'),
            'bet_journal': query_quant('SELECT * FROM bet_journal_mirror'),
            'odds_snapshots': query_quant('SELECT * FROM odds_snapshots_mirror'),
            'policy_research_history': query_quant('SELECT * FROM policy_research_recommendations ORDER BY as_of_report_date DESC, window_days ASC NULLS LAST, run_id DESC'),
            'policy_research_rankings': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_window_rankings WHERE run_id = ? ORDER BY window_days ASC NULLS LAST, stability_score DESC NULLS LAST, policy_name', [run_id]),
            'policy_research_shadow_daily': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_shadow_daily WHERE run_id = ? ORDER BY report_date, policy_name', [run_id]),
            'policy_research_shadow_summary': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_shadow_summary WHERE run_id = ? ORDER BY result_units DESC NULLS LAST, policy_name', [run_id]),
            'policy_research_market_daily': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_market_daily WHERE run_id = ? ORDER BY report_date', [run_id]),
            'policy_research_market_policy_daily': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_market_policy_daily WHERE run_id = ? ORDER BY report_date, policy_name', [run_id]),
            'policy_research_market_policy_summary': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM policy_research_market_policy_summary WHERE run_id = ? ORDER BY result_units DESC NULLS LAST, policy_name', [run_id]),
            'policy_research_market_type_history': query_quant('SELECT * FROM policy_research_market_type_recommendations ORDER BY as_of_report_date DESC, window_days ASC NULLS LAST, market_type, run_id DESC'),
            'market_proof_summary': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM market_proof_summary WHERE run_id = ?', [run_id]),
            'market_proof_by_market': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM market_proof_by_market WHERE run_id = ? ORDER BY market_type', [run_id]),
            'market_proof_by_market_history': query_quant('SELECT * FROM market_proof_by_market ORDER BY as_of_report_date DESC, market_type, run_id DESC'),
            'daily_kpi_snapshot': query_quant('SELECT * FROM daily_kpi_snapshot_daily ORDER BY report_date DESC, created_ts DESC'),
            'rolling_kpi_windows': query_quant('SELECT * FROM rolling_kpi_windows ORDER BY report_date DESC, window_label'),
            'alert_history': query_quant('SELECT * FROM alert_history ORDER BY as_of_report_date DESC, created_ts DESC'),
            'target_history': query_quant('SELECT * FROM target_history ORDER BY as_of_report_date DESC, metric_key, run_id DESC'),
            'attribution_market_mart': query_quant('SELECT * FROM attribution_market_mart'),
            'attribution_driver_mart': query_quant('SELECT * FROM attribution_driver_mart'),
        })
        if not latest_run.empty:
            state_json = latest_run.iloc[0].get('state_json')
            try:
                bundle['model_state'] = json.loads(state_json) if isinstance(state_json, str) and state_json.strip() else bundle['model_state']
            except Exception:
                pass
        raw_bet_journal = _load_sqlite_table(SPORTSBOOK_DB_PATH, 'bet_journal')
        raw_odds_snapshots = _load_sqlite_table(SPORTSBOOK_DB_PATH, 'odds_snapshots')
        if not raw_bet_journal.empty:
            bundle['bet_journal'] = raw_bet_journal
        if not raw_odds_snapshots.empty:
            bundle['odds_snapshots'] = raw_odds_snapshots
        return bundle

    live_archive = _load_csv(LIVE_PREDICTION_ARCHIVE_PATH)
    if not live_archive.empty and 'report_date' in live_archive.columns:
        latest_date = str(sorted(live_archive['report_date'].astype(str).dropna().unique().tolist())[-1])
        latest_predictions = live_archive[live_archive['report_date'].astype(str) == latest_date].copy()
    else:
        latest_predictions = pd.DataFrame()
    bundle.update({
        'predictions': latest_predictions,
        'clean_backtest': _load_csv(CLEAN_BACKTEST_LOG_PATH),
        'live_archive': live_archive,
        'bet_journal': _load_sqlite_table(SPORTSBOOK_DB_PATH, 'bet_journal'),
        'odds_snapshots': _load_sqlite_table(SPORTSBOOK_DB_PATH, 'odds_snapshots'),
    })
    return bundle


















