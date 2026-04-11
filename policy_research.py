from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import numpy as np
import pandas as pd

import portfolio_engine as portfolio

DEFAULT_POLICY_WINDOWS = [
    ('Last 7 settled days', 7),
    ('Last 14 settled days', 14),
    ('Last 30 settled days', 30),
    ('All settled history', None),
]

CANONICAL_BANKROLL = 5000.0
CANONICAL_DEPLOYMENT_SCALAR = 1.0
CANONICAL_POSTURE = 'Base'
REAL_MARKET_POLICY_MEDIUM_SLATES = 3
REAL_MARKET_POLICY_MEDIUM_PICKS = 10
REAL_MARKET_POLICY_HIGH_SLATES = 5
REAL_MARKET_POLICY_HIGH_PICKS = 20
REAL_MARKET_POLICY_THRESHOLDS = {
    'aggregate': {
        'medium_slates': REAL_MARKET_POLICY_MEDIUM_SLATES,
        'medium_picks': REAL_MARKET_POLICY_MEDIUM_PICKS,
        'high_slates': REAL_MARKET_POLICY_HIGH_SLATES,
        'high_picks': REAL_MARKET_POLICY_HIGH_PICKS,
    },
    'moneyline': {
        'medium_slates': 3,
        'medium_picks': 10,
        'high_slates': 5,
        'high_picks': 20,
    },
    'total': {
        'medium_slates': 4,
        'medium_picks': 12,
        'high_slates': 6,
        'high_picks': 24,
    },
    'run_line': {
        'medium_slates': 5,
        'medium_picks': 14,
        'high_slates': 7,
        'high_picks': 28,
    },
}


def market_policy_thresholds(market_type: str | None = None) -> Dict[str, int]:
    key = str(market_type or 'aggregate').lower().strip()
    if key not in REAL_MARKET_POLICY_THRESHOLDS:
        key = 'aggregate'
    return dict(REAL_MARKET_POLICY_THRESHOLDS[key])


def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def clip_value(value: Any, low: float, high: float) -> float:
    numeric = safe_float(value)
    if numeric is None:
        numeric = low
    return max(low, min(high, float(numeric)))


def policy_specs() -> Dict[str, Dict[str, Any]]:
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


def risk_posture_specs() -> Dict[str, Dict[str, Any]]:
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


def default_unit_risk_pct_for_phase(phase: str) -> float:
    config = portfolio.portfolio_config_for_phase(phase)
    return float(config.get('unit_risk_pct', 0.01)) * 100.0


def infer_phase_from_report_date(report_date: Any, explicit_phase: Any = None) -> str:
    phase_text = str(explicit_phase or '').strip().lower()
    if phase_text in {'spring', 'regular'}:
        return phase_text
    parsed = pd.to_datetime(report_date, errors='coerce')
    if pd.isna(parsed):
        return 'regular'
    target_date = parsed.date()
    if target_date.month < 3:
        return 'spring'
    if target_date.month == 3 and target_date.day < 25:
        return 'spring'
    return 'regular'


def market_expected_value_per_unit(model_probability: Any, american_price: Any, edge_value: Any = None, market_type: str = 'moneyline') -> float:
    probability = safe_float(model_probability)
    price = safe_float(american_price)
    if probability is not None and price is not None and abs(price) > 1e-9:
        probability = max(min(probability, 1.0 - 1e-6), 1e-6)
        if price > 0:
            win_profit = price / 100.0
        else:
            win_profit = 100.0 / abs(price)
        return max(min(float((probability * win_profit) - ((1.0 - probability) * 1.0)), 0.25), 0.0)
    edge = abs(safe_float(edge_value) or 0.0)
    market_name = str(market_type or '').lower()
    if market_name == 'total':
        return min(edge * 0.10, 0.20)
    if market_name == 'run_line':
        return min(edge * 0.85, 0.20)
    return min(edge * 0.95, 0.20)


def portfolio_team_keys(row: pd.Series) -> list[str]:
    market_type = str(row.get('market_type') or '').lower()
    selected_team = str(row.get('selected_team') or '').strip().upper()
    away_team = str(row.get('away_team') or '').strip().upper()
    home_team = str(row.get('home_team') or '').strip().upper()
    if market_type == 'total':
        return [team for team in [away_team, home_team] if team]
    if selected_team:
        return [selected_team]
    return [team for team in [away_team, home_team] if team][:1]


def portfolio_game_bucket(row: pd.Series) -> str:
    game_pk = row.get('game_pk')
    if pd.notna(game_pk):
        try:
            return f"gamepk:{int(float(game_pk))}"
        except Exception:
            pass
    return f"{str(row.get('away_team') or '').upper()}@{str(row.get('home_team') or '').upper()}"


def apply_policy_to_recommendations(summary_row: Dict[str, Any], recommendations: pd.DataFrame, policy_name: str, posture_name: str, bankroll: float, unit_risk_pct: float, deployment_scalar: float):
    policy = policy_specs().get(policy_name, policy_specs()['Flat'])
    posture = risk_posture_specs().get(posture_name, risk_posture_specs()['Base'])
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
            'daily_budget_units': ((safe_float(summary_row.get('daily_budget_units')) if summary_row else 0.0) or 0.0) * posture['deployment_mult'] * deployment_scalar,
            'budget_scale': 1.0,
            'policy_description': policy['technical'],
            'plain_english': f"{policy['plain_english']} {posture['plain_english']}",
            'avg_correlation_penalty': safe_float(summary_row.get('avg_correlation_penalty')) if summary_row else None,
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rec = recommendation_frame.copy()
    for col in ['allocated_units', 'expected_value_per_unit', 'edge_value', 'actionability_score', 'final_correlation_penalty', 'same_game_peer_count', 'same_team_peer_count']:
        rec[col] = pd.to_numeric(rec.get(col), errors='coerce').fillna(0.0)
    rec['game_bucket'] = rec.apply(portfolio_game_bucket, axis=1)
    rec['team_keys'] = rec.apply(portfolio_team_keys, axis=1)

    def _policy_factor(row: pd.Series):
        ev = clip_value(row.get('expected_value_per_unit'), 0.0, 0.20)
        edge = clip_value(abs(safe_float(row.get('edge_value')) or 0.0), 0.0, 0.15)
        action = clip_value((safe_float(row.get('actionability_score')) or 0.0) / 100.0, 0.40, 1.00)
        corr = clip_value(row.get('final_correlation_penalty'), 0.45, 1.00)
        same_game = int(safe_float(row.get('same_game_peer_count')) or 0)
        same_team = int(safe_float(row.get('same_team_peer_count')) or 0)
        if policy_name == 'Flat':
            return 1.0, 'Keeps the allocator size as-is.'
        if policy_name == 'Fractional Kelly Proxy':
            # True fractional Kelly (25% Kelly) sizing.
            # Full Kelly fraction f* = EV / b, where b is win_profit per unit risked.
            # For typical MLB near-even-money lines b ≈ 0.90.
            # EV = b*p - (1-p), so full_kelly = EV / b.
            # We use 25% Kelly for conservatism, then scale to a relative factor
            # anchored so that a solid EV of 0.05 maps to factor ≈ 1.0.
            b_estimate = max(0.80, 1.0 - ev * 0.5)  # slight downward adjust for favorite-heavy slates
            full_kelly = ev / b_estimate if b_estimate > 1e-6 else 0.0
            quarter_kelly = full_kelly * 0.25
            # Normalize: quarter_kelly of 0.0125 → factor 1.0 (represents ~EV 0.04-0.05 range)
            base_factor = 0.45 + (quarter_kelly / 0.0125) * 0.55
            # Apply correlation and actionability scaling
            factor = clip_value(base_factor * corr * (0.80 + (0.20 * action)), 0.45, 1.35)
            return factor, f'25% Kelly: full_kelly={full_kelly:.3f}, factor={factor:.3f}'
        # Risk-Capped Hybrid: true Kelly base with hard concentration drag
        b_estimate = max(0.80, 1.0 - ev * 0.5)
        full_kelly = ev / b_estimate if b_estimate > 1e-6 else 0.0
        quarter_kelly = full_kelly * 0.25
        base_factor = 0.50 + (quarter_kelly / 0.0125) * 0.50
        concentration_drag = clip_value(1.0 - (0.06 * same_game) - (0.05 * same_team), 0.70, 1.00)
        factor = clip_value(base_factor * concentration_drag * corr * (0.85 + (0.15 * action)), 0.50, 1.10)
        return factor, f'Kelly-hybrid: full_kelly={full_kelly:.3f}, conc_drag={concentration_drag:.2f}, factor={factor:.3f}'

    policy_details = rec.apply(_policy_factor, axis=1, result_type='expand')
    rec['policy_factor'] = pd.to_numeric(policy_details[0], errors='coerce').fillna(1.0)
    rec['policy_note'] = policy_details[1].astype(str)

    base_units = rec['allocated_units'] * posture['deployment_mult'] * deployment_scalar
    rec['policy_units_raw'] = base_units * rec['policy_factor']
    budget_units = (((safe_float(summary_row.get('daily_budget_units')) if summary_row else None) or float(rec['allocated_units'].sum()) or 0.0) * posture['deployment_mult'] * deployment_scalar)
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
    }
    return rec, summary, market_exposure, team_exposure, game_exposure

def confidence_tier_from_probability(probability: Any) -> str:
    numeric = safe_float(probability)
    if numeric is None:
        return 'LOW'
    confidence = max(min(max(numeric, 1.0 - numeric), 1.0), 0.5)
    if confidence >= 0.67:
        return 'HIGH'
    if confidence >= 0.60:
        return 'MEDIUM'
    return 'LOW'


def shadow_actionability_label(confidence: float) -> str:
    if confidence >= 0.67:
        return 'MUST TAKE'
    if confidence >= 0.60:
        return 'BET'
    if confidence >= 0.55:
        return 'WATCH'
    return 'PASS'


def build_shadow_candidate_frame(live_archive: pd.DataFrame) -> pd.DataFrame:
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
    work['confidence_tier'] = work['p_home'].map(confidence_tier_from_probability)
    work['actionability_label'] = work['model_confidence'].map(shadow_actionability_label)
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


def build_shadow_policy_backtest(live_archive: pd.DataFrame, bankroll: float, unit_risk_pct: float, deployment_scalar: float, posture_name: str):
    shadow_candidates = build_shadow_candidate_frame(live_archive)
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
        for policy_name in policy_specs().keys():
            policy_view, policy_summary, _, _, _ = apply_policy_to_recommendations(
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
            unit_size_dollars = float(safe_float(policy_summary.get('unit_size_dollars')) or 0.0)
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
                'avg_correlation_penalty': safe_float(policy_summary.get('avg_correlation_penalty')),
                'technical_note': policy_summary.get('policy_description') or '',
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
            'technical_note': policy_specs().get(policy_name, {}).get('technical', ''),
            'plain_english': takeaway,
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values('result_units', ascending=False).reset_index(drop=True)
    return daily_frame, summary_frame

def build_market_policy_candidate_frame(bet_journal: pd.DataFrame) -> pd.DataFrame:
    required_columns = {'report_date', 'away_team', 'home_team', 'market_type', 'selection', 'result_units'}
    if bet_journal.empty or not required_columns.issubset(bet_journal.columns):
        return pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').replace(0.0, np.nan).fillna(1.0)
    if 'book_price' in work.columns:
        work['book_price'] = pd.to_numeric(work.get('book_price'), errors='coerce')
    if 'model_price' in work.columns:
        work['model_price'] = pd.to_numeric(work.get('model_price'), errors='coerce')
    if 'model_probability' in work.columns:
        work['model_probability'] = pd.to_numeric(work.get('model_probability'), errors='coerce')
    if 'market_probability' in work.columns:
        work['market_probability'] = pd.to_numeric(work.get('market_probability'), errors='coerce')
    if 'edge_value' in work.columns:
        work['edge_value'] = pd.to_numeric(work.get('edge_value'), errors='coerce').fillna(0.0)
    if 'actionability_score' in work.columns:
        work['actionability_score'] = pd.to_numeric(work.get('actionability_score'), errors='coerce')
    if 'line_value' in work.columns:
        work['line_value'] = pd.to_numeric(work.get('line_value'), errors='coerce')
    if 'game_pk' in work.columns:
        work['game_pk'] = pd.to_numeric(work.get('game_pk'), errors='coerce')
    if 'clv_value' in work.columns:
        work['clv_value'] = pd.to_numeric(work.get('clv_value'), errors='coerce')
    work = work.dropna(subset=['report_date', 'result_units']).copy()
    if work.empty:
        return pd.DataFrame()
    if 'is_settled' in work.columns:
        settled_mask = work['is_settled'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        work = work[settled_mask | work['result_units'].notna()].copy()
    elif 'status' in work.columns:
        status_mask = work['status'].astype(str).str.lower().isin(['graded', 'settled'])
        work = work[status_mask | work['result_units'].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    explicit_phase = work.get('phase') if 'phase' in work.columns else None
    work['phase'] = [
        infer_phase_from_report_date(report_date, explicit_phase.iloc[idx] if explicit_phase is not None else None)
        for idx, report_date in enumerate(work['report_date'])
    ]
    work['away_team'] = work['away_team'].astype(str).str.upper()
    work['home_team'] = work['home_team'].astype(str).str.upper()
    work['market_type'] = work['market_type'].astype(str).str.lower()
    work['selection'] = work['selection'].astype(str).str.upper()
    work['confidence_tier'] = work.get('confidence_tier', 'LOW').fillna('LOW').astype(str).str.upper()
    work['actionability_label'] = work.get('actionability_label', 'WATCH').fillna('WATCH').astype(str).str.upper()
    score_defaults = np.select(
        [
            work['actionability_label'] == 'WATCH',
            work['actionability_label'] == 'BET',
            work['actionability_label'] == 'MUST TAKE',
        ],
        [55.0, 70.0, 82.0],
        default=45.0,
    )
    work['actionability_score'] = work.get('actionability_score', pd.Series(dtype=float))
    work['actionability_score'] = pd.to_numeric(work['actionability_score'], errors='coerce').fillna(pd.Series(score_defaults, index=work.index))
    base_units = work['actionability_label'].map({'WATCH': 0.25, 'BET': 0.50, 'MUST TAKE': 0.75}).fillna(0.20)
    score_factor = (0.60 + work['actionability_score'].clip(lower=0.0, upper=100.0) / 200.0).clip(lower=0.60, upper=1.10)
    confidence_factor = work['confidence_tier'].map({'LOW': 0.85, 'MEDIUM': 0.95, 'HIGH': 1.05}).fillna(0.90)
    work['suggested_units_preview'] = (base_units * score_factor * confidence_factor).clip(lower=0.0, upper=1.25).round(2)
    work['expected_value_per_unit'] = [
        market_expected_value_per_unit(row.get('model_probability'), row.get('book_price'), row.get('edge_value'), str(row.get('market_type') or 'moneyline'))
        for _, row in work.iterrows()
    ]
    work['candidate_score'] = (
        pd.Series(work['expected_value_per_unit'], index=work.index).fillna(0.0)
        * (1.0 + work['actionability_score'].fillna(0.0) / 100.0)
        * (1.0 + work['edge_value'].fillna(0.0).abs().clip(lower=0.0, upper=0.25))
    )
    work['market_label'] = work.get('sportsbook', 'Journal replay').fillna('Journal replay').astype(str)
    work['sportsbook_count'] = np.where(work['market_label'].astype(str).str.strip() != '', 1, 0)
    work['realized_result_per_unit'] = np.where(work['stake_units'].abs() > 1e-9, work['result_units'] / work['stake_units'], 0.0)
    work['actionability_notes'] = work.get('actionability_notes', '').fillna('').astype(str)
    work['selected_team'] = ''
    columns = [
        'report_date', 'phase', 'game_pk', 'away_team', 'home_team', 'market_label', 'sportsbook_count',
        'confidence_tier', 'market_type', 'selection', 'line_value', 'book_price', 'model_price',
        'edge_value', 'actionability_label', 'actionability_score', 'actionability_notes',
        'expected_value_per_unit', 'suggested_units_preview', 'candidate_score', 'selected_team',
        'realized_result_per_unit', 'clv_value',
    ]
    for col in columns:
        if col not in work.columns:
            work[col] = np.nan
    return work[columns].reset_index(drop=True)


def build_market_policy_backtest(bet_journal: pd.DataFrame, bankroll: float, unit_risk_pct: float, deployment_scalar: float, posture_name: str):
    market_candidates = build_market_policy_candidate_frame(bet_journal)
    if market_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_rows = []
    for report_date, day_candidates in market_candidates.groupby('report_date', dropna=True):
        phase = str(day_candidates['phase'].mode().iloc[0] if 'phase' in day_candidates.columns and not day_candidates['phase'].mode().empty else 'regular').lower()
        recs, summary_frame, _, _, _ = portfolio.build_portfolio_plan(
            day_candidates,
            phase,
            f"market_{report_date.isoformat()}",
            report_date,
        )
        summary_row = {} if summary_frame.empty else summary_frame.iloc[0].to_dict()
        if recs.empty:
            continue
        result_lookup = day_candidates[['away_team', 'home_team', 'market_type', 'selection', 'realized_result_per_unit', 'clv_value']].drop_duplicates()
        recs = recs.merge(result_lookup, on=['away_team', 'home_team', 'market_type', 'selection'], how='left')
        for policy_name in policy_specs().keys():
            policy_view, policy_summary, _, _, _ = apply_policy_to_recommendations(summary_row, recs, policy_name, posture_name, bankroll, unit_risk_pct, deployment_scalar)
            if policy_view.empty:
                continue
            realized_series = pd.to_numeric(policy_view['realized_result_per_unit'], errors='coerce').fillna(0.0)
            result_units = float((policy_view['policy_units'] * realized_series).sum())
            risk_units = float(policy_view['policy_units'].sum())
            unit_size_dollars = float(safe_float(policy_summary.get('unit_size_dollars')) or 0.0)
            picks = int(len(policy_view))
            hits = int((realized_series > 0).sum())
            clv_series = pd.to_numeric(policy_view.get('clv_value'), errors='coerce') if 'clv_value' in policy_view.columns else pd.Series(dtype=float)
            avg_clv = float(clv_series.dropna().mean()) if not clv_series.dropna().empty else None
            daily_rows.append({
                'report_date': report_date,
                'phase': phase,
                'policy_name': policy_name,
                'posture_name': posture_name,
                'games': int(len(day_candidates[['away_team', 'home_team']].drop_duplicates())),
                'picks': picks,
                'hits': hits,
                'hit_rate': (hits / picks) if picks else None,
                'risk_units': risk_units,
                'result_units': result_units,
                'yield_pct': (result_units / risk_units) if risk_units > 1e-9 else None,
                'unit_size_dollars': unit_size_dollars,
                'risk_dollars': risk_units * unit_size_dollars,
                'result_dollars': result_units * unit_size_dollars,
                'avg_clv': avg_clv,
                'avg_correlation_penalty': safe_float(policy_summary.get('avg_correlation_penalty')),
                'technical_note': policy_summary.get('policy_description') or '',
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
        avg_clv = float(grp['avg_clv'].dropna().mean()) if 'avg_clv' in grp.columns and not grp['avg_clv'].dropna().empty else None
        if max_drawdown_units <= 1.0 and total_result_units >= 0.0:
            takeaway = 'This was the cleanest real-market path, with smaller drawdowns on actual priced bets.'
        elif total_result_units > 0.0 and max_drawdown_units > 1.0:
            takeaway = 'This made more on actual priced bets, but the ride was bumpier.'
        else:
            takeaway = 'This policy has not earned trust yet on the settled priced-bet sample.'
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
            'avg_clv': avg_clv,
            'avg_correlation_penalty': float(grp['avg_correlation_penalty'].dropna().mean()) if 'avg_correlation_penalty' in grp.columns and not grp['avg_correlation_penalty'].dropna().empty else None,
            'technical_note': policy_specs().get(policy_name, {}).get('technical', ''),
            'plain_english': takeaway,
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values(['result_units', 'yield_pct'], ascending=[False, False]).reset_index(drop=True)
    return daily_frame, summary_frame


def build_market_policy_backtest_for_market_type(bet_journal: pd.DataFrame, market_type: str, bankroll: float, unit_risk_pct: float, deployment_scalar: float, posture_name: str):
    market_type_key = str(market_type or '').lower()
    market_candidates = build_market_policy_candidate_frame(bet_journal)
    if market_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    market_candidates = market_candidates[market_candidates['market_type'].astype(str).str.lower() == market_type_key].copy()
    if market_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_rows = []
    for report_date, day_candidates in market_candidates.groupby('report_date', dropna=True):
        phase = str(day_candidates['phase'].mode().iloc[0] if 'phase' in day_candidates.columns and not day_candidates['phase'].mode().empty else 'regular').lower()
        recs, summary_frame, _, _, _ = portfolio.build_portfolio_plan(
            day_candidates,
            phase,
            f"market_{market_type_key}_{report_date.isoformat()}",
            report_date,
        )
        summary_row = {} if summary_frame.empty else summary_frame.iloc[0].to_dict()
        if recs.empty:
            continue
        result_lookup = day_candidates[['away_team', 'home_team', 'market_type', 'selection', 'realized_result_per_unit', 'clv_value']].drop_duplicates()
        recs = recs.merge(result_lookup, on=['away_team', 'home_team', 'market_type', 'selection'], how='left')
        for policy_name in policy_specs().keys():
            policy_view, policy_summary, _, _, _ = apply_policy_to_recommendations(summary_row, recs, policy_name, posture_name, bankroll, unit_risk_pct, deployment_scalar)
            if policy_view.empty:
                continue
            realized_series = pd.to_numeric(policy_view['realized_result_per_unit'], errors='coerce').fillna(0.0)
            result_units = float((policy_view['policy_units'] * realized_series).sum())
            risk_units = float(policy_view['policy_units'].sum())
            unit_size_dollars = float(safe_float(policy_summary.get('unit_size_dollars')) or 0.0)
            picks = int(len(policy_view))
            hits = int((realized_series > 0).sum())
            clv_series = pd.to_numeric(policy_view.get('clv_value'), errors='coerce') if 'clv_value' in policy_view.columns else pd.Series(dtype=float)
            avg_clv = float(clv_series.dropna().mean()) if not clv_series.dropna().empty else None
            daily_rows.append({
                'report_date': report_date,
                'phase': phase,
                'market_type': market_type_key,
                'policy_name': policy_name,
                'posture_name': posture_name,
                'games': int(len(day_candidates[['away_team', 'home_team']].drop_duplicates())),
                'picks': picks,
                'hits': hits,
                'hit_rate': (hits / picks) if picks else None,
                'risk_units': risk_units,
                'result_units': result_units,
                'yield_pct': (result_units / risk_units) if risk_units > 1e-9 else None,
                'unit_size_dollars': unit_size_dollars,
                'risk_dollars': risk_units * unit_size_dollars,
                'result_dollars': result_units * unit_size_dollars,
                'avg_clv': avg_clv,
                'avg_correlation_penalty': safe_float(policy_summary.get('avg_correlation_penalty')),
                'technical_note': policy_summary.get('policy_description') or '',
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
        avg_clv = float(grp['avg_clv'].dropna().mean()) if 'avg_clv' in grp.columns and not grp['avg_clv'].dropna().empty else None
        if max_drawdown_units <= 1.0 and total_result_units >= 0.0:
            takeaway = f"{market_type_key.replace('_', ' ').title()} has the cleanest priced-bet path under this policy so far."
        elif total_result_units > 0.0 and max_drawdown_units > 1.0:
            takeaway = f"{market_type_key.replace('_', ' ').title()} has upside here, but the path has been bumpier."
        else:
            takeaway = f"{market_type_key.replace('_', ' ').title()} has not earned trust yet on the settled priced-bet sample."
        summary_rows.append({
            'market_type': market_type_key,
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
            'avg_clv': avg_clv,
            'avg_correlation_penalty': float(grp['avg_correlation_penalty'].dropna().mean()) if 'avg_correlation_penalty' in grp.columns and not grp['avg_correlation_penalty'].dropna().empty else None,
            'technical_note': policy_specs().get(policy_name, {}).get('technical', ''),
            'plain_english': takeaway,
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values(['result_units', 'yield_pct'], ascending=[False, False]).reset_index(drop=True)
    return daily_frame, summary_frame


def market_specific_policy_recommendations(bet_journal: pd.DataFrame, shadow_window_moneyline: pd.DataFrame, window_label: str, window_days: int | None, phase: str) -> pd.DataFrame:
    label_map = {
        'moneyline': 'Moneyline',
        'run_line': 'Run line',
        'total': 'Total',
    }
    bankroll = CANONICAL_BANKROLL
    posture_name = CANONICAL_POSTURE
    deployment_scalar = CANONICAL_DEPLOYMENT_SCALAR
    unit_risk_pct = default_unit_risk_pct_for_phase(phase)
    rows = []
    for market_type in ['moneyline', 'run_line', 'total']:
        market_key = str(market_type)
        thresholds = market_policy_thresholds(market_key)
        market_journal = pd.DataFrame()
        if bet_journal is not None and not bet_journal.empty and 'market_type' in bet_journal.columns:
            market_journal = bet_journal[bet_journal['market_type'].astype(str).str.lower() == market_key].copy()
        market_daily, _ = build_market_policy_backtest_for_market_type(
            market_journal,
            market_key,
            bankroll,
            unit_risk_pct,
            deployment_scalar,
            posture_name,
        )
        market_window = summarize_policy_window(market_daily, window_days) if not market_daily.empty else pd.DataFrame()
        realized_daily = build_market_realized_daily(market_journal) if not market_journal.empty else pd.DataFrame()
        shadow_window = shadow_window_moneyline if market_key == 'moneyline' else pd.DataFrame()
        takeover_probe = market_policy_takeover_status(market_window, None, market_key)
        market_ready = bool(takeover_probe.get('primary_ready'))
        usable_market_window = market_window if market_ready else pd.DataFrame()
        if not usable_market_window.empty or not shadow_window.empty:
            recommendation = recommend_policy_research(shadow_window, realized_daily, window_label, window_days, usable_market_window, market_key)
        else:
            recommendation = {
                'policy_name': 'No call yet',
                'posture_name': 'Conservative',
                'evidence_source': 'Below market-specific takeover threshold',
                'evidence_strength': 'Low',
                'used_real_market': False,
                'technical_note': (
                    f"{label_map.get(market_key, market_key.title())} has not yet cleared its own real-market takeover threshold "
                    f"of {thresholds['medium_slates']} slates / {thresholds['medium_picks']} picks, so a market-specific bankroll recommendation would still be guesswork."
                ),
                'plain_english': (
                    f"There is not enough settled priced-bet evidence yet for {label_map.get(market_key, market_key.title())}. "
                    f"This market needs its own threshold of {thresholds['medium_slates']} settled day(s) and {thresholds['medium_picks']} priced bet(s) before real-market evidence can take over."
                ),
            }
        takeover = market_policy_takeover_status(market_window, recommendation, market_key)
        rows.append({
            'market_type': market_key,
            'market_label': label_map.get(market_key, market_key.title()),
            'recommended_policy': recommendation.get('policy_name'),
            'recommended_posture': recommendation.get('posture_name'),
            'evidence_source': recommendation.get('evidence_source'),
            'evidence_strength': recommendation.get('evidence_strength'),
            'used_real_market': recommendation.get('used_real_market'),
            'current_slates': takeover.get('current_slates'),
            'current_picks': takeover.get('current_picks'),
            'remaining_medium_slates': takeover.get('remaining_medium_slates'),
            'remaining_medium_picks': takeover.get('remaining_medium_picks'),
            'remaining_high_slates': takeover.get('remaining_high_slates'),
            'remaining_high_picks': takeover.get('remaining_high_picks'),
            'primary_ready': takeover.get('primary_ready'),
            'high_ready': takeover.get('high_ready'),
            'medium_threshold_label': takeover.get('medium_threshold_label'),
            'high_threshold_label': takeover.get('high_threshold_label'),
            'avg_clv': takeover.get('avg_clv'),
            'technical_note': recommendation.get('technical_note'),
            'plain_english': recommendation.get('plain_english'),
        })
    return pd.DataFrame(rows)


def build_market_realized_daily(bet_journal: pd.DataFrame) -> pd.DataFrame:
    if bet_journal.empty or 'report_date' not in bet_journal.columns or 'result_units' not in bet_journal.columns:
        return pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    clv_column = 'clv_value' if 'clv_value' in work.columns else ('clv' if 'clv' in work.columns else None)
    if clv_column is not None:
        work['clv'] = pd.to_numeric(work.get(clv_column), errors='coerce')
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


def build_market_realized_summary(bet_journal: pd.DataFrame):
    if bet_journal.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = {'report_date', 'result_units'}
    if not required_columns.issubset(bet_journal.columns):
        return pd.DataFrame(), pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
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
    daily['roi_pct'] = np.where(daily['stake_units'].abs() > 1e-9, daily['result_units'] / daily['stake_units'], np.nan)
    summary_rows = [{
        'settled_bets': int(len(work)),
        'stake_units': float(work['stake_units'].sum()),
        'result_units': float(work['result_units'].sum()),
        'roi_pct': float(work['result_units'].sum() / work['stake_units'].sum()) if abs(float(work['stake_units'].sum())) > 1e-9 else None,
        'avg_clv': float(pd.to_numeric(work.get('clv'), errors='coerce').dropna().mean()) if 'clv' in work.columns else None,
        'max_drawdown_units': abs(float((daily['result_units'].cumsum() - daily['result_units'].cumsum().cummax()).min() or 0.0)),
    }]
    by_market = pd.DataFrame()
    if 'market_type' in work.columns:
        by_market = work.groupby('market_type', dropna=False).agg(
            bets=('result_units', 'size'),
            stake_units=('stake_units', 'sum'),
            result_units=('result_units', 'sum'),
        ).reset_index()
        by_market['roi_pct'] = np.where(by_market['stake_units'].abs() > 1e-9, by_market['result_units'] / by_market['stake_units'], np.nan)
    return pd.DataFrame(summary_rows), by_market


def filter_to_window(frame: pd.DataFrame, date_col: str, window_days: int | None) -> pd.DataFrame:
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


def summarize_policy_window(policy_daily: pd.DataFrame, window_days: int | None) -> pd.DataFrame:
    window = filter_to_window(policy_daily, 'report_date', window_days)
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
            'avg_clv': float(grp['avg_clv'].dropna().mean()) if 'avg_clv' in grp.columns and not grp['avg_clv'].dropna().empty else None,
            'avg_correlation_penalty': float(grp['avg_correlation_penalty'].dropna().mean()) if 'avg_correlation_penalty' in grp.columns and not grp['avg_correlation_penalty'].dropna().empty else None,
            'stability_score': stability_score,
            'sample_ok': bool((len(grp) >= 3) and (int(grp['picks'].sum()) >= 10 if 'picks' in grp.columns else len(grp) >= 10)),
        })
    return pd.DataFrame(rows).sort_values(['stability_score', 'result_units', 'yield_pct'], ascending=[False, False, False]).reset_index(drop=True)


def summarize_shadow_window(shadow_daily: pd.DataFrame, window_days: int | None) -> pd.DataFrame:
    return summarize_policy_window(shadow_daily, window_days)


def market_policy_takeover_status(market_window: pd.DataFrame, recommendation: Dict[str, Any] | None = None, market_type: str | None = None) -> Dict[str, Any]:
    recommendation = recommendation or {}
    thresholds = market_policy_thresholds(market_type)
    medium_slates = int(thresholds['medium_slates'])
    medium_picks = int(thresholds['medium_picks'])
    high_slates = int(thresholds['high_slates'])
    high_picks = int(thresholds['high_picks'])
    if market_window is None or market_window.empty:
        current_slates = 0
        current_picks = 0
        avg_clv = None
    else:
        current_slates = int(pd.to_numeric(market_window.get('settled_slates'), errors='coerce').fillna(0).max() or 0)
        current_picks = int(pd.to_numeric(market_window.get('settled_picks'), errors='coerce').fillna(0).max() or 0)
        avg_clv_series = pd.to_numeric(market_window.get('avg_clv'), errors='coerce')
        avg_clv = float(avg_clv_series.dropna().mean()) if not avg_clv_series.dropna().empty else None

    primary_ready = bool(current_slates >= medium_slates and current_picks >= medium_picks)
    high_ready = bool(current_slates >= high_slates and current_picks >= high_picks)
    remaining_medium_slates = max(0, medium_slates - current_slates)
    remaining_medium_picks = max(0, medium_picks - current_picks)
    remaining_high_slates = max(0, high_slates - current_slates)
    remaining_high_picks = max(0, high_picks - current_picks)
    used_real_market = bool(recommendation.get('used_real_market'))
    evidence_strength = str(recommendation.get('evidence_strength') or 'Low')
    label_map = {
        'moneyline': 'Moneyline',
        'run_line': 'Run line',
        'total': 'Total',
    }
    market_label = label_map.get(str(market_type or '').lower(), 'Overall market')

    if used_real_market and primary_ready:
        technical_note = (
            f"{market_label} real-market policy replay is active. The selected window has {current_slates} settled slate(s) and "
            f"{current_picks} priced bet(s), which clears the primary takeover threshold of "
            f"{medium_slates} slates / {medium_picks} picks."
        )
        plain_english = f"Priced-bet evidence is now driving the {market_label.lower()} policy recommendation instead of shadow replay."
    else:
        technical_note = (
            f"{market_label} real-market takeover is waiting on {remaining_medium_slates} more settled slate(s) and "
            f"{remaining_medium_picks} more priced bet(s) to reach its primary threshold of "
            f"{medium_slates} slates / {medium_picks} picks."
        )
        plain_english = (
            f"{market_label} still needs {remaining_medium_slates} more settled day(s) and {remaining_medium_picks} more settled priced bet(s) "
            f"before its own real-market policy evidence can take over."
        )
    if high_ready:
        technical_note += ' High-confidence real-market policy evidence is already unlocked.'
    else:
        technical_note += (
            f" High-confidence research would need {remaining_high_slates} more slate(s) and "
            f"{remaining_high_picks} more priced bet(s) to reach {high_slates} / {high_picks}."
        )
    if avg_clv is not None:
        technical_note += f" Window average CLV is {avg_clv:+.3f}."

    return {
        'used_real_market': used_real_market,
        'sample_ready': primary_ready,
        'primary_ready': primary_ready,
        'high_ready': high_ready,
        'evidence_strength': evidence_strength,
        'current_slates': current_slates,
        'current_picks': current_picks,
        'remaining_medium_slates': remaining_medium_slates,
        'remaining_medium_picks': remaining_medium_picks,
        'remaining_high_slates': remaining_high_slates,
        'remaining_high_picks': remaining_high_picks,
        'avg_clv': avg_clv,
        'technical_note': technical_note,
        'plain_english': plain_english,
        'medium_threshold_label': f"{medium_slates} slates / {medium_picks} picks",
        'high_threshold_label': f"{high_slates} slates / {high_picks} picks",
    }


def market_type_takeover_status(bet_journal: pd.DataFrame, window_days: int | None = None) -> pd.DataFrame:
    candidates = build_market_policy_candidate_frame(bet_journal)
    if candidates.empty:
        candidates = pd.DataFrame(columns=['report_date', 'market_type', 'clv_value'])
    candidates = filter_to_window(candidates, 'report_date', window_days)
    label_map = {
        'moneyline': 'Moneyline',
        'run_line': 'Run line',
        'total': 'Total',
    }
    rows = []
    for market_type in ['moneyline', 'run_line', 'total']:
        thresholds = market_policy_thresholds(market_type)
        medium_slates = int(thresholds['medium_slates'])
        medium_picks = int(thresholds['medium_picks'])
        high_slates = int(thresholds['high_slates'])
        high_picks = int(thresholds['high_picks'])
        if candidates.empty:
            grp = pd.DataFrame(columns=candidates.columns)
        else:
            grp = candidates[candidates.get('market_type', pd.Series(dtype=str)).astype(str) == market_type].copy()
        current_slates = int(grp['report_date'].nunique()) if not grp.empty and 'report_date' in grp.columns else 0
        current_picks = int(len(grp))
        avg_clv_series = pd.to_numeric(grp.get('clv_value'), errors='coerce') if not grp.empty else pd.Series(dtype=float)
        avg_clv = float(avg_clv_series.dropna().mean()) if not avg_clv_series.dropna().empty else None
        remaining_medium_slates = max(0, medium_slates - current_slates)
        remaining_medium_picks = max(0, medium_picks - current_picks)
        remaining_high_slates = max(0, high_slates - current_slates)
        remaining_high_picks = max(0, high_picks - current_picks)
        primary_ready = bool(current_slates >= medium_slates and current_picks >= medium_picks)
        high_ready = bool(current_slates >= high_slates and current_picks >= high_picks)
        if primary_ready:
            plain_english = f"{label_map.get(market_type, market_type.title())} is ready to participate in real-market policy evidence on its own threshold ladder."
        else:
            plain_english = (
                f"{label_map.get(market_type, market_type.title())} still needs {remaining_medium_slates} more settled day(s) "
                f"and {remaining_medium_picks} more priced bet(s) before it can carry real-market policy evidence on its own."
            )
        rows.append({
            'market_type': market_type,
            'market_label': label_map.get(market_type, market_type.title()),
            'current_slates': current_slates,
            'current_picks': current_picks,
            'remaining_medium_slates': remaining_medium_slates,
            'remaining_medium_picks': remaining_medium_picks,
            'remaining_high_slates': remaining_high_slates,
            'remaining_high_picks': remaining_high_picks,
            'primary_ready': primary_ready,
            'high_ready': high_ready,
            'avg_clv': avg_clv,
            'medium_threshold_label': f"{medium_slates} slates / {medium_picks} picks",
            'high_threshold_label': f"{high_slates} slates / {high_picks} picks",
            'plain_english': plain_english,
        })
    return pd.DataFrame(rows)

def recommend_policy_research(shadow_window: pd.DataFrame, realized_daily: pd.DataFrame, window_label: str, window_days: int | None, market_window: pd.DataFrame | None = None, market_type: str | None = None) -> Dict[str, Any]:
    fallback = {
        'policy_name': 'Flat',
        'posture_name': 'Conservative',
        'window_label': window_label,
        'window_days': window_days,
        'evidence_source': 'Insufficient evidence',
        'evidence_family': 'fallback',
        'used_real_market': False,
        'evidence_strength': 'Low',
        'technical_note': 'Not enough settled archive or priced-bet data exists to rank bankroll policies yet.',
        'plain_english': 'There is not enough evidence yet, so the safest default is to keep sizing simple and conservative.',
    }
    market_window = market_window if isinstance(market_window, pd.DataFrame) else pd.DataFrame()
    market_ranked = market_window[market_window['sample_ok']].copy() if not market_window.empty and 'sample_ok' in market_window.columns and market_window['sample_ok'].any() else market_window.copy()
    shadow_ranked = shadow_window[shadow_window['sample_ok']].copy() if not shadow_window.empty and 'sample_ok' in shadow_window.columns and shadow_window['sample_ok'].any() else shadow_window.copy()
    if market_ranked.empty and shadow_ranked.empty:
        return fallback

    using_market = not market_ranked.empty
    evidence_family = 'real_market' if using_market else 'shadow'
    ranked = market_ranked if using_market else shadow_ranked
    ranked = ranked.sort_values(['stability_score', 'result_units', 'yield_pct'], ascending=[False, False, False]).reset_index(drop=True)
    top = ranked.iloc[0]
    recommended_policy = str(top.get('policy_name') or 'Flat')
    if 'policy_name' in ranked.columns and 'Flat' in ranked['policy_name'].astype(str).tolist() and len(ranked) > 1:
        flat_row = ranked[ranked['policy_name'].astype(str) == 'Flat'].head(1)
        if not flat_row.empty:
            flat_row = flat_row.iloc[0]
            if (
                (safe_float(top.get('stability_score')) or 0.0) - (safe_float(flat_row.get('stability_score')) or 0.0) < 0.15
                and (safe_float(flat_row.get('max_drawdown_units')) or 0.0) <= ((safe_float(top.get('max_drawdown_units')) or 0.0) + 0.20)
            ):
                recommended_policy = 'Flat'
                top = flat_row

    posture_name = 'Base'
    evidence_source = 'Real-market policy ranking' if using_market else 'Shadow policy ranking'
    evidence_strength = 'Low'
    realized_window = filter_to_window(realized_daily, 'report_date', window_days)
    realized_bets = int(realized_window['bets'].sum()) if not realized_window.empty and 'bets' in realized_window.columns else 0
    realized_roi = float(realized_window['result_units'].sum() / realized_window['stake_units'].sum()) if not realized_window.empty and 'stake_units' in realized_window.columns and abs(float(realized_window['stake_units'].sum())) > 1e-9 else None
    realized_drawdown = abs(float((realized_window['cumulative_units'] - realized_window['cumulative_units'].cummax()).min() or 0.0)) if not realized_window.empty and 'cumulative_units' in realized_window.columns else None
    avg_clv = float(realized_window['avg_clv'].dropna().mean()) if not realized_window.empty and 'avg_clv' in realized_window.columns and not realized_window['avg_clv'].dropna().empty else None

    top_result = safe_float(top.get('result_units')) or 0.0
    top_drawdown = safe_float(top.get('max_drawdown_units')) or 0.0
    top_yield = safe_float(top.get('yield_pct'))
    top_clv = safe_float(top.get('avg_clv'))
    if using_market:
        if top_result < 0.0 or top_drawdown >= 2.0:
            posture_name = 'Conservative'
        elif (top_yield or 0.0) >= 0.05 and top_drawdown <= 1.0 and (top_clv is None or top_clv >= 0.0):
            posture_name = 'Aggressive'
        else:
            posture_name = 'Base'
    else:
        if realized_bets >= 20:
            evidence_source = 'Shadow ranking with real-market posture overlay'
            if (realized_roi is not None and realized_roi < 0.0) or ((realized_drawdown or 0.0) >= 2.0):
                posture_name = 'Conservative'
            elif (realized_roi is not None and realized_roi >= 0.05) and ((realized_drawdown or 0.0) <= 1.0) and (avg_clv is None or avg_clv >= 0.0):
                posture_name = 'Aggressive'
            else:
                posture_name = 'Base'
        else:
            if top_drawdown >= 2.0 or top_result < 0.0:
                posture_name = 'Conservative'

    settled_slates = int(top.get('settled_slates') or 0)
    settled_picks = int(top.get('settled_picks') or 0)
    thresholds = market_policy_thresholds(market_type if using_market else None)
    medium_slates = int(thresholds['medium_slates'])
    medium_picks = int(thresholds['medium_picks'])
    high_slates = int(thresholds['high_slates'])
    high_picks = int(thresholds['high_picks'])
    if using_market:
        if settled_slates >= high_slates and settled_picks >= high_picks:
            evidence_strength = 'High'
        elif settled_slates >= medium_slates and settled_picks >= medium_picks:
            evidence_strength = 'Medium'
    else:
        if settled_slates >= 7 and settled_picks >= 40 and realized_bets >= 20:
            evidence_strength = 'High'
        elif settled_slates >= 4 and settled_picks >= 20:
            evidence_strength = 'Medium'

    source_label = 'real-market replay' if using_market else 'shadow replay'
    technical_note = (
        f"{recommended_policy} leads the selected window on {source_label} stability score "
        f"({(safe_float(top.get('stability_score')) or 0.0):.2f}) with {(safe_float(top.get('result_units')) or 0.0):+.2f}u "
        f"and {(safe_float(top.get('max_drawdown_units')) or 0.0):.2f}u max drawdown."
    )
    if using_market and top_clv is not None:
        technical_note += f" Average CLV on the priced-bet replay was {top_clv:+.3f}."
    elif (not using_market) and realized_bets >= 20:
        technical_note += (
            f" Real-market overlay saw {realized_bets} settled bets"
            f" with ROI {((realized_roi or 0.0) * 100.0):.1f}% and"
            f" max drawdown {(realized_drawdown or 0.0):.2f}u."
        )

    if using_market:
        plain_english = f"{recommended_policy} has been the best bankroll style on the actual settled bets over {window_label.lower()}."
    else:
        plain_english = f"{recommended_policy} currently looks like the best balance of upside and pain over {window_label.lower()}."
    if posture_name == 'Conservative':
        plain_english += ' The evidence still points to a tighter risk dial right now.'
    elif posture_name == 'Aggressive':
        plain_english += ' The recent path has been stable enough to justify leaning in a bit more.'
    else:
        plain_english += ' The normal risk posture is still the most sensible setting.'

    return {
        'policy_name': recommended_policy,
        'posture_name': posture_name,
        'window_label': window_label,
        'window_days': window_days,
        'evidence_source': evidence_source,
        'evidence_family': evidence_family,
        'used_real_market': using_market,
        'evidence_strength': evidence_strength,
        'technical_note': technical_note,
        'plain_english': plain_english,
        'realized_bets': realized_bets,
        'realized_roi': realized_roi,
        'realized_drawdown': realized_drawdown,
        'avg_clv': avg_clv if avg_clv is not None else top_clv,
    }

def build_policy_research_snapshot(live_archive: pd.DataFrame, bet_journal: pd.DataFrame, phase: str, run_id: str, as_of_report_date: dt.date):
    bankroll = CANONICAL_BANKROLL
    posture_name = CANONICAL_POSTURE
    deployment_scalar = CANONICAL_DEPLOYMENT_SCALAR
    unit_risk_pct = default_unit_risk_pct_for_phase(phase)
    shadow_daily, shadow_summary = build_shadow_policy_backtest(
        live_archive,
        bankroll,
        unit_risk_pct,
        deployment_scalar,
        posture_name,
    )
    market_policy_daily, market_policy_summary = build_market_policy_backtest(
        bet_journal,
        bankroll,
        unit_risk_pct,
        deployment_scalar,
        posture_name,
    )
    realized_daily = build_market_realized_daily(bet_journal)
    recommendation_rows = []
    ranking_rows = []
    market_type_recommendation_rows = []
    for window_label, window_days in DEFAULT_POLICY_WINDOWS:
        shadow_window = summarize_policy_window(shadow_daily, window_days)
        market_window = summarize_policy_window(market_policy_daily, window_days)
        if not shadow_window.empty:
            shadow_rankings = shadow_window.copy()
            shadow_rankings['run_id'] = run_id
            shadow_rankings['as_of_report_date'] = as_of_report_date.isoformat()
            shadow_rankings['phase'] = phase
            shadow_rankings['window_label'] = window_label
            shadow_rankings['window_days'] = window_days
            shadow_rankings['canonical_posture'] = posture_name
            shadow_rankings['canonical_unit_risk_pct'] = unit_risk_pct
            shadow_rankings['canonical_deployment_scalar'] = deployment_scalar
            shadow_rankings['canonical_bankroll'] = bankroll
            shadow_rankings['evidence_family'] = 'shadow'
            ranking_rows.append(shadow_rankings)
        if not market_window.empty:
            market_rankings = market_window.copy()
            market_rankings['run_id'] = run_id
            market_rankings['as_of_report_date'] = as_of_report_date.isoformat()
            market_rankings['phase'] = phase
            market_rankings['window_label'] = window_label
            market_rankings['window_days'] = window_days
            market_rankings['canonical_posture'] = posture_name
            market_rankings['canonical_unit_risk_pct'] = unit_risk_pct
            market_rankings['canonical_deployment_scalar'] = deployment_scalar
            market_rankings['canonical_bankroll'] = bankroll
            market_rankings['evidence_family'] = 'real_market'
            ranking_rows.append(market_rankings)
        recommendation = recommend_policy_research(shadow_window, realized_daily, window_label, window_days, market_window)
        recommendation_rows.append({
            'run_id': run_id,
            'as_of_report_date': as_of_report_date.isoformat(),
            'phase': phase,
            'window_label': window_label,
            'window_days': window_days,
            'recommended_policy': recommendation.get('policy_name'),
            'recommended_posture': recommendation.get('posture_name'),
            'evidence_source': recommendation.get('evidence_source'),
            'evidence_family': recommendation.get('evidence_family'),
            'used_real_market': recommendation.get('used_real_market'),
            'evidence_strength': recommendation.get('evidence_strength'),
            'technical_note': recommendation.get('technical_note'),
            'plain_english': recommendation.get('plain_english'),
            'realized_bets': recommendation.get('realized_bets'),
            'realized_roi': recommendation.get('realized_roi'),
            'realized_drawdown': recommendation.get('realized_drawdown'),
            'avg_clv': recommendation.get('avg_clv'),
            'canonical_posture': posture_name,
            'canonical_unit_risk_pct': unit_risk_pct,
            'canonical_deployment_scalar': deployment_scalar,
            'canonical_bankroll': bankroll,
        })
        market_specific = market_specific_policy_recommendations(bet_journal, shadow_window, window_label, window_days, phase)
        if not market_specific.empty:
            market_specific = market_specific.copy()
            market_specific['run_id'] = run_id
            market_specific['as_of_report_date'] = as_of_report_date.isoformat()
            market_specific['phase'] = phase
            market_specific['window_label'] = window_label
            market_specific['window_days'] = window_days
            market_specific['canonical_posture'] = posture_name
            market_specific['canonical_unit_risk_pct'] = unit_risk_pct
            market_specific['canonical_deployment_scalar'] = deployment_scalar
            market_specific['canonical_bankroll'] = bankroll
            market_type_recommendation_rows.append(market_specific)

    def _attach_snapshot_meta(frame: pd.DataFrame, empty_columns: list[str] | None = None) -> pd.DataFrame:
        output = frame.copy()
        if output.empty and empty_columns is not None:
            output = pd.DataFrame(columns=empty_columns)
        output['run_id'] = run_id
        output['as_of_report_date'] = as_of_report_date.isoformat()
        output['canonical_posture'] = posture_name
        output['canonical_unit_risk_pct'] = unit_risk_pct
        output['canonical_deployment_scalar'] = deployment_scalar
        output['canonical_bankroll'] = bankroll
        return output

    shadow_daily_frame = _attach_snapshot_meta(shadow_daily)
    shadow_summary_frame = _attach_snapshot_meta(shadow_summary)
    market_policy_daily_frame = _attach_snapshot_meta(
        market_policy_daily,
        ['report_date', 'phase', 'policy_name', 'posture_name', 'games', 'picks', 'hits', 'hit_rate', 'risk_units', 'result_units', 'yield_pct', 'unit_size_dollars', 'risk_dollars', 'result_dollars', 'avg_clv', 'avg_correlation_penalty', 'technical_note', 'plain_english', 'cumulative_units', 'cumulative_dollars', 'rolling_peak_units', 'drawdown_units', 'rolling_peak_dollars', 'drawdown_dollars'],
    )
    market_policy_summary_frame = _attach_snapshot_meta(
        market_policy_summary,
        ['policy_name', 'posture_name', 'settled_slates', 'settled_games', 'settled_picks', 'hit_rate', 'risk_units', 'result_units', 'yield_pct', 'risk_dollars', 'result_dollars', 'max_drawdown_units', 'max_drawdown_dollars', 'worst_day_units', 'avg_daily_risk_dollars', 'avg_clv', 'avg_correlation_penalty', 'technical_note', 'plain_english'],
    )

    realized_daily_frame = realized_daily.copy()
    if not realized_daily_frame.empty:
        realized_daily_frame['run_id'] = run_id
        realized_daily_frame['as_of_report_date'] = as_of_report_date.isoformat()

    def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        valid_frames = []
        for frame in frames:
            if frame is None:
                continue
            if not isinstance(frame, pd.DataFrame):
                continue
            if frame.empty:
                continue
            trimmed = frame.dropna(axis=0, how='all').dropna(axis=1, how='all')
            if trimmed.empty:
                continue
            valid_frames.append(trimmed.copy())
        return pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()

    ranking_frame = _safe_concat(ranking_rows)
    recommendation_frame = pd.DataFrame(recommendation_rows)
    market_type_recommendation_frame = _safe_concat(market_type_recommendation_rows)
    return {
        'shadow_daily': shadow_daily_frame,
        'shadow_summary': shadow_summary_frame,
        'market_policy_daily': market_policy_daily_frame,
        'market_policy_summary': market_policy_summary_frame,
        'realized_daily': realized_daily_frame,
        'window_rankings': ranking_frame,
        'recommendations': recommendation_frame,
        'market_type_recommendations': market_type_recommendation_frame,
    }








