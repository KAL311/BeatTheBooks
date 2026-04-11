"""
Regular-season earn-back optimizer.

Uses the pre-computed feature columns in clean_backtest_log.csv to run forward
feature selection for the regular-season weight set.  Starting from a base model
(offense + home field only), each candidate feature is tried in turn; those that
improve out-of-sample log loss are retained.

Usage:
    python fit_regular_earn_back.py [--min-oos-games N] [--dry-run]

Writes updated regular_earn_back block to model_state.json when not in dry-run mode.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_daily_mlb_report as m

# ── constants ──────────────────────────────────────────────────────────────────
MIN_REGULAR_GAMES = 20          # minimum OOS games required to activate a feature
IMPROVEMENT_THRESHOLD = 0.0005  # minimum OOS log-loss improvement to keep a feature
OOS_DATE_HOLDOUT = 2            # number of most-recent dates held out as OOS
WEIGHT_BOUNDS = {               # (min, max) for scipy optimizer — tight bounds prevent overfit on small samples
    'w_off':      (0.05, 0.80),
    'w_starter':  (0.0,  0.80),
    'w_bullpen':  (0.0,  0.40),
    'w_park':     (0.0,  0.15),
    'w_weather':  (0.0,  0.10),
    'w_context':  (0.0,  0.40),
    'home_field': (0.02, 0.20),  # raw MLB HFA ~0.16 logit; full-feature models typically 0.05-0.10
}
CANDIDATE_FEATURES = ['w_starter', 'w_bullpen', 'w_park', 'w_weather', 'w_context']
FEATURE_COLUMNS = {
    'w_off':      'x_off',
    'w_starter':  'x_starter',
    'w_bullpen':  'x_bullpen',
    'w_park':     'x_park',
    'w_weather':  'x_weather',
    'w_context':  'x_context',
}


# ── helpers ───────────────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def build_logit(df: pd.DataFrame, active_weights: dict[str, float]) -> np.ndarray:
    logit = np.zeros(len(df))
    for weight_key, col in FEATURE_COLUMNS.items():
        w = active_weights.get(weight_key, 0.0)
        if w != 0.0 and col in df.columns:
            logit += w * pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
    logit += active_weights.get('home_field', 0.0)
    return logit


def fit_weights(train_df: pd.DataFrame, active_keys: list[str],
                start_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Fit weights for active_keys + home_field on train_df using scipy L-BFGS-B."""
    keys = active_keys + ['home_field']
    y = pd.to_numeric(train_df['y_home'], errors='coerce').fillna(0.5).values

    # Starting point: use provided start_weights, else use spring weights from model state
    state = m.load_model_state()
    spring_w = (state.get('weights') or {})
    x0 = np.array([
        (start_weights or spring_w).get(k, WEIGHT_BOUNDS[k][0])
        for k in keys
    ])
    bounds = [WEIGHT_BOUNDS[k] for k in keys]

    def objective(x: np.ndarray) -> float:
        w = dict(zip(keys, x))
        logit = build_logit(train_df, w)
        return log_loss(y, sigmoid(logit))

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-9})
    return dict(zip(keys, result.x))


def evaluate_oos(oos_df: pd.DataFrame, weights: dict[str, float]) -> float:
    y = pd.to_numeric(oos_df['y_home'], errors='coerce').fillna(0.5).values
    logit = build_logit(oos_df, weights)
    return log_loss(y, sigmoid(logit))


def accuracy(oos_df: pd.DataFrame, weights: dict[str, float]) -> float:
    y = pd.to_numeric(oos_df['y_home'], errors='coerce').fillna(0.5).values
    logit = build_logit(oos_df, weights)
    pred = (sigmoid(logit) >= 0.5).astype(int)
    return float((pred == y.astype(int)).mean())


# ── main optimizer ─────────────────────────────────────────────────────────────
def run_earn_back(min_oos_games: int = MIN_REGULAR_GAMES, dry_run: bool = False) -> dict[str, Any]:
    backtest_log = m.load_validation_log()
    if backtest_log.empty:
        return {'error': 'No backtest log available.'}

    reg = backtest_log[
        (backtest_log.get('phase', pd.Series(dtype=str)) == 'regular')
        if 'phase' in backtest_log.columns else [True] * len(backtest_log)
    ].copy()
    if 'date' not in reg.columns:
        return {'error': 'No date column in backtest log.'}

    reg['date'] = pd.to_datetime(reg['date'], errors='coerce').dt.date
    reg = reg.dropna(subset=['date', 'y_home'])
    dates = sorted(reg['date'].unique())

    if len(dates) < OOS_DATE_HOLDOUT + 1:
        return {
            'apply': False,
            'mode': 'awaiting_regular_sample',
            'reason': f'Only {len(dates)} regular dates available; need at least {OOS_DATE_HOLDOUT + 1}.',
        }

    oos_dates = set(dates[-OOS_DATE_HOLDOUT:])
    train_df = reg[~reg['date'].isin(oos_dates)].copy()
    oos_df   = reg[reg['date'].isin(oos_dates)].copy()

    print(f'Train: {len(train_df)} games across {len(dates) - OOS_DATE_HOLDOUT} dates '
          f'({min(d for d in dates if d not in oos_dates)} – {max(d for d in dates if d not in oos_dates)})')
    print(f'OOS:   {len(oos_df)} games across {OOS_DATE_HOLDOUT} dates '
          f'({min(oos_dates)} – {max(oos_dates)})')

    if len(oos_df) < min_oos_games:
        return {
            'apply': False,
            'mode': 'awaiting_regular_sample',
            'reason': f'Only {len(oos_df)} OOS games; need at least {min_oos_games} to run earn-back.',
        }

    # ── base model: offense + home field only ──────────────────────────────────
    print('\n── Base model (offense + home field) ──')
    base_active = ['w_off']
    base_weights = fit_weights(train_df, base_active)
    base_oos_ll = evaluate_oos(oos_df, base_weights)
    base_acc = accuracy(oos_df, base_weights)
    print(f'  weights: {_fmt_weights(base_weights)}')
    print(f'  OOS log loss: {base_oos_ll:.5f} | accuracy: {base_acc:.3f}')

    # ── forward feature selection ──────────────────────────────────────────────
    selected = list(base_active)
    current_weights = dict(base_weights)
    current_oos_ll = base_oos_ll
    feature_results: dict[str, Any] = {}

    print('\n── Forward feature selection ──')
    for candidate in CANDIDATE_FEATURES:
        trial_active = selected + [candidate]
        trial_weights = fit_weights(train_df, trial_active, start_weights=current_weights)
        trial_oos_ll = evaluate_oos(oos_df, trial_weights)
        trial_acc = accuracy(oos_df, trial_weights)
        improvement = current_oos_ll - trial_oos_ll  # positive = better
        kept = improvement >= IMPROVEMENT_THRESHOLD
        status = 'KEPT' if kept else 'skipped'
        print(f'  +{candidate}: OOS {trial_oos_ll:.5f} (Δ {improvement:+.5f}) {status}')
        feature_results[candidate] = {
            'oos_log_loss': trial_oos_ll,
            'improvement': improvement,
            'kept': kept,
            'fitted_weight': float(trial_weights.get(candidate, 0.0)),
        }
        if kept:
            selected.append(candidate)
            current_weights = trial_weights
            current_oos_ll = trial_oos_ll

    # ── build adjusted_weights: zeroed for non-selected features ──────────────
    state = m.load_model_state()
    spring_w = state.get('weights') or {}
    adjusted = {
        'w_off':      float(current_weights.get('w_off', spring_w.get('w_off', 0.272))),
        'w_starter':  float(current_weights.get('w_starter', 0.0))  if 'w_starter' in selected else 0.0,
        'w_bullpen':  float(current_weights.get('w_bullpen', 0.0))  if 'w_bullpen' in selected else 0.0,
        'w_park':     float(current_weights.get('w_park', 0.0))     if 'w_park'    in selected else 0.0,
        'w_weather':  float(current_weights.get('w_weather', 0.0))  if 'w_weather' in selected else 0.0,
        'w_context':  float(current_weights.get('w_context', 0.0))  if 'w_context' in selected else 0.0,
        'home_field': float(current_weights.get('home_field', spring_w.get('home_field', 0.068))),
        'spread_scale': float(spring_w.get('spread_scale', 8.0)),
    }

    enabled_features = [f for f in CANDIDATE_FEATURES if f in selected]

    # Compare against current in-use weights on the same OOS data before activating.
    # Only apply earn-back if it beats (or ties within threshold) the spring-tuned weights.
    spring_weights = state.get('weights') or {}
    spring_oos_ll = evaluate_oos(oos_df, spring_weights)
    spring_acc = accuracy(oos_df, spring_weights)
    earn_back_beats_spring = (spring_oos_ll - current_oos_ll) >= -IMPROVEMENT_THRESHOLD
    print(f'\n── Comparison vs spring-tuned weights ──')
    print(f'  Spring weights OOS log loss: {spring_oos_ll:.5f} | accuracy: {spring_acc:.3f}')
    print(f'  Earn-back OOS log loss:      {current_oos_ll:.5f} | accuracy: {base_acc:.3f}')
    print(f'  Earn-back beats spring: {earn_back_beats_spring}')

    # Require earn-back to strictly beat spring weights by at least the improvement threshold
    apply = len(oos_df) >= min_oos_games and (spring_oos_ll - current_oos_ll) >= IMPROVEMENT_THRESHOLD

    print(f'\n── Result ──')
    print(f'  Enabled features: {enabled_features}')
    print(f'  Base OOS log loss:    {base_oos_ll:.5f}')
    print(f'  Final OOS log loss:   {current_oos_ll:.5f}')
    print(f'  Improvement:          {base_oos_ll - current_oos_ll:+.5f}')
    print(f'  Adjusted weights: {_fmt_weights(adjusted)}')

    result = {
        'apply': apply,
        'mode': 'regular_earn_back',
        'enabled_features': enabled_features,
        'reason': (
            f'Regular earn-back: {len(enabled_features)}/{len(CANDIDATE_FEATURES)} features earned back '
            f'over {len(oos_df)} OOS games ({min(oos_dates)} to {max(oos_dates)}). '
            f'OOS log loss {base_oos_ll:.5f} → {current_oos_ll:.5f} '
            f'(Δ {base_oos_ll - current_oos_ll:+.5f}).'
        ),
        'base_oos_log_loss': base_oos_ll,
        'selected_oos_log_loss': current_oos_ll,
        'feature_results': feature_results,
        'adjusted_weights': adjusted,
        'train_games': int(len(train_df)),
        'oos_games': int(len(oos_df)),
        'oos_date_span': f'{min(oos_dates)} to {max(oos_dates)}',
    }

    if not dry_run:
        state['regular_earn_back'] = result
        state['last_update_date'] = dt.date.today().isoformat()
        m.save_model_state(state)
        print(f'\nmodel_state.json updated (regular_earn_back + last_update_date).')
    else:
        print('\n[dry-run] model_state.json not modified.')

    return result


def _fmt_weights(w: dict) -> str:
    parts = []
    for k in ['w_off', 'w_starter', 'w_bullpen', 'w_park', 'w_weather', 'w_context', 'home_field']:
        if k in w:
            parts.append(f'{k.replace("w_","").replace("home_field","hfa")}={w[k]:.3f}')
    return ' '.join(parts)


# ── entry point ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Regular-season earn-back weight optimizer.')
    p.add_argument('--min-oos-games', type=int, default=MIN_REGULAR_GAMES,
                   help=f'Minimum OOS games required (default {MIN_REGULAR_GAMES}).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print results without writing to model_state.json.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = run_earn_back(min_oos_games=args.min_oos_games, dry_run=args.dry_run)
    if 'error' in result:
        print('ERROR:', result['error'])
