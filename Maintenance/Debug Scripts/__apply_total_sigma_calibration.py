from pathlib import Path

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f'missing pattern for {label}')
    return text.replace(old, new, 1)

# run_daily_mlb_report.py
path = Path('run_daily_mlb_report.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """TOTAL_CALIBRATION_MIN_GAMES = 30
TOTAL_CALIBRATION_MIN_IMPROVEMENT = 0.03
SHRINKAGE_BRIDGE_DAYS = 21
""",
    """TOTAL_CALIBRATION_MIN_GAMES = 30
TOTAL_CALIBRATION_MIN_IMPROVEMENT = 0.03
TOTAL_SIGMA_CALIBRATION_MIN_GAMES = 24
TOTAL_SIGMA_CALIBRATION_MIN_IMPROVEMENT = 0.01
SHRINKAGE_BRIDGE_DAYS = 21
""",
    'sigma calibration constants',
)
text = replace_once(
    text,
    """def _mae_metric(actual, predicted):
    if actual is None or predicted is None:
        return None
    a = pd.to_numeric(pd.Series(actual), errors='coerce')
    p = pd.to_numeric(pd.Series(predicted), errors='coerce')
    valid = (~a.isna()) & (~p.isna())
    if not valid.any():
        return None
    diff = a[valid].astype(float) - p[valid].astype(float)
    return float(np.mean(np.abs(diff)))


def simple_total_baseline_from_phase(phase, state):
""",
    """def _mae_metric(actual, predicted):
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
""",
    'gaussian nll helper',
)
text = replace_once(
    text,
    """def fit_margin_shrinkage_profile(state, backtest_df):
""",
    """def fit_total_sigma_profile(state, backtest_df):
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
""",
    'fit total sigma profile',
)
text = replace_once(
    text,
    """def apply_margin_calibration(margin, state, phase):
    profile = ((state.get('margin_calibration') or {}).get(phase) or {})
    if not profile.get('apply'):
        return finite(margin, 0.0)
    return float(np.clip(finite(profile.get('intercept'), 0.0) + (finite(profile.get('slope'), 1.0) * finite(margin, 0.0)), -7.0, 7.0))

""",
    """def apply_total_sigma_calibration(total_sigma, state, phase):
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

""",
    'apply total sigma calibration',
)
text = replace_once(
    text,
    """    total_sigma = estimate_total_sigma(
        phase,
        shared_env,
        starter_uncertainty,
        short_start_total,
        bullpen_fragility,
        lineup_missing,
        finite(weather_adj.get('total_adj'), 0.0),
        offense_pressure,
    )
""",
    """    total_sigma_raw = estimate_total_sigma(
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
""",
    'predict_game sigma raw/calibrated',
)
text = replace_once(
    text,
    """        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple, 'probability_shrink_alpha': probability_shrink_alpha, 'p_home_no_lineup': p_home_no_lineup, 'p_home_full_lineup': p_home_full_lineup, 'lineup_multiplier': lineup_multiplier, 'total_raw': total_raw, 'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma': total_sigma, 'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    """        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple, 'probability_shrink_alpha': probability_shrink_alpha, 'p_home_no_lineup': p_home_no_lineup, 'p_home_full_lineup': p_home_full_lineup, 'lineup_multiplier': lineup_multiplier, 'total_raw': total_raw, 'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma_raw': total_sigma_raw, 'total_sigma': total_sigma, 'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    'return total sigma raw',
)
text = replace_once(
    text,
    """            'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma': total_sigma,
            'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    """            'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma_raw': total_sigma_raw, 'total_sigma': total_sigma,
            'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    'feature total sigma raw',
)
text = replace_once(
    text,
    """    probability_profile, total_profile, lineup_profile = ((state.get('probability_calibration') or {}).get(phase) or {}), ((state.get('total_calibration') or {}).get(phase) or {}), (state.get('lineup_earn_back') or {})
""",
    """    probability_profile, total_profile, total_sigma_profile, lineup_profile = ((state.get('probability_calibration') or {}).get(phase) or {}), ((state.get('total_calibration') or {}).get(phase) or {}), ((state.get('total_sigma_calibration') or {}).get(phase) or {}), (state.get('lineup_earn_back') or {})
""",
    'report profile tuple',
)
text = replace_once(
    text,
    """    if total_profile.get('apply'):
        lines.append(f"Total calibration: {total_profile.get('mode', 'unknown')} | slope {finite(total_profile.get('slope'), 1.0):.3f} | intercept {finite(total_profile.get('intercept'), 0.0):+.3f}")
""",
    """    if total_profile.get('apply'):
        lines.append(f"Total calibration: {total_profile.get('mode', 'unknown')} | slope {finite(total_profile.get('slope'), 1.0):.3f} | intercept {finite(total_profile.get('intercept'), 0.0):+.3f}")
    if total_sigma_profile:
        lines.append(f"Total sigma calibration: {'active' if total_sigma_profile.get('apply') else 'inactive'} | scale {finite(total_sigma_profile.get('scale'), 1.0):.2f} | raw NLL {display_number(total_sigma_profile.get('raw_nll'), '.3f')} | calibrated NLL {display_number(total_sigma_profile.get('calibrated_nll'), '.3f')}")
""",
    'report sigma calibration line',
)
text = replace_once(
    text,
    """    model_state, shrink_profile = fit_probability_shrinkage_profile(model_state, backtest_log)
    model_state, margin_shrink_profile = fit_margin_shrinkage_profile(model_state, backtest_log)
    model_state, lineup_profile = update_lineup_earn_back_state(model_state, live_archive)
""",
    """    model_state, shrink_profile = fit_probability_shrinkage_profile(model_state, backtest_log)
    model_state, total_sigma_profile = fit_total_sigma_profile(model_state, backtest_log)
    model_state, margin_shrink_profile = fit_margin_shrinkage_profile(model_state, backtest_log)
    model_state, lineup_profile = update_lineup_earn_back_state(model_state, live_archive)
""",
    'main fit sigma profile',
)
text = replace_once(
    text,
    """    total_shrink_phase = ((model_state.get('total_shrinkage') or {}).get(phase) or {})
    total_shrink_bridge = ((model_state.get('total_shrinkage') or {}).get('regular_bridge') or {})
    margin_shrink_phase = ((model_state.get('margin_shrinkage') or {}).get(phase) or {})
""",
    """    total_shrink_phase = ((model_state.get('total_shrinkage') or {}).get(phase) or {})
    total_shrink_bridge = ((model_state.get('total_shrinkage') or {}).get('regular_bridge') or {})
    total_sigma_phase = ((model_state.get('total_sigma_calibration') or {}).get(phase) or {})
    margin_shrink_phase = ((model_state.get('margin_shrinkage') or {}).get(phase) or {})
""",
    'main sigma phase lookup',
)
text = replace_once(
    text,
    """    if str(phase).lower() == 'spring' and total_shrink_phase:
        warnings.append(f"Spring total shrinkage: {'active' if total_shrink_phase.get('apply') else 'inactive'} | alpha {finite(total_shrink_phase.get('alpha'), 0.0):.2f}")
    elif str(phase).lower() == 'regular' and total_shrink_bridge.get('apply'):
        warnings.append(f"Early regular total bridge max alpha: {finite(total_shrink_bridge.get('max_alpha'), 0.0):.2f} over {int(finite(total_shrink_bridge.get('bridge_days'), 0.0))} days.")
    if str(phase).lower() == 'spring' and margin_shrink_phase:
""",
    """    if str(phase).lower() == 'spring' and total_shrink_phase:
        warnings.append(f"Spring total shrinkage: {'active' if total_shrink_phase.get('apply') else 'inactive'} | alpha {finite(total_shrink_phase.get('alpha'), 0.0):.2f}")
    elif str(phase).lower() == 'regular' and total_shrink_bridge.get('apply'):
        warnings.append(f"Early regular total bridge max alpha: {finite(total_shrink_bridge.get('max_alpha'), 0.0):.2f} over {int(finite(total_shrink_bridge.get('bridge_days'), 0.0))} days.")
    if total_sigma_phase:
        warnings.append(f"{phase.title()} total sigma calibration: {'active' if total_sigma_phase.get('apply') else 'inactive'} | scale {finite(total_sigma_phase.get('scale'), 1.0):.2f}")
    if str(phase).lower() == 'spring' and margin_shrink_phase:
""",
    'main sigma warning',
)
path.write_text(text, encoding='utf-8')

# clean_backtest_runner.py
path = Path('clean_backtest_runner.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """        'projected_total_simple': float(prediction.get('total_simple', prediction.get('total_calibrated', 0.0))),
        'projected_total_sigma': float(prediction.get('total_sigma', 4.0)),
        'away_expected_runs': float(prediction.get('away_runs', 0.0)),
""",
    """        'projected_total_simple': float(prediction.get('total_simple', prediction.get('total_calibrated', 0.0))),
        'projected_total_sigma_raw': float(prediction.get('total_sigma_raw', prediction.get('total_sigma', 4.0))),
        'projected_total_sigma': float(prediction.get('total_sigma', 4.0)),
        'away_expected_runs': float(prediction.get('away_runs', 0.0)),
""",
    'replay sigma raw column',
)
path.write_text(text, encoding='utf-8')

# quant_store.py
path = Path('quant_store.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """            'total_shrink_alpha': _safe_float(prediction.get('total_shrink_alpha')),
            'total_sigma': _safe_float(prediction.get('total_sigma')),
            'margin_calibrated': _safe_float(prediction.get('margin_calibrated')),
""",
    """            'total_shrink_alpha': _safe_float(prediction.get('total_shrink_alpha')),
            'total_sigma_raw': _safe_float(prediction.get('total_sigma_raw')),
            'total_sigma': _safe_float(prediction.get('total_sigma')),
            'margin_calibrated': _safe_float(prediction.get('margin_calibrated')),
""",
    'quant total sigma raw',
)
path.write_text(text, encoding='utf-8')

print('updated sigma calibration across runner, replay, and quant store')
