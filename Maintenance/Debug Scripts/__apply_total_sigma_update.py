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
    """TOTAL_MODEL_PHASE_PARAMS = {
    \"spring\": {
        \"base_runs\": 4.58,
        \"offense_coef\": 0.62,
        \"starter_coef\": 0.80,
        \"bullpen_coef\": 0.32,
        \"shared_total_coef\": 0.34,
        \"schedule_coef\": 0.04,
        \"defense_coef\": 0.08,
        \"lineup_coef\": 0.10,
        \"bridge_coef\": 0.08,
        \"availability_coef\": 0.03,
        \"shared_env_offense_coef\": 0.05,
        \"shared_env_support_coef\": 0.04,
        \"shared_env_starter_coef\": 0.08,
        \"shared_env_bullpen_coef\": 0.04,
        \"shared_env_short_coef\": 0.12,
        \"shared_env_bridge_coef\": 0.08,
        \"shared_env_uncertainty_coef\": 0.05,
        \"run_floor\": 1.7,
        \"run_cap\": 7.9,
        \"env_cap\": 0.90,
    },
    \"regular\": {
        \"base_runs\": 4.50,
        \"offense_coef\": 0.65,
        \"starter_coef\": 0.84,
        \"bullpen_coef\": 0.35,
        \"shared_total_coef\": 0.38,
        \"schedule_coef\": 0.05,
        \"defense_coef\": 0.09,
        \"lineup_coef\": 0.11,
        \"bridge_coef\": 0.09,
        \"availability_coef\": 0.03,
        \"shared_env_offense_coef\": 0.06,
        \"shared_env_support_coef\": 0.05,
        \"shared_env_starter_coef\": 0.09,
        \"shared_env_bullpen_coef\": 0.05,
        \"shared_env_short_coef\": 0.14,
        \"shared_env_bridge_coef\": 0.09,
        \"shared_env_uncertainty_coef\": 0.06,
        \"run_floor\": 1.8,
        \"run_cap\": 8.2,
        \"env_cap\": 1.05,
    },
}
""",
    """TOTAL_MODEL_PHASE_PARAMS = {
    \"spring\": {
        \"base_runs\": 4.58,
        \"offense_coef\": 0.62,
        \"starter_coef\": 0.80,
        \"bullpen_coef\": 0.32,
        \"shared_total_coef\": 0.34,
        \"schedule_coef\": 0.04,
        \"defense_coef\": 0.08,
        \"lineup_coef\": 0.10,
        \"bridge_coef\": 0.08,
        \"availability_coef\": 0.03,
        \"shared_env_offense_coef\": 0.05,
        \"shared_env_support_coef\": 0.04,
        \"shared_env_starter_coef\": 0.08,
        \"shared_env_bullpen_coef\": 0.04,
        \"shared_env_short_coef\": 0.12,
        \"shared_env_bridge_coef\": 0.08,
        \"shared_env_uncertainty_coef\": 0.05,
        \"run_floor\": 1.7,
        \"run_cap\": 7.9,
        \"env_cap\": 0.90,
    },
    \"regular\": {
        \"base_runs\": 4.50,
        \"offense_coef\": 0.65,
        \"starter_coef\": 0.84,
        \"bullpen_coef\": 0.35,
        \"shared_total_coef\": 0.38,
        \"schedule_coef\": 0.05,
        \"defense_coef\": 0.09,
        \"lineup_coef\": 0.11,
        \"bridge_coef\": 0.09,
        \"availability_coef\": 0.03,
        \"shared_env_offense_coef\": 0.06,
        \"shared_env_support_coef\": 0.05,
        \"shared_env_starter_coef\": 0.09,
        \"shared_env_bullpen_coef\": 0.05,
        \"shared_env_short_coef\": 0.14,
        \"shared_env_bridge_coef\": 0.09,
        \"shared_env_uncertainty_coef\": 0.06,
        \"run_floor\": 1.8,
        \"run_cap\": 8.2,
        \"env_cap\": 1.05,
    },
}
TOTAL_SIGMA_PHASE_PARAMS = {
    \"spring\": {
        \"base_sigma\": 4.10,
        \"starter_uncertainty_coef\": 0.34,
        \"short_start_coef\": 0.62,
        \"bullpen_fragility_coef\": 0.48,
        \"lineup_missing_coef\": 0.22,
        \"env_coef\": 0.28,
        \"weather_vol_coef\": 0.18,
        \"offense_pressure_coef\": 0.10,
        \"min_sigma\": 3.10,
        \"max_sigma\": 6.20,
    },
    \"regular\": {
        \"base_sigma\": 3.75,
        \"starter_uncertainty_coef\": 0.28,
        \"short_start_coef\": 0.48,
        \"bullpen_fragility_coef\": 0.38,
        \"lineup_missing_coef\": 0.14,
        \"env_coef\": 0.22,
        \"weather_vol_coef\": 0.14,
        \"offense_pressure_coef\": 0.08,
        \"min_sigma\": 2.85,
        \"max_sigma\": 5.60,
    },
}
""",
    'total sigma constants',
)
text = replace_once(
    text,
    """def total_model_params(phase):
    phase_name = str(phase or 'spring').lower()
    params = dict(TOTAL_MODEL_PHASE_PARAMS.get('regular') or {})
    params.update(TOTAL_MODEL_PHASE_PARAMS.get(phase_name) or {})
    return params

""",
    """def total_model_params(phase):
    phase_name = str(phase or 'spring').lower()
    params = dict(TOTAL_MODEL_PHASE_PARAMS.get('regular') or {})
    params.update(TOTAL_MODEL_PHASE_PARAMS.get(phase_name) or {})
    return params


def total_sigma_params(phase):
    phase_name = str(phase or 'spring').lower()
    params = dict(TOTAL_SIGMA_PHASE_PARAMS.get('regular') or {})
    params.update(TOTAL_SIGMA_PHASE_PARAMS.get(phase_name) or {})
    return params

""",
    'total sigma helper',
)
text = replace_once(
    text,
    """def fair_run_line_prices(margin, sigma):
    sigma_value = float(np.clip(finite(sigma, 3.25), 1.75, 6.0))
    home_cover = 1.0 - normal_cdf((1.5 - finite(margin, 0.0)) / sigma_value)
    home_cover = float(np.clip(home_cover, 1e-6, 1.0 - 1e-6))
    away_cover = 1.0 - home_cover
    return american_from_probability(home_cover), american_from_probability(away_cover)

""",
    """def estimate_total_sigma(phase, shared_env, starter_uncertainty, short_start_total, bullpen_fragility, lineup_missing, weather_total_adj, offense_pressure):
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

""",
    'total sigma functions',
)
text = replace_once(
    text,
    """    total_raw = away_runs_raw + home_runs_raw
    total_model = apply_total_calibration(total_raw, state, phase)
    total_calibrated, total_simple, total_shrink_alpha = blend_total_with_baseline(total_model, phase, state, target_date)
    scale = total_calibrated / total_raw if total_raw > 0 else 1.0
""",
    """    total_raw = away_runs_raw + home_runs_raw
    short_start_total = finite(away_starter_workload.get('short_start_risk'), 0.0) + finite(home_starter_workload.get('short_start_risk'), 0.0)
    bullpen_fragility = (
        (max(0.0, -away_bullpen) * finite(away_starter_workload.get('bullpen_share'), 0.45))
        + (max(0.0, -home_bullpen) * finite(home_starter_workload.get('bullpen_share'), 0.45))
        + (0.35 * max(0.0, -away_bullpen_availability))
        + (0.35 * max(0.0, -home_bullpen_availability))
    )
    lineup_missing = max(0.0, (18.0 - float((lineups.get('away', 0) or 0) + (lineups.get('home', 0) or 0))) / 9.0)
    offense_pressure = max(0.0, away_off) + max(0.0, home_off)
    total_sigma = estimate_total_sigma(
        phase,
        shared_env,
        starter_uncertainty,
        short_start_total,
        bullpen_fragility,
        lineup_missing,
        finite(weather_adj.get('total_adj'), 0.0),
        offense_pressure,
    )
    total_model = apply_total_calibration(total_raw, state, phase)
    total_calibrated, total_simple, total_shrink_alpha = blend_total_with_baseline(total_model, phase, state, target_date)
    scale = total_calibrated / total_raw if total_raw > 0 else 1.0
""",
    'predict_game total sigma block',
)
text = replace_once(
    text,
    """        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple, 'probability_shrink_alpha': probability_shrink_alpha, 'p_home_no_lineup': p_home_no_lineup, 'p_home_full_lineup': p_home_full_lineup, 'lineup_multiplier': lineup_multiplier, 'total_raw': total_raw, 'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    """        'game': game, 'winner': winner, 'winner_prob': winner_prob, 'p_home_raw': p_home_raw, 'p_home': p_home, 'p_home_model': p_home_model, 'p_home_simple': p_home_simple, 'probability_shrink_alpha': probability_shrink_alpha, 'p_home_no_lineup': p_home_no_lineup, 'p_home_full_lineup': p_home_full_lineup, 'lineup_multiplier': lineup_multiplier, 'total_raw': total_raw, 'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma': total_sigma, 'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    'predict_game return total sigma',
)
text = replace_once(
    text,
    """            'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha,
            'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    """            'total_model': total_model, 'total_simple': total_simple, 'total_shrink_alpha': total_shrink_alpha, 'total_sigma': total_sigma,
            'margin_model': margin_model, 'margin_simple': margin_simple, 'margin_shrink_alpha': margin_shrink_alpha,
""",
    'predict_game feature total sigma',
)
text = replace_once(
    text,
    """    market_total = market.parse_market_float(market_row.get('total_line'))
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
""",
    """    market_total = market.parse_market_float(market_row.get('total_line'))
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
        }
""",
    'market comparison total probability block',
)
text = replace_once(
    text,
    """        lines.append(f\"Model expected total: {prediction['total_calibrated']:.1f}\")
        lines.append(f\"Projected team runs: {game['away_team']} {prediction['away_runs']:.1f} | {game['home_team']} {prediction['home_runs']:.1f}\")
        lines.append(f\"Fair betting total: {prediction['total_bet_line']:.1f}\")
""",
    """        lines.append(f\"Model expected total: {prediction['total_calibrated']:.1f}\")
        lines.append(f\"Projected team runs: {game['away_team']} {prediction['away_runs']:.1f} | {game['home_team']} {prediction['home_runs']:.1f}\")
        lines.append(f\"Fair betting total: {prediction['total_bet_line']:.1f}\")
        lines.append(f\"Total volatility estimate (sigma): {finite(prediction.get('total_sigma'), 4.0):.2f}\")
""",
    'report total sigma line',
)
path.write_text(text, encoding='utf-8')

# quant_store.py
path = Path('quant_store.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """            'total_calibrated': _safe_float(prediction.get('total_calibrated')),
            'total_model': _safe_float(prediction.get('total_model')),
            'total_simple': _safe_float(prediction.get('total_simple')),
            'total_shrink_alpha': _safe_float(prediction.get('total_shrink_alpha')),
            'margin_calibrated': _safe_float(prediction.get('margin_calibrated')),
""",
    """            'total_calibrated': _safe_float(prediction.get('total_calibrated')),
            'total_model': _safe_float(prediction.get('total_model')),
            'total_simple': _safe_float(prediction.get('total_simple')),
            'total_shrink_alpha': _safe_float(prediction.get('total_shrink_alpha')),
            'total_sigma': _safe_float(prediction.get('total_sigma')),
            'margin_calibrated': _safe_float(prediction.get('margin_calibrated')),
""",
    'quant common total sigma',
)
text = replace_once(
    text,
    """        total_market = market_comp.get('total') or {}
        if total_market.get('available'):
            action = total_market.get('action') or {}
            lean = str(total_market.get('lean') or 'FLAT').upper()
            best_price = _safe_int(total_market.get('best_price'))
            raw_ev = _total_expected_value_proxy(total_market.get('diff'), best_price)
            uncertainty = _candidate_uncertainty(prediction, 'total', sportsbook_count, phase, uncertainty_context)
            adjusted_ev = _adjusted_expected_value(raw_ev, uncertainty, 'total')
            action_score = _safe_float(action.get('score')) or 0.0
            diff_value = abs(_safe_float(total_market.get('diff')) or 0.0)
            candidate_score = round(
                (diff_value * max(action_score, 0.0) * max((_safe_float(uncertainty.get('penalty')) or 0.0), 0.40))
                + ((adjusted_ev or 0.0) * 100.0),
                4,
            )
            rows.append({
                **common,
                'market_type': 'total',
                'selection': f"{lean} {(_safe_float(total_market.get('market_total')) or 0.0):.1f}",
                'line_value': _safe_float(total_market.get('market_total')),
                'book_price': best_price,
                'fair_price': None,
                'model_probability': None,
                'market_probability': None,
                'edge_value': _safe_float(total_market.get('diff')),
                'actionability_label': str(action.get('label') or 'PASS'),
                'actionability_score': _safe_float(action.get('score')),
                'actionability_notes': str(action.get('notes') or ''),
                'expected_value_per_unit': raw_ev,
                'raw_expected_value_per_unit': raw_ev,
                'adjusted_expected_value_per_unit': adjusted_ev,
                'expected_value_method': 'run-gap proxy EV',
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
                'suggested_units_preview': _suggested_units_preview(action.get('label'), action.get('score'), prediction, 'total'),
                'candidate_score': candidate_score,
            })
""",
    """        total_market = market_comp.get('total') or {}
        if total_market.get('available'):
            action = total_market.get('action') or {}
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
                'suggested_units_preview': _suggested_units_preview(action.get('label'), action.get('score'), prediction, 'total'),
                'candidate_score': candidate_score,
            })
""",
    'quant total EV block',
)
path.write_text(text, encoding='utf-8')

# clean_backtest_runner.py
path = Path('clean_backtest_runner.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """        'projected_total': float(prediction.get('total_calibrated', 0.0)),
        'projected_total_raw': float(prediction.get('total_raw', prediction.get('total_calibrated', 0.0))),
        'projected_total_model': float(prediction.get('total_model', prediction.get('total_calibrated', 0.0))),
        'projected_total_simple': float(prediction.get('total_simple', prediction.get('total_calibrated', 0.0))),
        'away_expected_runs': float(prediction.get('away_runs', 0.0)),
""",
    """        'projected_total': float(prediction.get('total_calibrated', 0.0)),
        'projected_total_raw': float(prediction.get('total_raw', prediction.get('total_calibrated', 0.0))),
        'projected_total_model': float(prediction.get('total_model', prediction.get('total_calibrated', 0.0))),
        'projected_total_simple': float(prediction.get('total_simple', prediction.get('total_calibrated', 0.0))),
        'projected_total_sigma': float(prediction.get('total_sigma', 4.0)),
        'away_expected_runs': float(prediction.get('away_runs', 0.0)),
""",
    'clean replay total sigma',
)
path.write_text(text, encoding='utf-8')

print('updated run_daily_mlb_report.py, quant_store.py, clean_backtest_runner.py')
