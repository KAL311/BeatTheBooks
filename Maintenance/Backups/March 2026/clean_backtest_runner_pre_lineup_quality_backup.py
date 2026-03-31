from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os

import pandas as pd

import run_daily_mlb_report as m

DEFAULT_OUTPUT_PATH = m.CLEAN_BACKTEST_LOG_PATH
DEFAULT_SUMMARY_PATH = os.path.join(m.OUT_DIR, 'clean_backtest_summary.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Replay historical MLB slates through the clean runner and build a clean evaluation log.')
    parser.add_argument('--start-date', type=str, default=None, help='Inclusive start date in YYYY-MM-DD format.')
    parser.add_argument('--end-date', type=str, default=None, help='Inclusive end date in YYYY-MM-DD format.')
    parser.add_argument('--max-dates', type=int, default=None, help='Optional cap on replay dates after filtering.')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH, help='CSV output path for clean replay rows.')
    parser.add_argument('--summary-output', type=str, default=DEFAULT_SUMMARY_PATH, help='JSON summary output path.')
    parser.add_argument('--rebuild', action='store_true', help='Ignore any existing clean replay log and rebuild from scratch.')
    return parser.parse_args()


def parse_date(text):
    return None if not text else dt.date.fromisoformat(str(text).strip())


def replay_dates_from_legacy_log(start_date=None, end_date=None, max_dates=None):
    legacy = m.load_backtest_log()
    if legacy.empty or 'date' not in legacy.columns:
        return []
    dates = sorted(pd.to_datetime(legacy['date'], errors='coerce').dropna().dt.date.unique())
    if start_date is not None:
        dates = [value for value in dates if value >= start_date]
    if end_date is not None:
        dates = [value for value in dates if value <= end_date]
    if max_dates is not None:
        dates = dates[-int(max_dates):]
    return dates


def prepare_statcast_frame(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=[])
    work = df.copy()
    work['game_date'] = pd.to_datetime(work.get('game_date'), errors='coerce').dt.date
    work = work[work['game_date'].notna()].copy()
    return work


def slice_statcast_frame(df, start_date, end_date_exclusive):
    if df is None or df.empty:
        return pd.DataFrame(columns=(list(df.columns) if df is not None else []))
    work = df[(df['game_date'] >= start_date) & (df['game_date'] < end_date_exclusive)].copy()
    return work.reset_index(drop=True)


def fetch_final_games_for_date(client, target_date):
    games, warnings = m.fetch_schedule_games(client, target_date)
    payload = m.schedule_for_date(client, target_date)
    final_scores = {}
    for day in payload.get('dates', []):
        for game in day.get('games', []):
            abstract_state = str(((game.get('status') or {}).get('abstractGameState')) or '')
            if abstract_state != 'Final':
                continue
            away = (game.get('teams') or {}).get('away') or {}
            home = (game.get('teams') or {}).get('home') or {}
            final_scores[int(game['gamePk'])] = {
                'away_score': int(m.finite(away.get('score'), 0.0)),
                'home_score': int(m.finite(home.get('score'), 0.0)),
            }
    finals = []
    for game in games:
        result = final_scores.get(int(game['game_pk']))
        if not result:
            continue
        row = dict(game)
        row.update(result)
        finals.append(row)
    return finals, warnings


def build_feature_tables_for_date(client, report_date, phase, batter_raw_by_phase, pitcher_raw_by_phase, prior_offense, prior_offense_lhp, prior_offense_rhp, prior_defense, prior_pitchers):
    season_start = dt.date(report_date.year, 1, 1)
    batter_raw = batter_raw_by_phase.get(phase)
    pitcher_raw = pitcher_raw_by_phase.get(phase)

    current_batter_raw = slice_statcast_frame(batter_raw, season_start, report_date)
    batter_raw_30 = slice_statcast_frame(batter_raw, report_date - dt.timedelta(days=30), report_date)
    batter_raw_14 = slice_statcast_frame(batter_raw, report_date - dt.timedelta(days=14), report_date)
    batter_raw_7 = slice_statcast_frame(batter_raw, report_date - dt.timedelta(days=7), report_date)

    strengths = m.blend_team_offense(
        m.aggregate_team_offense(current_batter_raw),
        m.aggregate_team_offense(batter_raw_30),
        m.aggregate_team_offense(batter_raw_14),
        m.aggregate_team_offense(batter_raw_7),
        prior_offense,
    )
    strengths = m.merge_handed_offense_splits(
        strengths,
        m.blend_team_offense(
            m.aggregate_team_offense(current_batter_raw, pitcher_hand='L'),
            m.aggregate_team_offense(batter_raw_30, pitcher_hand='L'),
            m.aggregate_team_offense(batter_raw_14, pitcher_hand='L'),
            m.aggregate_team_offense(batter_raw_7, pitcher_hand='L'),
            prior_offense_lhp,
        ),
        'vs_lhp',
    )
    strengths = m.merge_handed_offense_splits(
        strengths,
        m.blend_team_offense(
            m.aggregate_team_offense(current_batter_raw, pitcher_hand='R'),
            m.aggregate_team_offense(batter_raw_30, pitcher_hand='R'),
            m.aggregate_team_offense(batter_raw_14, pitcher_hand='R'),
            m.aggregate_team_offense(batter_raw_7, pitcher_hand='R'),
            prior_offense_rhp,
        ),
        'vs_rhp',
    )

    defense_strengths = m.blend_team_defense(
        m.aggregate_team_defense(current_batter_raw),
        m.aggregate_team_defense(batter_raw_30),
        m.aggregate_team_defense(batter_raw_14),
        m.aggregate_team_defense(batter_raw_7),
        prior_defense,
    )
    strengths = strengths.merge(defense_strengths, on='team', how='left')

    current_pitchers = m.aggregate_pitcher_quality(slice_statcast_frame(pitcher_raw, season_start, report_date))
    bullpen = m.bullpen_snapshot(client, report_date, 7, current_pitchers, prior_pitchers)
    bullpen_map = {str(row['team']): m.finite(row['bullpen_score'], 0.0) for _, row in bullpen.iterrows()} if not bullpen.empty else {}
    return {
        'strengths': strengths,
        'current_pitchers': current_pitchers,
        'prior_pitchers': prior_pitchers,
        'bullpen': bullpen,
        'bullpen_map': bullpen_map,
    }


def prediction_row(report_date, phase, game, prediction, weights):
    features = prediction.get('features', {}) or {}
    p_home_raw = float(m.finite(prediction.get('p_home_raw'), 0.5))
    p_home_raw = min(max(p_home_raw, 1e-6), 1.0 - 1e-6)
    return {
        'date': report_date.isoformat(),
        'phase': phase,
        'game_pk': int(game.get('game_pk') or 0),
        'away': str(game['away_team']),
        'home': str(game['home_team']),
        'venue_id': game.get('venue_id'),
        'venue': str(game.get('venue_name') or ''),
        'away_pitcher': str(game.get('away_pitcher') or ''),
        'home_pitcher': str(game.get('home_pitcher') or ''),
        'away_score': int(game.get('away_score') or 0),
        'home_score': int(game.get('home_score') or 0),
        'y_home': 1 if int(game.get('home_score') or 0) > int(game.get('away_score') or 0) else 0,
        'p_home': float(prediction.get('p_home', 0.5)),
        'pred_home': 1 if float(prediction.get('p_home', 0.5)) >= 0.5 else 0,
        'projected_total': float(prediction.get('total_calibrated', 0.0)),
        'away_expected_runs': float(prediction.get('away_runs', 0.0)),
        'home_expected_runs': float(prediction.get('home_runs', 0.0)),
        'p_home_raw': p_home_raw,
        'logit': float(math.log(p_home_raw / (1.0 - p_home_raw))),
        'projected_margin': float(prediction.get('margin_calibrated', 0.0)),
        'x_off': float(m.finite(features.get('x_off'), 0.0)),
        'x_starter': float(m.finite(features.get('x_starter'), 0.0)),
        'x_bullpen': float(m.finite(features.get('x_bullpen'), 0.0)),
        'x_park': float(m.finite(features.get('x_park'), 0.0)),
        'x_weather': float(m.finite(features.get('x_weather'), 0.0)),
        'x_context': float(m.finite(features.get('x_context'), 0.0)),
        'W_OFF': float(weights['w_off']),
        'W_ST': float(weights['w_starter']),
        'W_BP': float(weights['w_bullpen']),
        'W_PK': float(weights['w_park']),
        'W_WX': float(weights['w_weather']),
        'W_CTX': float(weights['w_context']),
        'HFA': float(weights['home_field']),
    }


def replay_clean_backtest(dates, output_path, summary_path, rebuild=False):
    if not dates:
        raise RuntimeError('No replay dates were available.')

    client = m.RequestClient()
    state = m.load_model_state()
    park_cache = m.load_park_cache()
    venue_cache = m.load_venue_metadata_cache()
    baseline_validation = m.load_validation_log()
    margin_sigma = m.estimate_margin_sigma(baseline_validation)

    replay_year = max(dates).year
    max_date = max(dates)
    spring_batter_raw = prepare_statcast_frame(m.savant_statcast_csv(client, 'batter', replay_year, dt.date(replay_year, 1, 1), max_date, 'S|'))
    spring_pitcher_raw = prepare_statcast_frame(m.savant_statcast_csv(client, 'pitcher', replay_year, dt.date(replay_year, 1, 1), max_date, 'S|'))
    regular_batter_raw = prepare_statcast_frame(m.savant_statcast_csv(client, 'batter', replay_year, dt.date(replay_year, 1, 1), max_date, 'R|'))
    regular_pitcher_raw = prepare_statcast_frame(m.savant_statcast_csv(client, 'pitcher', replay_year, dt.date(replay_year, 1, 1), max_date, 'R|'))
    batter_raw_by_phase = {'spring': spring_batter_raw, 'regular': regular_batter_raw}
    pitcher_raw_by_phase = {'spring': spring_pitcher_raw, 'regular': regular_pitcher_raw}

    prior_offense = m.previous_regular_season_team_offense(client, replay_year - 1)
    prior_offense_lhp = m.previous_regular_season_team_offense(client, replay_year - 1, pitcher_hand='L')
    prior_offense_rhp = m.previous_regular_season_team_offense(client, replay_year - 1, pitcher_hand='R')
    prior_defense = m.previous_regular_season_team_defense(client, replay_year - 1)
    prior_pitchers = m.previous_regular_season_pitchers(client, replay_year - 1)

    existing = pd.DataFrame()
    if (not rebuild) and os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path)
        except Exception:
            existing = pd.DataFrame()
    existing_dates = set(pd.to_datetime(existing.get('date'), errors='coerce').dropna().dt.date.unique()) if not existing.empty and 'date' in existing.columns else set()

    rows = [] if existing.empty else existing.to_dict('records')
    replayed_dates = []
    for index, report_date in enumerate(dates, start=1):
        if report_date in existing_dates:
            print(f'SKIP {report_date.isoformat()} already present in clean replay log')
            continue
        games, warnings = fetch_final_games_for_date(client, report_date)
        if warnings:
            print(f'WARN {report_date.isoformat()} ' + ' | '.join(str(item) for item in warnings[:3]))
        if not games:
            print(f'SKIP {report_date.isoformat()} no final games found')
            continue
        phase = m.season_phase_for_games(games)
        weights = m.phase_weights_from_state(state, phase)
        tables = build_feature_tables_for_date(
            client,
            report_date,
            phase,
            batter_raw_by_phase,
            pitcher_raw_by_phase,
            prior_offense,
            prior_offense_lhp,
            prior_offense_rhp,
            prior_defense,
            prior_pitchers,
        )
        for game in games:
            prediction = m.predict_game(
                game,
                phase,
                weights,
                state,
                tables['strengths'],
                tables['current_pitchers'],
                tables['prior_pitchers'],
                tables['bullpen_map'],
                park_cache,
                client,
                venue_cache,
                margin_sigma,
                lineup_counts_override={'away': 0, 'home': 0},
            )
            rows.append(prediction_row(report_date, phase, game, prediction, weights))
        replayed_dates.append(report_date.isoformat())
        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values(['date', 'away', 'home']).reset_index(drop=True)
        out_df.to_csv(output_path, index=False)
        print(f'REPLAY {index}/{len(dates)} {report_date.isoformat()} games={len(games)} rows={len(out_df)}')

    m.save_venue_metadata_cache(venue_cache)
    final_df = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame(rows)
    summary = m.build_validation_summary(final_df)
    summary['replayed_dates'] = replayed_dates
    summary['output_path'] = output_path
    summary['source'] = 'clean_backtest_runner'
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return final_df, summary


def main():
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    dates = replay_dates_from_legacy_log(start_date=start_date, end_date=end_date, max_dates=args.max_dates)
    final_df, summary = replay_clean_backtest(dates, args.output, args.summary_output, rebuild=args.rebuild)
    print('STATUS clean walk-forward replay built')
    print('OUTPUT', args.output)
    print('SUMMARY', args.summary_output)
    print('GAMES', int(summary.get('games', 0) or 0))
    print('OOS_GAMES', int(summary.get('oos_games', 0) or 0))
    print('DATE_SPAN', str(summary.get('date_span', 'N/A')))
    print('OOS_SPAN', str(summary.get('oos_span', 'N/A')))
    print('LOGLOSS', 'N/A' if summary.get('log_loss') is None else round(float(summary['log_loss']), 4))
    print('ACCURACY', 'N/A' if summary.get('accuracy') is None else round(float(summary['accuracy']), 4))
    print('TOTAL_MAE', 'N/A' if summary.get('total_mae') is None else round(float(summary['total_mae']), 3))
    print('TOTAL_RMSE', 'N/A' if summary.get('total_rmse') is None else round(float(summary['total_rmse']), 3))
    print('MARGIN_MAE', 'N/A' if summary.get('margin_mae') is None else round(float(summary['margin_mae']), 3))
    print('MARGIN_RMSE', 'N/A' if summary.get('margin_rmse') is None else round(float(summary['margin_rmse']), 3))


if __name__ == '__main__':
    main()

