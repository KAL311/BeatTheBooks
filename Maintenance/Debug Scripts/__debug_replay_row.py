import datetime as dt
import clean_backtest_runner as c
import run_daily_mlb_report as m

report_date = dt.date(2026, 2, 21)
client = m.RequestClient()
state = m.load_model_state()
park_cache = m.load_park_cache()
venue_cache = m.load_venue_metadata_cache()
baseline_validation = m.load_validation_log()
margin_sigma = m.estimate_margin_sigma(baseline_validation)
replay_year = report_date.year
max_date = report_date
spring_batter_raw = c.prepare_statcast_frame(m.savant_statcast_csv(client, 'batter', replay_year, dt.date(replay_year, 1, 1), max_date, 'S|'))
spring_pitcher_raw = c.prepare_statcast_frame(m.savant_statcast_csv(client, 'pitcher', replay_year, dt.date(replay_year, 1, 1), max_date, 'S|'))
regular_batter_raw = c.prepare_statcast_frame(m.savant_statcast_csv(client, 'batter', replay_year, dt.date(replay_year, 1, 1), max_date, 'R|'))
regular_pitcher_raw = c.prepare_statcast_frame(m.savant_statcast_csv(client, 'pitcher', replay_year, dt.date(replay_year, 1, 1), max_date, 'R|'))
phase = 'spring'
tables = c.build_feature_tables_for_date(
    client,
    report_date,
    phase,
    {'spring': spring_batter_raw, 'regular': regular_batter_raw},
    {'spring': spring_pitcher_raw, 'regular': regular_pitcher_raw},
    m.previous_regular_season_team_offense(client, replay_year - 1),
    m.previous_regular_season_team_offense(client, replay_year - 1, pitcher_hand='L'),
    m.previous_regular_season_team_offense(client, replay_year - 1, pitcher_hand='R'),
    m.previous_regular_season_team_defense(client, replay_year - 1),
    m.previous_regular_season_pitchers(client, replay_year - 1),
    m.previous_regular_season_batters(client, replay_year - 1),
)
games, warnings = c.fetch_final_games_for_date(client, report_date)
weights = m.phase_weights_from_state(state, phase)
game = games[0]
pred = m.predict_game(
    game,
    phase,
    weights,
    state,
    tables['strengths'],
    tables['current_pitchers'],
    tables['prior_pitchers'],
    tables['current_batters'],
    tables['prior_batters'],
    tables['bullpen_profiles'],
    park_cache,
    client,
    venue_cache,
    margin_sigma,
    lineup_counts_override={'away': 0, 'home': 0},
)
print('warnings', warnings[:2])
print({k: pred.get(k) for k in ['total_raw','total_model','total_simple','total_shrink_alpha','total_calibrated']})
row = c.prediction_row(report_date, phase, game, pred, weights)
print({k: row[k] for k in ['date','away','home','projected_total','projected_total_raw','projected_total_model','projected_total_simple']})
