import clean_backtest_runner as c
import run_daily_mlb_report as m
state = m.load_model_state()
print('replay_simple_total', m.simple_total_baseline_from_phase('spring', state))
print('replay_tc_apply', ((state.get('total_calibration') or {}).get('spring') or {}).get('apply'))
print('replay_ts_apply', ((state.get('total_shrinkage') or {}).get('spring') or {}).get('apply'))
