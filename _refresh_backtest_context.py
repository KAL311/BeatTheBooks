import os

import run_daily_mlb_report as m

state = m.load_model_state()
backtest_df = m.load_validation_log()
summary = m.build_validation_summary(backtest_df)
source = 'clean_backtest_log.csv' if os.path.exists(m.CLEAN_BACKTEST_LOG_PATH) and not m.load_clean_backtest_log().empty else 'backtest_log.csv'

print('STATUS clean validation snapshot')
print(f'SOURCE {source}')
print('NOTE Validation now prefers the clean replay log when it is available; otherwise it falls back to the legacy backtest log.')
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
