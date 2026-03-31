import pandas as pd
import json
import run_daily_mlb_report as m

state = m.load_model_state()
df = pd.read_csv('clean_backtest_log.csv')
new_state, cal = m.fit_total_calibration_profile(state, df)
new_state, shr = m.fit_total_shrinkage_profile(new_state, df)
print(json.dumps({'calibration': cal, 'shrinkage': shr}, indent=2, default=str))
