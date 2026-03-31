import json
import shutil
import pandas as pd
import run_daily_mlb_report as m

state_path = 'model_state.json'
backup_path = 'model_state.json.pre_totals_refresh.bak'
shutil.copyfile(state_path, backup_path)
state = m.load_model_state()
df = pd.read_csv('clean_backtest_log.csv')
state, _ = m.fit_total_calibration_profile(state, df)
state, _ = m.fit_total_shrinkage_profile(state, df)
with open(state_path, 'w', encoding='utf-8') as handle:
    json.dump(state, handle, indent=2)
print('updated', state_path)
print('backup', backup_path)
print(json.dumps({
    'total_calibration': state.get('total_calibration'),
    'total_shrinkage': state.get('total_shrinkage'),
}, indent=2))
