import pandas as pd
import json
from pathlib import Path
state = json.loads(Path('model_state.json').read_text(encoding='utf-8'))
print('spring_total_calibration_apply', state['total_calibration']['spring']['apply'])
print('spring_total_shrinkage_apply', state['total_shrinkage']['spring']['apply'])
df = pd.read_csv('clean_backtest_log.csv')
oos = df.tail(59).copy()
actual = pd.to_numeric(oos['home_score'], errors='coerce') + pd.to_numeric(oos['away_score'], errors='coerce')
proj = pd.to_numeric(oos['projected_total'], errors='coerce')
err = actual - proj
print('oos_bias', round(err.mean(), 3))
print('oos_mae', round(err.abs().mean(), 3))
