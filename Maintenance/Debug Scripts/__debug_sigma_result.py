import pandas as pd
import numpy as np

df = pd.read_csv('clean_backtest_log.csv').tail(59).copy()
actual = pd.to_numeric(df['home_score'], errors='coerce') + pd.to_numeric(df['away_score'], errors='coerce')
pred = pd.to_numeric(df['projected_total'], errors='coerce')
sigma = pd.to_numeric(df['projected_total_sigma'], errors='coerce')
raw_sigma = pd.to_numeric(df['projected_total_sigma_raw'], errors='coerce')
valid = (~actual.isna()) & (~pred.isna()) & (~sigma.isna()) & (~raw_sigma.isna()) & (sigma > 0) & (raw_sigma > 0)
actual = actual[valid].astype(float)
pred = pred[valid].astype(float)
sigma = sigma[valid].astype(float)
raw_sigma = raw_sigma[valid].astype(float)
print('z_std_raw', round(float(np.std((actual - pred) / raw_sigma)), 3))
print('z_std_cal', round(float(np.std((actual - pred) / sigma)), 3))
print('sigma_scale_mean', round(float((sigma / raw_sigma).mean()), 3))
