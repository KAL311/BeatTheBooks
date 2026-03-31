import pandas as pd
import numpy as np

df = pd.read_csv('clean_backtest_log.csv')
df['actual_total'] = pd.to_numeric(df['home_score'], errors='coerce') + pd.to_numeric(df['away_score'], errors='coerce')
df['projected_total'] = pd.to_numeric(df['projected_total'], errors='coerce')
df['projected_total_sigma'] = pd.to_numeric(df['projected_total_sigma'], errors='coerce')
df['z'] = (df['actual_total'] - df['projected_total']) / df['projected_total_sigma']
oos = df.tail(59).copy()
for name, frame in [('all', df), ('oos', oos)]:
    valid = frame.dropna(subset=['z']).copy()
    print('\n', name.upper())
    print('games', len(valid))
    print('z_mean', round(valid['z'].mean(), 3))
    print('z_std', round(valid['z'].std(ddof=0), 3))
    print('mae_sigma_ratio', round((valid['actual_total'] - valid['projected_total']).abs().mean() / valid['projected_total_sigma'].mean(), 3))
    by_phase = valid.groupby('phase').agg(games=('date','count'), z_mean=('z','mean'), z_std=('z','std'), sigma_mean=('projected_total_sigma','mean'))
    print(by_phase.round(3).to_string())
