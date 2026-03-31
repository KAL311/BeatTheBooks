import pandas as pd

df = pd.read_csv('clean_backtest_log.csv')
print('has_sigma', 'projected_total_sigma' in df.columns)
print('sigma_range', round(pd.to_numeric(df['projected_total_sigma'], errors='coerce').min(), 3), round(pd.to_numeric(df['projected_total_sigma'], errors='coerce').max(), 3))
print(df[['date','away','home','projected_total','projected_total_sigma']].head(3).to_string(index=False))
