import pandas as pd

df = pd.read_csv('clean_backtest_log.csv')
subset = df[(df['date']=='2026-02-21') & (df['away']=='HOU') & (df['home']=='WSH')]
cols = ['date','away','home','projected_total','projected_total_raw','projected_total_model','projected_total_simple']
print(subset[cols].to_string(index=False))
