import pandas as pd

df = pd.read_csv('clean_backtest_log.csv')
cols = ['date','away','home','projected_total','projected_total_raw','projected_total_model','projected_total_simple']
print(df.loc[:4, cols].to_string(index=False))
