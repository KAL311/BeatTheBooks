import pandas as pd
import numpy as np

df = pd.read_csv('clean_backtest_log.csv')
df['actual_total'] = pd.to_numeric(df['home_score'], errors='coerce') + pd.to_numeric(df['away_score'], errors='coerce')
df['projected_total'] = pd.to_numeric(df['projected_total'], errors='coerce')
df['projected_total_raw'] = pd.to_numeric(df['projected_total_raw'], errors='coerce')
df['error'] = df['actual_total'] - df['projected_total']
df['raw_error'] = df['actual_total'] - df['projected_total_raw']
oos = df.tail(59).copy()
print('OVERALL_OOS', {'mae': round((oos['error'].abs()).mean(),3), 'rmse': round(np.sqrt((oos['error']**2).mean()),3), 'bias': round(oos['error'].mean(),3), 'raw_bias': round(oos['raw_error'].mean(),3)})
for col in ['projected_total','projected_total_raw']:
    bands = pd.cut(oos[col], bins=[0,8.5,9.5,10.5,99], labels=['<=8.5','8.5-9.5','9.5-10.5','10.5+'])
    grp = oos.groupby(bands, observed=False).agg(games=('date','count'), avg_proj=(col,'mean'), avg_actual=('actual_total','mean'), bias=('error','mean'), mae=('error', lambda s: s.abs().mean()))
    print('\nBANDS', col)
    print(grp.round(3).to_string())
