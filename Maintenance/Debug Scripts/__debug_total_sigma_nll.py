import pandas as pd
import numpy as np

df = pd.read_csv('clean_backtest_log.csv').tail(59).copy()
actual = pd.to_numeric(df['home_score'], errors='coerce') + pd.to_numeric(df['away_score'], errors='coerce')
pred = pd.to_numeric(df['projected_total'], errors='coerce')
sigma = pd.to_numeric(df['projected_total_sigma'], errors='coerce')
valid = (~actual.isna()) & (~pred.isna()) & (~sigma.isna()) & (sigma > 0)
actual = actual[valid].astype(float).to_numpy()
pred = pred[valid].astype(float).to_numpy()
sigma = sigma[valid].astype(float).to_numpy()

def nll(scale):
    s = np.clip(sigma * scale, 1e-6, None)
    resid = actual - pred
    return float(np.mean(0.5*np.log(2*np.pi*(s**2)) + 0.5*((resid/s)**2)))

best = min([(round(scale,3), nll(scale)) for scale in np.linspace(0.72, 1.12, 41)], key=lambda t: t[1])
raw = nll(1.0)
print('raw_nll', raw)
print('best', best)
print('improvement', raw - best[1])
print('rms_z', float(np.sqrt(np.mean(((actual-pred)/sigma)**2))))
