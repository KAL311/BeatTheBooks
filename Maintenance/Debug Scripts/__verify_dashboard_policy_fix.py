import pandas as pd
import quant_dashboard as qd

summary = {'daily_budget_units': 1.0}
recs = pd.DataFrame([
    {'selection': 'TEST', 'market_type': 'moneyline', 'allocated_units': 0.5}
])
out = qd._apply_policy_to_recommendations(summary, recs, 'Flat', 'Base', 1000.0, 1.0, 1.0)
policy_view, policy_summary, _, _, _ = out
print('rows', len(policy_view))
print('cols_present', all(col in policy_view.columns for col in ['edge_value','actionability_score','uncertainty_calibration_bias','uncertainty_penalty_scale']))
print('policy_name', policy_summary.get('policy_name'))
