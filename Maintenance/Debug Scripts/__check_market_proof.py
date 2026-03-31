import duckdb
con = duckdb.connect('mlb_quant_dashboard.duckdb', read_only=True)
print('market_proof_summary', con.execute('select count(*) from market_proof_summary').fetchall())
print('market_proof_by_market', con.execute('select count(*) from market_proof_by_market').fetchall())
print('latest_run_totals', con.execute('select report_date, market_proof_total_bets, market_proof_total_units, market_proof_total_roi, market_proof_total_avg_clv from quant_runs order by created_ts desc limit 1').fetchall())
