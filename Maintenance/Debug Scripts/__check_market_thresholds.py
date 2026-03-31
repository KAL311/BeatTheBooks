import duckdb
con = duckdb.connect('mlb_quant_dashboard.duckdb', read_only=True)
print('latest_run', con.execute("select report_date, created_ts, market_policy_moneyline_policy, market_policy_total_policy, market_policy_run_line_policy from quant_runs order by created_ts desc limit 1").fetchall())
print('market_type_cols', con.execute("select * from policy_research_market_type_recommendations order by as_of_report_date desc, run_id desc limit 1").df().columns.tolist())
print('market_type_rows', con.execute("select market_type, recommended_policy, recommended_posture, evidence_source, used_real_market, current_slates, current_picks, medium_threshold_label, high_threshold_label, primary_ready, high_ready from policy_research_market_type_recommendations order by as_of_report_date desc, run_id desc, market_type limit 6").fetchall())
