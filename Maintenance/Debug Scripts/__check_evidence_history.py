import duckdb
con = duckdb.connect('mlb_quant_dashboard.duckdb', read_only=True)
rows = con.execute("select market_type, as_of_report_date, evidence_strength, used_real_market, current_slates, current_picks, medium_threshold_label, high_threshold_label from policy_research_market_type_recommendations order by as_of_report_date desc, run_id desc, market_type limit 12").fetchall()
print(rows)
