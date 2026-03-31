import duckdb
from pathlib import Path
p = Path('mlb_quant_dashboard.duckdb')
con = duckdb.connect(str(p), read_only=True)
for table in [
    'policy_research_market_policy_daily',
    'policy_research_market_policy_summary',
    'policy_research_recommendations',
    'policy_research_window_rankings',
]:
    try:
        rows = con.execute(f"select count(*) from {table}").fetchone()[0]
        print(table, rows)
        if 'market_policy' in table:
            cols = con.execute(f"describe {table}").fetchall()
            print('cols', [c[0] for c in cols])
    except Exception as exc:
        print(table, 'ERROR', exc)
