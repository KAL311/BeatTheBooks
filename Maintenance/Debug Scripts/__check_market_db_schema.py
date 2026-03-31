import sqlite3
conn=sqlite3.connect('sportsbook_lines.db')
cur=conn.cursor()
for table in ['odds_snapshots','bet_journal']:
    try:
        cols=cur.execute(f'PRAGMA table_info({table})').fetchall()
        print(table, [c[1] for c in cols])
        count=cur.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print('rows', count)
    except Exception as e:
        print(table, 'ERR', e)
