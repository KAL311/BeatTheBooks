import sqlite3
conn = sqlite3.connect(r'C:\Users\fraun\OneDrive\Documents\2026 MLB Report\sportsbook_lines.db')
cur = conn.cursor()
for row in cur.execute("PRAGMA table_info(bet_journal)"):
    print(row)
print('ROWS', cur.execute('SELECT COUNT(*) FROM bet_journal').fetchone()[0])
for row in cur.execute('SELECT * FROM bet_journal LIMIT 3'):
    print(row)
conn.close()
