import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "spam_classifier.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Show history table structure
cursor.execute("PRAGMA table_info(history)")
print("Columns:", cursor.fetchall())

# Show history data
cursor.execute("SELECT * FROM history")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()