"""
setup_kuzu.py
─────────────
Initialize the Kuzu embedded graph (creates the DB directory + schema).
Idempotent — the backend also ensures the schema on first connect, so this
is just an explicit way to verify the store works.

Run:  python setup_kuzu.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from kuzu_graph import DB_PATH, get_connection

if __name__ == "__main__":
    print(f"DB path: {DB_PATH}")
    print()
    print("Opening database + ensuring schema...")
    conn = get_connection()

    imgs = conn.execute(
        "MATCH (i:ImageRecord) RETURN count(*)").get_next()[0]
    patches = conn.execute(
        "MATCH (t:TerrainPatch) RETURN count(*)").get_next()[0]
    print(f"ImageRecord count : {imgs}")
    print(f"TerrainPatch count: {patches}")

    print("\n✅ Kuzu setup complete!")
    print("Now run: uvicorn api:app --port 8000")
