"""
setup_tigergraph.py
───────────────────
Run ONCE to initialize TigerGraph schema + queries.
After this, terrain_graph.py handles all uploads.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from terrain_graph import get_connection, setup_schema

if __name__ == "__main__":
    print("Connecting to TigerGraph...")
    conn = get_connection()
    print(f"Host:  {os.getenv('TIGERGRAPH_HOST')}")
    print(f"Graph: {os.getenv('TIGERGRAPH_GRAPH_NAME')}")
    print()

    print("Setting up schema...")
    setup_schema(conn)

    print("\n✅ TigerGraph setup complete!")
    print("Now run: python app.py")