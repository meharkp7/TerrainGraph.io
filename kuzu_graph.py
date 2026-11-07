"""
kuzu_graph.py
─────────────
Kuzu embedded graph backend — free forever, no server, no credits.

Same public API as terrain_graph.py (TigerGraph):
  get_connection() / setup_schema() / upload_terrain() /
  get_risk_zones() / find_similar_terrains()

Local graph building (patches, edges, Dijkstra) is shared via
terrain_graph.build_terrain_local; only persistence + queries differ.

Storage is a local directory (KUZU_DB_PATH, default ./data/terrain_kuzu).
On Railway's ephemeral filesystem the history resets on redeploy unless a
volume is mounted at that path — the pipeline works either way.

kuzu is imported lazily inside get_connection() so this module stays
importable on hosts without the driver (e.g. local Python 3.14, which has
no macOS wheel); pipeline.py auto-selects the backend by driver presence.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from terrain_graph import PATCH_SIZE, build_terrain_local

DB_PATH = os.getenv("KUZU_DB_PATH", "./data/terrain_kuzu")

# Container-safe limits. Kuzu defaults its buffer pool to ~80% of HOST RAM
# and thread count to host CPUs — it cannot see container cgroup limits, so
# on a 512MB instance the defaults get the process OOM-killed on open.
# Our data is tiny (hundreds of nodes/image), so cap aggressively.
BUFFER_POOL_MB = int(os.getenv("KUZU_BUFFER_POOL_MB", "128"))
MAX_THREADS = int(os.getenv("KUZU_MAX_THREADS", "2"))


# ─────────────────────────────────────────────
# SCHEMA  (same shape as the TigerGraph one)
# ─────────────────────────────────────────────

SCHEMA_STATEMENTS = [
    """CREATE NODE TABLE IF NOT EXISTS ImageRecord(
        image_id STRING, timestamp STRING, image_path STRING,
        dominant_class STRING, avg_traversability DOUBLE,
        brightness DOUBLE, rock_pct DOUBLE, vegetation_pct DOUBLE,
        sky_pct DOUBLE, miou_achieved DOUBLE, model_used STRING,
        PRIMARY KEY (image_id))""",
    """CREATE NODE TABLE IF NOT EXISTS TerrainPatch(
        patch_id STRING, image_id STRING,
        rowV INT64, colV INT64, dominant_class INT64,
        class_name STRING, traversability DOUBLE, risk_level STRING,
        confidence DOUBLE, class_distribution STRING,
        PRIMARY KEY (patch_id))""",
    """CREATE NODE TABLE IF NOT EXISTS RiskZone(
        zone_id STRING, image_id STRING, severity STRING,
        class_name STRING, patch_count INT64,
        center_row INT64, center_col INT64,
        PRIMARY KEY (zone_id))""",
    """CREATE NODE TABLE IF NOT EXISTS SafePath(
        path_id STRING, image_id STRING, total_cost DOUBLE,
        avg_traversability DOUBLE, hop_count INT64,
        patch_sequence STRING,
        PRIMARY KEY (path_id))""",
    """CREATE REL TABLE IF NOT EXISTS AdjacentTo(
        FROM TerrainPatch TO TerrainPatch,
        transition_cost DOUBLE, direction STRING)""",
    """CREATE REL TABLE IF NOT EXISTS BelongsTo(
        FROM TerrainPatch TO ImageRecord)""",
    """CREATE REL TABLE IF NOT EXISTS HasRiskZone(
        FROM ImageRecord TO RiskZone)""",
    """CREATE REL TABLE IF NOT EXISTS HasPath(
        FROM ImageRecord TO SafePath)""",
]


# ─────────────────────────────────────────────
# CONNECTION (singleton — Kuzu connections are cheap to keep)
# ─────────────────────────────────────────────

_db = None
_conn = None


def get_connection():
    """Open (and create if needed) the embedded DB; ensure schema."""
    global _db, _conn
    if _conn is None:
        import kuzu  # lazy: clear error instead of ImportError at import
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db = kuzu.Database(
            DB_PATH,
            buffer_pool_size=BUFFER_POOL_MB * 1024 * 1024,
            max_num_threads=MAX_THREADS)
        _conn = kuzu.Connection(_db)
        setup_schema(_conn)
    return _conn


def setup_schema(conn):
    """Create node/rel tables. Idempotent — safe to run on every boot."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    print(f"Kuzu schema ready at {DB_PATH}")


def _rows(result):
    """Drain a Kuzu QueryResult into a list (guarded: empty results raise)."""
    out = []
    while result.has_next():
        out.append(result.get_next())
    return out


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

def upload_terrain(conn,
                   image_id: str,
                   image_path: str,
                   seg_result: dict,
                   model_miou: float = 0.0,
                   patch_size: int = PATCH_SIZE) -> dict:
    """
    Full pipeline: mask → patches → persist to Kuzu.
    Same return shape as terrain_graph.upload_terrain.
    MERGE everywhere, so re-uploading an image_id is idempotent.
    """
    print(f"\n[Kuzu] Uploading terrain graph: {image_id}")
    local = build_terrain_local(
        image_id, image_path, seg_result, model_miou, patch_size)
    patches     = local["patch_list"]
    edges       = local["edge_list"]
    zones       = local["zone_list"]
    path_result = local["path"]
    class_dist  = local["class_dist"]
    bright      = local["brightness"]
    avg_trav    = local["avg_trav"]

    try:
        # 1. ImageRecord
        conn.execute(
            """MERGE (i:ImageRecord {image_id: $image_id})
               ON CREATE SET i.timestamp = $ts, i.image_path = $path,
                 i.dominant_class = $dom, i.avg_traversability = $trav,
                 i.brightness = $bright, i.rock_pct = $rock,
                 i.vegetation_pct = $veg, i.sky_pct = $sky,
                 i.miou_achieved = $miou, i.model_used = $model
               ON MATCH SET i.timestamp = $ts, i.image_path = $path,
                 i.dominant_class = $dom, i.avg_traversability = $trav,
                 i.brightness = $bright, i.rock_pct = $rock,
                 i.vegetation_pct = $veg, i.sky_pct = $sky,
                 i.miou_achieved = $miou, i.model_used = $model""",
            {"image_id": image_id,
             "ts":       time.strftime("%Y-%m-%dT%H:%M:%S"),
             "path":     str(image_path),
             "dom":      max(class_dist, key=class_dist.get),
             "trav":     float(avg_trav),
             "bright":   float(bright),
             "rock":     float(class_dist.get("Rocks", 0)),
             "veg":      float(class_dist.get("Dense_Vegetation", 0) +
                               class_dist.get("Dry_Vegetation", 0)),
             "sky":      float(class_dist.get("Sky", 0)),
             "miou":     float(model_miou),
             "model":    "DeepLabV3+"})

        # 2. TerrainPatch vertices (batched)
        conn.execute(
            """UNWIND $batch AS p
               MERGE (t:TerrainPatch {patch_id: p.patch_id})
               ON CREATE SET t.image_id = p.image_id, t.rowV = p.rowV,
                 t.colV = p.colV, t.dominant_class = p.dom,
                 t.class_name = p.cls, t.traversability = p.trav,
                 t.risk_level = p.risk, t.confidence = p.conf,
                 t.class_distribution = p.dist
               ON MATCH SET t.image_id = p.image_id, t.rowV = p.rowV,
                 t.colV = p.colV, t.dominant_class = p.dom,
                 t.class_name = p.cls, t.traversability = p.trav,
                 t.risk_level = p.risk, t.confidence = p.conf,
                 t.class_distribution = p.dist""",
            {"batch": [
                {"patch_id": p["patch_id"], "image_id": p["image_id"],
                 "rowV": int(p["rowV"]), "colV": int(p["colV"]),
                 "dom": int(p["dominant_class"]), "cls": p["class_name"],
                 "trav": float(p["traversability"]), "risk": p["risk_level"],
                 "conf": float(p["confidence"]),
                 "dist": p["class_distribution"]}
                for p in patches]})
        print(f"     ✅ {len(patches)} patch vertices")

        # 3. AdjacentTo edges (batched)
        conn.execute(
            """UNWIND $edges AS e
               MATCH (a:TerrainPatch {patch_id: e.from_id}),
                     (b:TerrainPatch {patch_id: e.to_id})
               MERGE (a)-[r:AdjacentTo]->(b)
               ON CREATE SET r.transition_cost = e.cost,
                 r.direction = e.dir""",
            {"edges": [
                {"from_id": e["from"], "to_id": e["to"],
                 "cost": float(e["transition_cost"]), "dir": e["direction"]}
                for e in edges]})
        print(f"     ✅ {len(edges)} adjacency edges")

        # 4. BelongsTo edges (batched)
        conn.execute(
            """UNWIND $ids AS pid
               MATCH (t:TerrainPatch {patch_id: pid}),
                     (i:ImageRecord {image_id: $image_id})
               MERGE (t)-[:BelongsTo]->(i)""",
            {"ids": [p["patch_id"] for p in patches],
             "image_id": image_id})

        # 5. RiskZone vertices + HasRiskZone edges
        for z in zones:
            conn.execute(
                """MERGE (r:RiskZone {zone_id: $zone_id})
                   ON CREATE SET r.image_id = $image_id,
                     r.severity = $sev, r.class_name = $cls,
                     r.patch_count = $cnt, r.center_row = $cr,
                     r.center_col = $cc
                   ON MATCH SET r.image_id = $image_id,
                     r.severity = $sev, r.class_name = $cls,
                     r.patch_count = $cnt, r.center_row = $cr,
                     r.center_col = $cc""",
                {"zone_id": z["zone_id"], "image_id": z["image_id"],
                 "sev": z["severity"], "cls": z["class_name"],
                 "cnt": int(z["patch_count"]), "cr": int(z["center_row"]),
                 "cc": int(z["center_col"])})
            conn.execute(
                """MATCH (i:ImageRecord {image_id: $image_id}),
                         (r:RiskZone {zone_id: $zone_id})
                   MERGE (i)-[:HasRiskZone]->(r)""",
                {"image_id": image_id, "zone_id": z["zone_id"]})
        print(f"     ✅ {len(zones)} risk zones")

        # 6. SafePath vertex + HasPath edge
        path_id = f"{image_id}_path"
        conn.execute(
            """MERGE (s:SafePath {path_id: $path_id})
               ON CREATE SET s.image_id = $image_id, s.total_cost = $cost,
                 s.avg_traversability = $trav, s.hop_count = $hops,
                 s.patch_sequence = $seq
               ON MATCH SET s.image_id = $image_id, s.total_cost = $cost,
                 s.avg_traversability = $trav, s.hop_count = $hops,
                 s.patch_sequence = $seq""",
            {"path_id": path_id, "image_id": image_id,
             "cost": float(path_result["total_cost"]),
             "trav": float(avg_trav),
             "hops": int(path_result["hop_count"]),
             "seq": json.dumps(path_result["path_ids"][:50])})
        conn.execute(
            """MATCH (i:ImageRecord {image_id: $image_id}),
                     (s:SafePath {path_id: $path_id})
               MERGE (i)-[:HasPath]->(s)""",
            {"image_id": image_id, "path_id": path_id})
        print("     ✅ Safe path stored")
        print("     ✅ Terrain graph complete!")

    except Exception as e:
        print(f"     ⚠️  Kuzu upload error: {e}")
        print("     (Path result still valid from Python Dijkstra)")

    return {
        "image_id":   image_id,
        "patches":    len(patches),
        "edges":      len(edges),
        "risk_zones": zones,
        "avg_trav":   float(avg_trav),
        "class_dist": class_dist,
        "brightness": float(bright),
        "path":       path_result,  # always present
    }


# ─────────────────────────────────────────────
# QUERIES — same {"attributes": {...}} shape as TigerGraph
# ─────────────────────────────────────────────

def get_risk_zones(conn, image_id: str) -> list:
    if conn is None:
        return []
    try:
        result = conn.execute(
            """MATCH (i:ImageRecord {image_id: $image_id})
                     -[:HasRiskZone]->(r:RiskZone)
               RETURN r.zone_id, r.image_id, r.severity, r.class_name,
                      r.patch_count, r.center_row, r.center_col
               ORDER BY r.patch_count DESC""",
            {"image_id": image_id})
        out = []
        for zid, iid, sev, cls, cnt, cr, cc in _rows(result):
            out.append({"attributes": {
                "zone_id": zid, "image_id": iid, "severity": sev,
                "class_name": cls, "patch_count": int(cnt),
                "center_row": int(cr), "center_col": int(cc)}})
        return out
    except Exception as e:
        print(f"getRiskZones query failed: {e}")
        return []


def find_similar_terrains(conn,
                          rock_pct: float,
                          veg_pct: float,
                          brightness: float,
                          top_k: int = 5) -> list:
    if conn is None:
        return []
    try:
        # LIMIT uses an int literal (params in LIMIT are version-sensitive)
        result = conn.execute(
            f"""MATCH (i:ImageRecord)
               WHERE abs(i.rock_pct - $rock) < 0.08
                 AND abs(i.vegetation_pct - $veg) < 0.10
                 AND abs(i.brightness - $bright) < 0.10
                 AND i.miou_achieved > 0
               RETURN i.image_id, i.dominant_class, i.avg_traversability,
                      i.miou_achieved, i.rock_pct, i.vegetation_pct,
                      i.brightness
               ORDER BY i.miou_achieved DESC
               LIMIT {int(top_k)}""",
            {"rock": float(rock_pct), "veg": float(veg_pct),
             "bright": float(brightness)})
        out = []
        for (iid, dom, trav, miou, rock, veg, bright) in _rows(result):
            out.append({"attributes": {
                "image_id": iid, "dominant_class": dom,
                "avg_traversability": float(trav),
                "miou_achieved": float(miou),
                "rock_pct": float(rock),
                "vegetation_pct": float(veg),
                "brightness": float(bright)}})
        return out
    except Exception as e:
        print(f"findSimilarTerrains query failed: {e}")
        return []


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Kuzu Terrain Graph — Self Test")
    print("=" * 50)

    conn = get_connection()

    n_img = _rows(conn.execute("MATCH (i:ImageRecord) RETURN count(*)"))[0][0]
    n_patch = _rows(conn.execute("MATCH (t:TerrainPatch) RETURN count(*)"))[0][0]
    print(f"ImageRecord count : {n_img}")
    print(f"TerrainPatch count: {n_patch}")
    print("✅ Kuzu connection OK")
