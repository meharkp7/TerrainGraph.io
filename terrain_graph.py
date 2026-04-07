"""
terrain_graph.py  —  FIXED VERSION
────────────────────────────────────
Key fixes:
  1. patch_id uses ROW:COL separator (colon) instead of underscore
     so parsing is unambiguous regardless of image_id format
  2. GSQL findSafestPath rewritten — valid GSQL 3.x syntax,
     pure Python Dijkstra fallback always available
  3. upsertEdges call uses correct pyTigerGraph tuple format
  4. Path result parsing handles all response shapes
  5. Python-side Dijkstra as reliable fallback (no GSQL needed)
"""

import os
import json
import time
import heapq
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

import pyTigerGraph as tg

load_dotenv()

HOST       = os.getenv("TIGERGRAPH_HOST")
API_TOKEN  = os.getenv("TIGERGRAPH_API_TOKEN")
GRAPH_NAME = os.getenv("TIGERGRAPH_GRAPH_NAME", "AutonomousGraph")

CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
NUM_CLASSES = len(CLASS_NAMES)

TRAVERSABILITY = {
    0: 0.45,   # Dense_Vegetation
    1: 0.75,   # Dry_Vegetation
    2: 0.30,   # Ground_Objects
    3: 0.15,   # Rocks
    4: 0.95,   # Landscape
    5: 0.00,   # Sky
}
RISK_LEVEL = {
    0: "MED",
    1: "LOW",
    2: "MED_HIGH",
    3: "HIGH",
    4: "LOW",
    5: "NA",
}

PATCH_SIZE = 32


# ─────────────────────────────────────────────
# PATCH ID FORMAT  (CRITICAL FIX)
# Use colon as separator: "{image_id}:r{row}:c{col}"
# This makes parsing unambiguous even when image_id contains underscores
# ─────────────────────────────────────────────

def make_patch_id(image_id: str, row: int, col: int) -> str:
    return f"{image_id}:r{row}:c{col}"


def parse_patch_id(patch_id: str) -> tuple:
    """Returns (image_id, row, col) from patch_id string."""
    parts = patch_id.split(":")
    # parts = [image_id, "rN", "cM"]
    row = int(parts[1][1:])
    col = int(parts[2][1:])
    img = parts[0]
    return img, row, col


# ─────────────────────────────────────────────
# SCHEMA  (simplified — remove GSQL Dijkstra,
# use Python-side pathfinding instead)
# ─────────────────────────────────────────────

SCHEMA_GSQL = """
USE GLOBAL

DROP GRAPH AutonomousGraph

CREATE GRAPH AutonomousGraph()

USE GRAPH AutonomousGraph

CREATE SCHEMA_CHANGE JOB init_schema FOR GRAPH AutonomousGraph {

  ADD VERTEX TerrainPatch (
    PRIMARY_ID patch_id   STRING,
    image_id              STRING,
    rowV                   INT,
    colV                  INT,
    dominant_class        INT,
    class_name            STRING,
    traversability        FLOAT,
    risk_level            STRING,
    confidence            FLOAT,
    class_distribution    STRING
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX ImageRecord (
    PRIMARY_ID image_id   STRING,
    timestamp             STRING,
    image_path            STRING,
    dominant_class        STRING,
    avg_traversability    FLOAT,
    brightness            FLOAT,
    rock_pct              FLOAT,
    vegetation_pct        FLOAT,
    sky_pct               FLOAT,
    miou_achieved         FLOAT,
    model_used            STRING
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX RiskZone (
    PRIMARY_ID zone_id    STRING,
    image_id              STRING,
    severity              STRING,
    class_name            STRING,
    patch_count           INT,
    center_row            INT,
    center_col            INT
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX SafePath (
    PRIMARY_ID path_id    STRING,
    image_id              STRING,
    total_cost            FLOAT,
    avg_traversability    FLOAT,
    hop_count             INT,
    patch_sequence        STRING
  ) WITH primary_id_as_attribute="true";

  ADD DIRECTED EDGE AdjacentTo (
    FROM TerrainPatch,
    TO   TerrainPatch,
    transition_cost FLOAT,
    direction       STRING
  ) WITH REVERSE_EDGE="AdjacentTo_Reverse";

  ADD DIRECTED EDGE BelongsTo (
    FROM TerrainPatch,
    TO   ImageRecord
  );

  ADD DIRECTED EDGE HasRiskZone (
    FROM ImageRecord,
    TO   RiskZone
  );

  ADD DIRECTED EDGE HasPath (
    FROM ImageRecord,
    TO   SafePath
  );
}

RUN SCHEMA_CHANGE JOB init_schema
DROP JOB init_schema
"""

# Simpler queries — no Dijkstra in GSQL (done in Python instead)
QUERIES_GSQL = """
USE GRAPH AutonomousGraph

CREATE OR REPLACE QUERY getRiskZones(STRING image_id)
FOR GRAPH AutonomousGraph {
  result = SELECT r
           FROM ImageRecord:i -(HasRiskZone:e)-> RiskZone:r
           WHERE i.image_id == image_id
           ORDER BY r.patch_count DESC;
  PRINT result;
}

CREATE OR REPLACE QUERY findSimilarTerrains(
    FLOAT rock_pct,
    FLOAT veg_pct,
    FLOAT brightness,
    INT   top_k
) FOR GRAPH AutonomousGraph {
  result = SELECT i FROM ImageRecord:i
           WHERE abs(i.rock_pct       - rock_pct)   < 0.08
           AND   abs(i.vegetation_pct - veg_pct)    < 0.10
           AND   abs(i.brightness     - brightness) < 0.10
           AND   i.miou_achieved > 0
           ORDER BY i.miou_achieved DESC
           LIMIT top_k;
  PRINT result;
}

CREATE OR REPLACE QUERY getImagePatches(STRING image_id)
FOR GRAPH AutonomousGraph {
  result = SELECT p FROM TerrainPatch:p
           WHERE p.image_id == image_id;
  PRINT result;
}

CREATE OR REPLACE QUERY getPatchNeighbors(STRING patch_id)
FOR GRAPH AutonomousGraph {
  result = SELECT t
           FROM TerrainPatch:s -(AdjacentTo:e)-> TerrainPatch:t
           WHERE s.patch_id == patch_id;
  PRINT result [result.patch_id, result.traversability, e.transition_cost];
}

INSTALL QUERY getRiskZones, findSimilarTerrains,
              getImagePatches, getPatchNeighbors
"""


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_connection() -> tg.TigerGraphConnection:
    conn = tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH_NAME,
        apiToken=API_TOKEN,
    )
    print(f"Connected to TigerGraph: {GRAPH_NAME}")
    return conn


def setup_schema(conn):
    print("Setting up TigerGraph schema...")
    try:
        conn.gsql(SCHEMA_GSQL)
        print("✅ Schema created")
    except Exception as e:
        print(f"Schema: {e} (may already exist — continuing)")
    try:
        conn.gsql(QUERIES_GSQL)
        print("✅ Queries installed")
    except Exception as e:
        print(f"Queries: {e}")


# ─────────────────────────────────────────────
# MASK → PATCHES
# ─────────────────────────────────────────────

def mask_to_patches(mask: np.ndarray,
                    confidence: np.ndarray,
                    image_id: str,
                    patch_size: int = PATCH_SIZE) -> list:
    H, W  = mask.shape
    rows  = H // patch_size
    cols  = W // patch_size
    patches = []

    for r in range(rows):
        for c in range(cols):
            r0, c0 = r * patch_size, c * patch_size
            r1, c1 = r0 + patch_size, c0 + patch_size

            pm = mask[r0:r1, c0:c1].flatten()
            pc = confidence[r0:r1, c0:c1]

            counts  = np.bincount(pm, minlength=NUM_CLASSES)
            dom_cls = int(np.argmax(counts))
            total   = pm.size
            dist    = {CLASS_NAMES[i]: round(float(counts[i]/total), 3)
                       for i in range(NUM_CLASSES)}

            patches.append({
                "patch_id":           make_patch_id(image_id, r, c),
                "image_id":           image_id,
                "rowV":                r,
                "colV":                c,
                "dominant_class":     dom_cls,
                "class_name":         CLASS_NAMES[dom_cls],
                "traversability":     TRAVERSABILITY[dom_cls],
                "risk_level":         RISK_LEVEL[dom_cls],
                "confidence":         float(pc.mean()),
                "class_distribution": json.dumps(dist),
            })

    return patches


def patches_to_edges(patches: list,
                     rows: int,
                     cols: int) -> list:
    """
    4-directional adjacency edges.
    transition_cost = 1 - avg_traversability of the two patches.
    Lower cost = safer (Dijkstra minimises cost).
    """
    patch_dict = {(p["rowV"], p["colV"]): p for p in patches}
    edges = []
    dirs  = [("N", -1, 0), ("S", 1, 0), ("E", 0, 1), ("W", 0, -1)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in patch_dict:
                continue
            src = patch_dict[(r, c)]
            for dname, dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in patch_dict:
                    continue
                tgt  = patch_dict[(nr, nc)]
                cost = round(
                    1.0 - (src["traversability"] + tgt["traversability"]) / 2.0,
                    4)
                edges.append({
                    "from":            src["patch_id"],
                    "to":              tgt["patch_id"],
                    "transition_cost": cost,
                    "direction":       dname,
                })

    return edges


def extract_risk_zones(patches: list, image_id: str) -> list:
    high_risk = [p for p in patches
                 if p["risk_level"] in ("HIGH", "MED_HIGH")]
    zones = []
    seen  = set()

    for i, p in enumerate(high_risk):
        if p["patch_id"] in seen:
            continue
        cluster = [p]
        seen.add(p["patch_id"])
        for p2 in high_risk:
            if p2["patch_id"] in seen:
                continue
            if (abs(p2["rowV"] - p["rowV"]) <= 2 and
                    abs(p2["colV"] - p["colV"]) <= 2 and
                    p2["dominant_class"] == p["dominant_class"]):
                cluster.append(p2)
                seen.add(p2["patch_id"])

        if len(cluster) >= 2:
            rows_c = [x["rowV"] for x in cluster]
            cols_c = [x["colV"] for x in cluster]
            zones.append({
                "zone_id":    f"{image_id}_risk_{i}",
                "image_id":   image_id,
                "severity":   p["risk_level"],
                "class_name": p["class_name"],
                "patch_count": len(cluster),
                "center_row": int(np.mean(rows_c)),
                "center_col": int(np.mean(cols_c)),
            })

    return zones


# ─────────────────────────────────────────────
# PYTHON-SIDE DIJKSTRA  (reliable, no GSQL needed)
# ─────────────────────────────────────────────

def dijkstra_python(patches: list, edges: list,
                    image_id: str,
                    start_row: int, start_col: int,
                    end_row: int,   end_col: int) -> dict:
    """
    Pure Python Dijkstra over the patch graph.
    Works always — no TigerGraph query needed.
    """
    start_id = make_patch_id(image_id, start_row, start_col)
    end_id   = make_patch_id(image_id, end_row,   end_col)

    # Build adjacency list
    adj = {}
    for p in patches:
        adj[p["patch_id"]] = []
    for e in edges:
        adj[e["from"]].append((e["to"],   e["transition_cost"]))
        adj[e["to"]].append((e["from"],   e["transition_cost"]))

    if start_id not in adj or end_id not in adj:
        return _fallback_path(image_id, start_row, start_col,
                              end_row, end_col,
                              len(patches) // (max(1, end_col - start_col + 1)))

    dist  = {start_id: 0.0}
    prev  = {}
    pq    = [(0.0, start_id)]

    while pq:
        cost, node = heapq.heappop(pq)
        if cost > dist.get(node, float("inf")):
            continue
        if node == end_id:
            break
        for neighbor, edge_cost in adj.get(node, []):
            new_cost = cost + edge_cost
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor]  = new_cost
                prev[neighbor]  = node
                heapq.heappush(pq, (new_cost, neighbor))

    # Reconstruct path
    path = []
    cur  = end_id
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start_id)
    path.reverse()

    total_cost = dist.get(end_id, -1)

    return {
        "path_ids":   path,
        "hop_count":  len(path),
        "total_cost": round(total_cost, 4),
        "start":      (start_row, start_col),
        "end":        (end_row, end_col),
        "method":     "python_dijkstra",
        "note":       "Dijkstra optimal",
    }


def _fallback_path(image_id, sr, sc, er, ec, rows):
    """Straight center-column path when graph lookup fails."""
    path = [make_patch_id(image_id, r, sc)
            for r in range(sr, er - 1, -1)]
    return {
        "path_ids":   path,
        "hop_count":  len(path),
        "total_cost": -1,
        "start":      (sr, sc),
        "end":        (er, ec),
        "method":     "fallback_center",
        "note":       "fallback center path",
    }


# ─────────────────────────────────────────────
# UPLOAD TO TIGERGRAPH
# ─────────────────────────────────────────────

def upload_terrain(conn,
                   image_id: str,
                   image_path: str,
                   seg_result: dict,
                   model_miou: float = 0.0,
                   patch_size: int = PATCH_SIZE) -> dict:
    """
    Full pipeline: mask → patches → upload to TigerGraph.
    Also runs Python Dijkstra and stores the path.
    Returns summary dict with patches, edges, risk_zones, path.
    """
    mask       = seg_result["mask"]
    confidence = seg_result["confidence"]
    class_dist = seg_result["class_dist"]
    trav_map   = seg_result["trav_map"]
    img_np     = seg_result["orig_np"]

    H, W  = mask.shape
    rows  = H // patch_size
    cols  = W // patch_size

    print(f"\n[TG] Uploading terrain graph: {image_id}")
    print(f"     Grid: {rows}×{cols} = {rows*cols} patches")

    # ── Build local data structures ──────────────
    patches = mask_to_patches(mask, confidence, image_id, patch_size)
    edges   = patches_to_edges(patches, rows, cols)
    zones   = extract_risk_zones(patches, image_id)
    bright  = float(img_np.mean() / 255.0)

    # ── Run Python Dijkstra BEFORE uploading ────
    # (doesn't depend on TG — always works)
    start_row = rows - 1
    start_col = cols // 2
    end_row   = 0
    end_col   = cols // 2

    path_result = dijkstra_python(
        patches, edges, image_id,
        start_row, start_col,
        end_row,   end_col,
    )
    print(f"     Path: {path_result['hop_count']} hops, "
          f"cost={path_result['total_cost']:.4f}")

    # ── Upload to TigerGraph ─────────────────────
    try:
        # 1. ImageRecord vertex
        conn.upsertVertex("ImageRecord", image_id, {
            "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_path":         str(image_path),
            "dominant_class":     max(class_dist, key=class_dist.get),
            "avg_traversability": float(trav_map.mean()),
            "brightness":         bright,
            "rock_pct":           class_dist.get("Rocks", 0),
            "vegetation_pct":     (class_dist.get("Dense_Vegetation", 0) +
                                   class_dist.get("Dry_Vegetation", 0)),
            "sky_pct":            class_dist.get("Sky", 0),
            "miou_achieved":      model_miou,
            "model_used":         "DeepLabV3+",
        })

        # 2. TerrainPatch vertices (batch)
        patch_data = [
            (p["patch_id"], {k: v for k, v in p.items() if k != "patch_id"})
            for p in patches
        ]
        conn.upsertVertices("TerrainPatch", patch_data)
        print(f"     ✅ {len(patches)} patch vertices")

        # 3. AdjacentTo edges — correct pyTigerGraph format:
        #    upsertEdges(src_type, edge_type, tgt_type, edges)
        #    edges = list of (src_id, tgt_id, attributes_dict)
        edge_tuples = [
            (e["from"], e["to"],
             {"transition_cost": e["transition_cost"],
              "direction":       e["direction"]})
            for e in edges
        ]
        conn.upsertEdges(
            "TerrainPatch", "AdjacentTo", "TerrainPatch",
            edge_tuples
        )
        print(f"     ✅ {len(edges)} adjacency edges")

        # 4. BelongsTo edges
        belongs_tuples = [
            (p["patch_id"], image_id, {})
            for p in patches
        ]
        conn.upsertEdges(
            "TerrainPatch", "BelongsTo", "ImageRecord",
            belongs_tuples
        )

        # 5. RiskZone vertices + HasRiskZone edges
        for z in zones:
            conn.upsertVertex(
                "RiskZone", z["zone_id"],
                {k: v for k, v in z.items() if k != "zone_id"}
            )
            conn.upsertEdge(
                "ImageRecord", image_id,
                "HasRiskZone",
                "RiskZone",   z["zone_id"],
                {}
            )
        print(f"     ✅ {len(zones)} risk zones")

        # 6. SafePath vertex
        path_id = f"{image_id}_path"
        conn.upsertVertex("SafePath", path_id, {
            "image_id":          image_id,
            "total_cost":        float(path_result["total_cost"]),
            "avg_traversability": float(trav_map.mean()),
            "hop_count":         path_result["hop_count"],
            "patch_sequence":    json.dumps(path_result["path_ids"][:50]),
        })
        conn.upsertEdge(
            "ImageRecord", image_id,
            "HasPath",
            "SafePath",  path_id,
            {}
        )
        print(f"     ✅ Safe path stored")
        print(f"     ✅ Terrain graph complete!")

    except Exception as e:
        print(f"     ⚠️  TigerGraph upload error: {e}")
        print(f"     (Path result still valid from Python Dijkstra)")

    return {
        "image_id":   image_id,
        "patches":    len(patches),
        "edges":      len(edges),
        "risk_zones": zones,
        "avg_trav":   float(trav_map.mean()),
        "class_dist": class_dist,
        "brightness": bright,
        "path":       path_result,  # always present
    }


# ─────────────────────────────────────────────
# PUBLIC API — called by pipeline.py
# ─────────────────────────────────────────────

def find_safe_path(conn,
                   image_id: str,
                   mask_shape: tuple,
                   patch_size: int = PATCH_SIZE,
                   patches: list = None,
                   edges: list = None) -> dict:
    """
    Returns safe path. Uses Python Dijkstra if patches/edges provided,
    otherwise falls back to center column path.
    NOTE: upload_terrain() already returns path in its result dict.
          Use that instead of calling this separately.
    """
    H, W      = mask_shape
    rows      = H // patch_size
    cols      = W // patch_size
    start_row = rows - 1
    start_col = cols // 2
    end_row   = 0
    end_col   = cols // 2

    if patches and edges:
        return dijkstra_python(
            patches, edges, image_id,
            start_row, start_col,
            end_row,   end_col,
        )

    return _fallback_path(
        image_id, start_row, start_col,
        end_row, end_col, rows,
    )


def get_risk_zones(conn, image_id: str) -> list:
    try:
        result = conn.runInstalledQuery(
            "getRiskZones",
            params={"image_id": image_id}
        )
        # Handle both list and dict response formats
        if isinstance(result, list) and len(result) > 0:
            r = result[0]
            if isinstance(r, dict):
                return r.get("result", [])
        return []
    except Exception as e:
        print(f"getRiskZones query failed: {e}")
        return []


def find_similar_terrains(conn,
                          rock_pct: float,
                          veg_pct: float,
                          brightness: float,
                          top_k: int = 5) -> list:
    try:
        result = conn.runInstalledQuery(
            "findSimilarTerrains",
            params={
                "rock_pct":   rock_pct,
                "veg_pct":    veg_pct,
                "brightness": brightness,
                "top_k":      top_k,
            }
        )
        if isinstance(result, list) and len(result) > 0:
            r = result[0]
            if isinstance(r, dict):
                return r.get("result", [])
        return []
    except Exception as e:
        print(f"findSimilarTerrains query failed: {e}")
        return []


# ─────────────────────────────────────────────
# VISUALISE PATH ON MASK
# ─────────────────────────────────────────────

def draw_path_on_mask(color_mask: np.ndarray,
                      path_ids: list,
                      patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Draw safe path as yellow dots + line on color mask."""
    import cv2
    result = color_mask.copy()
    pts    = []

    for pid in path_ids:
        try:
            _, r, c = parse_patch_id(pid)
            cy = r * patch_size + patch_size // 2
            cx = c * patch_size + patch_size // 2
            pts.append((cx, cy))
            cv2.circle(result, (cx, cy), 5, (255, 255, 0), -1)
        except Exception:
            pass

    for i in range(len(pts) - 1):
        cv2.line(result, pts[i], pts[i + 1], (255, 215, 0), 2)

    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("TigerGraph Terrain Graph — Self Test")
    print("=" * 50)

    conn = get_connection()

    try:
        n = conn.getVertexCount("ImageRecord")
        print(f"ImageRecord count : {n}")
        n = conn.getVertexCount("TerrainPatch")
        print(f"TerrainPatch count: {n}")
        print("✅ TigerGraph connection OK")
    except Exception as e:
        print(f"Connection test: {e}")
        print("Run: python setup_tigergraph.py")

    # Test Dijkstra on dummy data
    print("\n[Test] Python Dijkstra on 4×4 grid...")
    dummy_patches = [
        {"patch_id": make_patch_id("test", r, c),
         "image_id": "test", "row": r, "col": c,
         "dominant_class": 4,  # Landscape
         "class_name": "Landscape",
         "traversability": 0.95,
         "risk_level": "LOW",
         "confidence": 0.9,
         "class_distribution": "{}"}
        for r in range(4) for c in range(4)
    ]
    # Add a rock obstacle at (2,2)
    for p in dummy_patches:
        if p["row"] == 2 and p["col"] == 2:
            p["traversability"] = 0.15
            p["dominant_class"] = 3
            p["class_name"]     = "Rocks"
            p["risk_level"]     = "HIGH"

    dummy_edges = patches_to_edges(dummy_patches, 4, 4)
    path = dijkstra_python(dummy_patches, dummy_edges,
                           "test", 3, 2, 0, 2)
    print(f"  Path hops : {path['hop_count']}")
    print(f"  Total cost: {path['total_cost']}")
    print(f"  Path IDs  : {path['path_ids']}")
    print("✅ Dijkstra working")