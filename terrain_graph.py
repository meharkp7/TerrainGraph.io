"""
terrain_graph.py
────────────────
Converts segmentation mask into a spatial graph
and uploads it to TigerGraph.

Each 32x32 patch = one node
Edges = spatial adjacency with transition cost
"""

import heapq
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# TIGERGRAPH CONNECTION
# ─────────────────────────────────────────────

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

PATCH_SIZE = 32   # pixels per patch


# ─────────────────────────────────────────────
# SCHEMA SETUP — run once
# ─────────────────────────────────────────────

SCHEMA_GSQL = """
USE GLOBAL

DROP GRAPH AutonomousGraph

CREATE GRAPH AutonomousGraph()

USE GRAPH AutonomousGraph

CREATE SCHEMA_CHANGE JOB init_schema FOR GRAPH AutonomousGraph {

  ADD VERTEX TerrainPatch (
    PRIMARY_ID patch_id STRING,
    image_id          STRING,
    rowV              INT,
    colV              INT,
    dominant_class    INT,
    class_name        STRING,
    traversability    FLOAT,
    risk_level        STRING,
    confidence        FLOAT,
    class_distribution STRING
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX ImageRecord (
    PRIMARY_ID image_id STRING,
    timestamp         STRING,
    image_path        STRING,
    dominant_class    STRING,
    avg_traversability FLOAT,
    brightness        FLOAT,
    rock_pct          FLOAT,
    vegetation_pct    FLOAT,
    sky_pct           FLOAT,
    miou_achieved     FLOAT,
    model_used        STRING
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX RiskZone (
    PRIMARY_ID zone_id STRING,
    image_id    STRING,
    severity    STRING,
    class_name  STRING,
    patch_count INT,
    center_row  INT,
    center_col  INT
  ) WITH primary_id_as_attribute="true";

  ADD VERTEX SafePath (
    PRIMARY_ID path_id STRING,
    image_id        STRING,
    total_cost      FLOAT,
    avg_traversability FLOAT,
    hop_count       INT,
    patch_sequence  STRING
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

QUERIES_GSQL = """
USE GRAPH AutonomousGraph

CREATE OR REPLACE QUERY getRiskZones(STRING image_id)
FOR GRAPH AutonomousGraph {
  result = SELECT r FROM ImageRecord:i -(HasRiskZone:e)-> RiskZone:r
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
           WHERE abs(i.rock_pct        - rock_pct)   < 0.08
           AND   abs(i.vegetation_pct  - veg_pct)    < 0.10
           AND   abs(i.brightness      - brightness) < 0.10
           AND   i.miou_achieved > 0
           ORDER BY i.miou_achieved DESC
           LIMIT top_k;
  PRINT result;
}

CREATE OR REPLACE QUERY getImageSummary(STRING image_id)
FOR GRAPH AutonomousGraph {
  img     = SELECT i FROM ImageRecord:i
            WHERE i.image_id == image_id;
  patches = SELECT p FROM TerrainPatch:p
            WHERE p.image_id == image_id
            ORDER BY p.traversability DESC;
  risks   = SELECT r FROM ImageRecord:i -(HasRiskZone)-> RiskZone:r
            WHERE i.image_id == image_id;
  PRINT img, patches, risks;
}

INSTALL QUERY getRiskZones, findSimilarTerrains, getImageSummary
"""


def get_connection() -> tg.TigerGraphConnection:
    conn = tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH_NAME,
        apiToken=API_TOKEN,
    )
    return conn


def setup_schema(conn):
    print("Setting up TigerGraph schema...")
    conn.gsql(SCHEMA_GSQL)
    print("✅ Schema created")
    conn.gsql(QUERIES_GSQL)
    print("✅ Queries installed")


def _patch_id(image_id: str, row: int, col: int) -> str:
    return f"{image_id}_{row}_{col}"


def _parse_patch_id(patch_id: str) -> tuple[int, int]:
    _, row, col = patch_id.rsplit("_", 2)
    return int(row), int(col)


def _transition_cost(trav_a: float, trav_b: float, risk_a: str, risk_b: str) -> float:
    penalty = 0.0
    if "HIGH" in (risk_a, risk_b):
        penalty = 0.22
    elif "MED_HIGH" in (risk_a, risk_b):
        penalty = 0.12
    trav = (trav_a + trav_b) / 2.0
    return round(max(0.05, 1.0 - trav + penalty), 4)


def _flatten_query_result(result):
    if not result:
        return []

    parsed = []
    if isinstance(result, dict):
        result = [result]

    for part in result:
        if isinstance(part, dict):
            if "result" in part and isinstance(part["result"], list):
                parsed.extend(part["result"])
            elif "vertices" in part and isinstance(part["vertices"], dict):
                for vtype, collection in part["vertices"].items():
                    for vid, meta in collection.items():
                        parsed.append({
                            "type": vtype,
                            "id": vid,
                            "attributes": meta.get("attributes", {}),
                        })
            elif "attributes" in part:
                parsed.append(part)
            else:
                for value in part.values():
                    if isinstance(value, list):
                        parsed.extend(_flatten_query_result(value))
    return parsed


def upload_terrain(conn,
                   image_id: str,
                   image_path: str,
                   seg_result: dict,
                   model_miou: float,
                   model_name: str = "deeplabv3+") -> dict:
    mask = seg_result["mask"]
    trav_map = seg_result["trav_map"]
    confidence = seg_result["confidence"]

    rows = mask.shape[0] // PATCH_SIZE
    cols = mask.shape[1] // PATCH_SIZE

    conn.upsertVertex("ImageRecord", image_id, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": str(Path(image_path).resolve()),
        "dominant_class": max(seg_result["class_dist"], key=seg_result["class_dist"].get),
        "avg_traversability": float(np.mean(trav_map)),
        "brightness": float(np.mean(seg_result["orig_np"]) / 255.0),
        "rock_pct": float(seg_result["class_dist"].get("Rocks", 0.0)),
        "vegetation_pct": float(seg_result["class_dist"].get("Dense_Vegetation", 0.0) +
                                seg_result["class_dist"].get("Dry_Vegetation", 0.0)),
        "sky_pct": float(seg_result["class_dist"].get("Sky", 0.0)),
        "miou_achieved": float(model_miou),
        "model_used": model_name,
    })

    patch_vertices = []
    risk_zone_vertices = []
    for row in range(rows):
        for col in range(cols):
            patch = mask[row*PATCH_SIZE:(row+1)*PATCH_SIZE,
                         col*PATCH_SIZE:(col+1)*PATCH_SIZE]
            confidence_patch = confidence[row*PATCH_SIZE:(row+1)*PATCH_SIZE,
                                          col*PATCH_SIZE:(col+1)*PATCH_SIZE]
            counts = np.bincount(patch.flatten(), minlength=NUM_CLASSES)
            dominant = int(np.argmax(counts))
            patch_id = _patch_id(image_id, row, col)
            traversability = float(np.mean(trav_map[row*PATCH_SIZE:(row+1)*PATCH_SIZE,
                                                   col*PATCH_SIZE:(col+1)*PATCH_SIZE]))
            risk_level = RISK_LEVEL[dominant]
            patch_vertices.append((patch_id, {
                "image_id": image_id,
                "rowV": row,
                "colV": col,
                "dominant_class": dominant,
                "class_name": CLASS_NAMES[dominant],
                "traversability": traversability,
                "risk_level": risk_level,
                "confidence": float(np.mean(confidence_patch)),
                "class_distribution": json.dumps({
                    CLASS_NAMES[i]: round(float(counts[i]) / counts.sum(), 4)
                    for i in range(NUM_CLASSES)
                }),
            }))

            if risk_level in {"HIGH", "MED_HIGH"}:
                zone_id = f"{image_id}_risk_{row}_{col}"
                risk_zone_vertices.append((zone_id, {
                    "image_id": image_id,
                    "severity": risk_level,
                    "class_name": CLASS_NAMES[dominant],
                    "patch_count": 1,
                    "center_row": row,
                    "center_col": col,
                }))

    conn.upsertVertices("TerrainPatch", patch_vertices)
    if risk_zone_vertices:
        conn.upsertVertices("RiskZone", risk_zone_vertices)
        conn.upsertEdges(
            "ImageRecord",
            "HasRiskZone",
            "RiskZone",
            [(image_id, zone_id, {}) for zone_id, _ in risk_zone_vertices],
            vertexMustExist=True,
        )

    edge_rows = []
    for row in range(rows):
        for col in range(cols):
            source_id = _patch_id(image_id, row, col)
            trav_a = float(np.mean(trav_map[row*PATCH_SIZE:(row+1)*PATCH_SIZE,
                                        col*PATCH_SIZE:(col+1)*PATCH_SIZE]))
            window = mask[row*PATCH_SIZE:(row+1)*PATCH_SIZE,
                          col*PATCH_SIZE:(col+1)*PATCH_SIZE]
            risk_a = RISK_LEVEL[int(np.argmax(np.bincount(window.flatten(), minlength=NUM_CLASSES)))]
            for dr, dc, direction in [(0, 1, "E"), (1, 0, "S"), (0, -1, "W"), (-1, 0, "N")]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    target_id = _patch_id(image_id, nr, nc)
                    trav_b = float(np.mean(trav_map[nr*PATCH_SIZE:(nr+1)*PATCH_SIZE,
                                                    nc*PATCH_SIZE:(nc+1)*PATCH_SIZE]))
                    window_b = mask[nr*PATCH_SIZE:(nr+1)*PATCH_SIZE,
                                    nc*PATCH_SIZE:(nc+1)*PATCH_SIZE]
                    risk_b = RISK_LEVEL[int(np.argmax(np.bincount(window_b.flatten(), minlength=NUM_CLASSES)))]
                    edge_rows.append((source_id, target_id, {
                        "transition_cost": _transition_cost(trav_a, trav_b, risk_a, risk_b),
                        "direction": direction,
                    }))

    if edge_rows:
        conn.upsertEdges("TerrainPatch", "AdjacentTo", "TerrainPatch", edge_rows, vertexMustExist=True)

    return {
        "patches": len(patch_vertices),
        "edges": len(edge_rows),
        "risk_zones": len(risk_zone_vertices),
        "rows": rows,
        "cols": cols,
    }


def find_safe_path(conn,
                   image_id: str,
                   trav_map: np.ndarray,
                   start: tuple[int, int] | None = None,
                   end: tuple[int, int] | None = None) -> dict:
    if start is None:
        start = (0, 0)
    if end is None:
        height, width = trav_map.shape
        end = (height // PATCH_SIZE - 1, width // PATCH_SIZE - 1)

    rows = trav_map.shape[0] // PATCH_SIZE
    cols = trav_map.shape[1] // PATCH_SIZE
    start_id = _patch_id(image_id, start[0], start[1])
    end_id = _patch_id(image_id, end[0], end[1])

    def neighbors(r, c):
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    dist = {start_id: 0.0}
    prev = {}
    heap = [(0.0, start)]

    while heap:
        cost, (r, c) = heapq.heappop(heap)
        current_id = _patch_id(image_id, r, c)
        if cost > dist.get(current_id, float("inf")):
            continue
        if current_id == end_id:
            break

        current_trav = float(np.mean(trav_map[r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                                           c*PATCH_SIZE:(c+1)*PATCH_SIZE]))
        current_risk = "HIGH" if current_trav < 0.25 else "MED_HIGH" if current_trav < 0.5 else "LOW"
        for nr, nc in neighbors(r, c):
            neighbor_id = _patch_id(image_id, nr, nc)
            neighbor_trav = float(np.mean(trav_map[nr*PATCH_SIZE:(nr+1)*PATCH_SIZE,
                                                  nc*PATCH_SIZE:(nc+1)*PATCH_SIZE]))
            neighbor_risk = "HIGH" if neighbor_trav < 0.25 else "MED_HIGH" if neighbor_trav < 0.5 else "LOW"
            edge_cost = _transition_cost(current_trav, neighbor_trav, current_risk, neighbor_risk)
            candidate = cost + edge_cost
            if candidate < dist.get(neighbor_id, float("inf")):
                dist[neighbor_id] = candidate
                prev[neighbor_id] = current_id
                heapq.heappush(heap, (candidate, (nr, nc)))

    if end_id not in dist:
        return {"path_ids": [], "hop_count": 0, "total_cost": float("inf"), "note": "no feasible path"}

    path_ids = [end_id]
    current = end_id
    while current != start_id:
        current = prev[current]
        path_ids.append(current)
    path_ids.reverse()

    total_cost = round(dist[end_id], 4)
    path_id = f"{image_id}_path"
    conn.upsertVertex("SafePath", path_id, {
        "image_id": image_id,
        "total_cost": total_cost,
        "avg_traversability": float(np.mean(trav_map)),
        "hop_count": len(path_ids) - 1,
        "patch_sequence": json.dumps(path_ids),
    })
    conn.upsertEdge("ImageRecord", image_id, "HasPath", "SafePath", path_id, {})

    return {
        "path_ids": path_ids,
        "hop_count": len(path_ids) - 1,
        "total_cost": total_cost,
        "note": "local weighted Dijkstra on patch graph",
    }


def get_risk_zones(conn, image_id: str) -> list:
    try:
        output = conn.runInstalledQuery("getRiskZones", {"image_id": image_id})
        return _flatten_query_result(output)
    except Exception:
        return []


def find_similar_terrains(conn,
                           rock_pct: float,
                           veg_pct: float,
                           brightness: float,
                           top_k: int = 5) -> list:
    try:
        output = conn.runInstalledQuery(
            "findSimilarTerrains",
            {
                "rock_pct": rock_pct,
                "veg_pct": veg_pct,
                "brightness": brightness,
                "top_k": top_k,
            },
        )
        return _flatten_query_result(output)
    except Exception:
        return []


def draw_path_on_mask(color_mask: np.ndarray, path_ids: list[str]) -> np.ndarray:
    overlay = color_mask.copy()
    highlight = np.array([255, 255, 0], dtype=np.uint8)
    for patch_id in path_ids:
        try:
            row, col = _parse_patch_id(patch_id)
        except Exception:
            continue
        y1 = row * PATCH_SIZE
        x1 = col * PATCH_SIZE
        y2 = y1 + PATCH_SIZE
        x2 = x1 + PATCH_SIZE
        overlay[y1:y2, x1:x2] = (((overlay[y1:y2, x1:x2].astype(int) + highlight.astype(int)) // 2).astype(np.uint8))
    return overlay


def setup_schema(conn):
    """Run once to create schema and queries."""
    print("Setting up TigerGraph schema...")
    try:
        conn.gsql(SCHEMA_GSQL)
        print("✅ Schema created")
        conn.gsql(QUERIES_GSQL)
        print("✅ Queries installed")
    except Exception as e:
        print(f"Schema setup error: {e}")
        print("If graph already exists, this is okay.")


# ─────────────────────────────────────────────
# MASK → GRAPH CONVERSION
# ─────────────────────────────────────────────

def mask_to_patches(mask: np.ndarray,
                    confidence: np.ndarray,
                    image_id: str,
                    patch_size: int = PATCH_SIZE) -> list:
    """
    Divide mask into patch_size x patch_size patches.
    Returns list of patch dicts.
    """
    H, W    = mask.shape
    patches = []
    rows    = H // patch_size
    cols    = W // patch_size

    for r in range(rows):
        for c in range(cols):
            r0 = r * patch_size
            c0 = c * patch_size
            r1 = r0 + patch_size
            c1 = c0 + patch_size

            patch_mask = mask[r0:r1, c0:c1]
            patch_conf = confidence[r0:r1, c0:c1]

            # Dominant class
            counts = np.bincount(
                patch_mask.flatten(), minlength=NUM_CLASSES)
            dom_cls = int(np.argmax(counts))

            # Class distribution JSON
            total = patch_mask.size
            dist  = {CLASS_NAMES[i]: round(float(counts[i]/total), 3)
                     for i in range(NUM_CLASSES)}

            patches.append({
                "patch_id":          f"{image_id}_{r}_{c}",
                "image_id":          image_id,
                "rowV":              r,
                "colV":              c,
                "dominant_class":    dom_cls,
                "class_name":        CLASS_NAMES[dom_cls],
                "traversability":    TRAVERSABILITY[dom_cls],
                "risk_level":        RISK_LEVEL[dom_cls],
                "confidence":        float(patch_conf.mean()),
                "class_distribution":json.dumps(dist),
            })

    return patches


def patches_to_edges(patches: list,
                     image_id: str,
                     rows: int,
                     cols: int) -> list:
    """
    Create adjacency edges between neighbouring patches.
    transition_cost = 1 - avg(traversability of two patches)
    Lower cost = better path.
    """
    patch_dict = {(p["rowV"], p["colV"]): p for p in patches}
    edges = []
    dirs  = [("N",-1,0),("S",1,0),("E",0,1),("W",0,-1)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in patch_dict:
                continue
            src = patch_dict[(r, c)]
            for dname, dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if (nr, nc) not in patch_dict:
                    continue
                tgt  = patch_dict[(nr, nc)]
                cost = 1.0 - (src["traversability"] +
                               tgt["traversability"]) / 2.0
                edges.append({
                    "from": src["patch_id"],
                    "to":   tgt["patch_id"],
                    "transition_cost": round(cost, 4),
                    "direction": dname,
                })

    return edges


def extract_risk_zones(patches: list,
                       image_id: str) -> list:
    """Find contiguous high-risk patch clusters."""
    high_risk = [p for p in patches
                 if p["risk_level"] in ("HIGH", "MED_HIGH")]

    zones = []
    seen  = set()

    for i, p in enumerate(high_risk):
        if p["patch_id"] in seen:
            continue
        # Simple cluster: patches of same class within 2 hops
        cluster = [p]
        seen.add(p["patch_id"])
        for p2 in high_risk:
            if p2["patch_id"] in seen:
                continue
            if (abs(p2["rowV"]-p["rowV"]) <= 2 and
                    abs(p2["colV"]-p["colV"]) <= 2 and
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
                "patch_count":len(cluster),
                "center_row": int(np.mean(rows_c)),
                "center_col": int(np.mean(cols_c)),
            })

    return zones


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
    Full pipeline: mask → patches → TigerGraph.
    Returns summary dict.
    """
    mask       = seg_result["mask"]
    confidence = seg_result["confidence"]
    class_dist = seg_result["class_dist"]
    trav_map   = seg_result["trav_map"]

    H, W  = mask.shape
    rows  = H // patch_size
    cols  = W // patch_size

    print(f"\nUploading terrain graph for {image_id}...")
    print(f"  Grid: {rows}×{cols} = {rows*cols} patches")

    # 1. Create patches
    patches = mask_to_patches(mask, confidence,
                               image_id, patch_size)

    # 2. Create edges
    edges   = patches_to_edges(patches, image_id, rows, cols)

    # 3. Risk zones
    zones   = extract_risk_zones(patches, image_id)

    # 4. Image features
    img_np  = seg_result["orig_np"]
    bright  = float(img_np.mean() / 255.0)

    # 5. Upsert ImageRecord
    conn.upsertVertex("ImageRecord", image_id, {
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_path":         str(image_path),
        "dominant_class":     max(class_dist,
                                  key=class_dist.get),
        "avg_traversability": float(trav_map.mean()),
        "brightness":         bright,
        "rock_pct":           class_dist.get("Rocks", 0),
        "vegetation_pct":     (class_dist.get("Dense_Vegetation",0) +
                               class_dist.get("Dry_Vegetation",0)),
        "sky_pct":            class_dist.get("Sky", 0),
        "miou_achieved":      model_miou,
        "model_used":         "DeepLabV3+",
    })

    # 6. Upsert TerrainPatch vertices
    patch_vertices = [(p["patch_id"], {
        k: v for k, v in p.items()
        if k != "patch_id"
    }) for p in patches]
    conn.upsertVertices("TerrainPatch", patch_vertices)
    print(f"  ✅ {len(patches)} patch vertices uploaded")

    # 7. Upsert edges
    edge_data = [
        (e["from"], "AdjacentTo", e["to"],
         {"transition_cost": e["transition_cost"],
          "direction":       e["direction"]})
        for e in edges
    ]
    conn.upsertEdges("TerrainPatch", "AdjacentTo",
                     "TerrainPatch", edge_data)
    print(f"  ✅ {len(edges)} adjacency edges uploaded")

    # 8. BelongsTo edges
    belongs = [(p["patch_id"], "BelongsTo", image_id, {})
               for p in patches]
    conn.upsertEdges("TerrainPatch", "BelongsTo",
                     "ImageRecord", belongs)

    # 9. Risk zones
    for z in zones:
        conn.upsertVertex("RiskZone", z["zone_id"],
                          {k:v for k,v in z.items()
                           if k != "zone_id"})
        conn.upsertEdge("ImageRecord", image_id,
                        "HasRiskZone", "RiskZone",
                        z["zone_id"], {})
    print(f"  ✅ {len(zones)} risk zones uploaded")

    summary = {
        "image_id":    image_id,
        "patches":     len(patches),
        "edges":       len(edges),
        "risk_zones":  zones,
        "avg_trav":    float(trav_map.mean()),
        "class_dist":  class_dist,
        "brightness":  bright,
    }
    print(f"  ✅ Terrain graph ready!")
    return summary


# ─────────────────────────────────────────────
# PATH FINDING
# ─────────────────────────────────────────────

def get_risk_zones(conn, image_id: str) -> list:
    """Retrieve risk zones for an image."""
    try:
        result = conn.runInstalledQuery(
            "getRiskZones",
            params={"image_id": image_id}
        )
        return result[0].get("result", [])
    except Exception as e:
        print(f"Risk zone query failed: {e}")
        return []


def find_similar_terrains(conn,
                          rock_pct: float,
                          veg_pct: float,
                          brightness: float,
                          top_k: int = 5) -> list:
    """Find historically similar terrain images."""
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
        return result[0].get("result", [])
    except Exception as e:
        print(f"Similar terrain query failed: {e}")
        return []


# ─────────────────────────────────────────────
# VISUALISE PATH ON MASK
# ─────────────────────────────────────────────

def draw_path_on_mask(color_mask: np.ndarray,
                      path_ids: list,
                      patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Draw safe path as yellow line on color mask."""
    import cv2
    result = color_mask.copy()
    for pid in path_ids:
        parts = pid.split("_")
        if len(parts) >= 3:
            try:
                r = int(parts[-2])
                c = int(parts[-1])
                r0 = r * patch_size + patch_size // 2
                c0 = c * patch_size + patch_size // 2
                cv2.circle(result, (c0, r0), 6,
                           (255, 255, 0), -1)   # yellow dot
            except:
                pass
    # Connect dots
    pts = []
    for pid in path_ids:
        parts = pid.split("_")
        if len(parts) >= 3:
            try:
                r = int(parts[-2])
                c = int(parts[-1])
                pts.append((c*patch_size+patch_size//2,
                             r*patch_size+patch_size//2))
            except:
                pass
    if len(pts) > 1:
        for i in range(len(pts)-1):
            cv2.line(result, pts[i], pts[i+1],
                     (255, 255, 0), 2)
    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_connection()
    print(f"Connected to TigerGraph: {HOST}")
    print(f"Graph: {GRAPH_NAME}")

    # Test connection
    try:
        info = conn.getVertexCount("ImageRecord")
        print(f"ImageRecord count: {info}")
    except Exception as e:
        print(f"Connection test: {e}")
        print("Run setup_schema(conn) if first time")