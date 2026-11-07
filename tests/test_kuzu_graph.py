"""
Live Kuzu backend test: upload, risk-zone roundtrip, similarity memory,
and idempotent re-upload against a throwaway database.

Needs kuzu installed — which the py3.14 local .venv CANNOT take (no macOS
wheel). Run it in the 3.11 sandbox:
  PYTHONPATH=. KUZU_DB_PATH=$(mktemp -d)/kuzu /tmp/kuzuvenv/bin/python tests/test_kuzu_graph.py
"""

import os
import sys
import tempfile
from pathlib import Path

# DB path must be set before kuzu_graph is imported (module reads it then).
os.environ.setdefault("KUZU_DB_PATH",
                      str(Path(tempfile.mkdtemp(prefix="kuzutest_")) / "kuzu"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kuzu_graph import (
    find_similar_terrains,
    get_connection,
    get_risk_zones,
    upload_terrain,
)


def fake_seg(size=64, rock_row=True):
    mask = np.full((size, size), 4, dtype=np.int64)  # Landscape
    if rock_row:
        # Whole top patch row Rocks -> 2-patch HIGH cluster -> 1 risk zone
        mask[: size // 2, :] = 3
    return {
        "mask": mask,
        "confidence": np.full((size, size), 0.9, dtype=np.float32),
        "class_dist": {"Landscape": 0.5, "Rocks": 0.4,
                       "Dry_Vegetation": 0.1},
        "trav_map": np.full((size, size), 0.8, dtype=np.float32),
        "orig_np": np.full((size, size, 3), 128, dtype=np.uint8),
    }


def count(conn, label):
    r = conn.execute(f"MATCH (n:{label}) RETURN count(*)")
    assert r.has_next()
    return r.get_next()[0]


def main():
    conn = get_connection()

    # Upload + summary shape (same contract as TigerGraph flavor)
    tg = upload_terrain(conn, "img_a", "fake_a.png", fake_seg(),
                        model_miou=0.66)
    assert tg["image_id"] == "img_a"
    assert tg["path"]["hop_count"] > 0
    assert tg["patches"] == 4 and tg["edges"] > 0
    assert len(tg["risk_zones"]) >= 1, "expected a Rocks risk zone"

    # Risk zones roundtrip through the graph
    stored = get_risk_zones(conn, "img_a")
    assert len(stored) == len(tg["risk_zones"]), (stored, tg["risk_zones"])
    assert stored[0]["attributes"]["severity"] in ("HIGH", "MED_HIGH")
    assert stored[0]["attributes"]["class_name"] == "Rocks"
    assert get_risk_zones(conn, "img_missing") == []

    # Similarity memory finds the uploaded record...
    sim = find_similar_terrains(conn, rock_pct=0.4, veg_pct=0.1,
                                brightness=0.5)
    ids = [s["attributes"]["image_id"] for s in sim]
    assert "img_a" in ids, ids
    assert sim[0]["attributes"]["miou_achieved"] == 0.66

    # ...but not for distant compositions
    assert find_similar_terrains(conn, rock_pct=0.9, veg_pct=0.0,
                                 brightness=0.0) == []

    # Idempotent re-upload: counts must not move
    before = (count(conn, "ImageRecord"), count(conn, "TerrainPatch"),
              count(conn, "RiskZone"), count(conn, "SafePath"))
    upload_terrain(conn, "img_a", "fake_a.png", fake_seg(), model_miou=0.66)
    after = (count(conn, "ImageRecord"), count(conn, "TerrainPatch"),
             count(conn, "RiskZone"), count(conn, "SafePath"))
    assert before == after, (before, after)
    assert before[0] == 1 and before[1] == 4

    # A second, different image lands beside the first
    seg_b = fake_seg(rock_row=False)
    seg_b["class_dist"] = {"Landscape": 0.9, "Rocks": 0.0,
                           "Dry_Vegetation": 0.1}
    upload_terrain(conn, "img_b", "fake_b.png", seg_b, model_miou=0.70)
    assert count(conn, "ImageRecord") == 2

    print("test_kuzu_graph: ALL PASS")


if __name__ == "__main__":
    main()
