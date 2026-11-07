"""
Graceful-degradation test: the pipeline must survive TigerGraph being
unreachable or unbuildable (bad creds, version quirk, no network).

Drives the real terrain_graph functions with conn=None and a synthetic
segmentation — no torch, no network. Run:  python tests/test_tg_graceful.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from terrain_graph import (
    draw_path_on_mask,
    find_similar_terrains,
    get_risk_zones,
    upload_terrain,
)


def fake_seg(size=64):
    mask = np.zeros((size, size), dtype=np.int64)
    mask[:, size // 2:] = 4  # half Landscape
    mask[: size // 4, : size // 4] = 3  # a Rocks corner
    return {
        "mask": mask,
        "confidence": np.full((size, size), 0.9, dtype=np.float32),
        "class_dist": {"Landscape": 0.7, "Rocks": 0.2,
                       "Dry_Vegetation": 0.1},
        "trav_map": np.full((size, size), 0.8, dtype=np.float32),
        "orig_np": np.zeros((size, size, 3), dtype=np.uint8),
    }


def main():
    seg = fake_seg()

    # Upload with no connection: local path + zones must still come back
    tg = upload_terrain(None, "test_img", "fake.png", seg, model_miou=0.66)
    assert tg["image_id"] == "test_img"
    assert tg["path"]["hop_count"] > 0, "Dijkstra path missing"
    assert tg["path"]["total_cost"] >= 0
    assert isinstance(tg["risk_zones"], list)
    assert tg["patches"] == 4 and tg["edges"] > 0  # 64px / 32 = 2x2 grid

    # Queries with no connection: empty, never an exception
    assert get_risk_zones(None, "test_img") == []
    assert find_similar_terrains(None, 0.2, 0.1, 0.5) == []

    # Path drawing works on the locally computed path ids
    color = np.zeros((64, 64, 3), dtype=np.uint8)
    drawn = draw_path_on_mask(color, tg["path"]["path_ids"])
    assert drawn.sum() > 0, "path was not drawn"

    print("test_tg_graceful: ALL PASS")


if __name__ == "__main__":
    main()
