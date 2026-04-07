"""
pipeline.py  —  FIXED VERSION
──────────────────────────────
Key fix: upload_terrain() now returns path in its result dict.
No separate find_safe_path() call needed.
"""

import os
import time
import json
import uuid
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from segmentor     import load_model, predict, CLASS_NAMES, NUM_CLASSES
from terrain_graph import (
    get_connection,
    upload_terrain,
    get_risk_zones,
    find_similar_terrains,
    draw_path_on_mask,
)
from explainer import (
    generate_navigation_briefing,
    generate_system_explanation,
)

# ─────────────────────────────────────────────
# LOAD MODEL + CONNECT ONCE AT STARTUP
# ─────────────────────────────────────────────

CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "./runs/deeplabv3+_20260403_211706/best.pth"
)
MODEL_MIOU = 0.6676

print("Loading model...")
model, device, cfg = load_model(CHECKPOINT)

print("Connecting to TigerGraph...")
conn = get_connection()
print("✅ Pipeline ready\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(image_path: str,
                 use_tta: bool = True,
                 save_outputs: bool = True) -> dict:
    """
    Full pipeline:
      1. Segmentation (DeepLabV3+)
      2. Terrain graph upload to TigerGraph
         → Python Dijkstra runs INSIDE upload_terrain
      3. Risk zone + similar terrain queries
      4. LLM briefing (Groq / Gemini / Claude)
    Returns result dict for the API / UI.
    """
    t_start  = time.time()
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    out_dir  = Path("./pipeline_outputs") / image_id
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Image    : {image_path}")
    print(f"Image ID : {image_id}")
    print(f"{'='*60}")

    # ── Step 1: Segmentation ──────────────────────────────────
    print("\n[1/4] Segmentation...")
    t0 = time.time()
    seg = predict(model, device, image_path, use_tta=use_tta)
    print(f"      Done in {time.time()-t0:.1f}s")
    print("      Class distribution:")
    for name, pct in seg["class_dist"].items():
        if pct > 0.01:
            bar = '█' * int(pct * 30)
            print(f"        {name:25s}: {pct:.1%}  {bar}")

    # ── Step 2: Terrain graph + Dijkstra path ─────────────────
    print("\n[2/4] Building terrain graph + pathfinding...")
    t0 = time.time()
    terrain = upload_terrain(
        conn=conn,
        image_id=image_id,
        image_path=image_path,
        seg_result=seg,
        model_miou=MODEL_MIOU,
    )
    # Path is already computed inside upload_terrain via Python Dijkstra
    path_result = terrain["path"]
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      Patches : {terrain['patches']}")
    print(f"      Edges   : {terrain['edges']}")
    print(f"      Path    : {path_result['hop_count']} hops, "
          f"cost={path_result['total_cost']}")

    # ── Step 3: Risk zones + similar terrains ─────────────────
    print("\n[3/4] Querying graph...")
    t0 = time.time()

    # Risk zones from terrain (already computed locally)
    risk_zones_local = terrain["risk_zones"]

    # Also try to fetch from TigerGraph (may return richer data)
    risk_zones_tg = get_risk_zones(conn, image_id)
    risk_zones    = risk_zones_tg if risk_zones_tg else _format_local_risks(risk_zones_local)

    similar = find_similar_terrains(
        conn,
        rock_pct   = seg["class_dist"].get("Rocks", 0),
        veg_pct    = (seg["class_dist"].get("Dense_Vegetation", 0) +
                      seg["class_dist"].get("Dry_Vegetation", 0)),
        brightness = float(seg["orig_np"].mean() / 255),
        top_k      = 5,
    )
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      Risk zones: {len(risk_zones_local)} local, "
          f"{len(risk_zones_tg)} from TigerGraph")
    print(f"      Similar terrains: {len(similar)}")

    # ── Step 4: LLM reasoning ─────────────────────────────────
    print("\n[4/4] LLM briefing...")
    t0 = time.time()
    briefing    = generate_navigation_briefing(
        class_dist         = seg["class_dist"],
        path_result        = path_result,
        risk_zones         = risk_zones,
        similar_terrains   = similar,
        avg_traversability = float(seg["trav_map"].mean()),
    )
    explanation = generate_system_explanation(
        class_dist  = seg["class_dist"],
        path_result = path_result,
        risk_zones  = risk_zones,
        model_miou  = MODEL_MIOU,
    )
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── Draw path ─────────────────────────────────────────────
    path_visual = draw_path_on_mask(
        seg["color_mask"],
        path_result.get("path_ids", []),
    )

    # ── Save outputs ──────────────────────────────────────────
    if save_outputs:
        from PIL import Image as PILImage
        PILImage.fromarray(seg["orig_np"])\
                 .save(out_dir / "original.png")
        PILImage.fromarray(seg["color_mask"])\
                 .save(out_dir / "segmented.png")
        PILImage.fromarray(seg["overlay"])\
                 .save(out_dir / "overlay.png")
        PILImage.fromarray(path_visual)\
                 .save(out_dir / "path.png")
        with open(out_dir / "briefing.txt", "w") as f:
            f.write(briefing)
        with open(out_dir / "summary.json", "w") as f:
            json.dump({
                "image_id":  image_id,
                "class_dist":seg["class_dist"],
                "path":      path_result,
                "risk_zones":len(risk_zones_local),
                "similar":   len(similar),
                "avg_trav":  float(seg["trav_map"].mean()),
            }, f, indent=2)
        print(f"\n      Saved → {out_dir}")

    total_time = time.time() - t_start

    result = {
        # Images
        "original":    seg["orig_np"],
        "segmented":   seg["color_mask"],
        "overlay":     seg["overlay"],
        "path_visual": path_visual,

        # Data
        "image_id":    image_id,
        "class_dist":  seg["class_dist"],
        "path":        path_result,
        "risk_zones":  risk_zones,
        "similar":     similar,
        "avg_trav":    float(seg["trav_map"].mean()),

        # Text
        "briefing":    briefing,
        "explanation": explanation,

        # Stats
        "total_time":  round(total_time, 2),
        "model_miou":  MODEL_MIOU,
    }

    print(f"\n{'='*60}")
    print(f"✅ Complete in {total_time:.1f}s")
    print(f"\n📋 Briefing:\n{briefing}")
    print(f"{'='*60}")

    return result


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _format_local_risks(zones: list) -> list:
    """Convert local risk zone dicts to TigerGraph-style format."""
    return [{"attributes": z} for z in zones]


# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else None

    if img is None:
        imgs = list(Path("./dataset/val/Color_Images").glob("*.png"))
        if not imgs:
            print("No test image. Pass path as argument.")
            sys.exit(1)
        img = str(imgs[0])

    print(f"Test image: {img}")
    result = run_pipeline(img)