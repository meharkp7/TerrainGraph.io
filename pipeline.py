"""
pipeline.py
───────────
Full pipeline: Image → Segment → Graph → Path → Explain
"""

import os
import time
import json
import uuid
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from segmentor    import load_model, predict, CLASS_NAMES, NUM_CLASSES
from terrain_graph import (
    get_connection, upload_terrain,
    find_safe_path, get_risk_zones,
    find_similar_terrains, draw_path_on_mask,
)
from explainer import (
    generate_navigation_briefing,
    generate_system_explanation,
)

# ─────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────

CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "./runs/deeplabv3+_20260403_211706/best.pth"
)
MODEL_MIOU = 0.6676

print("Loading model...")
model, device, cfg = load_model(CHECKPOINT)

# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────

def run_pipeline(image_path: str,
                 use_tta: bool = True,
                 save_outputs: bool = True) -> dict:
    """
    Full pipeline for one image.

    Returns:
        result dict with all outputs for UI display
    """
    print(f"Connecting to TigerGraph...")
    conn = None
    print("✅ Pipeline ready\n")
    t_start   = time.time()
    image_id  = f"img_{uuid.uuid4().hex[:8]}"
    out_dir   = Path("./pipeline_outputs") / image_id
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Image ID:   {image_id}")
    print(f"{'='*60}")

    # ── STEP 1: Segmentation ──────────────────
    print("\n[1/5] Running segmentation...")
    t0 = time.time()
    seg_result = predict(model, device, image_path,
                         use_tta=use_tta)
    print(f"      Done in {time.time()-t0:.1f}s")

    print("\n      Class distribution:")
    for name, pct in seg_result["class_dist"].items():
        if pct > 0.01:
            bar = '█' * int(pct * 30)
            print(f"      {name:25s}: {pct:.1%}  {bar}")

    # ── STEP 2: Upload to TigerGraph ──────────
    print("\n[2/5] Building terrain graph...")
    t0 = time.time()
    terrain_summary = upload_terrain(
        conn=conn,
        image_id=image_id,
        image_path=image_path,
        seg_result=seg_result,
        model_miou=MODEL_MIOU,
    )
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      Patches: {terrain_summary['patches']}")
    print(f"      Edges:   {terrain_summary['edges']}")
    print(f"      Risks:   {len(terrain_summary['risk_zones'])}")

    # ── STEP 3: Find safest path ──────────────
    print("\n[3/5] Finding safest path (Dijkstra)...")
    t0 = time.time()
    path_result = find_safe_path(
        conn,
        image_id,
        seg_result["trav_map"],
    )
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      Hops: {path_result['hop_count']}")
    print(f"      Cost: {path_result['total_cost']}")

    # ── STEP 4: Get risk zones + similar ──────
    print("\n[4/5] Querying graph knowledge...")
    t0 = time.time()
    risk_zones = get_risk_zones(conn, image_id)
    similar    = find_similar_terrains(
        conn,
        rock_pct   = seg_result["class_dist"].get("Rocks", 0),
        veg_pct    = (seg_result["class_dist"].get("Dense_Vegetation",0) +
                      seg_result["class_dist"].get("Dry_Vegetation",0)),
        brightness = float(seg_result["orig_np"].mean()/255),
        top_k      = 5,
    )
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"      Risk zones: {len(risk_zones)}")
    print(f"      Similar terrains: {len(similar)}")

    # ── STEP 5: Claude reasoning ──────────────
    print("\n[5/5] Generating AI briefing...")
    t0 = time.time()
    briefing    = generate_navigation_briefing(
        class_dist          = seg_result["class_dist"],
        path_result         = path_result,
        risk_zones          = risk_zones,
        similar_terrains    = similar,
        avg_traversability  = float(seg_result["trav_map"].mean()),
    )
    explanation = generate_system_explanation(
        class_dist  = seg_result["class_dist"],
        path_result = path_result,
        risk_zones  = risk_zones,
        model_miou  = MODEL_MIOU,
    )
    print(f"      Done in {time.time()-t0:.1f}s")

    # ── Draw path on mask ─────────────────────
    path_visual = draw_path_on_mask(
        seg_result["color_mask"],
        path_result.get("path_ids", []),
    )

    # ── Save outputs ──────────────────────────
    if save_outputs:
        from PIL import Image as PILImage
        PILImage.fromarray(seg_result["orig_np"])\
                 .save(out_dir/"original.png")
        PILImage.fromarray(seg_result["color_mask"])\
                 .save(out_dir/"segmentation.png")
        PILImage.fromarray(seg_result["overlay"])\
                 .save(out_dir/"overlay.png")
        PILImage.fromarray(path_visual)\
                 .save(out_dir/"path.png")

        with open(out_dir/"briefing.txt", "w") as f:
            f.write(briefing)
        with open(out_dir/"summary.json", "w") as f:
            json.dump({
                "image_id":    image_id,
                "class_dist":  seg_result["class_dist"],
                "path":        path_result,
                "risk_zones":  len(risk_zones),
                "similar":     len(similar),
                "avg_trav":    float(seg_result["trav_map"].mean()),
            }, f, indent=2)

        print(f"\n      Outputs saved → {out_dir}")

    total_time = time.time() - t_start

    # ── Final result dict ─────────────────────
    result = {
        # Images (numpy arrays for Gradio)
        "original":    seg_result["orig_np"],
        "segmented":   seg_result["color_mask"],
        "overlay":     seg_result["overlay"],
        "path_visual": path_visual,

        # Data
        "image_id":    image_id,
        "class_dist":  seg_result["class_dist"],
        "path":        path_result,
        "risk_zones":  risk_zones,
        "similar":     similar,
        "avg_trav":    float(seg_result["trav_map"].mean()),

        # Text outputs
        "briefing":    briefing,
        "explanation": explanation,

        # Stats
        "total_time":  round(total_time, 2),
        "model_miou":  MODEL_MIOU,
    }

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete in {total_time:.1f}s")
    print(f"\n📋 Navigation Briefing:")
    print(f"{briefing}")
    print(f"{'='*60}")

    return result


# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else None

    if img is None:
        # Find first val image
        imgs = list(Path("./dataset/val/Color_Images")
                    .glob("*.png"))
        if not imgs:
            print("No test image found. Pass image path as argument.")
            sys.exit(1)
        img = str(imgs[0])

    print(f"Test image: {img}")
    result = run_pipeline(img)