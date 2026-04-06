"""
api.py
──────
FastAPI backend — bridges React frontend to pipeline.py
Serves images from pipeline_outputs/<image_id>/<type>.png

Run: uvicorn api:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from pipeline import run_pipeline, MODEL_MIOU

app = FastAPI(title="Terrain Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("./pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "model_miou": MODEL_MIOU}


@app.get("/health")
def health():
    return {
        "status":     "healthy",
        "model_miou": MODEL_MIOU,
        "tigergraph": os.getenv("TIGERGRAPH_HOST", "not set"),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accept image upload → run full pipeline → return JSON results."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, use_tta=False, save_outputs=True)
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return JSONResponse({
        "image_id":    result["image_id"],
        "class_dist":  result["class_dist"],        # fractions 0-1
        "path":        result["path"],               # {hop_count, total_cost}
        "risk_zones":  result["risk_zones"],         # [{attributes:{...}}]
        "similar":     result["similar"],            # [{attributes:{...}}]
        "avg_trav":    round(result["avg_trav"], 4),
        "briefing":    result["briefing"],
        "explanation": result.get("explanation", result["briefing"]),
        "total_time":  result["total_time"],
        "model_miou":  result.get("model_miou", MODEL_MIOU),
    })


@app.get("/image/{image_id}/{img_type}")
def get_image(image_id: str, img_type: str):
    """
    Serve saved output images.
    Tries the canonical filename first, then common aliases.
    """
    valid = {"original", "segmented", "overlay", "path"}
    if img_type not in valid:
        raise HTTPException(400, f"Invalid image type '{img_type}'. Use: {valid}")

    base_dir = OUTPUT_DIR / image_id
    if not base_dir.exists():
        raise HTTPException(404, f"No outputs found for image_id '{image_id}'")

    # Try canonical name first, then aliases
    aliases = {
        "original":  ["original.png", "input.png",       "image.png"],
        "segmented": ["segmented.png", "segmentation.png", "colored_mask.png", "seg.png"],
        "overlay":   ["overlay.png",  "blend.png",        "segmented.png"],
        "path":      ["path.png",     "path_viz.png",     "overlay.png", "segmented.png"],
    }

    for filename in aliases[img_type]:
        p = base_dir / filename
        if p.exists():
            return FileResponse(str(p), media_type="image/png")

    # Nothing found — list what IS there to help debug
    existing = [f.name for f in base_dir.iterdir()] if base_dir.exists() else []
    raise HTTPException(
        404,
        f"'{img_type}.png' not found for {image_id}. "
        f"Files in output dir: {existing}"
    )


@app.get("/debug/{image_id}")
def debug_outputs(image_id: str):
    """List all files saved for an image_id."""
    d = OUTPUT_DIR / image_id
    if not d.exists():
        raise HTTPException(404, f"No output dir for {image_id}")
    return {"image_id": image_id, "files": sorted(f.name for f in d.iterdir())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)