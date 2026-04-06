"""
api.py — FastAPI backend, Railway-safe
Nothing connects or loads at import time.
"""

import os, tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Lazy pipeline import — if it fails, API still starts and
# returns a 503 on /analyze rather than killing the container.
_pipeline_loaded  = False
_pipeline_error   = None
run_pipeline      = None
MODEL_MIOU        = 0.0

def _load_pipeline():
    global run_pipeline, MODEL_MIOU, _pipeline_loaded, _pipeline_error
    if _pipeline_loaded:
        return
    try:
        import pipeline as _p
        run_pipeline     = _p.run_pipeline
        MODEL_MIOU       = _p.MODEL_MIOU
        _pipeline_loaded = True
    except Exception as e:
        _pipeline_error  = str(e)
        _pipeline_loaded = True   # don't retry forever
        print(f"⚠️  Pipeline load error: {e}")


app = FastAPI(title="Terrain Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

OUTPUT_DIR = Path("./pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup():
    """Load model in background after server is ready — avoids OOM on boot."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_pipeline)


@app.get("/")
def root():
    return {"status":"ok","model_miou":MODEL_MIOU,
            "pipeline_ready": run_pipeline is not None}

@app.get("/health")
def health():
    return {"status":"healthy","model_miou":MODEL_MIOU,
            "tigergraph":os.getenv("TIGERGRAPH_HOST","not set"),
            "pipeline_ready": run_pipeline is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    _load_pipeline()   # ensure loaded (no-op if already done)
    if run_pipeline is None:
        raise HTTPException(503, f"Pipeline not ready: {_pipeline_error}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = run_pipeline(tmp_path, use_tta=False, save_outputs=True)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    return JSONResponse({
        "image_id":    result["image_id"],
        "class_dist":  result["class_dist"],
        "path":        result["path"],
        "risk_zones":  result["risk_zones"],
        "similar":     result["similar"],
        "avg_trav":    round(result["avg_trav"],4),
        "briefing":    result["briefing"],
        "explanation": result.get("explanation", result["briefing"]),
        "total_time":  result["total_time"],
        "model_miou":  result.get("model_miou", MODEL_MIOU),
    })

@app.get("/image/{image_id}/{img_type}")
def get_image(image_id: str, img_type: str):
    valid = {"original","segmented","overlay","path"}
    if img_type not in valid:
        raise HTTPException(400, f"img_type must be one of {valid}")
    base = OUTPUT_DIR/image_id
    if not base.exists():
        raise HTTPException(404, f"No outputs for '{image_id}'")
    aliases = {
        "original":  ["original.png","input.png"],
        "segmented": ["segmented.png","segmentation.png","colored_mask.png"],
        "overlay":   ["overlay.png","blend.png","segmented.png"],
        "path":      ["path.png","path_viz.png","overlay.png","segmented.png"],
    }
    for fname in aliases[img_type]:
        p = base/fname
        if p.exists(): return FileResponse(str(p), media_type="image/png")
    existing = [f.name for f in base.iterdir()]
    raise HTTPException(404, f"'{img_type}' not found. Available: {existing}")

@app.get("/debug/{image_id}")
def debug(image_id: str):
    d = OUTPUT_DIR/image_id
    if not d.exists(): raise HTTPException(404)
    return {"files": sorted(f.name for f in d.iterdir())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)