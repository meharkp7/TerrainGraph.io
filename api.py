"""
api.py — Railway-safe. /health responds instantly, model loads on first request.
"""
import os, tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Terrain Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

OUTPUT_DIR = Path("./pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

_run_pipeline = None
_model_miou   = 0.0
_model_error  = None


def _ensure_pipeline():
    global _run_pipeline, _model_miou, _model_error
    if _run_pipeline is not None:
        return True
    if _model_error is not None:
        return False
    try:
        import pipeline as _p
        _run_pipeline = _p.run_pipeline
        _model_miou   = getattr(_p, "MODEL_MIOU", 0.0)
        print("✅ Pipeline loaded")
        return True
    except Exception as e:
        _model_error = str(e)
        print(f"❌ Pipeline load failed: {e}")
        return False


# ── INSTANT health — Railway checks this immediately on startup ───
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "model_miou": _model_miou,
            "pipeline_ready": _run_pipeline is not None}


@app.get("/ready")
def ready():
    return {"ready": _run_pipeline is not None, "error": _model_error}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not _ensure_pipeline():
        raise HTTPException(503, f"Model not loaded: {_model_error}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = _run_pipeline(tmp_path, use_tta=False, save_outputs=True)
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
        "avg_trav":    round(result["avg_trav"], 4),
        "briefing":    result["briefing"],
        "explanation": result.get("explanation", result["briefing"]),
        "total_time":  result["total_time"],
        "model_miou":  result.get("model_miou", _model_miou),
    })


@app.get("/image/{image_id}/{img_type}")
def get_image(image_id: str, img_type: str):
    valid = {"original","segmented","overlay","path"}
    if img_type not in valid:
        raise HTTPException(400, f"img_type must be one of {valid}")
    base = OUTPUT_DIR / image_id
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