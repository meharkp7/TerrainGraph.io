"""
api.py — Railway-safe.
/health responds in <100ms always.
Model loads in the background AFTER uvicorn is up and serving /health.
This prevents Railway from OOM-killing the process before the port is bound.
"""
import os
import gc
import threading
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Terrain Intelligence API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

OUTPUT_DIR = Path("./pipeline_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── State ──────────────────────────────────────────────────────────────────────
_run_pipeline   = None
_model_miou     = 0.0
_model_error    = None
_model_loading  = False   # True while background thread is running
_model_ready    = False   # True once successfully loaded


def _load_pipeline_background():
    """
    Runs in a daemon thread so uvicorn can bind and pass the Railway
    healthcheck BEFORE we try to allocate ~400MB for the model weights.
    """
    global _run_pipeline, _model_miou, _model_error, _model_loading, _model_ready
    _model_loading = True
    try:
        _log_memory("before pipeline import")
        import pipeline as _p
        _log_memory("after pipeline import")

        # Force the segmentor to actually load (lazy singleton inside pipeline)
        _ = _p._get_segmentor()
        _log_memory("after model load")

        _run_pipeline = _p.run_pipeline
        _model_miou   = getattr(_p, "MODEL_MIOU", 0.0)
        _model_ready  = True
        print(f"✅ Pipeline ready — mIoU={_model_miou:.4f}")
    except Exception as e:
        _model_error = str(e)
        print(f"❌ Pipeline load failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        _model_loading = False
        gc.collect()


def _log_memory(label: str):
    """Log RSS memory if psutil is available (non-fatal if not)."""
    try:
        import psutil, os
        mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"  [mem] {label}: {mb:.0f} MB RSS")
    except Exception:
        pass


# ── Startup: bind the port first, THEN load the model ─────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    uvicorn calls this after the socket is bound.
    Railway healthcheck can now reach /health immediately.
    We kick off model loading in a background thread.
    """
    print("🚀 uvicorn bound — starting background model load...")
    t = threading.Thread(target=_load_pipeline_background, daemon=True)
    t.start()


# ── Health — always instant ────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Must return 200 within Railway's healthcheckTimeout.
    This endpoint does ZERO work — no model access, no imports.
    """
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_ready":   _model_ready,
        "model_loading": _model_loading,
        "model_miou":    _model_miou,
        "model_error":   _model_error,
    }


@app.get("/ready")
def ready():
    """Poll this to know when the model has finished loading."""
    return {
        "ready":   _model_ready,
        "loading": _model_loading,
        "error":   _model_error,
    }


# ── Analyze ────────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if _model_loading and not _model_ready:
        # Model is still loading — tell client to retry
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Poll /ready and retry in ~30s."
        )
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_model_error or 'unknown error'}"
        )
    if not file.content_type or not file.content_type.startswith("image/"):
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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

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


# ── Image serving ──────────────────────────────────────────────────────────────
@app.get("/image/{image_id}/{img_type}")
def get_image(image_id: str, img_type: str):
    valid = {"original", "segmented", "overlay", "path"}
    if img_type not in valid:
        raise HTTPException(400, f"img_type must be one of {valid}")
    base = OUTPUT_DIR / image_id
    if not base.exists():
        raise HTTPException(404, f"No outputs for '{image_id}'")
    aliases = {
        "original":  ["original.png", "input.png"],
        "segmented": ["segmented.png", "segmentation.png", "colored_mask.png"],
        "overlay":   ["overlay.png", "blend.png", "segmented.png"],
        "path":      ["path.png", "path_viz.png", "overlay.png", "segmented.png"],
    }
    for fname in aliases[img_type]:
        p = base / fname
        if p.exists():
            return FileResponse(str(p), media_type="image/png")
    existing = [f.name for f in base.iterdir()]
    raise HTTPException(404, f"'{img_type}' not found. Available: {existing}")


@app.get("/debug/{image_id}")
def debug(image_id: str):
    d = OUTPUT_DIR / image_id
    if not d.exists():
        raise HTTPException(404)
    return {"files": sorted(f.name for f in d.iterdir())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)