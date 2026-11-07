"""
segmentor_space.py
──────────────────
Torch-free segmentation path for lean backends (e.g. Railway 512MB).

- encode_mask_files(): Space side — pack mask + confidence into PNG files.
- predict_via_space(): Railway side — call the HF Space /segment endpoint
  and rebuild the exact dict shape segmentor.predict() returns, so the
  rest of the pipeline (graph, Dijkstra, LLM) runs unchanged.

Needs only numpy + Pillow (+ gradio_client for live calls, imported lazily
inside get_client so this module stays importable without it).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from terrain_const import CLASS_NAMES, NUM_CLASSES, TRAVERSABILITY, PALETTE


class SpaceInferenceError(RuntimeError):
    """The HF inference Space call failed (down, waking up, misconfigured)."""


def encode_mask_files(mask: np.ndarray, confidence: np.ndarray,
                      out_dir=None) -> tuple:
    """
    Pack class mask (HxW int, values 0-5) + confidence (HxW float 0-1)
    into two grayscale PNGs. Returns (mask_path, conf_path).
    """
    d = Path(out_dir or tempfile.mkdtemp(prefix="tgmask_"))
    d.mkdir(parents=True, exist_ok=True)
    mask_path = str(d / "mask.png")
    conf_path = str(d / "conf.png")
    Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
    Image.fromarray(
        (np.clip(confidence, 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).save(conf_path)
    return mask_path, conf_path


def decode_space_result(mask_path: str, conf_path: str, class_dist: dict,
                        image_path: str, img_size: int = 512) -> dict:
    """
    Rebuild the segmentor.predict() dict from Space outputs + local image.
    Keys: mask, confidence, class_dist, trav_map, color_mask, overlay, orig_np.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    conf = np.array(Image.open(conf_path).convert("L")).astype(np.float32) / 255.0

    if int(mask.max()) >= NUM_CLASSES:
        raise SpaceInferenceError(
            f"mask has invalid class id {mask.max()} "
            f"(expected 0-{NUM_CLASSES - 1})")

    if mask.shape != (img_size, img_size):
        mask = np.array(Image.fromarray(mask).resize(
            (img_size, img_size), Image.NEAREST))
        conf = np.array(Image.fromarray(
            (conf * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR)).astype(np.float32) / 255.0

    trav_map = np.zeros_like(mask, dtype=np.float32)
    for cls_idx, score in TRAVERSABILITY.items():
        trav_map[mask == cls_idx] = score

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color_mask[mask == c] = PALETTE[c]

    orig = np.array(Image.open(image_path).convert("RGB").resize(
        (img_size, img_size), Image.BILINEAR))
    overlay = (orig * 0.5 + color_mask * 0.5).astype(np.uint8)

    dist = {k: float(v) for k, v in class_dist.items() if k in CLASS_NAMES}

    return {
        "mask":        mask,
        "confidence":  conf,
        "class_dist":  dist,
        "trav_map":    trav_map,
        "color_mask":  color_mask,
        "overlay":     overlay,
        "orig_np":     orig,
    }


_client = None


def get_client():
    """Connect to the HF inference Space once (reads HF_SPACE_ID/HF_TOKEN)."""
    global _client
    if _client is None:
        space = os.getenv("HF_SPACE_ID", "").strip()
        if not space:
            raise SpaceInferenceError(
                "HF_SPACE_ID is not set — point it at the inference Space "
                "(e.g. your-user/terrain-space). See docs/DEPLOYMENT.md.")
        from gradio_client import Client  # lazy: keeps module lean-safe
        # Hard timeout: a dead/sleeping Space must fail fast into OUR 502
        # (with CORS headers + retry message), never hang into an edge 502.
        _client = Client(space, hf_token=os.getenv("HF_TOKEN") or None,
                         httpx_kwargs={"timeout": 300.0})
    return _client


def predict_via_space(image_path: str, img_size: int = 512) -> dict:
    """Run segmentation on the HF Space; same return shape as predict()."""
    client = get_client()
    try:
        mask_path, conf_path, class_dist = client.predict(
            image_path, api_name="/segment")
    except SpaceInferenceError:
        raise
    except Exception as e:
        raise SpaceInferenceError(
            f"inference Space call failed ({e}). The free Space may be "
            f"waking up from sleep — retry in a minute.") from e
    return decode_space_result(
        mask_path, conf_path, dict(class_dist), image_path, img_size)
