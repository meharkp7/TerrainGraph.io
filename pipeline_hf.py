"""
pipeline_hf.py
──────────────
Lightweight pipeline for the HuggingFace Space demo (app.py).

Same stages as pipeline.py, minus the services a Space can't reach:
  1. Segmentation (DeepLabV3+, weights from the HF model repo)
  2. Dijkstra pathfinding on a downsampled traversability grid
     (pure Python — no TigerGraph needed)
  3. Template briefing (no LLM API keys needed)

Kept import-compatible with pipeline.py's result dict where it matters:
  original / overlay / briefing / class_dist
"""

import heapq
import tempfile
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from segmentor import load_model, predict

HF_REPO = "mk1647/terrain-intelligence"

# Pathfinding grid: block-average the 512×512 traversability map to this.
# 64×64 = 4096 nodes — Dijkstra runs in ~1s on a free CPU Space.
# (Full-resolution Dijkstra would be 262k nodes and time out.)
PATH_GRID = 64

model = None
device = None


def get_checkpoint() -> str:
    """Download weights once (cached by huggingface_hub), return local path."""
    return hf_hub_download(repo_id=HF_REPO, filename="best.pth")


def get_model():
    global model, device
    if model is None:
        print("Loading model from HuggingFace...")
        model, device, _ = load_model(get_checkpoint())
    return model, device


def _block_average(trav_map: np.ndarray, grid: int = PATH_GRID) -> np.ndarray:
    """Block-average H×W traversability map down to grid×grid."""
    h, w = trav_map.shape
    bh, bw = h // grid, w // grid
    small = trav_map[: grid * bh, : grid * bw].reshape(grid, bh, grid, bw)
    return small.mean(axis=(1, 3)).astype(np.float32)


def compute_path(trav_map: np.ndarray, grid: int = PATH_GRID) -> list:
    """
    Dijkstra from bottom-center to top-center on the downsampled grid.
    Returns path as [(row, col), ...] in ORIGINAL pixel coordinates.
    """
    small = _block_average(trav_map, grid)
    gh, gw = small.shape
    start, goal = (gh - 1, gw // 2), (0, gw // 2)

    pq = [(0.0, start)]
    dist = {start: 0.0}
    parent = {}
    visited = set()

    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        x, y = node
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < gh and 0 <= ny < gw and (nx, ny) not in visited:
                new_cost = cost + (1.0 - float(small[nx, ny]))
                if new_cost < dist.get((nx, ny), float("inf")):
                    dist[(nx, ny)] = new_cost
                    parent[(nx, ny)] = node
                    heapq.heappush(pq, (new_cost, (nx, ny)))

    # Reconstruct in grid coords, then scale to pixel coords (block centers)
    h, w = trav_map.shape
    path = []
    cur = goal
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()

    return [
        (int((gx + 0.5) * h / gh), int((gy + 0.5) * w / gw))
        for gx, gy in path
    ]


def draw_path(overlay: np.ndarray, path: list) -> np.ndarray:
    """Draw the safe path as a visible yellow line + dots (BGR for cv2)."""
    img = overlay.copy()
    pts = [(c, r) for r, c in path]  # (x, y) for cv2
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (0, 215, 255), 3)
    for x, y in pts[:: max(1, len(pts) // 40)]:
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
    return img


def generate_briefing(class_dist: dict, avg_trav: float, n_hops: int) -> str:
    dominant = max(class_dist, key=class_dist.get)
    dom_pct = class_dist[dominant] * 100
    risk = "high" if avg_trav < 0.4 else "moderate" if avg_trav < 0.7 else "low"

    parts = [f"{k.replace('_', ' ')} {v * 100:.0f}%" for k, v in class_dist.items() if v > 0.01]
    return (
        f"Terrain is dominated by {dominant.replace('_', ' ')} ({dom_pct:.0f}%). "
        f"Composition: {', '.join(parts)}. "
        f"Overall traversability is {avg_trav:.2f}, indicating {risk} difficulty. "
        f"Dijkstra found a {n_hops}-hop safe path (bottom-center to top-center) "
        f"that avoids low-traversability regions. "
        f"Recommendation: proceed via high-traversability classes "
        f"(Landscape, Dry Vegetation) and avoid Rocks and Ground Objects."
    )


def run_pipeline(image) -> dict:
    """Full HF demo pipeline. `image` is a PIL Image (from Gradio)."""
    seg_model, seg_device = get_model()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(np.array(image)).save(tmp_path)
        seg = predict(seg_model, seg_device, tmp_path, use_tta=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    trav_map = seg["trav_map"]
    avg_trav = float(trav_map.mean())
    path = compute_path(trav_map)
    path_img = draw_path(seg["overlay"], path)
    briefing = generate_briefing(seg["class_dist"], avg_trav, len(path))

    return {
        "original": seg["orig_np"],
        "segmented": seg["color_mask"],
        "overlay": path_img,
        "briefing": briefing,
        "class_dist": seg["class_dist"],
        "avg_trav": round(avg_trav, 4),
        "path_hops": len(path),
        # Raw outputs for the /segment machine endpoint (demo ignores these)
        "mask": seg["mask"],
        "confidence": seg["confidence"],
    }
