import numpy as np
from PIL import Image
from segmentor import load_model, predict
from huggingface_hub import hf_hub_download
import heapq

# 🔥 Load model from HF
CHECKPOINT = hf_hub_download(
    repo_id="mk1647/terrain-intelligence",
    filename="best.pth"
)

model = None
device = None

def get_model():
    global model, device
    if model is None:
        print("🔥 Loading model...")
        model, device, _ = load_model(CHECKPOINT)
    return model, device


# 🔥 Dijkstra pathfinding
def compute_path(trav_map):
    h, w = trav_map.shape
    pq = [(0, 0, 0)]
    parent = {}
    visited = set()

    while pq:
        cost, x, y = heapq.heappop(pq)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        if (x, y) == (h - 1, w - 1):
            break

        for dx, dy in [(1,0),(0,1),(-1,0),(0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                new_cost = cost + (1 - trav_map[nx, ny])
                heapq.heappush(pq, (new_cost, nx, ny))
                parent[(nx, ny)] = (x, y)

    path = []
    cur = (h - 1, w - 1)
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append((0, 0))
    path.reverse()

    return path


def draw_path(mask, path):
    mask = mask.copy()
    for x, y in path:
        mask[x, y] = [255, 0, 0]
    return mask


def generate_briefing(class_dist, avg_trav):
    dominant = max(class_dist, key=class_dist.get)
    risk = "high" if avg_trav < 0.4 else "moderate" if avg_trav < 0.7 else "low"

    return f"""
Terrain is dominated by {dominant}.
Overall traversability is {avg_trav:.2f}, indicating {risk} difficulty.
Recommended path avoids low-access regions and minimizes risk.
"""


def run_pipeline(image):
    model, device = get_model()

    image_np = np.array(image)
    temp_path = "temp.png"
    Image.fromarray(image_np).save(temp_path)

    seg = predict(model, device, temp_path, use_tta=False)

    trav_map = seg["trav_map"]
    path = compute_path(trav_map)

    path_img = draw_path(seg["overlay"], path)

    briefing = generate_briefing(
        seg["class_dist"],
        float(trav_map.mean())
    )

    return {
        "original": seg["orig_np"],
        "overlay": path_img,
        "briefing": briefing,
        "class_dist": seg["class_dist"]
    }