# API Reference

Base URL: `http://localhost:8000` locally, or your Railway/ngrok host.
Interactive docs: `GET /docs` (Swagger UI). CORS is open (`*`).

## `GET /health`

Liveness + model info. The frontend calls this on load for its API status pill.

```bash
curl -s $API/health
```

```json
{
  "status": "healthy",
  "model_loaded": false,
  "model_miou": 0.6676,
  "tigergraph": "https://<host> or \"not set\""
}
```

`model_loaded` is `false` until the first `/analyze` call — the 550MB
weights load lazily so container boot stays light (see deployment notes).

## `POST /analyze`

Multipart upload, field name **`file`**. Runs the full pipeline
(segment → graph → Dijkstra → LLM brief).

```bash
curl -s -F "file=@scene.png" $API/analyze
```

Success response:

```json
{
  "image_id": "img_a1b2c3d4",
  "class_dist": { "Landscape": 0.41, "Dry_Vegetation": 0.28, "...": "..." },
  "path": { "hop_count": 14, "total_cost": 1.23, "method": "python_dijkstra" },
  "risk_zones": [{ "attributes": { "class_name": "Rocks", "severity": "HIGH", "patch_count": 5, "center_row": 3, "center_col": 9 } }],
  "similar": [{ "attributes": { "dominant_class": "Landscape", "avg_traversability": 0.71, "miou_achieved": 0.68 } }],
  "avg_trav": 0.63,
  "briefing": "Scene is 41% Landscape ...",
  "explanation": "This system fuses DeepLabV3+ ...",
  "total_time": 12.4,
  "model_miou": 0.6676
}
```

- `class_dist` values are fractions (0–1).
- `risk_zones` / `similar` use TigerGraph `{"attributes": {...}}` shape even
  for locally computed fallbacks, so clients parse one shape.
- Errors: `400` non-image upload, `500` pipeline failure (message in `detail`).

## `GET /image/{image_id}/{type}`

Saved output PNGs. `{type}` is one of `original | segmented | overlay | path`.
Each type also accepts common filename aliases from older runs.

```bash
curl -s $API/image/img_a1b2c3d4/path --output path.png
```

Errors: `400` bad type or malformed id, `404` unknown id / missing file
(the 404 message lists which files *do* exist for that id).

## `GET /debug/{image_id}`

Lists saved files for an id — handy when an `/image` call 404s.

```json
{ "image_id": "img_a1b2c3d4", "files": ["briefing.txt", "original.png", "overlay.png", "path.png", "segmented.png", "summary.json"] }
```

## Frontend contract (`frontend/src/api.js`)

The dashboard is the reference client: `checkHealth()` → `GET /health`,
`analyzeImage(file)` → `POST /analyze`, `imageUrl(id, type)` → `GET /image/...`.
Requests carry the `ngrok-skip-browser-warning` header so ngrok-hosted
backends don't serve the interstitial page to `fetch()`.
