# Architecture

End-to-end flow for one image (`pipeline.run_pipeline`):

## Stage 1 — Segmentation (`segmentor.py` or HF Space)

- DeepLabV3+ / ResNet-101, 512×512 input, ImageNet normalization.
- Optional test-time augmentation: horizontal-flip averaging (`use_tta`, local only).
- Two backends, one contract (`INFERENCE_BACKEND`):
  - `local` — torch in-process (`segmentor.predict`), needs ~1GB+ RAM.
  - `space` — HTTP call to the HF Space `/segment` endpoint
    (`segmentor_space.predict_via_space`), torch-free, fits 512MB.
- Wire format: Space returns `mask.png` (class ids 0-5), `conf.png`
  (confidence 0-255) and `class_dist` JSON; the backend rebuilds the
  identical dict local inference returns (see `tests/test_space_codec.py`).
- Outputs: class mask (6 classes), per-pixel confidence, class distribution,
  **traversability map** (per-class score → pixel), color mask, 50/50 overlay.

| # | Class | Traversability | Risk |
|---|---|---|---|
| 0 | Dense_Vegetation | 0.45 | MED |
| 1 | Dry_Vegetation | 0.75 | LOW |
| 2 | Ground_Objects | 0.30 | MED-HIGH |
| 3 | Rocks | 0.15 | HIGH |
| 4 | Landscape | 0.95 | LOW |
| 5 | Sky | 0.00 | N/A |

## Stage 2 — Terrain graph + pathfinding (`kuzu_graph.py` / `terrain_graph.py`)

Two interchangeable stores behind `GRAPH_BACKEND` (`kuzu` default, free and
embedded; `tigergraph` needs cloud credits). Same node/edge shape, same
`{"attributes": {...}}` query contract; local building shared in
`terrain_graph.build_terrain_local`.

- The 512×512 mask is tiled into 32px patches → **16×16 grid (256 nodes)**.
- Each patch becomes a `TerrainPatch` vertex (dominant class, traversability,
  risk level, confidence, class distribution).
- 4-connected `AdjacentTo` edges with
  `transition_cost = 1 − mean(traversability)` — Dijkstra minimizes cost,
  so the path prefers Landscape and Dry Vegetation.
- Patch ids use `:` separators (`{image_id}:r{row}:c{col}`) so parsing is
  unambiguous even when `image_id` contains underscores.
- **Dijkstra runs in Python** over the local patch/edge lists
  (bottom-center → top-center). No database pathfinding needed, so a path is
  always returned even if the graph store is unreachable.

## Stage 3 — Graph memory queries

- `upload_terrain` stores `ImageRecord`, patches, edges, `RiskZone`s
  (clusters of ≥2 adjacent HIGH/MED-HIGH patches) and the `SafePath`.
- `getRiskZones(image_id)` — risk zones for this image (falls back to the
  locally computed zones when the store is unavailable).
- `findSimilarTerrains(rock_pct, veg_pct, brightness)` — past images with
  similar composition, ranked by mIoU, for historical context.

Kuzu creates its schema automatically on first connect (verify explicitly
with `python setup_kuzu.py`); TigerGraph mode installs GSQL schema + queries
once via `python setup_tigergraph.py`.
Vertex/edge types: `TerrainPatch`, `ImageRecord`, `RiskZone`, `SafePath`,
`AdjacentTo`, `BelongsTo`, `HasRiskZone`, `HasPath`.

## Stage 4 — LLM briefing (`explainer.py`)

One provider call per text, with fallback chain:

1. Groq (`llama-3.3-70b-versatile`) → 2. Gemini (`gemini-2.0-flash-lite`) →
   3. Claude → 4. deterministic template (no API needed).

- `generate_navigation_briefing` — ≤100-word ops briefing grounded only in
  the pipeline outputs (risk zones named, path stated, one recommendation).
- `generate_system_explanation` — 60–80-word "what happened and why it
  matters" pitch of the result.

## Outputs

`pipeline_outputs/<image_id>/`: `original.png`, `segmented.png`,
`overlay.png`, `path.png` (yellow Dijkstra path on the color mask),
`briefing.txt`, `summary.json`. Served over HTTP by `api.py`
(`GET /image/{image_id}/{original|segmented|overlay|path}`).

## HF Space variant (`pipeline_hf.py` + `app.py`)

Same stages, Space-safe: weights download from the Hub, Dijkstra runs on a
block-averaged 64×64 traversability grid (~0.01 s), briefing is a template.
No TigerGraph, no LLM keys.
