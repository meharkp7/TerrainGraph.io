---
# HuggingFace Space config — this repo deploys to HF Spaces straight from GitHub.
title: TerrainGraph AI
emoji: 🌍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# TerrainGraph AI

Intelligent terrain analysis and safe navigation for autonomous offroad vehicles (UGVs).

Upload a desert-scene image → **DeepLabV3+** segments it into 6 terrain classes →
a **TigerGraph** terrain graph is built → **Dijkstra** finds the safest path →
an **LLM** writes a navigation briefing → results render in a live **React dashboard**.

| Surface | Entry | Deploy target |
|---|---|---|
| React dashboard | [`frontend/`](frontend/) → `TerrainIntelligence.jsx` | Vercel |
| FastAPI backend | [`api.py`](api.py) | Railway (Docker) |
| Gradio demo | [`app.py`](app.py) | HuggingFace Space |
| Weights | [`mk1647/terrain-intelligence`](https://huggingface.co/mk1647/terrain-intelligence) | HuggingFace Hub |

## How it works

```
image ──▶ DeepLabV3+ (ResNet-101, mIoU 0.6676)
              │  mask + class distribution + traversability map
              ▼
        TigerGraph terrain graph (16×16 patches, 4-connected)
              │  Python Dijkstra, bottom-center → top-center
              ▼
        risk zones + similar-terrain memory queries
              │
              ▼
        LLM briefing (Groq → Gemini → Claude → template fallback)
              │
              ▼
        React dashboard · Gradio demo · saved PNGs + JSON
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Quickstart

**1. Backend**

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in TigerGraph + LLM keys (all optional except model)
python setup_tigergraph.py  # once: create graph schema + queries
uvicorn api:app --port 8000
```

On machines without the checkpoint, fetch it first: `python download_model.py`
(downloads `best.pth` from the HF model repo into `runs/deployed/`).

**2. Frontend**

```bash
cd frontend
npm install
cp .env.example .env.local  # VITE_API_BASE=http://localhost:8000
npm run dev                 # → http://localhost:3000
```

**3. Analyze** — open the dashboard, drop/paste/click an image, hit **Analyze Terrain**.
API docs for the contract: [`docs/API.md`](docs/API.md)

**4. HF demo (no backend needed)**

```bash
pip install -r requirements.txt   # includes gradio
python app.py
```

## Project layout

```
├── api.py               # FastAPI backend (Railway): /analyze, /image, /health
├── app.py               # Gradio demo (HuggingFace Space)
├── pipeline.py          # Full pipeline: segment → graph → query → LLM brief
├── pipeline_hf.py       # Lightweight Space pipeline (no TigerGraph/LLM needed)
├── segmentor.py         # DeepLabV3+ loader + inference, traversability maps
├── terrain_graph.py     # TigerGraph schema, patch graph, Dijkstra, risk zones
├── explainer.py         # LLM layer: Groq → Gemini → Claude → template
├── utils.py             # Chart helper for the Gradio demo
├── download_model.py    # Fetch best.pth from HuggingFace Hub
├── setup_tigergraph.py  # One-time TigerGraph schema setup
├── train_fixed.py       # Training (DeepLabV3+, corrected mask decoding)
├── train_segformer.py   # Training (SegFormer-B4 variant)
├── Dockerfile / railway.toml / requirements.txt
├── frontend/            # React + Vite dashboard (Vercel) — see frontend/README.md
├── dataset/             # train/val images + masks (gitignored)
├── runs/                # checkpoints + training history (gitignored)
└── docs/                # architecture, API, deployment, training guides
```

## Model

DeepLabV3+ with ResNet-101 encoder, 512×512 input, trained on 2,857 desert
images (317 val) with corrected Duality-Falcon mask decoding. **mIoU 0.6676.**

| Class | IoU | Traversability | Risk |
|---|---|---|---|
| Dense Vegetation | 0.755 | 0.45 | MED |
| Dry Vegetation | 0.700 | 0.75 | LOW |
| Ground Objects | 0.453 | 0.30 | MED-HIGH |
| Rocks | 0.464 | 0.15 | HIGH |
| Landscape | 0.651 | 0.95 | LOW |
| Sky | 0.982 | 0.00 | N/A |

Training details: [`docs/TRAINING.md`](docs/TRAINING.md)

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT` | yes | Path to `best.pth` (or set by `download_model.py`) |
| `TIGERGRAPH_HOST` / `TIGERGRAPH_API_TOKEN` / `TIGERGRAPH_GRAPH_NAME` | no¹ | Graph memory; pathfinding still works without it |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | no² | LLM briefing, tried in that order |
| `HF_MODEL_REPO` / `HF_TOKEN` | no | Weight download (defaults to public repo) |
| `VITE_API_BASE` (frontend) | yes³ | Backend URL baked into the frontend build |

¹ Without TigerGraph, uploads/queries log a warning and the pipeline uses local
risk zones + Python Dijkstra. ² Without any LLM key, briefings use a template.
³ Dev default `http://localhost:8000` via the Vite proxy.

Template: [`.env.example`](.env.example), [`frontend/.env.example`](frontend/.env.example)

## Deployment

[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) covers all three targets and the
checklist that keeps them in sync:

- **Railway** (backend): Docker build, model fetched at build time, `/health` check
- **HuggingFace Space** (demo): deploys from this repo — `app.py` + `requirements.txt`
- **Vercel** (frontend): root directory `frontend/`, `VITE_API_BASE` env var

## API (summary)

```bash
curl -s http://localhost:8000/health
curl -s -F "file=@scene.png" http://localhost:8000/analyze
curl -s http://localhost:8000/image/<image_id>/path --output path.png
```

Full reference: [`docs/API.md`](docs/API.md) · Interactive: `/docs` (Swagger UI)
