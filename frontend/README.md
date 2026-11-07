# TerrainGraph AI — Dashboard (React + Vite)

Live monitoring UI for the TerrainGraph backend: upload → segmentation,
Dijkstra safe path, risk zones, traversability charts, LLM briefing.

## Run

```bash
npm install
cp .env.example .env.local   # VITE_API_BASE=http://localhost:8000
npm run dev                  # → http://localhost:3000 (proxies /analyze /image /health)
```

Production: `npm run build` → `dist/` (deployed on Vercel, root dir `frontend/`).
Set **`VITE_API_BASE`** to the public backend URL — it's baked in at build time.

## Backend contract

All API access goes through [`src/api.js`](src/api.js):

| Call | Endpoint |
|---|---|
| `checkHealth()` | `GET /health` → header status pill + offline banner |
| `analyzeImage(file)` | `POST /analyze` (multipart `file`) → full result JSON |
| `imageUrl(id, type)` | `GET /image/{id}/{original\|segmented\|overlay\|path}` |

Full request/response shapes: [`../docs/API.md`](../docs/API.md).
Requests send `ngrok-skip-browser-warning` so ngrok-hosted backends work.

## Layout

```
src/
  main.jsx                 # entry (imports App + index.css)
  App.jsx                  # renders TerrainIntelligence
  TerrainIntelligence.jsx  # dashboard: upload, viewer tabs, charts, briefing, logs
  api.js                   # backend client (base URL, health, analyze, image URLs)
  index.css                # page shell + responsive grid breakpoints
```

- Charts use the `chart.js` npm package (bundled, works offline).
- Upload via click, drag-drop, or paste (images ≤ 15 MB).
- Grids collapse at 1100 px (two columns) and 760 px (single column).
- `WHY IT WORKS` tab shows the `explanation` field from `/analyze`.
