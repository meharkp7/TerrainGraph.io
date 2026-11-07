# Deployment

Three targets, one repo. They share `requirements.txt`, `segmentor.py`,
and the weights on the Hub — the checklist at the bottom keeps them in sync.

## Backend → Railway (FastAPI + Docker)

- Build: `Dockerfile` (`railway.toml` selects it). Model weights download
  from `HF_MODEL_REPO` at build time into `runs/deployed/best.pth`;
  `MODEL_CHECKPOINT` is set to that path in the image.
- Start: `uvicorn api:app --host 0.0.0.0 --port ${PORT}`
- Health check: `GET /health` (timeout 120 s — torch cold-start is slow).
- Env vars on Railway: `TIGERGRAPH_*`, `GROQ_API_KEY`/`GEMINI_API_KEY`/
  `ANTHROPIC_API_KEY`, `HF_MODEL_REPO`, `HF_TOKEN` (only for private repos).
- CPU-only torch is pinned in `requirements.txt`; `opencv-python-headless`
  avoids the ~200 MB Qt/RAM hit of the GUI build.
- First boot also needs the graph schema once: run
  `python setup_tigergraph.py` locally with the same TigerGraph credentials
  (or accept graceful local-only mode).

## Railway troubleshooting

**Build log doesn't match this repo's `Dockerfile`?** Then Railway is
auto-detecting (Railpack) instead of using `railway.toml` — e.g. a log
showing `python:3.10-slim` (from `runtime.txt`) plus its own
`pip install torch ... --index-url ...` step. That generated install is
broken: `--index-url` *replaces* PyPI, so transitive build deps
(`flit_core`) can't be found and the build fails. Fix:

1. Commit and push this repo — Railway only builds pushed commits.
2. Railway dashboard → service → Settings → Builder: must be
   **Dockerfile** (not Railpack/Nixpacks).
3. Same page: Branch = `main`, Root Directory = empty (repo root).
4. Deploy → Redeploy (clear the build cache if offered).

**`typing_extensions` / `flit_core` errors in a Docker build?** The repo
`Dockerfile` upgrades pip before installing requirements, which avoids the
old-pip wheel rejection that triggers the `flit_core` sdist path. If you
see it anyway, you're not building the repo `Dockerfile` — see above.

## Temporary tunnels (when Railway isn't up)

**ngrok, permanent-ish (free):** reserve one free static domain at
dashboard.ngrok.com → Domains, then always serve with it — the URL
survives agent restarts (it still dies if the laptop sleeps or the
agent stops):

```bash
ngrok http 8000 --domain=<your-name>.ngrok-free.dev
```

Keep it alive on macOS with a LaunchAgent (`~/Library/LaunchAgents/ngrok-api.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>Label</key><string>ngrok-api</string>
  <key>ProgramArguments</key><array>
    <string>/usr/local/bin/ngrok</string><string>http</string>
    <string>8000</string><string>--domain</string>
    <string>&lt;your-name&gt;.ngrok-free.dev</string>
  </array>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/ngrok-api.plist
```

**More permanent free alternative:** Tailscale Funnel gives a stable
public `https://` URL with no rotation and no own-domain needed:

```bash
tailscale up
tailscale funnel 8000
```

Whichever tunnel you use, update `VITE_API_BASE` and rebuild/redeploy the
frontend — the URL is baked in at build time. Tunnels are a stopgap;
Railway above remains the permanent backend.

## Demo → HuggingFace Space (Gradio)

The Space deploys **from this repo**:

1. Create a Space (Gradio SDK, CPU) backed by / mirroring this GitHub repo,
   or push directly:
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space>
   git push hf main
   ```
2. Space entrypoint is `app.py`; dependencies come from the same root
   `requirements.txt` (it includes `gradio`, `huggingface_hub`, `matplotlib`).
3. The Space header lives at the top of [`README.md`](../README.md)
   (`sdk: gradio`, `app_file: app.py`) — that's what makes the file double
   as the Space card. Keep it.
4. Weights: `pipeline_hf.py` pulls `best.pth` from
   `mk1647/terrain-intelligence` on first run (cached after that).
5. No secrets needed: no TigerGraph, no LLM keys (template briefing).

## Frontend → Vercel (React + Vite)

- Root directory: `frontend/`. Build: `npm run build`, output: `dist/`.
- Env var **`VITE_API_BASE`** = public backend URL
  (e.g. `https://<app>.up.railway.app`). It is baked in at build time —
  changing the backend URL requires a rebuild/redeploy.
- No backend rewrites needed: the dashboard calls the API over absolute URLs.
- Local dev instead uses the Vite proxy (`vite.config.js`):
  `VITE_API_BASE=http://localhost:8000` in `frontend/.env.local`.

## Sync checklist (run before pushing)

- [ ] `requirements.txt` installs **both** entrypoints:
      `uvicorn api:app` (Railway) and `python app.py` (Space)
- [ ] `app.py` + `pipeline_hf.py` importable without TigerGraph/LLM keys
- [ ] `frontend/src/api.js` matches `docs/API.md` (paths, field names)
- [ ] Root `README.md` keeps the HF frontmatter block at the top
- [ ] `VITE_API_BASE` on Vercel points at the current backend host
- [ ] `MODEL_MIOU` in `pipeline.py` matches the deployed checkpoint's mIoU
