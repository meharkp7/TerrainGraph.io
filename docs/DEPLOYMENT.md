# Deployment

Three targets, one repo. Railway is a lean API (no torch) that calls the
HF Space for segmentation — the checklist at the bottom keeps them in sync.

## Backend → Railway (FastAPI + Docker, lean)

- Build: `Dockerfile` (`railway.toml` selects it). Installs
  `requirements-railway.txt` — no torch/smp/albumentations, ~150MB RAM,
  fits the 512MB tier. No model download (weights live on the Space).
- Start: `uvicorn api:app --host 0.0.0.0 --port ${PORT}` (via `sh -c`).
- Health check: `GET /health` — green in seconds (nothing heavy at boot).
- Env vars on Railway: `INFERENCE_BACKEND=space` (set in the image, but
  make sure no dashboard override says `local`), `HF_SPACE_ID`
  (`user/space-name` of the deployed Space below), `TIGERGRAPH_*`,
  `GROQ_API_KEY`/`GEMINI_API_KEY`/`ANTHROPIC_API_KEY`, `HF_TOKEN`
  (only for private Spaces/repos).
- First boot also needs the graph schema once: run
  `python setup_tigergraph.py` locally with the same TigerGraph credentials
  (or accept graceful local-only mode).
- First `/analyze` after idle takes a while: the free Space sleeps when
  unused, so the first call wakes it (~30–60s) and may 502 — the API
  returns `502 Inference Space unavailable … retry`, and the retry
  succeeds once the Space is warm.

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

**`Package 'libgl1-mesa-glx' has no installation candidate`?** The
`python:3.11-slim` base moved to Debian trixie, which dropped that
transitional package. The repo `Dockerfile` uses its replacement `libgl1` —
pull the latest `main` if your build still references the old name.

**`'${PORT}' is not a valid integer` at container start?** Railway runs
`startCommand` without a shell, so `${PORT}` was passed to uvicorn
literally. The repo wraps the command in `sh -c '... ${PORT:-8000}'`
(both in `railway.toml` and the `Dockerfile` CMD) so the variable expands.

**Container prints `Killed` with no traceback?** That's the OOM killer —
the 550MB model plus torch exceeds the service's RAM. Mitigations already
in the repo: torch/model/connection all load lazily on first `/analyze`
(boot is seconds, `/health` passes immediately), memory-mapped weight
loading, capped BLAS threads, single worker. If it still dies, check
Metrics → Memory to confirm the spike, then raise the service memory
limit (Settings → Resources) — a ResNet-101 fp32 model realistically
wants ≥2GB. The first `/analyze` after a deploy takes ~1–2 min (one-time
torch import + model load); after that it stays warm. If that first
request times out at the edge, just retry — the load continues.

**Deploy green + internal `/health` 200, but public URL gives 502
`connection refused`?** The edge is routing to a different internal port
than the app listens on. The app binds Railway's `$PORT` (8080); keep the
`Dockerfile`'s `EXPOSE` on the same port, and check the domain's target
port under service Settings → Networking.

**`/analyze` fails with `'TigerGraphConnection' object has no attribute
'restppPort'`?** The deprecated `apiToken=` constructor parameter crashes
on some pyTigerGraph versions. The repo sets the token + auth header after
construction instead, and `get_conn()` degrades to local-only mode (path +
risk zones still computed) whenever the connection can't be built — the
pipeline never 500s on TigerGraph issues. Covered by
`tests/test_tg_graceful.py`.

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

## Inference + demo → HuggingFace Space (Gradio, free 16GB CPU)

The Space deploys **from this repo** and serves two roles: the human demo
(`Demo` tab) and the machine endpoint the Railway backend calls
(`API` tab, `/segment` → `mask.png` + `conf.png` + `class_dist` JSON).

1. Use the existing Space `mk1647/devcation_tg` (Gradio SDK, CPU).
   Push this repo's `main` to it (HF login required — use a write token
   as the password):
   ```bash
   git remote add hf https://huggingface.co/spaces/mk1647/devcation_tg
   git push hf main
   ```
   The Space rebuilds from `app.py` + root `requirements.txt` (~5–10 min
   for the torch stack). Creating a fresh Space instead works the same way.
2. Space entrypoint is `app.py`; dependencies come from the root
   `requirements.txt` (torch CPU stack + `gradio` + `huggingface_hub`).
   Do NOT point the Space at `requirements-railway.txt` (no torch there).
3. The Space header lives at the top of [`README.md`](../README.md)
   (`sdk: gradio`, `app_file: app.py`) — that's what makes the file double
   as the Space card. Keep it.
4. Weights: `pipeline_hf.py` pulls `best.pth` from
   `mk1647/terrain-intelligence` on first run (cached after that).
5. No secrets needed on the Space: no TigerGraph, no LLM keys (the demo
   briefing is a template; graph + LLM run on Railway).
6. After deploy, copy the Space id (`user/space-name`) into Railway's
   `HF_SPACE_ID` env var and redeploy Railway. Verify:
   `GET /health` shows `"inference_backend": "space"`.
7. Free Spaces sleep after inactivity — the first call wakes them
   (~30–60s). Railway surfaces this as a retryable 502; the retry lands
   on a warm Space.

## Frontend → Vercel (React + Vite)

- Root directory: `frontend/`. Build: `npm run build`, output: `dist/`.
- Env var **`VITE_API_BASE`** = public backend URL
  (e.g. `https://<app>.up.railway.app`). It is baked in at build time —
  changing the backend URL requires a rebuild/redeploy.
- No backend rewrites needed: the dashboard calls the API over absolute URLs.
- Local dev instead uses the Vite proxy (`vite.config.js`):
  `VITE_API_BASE=http://localhost:8000` in `frontend/.env.local`.

## Sync checklist (run before pushing)

- [ ] `requirements-railway.txt` installs the lean backend (`uvicorn api:app`,
      no torch); root `requirements.txt` installs the Space (`python app.py`)
- [ ] `app.py` + `pipeline_hf.py` importable without TigerGraph/LLM keys
- [ ] `/segment` wire format matches `segmentor_space.py`
      (`mask.png` + `conf.png` + `class_dist`) — run `tests/test_space_codec.py`
- [ ] `frontend/src/api.js` matches `docs/API.md` (paths, field names)
- [ ] Root `README.md` keeps the HF frontmatter block at the top
- [ ] `VITE_API_BASE` on Vercel points at the current backend host
- [ ] `HF_SPACE_ID` on Railway points at the current Space id
- [ ] `MODEL_MIOU` in `pipeline.py` matches the deployed checkpoint's mIoU
