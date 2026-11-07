// api.js — single place where the frontend talks to the FastAPI backend.
//
// Backend contract (api.py):
//   GET  /health                  → { status, model_miou, tigergraph }
//   POST /analyze  (multipart, field "file")
//        → { image_id, class_dist, path, risk_zones, similar,
//            avg_trav, briefing, explanation, total_time, model_miou }
//   GET  /image/{image_id}/{original|segmented|overlay|path} → PNG
//
// Base URL resolution:
//   1. VITE_API_BASE baked in at build time (Vercel env var, or .env.local)
//   2. Same origin ("") — works when frontend is served by the backend,
//      or with the vite dev proxy (see vite.config.js)

export const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

// Lets ngrok tunnels skip their interstitial warning page.
const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" };

async function readError(res) {
  try {
    const body = await res.json();
    return body.detail || body.message || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function checkHealth(signal) {
  const res = await fetch(`${API_BASE}/health`, {
    headers: NGROK_HEADER,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function analyzeImage(file, signal) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
    headers: NGROK_HEADER,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function imageUrl(imageId, type) {
  return `${API_BASE}/image/${imageId}/${type}`;
}
