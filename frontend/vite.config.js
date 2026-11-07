import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to the FastAPI backend so the app can use
// relative URLs locally. Set VITE_API_BASE (or legacy NGROK_URL) in
// frontend/.env.local — production builds bake VITE_API_BASE in instead.
export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_BASE || env.NGROK_URL || "";

  return defineConfig({
    plugins: [react()],
    server: {
      port: 3000,
      proxy: target
        ? {
            "/analyze": target,
            "/image": target,
            "/health": target,
            "/debug": target,
          }
        : undefined,
    },
  });
};
