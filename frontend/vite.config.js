import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const ngrokUrl = env.NGROK_URL || '';

  return defineConfig({
    plugins: [react()],
    server: {
      port: 3000,
      proxy: ngrokUrl ? {
        '/analyze': ngrokUrl,
        '/image':   ngrokUrl,
        '/health':  ngrokUrl,
      } : undefined,
    }
  })
}