import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/analyze': 'https://rosaline-beeriest-camie.ngrok-free.dev',
      '/image':   'https://rosaline-beeriest-camie.ngrok-free.dev',
      '/health':  'https://rosaline-beeriest-camie.ngrok-free.dev',
    }
  }
})