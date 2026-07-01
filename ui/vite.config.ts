import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            // Suppress ECONNABORTED — fires when a browser tab closes mid-proxy
            if ((err as NodeJS.ErrnoException).code === 'ECONNABORTED') return
            console.error('[proxy]', err.message)
          })
        },
      },
    },
  },
})
