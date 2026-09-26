import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the backend so the browser sees one origin.
// Without this, every fetch would need an absolute URL and the app would break
// the moment it was served from the FastAPI process at :8000 instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.OMNI_API ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Split D3 out of the main bundle. It is ~250 kB and only the graph view
    // needs it, so loading the dashboard should not pay for a force simulation
    // the user may never open.
    rollupOptions: {
      output: {
        manualChunks: {
          d3: ['d3'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
})
