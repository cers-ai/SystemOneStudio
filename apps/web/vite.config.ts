import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const API_ORIGIN = process.env.SON_API_ORIGIN ?? 'http://127.0.0.1:9969';

/**
 * Dev server port.
 *
 * 5173 is occupied by an unrelated project on this machine, so the dev server
 * takes 5174. The production container serves everything on 9969 via nginx.
 */
const DEV_PORT = Number(process.env.SON_WEB_PORT ?? 5174);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: DEV_PORT,
    strictPort: false,
    proxy: {
      // The web client talks to the control plane over same-origin paths in dev,
      // so the browser never needs CORS and cookie handling stays simple.
      '/api': { target: API_ORIGIN, changeOrigin: true },
      '/meta': { target: API_ORIGIN, changeOrigin: true },
      '/health': { target: API_ORIGIN, changeOrigin: true },
      '/v1': { target: API_ORIGIN, changeOrigin: true },
      '/docs': { target: API_ORIGIN, changeOrigin: true },
      '/openapi.json': { target: API_ORIGIN, changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
});
