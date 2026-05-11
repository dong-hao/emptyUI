import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // Exposed on 0.0.0.0 so a browser on another machine can reach the dev
    // server (remote testing). The API stays private behind the proxy —
    // only :7860 is reachable externally. To go back to localhost-only:
    //   npm run dev -- --host 127.0.0.1
    // (CLI flags override this file), or flip the line back to
    // host: "127.0.0.1".
    host: true,
    port: 7860,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, ws: true },
    },
  },
});
