import { defineConfig } from "vitest/config";

// Unit tests cover the pure frontend logic: connection rules, wiring
// enforcement and the inheritance walk. They run in Node (no DOM, no React
// rendering); the Playwright suite covers the UI (P0.2).
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
