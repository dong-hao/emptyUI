import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * emptyUI end-to-end tests (headless, repeatable — runnable locally and in CI).
 *
 * The suite is full-stack: the frontend catalog/run API comes from the Python
 * backend, so both servers are started by Playwright's `webServer`:
 *   - Vite dev server (http://localhost:7860), which proxies /api -> :8000
 *   - The FastAPI backend (uvicorn, http://127.0.0.1:8000)
 *
 * The backend command auto-resolves to the repo-root `.venv` when present
 * (README development setup) and falls back to `python3` (CI installs the
 * editable packages system-wide).
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");
const venvPython = path.join(repoRoot, ".venv", "bin", "python");
const python = existsSync(venvPython) ? venvPython : "python3";

const backendCommand = [
  python,
  "-m",
  "uvicorn",
  "emptyui.server.app:create_app",
  "--factory",
  "--host",
  "127.0.0.1",
  "--port",
  "8000",
].join(" ");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  outputDir: "test-results",
  use: {
    baseURL: "http://localhost:7860",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: backendCommand,
      cwd: repoRoot,
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      command: "npm run dev",
      cwd: path.join(repoRoot, "packages", "web"),
      url: "http://localhost:7860",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
